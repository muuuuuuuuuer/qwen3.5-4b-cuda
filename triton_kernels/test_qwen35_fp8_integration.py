"""Tests for FP8 integration module.

Tests the monkey-patch logic on a synthetic model without loading
the full Qwen3.5 checkpoint.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn


class SyntheticMLP(nn.Module):
    def __init__(self, hidden=2560, intermediate=9216):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class FP8IntegrationTests(unittest.TestCase):
    """Tests for FP8 integration module on synthetic model."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            import triton
        except ModuleNotFoundError:
            raise unittest.SkipTest("Triton not available")

    def setUp(self):
        from triton_kernels.qwen35_fp8_integration import _PATCHED_LAYERS
        _PATCHED_LAYERS.clear()

    def test_apply_and_restore(self):
        from triton_kernels.qwen35_fp8_integration import (
            apply_qwen35_fp8,
            restore_original_fp8_layers,
        )
        model = SyntheticMLP().cuda().to(torch.float16)

        result = apply_qwen35_fp8(model, target_ffn=True, target_lm_head=False)
        self.assertEqual(result["patched_count"], 3)  # gate, up, down
        self.assertIn("gate_proj", str(result["patched_names"]))
        self.assertIn("up_proj", str(result["patched_names"]))
        self.assertIn("down_proj", str(result["patched_names"]))

        restored = restore_original_fp8_layers(model)
        self.assertEqual(restored, 3)

    def test_decode_forward_correctness(self):
        from triton_kernels.qwen35_fp8_integration import (
            apply_qwen35_fp8,
            restore_original_fp8_layers,
        )
        from triton_kernels.fp8_gemv import fp8_gemv_reference

        torch.manual_seed(42)
        model = SyntheticMLP().cuda().to(torch.float16)

        # Record original output
        x = torch.randn(1, 2560, dtype=torch.float16, device="cuda")
        with torch.no_grad():
            y_orig = model(x)

        # Apply FP8
        apply_qwen35_fp8(model, target_ffn=True, target_lm_head=False)

        # FP8 output
        torch.manual_seed(42)
        with torch.no_grad():
            y_fp8 = model(x)

        # Should match reasonably
        max_diff = (y_fp8 - y_orig).abs().max().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            y_fp8.float().reshape(-1).unsqueeze(0),
            y_orig.float().reshape(-1).unsqueeze(0),
        ).item()

        self.assertGreater(cos_sim, 0.99, f"cos_sim={cos_sim:.6f}, max_diff={max_diff:.6f}")

        restore_original_fp8_layers(model)

    def test_prefill_fallback(self):
        from triton_kernels.qwen35_fp8_integration import (
            apply_qwen35_fp8,
            restore_original_fp8_layers,
            get_fp8_integration_stats,
        )

        torch.manual_seed(42)
        model = SyntheticMLP().cuda().to(torch.float16)
        apply_qwen35_fp8(model, target_ffn=True, target_lm_head=False)

        # Batch > 1 should trigger fallback
        x = torch.randn(4, 2560, dtype=torch.float16, device="cuda")
        with torch.no_grad():
            _ = model(x)

        stats = get_fp8_integration_stats()
        self.assertGreater(stats["fallback_calls"], 0)

        restore_original_fp8_layers(model)

    def test_multiple_forward_calls(self):
        from triton_kernels.qwen35_fp8_integration import (
            apply_qwen35_fp8,
            restore_original_fp8_layers,
            get_fp8_integration_stats,
            reset_fp8_integration_stats,
        )

        torch.manual_seed(42)
        model = SyntheticMLP().cuda().to(torch.float16)
        apply_qwen35_fp8(model, target_ffn=True, target_lm_head=False)

        reset_fp8_integration_stats()
        for _ in range(5):
            x = torch.randn(1, 2560, dtype=torch.float16, device="cuda")
            with torch.no_grad():
                _ = model(x)

        stats = get_fp8_integration_stats()
        # 5 calls × 3 layers = 15 total calls, all fp8
        self.assertEqual(stats["calls"], 15)
        self.assertEqual(stats["fp8_calls"], 15)
        self.assertEqual(stats["fallback_calls"], 0)

        restore_original_fp8_layers(model)

    def test_restore_preserves_original(self):
        """After restore, model should behave identically to pre-patch."""
        from triton_kernels.qwen35_fp8_integration import (
            apply_qwen35_fp8,
            restore_original_fp8_layers,
        )

        torch.manual_seed(42)
        model = SyntheticMLP().cuda().to(torch.float16)

        x = torch.randn(1, 2560, dtype=torch.float16, device="cuda")
        with torch.no_grad():
            y_before = model(x)

        apply_qwen35_fp8(model, target_ffn=True, target_lm_head=False)
        restore_original_fp8_layers(model)

        torch.manual_seed(42)
        with torch.no_grad():
            y_after = model(x)

        self.assertTrue(torch.equal(y_before, y_after),
                        "After restore, output should be identical to pre-patch")


if __name__ == "__main__":
    unittest.main()
