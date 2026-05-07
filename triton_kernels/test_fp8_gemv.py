"""Correctness tests for FP8 GEMV Triton kernel.

Compares the Triton FP8 GEMV kernel against a CPU reference implementation
across all Qwen3.5-4B representative projection shapes.
"""

from __future__ import annotations

import unittest

import torch


class FP8GEMVReferenceTests(unittest.TestCase):
    """Tests for the PyTorch reference implementation."""

    def setUp(self):
        torch.manual_seed(42)

    def test_reference_output_shape(self):
        from triton_kernels.fp8_gemv import fp8_gemv_reference
        N, K = 64, 128
        w = torch.randn(N, K, dtype=torch.float16)
        from quantize_fp8_weights import quantize_weight_fp8
        w_fp8, scale = quantize_weight_fp8(w, dim=0)
        x = torch.randn(K, dtype=torch.float16)
        y = fp8_gemv_reference(w_fp8, scale, x)
        self.assertEqual(y.shape, (N,))
        self.assertEqual(y.dtype, torch.float16)

    def test_reference_vs_cpu_small(self):
        from triton_kernels.fp8_gemv import fp8_gemv_reference, fp8_gemv_cpu_reference
        N, K = 16, 32
        w = torch.randn(N, K, dtype=torch.float16)
        from quantize_fp8_weights import quantize_weight_fp8
        w_fp8, scale = quantize_weight_fp8(w, dim=0)
        x = torch.randn(K, dtype=torch.float16)
        y_ref = fp8_gemv_reference(w_fp8, scale, x)
        y_cpu = fp8_gemv_cpu_reference(w_fp8, scale, x)
        self.assertTrue(torch.allclose(y_ref.float(), y_cpu.float(), atol=1e-3, rtol=5e-3),
                        f"max diff: {(y_ref.float() - y_cpu.float()).abs().max()}")

    def test_reference_mathematical_correctness(self):
        from triton_kernels.fp8_gemv import fp8_gemv_reference
        N, K = 32, 64
        w = torch.randn(N, K, dtype=torch.float16)
        from quantize_fp8_weights import quantize_weight_fp8
        w_fp8, scale = quantize_weight_fp8(w, dim=0)
        x = torch.ones(K, dtype=torch.float16)
        y = fp8_gemv_reference(w_fp8, scale, x)

        w_deq = w_fp8.float() * scale.float().squeeze(-1).unsqueeze(-1)
        x_fp32 = x.float()
        expected = w_deq @ x_fp32
        # Reference returns FP16, expected is FP32; account for cast precision loss
        self.assertTrue(
            torch.allclose(y.float(), expected, atol=5e-2, rtol=1e-2),
            f"max diff: {(y.float() - expected).abs().max().item():.6f}",
        )


class FP8GEMVKernelTests(unittest.TestCase):
    """CUDA tests comparing Triton FP8 GEMV kernel against reference."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            import triton
        except ModuleNotFoundError:
            raise unittest.SkipTest("Triton not available")

    def setUp(self):
        torch.manual_seed(42)

    def _run_test_shape(self, N: int, K: int, atol: float = 5e-3, rtol: float = 1e-2):
        from triton_kernels.fp8_gemv import fp8_gemv, fp8_gemv_reference
        from quantize_fp8_weights import quantize_weight_fp8

        w = torch.randn(N, K, dtype=torch.float16, device="cuda")
        w_fp8, scale = quantize_weight_fp8(w, dim=0)
        w_fp8 = w_fp8.cuda()
        scale = scale.cuda()
        x = torch.randn(K, dtype=torch.float16, device="cuda")

        y_triton = fp8_gemv(w_fp8, scale, x, use_autotune=True)
        y_ref = fp8_gemv_reference(w_fp8.cpu(), scale.cpu(), x.cpu()).cuda()

        max_abs_err = (y_triton.float() - y_ref.float()).abs().max().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            y_triton.float().unsqueeze(0), y_ref.float().unsqueeze(0)
        ).item()

        self.assertTrue(
            torch.allclose(y_triton.float(), y_ref.float(), atol=atol, rtol=rtol),
            f"Shape [{N},{K}]: max_abs_err={max_abs_err:.6f}, cos_sim={cos_sim:.8f}",
        )

    def test_small_square(self):
        self._run_test_shape(64, 128, rtol=1e-2)

    def test_medium_shape(self):
        self._run_test_shape(256, 512, rtol=1e-2)

    def test_ffn_gate_shape(self):
        self._run_test_shape(9216, 2560, rtol=1e-2)

    def test_ffn_down_shape(self):
        self._run_test_shape(2560, 9216, rtol=1e-2)

    def test_full_attention_q_shape(self):
        self._run_test_shape(8192, 2560, rtol=1e-2)

    def test_full_attention_k_shape(self):
        self._run_test_shape(1024, 2560, rtol=1e-2)

    def test_deltanet_qkv_shape(self):
        self._run_test_shape(8192, 2560, rtol=1e-2)

    def test_deltanet_z_shape(self):
        self._run_test_shape(4096, 2560, rtol=1e-2)

    def test_output_is_fp16(self):
        from triton_kernels.fp8_gemv import fp8_gemv
        from quantize_fp8_weights import quantize_weight_fp8

        N, K = 256, 512
        w = torch.randn(N, K, dtype=torch.float16, device="cuda")
        w_fp8, scale = quantize_weight_fp8(w, dim=0)
        w_fp8 = w_fp8.cuda()
        scale = scale.cuda()
        x = torch.randn(K, dtype=torch.float16, device="cuda")

        y = fp8_gemv(w_fp8, scale, x, use_autotune=False)
        self.assertEqual(y.dtype, torch.float16)
        self.assertEqual(y.shape, (N,))

    def test_manual_config(self):
        from triton_kernels.fp8_gemv import fp8_gemv, FP8GEMVKernelConfig
        from quantize_fp8_weights import quantize_weight_fp8

        N, K = 1024, 2560
        w = torch.randn(N, K, dtype=torch.float16, device="cuda")
        w_fp8, scale = quantize_weight_fp8(w, dim=0)
        w_fp8 = w_fp8.cuda()
        scale = scale.cuda()
        x = torch.randn(K, dtype=torch.float16, device="cuda")

        cfg = FP8GEMVKernelConfig(bn=128, bk=128, num_warps=4, num_stages=1)
        y = fp8_gemv(w_fp8, scale, x, kernel_config=cfg, use_autotune=False)

        from triton_kernels.fp8_gemv import fp8_gemv_reference
        y_ref = fp8_gemv_reference(w_fp8.cpu(), scale.cpu(), x.cpu()).cuda()
        self.assertTrue(torch.allclose(y.float(), y_ref.float(), atol=1e-2, rtol=1e-2))


class FP8GEMVAutotuneTests(unittest.TestCase):
    """Tests for autotuning support across different shapes."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        try:
            import triton
        except ModuleNotFoundError:
            raise unittest.SkipTest("Triton not available")

    def test_autotune_multiple_configs_available(self):
        from triton_kernels.fp8_gemv import FP8_GEMV_AUTOTUNE_CONFIGS
        self.assertGreater(len(FP8_GEMV_AUTOTUNE_CONFIGS), 2)

    def test_autotune_caches_result(self):
        from triton_kernels.fp8_gemv import fp8_gemv, _AUTOTUNE_CACHE
        from quantize_fp8_weights import quantize_weight_fp8

        torch.manual_seed(42)
        _AUTOTUNE_CACHE.clear()

        N, K = 512, 512
        w = torch.randn(N, K, dtype=torch.float16, device="cuda")
        w_fp8, scale = quantize_weight_fp8(w, dim=0)
        w_fp8 = w_fp8.cuda()
        scale = scale.cuda()
        x = torch.randn(K, dtype=torch.float16, device="cuda")

        y1 = fp8_gemv(w_fp8, scale, x, use_autotune=True)
        self.assertGreater(len(_AUTOTUNE_CACHE), 0, "Autotune should cache result")

        # Second call should use cached config
        x2 = torch.randn(K, dtype=torch.float16, device="cuda")
        y2 = fp8_gemv(w_fp8, scale, x2, use_autotune=True)
        self.assertEqual(y2.shape, (N,))


if __name__ == "__main__":
    unittest.main()
