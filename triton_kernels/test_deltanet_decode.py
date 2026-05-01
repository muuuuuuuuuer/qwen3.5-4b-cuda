"""Regression tests for the Phase 2A DeltaNet decode kernels and references."""

import math
import unittest

import torch
import torch.nn.functional as F

from triton_kernels.deltanet_decode import (
    DELTANET_AUTOTUNE_CONFIGS,
    DeltaNetKernelConfig,
    deltanet_decode_lowrank_beta_gate_reference,
    deltanet_decode_reference,
    deltanet_l2_normalize_qk,
    deltanet_decode_step,
    deltanet_decode_step_fused_gates,
)


HV = 32
K = 128
V = 128


def make_decode_inputs(device: str = "cpu", state_dtype: torch.dtype = torch.float32):
    torch.manual_seed(0)
    q = torch.randn(HV, K, dtype=torch.float16, device=device)
    k = torch.randn(HV, K, dtype=torch.float16, device=device)
    v = torch.randn(HV, V, dtype=torch.float16, device=device)
    g = -torch.rand(HV, dtype=torch.float32, device=device)
    beta = torch.sigmoid(torch.randn(HV, dtype=torch.float32, device=device))
    state = torch.randn(HV, K, V, dtype=state_dtype, device=device) * 0.01
    return q, k, v, g, beta, state


class DeltaNetDecodeReferenceTests(unittest.TestCase):
    def test_autotune_configs_cover_default_and_large_v_tiles(self):
        config_values = {(config.bv, config.num_warps, config.num_stages) for config in DELTANET_AUTOTUNE_CONFIGS}

        self.assertIn((32, 4, 1), config_values)
        self.assertIn((64, 8, 1), config_values)
        self.assertTrue(any(config.bv == 128 for config in DELTANET_AUTOTUNE_CONFIGS))

    def test_l2_normalize_qk_returns_unit_norms(self):
        q, k, _, _, _, _ = make_decode_inputs()

        q_norm, k_norm = deltanet_l2_normalize_qk(q, k)

        self.assertTrue(torch.allclose(q_norm.float().norm(p=2, dim=-1), torch.ones(HV), atol=1e-3))
        self.assertTrue(torch.allclose(k_norm.float().norm(p=2, dim=-1), torch.ones(HV), atol=1e-3))
        self.assertEqual(q_norm.dtype, q.dtype)
        self.assertEqual(k_norm.dtype, k.dtype)

    def test_reference_returns_expected_shapes_and_updates_state(self):
        q, k, v, g, beta, state = make_decode_inputs()
        state_before = state.clone()

        output = deltanet_decode_reference(q, k, v, g, beta, state)

        self.assertEqual(tuple(output.shape), (HV, V))
        self.assertEqual(tuple(state.shape), (HV, K, V))
        self.assertFalse(torch.equal(state, state_before))

    def test_lowrank_beta_gate_zero_up_matches_scalar_fused_math(self):
        q, k, v, _, _, state = make_decode_inputs()
        torch.manual_seed(3)
        rank = 8
        a = torch.randn(HV, dtype=torch.float32)
        b = torch.randn(HV, dtype=torch.float32)
        a_log = torch.randn(HV, dtype=torch.float32)
        dt_bias = torch.randn(HV, dtype=torch.float32)
        w_down = torch.randn(rank, K, dtype=torch.float32) * 1e-3
        w_up = torch.zeros(V, rank, dtype=torch.float32)
        b_up = torch.zeros(V, dtype=torch.float32)
        g = -torch.exp(a_log) * F.softplus(a + dt_bias)
        beta = torch.sigmoid(b)
        state_scalar = state.clone()
        state_lowrank = state.clone()

        output_scalar = deltanet_decode_reference(q, k, v, g, beta, state_scalar)
        output_lowrank = deltanet_decode_lowrank_beta_gate_reference(
            q,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
            w_down,
            w_up,
            b_up,
            state_lowrank,
        )

        self.assertLess((output_scalar - output_lowrank).abs().max().item(), 1e-6)
        self.assertLess((state_scalar - state_lowrank).abs().max().item(), 1e-6)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton kernel tests")
class DeltaNetDecodeKernelTests(unittest.TestCase):
    def test_triton_wrapper_matches_reference_on_real_qwen_dimensions(self):
        q, k, v, g, beta, state = make_decode_inputs(device="cuda")
        state_ref = state.clone()
        state_test = state.clone()

        output_ref = deltanet_decode_reference(q, k, v, g, beta, state_ref)
        output_test = deltanet_decode_step(q, k, v, g, beta, state_test)

        output_test_f32 = output_test.float()
        cosine_similarity = F.cosine_similarity(
            output_ref.flatten().unsqueeze(0),
            output_test_f32.flatten().unsqueeze(0),
        ).item()

        self.assertLess((output_ref - output_test_f32).abs().max().item(), 1e-3)
        self.assertGreater(cosine_similarity, 0.9999)
        self.assertLess((state_ref - state_test).abs().max().item(), 1e-3)

    def test_triton_wrapper_accepts_fp16_state_like_qwen_cache(self):
        q, k, v, g, beta, state = make_decode_inputs(device="cuda", state_dtype=torch.float16)
        state_ref = state.clone()
        state_test = state.clone()

        output_ref = deltanet_decode_reference(q, k, v, g, beta, state_ref)
        output_test = deltanet_decode_step(q, k, v, g, beta, state_test)

        self.assertLess((output_ref - output_test.float()).abs().max().item(), 5e-3)
        self.assertEqual(state_test.dtype, torch.float16)
        self.assertLess((state_ref.float() - state_test.float()).abs().max().item(), 5e-3)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton kernel tests")
class DeltaNetDecodeFusedGateTests(unittest.TestCase):
    def test_fused_gate_wrapper_matches_reference(self):
        q, k, v, _, _, state = make_decode_inputs(device="cuda")
        torch.manual_seed(1)
        a = torch.randn(HV, dtype=torch.float32, device="cuda")
        b = torch.randn(HV, dtype=torch.float32, device="cuda")
        a_log = torch.randn(HV, dtype=torch.float32, device="cuda")
        dt_bias = torch.randn(HV, dtype=torch.float32, device="cuda")

        g = -torch.exp(a_log) * F.softplus(a + dt_bias)
        beta = torch.sigmoid(b)

        state_ref = state.clone()
        state_test = state.clone()

        output_ref = deltanet_decode_reference(q, k, v, g, beta, state_ref)
        output_test = deltanet_decode_step_fused_gates(q, k, v, a, b, a_log, dt_bias, state_test)
        output_test_f32 = output_test.float()
        cosine_similarity = F.cosine_similarity(
            output_ref.flatten().unsqueeze(0),
            output_test_f32.flatten().unsqueeze(0),
        ).item()

        self.assertLess((output_ref - output_test_f32).abs().max().item(), 1e-3)
        self.assertGreater(cosine_similarity, 0.9999)
        self.assertLess((state_ref - state_test).abs().max().item(), 1e-3)

    def test_fused_gate_wrapper_accepts_pre_normalized_qk(self):
        q, k, v, _, _, state = make_decode_inputs(device="cuda")
        torch.manual_seed(1)
        a = torch.randn(HV, dtype=torch.float32, device="cuda")
        b = torch.randn(HV, dtype=torch.float32, device="cuda")
        a_log = torch.randn(HV, dtype=torch.float32, device="cuda")
        dt_bias = torch.randn(HV, dtype=torch.float32, device="cuda")

        g = -torch.exp(a_log) * F.softplus(a + dt_bias)
        beta = torch.sigmoid(b)
        q_norm, k_norm = deltanet_l2_normalize_qk(q, k)

        state_ref = state.clone()
        state_test = state.clone()

        output_ref = deltanet_decode_reference(q, k, v, g, beta, state_ref)
        output_test = deltanet_decode_step_fused_gates(
            q_norm,
            k_norm,
            v,
            a,
            b,
            a_log,
            dt_bias,
            state_test,
            use_qk_l2norm=False,
        )

        self.assertLess((output_ref - output_test.float()).abs().max().item(), 1e-3)
        self.assertLess((state_ref - state_test).abs().max().item(), 1e-3)

@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton kernel tests")
class DeltaNetDecodeTuningConfigTests(unittest.TestCase):
    def test_base_wrapper_accepts_custom_kernel_config(self):
        q, k, v, g, beta, state = make_decode_inputs(device="cuda")
        state_ref = state.clone()
        state_test = state.clone()
        config = DeltaNetKernelConfig(bv=64, num_warps=8, num_stages=2)

        output_ref = deltanet_decode_reference(q, k, v, g, beta, state_ref)
        output_test = deltanet_decode_step(q, k, v, g, beta, state_test, kernel_config=config)

        self.assertLess((output_ref - output_test.float()).abs().max().item(), 1e-3)
        self.assertLess((state_ref - state_test).abs().max().item(), 1e-3)

    def test_fused_gate_wrapper_accepts_custom_kernel_config(self):
        q, k, v, _, _, state = make_decode_inputs(device="cuda")
        torch.manual_seed(1)
        a = torch.randn(HV, dtype=torch.float32, device="cuda")
        b = torch.randn(HV, dtype=torch.float32, device="cuda")
        a_log = torch.randn(HV, dtype=torch.float32, device="cuda")
        dt_bias = torch.randn(HV, dtype=torch.float32, device="cuda")
        g = -torch.exp(a_log) * F.softplus(a + dt_bias)
        beta = torch.sigmoid(b)
        state_ref = state.clone()
        state_test = state.clone()
        config = DeltaNetKernelConfig(bv=16, num_warps=2, num_stages=2)

        output_ref = deltanet_decode_reference(q, k, v, g, beta, state_ref)
        output_test = deltanet_decode_step_fused_gates(
            q,
            k,
            v,
            a,
            b,
            a_log,
            dt_bias,
            state_test,
            kernel_config=config,
        )

        self.assertLess((output_ref - output_test.float()).abs().max().item(), 1e-3)
        self.assertLess((state_ref - state_test).abs().max().item(), 1e-3)


if __name__ == "__main__":
    unittest.main()
