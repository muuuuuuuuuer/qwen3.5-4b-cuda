"""Regression tests for the grouped Qwen3.5 DeltaNet decode projection pack."""

import unittest

import torch
import torch.nn.functional as F

from triton_kernels.qwen35_projection_pack import (
    qwen35_grouped_projection_conv_lowrank_deltanet,
    qwen35_grouped_projection_conv_pack_reference,
)
from triton_kernels.deltanet_decode import (
    deltanet_decode_lowrank_beta_gate_reference,
)


def make_pack_inputs(device: str = "cpu"):
    torch.manual_seed(7)
    batch = 2
    d_model = 16
    num_k_heads = 2
    num_v_heads = 4
    head_k_dim = 4
    head_v_dim = 4
    conv_width = 4
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_dim = key_dim * 2 + value_dim
    hidden_states = torch.randn(batch, 1, d_model, dtype=torch.float32, device=device)
    conv_state = torch.randn(batch, conv_dim, conv_width, dtype=torch.float32, device=device)
    w_qkv = torch.randn(conv_dim, d_model, dtype=torch.float32, device=device) * 0.1
    w_z = torch.randn(value_dim, d_model, dtype=torch.float32, device=device) * 0.1
    w_a = torch.randn(num_v_heads, d_model, dtype=torch.float32, device=device) * 0.1
    w_b = torch.randn(num_v_heads, d_model, dtype=torch.float32, device=device) * 0.1
    conv_weight = torch.randn(conv_dim, conv_width, dtype=torch.float32, device=device) * 0.1
    return {
        "hidden_states": hidden_states,
        "conv_state": conv_state,
        "w_qkv": w_qkv,
        "w_z": w_z,
        "w_a": w_a,
        "w_b": w_b,
        "conv_weight": conv_weight,
        "num_k_heads": num_k_heads,
        "num_v_heads": num_v_heads,
        "head_k_dim": head_k_dim,
        "head_v_dim": head_v_dim,
    }


def original_projection_conv_reference(inputs: dict[str, torch.Tensor]):
    hidden_states = inputs["hidden_states"]
    conv_state = inputs["conv_state"]
    w_qkv = inputs["w_qkv"]
    w_z = inputs["w_z"]
    w_a = inputs["w_a"]
    w_b = inputs["w_b"]
    conv_weight = inputs["conv_weight"]
    num_k_heads = inputs["num_k_heads"]
    num_v_heads = inputs["num_v_heads"]
    head_k_dim = inputs["head_k_dim"]
    head_v_dim = inputs["head_v_dim"]
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim

    mixed_qkv = F.linear(hidden_states, w_qkv).transpose(1, 2)
    hidden_states_new = torch.cat([conv_state, mixed_qkv], dim=-1).to(conv_weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -conv_state.shape[-1] :])
    mixed_qkv = F.conv1d(hidden_states_new, conv_weight.unsqueeze(1), padding=0, groups=conv_weight.shape[0])
    mixed_qkv = F.silu(mixed_qkv[:, :, -1:]).transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
    query = query.reshape(hidden_states.shape[0], 1, num_k_heads, head_k_dim)
    key = key.reshape(hidden_states.shape[0], 1, num_k_heads, head_k_dim)
    value = value.reshape(hidden_states.shape[0], 1, num_v_heads, head_v_dim)
    z = F.linear(hidden_states, w_z).reshape(hidden_states.shape[0], 1, num_v_heads, head_v_dim)
    a = F.linear(hidden_states, w_a).reshape(hidden_states.shape[0], 1, num_v_heads)
    b = F.linear(hidden_states, w_b).reshape(hidden_states.shape[0], 1, num_v_heads)
    return query[:, 0], key[:, 0], value[:, 0], z[:, 0], a[:, 0], b[:, 0]


class Qwen35ProjectionPackReferenceTests(unittest.TestCase):
    def test_reference_matches_original_projection_conv_flow(self):
        inputs = make_pack_inputs()
        expected_state = inputs["conv_state"].clone()
        expected = original_projection_conv_reference({**inputs, "conv_state": expected_state})
        packed_state = inputs["conv_state"].clone()

        actual = qwen35_grouped_projection_conv_pack_reference(
            inputs["hidden_states"],
            packed_state,
            inputs["w_qkv"],
            inputs["w_z"],
            inputs["w_a"],
            inputs["w_b"],
            inputs["conv_weight"],
            num_k_heads=inputs["num_k_heads"],
            num_v_heads=inputs["num_v_heads"],
            head_k_dim=inputs["head_k_dim"],
            head_v_dim=inputs["head_v_dim"],
        )

        for expected_tensor, actual_tensor in zip(expected, actual):
            self.assertLess((expected_tensor - actual_tensor).abs().max().item(), 1e-6)
        self.assertLess((expected_state - packed_state).abs().max().item(), 1e-6)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton projection-pack tests")
class Qwen35ProjectionPackKernelTests(unittest.TestCase):
    def test_fused_projection_pack_lowrank_decode_matches_reference(self):
        inputs = make_pack_inputs(device="cuda")
        torch.manual_seed(11)
        rank = 2
        batch = inputs["hidden_states"].shape[0]
        num_v_heads = inputs["num_v_heads"]
        head_k_dim = inputs["head_k_dim"]
        head_v_dim = inputs["head_v_dim"]
        a_log = torch.randn(num_v_heads, dtype=torch.float32, device="cuda") * 0.02
        dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device="cuda") * 0.02
        w_down = torch.randn(rank, head_k_dim, dtype=torch.float32, device="cuda") * 0.02
        w_up = torch.randn(head_v_dim, rank, dtype=torch.float32, device="cuda") * 0.02
        b_up = torch.randn(head_v_dim, dtype=torch.float32, device="cuda") * 0.02
        recurrent_state = torch.randn(
            batch,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            dtype=torch.float32,
            device="cuda",
        ) * 0.02

        expected_conv_state = inputs["conv_state"].clone()
        expected_recurrent_state = recurrent_state.clone()
        q_raw, k_raw, value, expected_z, a, b = qwen35_grouped_projection_conv_pack_reference(
            inputs["hidden_states"],
            expected_conv_state,
            inputs["w_qkv"],
            inputs["w_z"],
            inputs["w_a"],
            inputs["w_b"],
            inputs["conv_weight"],
            num_k_heads=inputs["num_k_heads"],
            num_v_heads=inputs["num_v_heads"],
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
        )
        expected_core = torch.empty_like(value)
        qk_repeat = num_v_heads // inputs["num_k_heads"]
        for batch_idx in range(batch):
            expected_core[batch_idx] = deltanet_decode_lowrank_beta_gate_reference(
                q_raw[batch_idx].repeat_interleave(qk_repeat, dim=0),
                k_raw[batch_idx].repeat_interleave(qk_repeat, dim=0),
                value[batch_idx],
                a[batch_idx],
                b[batch_idx],
                a_log,
                dt_bias,
                w_down,
                w_up,
                b_up,
                expected_recurrent_state[batch_idx],
            )

        fused_conv_state = inputs["conv_state"].clone()
        fused_recurrent_state = recurrent_state.clone()
        actual_core, actual_z = qwen35_grouped_projection_conv_lowrank_deltanet(
            inputs["hidden_states"],
            fused_conv_state,
            inputs["w_qkv"],
            inputs["w_z"],
            inputs["w_a"],
            inputs["w_b"],
            inputs["conv_weight"],
            a_log,
            dt_bias,
            w_down,
            w_up,
            b_up,
            fused_recurrent_state,
            num_k_heads=inputs["num_k_heads"],
            num_v_heads=inputs["num_v_heads"],
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            block_d=16,
            num_warps=4,
        )

        self.assertLess((expected_core - actual_core.float()).abs().max().item(), 1e-4)
        self.assertLess((expected_z - actual_z.float()).abs().max().item(), 1e-4)
        self.assertLess((expected_conv_state - fused_conv_state).abs().max().item(), 1e-4)
        self.assertLess((expected_recurrent_state - fused_recurrent_state).abs().max().item(), 1e-4)


if __name__ == "__main__":
    unittest.main()
