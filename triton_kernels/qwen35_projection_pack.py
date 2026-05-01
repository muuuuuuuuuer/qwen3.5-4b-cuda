"""Fused grouped projection+conv plus low-rank DeltaNet decode for Qwen3.5."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - exercised only in non-Triton environments
    triton = None
    tl = None


def qwen35_grouped_projection_conv_pack_reference(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    w_qkv: torch.Tensor,
    w_z: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    conv_weight: torch.Tensor,
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference for the decode-only projection pack; updates `conv_state` in place."""
    if hidden_states.ndim == 2:
        hidden_states_3d = hidden_states.unsqueeze(1)
    elif hidden_states.ndim == 3 and hidden_states.shape[1] == 1:
        hidden_states_3d = hidden_states
    else:
        raise ValueError("hidden_states must have shape [B, D] or [B, 1, D]")

    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_dim = key_dim * 2 + value_dim
    if conv_state.ndim != 3 or conv_state.shape[1] != conv_dim:
        raise ValueError("conv_state must have shape [B, 2*key_dim + value_dim, conv_width]")
    if w_qkv.shape != (conv_dim, hidden_states_3d.shape[-1]):
        raise ValueError("w_qkv has an unexpected shape")
    if w_z.shape != (value_dim, hidden_states_3d.shape[-1]):
        raise ValueError("w_z has an unexpected shape")
    if w_a.shape != (num_v_heads, hidden_states_3d.shape[-1]):
        raise ValueError("w_a has an unexpected shape")
    if w_b.shape != (num_v_heads, hidden_states_3d.shape[-1]):
        raise ValueError("w_b has an unexpected shape")
    if conv_weight.shape != (conv_dim, conv_state.shape[-1]):
        raise ValueError("conv_weight has an unexpected shape")

    mixed_qkv = F.linear(hidden_states_3d, w_qkv).transpose(1, 2)
    hidden_states_new = torch.cat([conv_state, mixed_qkv], dim=-1).to(conv_weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -conv_state.shape[-1] :])
    mixed_qkv = F.conv1d(hidden_states_new, conv_weight.unsqueeze(1), padding=0, groups=conv_dim)
    mixed_qkv = F.silu(mixed_qkv[:, :, -1:]).transpose(1, 2).to(hidden_states_3d.dtype)

    q_raw, k_raw, value = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
    batch = hidden_states_3d.shape[0]
    q_raw = q_raw.reshape(batch, 1, num_k_heads, head_k_dim)[:, 0]
    k_raw = k_raw.reshape(batch, 1, num_k_heads, head_k_dim)[:, 0]
    value = value.reshape(batch, 1, num_v_heads, head_v_dim)[:, 0]
    z = F.linear(hidden_states_3d, w_z).reshape(batch, 1, num_v_heads, head_v_dim)[:, 0]
    a = F.linear(hidden_states_3d, w_a).reshape(batch, 1, num_v_heads)[:, 0]
    b = F.linear(hidden_states_3d, w_b).reshape(batch, 1, num_v_heads)[:, 0]
    return q_raw, k_raw, value, z, a, b


if triton is not None:

    @triton.jit
    def qwen35_grouped_projection_conv_lowrank_deltanet_kernel(
        x_ptr,
        w_qkv_ptr,
        w_z_ptr,
        w_a_ptr,
        w_b_ptr,
        conv_w_ptr,
        conv_state_ptr,
        conv_state_stride_b,
        conv_state_stride_c,
        conv_state_stride_w,
        a_log_ptr,
        dt_bias_ptr,
        w_down_ptr,
        w_up_ptr,
        b_up_ptr,
        recurrent_state_ptr,
        o_ptr,
        z_out_ptr,
        scale,
        D_MODEL: tl.constexpr,
        NUM_K_HEADS: tl.constexpr,
        NUM_V_HEADS: tl.constexpr,
        HEAD_K_DIM: tl.constexpr,
        HEAD_V_DIM: tl.constexpr,
        QK_REPEAT: tl.constexpr,
        CONV_DIM: tl.constexpr,
        CONV_WIDTH: tl.constexpr,
        R: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        BR: tl.constexpr,
        BLOCK_D: tl.constexpr,
        USE_QK_L2NORM: tl.constexpr,
    ):
        i_group = tl.program_id(0)
        i_batch = tl.program_id(1)

        o_k = tl.arange(0, BK)
        o_v = tl.arange(0, BV)
        o_r = tl.arange(0, BR)
        o_d = tl.arange(0, BLOCK_D)
        mask_k = o_k < HEAD_K_DIM
        mask_v = o_v < HEAD_V_DIM
        mask_r = o_r < R
        mask_h = mask_k[:, None] & mask_v[None, :]

        key_dim: tl.constexpr = NUM_K_HEADS * HEAD_K_DIM
        hv0 = i_group * QK_REPEAT
        hv1 = hv0 + 1

        q_row = i_group * HEAD_K_DIM + o_k
        k_row = key_dim + i_group * HEAD_K_DIM + o_k
        v0_row = 2 * key_dim + hv0 * HEAD_V_DIM + o_v
        v1_row = 2 * key_dim + hv1 * HEAD_V_DIM + o_v
        z0_row = hv0 * HEAD_V_DIM + o_v
        z1_row = hv1 * HEAD_V_DIM + o_v

        acc_q = tl.zeros((BK,), dtype=tl.float32)
        acc_k = tl.zeros((BK,), dtype=tl.float32)
        acc_v0 = tl.zeros((BV,), dtype=tl.float32)
        acc_v1 = tl.zeros((BV,), dtype=tl.float32)
        acc_z0 = tl.zeros((BV,), dtype=tl.float32)
        acc_z1 = tl.zeros((BV,), dtype=tl.float32)
        acc_a0 = tl.full((), 0.0, dtype=tl.float32)
        acc_a1 = tl.full((), 0.0, dtype=tl.float32)
        acc_b0 = tl.full((), 0.0, dtype=tl.float32)
        acc_b1 = tl.full((), 0.0, dtype=tl.float32)

        for d_start in tl.static_range(0, D_MODEL, BLOCK_D):
            d = d_start + o_d
            mask_d = d < D_MODEL
            x = tl.load(x_ptr + i_batch * D_MODEL + d, mask=mask_d, other=0.0).to(tl.float32)

            w_q = tl.load(
                w_qkv_ptr + q_row[:, None] * D_MODEL + d[None, :],
                mask=mask_k[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            w_k = tl.load(
                w_qkv_ptr + k_row[:, None] * D_MODEL + d[None, :],
                mask=mask_k[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            w_v0 = tl.load(
                w_qkv_ptr + v0_row[:, None] * D_MODEL + d[None, :],
                mask=mask_v[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            w_v1 = tl.load(
                w_qkv_ptr + v1_row[:, None] * D_MODEL + d[None, :],
                mask=mask_v[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            w_z0 = tl.load(
                w_z_ptr + z0_row[:, None] * D_MODEL + d[None, :],
                mask=mask_v[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            w_z1 = tl.load(
                w_z_ptr + z1_row[:, None] * D_MODEL + d[None, :],
                mask=mask_v[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            w_a0 = tl.load(w_a_ptr + hv0 * D_MODEL + d, mask=mask_d, other=0.0).to(tl.float32)
            w_a1 = tl.load(w_a_ptr + hv1 * D_MODEL + d, mask=mask_d, other=0.0).to(tl.float32)
            w_b0 = tl.load(w_b_ptr + hv0 * D_MODEL + d, mask=mask_d, other=0.0).to(tl.float32)
            w_b1 = tl.load(w_b_ptr + hv1 * D_MODEL + d, mask=mask_d, other=0.0).to(tl.float32)

            acc_q += tl.sum(w_q * x[None, :], axis=1)
            acc_k += tl.sum(w_k * x[None, :], axis=1)
            acc_v0 += tl.sum(w_v0 * x[None, :], axis=1)
            acc_v1 += tl.sum(w_v1 * x[None, :], axis=1)
            acc_z0 += tl.sum(w_z0 * x[None, :], axis=1)
            acc_z1 += tl.sum(w_z1 * x[None, :], axis=1)
            acc_a0 += tl.sum(w_a0 * x, axis=0)
            acc_a1 += tl.sum(w_a1 * x, axis=0)
            acc_b0 += tl.sum(w_b0 * x, axis=0)
            acc_b1 += tl.sum(w_b1 * x, axis=0)

        q_state_base = i_batch * conv_state_stride_b + q_row * conv_state_stride_c
        k_state_base = i_batch * conv_state_stride_b + k_row * conv_state_stride_c
        v0_state_base = i_batch * conv_state_stride_b + v0_row * conv_state_stride_c
        v1_state_base = i_batch * conv_state_stride_b + v1_row * conv_state_stride_c

        q_s1 = tl.load(conv_state_ptr + q_state_base + 1 * conv_state_stride_w, mask=mask_k, other=0.0).to(tl.float32)
        q_s2 = tl.load(conv_state_ptr + q_state_base + 2 * conv_state_stride_w, mask=mask_k, other=0.0).to(tl.float32)
        q_s3 = tl.load(conv_state_ptr + q_state_base + 3 * conv_state_stride_w, mask=mask_k, other=0.0).to(tl.float32)
        q_cw0 = tl.load(conv_w_ptr + q_row * CONV_WIDTH + 0, mask=mask_k, other=0.0).to(tl.float32)
        q_cw1 = tl.load(conv_w_ptr + q_row * CONV_WIDTH + 1, mask=mask_k, other=0.0).to(tl.float32)
        q_cw2 = tl.load(conv_w_ptr + q_row * CONV_WIDTH + 2, mask=mask_k, other=0.0).to(tl.float32)
        q_cw3 = tl.load(conv_w_ptr + q_row * CONV_WIDTH + 3, mask=mask_k, other=0.0).to(tl.float32)
        b_q = q_cw0 * q_s1 + q_cw1 * q_s2 + q_cw2 * q_s3 + q_cw3 * acc_q
        b_q = b_q / (1.0 + tl.exp(-b_q))
        b_q = b_q.to(o_ptr.dtype.element_ty).to(tl.float32)

        k_s1 = tl.load(conv_state_ptr + k_state_base + 1 * conv_state_stride_w, mask=mask_k, other=0.0).to(tl.float32)
        k_s2 = tl.load(conv_state_ptr + k_state_base + 2 * conv_state_stride_w, mask=mask_k, other=0.0).to(tl.float32)
        k_s3 = tl.load(conv_state_ptr + k_state_base + 3 * conv_state_stride_w, mask=mask_k, other=0.0).to(tl.float32)
        k_cw0 = tl.load(conv_w_ptr + k_row * CONV_WIDTH + 0, mask=mask_k, other=0.0).to(tl.float32)
        k_cw1 = tl.load(conv_w_ptr + k_row * CONV_WIDTH + 1, mask=mask_k, other=0.0).to(tl.float32)
        k_cw2 = tl.load(conv_w_ptr + k_row * CONV_WIDTH + 2, mask=mask_k, other=0.0).to(tl.float32)
        k_cw3 = tl.load(conv_w_ptr + k_row * CONV_WIDTH + 3, mask=mask_k, other=0.0).to(tl.float32)
        b_k = k_cw0 * k_s1 + k_cw1 * k_s2 + k_cw2 * k_s3 + k_cw3 * acc_k
        b_k = b_k / (1.0 + tl.exp(-b_k))
        b_k = b_k.to(o_ptr.dtype.element_ty).to(tl.float32)

        v0_s1 = tl.load(conv_state_ptr + v0_state_base + 1 * conv_state_stride_w, mask=mask_v, other=0.0).to(tl.float32)
        v0_s2 = tl.load(conv_state_ptr + v0_state_base + 2 * conv_state_stride_w, mask=mask_v, other=0.0).to(tl.float32)
        v0_s3 = tl.load(conv_state_ptr + v0_state_base + 3 * conv_state_stride_w, mask=mask_v, other=0.0).to(tl.float32)
        v0_cw0 = tl.load(conv_w_ptr + v0_row * CONV_WIDTH + 0, mask=mask_v, other=0.0).to(tl.float32)
        v0_cw1 = tl.load(conv_w_ptr + v0_row * CONV_WIDTH + 1, mask=mask_v, other=0.0).to(tl.float32)
        v0_cw2 = tl.load(conv_w_ptr + v0_row * CONV_WIDTH + 2, mask=mask_v, other=0.0).to(tl.float32)
        v0_cw3 = tl.load(conv_w_ptr + v0_row * CONV_WIDTH + 3, mask=mask_v, other=0.0).to(tl.float32)
        b_v0 = v0_cw0 * v0_s1 + v0_cw1 * v0_s2 + v0_cw2 * v0_s3 + v0_cw3 * acc_v0
        b_v0 = b_v0 / (1.0 + tl.exp(-b_v0))
        b_v0 = b_v0.to(o_ptr.dtype.element_ty).to(tl.float32)

        v1_s1 = tl.load(conv_state_ptr + v1_state_base + 1 * conv_state_stride_w, mask=mask_v, other=0.0).to(tl.float32)
        v1_s2 = tl.load(conv_state_ptr + v1_state_base + 2 * conv_state_stride_w, mask=mask_v, other=0.0).to(tl.float32)
        v1_s3 = tl.load(conv_state_ptr + v1_state_base + 3 * conv_state_stride_w, mask=mask_v, other=0.0).to(tl.float32)
        v1_cw0 = tl.load(conv_w_ptr + v1_row * CONV_WIDTH + 0, mask=mask_v, other=0.0).to(tl.float32)
        v1_cw1 = tl.load(conv_w_ptr + v1_row * CONV_WIDTH + 1, mask=mask_v, other=0.0).to(tl.float32)
        v1_cw2 = tl.load(conv_w_ptr + v1_row * CONV_WIDTH + 2, mask=mask_v, other=0.0).to(tl.float32)
        v1_cw3 = tl.load(conv_w_ptr + v1_row * CONV_WIDTH + 3, mask=mask_v, other=0.0).to(tl.float32)
        b_v1 = v1_cw0 * v1_s1 + v1_cw1 * v1_s2 + v1_cw2 * v1_s3 + v1_cw3 * acc_v1
        b_v1 = b_v1 / (1.0 + tl.exp(-b_v1))
        b_v1 = b_v1.to(o_ptr.dtype.element_ty).to(tl.float32)

        tl.store(conv_state_ptr + q_state_base + 0 * conv_state_stride_w, q_s1, mask=mask_k)
        tl.store(conv_state_ptr + q_state_base + 1 * conv_state_stride_w, q_s2, mask=mask_k)
        tl.store(conv_state_ptr + q_state_base + 2 * conv_state_stride_w, q_s3, mask=mask_k)
        tl.store(conv_state_ptr + q_state_base + 3 * conv_state_stride_w, acc_q, mask=mask_k)
        tl.store(conv_state_ptr + k_state_base + 0 * conv_state_stride_w, k_s1, mask=mask_k)
        tl.store(conv_state_ptr + k_state_base + 1 * conv_state_stride_w, k_s2, mask=mask_k)
        tl.store(conv_state_ptr + k_state_base + 2 * conv_state_stride_w, k_s3, mask=mask_k)
        tl.store(conv_state_ptr + k_state_base + 3 * conv_state_stride_w, acc_k, mask=mask_k)
        tl.store(conv_state_ptr + v0_state_base + 0 * conv_state_stride_w, v0_s1, mask=mask_v)
        tl.store(conv_state_ptr + v0_state_base + 1 * conv_state_stride_w, v0_s2, mask=mask_v)
        tl.store(conv_state_ptr + v0_state_base + 2 * conv_state_stride_w, v0_s3, mask=mask_v)
        tl.store(conv_state_ptr + v0_state_base + 3 * conv_state_stride_w, acc_v0, mask=mask_v)
        tl.store(conv_state_ptr + v1_state_base + 0 * conv_state_stride_w, v1_s1, mask=mask_v)
        tl.store(conv_state_ptr + v1_state_base + 1 * conv_state_stride_w, v1_s2, mask=mask_v)
        tl.store(conv_state_ptr + v1_state_base + 2 * conv_state_stride_w, v1_s3, mask=mask_v)
        tl.store(conv_state_ptr + v1_state_base + 3 * conv_state_stride_w, acc_v1, mask=mask_v)

        acc_a0 = acc_a0.to(o_ptr.dtype.element_ty).to(tl.float32)
        acc_a1 = acc_a1.to(o_ptr.dtype.element_ty).to(tl.float32)
        acc_b0 = acc_b0.to(o_ptr.dtype.element_ty).to(tl.float32)
        acc_b1 = acc_b1.to(o_ptr.dtype.element_ty).to(tl.float32)

        if USE_QK_L2NORM:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        p_down = w_down_ptr + o_r[:, None] * HEAD_K_DIM + o_k[None, :]
        b_down = tl.load(p_down, mask=mask_r[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
        b_lowrank = tl.sum(b_down * b_k[None, :], axis=1)

        p_up = w_up_ptr + o_r[:, None] + o_v[None, :] * R
        b_up_w = tl.load(p_up, mask=mask_r[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
        b_beta_delta = tl.sum(b_up_w * b_lowrank[:, None], axis=0)
        b_beta_delta += tl.load(b_up_ptr + o_v, mask=mask_v, other=0.0).to(tl.float32)
        b_beta_delta = 2.0 / (1.0 + tl.exp(-2.0 * b_beta_delta)) - 1.0

        z_base = i_batch * NUM_V_HEADS * HEAD_V_DIM
        tl.store(z_out_ptr + z_base + hv0 * HEAD_V_DIM + o_v, acc_z0.to(z_out_ptr.dtype.element_ty), mask=mask_v)
        tl.store(z_out_ptr + z_base + hv1 * HEAD_V_DIM + o_v, acc_z1.to(z_out_ptr.dtype.element_ty), mask=mask_v)

        recurrent_batch_base: tl.constexpr = NUM_V_HEADS * HEAD_K_DIM * HEAD_V_DIM
        state_base0 = i_batch * recurrent_batch_base + hv0 * HEAD_K_DIM * HEAD_V_DIM
        p_state0 = recurrent_state_ptr + state_base0 + o_k[:, None] * HEAD_V_DIM + o_v[None, :]
        b_h0 = tl.load(p_state0, mask=mask_h, other=0.0).to(tl.float32)
        b_beta_vec0 = 1.0 / (1.0 + tl.exp(-(acc_b0 + b_beta_delta)))
        b_gate_input0 = acc_a0 + tl.load(dt_bias_ptr + hv0).to(tl.float32)
        b_softplus0 = tl.where(b_gate_input0 > 20.0, b_gate_input0, tl.log(1.0 + tl.exp(b_gate_input0)))
        b_g0 = -tl.exp(tl.load(a_log_ptr + hv0).to(tl.float32)) * b_softplus0
        b_h0 = b_h0 * tl.exp(b_g0)
        b_proj0 = tl.sum(b_h0 * b_k[:, None], axis=0)
        b_delta0 = b_beta_vec0 * (b_v0 - b_proj0)
        b_h0 = b_h0 + b_k[:, None] * b_delta0[None, :]
        b_o0 = tl.sum(b_h0 * b_q[:, None], axis=0)
        tl.store(o_ptr + z_base + hv0 * HEAD_V_DIM + o_v, b_o0.to(o_ptr.dtype.element_ty), mask=mask_v)
        tl.store(p_state0, b_h0, mask=mask_h)

        state_base1 = i_batch * recurrent_batch_base + hv1 * HEAD_K_DIM * HEAD_V_DIM
        p_state1 = recurrent_state_ptr + state_base1 + o_k[:, None] * HEAD_V_DIM + o_v[None, :]
        b_h1 = tl.load(p_state1, mask=mask_h, other=0.0).to(tl.float32)
        b_beta_vec1 = 1.0 / (1.0 + tl.exp(-(acc_b1 + b_beta_delta)))
        b_gate_input1 = acc_a1 + tl.load(dt_bias_ptr + hv1).to(tl.float32)
        b_softplus1 = tl.where(b_gate_input1 > 20.0, b_gate_input1, tl.log(1.0 + tl.exp(b_gate_input1)))
        b_g1 = -tl.exp(tl.load(a_log_ptr + hv1).to(tl.float32)) * b_softplus1
        b_h1 = b_h1 * tl.exp(b_g1)
        b_proj1 = tl.sum(b_h1 * b_k[:, None], axis=0)
        b_delta1 = b_beta_vec1 * (b_v1 - b_proj1)
        b_h1 = b_h1 + b_k[:, None] * b_delta1[None, :]
        b_o1 = tl.sum(b_h1 * b_q[:, None], axis=0)
        tl.store(o_ptr + z_base + hv1 * HEAD_V_DIM + o_v, b_o1.to(o_ptr.dtype.element_ty), mask=mask_v)
        tl.store(p_state1, b_h1, mask=mask_h)


def qwen35_grouped_projection_conv_lowrank_deltanet(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    w_qkv: torch.Tensor,
    w_z: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    conv_weight: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    b_up: torch.Tensor,
    recurrent_state: torch.Tensor,
    *,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    use_qk_l2norm: bool = True,
    block_d: int = 64,
    num_warps: int = 8,
    num_stages: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused decode-only projection+conv pack followed by grouped-QK low-rank DeltaNet."""
    if triton is None:
        raise RuntimeError("Triton is not available in this environment")
    tensors = (
        hidden_states,
        conv_state,
        w_qkv,
        w_z,
        w_a,
        w_b,
        conv_weight,
        a_log,
        dt_bias,
        w_down,
        w_up,
        b_up,
        recurrent_state,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("All tensors must be CUDA tensors")
    if hidden_states.ndim == 3:
        if hidden_states.shape[1] != 1:
            raise ValueError("hidden_states must have seq_len=1")
        hidden_flat = hidden_states[:, 0].contiguous()
    elif hidden_states.ndim == 2:
        hidden_flat = hidden_states.contiguous()
    else:
        raise ValueError("hidden_states must have shape [B, D] or [B, 1, D]")
    if not recurrent_state.is_contiguous():
        raise ValueError("recurrent_state must be contiguous so in-place updates reach the cache")
    if conv_state.shape[-1] != 4 or conv_weight.shape[-1] != 4:
        raise ValueError("The fused packed kernel supports Qwen3.5 conv width 4 only")
    if num_v_heads % num_k_heads != 0:
        raise ValueError("num_v_heads must be an integer multiple of num_k_heads")
    qk_repeat = num_v_heads // num_k_heads
    if qk_repeat != 2:
        raise ValueError("The fused packed kernel currently supports Qwen3.5 q/k repeat factor 2 only")

    batch, d_model = hidden_flat.shape
    rank = w_down.shape[0]
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_dim = key_dim * 2 + value_dim
    if conv_state.shape != (batch, conv_dim, 4):
        raise ValueError("conv_state must have shape [B, 2*key_dim + value_dim, 4]")
    if recurrent_state.shape != (batch, num_v_heads, head_k_dim, head_v_dim):
        raise ValueError("recurrent_state must have shape [B, HV, K, V]")
    if w_qkv.shape != (conv_dim, d_model):
        raise ValueError("w_qkv has an unexpected shape")
    if w_z.shape != (value_dim, d_model):
        raise ValueError("w_z has an unexpected shape")
    if w_a.shape != (num_v_heads, d_model):
        raise ValueError("w_a has an unexpected shape")
    if w_b.shape != (num_v_heads, d_model):
        raise ValueError("w_b has an unexpected shape")
    if conv_weight.shape != (conv_dim, 4):
        raise ValueError("conv_weight has an unexpected shape")
    if a_log.shape != (num_v_heads,) or dt_bias.shape != (num_v_heads,):
        raise ValueError("a_log and dt_bias must have shape [HV]")
    if rank <= 0:
        raise ValueError("w_down rank dimension must be positive")
    if w_down.shape != (rank, head_k_dim):
        raise ValueError("w_down must have shape [R, K]")
    if w_up.shape != (head_v_dim, rank):
        raise ValueError("w_up must have shape [V, R]")
    if b_up.shape != (head_v_dim,):
        raise ValueError("b_up must have shape [V]")
    if block_d <= 0 or num_warps <= 0 or num_stages <= 0:
        raise ValueError("block_d, num_warps, and num_stages must be positive")

    w_qkv = w_qkv.contiguous()
    w_z = w_z.contiguous()
    w_a = w_a.contiguous()
    w_b = w_b.contiguous()
    conv_weight = conv_weight.contiguous()
    a_log = a_log.contiguous()
    dt_bias = dt_bias.contiguous()
    w_down = w_down.contiguous()
    w_up = w_up.contiguous()
    b_up = b_up.contiguous()

    output_dtype = hidden_flat.dtype
    core_attn_out = torch.empty(
        batch,
        num_v_heads,
        head_v_dim,
        dtype=output_dtype,
        device=hidden_flat.device,
    )
    z = torch.empty_like(core_attn_out)
    bk = triton.next_power_of_2(head_k_dim)
    bv = triton.next_power_of_2(head_v_dim)
    br = triton.next_power_of_2(rank)
    scale = 1.0 / math.sqrt(head_k_dim)
    qwen35_grouped_projection_conv_lowrank_deltanet_kernel[(num_k_heads, batch)](
        x_ptr=hidden_flat,
        w_qkv_ptr=w_qkv,
        w_z_ptr=w_z,
        w_a_ptr=w_a,
        w_b_ptr=w_b,
        conv_w_ptr=conv_weight,
        conv_state_ptr=conv_state,
        conv_state_stride_b=conv_state.stride(0),
        conv_state_stride_c=conv_state.stride(1),
        conv_state_stride_w=conv_state.stride(2),
        a_log_ptr=a_log,
        dt_bias_ptr=dt_bias,
        w_down_ptr=w_down,
        w_up_ptr=w_up,
        b_up_ptr=b_up,
        recurrent_state_ptr=recurrent_state,
        o_ptr=core_attn_out,
        z_out_ptr=z,
        scale=scale,
        D_MODEL=d_model,
        NUM_K_HEADS=num_k_heads,
        NUM_V_HEADS=num_v_heads,
        HEAD_K_DIM=head_k_dim,
        HEAD_V_DIM=head_v_dim,
        QK_REPEAT=qk_repeat,
        CONV_DIM=conv_dim,
        CONV_WIDTH=4,
        R=rank,
        BK=bk,
        BV=bv,
        BR=br,
        BLOCK_D=block_d,
        USE_QK_L2NORM=use_qk_l2norm,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return core_attn_out, z
