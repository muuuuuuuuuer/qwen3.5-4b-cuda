from __future__ import annotations

import math
from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - exercised only in non-Triton environments
    triton = None
    tl = None


@dataclass(frozen=True)
class DeltaNetKernelConfig:
    bv: int = 32
    num_warps: int = 4
    num_stages: int = 1


DEFAULT_DELTANET_KERNEL_CONFIG = DeltaNetKernelConfig()
DEFAULT_FUSED_GATE_KERNEL_CONFIG = DeltaNetKernelConfig(bv=64, num_warps=8, num_stages=1)


_SUPPORTED_STATE_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def _normalize_kernel_config(
    kernel_config: DeltaNetKernelConfig | None,
    v_dim: int,
) -> DeltaNetKernelConfig:
    config = kernel_config or DEFAULT_DELTANET_KERNEL_CONFIG
    if config.bv <= 0 or config.bv > v_dim:
        raise ValueError("kernel_config.bv must be in the range [1, V]")
    if config.bv & (config.bv - 1):
        raise ValueError("kernel_config.bv must be a power of 2")
    if config.num_warps <= 0:
        raise ValueError("kernel_config.num_warps must be positive")
    if config.num_stages <= 0:
        raise ValueError("kernel_config.num_stages must be positive")
    return config


def deltanet_decode_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    use_qk_l2norm: bool = True,
) -> torch.Tensor:
    """
    Pure PyTorch decode-step reference for Gated DeltaNet.
    The recurrent state is updated in place.
    """
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("q, k, and v must have shape [HV, D]")
    if state.ndim != 3:
        raise ValueError("state must have shape [HV, K, V]")
    if state.dtype not in _SUPPORTED_STATE_DTYPES:
        raise ValueError("state must be float16, bfloat16, or float32")

    hv, k_dim = q.shape
    v_dim = v.shape[-1]
    if k.shape != (hv, k_dim):
        raise ValueError("k must match q shape")
    if v.shape[0] != hv:
        raise ValueError("v must have the same head dimension as q")
    if state.shape != (hv, k_dim, v_dim):
        raise ValueError("state shape must match [HV, K, V]")
    if g.shape != (hv,) or beta.shape != (hv,):
        raise ValueError("g and beta must have shape [HV]")

    scale = 1.0 / math.sqrt(k_dim)
    output = torch.empty(hv, v_dim, dtype=torch.float32, device=q.device)

    for head in range(hv):
        q_head = q[head].float()
        k_head = k[head].float()
        if use_qk_l2norm:
            q_head = q_head / (q_head.norm(p=2) + 1e-6)
            k_head = k_head / (k_head.norm(p=2) + 1e-6)
        q_head = q_head * scale

        v_head = v[head].float()
        state_head = state[head].float()
        state_head.mul_(torch.exp(g[head].float()))
        projection = state_head.transpose(0, 1) @ k_head
        delta = beta[head].float() * (v_head - projection)
        state_head.add_(k_head.unsqueeze(1) * delta.unsqueeze(0))
        state[head].copy_(state_head.to(state.dtype))
        output[head] = state_head.transpose(0, 1) @ q_head

    return output


if triton is not None:

    @triton.jit
    def deltanet_decode_fused_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        state_ptr,
        o_ptr,
        scale,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        USE_QK_L2NORM: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        i_h = tl.program_id(1)

        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_k[:, None] & mask_v[None, :]

        b_q = tl.load(q_ptr + i_h * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
        b_k = tl.load(k_ptr + i_h * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
        if USE_QK_L2NORM:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        b_v = tl.load(v_ptr + i_h * V + o_v, mask=mask_v, other=0.0).to(tl.float32)
        b_g = tl.load(g_ptr + i_h).to(tl.float32)
        b_beta = tl.load(beta_ptr + i_h).to(tl.float32)

        state_offset = i_h * K * V
        p_state = state_ptr + state_offset + o_k[:, None] * V + o_v[None, :]
        b_h = tl.load(p_state, mask=mask_h, other=0.0).to(tl.float32)

        b_h = b_h * tl.exp(b_g)
        b_proj = tl.sum(b_h * b_k[:, None], axis=0)
        b_delta = b_beta * (b_v - b_proj)
        b_h = b_h + b_k[:, None] * b_delta[None, :]
        b_o = tl.sum(b_h * b_q[:, None], axis=0)

        tl.store(o_ptr + i_h * V + o_v, b_o.to(o_ptr.dtype.element_ty), mask=mask_v)
        tl.store(p_state, b_h, mask=mask_h)


    @triton.jit
    def deltanet_decode_fused_with_gates_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        a_ptr,
        b_ptr,
        a_log_ptr,
        dt_bias_ptr,
        state_ptr,
        o_ptr,
        scale,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        USE_QK_L2NORM: tl.constexpr,
    ):
        i_v = tl.program_id(0)
        i_h = tl.program_id(1)

        o_k = tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_k[:, None] & mask_v[None, :]

        b_q = tl.load(q_ptr + i_h * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
        b_k = tl.load(k_ptr + i_h * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
        if USE_QK_L2NORM:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        b_v = tl.load(v_ptr + i_h * V + o_v, mask=mask_v, other=0.0).to(tl.float32)
        b_a = tl.load(a_ptr + i_h).to(tl.float32)
        b_b = tl.load(b_ptr + i_h).to(tl.float32)
        b_a_log = tl.load(a_log_ptr + i_h).to(tl.float32)
        b_dt_bias = tl.load(dt_bias_ptr + i_h).to(tl.float32)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))
        b_gate_input = b_a + b_dt_bias
        b_softplus = tl.where(b_gate_input > 20.0, b_gate_input, tl.log(1.0 + tl.exp(b_gate_input)))
        b_g = -tl.exp(b_a_log) * b_softplus

        state_offset = i_h * K * V
        p_state = state_ptr + state_offset + o_k[:, None] * V + o_v[None, :]
        b_h = tl.load(p_state, mask=mask_h, other=0.0).to(tl.float32)

        b_h = b_h * tl.exp(b_g)
        b_proj = tl.sum(b_h * b_k[:, None], axis=0)
        b_delta = b_beta * (b_v - b_proj)
        b_h = b_h + b_k[:, None] * b_delta[None, :]
        b_o = tl.sum(b_h * b_q[:, None], axis=0)

        tl.store(o_ptr + i_h * V + o_v, b_o.to(o_ptr.dtype.element_ty), mask=mask_v)
        tl.store(p_state, b_h, mask=mask_h)


def deltanet_decode_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    use_qk_l2norm: bool = True,
    kernel_config: DeltaNetKernelConfig | None = None,
) -> torch.Tensor:
    """
    Single-token decode step for Gated DeltaNet.
    Returns output [HV, V] and updates `state` in place.
    """
    if triton is None:
        raise RuntimeError("Triton is not available in this environment")
    if not all(tensor.is_cuda for tensor in (q, k, v, g, beta, state)):
        raise ValueError("All tensors must be CUDA tensors")
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("q, k, and v must have shape [HV, D]")
    if state.ndim != 3:
        raise ValueError("state must have shape [HV, K, V]")
    if state.dtype not in _SUPPORTED_STATE_DTYPES:
        raise ValueError("state must be float16, bfloat16, or float32")

    hv, k_dim = q.shape
    v_dim = v.shape[-1]
    if k.shape != (hv, k_dim):
        raise ValueError("k must match q shape")
    if v.shape != (hv, v_dim):
        raise ValueError("v must have shape [HV, V]")
    if g.shape != (hv,) or beta.shape != (hv,):
        raise ValueError("g and beta must have shape [HV]")
    if state.shape != (hv, k_dim, v_dim):
        raise ValueError("state shape must match [HV, K, V]")
    config = _normalize_kernel_config(kernel_config, v_dim)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    state = state.contiguous()

    output = torch.empty(hv, v_dim, dtype=q.dtype, device=q.device)
    bk = triton.next_power_of_2(k_dim)
    bv = config.bv
    nv = triton.cdiv(v_dim, bv)
    scale = 1.0 / math.sqrt(k_dim)

    deltanet_decode_fused_kernel[(nv, hv)](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        g_ptr=g,
        beta_ptr=beta,
        state_ptr=state,
        o_ptr=output,
        scale=scale,
        K=k_dim,
        V=v_dim,
        BK=bk,
        BV=bv,
        USE_QK_L2NORM=use_qk_l2norm,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return output


def deltanet_decode_step_fused_gates(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    use_qk_l2norm: bool = True,
    kernel_config: DeltaNetKernelConfig | None = None,
) -> torch.Tensor:
    """
    Single-token decode step with gate computation fused inside the Triton kernel.
    Returns output [HV, V] and updates `state` in place.
    """
    if triton is None:
        raise RuntimeError("Triton is not available in this environment")
    if not all(tensor.is_cuda for tensor in (q, k, v, a, b, a_log, dt_bias, state)):
        raise ValueError("All tensors must be CUDA tensors")
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("q, k, and v must have shape [HV, D]")
    if state.ndim != 3:
        raise ValueError("state must have shape [HV, K, V]")
    if state.dtype not in _SUPPORTED_STATE_DTYPES:
        raise ValueError("state must be float16, bfloat16, or float32")

    hv, k_dim = q.shape
    v_dim = v.shape[-1]
    if k.shape != (hv, k_dim):
        raise ValueError("k must match q shape")
    if v.shape != (hv, v_dim):
        raise ValueError("v must have shape [HV, V]")
    if state.shape != (hv, k_dim, v_dim):
        raise ValueError("state shape must match [HV, K, V]")
    if any(tensor.shape != (hv,) for tensor in (a, b, a_log, dt_bias)):
        raise ValueError("a, b, a_log, and dt_bias must have shape [HV]")
    config = _normalize_kernel_config(kernel_config or DEFAULT_FUSED_GATE_KERNEL_CONFIG, v_dim)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    a_log = a_log.contiguous()
    dt_bias = dt_bias.contiguous()
    state = state.contiguous()

    output = torch.empty(hv, v_dim, dtype=q.dtype, device=q.device)
    bk = triton.next_power_of_2(k_dim)
    bv = config.bv
    nv = triton.cdiv(v_dim, bv)
    scale = 1.0 / math.sqrt(k_dim)

    deltanet_decode_fused_with_gates_kernel[(nv, hv)](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        a_ptr=a,
        b_ptr=b,
        a_log_ptr=a_log,
        dt_bias_ptr=dt_bias,
        state_ptr=state,
        o_ptr=output,
        scale=scale,
        K=k_dim,
        V=v_dim,
        BK=bk,
        BV=bv,
        USE_QK_L2NORM=use_qk_l2norm,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return output
