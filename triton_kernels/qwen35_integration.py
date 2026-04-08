from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    apply_mask_to_padding_states,
    causal_conv1d_fn as fla_causal_conv1d_fn,
    causal_conv1d_update as fla_causal_conv1d_update,
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule as fla_fused_recurrent_gated_delta_rule,
    torch_causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

from triton_kernels.deltanet_decode import (
    DEFAULT_DELTANET_KERNEL_CONFIG,
    DEFAULT_FUSED_GATE_KERNEL_CONFIG,
    deltanet_decode_step,
    deltanet_decode_step_fused_gates,
)


_PATCH_STATS = {
    "calls": 0,
    "batch_items": 0,
    "tokens": 0,
}

_RUNTIME_MODES = {"torch", "fla", "triton", "triton_base", "triton_fused"}


def reset_qwen35_triton_patch_stats() -> None:
    _PATCH_STATS["calls"] = 0
    _PATCH_STATS["batch_items"] = 0
    _PATCH_STATS["tokens"] = 0


def get_qwen35_triton_patch_stats() -> dict[str, int]:
    return dict(_PATCH_STATS)


def _record_patch_call(batch_size: int, seq_len: int) -> None:
    _PATCH_STATS["calls"] += 1
    _PATCH_STATS["batch_items"] += batch_size
    _PATCH_STATS["tokens"] += int(batch_size * seq_len)


def _iter_qwen35_linear_attn_modules(model: Any) -> list[Any]:
    candidate_roots = []
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        candidate_roots.append(model.model.language_model)
    if hasattr(model, "language_model"):
        candidate_roots.append(model.language_model)

    for root in candidate_roots:
        layers = getattr(root, "layers", None)
        if layers is None:
            continue
        modules = [getattr(layer, "linear_attn", None) for layer in layers]
        modules = [module for module in modules if module is not None]
        if modules:
            return modules
    raise ValueError("No Qwen3.5 linear_attn layers were found")


def _ensure_original_forward(linear_attn: Any) -> None:
    if not hasattr(linear_attn, "_original_forward"):
        linear_attn._original_forward = linear_attn.forward


def _restore_original_forward(linear_attn: Any) -> None:
    if hasattr(linear_attn, "_original_forward"):
        linear_attn.forward = linear_attn._original_forward


def _require_fla_fast_path() -> None:
    required = (
        fla_causal_conv1d_fn,
        fla_causal_conv1d_update,
        fla_chunk_gated_delta_rule,
        fla_fused_recurrent_gated_delta_rule,
    )
    if not all(required):
        raise RuntimeError("FLA fast path is unavailable; install flash-linear-attention and causal-conv1d first")


def qwen35_triton_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Decode-only adapter matching the FLA recurrent API used by Qwen3.5.
    Kept for targeted experiments and correctness checks; the end-to-end Triton runtime
    uses `qwen35_triton_decode_forward` to avoid extra Python overhead in the hot path.
    """
    if gk is not None or gv is not None or cu_seqlens is not None or transpose_state_layout:
        raise NotImplementedError("This adapter only supports the decode-only Qwen3.5 recurrent path")
    if g is None or beta is None:
        raise ValueError("g and beta are required")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [B, T, H/HV, D]")
    if q.shape[1] != 1 or k.shape[1] != 1 or v.shape[1] != 1:
        raise ValueError("This adapter only supports decode steps with T=1")

    batch_size = q.shape[0]
    _record_patch_call(batch_size, q.shape[1])

    hv = v.shape[2]
    k_dim = q.shape[-1]
    v_dim = v.shape[-1]

    if initial_state is None:
        state = torch.zeros(batch_size, hv, k_dim, v_dim, dtype=torch.float32, device=q.device)
    else:
        state = initial_state.clone()

    output = torch.empty(batch_size, 1, hv, v_dim, dtype=v.dtype, device=v.device)
    for batch_idx in range(batch_size):
        output[batch_idx, 0] = deltanet_decode_step(
            q[batch_idx, 0],
            k[batch_idx, 0],
            v[batch_idx, 0],
            g[batch_idx, 0],
            beta[batch_idx, 0],
            state[batch_idx],
            use_qk_l2norm=use_qk_l2norm_in_kernel,
        )

    final_state = state if output_final_state else None
    return output, final_state


def _qwen35_triton_decode_forward_impl(
    self,
    hidden_states: torch.Tensor,
    cache_params: Any | None = None,
    attention_mask: torch.Tensor | None = None,
    *,
    fuse_gates: bool,
):
    """
    Decode-specialized Qwen3.5 DeltaNet forward implementation.
    Prefill falls back to the original implementation; decode uses a Triton kernel directly from
    inside the layer to avoid the extra Python adapter and state clone overhead.
    """
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape

    use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx) and seq_len == 1
    if not use_precomputed_states:
        return self._original_forward(
            hidden_states=hidden_states,
            cache_params=cache_params,
            attention_mask=attention_mask,
        )

    conv_state = cache_params.layers[self.layer_idx].conv_states
    recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states).reshape(batch_size, seq_len, self.num_v_heads)
    a = self.in_proj_a(hidden_states).reshape(batch_size, seq_len, self.num_v_heads)

    mixed_qkv = self.causal_conv1d_update(
        mixed_qkv,
        conv_state,
        self.conv1d.weight.squeeze(1),
        self.conv1d.bias,
        self.activation,
    )
    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    if self.num_v_heads // self.num_k_heads > 1:
        repeat_factor = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(repeat_factor, dim=2)
        key = key.repeat_interleave(repeat_factor, dim=2)

    _record_patch_call(batch_size, seq_len)
    core_attn_out = torch.empty(batch_size, seq_len, self.num_v_heads, self.head_v_dim, dtype=value.dtype, device=value.device)
    default_kernel_config = DEFAULT_FUSED_GATE_KERNEL_CONFIG if fuse_gates else DEFAULT_DELTANET_KERNEL_CONFIG
    kernel_config = getattr(self, "_triton_kernel_config", default_kernel_config)

    if not fuse_gates:
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

    for batch_idx in range(batch_size):
        if fuse_gates:
            core_attn_out[batch_idx, 0] = deltanet_decode_step_fused_gates(
                query[batch_idx, 0],
                key[batch_idx, 0],
                value[batch_idx, 0],
                a[batch_idx, 0],
                b[batch_idx, 0],
                self.A_log,
                self.dt_bias,
                recurrent_state[batch_idx],
                use_qk_l2norm=True,
                kernel_config=kernel_config,
            )
        else:
            core_attn_out[batch_idx, 0] = deltanet_decode_step(
                query[batch_idx, 0],
                key[batch_idx, 0],
                value[batch_idx, 0],
                g[batch_idx, 0],
                beta[batch_idx, 0],
                recurrent_state[batch_idx],
                use_qk_l2norm=True,
                kernel_config=kernel_config,
            )

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    return self.out_proj(core_attn_out)


def qwen35_triton_decode_forward_base(
    self,
    hidden_states: torch.Tensor,
    cache_params: Any | None = None,
    attention_mask: torch.Tensor | None = None,
):
    return _qwen35_triton_decode_forward_impl(
        self,
        hidden_states=hidden_states,
        cache_params=cache_params,
        attention_mask=attention_mask,
        fuse_gates=False,
    )


def qwen35_triton_decode_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params: Any | None = None,
    attention_mask: torch.Tensor | None = None,
):
    return _qwen35_triton_decode_forward_impl(
        self,
        hidden_states=hidden_states,
        cache_params=cache_params,
        attention_mask=attention_mask,
        fuse_gates=True,
    )


def apply_qwen35_deltanet_triton_patch(model: Any, patch_mode: str = "recurrent_only") -> int:
    """
    Patch Qwen3.5 linear attention modules to use Triton.

    patch_mode:
      - "recurrent_only": replace `recurrent_gated_delta_rule`
      - "full_forward": replace the whole decode forward path with the fused-gates Triton kernel
      - "full_forward_base": replace the whole decode forward path with the base Triton kernel
      - "full_forward_fused": alias for "full_forward"
    """
    linear_attn_modules = _iter_qwen35_linear_attn_modules(model)
    patched = 0
    for linear_attn in linear_attn_modules:
        _ensure_original_forward(linear_attn)
        if patch_mode == "recurrent_only":
            linear_attn.recurrent_gated_delta_rule = qwen35_triton_recurrent_gated_delta_rule
        elif patch_mode == "full_forward_base":
            linear_attn.recurrent_gated_delta_rule = qwen35_triton_recurrent_gated_delta_rule
            linear_attn.forward = types.MethodType(qwen35_triton_decode_forward_base, linear_attn)
        elif patch_mode in {"full_forward", "full_forward_fused"}:
            linear_attn.recurrent_gated_delta_rule = qwen35_triton_recurrent_gated_delta_rule
            linear_attn.forward = types.MethodType(qwen35_triton_decode_forward, linear_attn)
        else:
            raise ValueError(f"Unsupported patch_mode: {patch_mode}")
        patched += 1
    return patched


def configure_qwen35_deltanet_runtime(model: Any, mode: str) -> int:
    """
    Configure Qwen3.5 DeltaNet runtime for end-to-end comparisons.

    Modes:
      - "torch": force upstream torch fallback
      - "fla": use upstream FLA fast path
      - "triton": alias for "triton_fused"
      - "triton_base": use upstream FLA conv/chunk path plus Triton decode forward with external gate ops
      - "triton_fused": use upstream FLA conv/chunk path plus Triton decode forward with fused gate ops
    """
    if mode not in _RUNTIME_MODES:
        raise ValueError(f"Unsupported mode: {mode}")

    normalized_mode = "triton_fused" if mode == "triton" else mode

    linear_attn_modules = _iter_qwen35_linear_attn_modules(model)
    for linear_attn in linear_attn_modules:
        _ensure_original_forward(linear_attn)
        _restore_original_forward(linear_attn)

        if normalized_mode == "torch":
            linear_attn.causal_conv1d_fn = None
            linear_attn.causal_conv1d_update = torch_causal_conv1d_update
            linear_attn.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
            linear_attn.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
            continue

        _require_fla_fast_path()
        linear_attn.causal_conv1d_fn = fla_causal_conv1d_fn
        linear_attn.causal_conv1d_update = fla_causal_conv1d_update
        linear_attn.chunk_gated_delta_rule = fla_chunk_gated_delta_rule
        linear_attn.recurrent_gated_delta_rule = fla_fused_recurrent_gated_delta_rule

        if normalized_mode == "triton_base":
            linear_attn.recurrent_gated_delta_rule = qwen35_triton_recurrent_gated_delta_rule
            linear_attn.forward = types.MethodType(qwen35_triton_decode_forward_base, linear_attn)
        elif normalized_mode == "triton_fused":
            linear_attn.recurrent_gated_delta_rule = qwen35_triton_recurrent_gated_delta_rule
            linear_attn.forward = types.MethodType(qwen35_triton_decode_forward, linear_attn)

    return len(linear_attn_modules)


def describe_qwen35_deltanet_runtime(model: Any) -> dict[str, str]:
    linear_attn = _iter_qwen35_linear_attn_modules(model)[0]
    forward = getattr(linear_attn.forward, "__func__", linear_attn.forward)
    return {
        "forward_impl": getattr(forward, "__name__", type(forward).__name__),
        "causal_conv1d_update_impl": getattr(linear_attn.causal_conv1d_update, "__name__", type(linear_attn.causal_conv1d_update).__name__),
        "chunk_impl": getattr(linear_attn.chunk_gated_delta_rule, "__name__", type(linear_attn.chunk_gated_delta_rule).__name__),
        "recurrent_impl": getattr(linear_attn.recurrent_gated_delta_rule, "__name__", type(linear_attn.recurrent_gated_delta_rule).__name__),
    }
