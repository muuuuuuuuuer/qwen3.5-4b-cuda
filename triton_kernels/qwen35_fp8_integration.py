"""FP8 integration for Qwen3.5-4B linear layers.

Monkey-patches nn.Linear layers to use the FP8 GEMV Triton kernel for
batch-1 decode, while keeping the original PyTorch path for prefill.

Target layers (configurable, default: all FFN + lm_head):
  - MLP gate_proj / up_proj / down_proj (every layer, N >= 2560)
  - lm_head [vocab_size, hidden_size]

Usage:
    from triton_kernels.qwen35_fp8_integration import apply_qwen35_fp8, restore_qwen35_fp8

    model, tokenizer, config = load_model_and_tokenizer()
    stats = apply_qwen35_fp8(model)  # quantizes target layers, patches forward
    # ... run benchmarks ...
    restore_qwen35_fp8(model)  # restore original forward methods
"""

from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantize_fp8_weights import quantize_weight_fp8, compute_quantization_error
from triton_kernels.fp8_gemv import fp8_gemv


_PATCHED_LAYERS: list[nn.Linear] = []

FP8_INTEGRATION_STATS: dict[str, int] = {
    "calls": 0,
    "fp8_calls": 0,
    "fallback_calls": 0,
}

# Quantization error summary per layer
_FP8_QUANT_ERRORS: dict[str, dict] = {}


def reset_fp8_integration_stats() -> None:
    FP8_INTEGRATION_STATS["calls"] = 0
    FP8_INTEGRATION_STATS["fp8_calls"] = 0
    FP8_INTEGRATION_STATS["fallback_calls"] = 0


def get_fp8_integration_stats() -> dict[str, int]:
    return dict(FP8_INTEGRATION_STATS)


def get_fp8_quantization_errors() -> dict[str, dict]:
    return dict(_FP8_QUANT_ERRORS)


def _make_fp8_linear_forward(
    layer: nn.Linear,
    use_autotune: bool = True,
) -> types.MethodType:
    """Create a replacement forward method that uses FP8 GEMV for batch-1 decode."""

    w_fp8 = layer._fp8_weight
    scale = layer._fp8_scale
    bias = layer.bias

    def fp8_forward(self: nn.Linear, input: torch.Tensor) -> torch.Tensor:
        FP8_INTEGRATION_STATS["calls"] += 1

        # Decode: input is [batch=1, seq=1, hidden] → flatten to [1, hidden]
        # Prefill: input is [batch=1, seq>1, hidden] → fallback
        is_decode = (
            input.ndim >= 2
            and int(input.shape[0]) == 1
            and (input.ndim == 2 or int(input.shape[1]) == 1)
        )

        if is_decode:
            x = input.reshape(-1, input.shape[-1]).squeeze(0)  # → [hidden]
            y = fp8_gemv(w_fp8, scale, x, use_autotune=use_autotune)
            FP8_INTEGRATION_STATS["fp8_calls"] += 1
            if bias is not None:
                y = y + bias
            # Preserve batch dimensions: [1, 1, h] → output [N] → [1, 1, N]
            return y.reshape(*input.shape[:-1], -1)
        else:
            FP8_INTEGRATION_STATS["fallback_calls"] += 1
            w_deq = w_fp8.float() * scale.float().unsqueeze(-1)
            w_deq = w_deq.to(input.dtype)
            return F.linear(input, w_deq, bias)

    return types.MethodType(fp8_forward, layer)


def _quantize_linear_layer(layer: nn.Linear) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Quantize a single nn.Linear layer's weight to FP8.

    Returns (weight_fp8, scale, error_dict).
    """
    weight = layer.weight.data
    if weight.device.type == "meta":
        raise ValueError("Cannot quantize meta tensor, load model to GPU first")
    weight_c = weight.cuda() if not weight.is_cuda else weight
    w_fp8, scale = quantize_weight_fp8(weight_c, dim=0)
    w_fp8_cuda = w_fp8.cuda()
    scale_cuda = scale.squeeze(-1).cuda()
    scale_c = scale.clone()
    err = compute_quantization_error(weight_c.cpu(), w_fp8.cpu(), scale_c.cpu(), dim=0)
    err["params"] = weight_c.numel()
    err["shape"] = list(weight_c.shape)
    return w_fp8_cuda, scale_cuda, err


def _should_quantize_layer(name: str, module: nn.Module, min_params: int) -> bool:
    """Decide whether to quantize this linear layer."""
    if not isinstance(module, nn.Linear):
        return False
    if any(skip in name for skip in ("embed_tokens",)):
        return False
    if module.weight.numel() < min_params:
        return False
    return True


def apply_qwen35_fp8(
    model: Any,
    target_ffn: bool = True,
    target_lm_head: bool = True,
    target_attention: bool = False,
    min_params: int = 500_000,
    use_autotune: bool = True,
) -> dict:
    """Apply FP8 quantization to selected nn.Linear layers in a Qwen3.5 model.

    Replaces the forward method of each target layer with an FP8 GEMV kernel
    for batch-1 decode. Store original forward as _fp8_original_forward for
    later restoration.

    Args:
        model: Loaded Qwen3.5 HuggingFace model.
        target_ffn: Quantize MLP gate/up/down layers (largest impact).
        target_lm_head: Quantize lm_head (vocab projection).
        target_attention: Also quantize attention q/k/v/o projections.
        min_params: Minimum number of parameters for a layer to be quantized.
        use_autotune: Whether to autotune FP8 kernel (recommended).

    Returns:
        dict with keys: patched_count, patched_names, quant_errors, stats_reset
    """
    patched = []
    total_params_quantized = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module in _PATCHED_LAYERS:
            continue  # already patched

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module in _PATCHED_LAYERS:
            continue
        if module.weight.device.type == "meta":
            continue  # skip meta-tensor layers (offloaded, not fully loaded)

        suffix = name.rsplit(".", 1)[-1] if "." in name else name
        is_ffn = suffix in ("gate_proj", "up_proj", "down_proj")
        is_lm_head = suffix == "lm_head" or name.endswith("lm_head")
        is_attention = suffix in ("q_proj", "k_proj", "v_proj", "o_proj",
                                   "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")

        if is_ffn and not target_ffn:
            continue
        if is_lm_head and not target_lm_head:
            continue
        if is_attention and not target_attention:
            continue
        if module.weight.numel() < min_params:
            continue

        w_fp8, scale, err = _quantize_linear_layer(module)
        err["params"] = module.weight.numel()
        err["shape"] = list(module.weight.shape)
        _FP8_QUANT_ERRORS[name] = err

        module._fp8_weight = w_fp8
        module._fp8_scale = scale
        module._fp8_original_forward = module.forward
        module._fp8_use_autotune = use_autotune

        module.forward = _make_fp8_linear_forward(module, use_autotune=use_autotune)
        _PATCHED_LAYERS.append(module)
        patched.append(name)
        total_params_quantized += module.weight.numel()

    reset_fp8_integration_stats()

    return {
        "patched_count": len(patched),
        "patched_names": patched,
        "total_params_quantized": total_params_quantized,
        "quant_errors": dict(_FP8_QUANT_ERRORS),
    }


def restore_original_fp8_layers(model: Any) -> int:
    """Restore the original forward methods of all FP8-patched layers."""
    restored = 0
    for module in _PATCHED_LAYERS[:]:
        if hasattr(module, "_fp8_original_forward"):
            module.forward = module._fp8_original_forward
            restored += 1
        _PATCHED_LAYERS.remove(module)
    _FP8_QUANT_ERRORS.clear()
    return restored


def apply_qwen35_fp8_from_disk(
    model: Any,
    quant_dir: str,
    target_attention: bool = False,
    use_autotune: bool = True,
    free_fp16_weight: bool = True,
) -> dict:
    """Load pre-quantized FP8 weights from disk and patch model.

    Assumes weights were saved by quantize_qwen35_fp8_offline.py.
    Optionally frees the original FP16 weight to save GPU memory.

    Args:
        model: Loaded Qwen3.5 model on GPU.
        quant_dir: Directory containing .pt files with FP8 weights.
        target_attention: Also patch attention q/k/v/o projections.
        use_autotune: Whether to autotune FP8 kernel.
        free_fp16_weight: If True, replace weight.data with an empty tensor
            to free FP16 GPU memory.

    Returns:
        dict with patched_count, patched_names, etc.
    """
    import json
    from pathlib import Path

    quant_dir = Path(quant_dir)
    summary_path = quant_dir / "quant_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"quant_summary.json not found in {quant_dir}. "
                                "Run quantize_qwen35_fp8_offline.py first.")

    with open(summary_path) as f:
        summary = json.load(f)

    patched = []
    total_params = 0
    not_found = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module in _PATCHED_LAYERS:
            continue
        if module.weight.device.type == "meta":
            continue

        suffix = name.rsplit(".", 1)[-1] if "." in name else name
        is_ffn = suffix in ("gate_proj", "up_proj", "down_proj")
        is_lm_head = suffix == "lm_head" or name.endswith("lm_head")
        is_attention = suffix in ("q_proj", "k_proj", "v_proj", "o_proj",
                                   "in_proj_qkv", "in_proj_z", "in_proj_a",
                                   "in_proj_b", "out_proj")

        if is_ffn and not (summary.get("target") == "ffn" or summary.get("target") == "all"):
            continue
        if is_lm_head and not summary.get("target") == "all":
            continue
        if is_attention and not target_attention:
            continue

        safe_name = name.replace(".", "_")
        pt_path = quant_dir / f"{safe_name}.pt"
        if not pt_path.exists():
            not_found += 1
            continue

        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        w_fp8 = data["weight_fp8"]
        scale = data["scale"]

        device = module.weight.device
        module._fp8_weight = w_fp8.to(device)
        module._fp8_scale = scale.to(device)
        module._fp8_original_forward = module.forward
        module._fp8_use_autotune = use_autotune

        if free_fp16_weight:
            module.weight.data = torch.empty(0, device=device)

        module.forward = _make_fp8_linear_forward(module, use_autotune=use_autotune)
        _PATCHED_LAYERS.append(module)
        patched.append(name)
        total_params += module._fp8_weight.numel()

    reset_fp8_integration_stats()

    return {
        "patched_count": len(patched),
        "patched_names": patched,
        "total_params_quantized": total_params,
        "not_found": not_found,
        "free_fp16_memory": free_fp16_weight,
    }
