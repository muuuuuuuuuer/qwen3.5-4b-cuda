"""Offline FP8 (E4M3) weight quantization for Qwen3.5-4B linear layers.

Converts FP16 weights to FP8 E4M3 with per-channel scaling. Supports both
per-row and per-column quantization (quantize along the K or N dimension).

Usage:
    python quantize_fp8_weights.py                          # smoke test
    python quantize_fp8_weights.py --model-dir models/Qwen3.5-4B --output-dir models/qwen35-fp8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import torch


FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0


def quantize_weight_fp8(
    weight_fp16: torch.Tensor, dim: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP16 weight tensor to FP8 E4M3 with per-channel scaling.

    Args:
        weight_fp16: FP16 weight tensor of shape [N, K].
        dim: Dimension along which to compute per-channel scales (0 = per-row,
            1 = per-column).

    Returns:
        (weight_fp8, scale) where weight_fp8 is torch.float8_e4m3fn of same
        shape and scale is FP32 of shape [N, 1] (dim=0) or [1, K] (dim=1).
    """
    if weight_fp16.dtype != torch.float16:
        weight_fp16 = weight_fp16.to(torch.float16)

    reduce_dim = tuple(i for i in range(weight_fp16.ndim) if i != dim)
    max_abs = weight_fp16.abs().amax(dim=reduce_dim, keepdim=True).clamp(min=1e-12)

    scale = max_abs / FP8_MAX

    scaled = weight_fp16.float() / scale.float()
    scaled_clamped = scaled.clamp(-FP8_MAX, FP8_MAX)
    weight_fp8 = scaled_clamped.to(torch.float8_e4m3fn)

    return weight_fp8, scale.squeeze(-1) if dim == 1 else scale


def reconstruct_fp16(weight_fp8: torch.Tensor, scale: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Reconstruct FP16 weights from FP8 + scale for error measurement."""
    s = scale.float()
    if s.ndim == 2 and s.shape[-1] == 1:
        s = s.squeeze(-1)
    if dim == 0:
        return weight_fp8.float() * s.unsqueeze(-1)
    else:
        return weight_fp8.float() * s.unsqueeze(0)


def compute_quantization_error(
    weight_orig: torch.Tensor,
    weight_fp8: torch.Tensor,
    scale: torch.Tensor,
    dim: int = 0,
) -> dict:
    """Compute error metrics between original and FP8-quantized weights."""
    weight_recon = reconstruct_fp16(weight_fp8, scale, dim)

    diff = (weight_orig.float() - weight_recon).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()

    orig_norm = weight_orig.float().norm()
    if orig_norm > 0:
        rel_err = (weight_orig.float() - weight_recon).norm() / orig_norm
    else:
        rel_err = torch.tensor(0.0)

    flat_orig = weight_orig.float().reshape(-1)
    flat_recon = weight_recon.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(
        flat_orig.unsqueeze(0), flat_recon.unsqueeze(0)
    ).item()

    return {
        "max_abs_err": float(max_err),
        "mean_abs_err": float(mean_err),
        "rel_err": float(rel_err),
        "cos_sim": float(cos_sim),
    }


def quantize_linear_layer(
    linear: torch.nn.Linear, dim: int = 0
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Quantize the weight of a single nn.Linear layer.

    Returns (weight_fp8, scale, bias) where bias may be None.
    """
    w_fp8, scale = quantize_weight_fp8(linear.weight.data, dim=dim)
    bias = linear.bias.data.clone() if linear.bias is not None else None
    return w_fp8, scale, bias


def quantize_model_linear_layers(
    model,
    output_dir: Path,
    dim: int = 0,
    skip_patterns: tuple[str, ...] = ("lm_head", "embed"),
    min_params: int = 0,
    dry_run: bool = False,
) -> dict:
    """Quantize all nn.Linear layers in a HuggingFace model and save artifacts.

    Args:
        model: The loaded HuggingFace model.
        output_dir: Directory to save quantized weight files.
        dim: Quantization dimension (0 = per-row for GEMV [N,K] × [K]).
        skip_patterns: Module name substrings to skip.
        min_params: Skip layers with fewer than this many parameters.
        dry_run: If True, only print what would be done, don't save.

    Returns:
        Summary dict with per-layer error metrics.
    """
    output_dir = Path(output_dir)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"layers": {}, "global_stats": {}}
    skipped = 0
    quantized = 0

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(p in name for p in skip_patterns):
            skipped += 1
            continue
        n_params = module.weight.numel()
        if n_params < min_params:
            skipped += 1
            continue

        w_fp8, scale, bias = quantize_linear_layer(module, dim=dim)
        err = compute_quantization_error(module.weight.data, w_fp8, scale, dim)

        safe_name = name.replace(".", "_")
        shape_str = f"{list(module.weight.shape)}"
        err["shape"] = shape_str
        err["params"] = n_params
        summary["layers"][name] = err

        if not dry_run:
            torch.save(
                {"weight_fp8": w_fp8, "scale": scale, "bias": bias, "name": name},
                output_dir / f"{safe_name}.pt",
            )

        quantized += 1

    all_cos = [v["cos_sim"] for v in summary["layers"].values()]
    summary["global_stats"] = {
        "quantized_layers": quantized,
        "skipped_layers": skipped,
        "total_layers": quantized + skipped,
        "mean_cos_sim": float(sum(all_cos) / len(all_cos)) if all_cos else 0.0,
        "min_cos_sim": float(min(all_cos)) if all_cos else 0.0,
        "max_cos_sim": float(max(all_cos)) if all_cos else 0.0,
        "quantization_format": "FP8-E4M3",
        "fp8_max_value": FP8_MAX,
        "quantize_dim": dim,
    }

    return summary


def smoke_test():
    """Quick smoke test with a random weight matrix."""
    torch.manual_seed(42)
    N, K = 1024, 2560  # FullAttn_k shape
    w = torch.randn(N, K, dtype=torch.float16)

    w_fp8, scale = quantize_weight_fp8(w, dim=0)
    err = compute_quantization_error(w, w_fp8, scale, dim=0)

    print(f"Smoke test: FP8 quantize [{N}, {K}]")
    print(f"  weight byte size: {w.numel() * 2} -> {w_fp8.numel() * 1}")
    print(f"  scale shape: {scale.shape}")
    print(f"  max_abs_err: {err['max_abs_err']:.6f}")
    print(f"  mean_abs_err: {err['mean_abs_err']:.6f}")
    print(f"  rel_err: {err['rel_err']:.6f}")
    print(f"  cos_sim: {err['cos_sim']:.8f}")
    print("  PASS" if err["cos_sim"] > 0.99 else "  WARN: low cos_sim")


def main():
    parser = argparse.ArgumentParser(description="Quantize FP16 model weights to FP8 E4M3")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Path to HuggingFace model directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save quantized weights")
    parser.add_argument("--dim", type=int, default=0,
                        help="Quantization dimension (0=per-row for GEMV)")
    parser.add_argument("--min-params", type=int, default=0,
                        help="Skip layers with fewer than this many parameters")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only compute errors, don't save")
    args = parser.parse_args()

    if args.model_dir is None:
        smoke_test()
    else:
        # Dynamic import to avoid loading HF unless needed
        from transformers import AutoModel

        model_dir = Path(args.model_dir)
        output_dir = Path(args.output_dir) if args.output_dir else model_dir / "fp8_quantized"

        print(f"Loading model from {model_dir} ...")
        model = AutoModel.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )

        summary = quantize_model_linear_layers(
            model,
            output_dir=output_dir,
            dim=args.dim,
            min_params=args.min_params,
            dry_run=args.dry_run,
        )

        print(f"\nQuantized {summary['global_stats']['quantized_layers']} layers")
        print(f"Skipped {summary['global_stats']['skipped_layers']} layers")
        print(f"Mean cos_sim: {summary['global_stats']['mean_cos_sim']:.6f}")
        print(f"Min  cos_sim: {summary['global_stats']['min_cos_sim']:.6f}")

        summary_path = output_dir / "quantization_summary.json"
        if not args.dry_run:
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
