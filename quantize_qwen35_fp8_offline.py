"""Offline CPU-based FP8 quantization for Qwen3.5-4B linear layers.

Loads the model on CPU, quantizes selected nn.Linear layers to FP8 E4M3,
and saves (weight_fp8, scale, bias) to disk.

Usage:
    python quantize_qwen35_fp8_offline.py                                    # default: FFN only
    python quantize_qwen35_fp8_offline.py --target all                       # all layers
    python quantize_qwen35_fp8_offline.py --model-dir /path/to/model         # custom model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT_DIR / "models" / "Qwen3.5-4B"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "fp8_quantized_weights"


def quantize_and_save(
    model_dir: str,
    output_dir: str,
    target: str = "ffn",
) -> dict:
    """Quantize model layers on CPU and save FP8 weights to disk.

    Args:
        model_dir: Path to HuggingFace model directory.
        output_dir: Directory to save .pt files.
        target: "ffn", "attention", or "all".

    Returns:
        Summary dict with per-layer error metrics.
    """
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_dir} on CPU ...")
    config = AutoConfig.from_pretrained(str(model_dir))
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    print(f"Loaded {sum(p.numel() for p in model.parameters()):,} params on CPU")

    from quantize_fp8_weights import quantize_weight_fp8, compute_quantization_error

    saved_files = []
    quant_errors = {}
    skipped_meta = 0
    skipped_small = 0

    target_ffn = target in ("ffn", "all")
    target_attention = target in ("attention", "all")
    target_lm_head = target in ("all")

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if module.weight.device.type == "meta":
            skipped_meta += 1
            continue

        suffix = name.rsplit(".", 1)[-1] if "." in name else name
        is_ffn = suffix in ("gate_proj", "up_proj", "down_proj")
        is_lm_head = suffix == "lm_head" or name.endswith("lm_head")
        is_attention = suffix in ("q_proj", "k_proj", "v_proj", "o_proj",
                                   "in_proj_qkv", "in_proj_z", "in_proj_a",
                                   "in_proj_b", "out_proj")

        if is_ffn and not target_ffn:
            continue
        if is_lm_head and not target_lm_head:
            continue
        if is_attention and not target_attention:
            continue

        n_params = module.weight.numel()
        if n_params < 500_000:
            skipped_small += 1
            continue

        weight = module.weight.data
        w_fp8, scale = quantize_weight_fp8(weight, dim=0)
        scale_1d = scale.squeeze(-1)
        bias = module.bias.data.clone() if module.bias is not None else None

        err = compute_quantization_error(weight, w_fp8, scale, dim=0)
        err["params"] = n_params
        err["shape"] = list(weight.shape)
        quant_errors[name] = err

        safe_name = name.replace(".", "_")
        out_path = output_dir / f"{safe_name}.pt"
        torch.save({
            "name": name,
            "weight_fp8": w_fp8,
            "scale": scale_1d,
            "bias": bias,
            "shape": list(weight.shape),
        }, out_path)
        saved_files.append(str(out_path))

        print(f"  {name:<60s} [{weight.shape[0]:>5d},{weight.shape[1]:>5d}] "
              f"cos={err['cos_sim']:.6f}")

    mean_cos = sum(e["cos_sim"] for e in quant_errors.values()) / len(quant_errors) if quant_errors else 0
    min_cos = min(e["cos_sim"] for e in quant_errors.values()) if quant_errors else 0

    summary = {
        "target": target,
        "saved_layers": len(saved_files),
        "skipped_meta": skipped_meta,
        "skipped_small": skipped_small,
        "mean_cos_sim": round(mean_cos, 6),
        "min_cos_sim": round(min_cos, 6),
        "output_dir": str(output_dir),
        "files": saved_files,
    }

    summary_path = output_dir / "quant_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved {len(saved_files)} layers to {output_dir}")
    print(f"  Skipped meta: {skipped_meta}, too small: {skipped_small}")
    print(f"  Mean cos_sim: {mean_cos:.6f}, min: {min_cos:.6f}")
    print(f"  Summary: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Offline FP8 quantization for Qwen3.5-4B")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Path to model directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save quantized weights")
    parser.add_argument("--target", type=str, default="ffn",
                        choices=["ffn", "attention", "all"],
                        help="Which layers to quantize")
    args = parser.parse_args()

    model_dir = args.model_dir or str(DEFAULT_MODEL_DIR)
    output_dir = args.output_dir or str(DEFAULT_OUTPUT_DIR)

    if not Path(model_dir).is_dir():
        print(f"Model directory not found: {model_dir}")
        sys.exit(1)

    quantize_and_save(model_dir, output_dir, args.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
