"""End-to-end FP8 GEMV benchmark for Qwen3.5-4B single-user decode.

Loads the real Qwen3.5-4B model, replaces selected nn.Linear layers with
FP8 GEMV kernels, and measures decode latency on synthetic prompts.

Compares FP16 baseline against FP8 patched model. Reports TTFT
(time-to-first-token), per-token decode mean/median, and end-to-end time.

Usage:
    python benchmark_qwen35_fp8_e2e.py --gen-tokens 16 --runs 3
    python benchmark_qwen35_fp8_e2e.py --gen-tokens 128 --runs 3 --target attention
    python benchmark_qwen35_fp8_e2e.py --smoke   # quick 8-token sanity check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from phase1_utils import load_model_and_tokenizer, get_model_device, build_fixed_prompt
from triton_kernels.fp8_gemv import fp8_gemv
from triton_kernels.qwen35_fp8_integration import (
    apply_qwen35_fp8_from_disk,
    restore_original_fp8_layers,
    get_fp8_integration_stats,
    reset_fp8_integration_stats,
)

DEFAULT_QUANT_DIR = Path(__file__).resolve().parent / "artifacts" / "fp8_quantized_weights"


def run_single_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> dict:
    """Run one generation pass. Returns timing dict and output text."""
    device = get_model_device(model)
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[-1]
    input_ids = inputs["input_ids"]

    ttft_ms = 0.0
    decode_times_ms: list[float] = []
    first_token_time = None

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        past_key_values = None
        current_ids = input_ids

        for step in range(max_new_tokens):
            if step == 0:
                # Prefill
                outputs = model(input_ids=current_ids, use_cache=True)
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
            else:
                # Decode
                outputs = model(
                    input_ids=current_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]

            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            current_ids = next_token

            if step == 0:
                torch.cuda.synchronize()
                ttft_ms = (time.perf_counter() - t0) * 1000
                first_token_time = time.perf_counter()
            else:
                torch.cuda.synchronize()
                step_time = time.perf_counter() - first_token_time
                decode_times_ms.append(step_time * 1000)
                first_token_time = time.perf_counter()

            if next_token.item() == tokenizer.eos_token_id:
                break

    torch.cuda.synchronize()
    total_time = (time.perf_counter() - t0) * 1000

    output_text = tokenizer.decode(current_ids[0], skip_special_tokens=True)

    return {
        "ttft_ms": round(ttft_ms, 3),
        "decode_mean_ms": round(sum(decode_times_ms) / len(decode_times_ms), 3) if decode_times_ms else 0,
        "decode_median_ms": round(sorted(decode_times_ms)[len(decode_times_ms)//2], 3) if decode_times_ms else 0,
        "end_to_end_ms": round(total_time, 3),
        "generated_tokens": len(decode_times_ms) + 1,
        "output_text": output_text,
        "token_diff_from_baseline": None,
    }


def compare_fp16_vs_fp8(
    model_dir: str,
    gen_tokens: int = 16,
    warmup_runs: int = 1,
    bench_runs: int = 3,
    target_attention: bool = False,
    quant_dir: str | None = None,
) -> dict:
    """Compare FP16 baseline vs FP8 over multiple runs.

    1. Load model on GPU for FP16 baseline.
    2. Run FP16 benchmark.
    3. Load pre-quantized FP8 weights from disk, patch layers.
    4. Run FP8 benchmark.
    5. Compare.

    Returns comparison dict suitable for JSON output.
    """
    qdir = Path(quant_dir or DEFAULT_QUANT_DIR)

    # Ensure pre-quantized weights exist
    if not (qdir / "quant_summary.json").exists():
        print(f"Pre-quantized weights not found at {qdir}")
        print("Running offline quantization first ...")
        from quantize_qwen35_fp8_offline import quantize_and_save
        quantize_and_save(model_dir, str(qdir), target="ffn")

    print(f"Loading model from {model_dir} ...")
    model, tokenizer, config = load_model_and_tokenizer(model_dir, torch.float16, "auto")
    device = get_model_device(model)
    print(f"Model loaded on {device}, {sum(p.numel() for p in model.parameters()):,} params")

    prompt = build_fixed_prompt(tokenizer, 128, "Explain the meaning of life briefly.")

    results = {"fp16_baseline": [], "fp8_patched": [], "comparison": {}}

    # FP16 baseline
    print(f"\n{'='*60}")
    print(f"FP16 baseline (gen={gen_tokens}, warmup={warmup_runs}, runs={bench_runs})")
    print(f"{'='*60}")

    for i in range(warmup_runs + bench_runs):
        r = run_single_generate(model, tokenizer, prompt, gen_tokens)
        label = "warmup" if i < warmup_runs else f"run {i - warmup_runs + 1}"
        print(f"  [{label}] ttft={r['ttft_ms']:.1f}ms, decode={r['decode_mean_ms']:.1f}ms/tok, "
              f"e2e={r['end_to_end_ms']:.1f}ms, tokens={r['generated_tokens']}")
        if i >= warmup_runs:
            results["fp16_baseline"].append(r)

    fp16_decode = [r["decode_mean_ms"] for r in results["fp16_baseline"]]
    fp16_e2e = [r["end_to_end_ms"] for r in results["fp16_baseline"]]
    fp16_baseline_output = results["fp16_baseline"][0]["output_text"]

    # Apply pre-quantized FP8 weights
    print(f"\n{'='*60}")
    print(f"FP8 patched (gen={gen_tokens}, warmup={warmup_runs}, runs={bench_runs})")
    print(f"{'='*60}")

    patch_stats = apply_qwen35_fp8_from_disk(
        model, str(qdir),
        target_attention=target_attention,
        use_autotune=True,
        free_fp16_weight=True,
    )
    print(f"Patched {patch_stats['patched_count']} layers from {qdir} "
          f"(FP16 weight freed: {patch_stats['free_fp16_memory']})")
    if patch_stats.get("not_found", 0) > 0:
        print(f"  Warning: {patch_stats['not_found']} layers had no .pt file")

    # Pre-warm autotuner
    print("Pre-warming FP8 autotuner (one call per shape) ...")
    seen_shapes = set()
    for name in patch_stats["patched_names"]:
        for m_name, module in model.named_modules():
            if name == m_name and hasattr(module, "_fp8_weight"):
                shape = tuple(module._fp8_weight.shape)
                if shape not in seen_shapes:
                    seen_shapes.add(shape)
                    x_dummy = torch.randn(shape[1], dtype=torch.float16, device=device)
                    _ = fp8_gemv(module._fp8_weight, module._fp8_scale, x_dummy, use_autotune=True)
    print(f"  {len(seen_shapes)} unique shapes pre-warmed")

    for i in range(warmup_runs + bench_runs):
        r = run_single_generate(model, tokenizer, prompt, gen_tokens)
        label = "warmup" if i < warmup_runs else f"run {i - warmup_runs + 1}"
        print(f"  [{label}] ttft={r['ttft_ms']:.1f}ms, decode={r['decode_mean_ms']:.1f}ms/tok, "
              f"e2e={r['end_to_end_ms']:.1f}ms, tokens={r['generated_tokens']}")
        if i >= warmup_runs:
            r["token_diff_from_baseline"] = r["output_text"] != fp16_baseline_output
            results["fp8_patched"].append(r)

    stats = get_fp8_integration_stats()
    print(f"\nFP8 kernel stats: calls={stats['calls']}, fp8={stats['fp8_calls']}, "
          f"fallback={stats['fallback_calls']}")

    fp8_decode = [r["decode_mean_ms"] for r in results["fp8_patched"]]
    fp8_e2e = [r["end_to_end_ms"] for r in results["fp8_patched"]]

    fp16_avg_decode = sum(fp16_decode) / len(fp16_decode)
    fp16_avg_e2e = sum(fp16_e2e) / len(fp16_e2e)
    fp8_avg_decode = sum(fp8_decode) / len(fp8_decode)
    fp8_avg_e2e = sum(fp8_e2e) / len(fp8_e2e)

    decode_speedup = fp16_avg_decode / fp8_avg_decode if fp8_avg_decode > 0 else 0
    e2e_speedup = fp16_avg_e2e / fp8_avg_e2e if fp8_avg_e2e > 0 else 0

    results["comparison"] = {
        "gen_tokens": gen_tokens,
        "warmup_runs": warmup_runs,
        "bench_runs": bench_runs,
        "fp16_decode_mean_ms": round(fp16_avg_decode, 3),
        "fp8_decode_mean_ms": round(fp8_avg_decode, 3),
        "decode_speedup": round(decode_speedup, 4),
        "fp16_e2e_mean_ms": round(fp16_avg_e2e, 3),
        "fp8_e2e_mean_ms": round(fp8_avg_e2e, 3),
        "e2e_speedup": round(e2e_speedup, 4),
        "same_generation": not any(r.get("token_diff_from_baseline", False) for r in results["fp8_patched"]),
        "kernel_calls": stats["calls"],
        "fp8_kernel_calls": stats["fp8_calls"],
        "fallback_calls": stats["fallback_calls"],
        "patched_layers": patch_stats["patched_count"],
    }

    print(f"\n{'='*60}")
    print(f"Comparison: FP8 vs FP16 (gen={gen_tokens})")
    print(f"{'='*60}")
    print(f"  FP16 decode: {fp16_avg_decode:.1f} ms/tok")
    print(f"  FP8  decode: {fp8_avg_decode:.1f} ms/tok")
    print(f"  Decode speedup: {decode_speedup:.3f}x")
    print(f"  FP16 e2e: {fp16_avg_e2e:.1f} ms")
    print(f"  FP8  e2e: {fp8_avg_e2e:.1f} ms")
    print(f"  E2E speedup: {e2e_speedup:.3f}x")
    print(f"  Same generation: {results['comparison']['same_generation']}")

    restore_original_fp8_layers(model)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end FP8 GEMV benchmark for Qwen3.5-4B"
    )
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Path to model directory (default: models/Qwen3.5-4B)")
    parser.add_argument("--gen-tokens", type=int, default=16,
                        help="Number of tokens to generate")
    parser.add_argument("--warmup-runs", type=int, default=1,
                        help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of benchmark runs")
    parser.add_argument("--target", type=str, default="ffn",
                        choices=["ffn", "attention", "all"],
                        help="Which layers to quantize (default: ffn)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test with 8 tokens, 1 run")
    args = parser.parse_args()

    model_dir = args.model_dir or str(Path(__file__).resolve().parent / "models" / "Qwen3.5-4B")

    if args.smoke:
        args.gen_tokens = 8
        args.warmup_runs = 0
        args.runs = 1

    results = compare_fp16_vs_fp8(
        model_dir=model_dir,
        gen_tokens=args.gen_tokens,
        warmup_runs=args.warmup_runs,
        bench_runs=args.runs,
        target_attention=(args.target in ("attention", "all")),
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
