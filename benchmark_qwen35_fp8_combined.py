"""Combined benchmark: FP8 FFN + Teammate Packed DeltaNet.

Compares 4 modes on the same model:
  - torch:      raw PyTorch (baseline)
  - packed:     teammate's packed DeltaNet kernel only
  - fp8:        FP8 FFN GEMV kernel only
  - combined:   both optimizations applied

Usage:
    python benchmark_qwen35_fp8_combined.py --gen-tokens 16 --runs 2
    python benchmark_qwen35_fp8_combined.py --smoke  # 8 tokens, 1 run
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
from triton_kernels.qwen35_integration import (
    apply_qwen35_deltanet_triton_patch,
    _restore_original_forward,
    _iter_qwen35_linear_attn_modules,
    get_qwen35_triton_patch_stats,
    reset_qwen35_triton_patch_stats,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_QUANT_DIR = ROOT / "artifacts" / "fp8_quantized_weights"


def run_single_generate(model, tokenizer, prompt: str, max_new_tokens: int) -> dict:
    """Run one generation pass. Returns timing dict."""
    device = get_model_device(model)
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)
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
                outputs = model(input_ids=current_ids, use_cache=True)
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
            else:
                outputs = model(input_ids=current_ids, past_key_values=past_key_values,
                                use_cache=True)
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

    return {
        "ttft_ms": round(ttft_ms, 3),
        "decode_mean_ms": round(sum(decode_times_ms) / len(decode_times_ms), 3) if decode_times_ms else 0,
        "decode_median_ms": round(sorted(decode_times_ms)[len(decode_times_ms)//2], 3) if decode_times_ms else 0,
        "end_to_end_ms": round(total_time, 3),
        "generated_tokens": len(decode_times_ms) + 1,
    }


def apply_packed(model) -> int:
    """Apply teammate's packed DeltaNet kernel (bypasses FLA check)."""
    return apply_qwen35_deltanet_triton_patch(model, "full_forward_lowrank_beta_gate_packed")


def restore_packed(model) -> int:
    """Restore original linear_attn forward."""
    modules = _iter_qwen35_linear_attn_modules(model)
    restored = 0
    for m in modules:
        if hasattr(m, "_original_forward"):
            m.forward = m._original_forward
            restored += 1
    return restored


def apply_fp8(model, quant_dir: str, use_autotune: bool = True) -> dict:
    """Apply FP8 FFN quantization from pre-computed weights."""
    return apply_qwen35_fp8_from_disk(
        model, quant_dir, target_attention=False,
        use_autotune=use_autotune, free_fp16_weight=True,
    )


def run_bench(
    model, tokenizer, prompt, gen_tokens: int,
    warmup_runs: int, bench_runs: int,
    label: str,
) -> dict:
    """Run benchmark for a single configuration (model is already configured)."""
    print(f"\n{'='*60}")
    print(f"  {label} (gen={gen_tokens}, warmup={warmup_runs}, runs={bench_runs})")
    print(f"{'='*60}")

    decode_list = []
    e2e_list = []
    ttft_list = []

    for i in range(warmup_runs + bench_runs):
        r = run_single_generate(model, tokenizer, prompt, gen_tokens)
        tag = "warmup" if i < warmup_runs else f"run {i - warmup_runs + 1}"
        print(f"  [{tag}] ttft={r['ttft_ms']:.0f}ms, decode={r['decode_mean_ms']:.1f}ms/tok, "
              f"e2e={r['end_to_end_ms']:.0f}ms, tokens={r['generated_tokens']}")
        if i >= warmup_runs:
            decode_list.append(r["decode_mean_ms"])
            e2e_list.append(r["end_to_end_ms"])
            ttft_list.append(r["ttft_ms"])

    avg_decode = round(sum(decode_list) / len(decode_list), 3) if decode_list else 0
    avg_e2e = round(sum(e2e_list) / len(e2e_list), 3) if e2e_list else 0
    avg_ttft = round(sum(ttft_list) / len(ttft_list), 3) if ttft_list else 0

    return {
        "label": label,
        "decode_mean_ms": avg_decode,
        "decode_list": decode_list,
        "e2e_mean_ms": avg_e2e,
        "e2e_list": e2e_list,
        "ttft_mean_ms": avg_ttft,
    }


def _load_model(model_dir, device_map="auto"):
    model, tokenizer, config = load_model_and_tokenizer(model_dir, torch.float16, device_map)
    print(f"  Model on {get_model_device(model)}, {sum(p.numel() for p in model.parameters()):,} params")
    return model, tokenizer


def run_combined_benchmark(
    model_dir: str,
    gen_tokens: int = 16,
    warmup_runs: int = 1,
    bench_runs: int = 2,
    quant_dir: str | None = None,
) -> dict:
    """Run all 4 modes. Each mode may reload the model for memory hygiene."""

    qdir = Path(quant_dir or DEFAULT_QUANT_DIR)
    if not (qdir / "quant_summary.json").exists():
        print("Running offline quantization first ...")
        from quantize_qwen35_fp8_offline import quantize_and_save
        quantize_and_save(model_dir, str(qdir), target="ffn")

    # ---- Pre-warm FP8 autotuner ----
    print("Pre-warming FP8 autotuner ...")
    for pt_file in sorted(qdir.glob("model_language_model_layers_0_mlp_*.pt")):
        data = torch.load(pt_file, map_location="cpu", weights_only=False)
        w_fp8, scale = data["weight_fp8"].cuda(), data["scale"].cuda()
        x = torch.randn(w_fp8.shape[1], dtype=torch.float16, device="cuda")
        fp8_gemv(w_fp8, scale, x, use_autotune=True)
    print("FP8 autotune pre-warmed.")

    results: list[dict] = []

    # ---- torch baseline & fp8: use device_map="auto" (saves VRAM) ----
    print(f"\n{'#'*60}")
    print(f" torch + FP8 (device_map=auto) on {model_dir}")
    print(f"{'#'*60}")
    model, tokenizer = _load_model(model_dir, "auto")
    device = get_model_device(model)
    prompt = build_fixed_prompt(tokenizer, 128, "Explain the meaning of life briefly.")

    r = run_bench(model, tokenizer, prompt, gen_tokens, warmup_runs, bench_runs, "torch (baseline)")
    results.append(r)

    apply_fp8(model, str(qdir), use_autotune=True)
    reset_fp8_integration_stats()
    r = run_bench(model, tokenizer, prompt, gen_tokens, warmup_runs, bench_runs, "fp8 (FP8 FFN)")
    results.append(r)
    del model
    torch.cuda.empty_cache()

    # ---- packed & combined: use device_map="cuda:0" (packed needs all on GPU) ----
    print(f"\n{'#'*60}")
    print(f" packed + combined (device_map=cuda:0) on {model_dir}")
    print(f"{'#'*60}")
    model, tokenizer = _load_model(model_dir, "cuda:0")
    device = get_model_device(model)
    prompt = build_fixed_prompt(tokenizer, 128, "Explain the meaning of life briefly.")

    # Baseline for this model load
    r = run_bench(model, tokenizer, prompt, gen_tokens, warmup_runs, bench_runs,
                  "torch_cuda0")
    results.append(r)

    # Apply packed → run
    apply_packed(model)
    reset_qwen35_triton_patch_stats()
    r = run_bench(model, tokenizer, prompt, gen_tokens, warmup_runs, bench_runs,
                  "packed (teammate DeltaNet)")
    results.append(r)

    # Apply FP8 on top → run combined
    apply_fp8(model, str(qdir), use_autotune=True)
    reset_fp8_integration_stats()
    r = run_bench(model, tokenizer, prompt, gen_tokens, warmup_runs, bench_runs,
                  "combined (packed+FP8)")
    results.append(r)
    del model
    torch.cuda.empty_cache()

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"Summary (gen={gen_tokens})")
    print(f"{'='*70}")
    print(f"  {'Mode':<25s} {'Decode ms/tok':>14s} {'vs torch':>10s} {'E2E ms':>10s} {'vs torch':>10s}")
    print(f"  {'-'*25} {'-'*14} {'-'*10} {'-'*10} {'-'*10}")

    # Group 1: torch (auto) vs fp8 (auto)
    r0, r1 = results[0], results[1]
    ds1 = r0["decode_mean_ms"] / r1["decode_mean_ms"] if r1["decode_mean_ms"] > 0 else 0
    es1 = r0["e2e_mean_ms"] / r1["e2e_mean_ms"] if r1["e2e_mean_ms"] > 0 else 0
    print(f"  torch (auto)               {r0['decode_mean_ms']:>14.3f} {r0['e2e_mean_ms']:>10.0f}     1.000x")
    print(f"  fp8 (auto)                 {r1['decode_mean_ms']:>14.3f} {r1['e2e_mean_ms']:>10.0f}    {ds1:>7.3f}x")

    # Group 2: torch_cuda0 → packed → combined
    r2, r3, r4 = results[2], results[3], results[4]
    ds3 = r2["decode_mean_ms"] / r3["decode_mean_ms"] if r3["decode_mean_ms"] > 0 else 0
    es3 = r2["e2e_mean_ms"] / r3["e2e_mean_ms"] if r3["e2e_mean_ms"] > 0 else 0
    ds4 = r2["decode_mean_ms"] / r4["decode_mean_ms"] if r4["decode_mean_ms"] > 0 else 0
    es4 = r2["e2e_mean_ms"] / r4["e2e_mean_ms"] if r4["e2e_mean_ms"] > 0 else 0
    print(f"  torch (cuda:0)             {r2['decode_mean_ms']:>14.3f} {r2['e2e_mean_ms']:>10.0f}     1.000x")
    print(f"  packed (cuda:0)            {r3['decode_mean_ms']:>14.3f} {r3['e2e_mean_ms']:>10.0f}    {ds3:>7.3f}x")
    print(f"  combined (cuda:0)          {r4['decode_mean_ms']:>14.3f} {r4['e2e_mean_ms']:>10.0f}    {ds4:>7.3f}x")
    print(f"")
    print(f"  Overall: packed {ds3:.2f}x + FP8 {ds1:.2f}x = combined {ds4:.2f}x")

    output = {
        "gen_tokens": gen_tokens,
        "warmup_runs": warmup_runs,
        "bench_runs": bench_runs,
        "results": results,
    }
    summary_path = ROOT / "artifacts" / "qwen35_integration" / "qwen35_fp8_combined.json"
    with open(summary_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {summary_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Combined FP8 + Packed DeltaNet benchmark")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--gen-tokens", type=int, default=16)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="8 tokens, 1 run")
    args = parser.parse_args()

    model_dir = args.model_dir or str(ROOT / "models" / "Qwen3.5-4B")
    if args.smoke:
        args.gen_tokens = 8
        args.warmup_runs = 0
        args.runs = 1

    run_combined_benchmark(
        model_dir=model_dir,
        gen_tokens=args.gen_tokens,
        warmup_runs=args.warmup_runs,
        bench_runs=args.runs,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
