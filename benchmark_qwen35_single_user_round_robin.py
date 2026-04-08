from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from pprint import pprint

import torch

from benchmark_qwen35_single_user import _measure_once
from phase1_utils import MODEL_DIR, load_model_and_tokenizer
from triton_kernels.qwen35_integration import (
    configure_qwen35_deltanet_runtime,
    describe_qwen35_deltanet_runtime,
    get_qwen35_triton_patch_stats,
    reset_qwen35_triton_patch_stats,
)
from triton_kernels.qwen35_single_user_benchmark import (
    build_round_robin_mode_orders,
    compare_single_user_results,
    summarize_latency_trace,
    summarize_mode_position_counts,
)


ARTIFACT_DIR = Path("artifacts/qwen35_integration")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MODES = ("fla", "triton_base", "triton_fused")


def _is_triton_mode(mode: str) -> bool:
    return mode.startswith("triton")


def _merge_patch_stats(total: dict[str, int], update: dict[str, int]) -> dict[str, int]:
    merged = dict(total)
    for key, value in update.items():
        merged[key] = merged.get(key, 0) + int(value)
    return merged


def _mean_per_step_trace(runs: list[dict[str, object]]) -> list[float]:
    if not runs:
        return []
    num_steps = len(runs[0]["decode_step_latencies_ms"])
    if num_steps == 0:
        return []
    means = []
    for step_idx in range(num_steps):
        values = [run["decode_step_latencies_ms"][step_idx] for run in runs]
        means.append(round(sum(values) / len(values), 6))
    return means


def _summarize_mode(mode: str, runtime_description: dict[str, str], runs: list[dict[str, object]], patch_stats: dict[str, int]) -> dict[str, object]:
    if not runs:
        raise ValueError(f"Mode {mode} did not record any runs")

    generated_token_ids = runs[0]["generated_token_ids"]
    consistent_across_runs = all(run["generated_token_ids"] == generated_token_ids for run in runs)
    if not consistent_across_runs:
        raise RuntimeError(f"Mode {mode} produced inconsistent greedy generations across runs")

    return {
        "mode": mode,
        "runtime_description": runtime_description,
        "measured_runs": len(runs),
        "ttft_summary": summarize_latency_trace([run["ttft_ms"] for run in runs]),
        "decode_summary": summarize_latency_trace(
            [latency for run in runs for latency in run["decode_step_latencies_ms"]]
        ),
        "end_to_end_summary": summarize_latency_trace([run["end_to_end_ms"] for run in runs]),
        "per_step_mean_decode_ms": _mean_per_step_trace(runs),
        "generated_token_ids": generated_token_ids,
        "generated_text": runs[0]["generated_text"],
        "runs": runs,
        "patch_stats": patch_stats if _is_triton_mode(mode) else {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-robin single-user Qwen3.5 latency benchmark")
    parser.add_argument("--prompt", default="Explain what DeltaNet decode latency means in one short paragraph.")
    parser.add_argument("--gen-tokens", type=int, default=16)
    parser.add_argument("--warmup-cycles", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--no-chat-template", action="store_true")
    args = parser.parse_args()

    use_chat_template = not args.no_chat_template
    warmup_orders = build_round_robin_mode_orders(args.modes, args.warmup_cycles)
    measured_orders = build_round_robin_mode_orders(args.modes, args.runs)

    model, tokenizer, _ = load_model_and_tokenizer(
        MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    runtime_descriptions: dict[str, dict[str, str]] = {}
    runs_by_mode: dict[str, list[dict[str, object]]] = {mode: [] for mode in args.modes}
    patch_stats_by_mode: dict[str, dict[str, int]] = {mode: {} for mode in args.modes}

    print("warmup_mode_orders:", warmup_orders)
    for cycle_idx, order in enumerate(warmup_orders):
        print(f"warmup_cycle:{cycle_idx}:{order}")
        for mode in order:
            configure_qwen35_deltanet_runtime(model, mode)
            runtime_descriptions.setdefault(mode, describe_qwen35_deltanet_runtime(model))
            if _is_triton_mode(mode):
                reset_qwen35_triton_patch_stats()
            _measure_once(model, tokenizer, args.prompt, args.gen_tokens, use_chat_template)

    print("measured_mode_orders:", measured_orders)
    for cycle_idx, order in enumerate(measured_orders):
        print(f"measured_cycle:{cycle_idx}:{order}")
        for mode in order:
            configure_qwen35_deltanet_runtime(model, mode)
            runtime_descriptions.setdefault(mode, describe_qwen35_deltanet_runtime(model))
            if _is_triton_mode(mode):
                reset_qwen35_triton_patch_stats()
            run = _measure_once(model, tokenizer, args.prompt, args.gen_tokens, use_chat_template)
            runs_by_mode[mode].append(run)
            if _is_triton_mode(mode):
                patch_stats_by_mode[mode] = _merge_patch_stats(
                    patch_stats_by_mode[mode],
                    get_qwen35_triton_patch_stats(),
                )

    mode_results = {
        mode: _summarize_mode(
            mode,
            runtime_descriptions[mode],
            runs_by_mode[mode],
            patch_stats_by_mode[mode],
        )
        for mode in args.modes
    }

    reference_mode = "torch" if "torch" in mode_results else args.modes[0]
    pairwise_comparisons: dict[str, dict[str, object]] = {}
    for base_mode, base_result in mode_results.items():
        pairwise_comparisons[base_mode] = {}
        for candidate_mode, candidate_result in mode_results.items():
            if base_mode == candidate_mode:
                continue
            pairwise_comparisons[base_mode][candidate_mode] = compare_single_user_results(
                base_result,
                candidate_result,
            )

    payload = {
        "prompt": args.prompt,
        "use_chat_template": use_chat_template,
        "gen_tokens": args.gen_tokens,
        "warmup_cycles": args.warmup_cycles,
        "runs": args.runs,
        "reuse_model": True,
        "warmup_mode_orders": warmup_orders,
        "measured_mode_orders": measured_orders,
        "measured_position_counts": summarize_mode_position_counts(measured_orders),
        "reference_mode": reference_mode,
        "modes": mode_results,
        "pairwise_comparisons": pairwise_comparisons,
    }
    output_path = ARTIFACT_DIR / "qwen35_single_user_round_robin_compare.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for mode, result in mode_results.items():
        print(f"mode:{mode}")
        pprint(result["runtime_description"])
        print("ttft_summary:", result["ttft_summary"])
        print("decode_summary:", result["decode_summary"])
        print("end_to_end_summary:", result["end_to_end_summary"])
        print("per_step_mean_decode_ms:", result["per_step_mean_decode_ms"])
        if result["patch_stats"]:
            print("patch_stats:", result["patch_stats"])
        print("generated_text:", result["generated_text"])
        print()

    print("pairwise_comparisons:")
    pprint(pairwise_comparisons)
    print()
    print("saved_results:", output_path.resolve())

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
