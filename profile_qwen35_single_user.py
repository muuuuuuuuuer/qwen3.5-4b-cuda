"""Profiling companion for Phase 2B and later reality checks.

Collects operator-level evidence showing decode is dominated by projection work
and launch overhead, not only the recurrent DeltaNet update.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from pprint import pprint

import torch

from benchmark_qwen35_single_user import _measure_once, reset_model_generation_state
from phase1_utils import MODEL_DIR, load_model_and_tokenizer
from triton_kernels.qwen35_integration import (
    configure_qwen35_deltanet_runtime,
    describe_qwen35_deltanet_runtime,
    get_qwen35_triton_patch_stats,
    reset_qwen35_triton_patch_stats,
)
from triton_kernels.qwen35_profiler import extract_top_ops


ARTIFACT_DIR = Path("artifacts/qwen35_integration/profiler")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _is_triton_mode(mode: str) -> bool:
    return mode.startswith("triton")


def _profile_one_mode(
    mode: str,
    prompt: str,
    gen_tokens: int,
    warmup_runs: int,
    use_chat_template: bool,
    row_limit: int,
) -> dict[str, object]:
    model, tokenizer, _ = load_model_and_tokenizer(
        MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    configure_qwen35_deltanet_runtime(model, mode)
    runtime_description = describe_qwen35_deltanet_runtime(model)

    print(f"profiling_mode:{mode}")
    for _ in range(warmup_runs):
        if _is_triton_mode(mode):
            reset_qwen35_triton_patch_stats()
        _measure_once(model, tokenizer, prompt, gen_tokens, use_chat_template)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    if _is_triton_mode(mode):
        reset_qwen35_triton_patch_stats()
    reset_model_generation_state(model)
    trace_path = ARTIFACT_DIR / f"{mode}_single_user_trace.json"
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        run_result = _measure_once(model, tokenizer, prompt, gen_tokens, use_chat_template)
    prof.export_chrome_trace(str(trace_path))

    events = prof.key_averages()
    result = {
        "mode": mode,
        "runtime_description": runtime_description,
        "trace_path": str(trace_path.resolve()),
        "top_self_cuda_ops": extract_top_ops(events, sort_key="self_cuda_time_total", row_limit=row_limit),
        "top_cuda_ops": extract_top_ops(events, sort_key="cuda_time_total", row_limit=row_limit),
        "top_self_cpu_ops": extract_top_ops(events, sort_key="self_cpu_time_total", row_limit=row_limit),
        "run_result": run_result,
        "patch_stats": get_qwen35_triton_patch_stats() if _is_triton_mode(mode) else {},
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile single-user Qwen3.5 request for FLA vs Triton")
    parser.add_argument("--prompt", default="Explain what DeltaNet decode latency means in one short paragraph.")
    parser.add_argument("--gen-tokens", type=int, default=16)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--modes", nargs="+", default=["fla", "triton_base", "triton_fused"])
    parser.add_argument("--row-limit", type=int, default=20)
    parser.add_argument("--no-chat-template", action="store_true")
    args = parser.parse_args()

    use_chat_template = not args.no_chat_template
    mode_results = {
        mode: _profile_one_mode(
            mode=mode,
            prompt=args.prompt,
            gen_tokens=args.gen_tokens,
            warmup_runs=args.warmup_runs,
            use_chat_template=use_chat_template,
            row_limit=args.row_limit,
        )
        for mode in args.modes
    }

    output_path = ARTIFACT_DIR / "single_user_profile_compare.json"
    output_path.write_text(json.dumps(mode_results, indent=2), encoding="utf-8")

    for mode, result in mode_results.items():
        print(f"mode:{mode}")
        pprint(result["runtime_description"])
        print("run_result:")
        pprint(
            {
                "ttft_ms": result["run_result"]["ttft_ms"],
                "decode_step_latencies_ms": result["run_result"]["decode_step_latencies_ms"],
                "end_to_end_ms": result["run_result"]["end_to_end_ms"],
                "generated_text": result["run_result"]["generated_text"],
            }
        )
        if result["patch_stats"]:
            print("patch_stats:", result["patch_stats"])
        print("top_self_cuda_ops:")
        pprint(result["top_self_cuda_ops"][:10])
        print("top_self_cpu_ops:")
        pprint(result["top_self_cpu_ops"][:10])
        print("trace_path:", result["trace_path"])
        print()

    print("saved_results:", output_path.resolve())


if __name__ == "__main__":
    main()
