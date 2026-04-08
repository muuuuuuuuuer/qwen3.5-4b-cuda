from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from pprint import pprint

import torch

from phase1_utils import MODEL_DIR, load_model_and_tokenizer
from triton_kernels.qwen35_integration import (
    configure_qwen35_deltanet_runtime,
    describe_qwen35_deltanet_runtime,
    get_qwen35_triton_patch_stats,
    reset_qwen35_triton_patch_stats,
)
from triton_kernels.qwen35_single_user_benchmark import (
    compare_single_user_results,
    summarize_latency_trace,
)


ARTIFACT_DIR = Path("artifacts/qwen35_integration")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MODES = ("torch", "fla", "triton_base", "triton_fused")


def reset_model_generation_state(model) -> None:
    if hasattr(model, "model") and hasattr(model.model, "rope_deltas"):
        model.model.rope_deltas = None
    if hasattr(model, "language_model") and hasattr(model.language_model, "rope_deltas"):
        model.language_model.rope_deltas = None


def build_prompt_text(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template or not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def to_model_inputs(tokenizer, prompt: str, device: torch.device, use_chat_template: bool) -> dict[str, torch.Tensor]:
    prompt_text = build_prompt_text(tokenizer, prompt, use_chat_template)
    model_inputs = tokenizer(prompt_text, return_tensors="pt")
    return {key: value.to(device) for key, value in model_inputs.items()}


def _sync_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _is_triton_mode(mode: str) -> bool:
    return mode.startswith("triton")


def _measure_once(model, tokenizer, prompt: str, gen_tokens: int, use_chat_template: bool) -> dict[str, object]:
    device = next(model.parameters()).device
    model_inputs = to_model_inputs(tokenizer, prompt, device, use_chat_template)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask", torch.ones_like(input_ids))

    reset_model_generation_state(model)

    _sync_if_needed()
    prefill_start = time.perf_counter()
    with torch.no_grad():
        prefill_outputs = model(**model_inputs, use_cache=True)
    first_token = prefill_outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
    _sync_if_needed()
    ttft_ms = (time.perf_counter() - prefill_start) * 1000.0

    generated_tokens = [first_token.detach().cpu()]
    decode_step_latencies_ms: list[float] = []
    past_key_values = prefill_outputs.past_key_values
    next_token = first_token

    for _ in range(1, gen_tokens):
        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device)],
            dim=1,
        )
        _sync_if_needed()
        decode_start = time.perf_counter()
        prepared = model.prepare_inputs_for_generation(
            input_ids=input_ids,
            next_sequence_length=1,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=True,
        )
        with torch.no_grad():
            decode_outputs = model(**prepared)
        next_token = decode_outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
        past_key_values = decode_outputs.past_key_values
        _sync_if_needed()
        decode_step_latencies_ms.append((time.perf_counter() - decode_start) * 1000.0)
        generated_tokens.append(next_token.detach().cpu())

    generated_token_ids = torch.cat(generated_tokens, dim=1)
    end_to_end_ms = ttft_ms + sum(decode_step_latencies_ms)
    return {
        "ttft_ms": round(ttft_ms, 6),
        "decode_step_latencies_ms": [round(value, 6) for value in decode_step_latencies_ms],
        "end_to_end_ms": round(end_to_end_ms, 6),
        "generated_token_ids": generated_token_ids.squeeze(0).tolist(),
        "generated_text": tokenizer.decode(generated_token_ids[0], skip_special_tokens=True),
    }


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


def run_mode(mode: str, prompt: str, gen_tokens: int, warmup_runs: int, measured_runs: int, use_chat_template: bool) -> dict[str, object]:
    model, tokenizer, _ = load_model_and_tokenizer(
        MODEL_DIR,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    configure_qwen35_deltanet_runtime(model, mode)
    runtime_description = describe_qwen35_deltanet_runtime(model)

    print(f"running_single_user_mode:{mode}")

    for _ in range(warmup_runs):
        if _is_triton_mode(mode):
            reset_qwen35_triton_patch_stats()
        _measure_once(model, tokenizer, prompt, gen_tokens, use_chat_template)

    if _is_triton_mode(mode):
        reset_qwen35_triton_patch_stats()

    runs = [_measure_once(model, tokenizer, prompt, gen_tokens, use_chat_template) for _ in range(measured_runs)]
    generated_token_ids = runs[0]["generated_token_ids"]
    consistent_across_runs = all(run["generated_token_ids"] == generated_token_ids for run in runs)
    if not consistent_across_runs:
        raise RuntimeError(f"Mode {mode} produced inconsistent greedy generations across runs")

    result = {
        "mode": mode,
        "runtime_description": runtime_description,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "ttft_summary": summarize_latency_trace([run["ttft_ms"] for run in runs]),
        "decode_summary": summarize_latency_trace(
            [latency for run in runs for latency in run["decode_step_latencies_ms"]]
        ),
        "end_to_end_summary": summarize_latency_trace([run["end_to_end_ms"] for run in runs]),
        "per_step_mean_decode_ms": _mean_per_step_trace(runs),
        "generated_token_ids": generated_token_ids,
        "generated_text": runs[0]["generated_text"],
        "runs": runs,
        "patch_stats": get_qwen35_triton_patch_stats() if _is_triton_mode(mode) else {},
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-user Qwen3.5 latency benchmark")
    parser.add_argument("--prompt", default="Explain what DeltaNet decode latency means in one short paragraph.")
    parser.add_argument("--gen-tokens", type=int, default=16)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--no-chat-template", action="store_true")
    args = parser.parse_args()

    use_chat_template = not args.no_chat_template
    mode_results = {
        mode: run_mode(mode, args.prompt, args.gen_tokens, args.warmup_runs, args.runs, use_chat_template)
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
        "warmup_runs": args.warmup_runs,
        "runs": args.runs,
        "reference_mode": reference_mode,
        "modes": mode_results,
        "pairwise_comparisons": pairwise_comparisons,
    }
    output_path = ARTIFACT_DIR / "qwen35_single_user_compare.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("reference_mode:", reference_mode)
    print()
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


if __name__ == "__main__":
    main()
