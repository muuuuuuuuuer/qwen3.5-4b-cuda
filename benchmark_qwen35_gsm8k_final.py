"""Final GSM8K quality/latency benchmark for Qwen3.5 edge decode paths."""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from benchmark_qwen35_single_user import reset_model_generation_state
from phase1_utils import MODEL_DIR, load_model_and_tokenizer
from triton_kernels.qwen35_compile_integration import (
    configure_qwen35_compile_runtime,
    describe_qwen35_compile_runtime,
)
from triton_kernels.qwen35_integration import (
    configure_qwen35_deltanet_runtime,
    describe_qwen35_deltanet_runtime,
    get_qwen35_triton_patch_stats,
    reset_qwen35_triton_patch_stats,
)
from triton_kernels.qwen35_static_cache_integration import (
    build_qwen35_static_cache,
    configure_qwen35_static_cache_runtime,
    describe_qwen35_static_cache_runtime,
)


ARTIFACT_DIR = Path("artifacts/qwen35_integration")
QUESTION_IDS_PATH = ARTIFACT_DIR / "gsm8k_50_question_ids.json"
EAGER_PACKED_MODE = "fp16_eager_packed"
STATIC_COMPILED_MODE = "fp16_static_compiled_attn_only_deltanet"
STATIC_COMPILED_PACKED_MODE = "fp16_static_compiled_attn_only_deltanet_packed"
SEED = 42
NUM_QUESTIONS = 50
SMOKE_QUESTIONS = 3
WARMUP_QUESTIONS = 2
MAX_NEW_TOKENS = 256

# Actual mapping note:
# - torch/fla are native runtime modes from benchmark_qwen35_single_user.py.
# - fp16_eager is eager Triton fused DeltaNet.
# - fp16_eager_packed is the packed low-rank beta gate DeltaNet endpoint.
EVAL_MODES = [
    "torch",
    "fla",
    "fp16_eager",
    EAGER_PACKED_MODE,
    STATIC_COMPILED_MODE,
    STATIC_COMPILED_PACKED_MODE,
]

GSM8K_FEWSHOT = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Answer: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 = 8 dollars left. The answer is 8.

Question: {test_question}
Answer:"""


def build_prompt(test_question: str) -> str:
    return GSM8K_FEWSHOT.format(test_question=test_question)


def extract_answer(generated_text: str) -> str | None:
    """Extract the final numeric answer from a GSM8K CoT generation."""
    m = re.search(r"[Tt]he answer is\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)", generated_text)
    if m:
        return m.group(1).replace(",", "")

    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", generated_text)
    if nums:
        return nums[-1].replace(",", "")

    return None


def normalize_gt(answer_field: str) -> str:
    """GSM8K ground truth stores the final answer after a #### marker."""
    m = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", answer_field)
    if m:
        return m.group(1).replace(",", "")
    return answer_field.strip()


def is_correct(predicted: str | None, ground_truth: str) -> bool:
    if predicted is None:
        return False
    try:
        return float(predicted) == float(ground_truth)
    except ValueError:
        return predicted.strip() == ground_truth.strip()


def load_gsm8k_dataset(local_path: str | None = None):
    from datasets import DatasetDict, load_dataset, load_from_disk

    if local_path:
        path = Path(local_path)
        try:
            dataset = load_from_disk(str(path))
        except Exception:
            return load_dataset(str(path), "main", split="test")
        if isinstance(dataset, DatasetDict):
            return dataset["test"]
        if isinstance(dataset, dict) and "test" in dataset:
            return dataset["test"]
        return dataset

    return load_dataset("gsm8k", "main", split="test")


def prepare_question_rows(
    dataset: Sequence[dict[str, Any]],
    *,
    num_questions: int,
    seed: int,
    cache_path: Path,
) -> list[dict[str, Any]]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        question_ids = [int(value) for value in json.loads(cache_path.read_text())]
        if len(question_ids) < num_questions:
            raise ValueError(
                f"Cached GSM8K question id file {cache_path} has {len(question_ids)} ids, "
                f"but {num_questions} are required"
            )
    else:
        if len(dataset) < num_questions:
            raise ValueError(f"GSM8K test split has only {len(dataset)} rows; need {num_questions}")
        question_ids = random.Random(seed).sample(range(len(dataset)), num_questions)
        cache_path.write_text(json.dumps(question_ids, indent=2) + "\n")

    rows: list[dict[str, Any]] = []
    for question_id in question_ids[:num_questions]:
        row = dict(dataset[question_id])
        row["question_id"] = int(question_id)
        rows.append(row)
    return rows


def resolve_mode_spec(mode: str) -> dict[str, object]:
    if mode == "torch":
        return _base_spec("torch")
    if mode == "fla":
        return _base_spec("fla")
    if mode == "fp16_eager":
        spec = _base_spec("triton_fused")
        spec["mode_note"] = "fp16_eager maps to eager triton_fused for Phase 2"
        return spec
    if mode == EAGER_PACKED_MODE:
        spec = _base_spec("triton_lowrank_beta_gate_packed")
        spec["mode_note"] = "fp16_eager_packed maps to eager packed low-rank beta DeltaNet"
        return spec
    if mode == STATIC_COMPILED_MODE:
        spec = _base_spec("triton_fused")
        spec.update({"use_static_cache": True, "compile_decode": True, "compile_after_prefill": True})
        spec["mode_note"] = "static-cache torch.compile report mode with fused DeltaNet"
        return spec
    if mode == STATIC_COMPILED_PACKED_MODE:
        spec = _base_spec("triton_lowrank_beta_gate_packed")
        spec.update({"use_static_cache": True, "compile_decode": True, "compile_after_prefill": True})
        spec["mode_note"] = "static-cache torch.compile report mode with packed DeltaNet"
        return spec
    raise ValueError(f"Unsupported GSM8K eval mode: {mode}")


def _base_spec(deltanet_mode: str) -> dict[str, object]:
    return {
        "deltanet_mode": deltanet_mode,
        "use_static_cache": False,
        "compile_decode": False,
        "compile_after_prefill": False,
        "compile_mode": "reduce-overhead",
    }


def configure_mode(model, mode: str) -> tuple[dict[str, object], dict[str, object]]:
    spec = resolve_mode_spec(mode)
    configure_qwen35_compile_runtime(model, enabled=False)
    configure_qwen35_static_cache_runtime(model, enabled=False)
    configure_qwen35_deltanet_runtime(model, str(spec["deltanet_mode"]))
    if bool(spec["use_static_cache"]):
        configure_qwen35_static_cache_runtime(model, enabled=True)
    runtime_description = {
        "mode": mode,
        "spec": spec,
        "deltanet_runtime": describe_qwen35_deltanet_runtime(model),
        "static_cache_runtime": describe_qwen35_static_cache_runtime(model),
        "compile_runtime": describe_qwen35_compile_runtime(model),
    }
    return runtime_description, spec


def _sync_device(device: torch.device) -> None:
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)


def _is_eos(token: torch.Tensor, eos_token_ids: set[int]) -> bool:
    if not eos_token_ids:
        return False
    return int(token.item()) in eos_token_ids


def _eos_token_ids(tokenizer) -> set[int]:
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, (list, tuple, set)):
        return {int(value) for value in eos_token_id}
    return {int(eos_token_id)}


def generate_one(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    mode_spec: dict[str, object],
) -> dict[str, object]:
    device = next(model.parameters()).device
    model_inputs = tokenizer(prompt, return_tensors="pt")
    model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask", torch.ones_like(input_ids))
    initial_context_tokens = int(input_ids.shape[1])
    compile_enabled = bool(mode_spec["compile_decode"])
    use_static_cache = bool(mode_spec["use_static_cache"])
    eos_token_ids = _eos_token_ids(tokenizer)

    reset_model_generation_state(model)
    if compile_enabled and bool(mode_spec["compile_after_prefill"]):
        configure_qwen35_compile_runtime(model, enabled=False)

    prefill_cache = None
    if use_static_cache:
        prefill_cache = build_qwen35_static_cache(model, max_cache_len=initial_context_tokens + max_new_tokens)

    generated_tokens: list[torch.Tensor] = []
    past_key_values = None
    next_token = None
    use_cuda_events = torch.cuda.is_available() and device.type == "cuda"

    try:
        if compile_enabled:
            torch.compiler.cudagraph_mark_step_begin()
        _sync_device(device)

        if use_cuda_events:
            total_start = torch.cuda.Event(enable_timing=True)
            prefill_start = torch.cuda.Event(enable_timing=True)
            prefill_end = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            total_start.record()
            prefill_start.record()
        else:
            total_start_time = time.perf_counter()
            prefill_start_time = time.perf_counter()

        with torch.no_grad():
            if prefill_cache is None:
                prefill_outputs = model(**model_inputs, use_cache=True)
            else:
                prefill_outputs = model(**model_inputs, use_cache=True, past_key_values=prefill_cache)
        next_token = prefill_outputs.logits[:, -1].argmax(dim=-1, keepdim=True)

        if use_cuda_events:
            prefill_end.record()
        else:
            prefill_ms = (time.perf_counter() - prefill_start_time) * 1000.0

        generated_tokens.append(next_token.detach().cpu())
        past_key_values = prefill_outputs.past_key_values
        if compile_enabled and bool(mode_spec["compile_after_prefill"]):
            configure_qwen35_compile_runtime(
                model,
                enabled=True,
                mode=str(mode_spec["compile_mode"]),
                fullgraph=False,
                compile_mlp=False,
                compile_self_attn=True,
                disable_linear_attn=False,
            )

        if not _is_eos(next_token, eos_token_ids):
            for _ in range(1, max_new_tokens):
                input_ids = torch.cat([input_ids, next_token], dim=1)
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device),
                    ],
                    dim=1,
                )
                prepared = model.prepare_inputs_for_generation(
                    input_ids=input_ids,
                    next_sequence_length=1,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
                if compile_enabled:
                    torch.compiler.cudagraph_mark_step_begin()
                with torch.no_grad():
                    decode_outputs = model(**prepared)
                next_token = decode_outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
                past_key_values = decode_outputs.past_key_values
                generated_tokens.append(next_token.detach().cpu())
                if _is_eos(next_token, eos_token_ids):
                    break

        if use_cuda_events:
            total_end.record()
            _sync_device(device)
            prefill_ms = float(prefill_start.elapsed_time(prefill_end))
            total_ms = float(total_start.elapsed_time(total_end))
        else:
            total_ms = (time.perf_counter() - total_start_time) * 1000.0
    finally:
        if compile_enabled:
            configure_qwen35_compile_runtime(model, enabled=False)

    generated_token_ids = torch.cat(generated_tokens, dim=1).squeeze(0).tolist()
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    generated_count = len(generated_token_ids)
    truncated = generated_count >= max_new_tokens and (
        not eos_token_ids or int(generated_token_ids[-1]) not in eos_token_ids
    )
    decode_mean_ms = (total_ms - prefill_ms) / max(generated_count - 1, 1)

    return {
        "context_tokens": initial_context_tokens,
        "generated_token_ids": generated_token_ids,
        "generated_text": generated_text,
        "generated_tokens": generated_count,
        "prefill_ms": round(prefill_ms, 6),
        "total_ms": round(total_ms, 6),
        "decode_mean_ms_per_token": round(decode_mean_ms, 6),
        "truncated": bool(truncated),
    }


def summarize_results(rows: list[dict[str, object]]) -> dict[str, object]:
    prefill_values = [float(row["prefill_ms"]) for row in rows]
    decode_values = [float(row["decode_mean_ms_per_token"]) for row in rows]
    total_values = [float(row["total_ms"]) for row in rows]
    token_counts = [int(row["generated_tokens"]) for row in rows]
    correct_count = sum(1 for row in rows if bool(row["correct"]))

    return {
        "accuracy": round(correct_count / len(rows), 6) if rows else 0.0,
        "invalid_count": sum(1 for row in rows if row.get("predicted") is None),
        "prefill_mean_ms": _mean(prefill_values),
        "prefill_median_ms": _median(prefill_values),
        "decode_mean_ms_per_token": _mean(decode_values),
        "decode_median_ms_per_token": _median(decode_values),
        "total_mean_ms": _mean(total_values),
        "total_median_ms": _median(total_values),
        "generated_tokens_mean": _mean([float(value) for value in token_counts]),
        "generated_tokens_total": sum(token_counts),
    }


def _mean(values: list[float]) -> float:
    return round(float(statistics.fmean(values)), 6) if values else 0.0


def _median(values: list[float]) -> float:
    return round(float(statistics.median(values)), 6) if values else 0.0


def run_mode(
    mode: str,
    rows: list[dict[str, Any]],
    *,
    model_path: str,
    max_new_tokens: int,
    smoke: bool,
) -> dict[str, object]:
    print(f"\n=== running GSM8K mode: {mode} ===", flush=True)
    model = None
    tokenizer = None
    try:
        model, tokenizer, _ = load_model_and_tokenizer(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model.eval()
        runtime_description, mode_spec = configure_mode(model, mode)

        warmup_rows = rows[: min(WARMUP_QUESTIONS, len(rows))]
        for idx, row in enumerate(warmup_rows, start=1):
            print(f"[{mode}] warmup {idx}/{len(warmup_rows)} qid={row['question_id']}", flush=True)
            generate_one(
                model,
                tokenizer,
                build_prompt(str(row["question"])),
                max_new_tokens=max_new_tokens,
                mode_spec=mode_spec,
            )

        if _uses_triton_deltanet(str(mode_spec["deltanet_mode"])):
            reset_qwen35_triton_patch_stats()

        per_question_results: list[dict[str, object]] = []
        for idx, row in enumerate(rows, start=1):
            question = str(row["question"])
            ground_truth = normalize_gt(str(row["answer"]))
            run = generate_one(
                model,
                tokenizer,
                build_prompt(question),
                max_new_tokens=max_new_tokens,
                mode_spec=mode_spec,
            )
            predicted = extract_answer(str(run["generated_text"]))
            correct = is_correct(predicted, ground_truth)
            result_row = {
                "question_id": int(row["question_id"]),
                "question": question,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "correct": correct,
                "raw_generation": run["generated_text"],
                "generated_tokens": run["generated_tokens"],
                "prefill_ms": run["prefill_ms"],
                "total_ms": run["total_ms"],
                "decode_mean_ms_per_token": run["decode_mean_ms_per_token"],
                "truncated": run["truncated"],
            }
            per_question_results.append(result_row)
            print(
                f"[{mode}] {idx}/{len(rows)} qid={row['question_id']} "
                f"correct={int(correct)} tokens={run['generated_tokens']} "
                f"decode={run['decode_mean_ms_per_token']:.3f}ms/tok "
                f"total={run['total_ms']:.1f}ms",
                flush=True,
            )

        summary = summarize_results(per_question_results)
        payload = {
            "mode": mode,
            "dataset": "gsm8k",
            "num_prompts": len(rows),
            "seed": SEED,
            "smoke": smoke,
            "generation_config": {
                "do_sample": False,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.eos_token_id,
            },
            "runtime_description": runtime_description,
            "per_question_results": per_question_results,
            "summary": summary,
            "patch_stats": get_qwen35_triton_patch_stats()
            if _uses_triton_deltanet(str(mode_spec["deltanet_mode"]))
            else {},
        }
        return payload
    finally:
        if model is not None:
            configure_qwen35_compile_runtime(model, enabled=False)
            configure_qwen35_static_cache_runtime(model, enabled=False)
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def output_path_for_mode(output_dir: Path, mode: str, smoke: bool) -> Path:
    suffix = "_smoke" if smoke else ""
    return output_dir / f"qwen35_gsm8k_final_{mode}{suffix}.json"


def summary_path(output_dir: Path, smoke: bool) -> Path:
    suffix = "_smoke" if smoke else ""
    return output_dir / f"qwen35_gsm8k_final_summary{suffix}.md"


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def collect_summary_payloads(
    output_dir: Path,
    *,
    smoke: bool,
    updated_payloads: dict[str, dict[str, object]],
    summary_modes: Sequence[str] = EVAL_MODES,
) -> dict[str, dict[str, object]]:
    mode_payloads: dict[str, dict[str, object]] = {}
    for mode in summary_modes:
        if mode in updated_payloads:
            mode_payloads[mode] = updated_payloads[mode]
            continue
        path = output_path_for_mode(output_dir, mode, smoke)
        if path.exists():
            mode_payloads[mode] = load_json(path)
    for mode, payload in updated_payloads.items():
        mode_payloads.setdefault(mode, payload)
    return mode_payloads


def write_summary_markdown(
    path: Path,
    mode_payloads: dict[str, dict[str, object]],
    *,
    num_questions: int,
    max_new_tokens: int,
) -> None:
    lines = [
        f"# GSM8K {num_questions}-Question Final Benchmark",
        "",
        f"Seed: {SEED}",
        f"Generation: greedy, max_new_tokens={max_new_tokens}, 8-shot CoT prompt",
        "",
        "| Mode | Accuracy | Decode mean (ms/tok) | Decode median (ms/tok) | Total mean (ms) | Total median (ms) | Tokens/sec |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    sorted_payloads = sorted(
        mode_payloads.items(),
        key=lambda item: 1000.0 / float(item[1]["summary"]["decode_mean_ms_per_token"])
        if float(item[1]["summary"]["decode_mean_ms_per_token"]) > 0
        else 0.0,
        reverse=True,
    )
    for mode, payload in sorted_payloads:
        summary = payload["summary"]
        decode_mean = float(summary["decode_mean_ms_per_token"])
        tokens_per_sec = 1000.0 / decode_mean if decode_mean > 0 else 0.0
        lines.append(
            f"| {mode} | {float(summary['accuracy']):.3f} | "
            f"{decode_mean:.3f} | {float(summary['decode_median_ms_per_token']):.3f} | "
            f"{float(summary['total_mean_ms']):.3f} | {float(summary['total_median_ms']):.3f} | "
            f"{tokens_per_sec:.2f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final Qwen3.5 GSM8K latency/accuracy benchmark")
    parser.add_argument("--model-path", default=str(MODEL_DIR))
    parser.add_argument("--gsm8k-local-path", default=None)
    parser.add_argument("--output-dir", default=str(ARTIFACT_DIR))
    parser.add_argument("--modes", nargs="+", default=list(EVAL_MODES))
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_gsm8k_dataset(args.gsm8k_local_path)
    all_rows = prepare_question_rows(
        dataset,
        num_questions=NUM_QUESTIONS,
        seed=SEED,
        cache_path=output_dir / QUESTION_IDS_PATH.name,
    )
    rows = all_rows[:SMOKE_QUESTIONS] if args.smoke else all_rows

    updated_payloads: dict[str, dict[str, object]] = {}
    for mode in args.modes:
        payload = run_mode(
            mode,
            rows,
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            smoke=args.smoke,
        )
        output_path = output_path_for_mode(output_dir, mode, args.smoke)
        save_json(output_path, payload)
        updated_payloads[mode] = payload
        print(f"[{mode}] wrote {output_path}", flush=True)

    markdown_path = summary_path(output_dir, args.smoke)
    mode_payloads = collect_summary_payloads(output_dir, smoke=args.smoke, updated_payloads=updated_payloads)
    write_summary_markdown(
        markdown_path,
        mode_payloads,
        num_questions=len(rows),
        max_new_tokens=args.max_new_tokens,
    )
    print(f"wrote {markdown_path}", flush=True)


if __name__ == "__main__":
    main()
