"""Shared utilities for model loading, layer enumeration, and baselines.

These helpers produce the layer inventory and FP16 measurements that guide the
DeltaNet decode-operator work.
"""

from __future__ import annotations

import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models" / "Qwen3.5-4B"
CONFIG_PATH = MODEL_DIR / "config.json"
LAYER_INDEX_RE = re.compile(r"model\.layers\.(\d+)\.")
FFN_SUFFIXES = {"gate_proj", "up_proj", "down_proj"}
ATTN_SUFFIXES = {"q_proj", "k_proj", "v_proj", "o_proj"}
LINEAR_ATTN_SUFFIXES = {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"}
PHASE1_BENCHMARK_TARGETS = (
    ("FFN_gate_proj", "FFN", "gate_proj"),
    ("FFN_down_proj", "FFN", "down_proj"),
    ("FullAttn_q_proj", "FullAttn", "q_proj"),
    ("FullAttn_k_proj", "FullAttn", "k_proj"),
    ("DeltaNet_in_proj_qkv", "DeltaNet_Attn", "in_proj_qkv"),
    ("DeltaNet_in_proj_z", "DeltaNet_Attn", "in_proj_z"),
)


def extract_text_config(config_path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return payload.get("text_config", payload)


def load_text_config(model_dir: str | Path = MODEL_DIR) -> dict[str, Any]:
    return extract_text_config(Path(model_dir) / "config.json")


def load_layer_types(model_dir: str | Path = MODEL_DIR) -> list[str]:
    text_config = load_text_config(model_dir)
    return list(text_config.get("layer_types", []))


def _select_model_loader(config: AutoConfig):
    architectures = getattr(config, "architectures", None) or []
    if any("ConditionalGeneration" in architecture for architecture in architectures):
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def load_model_and_tokenizer(
    model_dir: str | Path = MODEL_DIR,
    torch_dtype: torch.dtype = torch.float16,
    device_map: str | dict[str, str] = "auto",
):
    config = AutoConfig.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_loader = _select_model_loader(config)
    model = model_loader.from_pretrained(
        model_dir,
        dtype=torch_dtype,
        device_map=device_map,
    )
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    return model, tokenizer, config


def instantiate_model_for_structure(model_dir: str | Path = MODEL_DIR):
    config = AutoConfig.from_pretrained(model_dir)
    model_loader = _select_model_loader(config)
    with init_empty_weights():
        model = model_loader.from_config(config)
    return model, config


def get_model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def generate_text_smoke(
    model: torch.nn.Module,
    tokenizer,
    prompt: str = "Hello, how are you?",
    max_new_tokens: int = 10,
) -> str:
    device = get_model_device(model)
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = _prepare_model_inputs(tokenizer, chat_prompt, device)
    prompt_length = model_inputs["input_ids"].shape[-1]
    with torch.no_grad():
        output_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output_ids[0][prompt_length:], skip_special_tokens=True)


def extract_layer_index(name: str) -> int | None:
    match = LAYER_INDEX_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def _is_attention_projection(name: str, suffix: str) -> bool:
    if suffix in ATTN_SUFFIXES and any(token in name for token in (".self_attn.", ".attention.", ".attn.")):
        return True
    if suffix in LINEAR_ATTN_SUFFIXES and ".linear_attn." in name:
        return True
    return False


def classify_linear_subtype(layer_type: str, name: str) -> str | None:
    suffix = name.rsplit(".", 1)[-1]
    if ".mlp." in name and suffix in FFN_SUFFIXES:
        return "FFN"
    if not _is_attention_projection(name, suffix):
        return None
    if layer_type == "full_attention":
        return "FullAttn"
    if layer_type == "linear_attention":
        return "DeltaNet_Attn"
    return None


def classify_linear_priority(layer_type: str, name: str) -> str | None:
    sub_type = classify_linear_subtype(layer_type, name)
    if sub_type == "FFN":
        return "P0-优先"
    if sub_type == "FullAttn":
        return "P1-次优"
    if sub_type == "DeltaNet_Attn":
        return "P2-暂缓"
    return None


def build_layer_record(
    name: str,
    module: torch.nn.Module,
    layer_types: list[str],
) -> dict[str, Any] | None:
    if not isinstance(module, torch.nn.Linear):
        return None
    if name.startswith("visual.") or ".visual." in name:
        return None

    layer_index = extract_layer_index(name)
    if layer_index is None or layer_index >= len(layer_types):
        return None

    layer_type = layer_types[layer_index]
    sub_type = classify_linear_subtype(layer_type, name)
    priority = classify_linear_priority(layer_type, name)
    if priority is None or sub_type is None:
        return None

    return {
        "name": name,
        "layer_index": layer_index,
        "layer_type": layer_type,
        "sub_type": sub_type,
        "proj_name": name.rsplit(".", 1)[-1],
        "submodule_name": name,
        "out_features": module.out_features,
        "in_features": module.in_features,
        "shape": f"[{module.out_features}, {module.in_features}]",
        "params": module.out_features * module.in_features,
        "params_M": module.out_features * module.in_features / 1e6,
        "priority": priority,
    }


def collect_layer_rows(model: torch.nn.Module, layer_types: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        record = build_layer_record(name, module, layer_types)
        if record is not None:
            rows.append(record)
    rows.sort(key=lambda row: (row["layer_index"], row["submodule_name"]))
    return rows


def save_layer_rows_csv(rows: list[dict[str, Any]], csv_path: str | Path) -> Path:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "layer_index",
        "layer_type",
        "sub_type",
        "proj_name",
        "submodule_name",
        "out_features",
        "in_features",
        "shape",
        "params",
        "params_M",
        "priority",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def select_target_layers(
    rows: list[dict[str, Any]],
    priorities: tuple[str, ...] = ("P0-优先",),
) -> list[str]:
    return [row["submodule_name"] for row in rows if row["priority"] in priorities]


def select_phase1_benchmark_rows(
    rows: list[dict[str, Any]],
    targets: tuple[tuple[str, str, str], ...] = PHASE1_BENCHMARK_TARGETS,
) -> list[dict[str, Any]]:
    first_match_by_target: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("sub_type"), row.get("proj_name"))
        first_match_by_target.setdefault(key, row)

    selected: list[dict[str, Any]] = []
    for benchmark_name, sub_type, proj_name in targets:
        row = first_match_by_target.get((sub_type, proj_name))
        if row is None:
            continue
        selected.append({**row, "benchmark_name": benchmark_name})
    return selected


def build_fixed_prompt(tokenizer, target_tokens: int, seed_text: str) -> str:
    base_ids = tokenizer(seed_text, add_special_tokens=False)["input_ids"]
    if not base_ids:
        raise ValueError("seed_text must tokenize to at least one token")

    repeat = max(1, math.ceil(target_tokens / len(base_ids)))
    prompt = " ".join([seed_text] * repeat)
    while len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) < target_tokens:
        prompt = f"{prompt} {seed_text}"
    return prompt


def default_test_prompts(tokenizer) -> dict[str, str]:
    return {
        "short_128": build_fixed_prompt(
            tokenizer,
            128,
            "Explain quantum computing in simple terms with one concrete analogy.",
        ),
        "mid_512": build_fixed_prompt(
            tokenizer,
            512,
            "Write a detailed essay about climate change, mitigation, adaptation, and policy tradeoffs.",
        ),
        "long_2048": build_fixed_prompt(
            tokenizer,
            2048,
            "Summarize a long technical report about distributed systems, kernels, memory bandwidth, and GPU inference.",
        ),
    }


def _prepare_model_inputs(tokenizer, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
    model_inputs = tokenizer(prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in model_inputs.items()}


def measure_prefill(
    model: torch.nn.Module,
    model_inputs: dict[str, torch.Tensor],
    n_warmup: int = 3,
    n_runs: int = 10,
) -> float:
    for _ in range(n_warmup):
        with torch.no_grad():
            model.generate(**model_inputs, max_new_tokens=1)

    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model.generate(**model_inputs, max_new_tokens=1)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)


def measure_decode(
    model: torch.nn.Module,
    model_inputs: dict[str, torch.Tensor],
    gen_tokens: int = 128,
    n_warmup: int = 3,
    n_runs: int = 5,
) -> float:
    prefill_time = measure_prefill(model, model_inputs, n_warmup=n_warmup, n_runs=n_runs)

    for _ in range(n_warmup):
        with torch.no_grad():
            model.generate(**model_inputs, max_new_tokens=gen_tokens)

    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model.generate(**model_inputs, max_new_tokens=gen_tokens)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_total = sum(times) / len(times)
    decode_time = avg_total - prefill_time
    return decode_time / gen_tokens


def measure_peak_vram_mib(
    model: torch.nn.Module,
    model_inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
) -> float:
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def benchmark_single_layer_fp16(
    weight: torch.Tensor,
    x: torch.Tensor,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> float:
    if weight.is_cuda and x.is_cuda:
        for _ in range(n_warmup):
            _ = x @ weight.T

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(n_runs):
            _ = x @ weight.T
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / n_runs

    for _ in range(n_warmup):
        _ = x @ weight.T
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = x @ weight.T
    end = time.perf_counter()
    return (end - start) * 1000.0 / n_runs


def build_baseline_rows(
    model: torch.nn.Module,
    tokenizer,
    prompts: dict[str, str] | None = None,
    gen_tokens: int = 128,
) -> list[dict[str, Any]]:
    device = get_model_device(model)
    prompts = prompts or default_test_prompts(tokenizer)
    rows: list[dict[str, Any]] = []
    for name, prompt in prompts.items():
        model_inputs = _prepare_model_inputs(tokenizer, prompt, device)
        token_count = int(model_inputs["input_ids"].shape[-1])
        prefill_s = measure_prefill(model, model_inputs)
        decode_s_per_token = measure_decode(model, model_inputs, gen_tokens=gen_tokens)
        peak_vram_mib = measure_peak_vram_mib(model, model_inputs, max_new_tokens=gen_tokens)
        rows.append(
            {
                "scenario": name,
                "input_tokens": token_count,
                "prefill_latency_s": prefill_s,
                "decode_latency_s_per_token": decode_s_per_token,
                "peak_vram_mib": peak_vram_mib,
            }
        )
    return rows


def save_baseline_rows_csv(rows: list[dict[str, Any]], csv_path: str | Path) -> Path:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scenario",
        "input_tokens",
        "prefill_latency_s",
        "decode_latency_s_per_token",
        "peak_vram_mib",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def resolve_module(model: torch.nn.Module, dotted_name: str) -> torch.nn.Module:
    modules = dict(model.named_modules())
    return modules[dotted_name]
