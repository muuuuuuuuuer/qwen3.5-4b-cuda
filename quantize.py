from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def quantize_symmetric_int8(weight_fp16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight_float = weight_fp16.float()
    scale = weight_float.abs().amax(dim=1) / 127.0
    scale = scale.clamp(min=1e-8)
    qweight = (weight_float / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return qweight, scale.half()


def dequantize_int8(qweight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return qweight.float() * scale.float().unsqueeze(1)


def quantize_error_analysis(
    weight_fp16: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, float]:
    weight_approx = dequantize_int8(qweight, scale)
    weight_float = weight_fp16.float()
    abs_err = (weight_float - weight_approx).abs()
    rel_err = (abs_err / (weight_float.abs() + 1e-8)).mean().item()
    return {
        "max_abs_err": abs_err.max().item(),
        "mean_abs_err": abs_err.mean().item(),
        "mean_rel_err": rel_err,
    }


def quantize_model_layers(
    model: torch.nn.Module,
    target_layers: list[str],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    target_set = set(target_layers)
    for name, module in model.named_modules():
        if name in target_set and isinstance(module, torch.nn.Linear):
            weight = module.weight.data
            qweight, scale = quantize_symmetric_int8(weight)
            errors = quantize_error_analysis(weight, qweight, scale)
            results[name] = {
                "qweight": qweight.cpu(),
                "scale": scale.cpu(),
                "errors": errors,
                "shape": list(weight.shape),
            }
    return results


def load_layer_rows(csv_path: str | Path) -> list[dict[str, Any]]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def quantize_layer_records(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    name_to_module = dict(model.named_modules())
    results: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row["submodule_name"]
        module = name_to_module.get(name)
        if not isinstance(module, torch.nn.Linear):
            continue

        weight = module.weight.data
        qweight, scale = quantize_symmetric_int8(weight)
        errors = quantize_error_analysis(weight, qweight, scale)
        results[name] = {
            "qweight": qweight.cpu(),
            "scale": scale.cpu(),
            "errors": errors,
            "shape": list(weight.shape),
            "sub_type": row.get("sub_type"),
            "proj_name": row.get("proj_name"),
            "priority": row.get("priority"),
            "layer_index": row.get("layer_index"),
        }
    return results


def attach_layer_metadata(
    results: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    metadata_by_name = {row["submodule_name"]: row for row in rows}
    hydrated: dict[str, dict[str, Any]] = {}
    for name, info in results.items():
        row = metadata_by_name.get(name, {})
        hydrated[name] = {
            **info,
            "sub_type": info.get("sub_type") or row.get("sub_type"),
            "proj_name": info.get("proj_name") or row.get("proj_name"),
            "priority": info.get("priority") or row.get("priority"),
            "layer_index": info.get("layer_index") or row.get("layer_index"),
        }
    return hydrated


def quantize_all_target_layers(
    model: torch.nn.Module,
    layer_list_csv: str | Path = "layer_list.csv",
) -> dict[str, dict[str, Any]]:
    rows = load_layer_rows(layer_list_csv)
    return quantize_layer_records(model, rows)


def summarize_errors_by_type(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_errors: dict[str, list[dict[str, float]]] = defaultdict(list)
    for info in results.values():
        sub_type = info.get("sub_type") or "unknown"
        grouped_errors[sub_type].append(info["errors"])

    summary_rows: list[dict[str, Any]] = []
    for sub_type in sorted(grouped_errors):
        errors = grouped_errors[sub_type]
        count = len(errors)
        summary_rows.append(
            {
                "sub_type": sub_type,
                "count": count,
                "avg_max_abs_err": sum(item["max_abs_err"] for item in errors) / count,
                "avg_mean_abs_err": sum(item["mean_abs_err"] for item in errors) / count,
                "avg_mean_rel_err": sum(item["mean_rel_err"] for item in errors) / count,
            }
        )
    return summary_rows


def save_quantized(results: dict[str, dict[str, Any]], save_path: str | Path = "quantized_weights.pt") -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, save_path)
    return save_path


def load_quantized(save_path: str | Path = "quantized_weights.pt") -> dict[str, dict[str, Any]]:
    return torch.load(Path(save_path), map_location="cpu", weights_only=False)
