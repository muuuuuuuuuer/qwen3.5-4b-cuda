from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import torch


def _tensor_row(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def _flatten_tensors(value: Any, prefix: str) -> list[dict[str, Any]]:
    if isinstance(value, torch.Tensor):
        return [_tensor_row(prefix, value)]

    if isinstance(value, Mapping):
        rows: list[dict[str, Any]] = []
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_tensors(item, child_prefix))
        return rows

    if isinstance(value, (tuple, list)):
        rows = []
        for index, item in enumerate(value):
            child_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            rows.extend(_flatten_tensors(item, child_prefix))
        return rows

    return []


def _flatten_object_tensors(value: Any, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, item in getattr(value, "__dict__", {}).items():
        if key.startswith("_"):
            continue
        child_prefix = f"{prefix}.{key}" if prefix else key
        rows.extend(_flatten_tensors(item, child_prefix))
    return rows


def find_first_deltanet_layer(model: torch.nn.Module) -> dict[str, Any]:
    candidates = (
        ("model.language_model.layers[0].linear_attn", lambda root: root.model.language_model.layers[0].linear_attn),
        ("language_model.layers[0].linear_attn", lambda root: root.language_model.layers[0].linear_attn),
    )

    for attr_path, resolver in candidates:
        try:
            module = resolver(model)
        except (AttributeError, IndexError, TypeError):
            continue

        module_name = next((name for name, submodule in model.named_modules() if submodule is module), None)
        return {
            "attr_path": attr_path,
            "module_name": module_name,
            "module": module,
        }

    raise AttributeError("Unable to resolve a DeltaNet linear_attn module from the loaded model")


def list_module_tensors(module: torch.nn.Module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, param in module.named_parameters():
        rows.append(
            {
                "kind": "parameter",
                "name": name,
                "shape": list(param.shape),
                "dtype": str(param.dtype),
            }
        )
    for name, buffer in module.named_buffers():
        rows.append(
            {
                "kind": "buffer",
                "name": name,
                "shape": list(buffer.shape),
                "dtype": str(buffer.dtype),
            }
        )
    return rows


def capture_forward_io(
    module: torch.nn.Module,
    runner: Callable[[], Any],
) -> list[dict[str, list[dict[str, Any]]]]:
    calls: list[dict[str, list[dict[str, Any]]]] = []

    def hook_with_kwargs(
        _module: torch.nn.Module,
        inputs: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        calls.append(
            {
                "inputs": _flatten_tensors(inputs, "inputs") + _flatten_tensors(kwargs, "kwargs"),
                "outputs": _flatten_tensors(output, "output"),
            }
        )

    def hook_legacy(_module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        calls.append(
            {
                "inputs": _flatten_tensors(inputs, "inputs"),
                "outputs": _flatten_tensors(output, "output"),
            }
        )

    try:
        handle = module.register_forward_hook(hook_with_kwargs, with_kwargs=True)
    except TypeError:
        handle = module.register_forward_hook(hook_legacy)
    try:
        runner()
    finally:
        handle.remove()
    return calls


def summarize_past_key_values(cache: Any) -> list[dict[str, Any]]:
    if cache is None:
        return []

    if hasattr(cache, "layers"):
        layer_entries = list(cache.layers)
    else:
        try:
            layer_entries = list(cache)
        except TypeError:
            layer_entries = [cache]

    rows: list[dict[str, Any]] = []
    for index, layer_cache in enumerate(layer_entries):
        layer_rows = _flatten_tensors(layer_cache, f"layer {index}")
        if not layer_rows:
            layer_rows = _flatten_object_tensors(layer_cache, f"layer {index}")
        rows.extend(layer_rows)
    return rows
