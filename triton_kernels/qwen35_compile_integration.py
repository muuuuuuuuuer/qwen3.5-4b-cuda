"""Lean torch.compile runtime hooks for the retained static-cache report modes."""

from __future__ import annotations

from typing import Any

import torch


def _iter_decoder_layers(model: Any) -> list[Any]:
    candidate_roots = []
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        candidate_roots.append(model.model.language_model)
    if hasattr(model, "language_model"):
        candidate_roots.append(model.language_model)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        candidate_roots.append(model.model)

    for root in candidate_roots:
        layers = getattr(root, "layers", None)
        if layers is not None:
            return list(layers)
    return []


def _patch_forward(module: Any, *, mode: str | None, fullgraph: bool, options: dict[str, object] | None) -> bool:
    if module is None:
        return False
    if not hasattr(module, "_qwen35_compile_original_forward"):
        module._qwen35_compile_original_forward = module.forward
    module.forward = torch.compile(
        module._qwen35_compile_original_forward,
        mode=mode,
        fullgraph=fullgraph,
        options=options,
    )
    return True


def _restore_forward(module: Any) -> bool:
    if module is None or not hasattr(module, "_qwen35_compile_original_forward"):
        return False
    module.forward = module._qwen35_compile_original_forward
    return True


def configure_qwen35_compile_runtime(
    model: Any,
    *,
    enabled: bool,
    mode: str | None = None,
    fullgraph: bool = False,
    compile_mlp: bool = False,
    compile_self_attn: bool = True,
    disable_linear_attn: bool = False,
    options: dict[str, object] | None = None,
) -> dict[str, Any]:
    layers = _iter_decoder_layers(model)
    compiled_mlp_layers = 0
    compiled_self_attn_layers = 0
    disabled_linear_attn_layers = 0

    for layer in layers:
        if not enabled:
            _restore_forward(getattr(layer, "mlp", None))
            _restore_forward(getattr(layer, "self_attn", None))
            continue

        if compile_mlp and _patch_forward(getattr(layer, "mlp", None), mode=mode, fullgraph=fullgraph, options=options):
            compiled_mlp_layers += 1
        if compile_self_attn and _patch_forward(
            getattr(layer, "self_attn", None),
            mode=mode,
            fullgraph=fullgraph,
            options=options,
        ):
            compiled_self_attn_layers += 1
        if disable_linear_attn and hasattr(layer, "linear_attn"):
            disabled_linear_attn_layers += 1

    state = {
        "enabled": bool(enabled),
        "mode": mode if enabled else None,
        "fullgraph": bool(fullgraph) if enabled else None,
        "compile_mlp": bool(compile_mlp) if enabled else False,
        "compile_self_attn": bool(compile_self_attn) if enabled else False,
        "disable_linear_attn": bool(disable_linear_attn) if enabled else False,
        "compiled_mlp_layers": int(compiled_mlp_layers),
        "compiled_self_attn_layers": int(compiled_self_attn_layers),
        "disabled_linear_attn_layers": int(disabled_linear_attn_layers),
    }
    model._qwen35_compile_runtime_state = state
    return dict(state)


def describe_qwen35_compile_runtime(model: Any) -> dict[str, Any]:
    return dict(
        getattr(
            model,
            "_qwen35_compile_runtime_state",
            {
                "enabled": False,
                "mode": None,
                "fullgraph": None,
                "compile_mlp": False,
                "compile_self_attn": False,
                "disable_linear_attn": False,
                "compiled_mlp_layers": 0,
                "compiled_self_attn_layers": 0,
                "disabled_linear_attn_layers": 0,
            },
        )
    )


__all__ = [
    "configure_qwen35_compile_runtime",
    "describe_qwen35_compile_runtime",
]
