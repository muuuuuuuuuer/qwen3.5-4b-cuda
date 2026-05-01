"""Small StaticCache helpers kept for static compiled DeltaNet report modes."""

from __future__ import annotations

from typing import Any

from transformers.cache_utils import StaticCache


def configure_qwen35_static_cache_runtime(model: Any, *, enabled: bool) -> dict[str, Any]:
    state = {"enabled": bool(enabled), "patched_mask_creator": False}
    model._qwen35_static_cache_runtime_state = state
    return dict(state)


def describe_qwen35_static_cache_runtime(model: Any) -> dict[str, Any]:
    return dict(
        getattr(
            model,
            "_qwen35_static_cache_runtime_state",
            {"enabled": False, "patched_mask_creator": False},
        )
    )


def build_qwen35_static_cache(model: Any, *, max_cache_len: int) -> StaticCache:
    return StaticCache(config=model.config, max_cache_len=max_cache_len)


__all__ = [
    "build_qwen35_static_cache",
    "configure_qwen35_static_cache_runtime",
    "describe_qwen35_static_cache_runtime",
]
