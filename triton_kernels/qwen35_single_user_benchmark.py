"""Shared summary helpers for single-user serving benchmarks.

These utilities define how the project reports mean latency, percentile
latency, round-robin ordering, and strict same-generation checks.
"""

from __future__ import annotations

import math
import statistics


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    rank = max(1, math.ceil(percentile * len(sorted_values)))
    return float(sorted_values[rank - 1])


def summarize_latency_trace(latencies_ms: list[float]) -> dict[str, float | int | None]:
    if not latencies_ms:
        return {
            "count": 0,
            "min_ms": None,
            "max_ms": None,
            "mean_ms": None,
            "p50_ms": None,
            "p90_ms": None,
        }

    sorted_values = sorted(float(value) for value in latencies_ms)
    return {
        "count": len(sorted_values),
        "min_ms": sorted_values[0],
        "max_ms": sorted_values[-1],
        "mean_ms": round(sum(sorted_values) / len(sorted_values), 6),
        "p50_ms": round(float(statistics.median(sorted_values)), 6),
        "p90_ms": round(_percentile(sorted_values, 0.9), 6),
    }


def build_round_robin_mode_orders(modes: list[str], cycles: int) -> list[list[str]]:
    if cycles < 0:
        raise ValueError("cycles must be non-negative")
    if not modes:
        return []

    orders: list[list[str]] = []
    num_modes = len(modes)
    for cycle_idx in range(cycles):
        offset = cycle_idx % num_modes
        orders.append(list(modes[offset:]) + list(modes[:offset]))
    return orders


def summarize_mode_position_counts(orders: list[list[str]]) -> dict[str, dict[int, int]]:
    counts: dict[str, dict[int, int]] = {}
    for order in orders:
        for position, mode in enumerate(order):
            per_mode = counts.setdefault(mode, {})
            per_mode[position] = per_mode.get(position, 0) + 1
    return counts


def _safe_speedup(reference_ms: float | None, candidate_ms: float | None) -> float | None:
    if reference_ms in (None, 0) or candidate_ms in (None, 0):
        return None
    return round(reference_ms / candidate_ms, 6)


def _token_diff_stats(reference_tokens: list[int], candidate_tokens: list[int]) -> tuple[int, float]:
    common = min(len(reference_tokens), len(candidate_tokens))
    mismatches = sum(
        1
        for ref_token, candidate_token in zip(reference_tokens[:common], candidate_tokens[:common])
        if ref_token != candidate_token
    )
    diff_count = mismatches + abs(len(reference_tokens) - len(candidate_tokens))
    denominator = max(len(reference_tokens), len(candidate_tokens))
    if denominator == 0:
        return diff_count, 0.0
    return diff_count, round(diff_count / denominator, 6)


def compare_single_user_results(reference: dict, candidate: dict) -> dict[str, float | bool | None]:
    token_diff_count, token_diff_rate = _token_diff_stats(
        reference["generated_token_ids"],
        candidate["generated_token_ids"],
    )
    return {
        "ttft_speedup": _safe_speedup(reference["ttft_summary"]["mean_ms"], candidate["ttft_summary"]["mean_ms"]),
        "decode_mean_speedup": _safe_speedup(
            reference["decode_summary"]["mean_ms"],
            candidate["decode_summary"]["mean_ms"],
        ),
        "end_to_end_speedup": _safe_speedup(
            reference["end_to_end_summary"]["mean_ms"],
            candidate["end_to_end_summary"]["mean_ms"],
        ),
        # The project log uses strict token-by-token equality when deciding
        # whether an optimization preserved generation quality.
        "same_generation": reference["generated_token_ids"] == candidate["generated_token_ids"],
        "token_diff_count": token_diff_count,
        "token_diff_rate": token_diff_rate,
    }
