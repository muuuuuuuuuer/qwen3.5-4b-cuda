"""Phase 2A microbenchmark driver for decode-specialized DeltaNet Triton kernels.

Reproduces the kernel-only comparison against PyTorch naive and FLA recorded in
the project log before end-to-end integration.
"""

from __future__ import annotations

import argparse
import math
from typing import Callable

import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
from triton_kernels.deltanet_decode import (
    DEFAULT_DELTANET_KERNEL_CONFIG,
    DEFAULT_FUSED_GATE_KERNEL_CONFIG,
    DeltaNetKernelConfig,
    deltanet_decode_step,
    deltanet_decode_step_fused_gates,
    deltanet_l2_normalize_qk,
)


H = 16
HV = 32
K = 128
V = 128
GVA_RATIO = HV // H


def make_inputs(device: str = "cuda") -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    q = torch.randn(H, K, dtype=torch.float16, device=device)
    k = torch.randn(H, K, dtype=torch.float16, device=device)
    v = torch.randn(HV, V, dtype=torch.float16, device=device)
    q_expanded = q.repeat_interleave(GVA_RATIO, dim=0)
    k_expanded = k.repeat_interleave(GVA_RATIO, dim=0)
    a = torch.randn(HV, dtype=torch.float32, device=device)
    b = torch.randn(HV, dtype=torch.float32, device=device)
    a_log = torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)
    g = -torch.exp(a_log) * F.softplus(a + dt_bias)
    beta = torch.sigmoid(b)
    state = torch.randn(HV, K, V, dtype=torch.float32, device=device) * 0.01
    return {
        "q": q,
        "k": k,
        "v": v,
        "q_expanded": q_expanded,
        "k_expanded": k_expanded,
        "a": a,
        "b": b,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "g": g,
        "beta": beta,
        "state": state,
    }


def pytorch_naive_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    q_f32 = F.normalize(q.float(), dim=-1) * scale
    k_f32 = F.normalize(k.float(), dim=-1)
    state.mul_(torch.exp(g).view(-1, 1, 1))
    projection = torch.einsum("hkv,hk->hv", state, k_f32)
    delta = beta.view(-1, 1) * (v.float() - projection)
    state.add_(torch.einsum("hk,hv->hkv", k_f32, delta))
    return torch.einsum("hkv,hk->hv", state, q_f32).to(q.dtype)


def fla_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    output, final_state = fused_recurrent_gated_delta_rule(
        q=q.view(1, 1, H, K),
        k=k.view(1, 1, H, K),
        v=v.view(1, 1, HV, V),
        g=g.view(1, 1, HV),
        beta=beta.view(1, 1, HV),
        scale=1.0 / math.sqrt(K),
        initial_state=state.view(1, HV, K, V),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    state.copy_(final_state[0])
    return output[0, 0]


def measure_latency_us(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def build_benchmarks(inputs: dict[str, torch.Tensor]) -> list[tuple[str, Callable[[], torch.Tensor]]]:
    state_naive = inputs["state"].clone()
    state_fla = inputs["state"].clone()
    state_triton_manual = inputs["state"].clone()
    state_triton_autotune = inputs["state"].clone()
    state_fused_manual = inputs["state"].clone()
    state_fused_autotune = inputs["state"].clone()
    state_fused_prenorm_manual = inputs["state"].clone()
    state_fused_prenorm_kernel_only = inputs["state"].clone()
    state_fused_prenorm_with_normalize = inputs["state"].clone()
    q_norm, k_norm = deltanet_l2_normalize_qk(inputs["q_expanded"], inputs["k_expanded"])

    def fused_prenorm_with_normalize_step() -> torch.Tensor:
        current_q_norm, current_k_norm = deltanet_l2_normalize_qk(inputs["q_expanded"], inputs["k_expanded"])
        return deltanet_decode_step_fused_gates(
            current_q_norm,
            current_k_norm,
            inputs["v"],
            inputs["a"],
            inputs["b"],
            inputs["a_log"],
            inputs["dt_bias"],
            state_fused_prenorm_with_normalize,
            use_qk_l2norm=False,
        )

    return [
        (
            "pytorch_naive",
            lambda: pytorch_naive_step(
                inputs["q_expanded"],
                inputs["k_expanded"],
                inputs["v"],
                inputs["g"],
                inputs["beta"],
                state_naive,
            ),
        ),
        (
            "fla_fused_recurrent",
            lambda: fla_step(
                inputs["q"],
                inputs["k"],
                inputs["v"],
                inputs["g"],
                inputs["beta"],
                state_fla,
            ),
        ),
        (
            "triton_decode_manual_bv32",
            lambda: deltanet_decode_step(
                inputs["q_expanded"],
                inputs["k_expanded"],
                inputs["v"],
                inputs["g"],
                inputs["beta"],
                state_triton_manual,
                kernel_config=DEFAULT_DELTANET_KERNEL_CONFIG,
            ),
        ),
        (
            "triton_decode_autotune",
            lambda: deltanet_decode_step(
                inputs["q_expanded"],
                inputs["k_expanded"],
                inputs["v"],
                inputs["g"],
                inputs["beta"],
                state_triton_autotune,
            ),
        ),
        (
            "triton_fused_manual_bv64",
            lambda: deltanet_decode_step_fused_gates(
                inputs["q_expanded"],
                inputs["k_expanded"],
                inputs["v"],
                inputs["a"],
                inputs["b"],
                inputs["a_log"],
                inputs["dt_bias"],
                state_fused_manual,
                kernel_config=DEFAULT_FUSED_GATE_KERNEL_CONFIG,
            ),
        ),
        (
            "triton_fused_autotune",
            lambda: deltanet_decode_step_fused_gates(
                inputs["q_expanded"],
                inputs["k_expanded"],
                inputs["v"],
                inputs["a"],
                inputs["b"],
                inputs["a_log"],
                inputs["dt_bias"],
                state_fused_autotune,
            ),
        ),
        (
            "triton_fused_prenorm_bv64",
            lambda: deltanet_decode_step_fused_gates(
                q_norm,
                k_norm,
                inputs["v"],
                inputs["a"],
                inputs["b"],
                inputs["a_log"],
                inputs["dt_bias"],
                state_fused_prenorm_manual,
                use_qk_l2norm=False,
                kernel_config=DEFAULT_FUSED_GATE_KERNEL_CONFIG,
            ),
        ),
        (
            "triton_fused_prenorm_kernel",
            lambda: deltanet_decode_step_fused_gates(
                q_norm,
                k_norm,
                inputs["v"],
                inputs["a"],
                inputs["b"],
                inputs["a_log"],
                inputs["dt_bias"],
                state_fused_prenorm_kernel_only,
                use_qk_l2norm=False,
            ),
        ),
        (
            "triton_fused_prenorm_included",
            fused_prenorm_with_normalize_step,
        ),
    ]


def build_fused_sweep_benchmarks(
    inputs: dict[str, torch.Tensor],
    configs: list[DeltaNetKernelConfig],
) -> list[tuple[str, Callable[[], torch.Tensor]]]:
    benchmarks = []
    for config in configs:
        state = inputs["state"].clone()
        name = f"fused_bv{config.bv}_w{config.num_warps}_s{config.num_stages}"
        benchmarks.append(
            (
                name,
                lambda config=config, state=state: deltanet_decode_step_fused_gates(
                    inputs["q_expanded"],
                    inputs["k_expanded"],
                    inputs["v"],
                    inputs["a"],
                    inputs["b"],
                    inputs["a_log"],
                    inputs["dt_bias"],
                    state,
                    kernel_config=config,
                ),
            )
        )
    return benchmarks


def run_benchmarks(
    benchmarks: list[tuple[str, Callable[[], torch.Tensor]]],
    warmup: int,
    iters: int,
) -> list[dict[str, float | str]]:
    rows = []
    baseline_us = None
    for name, fn in benchmarks:
        latency_us = measure_latency_us(fn, warmup=warmup, iters=iters)
        if baseline_us is None:
            baseline_us = latency_us
        rows.append(
            {
                "implementation": name,
                "latency_us": latency_us,
                "relative_speedup": baseline_us / latency_us,
            }
        )
    return rows


def print_rows(rows: list[dict[str, float | str]]) -> None:
    print(f"{'implementation':28s} {'latency_us':>12s} {'relative_speedup':>18s}")
    print(f"{'-' * 28} {'-' * 12} {'-' * 18}")
    for row in rows:
        print(
            f"{str(row['implementation']):28s} "
            f"{float(row['latency_us']):12.3f} "
            f"{float(row['relative_speedup']):18.3f}"
        )


def candidate_fused_configs() -> list[DeltaNetKernelConfig]:
    return [
        DeltaNetKernelConfig(bv=bv, num_warps=num_warps, num_stages=num_stages)
        for bv in (16, 32, 64, 128)
        for num_warps in (2, 4, 8)
        for num_stages in (1, 2)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark decode-specialized DeltaNet kernels")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--sweep-fused", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    inputs = make_inputs(device="cuda")
    rows = run_benchmarks(build_benchmarks(inputs), warmup=args.warmup, iters=args.iters)
    print_rows(rows)

    if args.sweep_fused:
        print()
        print("Fused-gate config sweep:")
        sweep_rows = run_benchmarks(
            build_fused_sweep_benchmarks(inputs, candidate_fused_configs()),
            warmup=args.warmup,
            iters=args.iters,
        )
        sweep_rows = sorted(sweep_rows, key=lambda row: float(row["latency_us"]))
        best_latency = float(sweep_rows[0]["latency_us"])
        for row in sweep_rows:
            row["relative_speedup"] = best_latency / float(row["latency_us"])
        print_rows(sweep_rows)
        best = sweep_rows[0]
        print()
        print(
            "best_fused_config:",
            best["implementation"],
            f"latency_us={float(best['latency_us']):.3f}",
        )


if __name__ == "__main__":
    main()
