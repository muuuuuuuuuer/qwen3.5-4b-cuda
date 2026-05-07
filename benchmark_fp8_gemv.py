"""FP8 GEMV microbenchmark for Qwen3.5-4B decode projection shapes.

Compares FP8 Triton GEMV kernel against FP16 PyTorch native (cuBLAS) on
representative batch-1 decode projection shapes.

Usage:
    python benchmark_fp8_gemv.py                      # default shapes, warmup=100, iters=500
    python benchmark_fp8_gemv.py --warmup 200 --iters 1000
    python benchmark_fp8_gemv.py --output results.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from quantize_fp8_weights import quantize_weight_fp8
from triton_kernels.fp8_gemv import fp8_gemv, fp8_gemv_reference


QWEN35_PROJECTION_SHAPES: list[tuple[str, int, int]] = [
    ("FFN_gate", 9216, 2560),
    ("FFN_up", 9216, 2560),
    ("FFN_down", 2560, 9216),
    ("FullAttn_q", 8192, 2560),
    ("FullAttn_k", 1024, 2560),
    ("FullAttn_v", 1024, 2560),
    ("FullAttn_o", 2560, 4096),
    ("DeltaNet_qkv", 8192, 2560),
    ("DeltaNet_z", 4096, 2560),
    ("DeltaNet_out", 2560, 4096),
    ("DeltaNet_a", 32, 2560),
    ("DeltaNet_b", 32, 2560),
]


def benchmark_fp8_gemv(
    warmup: int = 100,
    iters: int = 500,
    output_csv: str | None = None,
) -> list[dict]:
    """Benchmark FP8 GEMV vs FP16 across Qwen3.5-4B projection shapes.

    Returns a list of result dictionaries.
    """
    import triton

    results = []
    header = f"{'Shape':<22s} {'FP16_ms':>9s} {'FP8_ms':>9s} {'Speedup':>8s}"
    print(header)
    print("-" * len(header))

    for name, N, K in QWEN35_PROJECTION_SHAPES:
        shape_str = f"{name} [{N},{K}]"

        # Prepare data
        w_fp16 = torch.randn(N, K, dtype=torch.float16, device="cuda")
        w_fp8, scale = quantize_weight_fp8(w_fp16, dim=0)
        w_fp8 = w_fp8.cuda()
        scale = scale.cuda()
        x = torch.randn(K, dtype=torch.float16, device="cuda")

        # Warm up cuBLAS
        for _ in range(warmup):
            torch.nn.functional.linear(x.unsqueeze(0), w_fp16).squeeze(0)
        torch.cuda.synchronize()

        # FP16 benchmark (cuBLAS)
        start = time.perf_counter()
        for _ in range(iters):
            torch.nn.functional.linear(x.unsqueeze(0), w_fp16).squeeze(0)
        torch.cuda.synchronize()
        fp16_elapsed = time.perf_counter() - start
        fp16_ms = fp16_elapsed / iters * 1000

        # Warm up Triton FP8
        for _ in range(warmup):
            fp8_gemv(w_fp8, scale, x, use_autotune=True)
        torch.cuda.synchronize()

        # FP8 benchmark
        start = time.perf_counter()
        for _ in range(iters):
            fp8_gemv(w_fp8, scale, x, use_autotune=True)
        torch.cuda.synchronize()
        fp8_elapsed = time.perf_counter() - start
        fp8_ms = fp8_elapsed / iters * 1000

        # Also try triton.testing.do_bench for more precise GPU timing
        fp16_ms_gpu = triton.testing.do_bench(
            lambda: torch.nn.functional.linear(x.unsqueeze(0), w_fp16).squeeze(0),
            warmup=warmup,
            rep=iters,
        )
        fp8_ms_gpu = triton.testing.do_bench(
            lambda: fp8_gemv(w_fp8, scale, x, use_autotune=True),
            warmup=warmup,
            rep=iters,
        )

        speedup = fp16_ms_gpu / fp8_ms_gpu if fp8_ms_gpu > 0 else 0.0

        result = {
            "shape": shape_str,
            "fp16_ms": round(fp16_ms_gpu, 4),
            "fp8_ms": round(fp8_ms_gpu, 4),
            "speedup": round(speedup, 4),
        }
        results.append(result)

        bar = "=" if speedup >= 1.0 else "-"
        print(
            f"{shape_str:<22s} {fp16_ms_gpu:>9.4f} {fp8_ms_gpu:>9.4f} "
            f"{speedup:>7.3f}x {bar}"
        )

    # Summary
    avg_speedup = sum(r["speedup"] for r in results) / len(results) if results else 0
    total_fp16 = sum(r["fp16_ms"] for r in results)
    total_fp8 = sum(r["fp8_ms"] for r in results)
    print(f"\nAverage speedup: {avg_speedup:.3f}x")
    print(f"Total FP16: {total_fp16:.3f} ms, Total FP8: {total_fp8:.3f} ms")

    if output_csv:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("shape,fp16_ms,fp8_ms,speedup\n")
            for r in results:
                f.write(f"{r['shape']},{r['fp16_ms']},{r['fp8_ms']},{r['speedup']}\n")
        print(f"Results saved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="FP8 GEMV microbenchmark for Qwen3.5-4B decode shapes"
    )
    parser.add_argument("--warmup", type=int, default=100,
                        help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=500,
                        help="Number of benchmark iterations")
    parser.add_argument("--output", type=str, default=None,
                        help="CSV output path for results")
    args = parser.parse_args()

    benchmark_fp8_gemv(
        warmup=args.warmup,
        iters=args.iters,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
