# Qwen3.5-4B INT8 Quantized CUDA Kernel

This repository contains the phase-one quantization deliverables and the follow-on Triton decode-kernel work for Qwen3.5-4B. The current performance work focuses on the DeltaNet decode path inside Qwen3.5 and compares PyTorch fallback, FLA, and custom Triton kernels under both microbenchmark and end-to-end serving-style measurements.

## What is included

- `phase1.ipynb`: notebook for the original phase-one tasks.
- `phase1_utils.py`: model loading, layer classification, benchmarking, and CSV export helpers.
- `quantize.py`: symmetric INT8 quantization, error analysis, and quantized artifact helpers.
- `cpu_reference.py`: CPU reference matmul / matvec implementations and correctness checks.
- `triton_kernels/deltanet_decode.py`: decode-specialized Triton kernels and PyTorch reference path for DeltaNet recurrent updates.
- `triton_kernels/qwen35_integration.py`: Qwen3.5 runtime integration for `fla`, `triton_base`, and `triton_fused`.
- `benchmark_deltanet_decode.py`: kernel-only recurrent microbenchmark.
- `benchmark_qwen35_single_user_round_robin.py`: shared-model round-robin single-user benchmark for fair end-to-end comparisons.
- `profile_qwen35_single_user.py`: profiler entry point for locating real end-to-end hotspots.
- `triton_kernels/` and `tests/`: unit tests for the Triton integration, benchmark helpers, and profiler utilities.

## DeltaNet Decode Kernel

### Design Idea

The custom kernel targets the exact Qwen3.5 DeltaNet decode case we care about:

- single-token decode (`T=1`)
- low-batch online serving
- recurrent state update repeated once per generated token per DeltaNet layer

Instead of keeping the full generality of the FLA recurrent kernel, the Triton kernel specializes for this serving path:

- no time-step loop for long sequences
- no backward path
- no variable-length machinery
- full `K=128` tile
- larger `V`-tile than the generic FLA recurrent kernel

The repository currently exposes two decode kernels:

- `triton_base`: the recurrent update is done in Triton, but gate math stays outside the kernel
- `triton_fused`: the recurrent update and gate math are fused into one decode-specialized path

### Why This Kernel Exists

The target use case is interactive generation:

- single-user chat
- agent rollouts
- streaming decode
- any batch-1 or small-batch serving path where `decode` dominates over `prefill`

This matters because the DeltaNet recurrent step runs for every generated token across every DeltaNet layer. Even a modest per-token saving accumulates over longer generations. At the same time, this kernel is not designed to accelerate prefill or high-throughput large-batch training, so those scenarios should not be expected to benefit as much.

## Performance Trend

Latest summary artifact:

- [deltanet_trend_summary.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/deltanet_trend_summary.json)

### Report Table 1: Kernel And Decode Summary

| Scenario | Scope | FLA | Triton Base | Triton Fused | Fused vs FLA |
|---|---|---:|---:|---:|---:|
| Microbenchmark | recurrent kernel only (`us`) | `94.446` | `28.965` | `33.580` | `2.813x` |
| `gen=16` | decode mean (`ms/token`) | `33.709` | `33.317` | `31.241` | `1.079x` |
| `gen=128` | decode mean (`ms/token`) | `54.522` | `52.006` | `51.229` | `1.064x` |
| `gen=256` | decode mean (`ms/token`) | `54.905` | `54.043` | `50.964` | `1.077x` |

Reference files:

- [deltanet_microbenchmark_latest.txt](/home/haozhong/ECE9483/artifacts/qwen35_integration/deltanet_microbenchmark_latest.txt)
- [qwen35_single_user_round_robin_compare_run3.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_single_user_round_robin_compare_run3.json)
- [qwen35_single_user_round_robin_compare_gen128.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_single_user_round_robin_compare_gen128.json)
- [qwen35_single_user_round_robin_compare_gen256.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_single_user_round_robin_compare_gen256.json)

### Report Table 2: End-To-End Summary

| Scenario | FLA e2e (`ms`) | Triton Base e2e (`ms`) | Triton Fused e2e (`ms`) | Fused vs FLA |
|---|---:|---:|---:|---:|
| `gen=16` | `555.854` | `549.844` | `518.487` | `1.072x` |
| `gen=128` | `7008.485` | `6677.568` | `6584.444` | `1.064x` |
| `gen=256` | `14077.691` | `13859.910` | `13076.210` | `1.077x` |

### Line Chart Data Table

Plot-ready CSV:

- [deltanet_decode_line_chart_data.csv](/home/haozhong/ECE9483/artifacts/qwen35_integration/deltanet_decode_line_chart_data.csv)

| gen tokens | FLA decode (`ms/token`) | Triton Base decode (`ms/token`) | Triton Fused decode (`ms/token`) | FLA e2e (`ms`) | Triton Base e2e (`ms`) | Triton Fused e2e (`ms`) |
|---|---:|---:|---:|---:|---:|---:|
| `16` | `33.709` | `33.317` | `31.241` | `555.854` | `549.844` | `518.487` |
| `128` | `54.522` | `52.006` | `51.229` | `7008.485` | `6677.568` | `6584.444` |
| `256` | `54.905` | `54.043` | `50.964` | `14077.691` | `13859.910` | `13076.210` |

## How To Read These Results

### Where The Biggest Improvement Shows Up

The largest relative gain appears in the kernel-only microbenchmark because that experiment isolates the exact operation the Triton kernel was written to accelerate.

### Where The Practical Improvement Matters

The practical serving win shows up in decode-heavy single-user generation:

- the kernel runs once per generated token
- the same savings are applied across all DeltaNet layers
- longer outputs amortize TTFT and make decode behavior more representative

In the current repository state, `triton_fused` stays modestly ahead of FLA in these end-to-end round-robin benchmarks, typically around `6%` to `8%`.

### Why The End-to-End Win Is Smaller Than The Microbenchmark Win

The profiler shows that the recurrent update is not the only hot path. Real end-to-end runs are still heavily dominated by projection work such as:

- `aten::mm`
- `gemvx::kernel`

That means the recurrent kernel can improve a lot in isolation while the total request only improves by a smaller amount. This is expected and is the main reason the microbenchmark gain is much larger than the full-model gain.

Relevant profiler outputs:

- [single_user_profile_compare.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/profiler/single_user_profile_compare.json)
- [recurrent_kernel_summary.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/profiler/recurrent_kernel_summary.json)

## Benchmark Method Notes

The current source-of-truth benchmark is the shared-model round-robin script:

- [benchmark_qwen35_single_user_round_robin.py](/home/haozhong/ECE9483/benchmark_qwen35_single_user_round_robin.py)

We use this because an older fixed-order benchmark was order-sensitive: changing the order of `fla` and Triton modes could change the apparent winner. The round-robin version reuses one loaded model and rotates mode order to make the comparison fairer.

The full investigation write-up is here:

- [2026-04-08-qwen35-benchmark-investigation.md](/home/haozhong/ECE9483/docs/superpowers/plans/2026-04-08-qwen35-benchmark-investigation.md)

## Notes on Large Files

Large model artifacts are intentionally not tracked in GitHub:

- `models/**/*.safetensors`
- `models/**/*.gguf`
- `quantized_weights.pt`

Those files stay local so the repository can remain public and lightweight. To run the notebooks or benchmarks locally, place the Qwen3.5-4B checkpoint under `models/Qwen3.5-4B/` and use the `ECE9483 vllm-env` kernel.

## Verification

The current Triton benchmark and integration helpers were verified with:

- `~/vllm-env/bin/python -m unittest triton_kernels.test_qwen35_single_user_benchmark triton_kernels.test_qwen35_integration triton_kernels.test_qwen35_profiler -v`
- `~/vllm-env/bin/python -m py_compile benchmark_qwen35_single_user.py benchmark_qwen35_single_user_round_robin.py profile_qwen35_single_user.py triton_kernels/qwen35_integration.py`
