# Qwen3.5-4B DeltaNet Decode Operators

Focused CUDA/Triton work for accelerating the Qwen3.5-4B DeltaNet decode path.

The repository now keeps one clean optimization ladder:

1. PyTorch / FLA baselines
2. Triton DeltaNet recurrent decode
3. Fused gate math inside the DeltaNet kernel
4. Packed low-rank beta gate DeltaNet path
5. Static-cache + `torch.compile` report modes for comparison

Removed from the active code path: full-attention fusion experiments, MTP self-speculation, INT8/AWQ/GPTQ quantization, and fused-FFN/back8 side tracks.

## Repository Layout

| Path | Purpose |
|---|---|
| [`triton_kernels/deltanet_decode.py`](triton_kernels/deltanet_decode.py) | Triton decode kernels and PyTorch references for DeltaNet recurrent updates |
| [`triton_kernels/qwen35_projection_pack.py`](triton_kernels/qwen35_projection_pack.py) | Packed projection/conv helper used by the final packed path |
| [`triton_kernels/qwen35_integration.py`](triton_kernels/qwen35_integration.py) | Runtime patching for `torch`, `fla`, `triton_fused`, and `triton_lowrank_beta_gate_packed` |
| [`triton_kernels/qwen35_static_cache_integration.py`](triton_kernels/qwen35_static_cache_integration.py) | Minimal `StaticCache` helper for static compiled report modes |
| [`triton_kernels/qwen35_compile_integration.py`](triton_kernels/qwen35_compile_integration.py) | Minimal `torch.compile` wrapper for static compiled report modes |
| [`benchmark_deltanet_decode.py`](benchmark_deltanet_decode.py) | Kernel-only DeltaNet microbenchmark |
| [`benchmark_qwen35_single_user_round_robin.py`](benchmark_qwen35_single_user_round_robin.py) | Shared-model round-robin end-to-end latency benchmark |
| [`benchmark_qwen35_gsm8k_final.py`](benchmark_qwen35_gsm8k_final.py) | GSM8K quality/latency benchmark |
| [`profile_qwen35_single_user.py`](profile_qwen35_single_user.py) | Profiler entry point for end-to-end hotspot attribution |

## Supported Modes

The final GSM8K benchmark intentionally exposes only these modes:

| Mode | Runtime |
|---|---|
| `torch` | Transformers/PyTorch fallback |
| `fla` | Flash Linear Attention runtime |
| `fp16_eager` | Eager Triton fused DeltaNet |
| `fp16_eager_packed` | Eager packed low-rank beta gate DeltaNet |
| `fp16_static_compiled_attn_only_deltanet` | Static cache + compiled full-attention blocks + fused DeltaNet |
| `fp16_static_compiled_attn_only_deltanet_packed` | Static cache + compiled full-attention blocks + packed DeltaNet |

## Key Results

### GSM8K 50-Question Final Benchmark

Source: [`artifacts/qwen35_integration/qwen35_gsm8k_final_summary.md`](artifacts/qwen35_integration/qwen35_gsm8k_final_summary.md)

| Mode | Accuracy | Decode mean (ms/token) | Tokens/sec |
|---|---:|---:|---:|
| `fp16_eager_packed` | 0.900 | 29.705 | 33.66 |
| `fp16_static_compiled_attn_only_deltanet_packed` | 0.900 | 33.964 | 29.44 |
| `fp16_eager` | 0.900 | 36.047 | 27.74 |
| `fla` | 0.900 | 38.831 | 25.75 |
| `fp16_static_compiled_attn_only_deltanet` | 0.880 | 39.141 | 25.55 |
| `torch` | 0.900 | 46.996 | 21.28 |

### Packed DeltaNet Round-Robin Decode

Sources:

- [`qwen35_phase17_fused_packed_round_robin_gen16.json`](artifacts/qwen35_integration/qwen35_phase17_fused_packed_round_robin_gen16.json)
- [`qwen35_phase17_fused_packed_round_robin_gen128.json`](artifacts/qwen35_integration/qwen35_phase17_fused_packed_round_robin_gen128.json)

| Mode | gen=16 decode mean | gen=128 decode mean |
|---|---:|---:|
| `fla` | 34.540 ms/token | 32.534 ms/token |
| `triton_fused` | 30.783 ms/token | 30.432 ms/token |
| `triton_lowrank_beta_gate` | 33.316 ms/token | 33.314 ms/token |
| `triton_lowrank_beta_gate_packed` | 27.763 ms/token | 26.996 ms/token |

### Static Compiled Packed Comparison

Sources:

- [`qwen35_static_compile_packed_deltanet_gen16.json`](artifacts/qwen35_integration/qwen35_static_compile_packed_deltanet_gen16.json)
- [`qwen35_static_compile_packed_deltanet_gen128.json`](artifacts/qwen35_integration/qwen35_static_compile_packed_deltanet_gen128.json)

| Mode | gen=16 decode mean | gen=128 decode mean |
|---|---:|---:|
| `fp16_static_compiled_attn_only_deltanet` | 28.330 ms/token | 27.829 ms/token |
| `fp16_static_compiled_attn_only_deltanet_packed` | 29.269 ms/token | 24.098 ms/token |

## Running Benchmarks

Place the Qwen3.5-4B checkpoint under `models/Qwen3.5-4B/`.

Kernel-only microbenchmark:

```bash
~/vllm-env/bin/python benchmark_deltanet_decode.py
```

Round-robin single-user decode benchmark:

```bash
~/vllm-env/bin/python benchmark_qwen35_single_user_round_robin.py \
  --modes fla triton_fused triton_lowrank_beta_gate_packed \
  --gen-tokens 128 \
  --runs 3
```

GSM8K final benchmark:

```bash
~/vllm-env/bin/python benchmark_qwen35_gsm8k_final.py \
  --modes torch fla fp16_eager fp16_eager_packed \
          fp16_static_compiled_attn_only_deltanet \
          fp16_static_compiled_attn_only_deltanet_packed
```

## Verification

Unit tests for the retained project surface:

```bash
~/vllm-env/bin/python -m unittest \
  triton_kernels.test_deltanet_decode \
  triton_kernels.test_qwen35_integration \
  triton_kernels.test_qwen35_projection_pack \
  triton_kernels.test_qwen35_single_user_benchmark \
  triton_kernels.test_qwen35_profiler \
  triton_kernels.test_benchmark_qwen35_gsm8k_final -v
```

Syntax check:

```bash
~/vllm-env/bin/python -m py_compile \
  benchmark_deltanet_decode.py \
  benchmark_qwen35_single_user.py \
  benchmark_qwen35_single_user_round_robin.py \
  benchmark_qwen35_gsm8k_final.py \
  profile_qwen35_single_user.py \
  phase1_utils.py \
  triton_kernels/deltanet_decode.py \
  triton_kernels/qwen35_integration.py \
  triton_kernels/qwen35_projection_pack.py \
  triton_kernels/qwen35_compile_integration.py \
  triton_kernels/qwen35_static_cache_integration.py
```

## Benchmark Artifacts

The comparison artifacts pushed with this repository live in [`artifacts/qwen35_integration/`](artifacts/qwen35_integration/).

Most useful files:

- [`deltanet_trend_summary.json`](artifacts/qwen35_integration/deltanet_trend_summary.json)
- [`deltanet_decode_line_chart_data.csv`](artifacts/qwen35_integration/deltanet_decode_line_chart_data.csv)
- [`qwen35_phase17_fused_packed_round_robin_gen16.json`](artifacts/qwen35_integration/qwen35_phase17_fused_packed_round_robin_gen16.json)
- [`qwen35_phase17_fused_packed_round_robin_gen128.json`](artifacts/qwen35_integration/qwen35_phase17_fused_packed_round_robin_gen128.json)
- [`qwen35_static_compile_packed_deltanet_gen16.json`](artifacts/qwen35_integration/qwen35_static_compile_packed_deltanet_gen16.json)
- [`qwen35_static_compile_packed_deltanet_gen128.json`](artifacts/qwen35_integration/qwen35_static_compile_packed_deltanet_gen128.json)
- [`qwen35_gsm8k_final_summary.md`](artifacts/qwen35_integration/qwen35_gsm8k_final_summary.md)

## Large Files

Model weights are intentionally not tracked:

- `models/**/*.safetensors`
- `models/**/*.gguf`
- `*.pt`
- `*.pth`
- `*.bin`
- `*.ckpt`

This keeps the GitHub repository lightweight while preserving code, tests, and benchmark comparisons.
