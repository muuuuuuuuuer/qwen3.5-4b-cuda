# Qwen3.5 Triton Benchmark Investigation Notes

## Goal

Figure out why the custom Triton DeltaNet decode path sometimes looked slower than FLA in end-to-end Qwen3.5 runs, despite looking much better in the isolated recurrent microbenchmark, then replace the benchmark method with a fairer one.

## Starting Point

The investigation started from two results that did not agree:

- Isolated recurrent microbenchmark:
  - `fla_fused_recurrent` around `74 us`
  - `triton_decode` around `26 us`
  - `triton_decode_fused_gates` around `29 us`
- Early end-to-end Qwen3.5 benchmark:
  - Triton sometimes trailed FLA

At face value, those two facts should not both be true unless something about the integration path or the benchmark method was distorting the comparison.

## Investigation Path

### 1. Verify the Triton path was actually being hit

The first question was whether the model was truly using the Triton decode path or silently falling back elsewhere.

What we checked:

- Patched Qwen3.5 DeltaNet decode into the real model path
- Counted decode calls through patch stats
- Compared generations and logits against the reference path

Outcome:

- The Triton decode path was definitely being called
- Generations stayed consistent
- This ruled out "the benchmark never used Triton" as the root cause

Relevant file:

- [qwen35_integration.py](/home/haozhong/ECE9483/triton_kernels/qwen35_integration.py)

### 2. Profile the real model instead of reasoning from the microbenchmark

The second question was whether the recurrent kernel was still a major bottleneck after integration.

What we checked:

- Real single-user profiler runs for FLA and Triton
- GPU hot ops and CPU launch overhead
- Trace-level recurrent and conv timings

Outcome:

- The real end-to-end request was dominated by projection work such as `aten::mm` and `gemvx::kernel`
- The recurrent kernel was only a small part of total request time
- This explained why a large microbenchmark win would not automatically turn into a large end-to-end win

Relevant files:

- [profile_qwen35_single_user.py](/home/haozhong/ECE9483/profile_qwen35_single_user.py)
- [qwen35_profiler.py](/home/haozhong/ECE9483/triton_kernels/qwen35_profiler.py)
- [single_user_profile_compare.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/profiler/single_user_profile_compare.json)
- [recurrent_kernel_summary.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/profiler/recurrent_kernel_summary.json)

### 3. Split Triton into two real model modes

The third question was whether the integrated `fused_gates` path itself was the problem.

We added two explicit decode modes:

- `triton_base`: gate ops outside the Triton kernel
- `triton_fused`: gate ops fused into the Triton kernel

Outcome:

- This made it possible to compare three real decode paths cleanly:
  - `fla`
  - `triton_base`
  - `triton_fused`

Relevant file:

- [qwen35_integration.py](/home/haozhong/ECE9483/triton_kernels/qwen35_integration.py)

### 4. Discover that the old single-user benchmark was order-sensitive

The most important turning point was testing whether simply reversing the mode order changed the winner.

Old method:

- Load model for one mode
- Run that mode to completion
- Load model for the next mode
- Keep a fixed order

What happened:

- When the order was `fla -> triton_base -> triton_fused`, FLA often looked best
- When the order was reversed, the later conclusion could flip

Outcome:

- This proved the old benchmark was not isolating the mode effect
- It was mixing in order-dependent effects such as GPU thermal state, allocator state, CUDA caches, and other run-history effects

This was the core benchmark fairness bug.

## Why the Old Benchmark Was Unfair

The old benchmark was unfair because it allowed mode comparison to be contaminated by run order.

Specifically, it did all of the following:

- loaded the model separately for each mode
- ran modes in a fixed sequence
- let each mode inherit a different GPU state depending on whether it ran early or late
- compared numbers gathered under different allocator and cache histories

That meant it was not just measuring:

- `mode A` vs `mode B`

It was also measuring:

- cold-ish GPU vs warmed GPU
- earlier allocator state vs later allocator state
- different CUDA runtime state

Once the order was shown to change the apparent winner, the old benchmark stopped being trustworthy as the primary source of truth.

## The Improved Benchmark Design

The replacement benchmark uses a shared model and round-robin mode rotation.

Design:

- load the model once
- reuse the same model instance
- switch runtime mode in place with `configure_qwen35_deltanet_runtime(...)`
- run warmup cycles first
- run measured cycles in round-robin order
- ensure each mode appears equally often in position 0, 1, and 2

For three modes and six measured cycles, the order is:

1. `fla, triton_base, triton_fused`
2. `triton_base, triton_fused, fla`
3. `triton_fused, fla, triton_base`
4. repeat

This does not make the benchmark perfect, but it removes the biggest fairness bug from the previous method.

Relevant files:

- [benchmark_qwen35_single_user_round_robin.py](/home/haozhong/ECE9483/benchmark_qwen35_single_user_round_robin.py)
- [qwen35_single_user_benchmark.py](/home/haozhong/ECE9483/triton_kernels/qwen35_single_user_benchmark.py)

## Current Conclusions

Under the fairer shared-model round-robin benchmark:

- `triton_base` is roughly on par with FLA
- `triton_fused` is modestly but consistently faster than FLA in this environment

Representative latest round-robin run:

- `fla`
  - TTFT: `50.212 ms`
  - decode mean: `33.709 ms`
  - end-to-end: `555.854 ms`
- `triton_base`
  - TTFT: `50.083 ms`
  - decode mean: `33.317 ms`
  - end-to-end: `549.844 ms`
- `triton_fused`
  - TTFT: `49.865 ms`
  - decode mean: `31.241 ms`
  - end-to-end: `518.487 ms`

Recent round-robin runs have been consistent on the main point:

- `triton_base` is not meaningfully worse than FLA
- `triton_fused` tends to lead by a small but real margin

Summary artifacts:

- [qwen35_single_user_round_robin_compare.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_single_user_round_robin_compare.json)
- [qwen35_single_user_round_robin_compare_run1.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_single_user_round_robin_compare_run1.json)
- [qwen35_single_user_round_robin_compare_run2.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_single_user_round_robin_compare_run2.json)
- [qwen35_single_user_round_robin_compare_run3.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_single_user_round_robin_compare_run3.json)
- [qwen35_single_user_round_robin_summary.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_single_user_round_robin_summary.json)

## Why `triton_fused` Wins in the Fairer Benchmark

This does not mean the fused recurrent kernel is universally faster in every narrow measurement.

What it means is:

- in the real Qwen3.5 decode path
- with the current integration
- under a fairer shared-model benchmark

the fused path reduces enough surrounding overhead that the whole decode step ends up faster.

The important distinction is:

- kernel-only timing and end-to-end timing are not the same thing
- fairness bugs in the benchmark can completely hide that difference

## GPU Clock Locking Attempt

We also tried to reduce frequency noise further by locking GPU graphics clocks, but the current user does not have permission to do that on this machine.

Relevant artifacts:

- [gpu_clock_query.txt](/home/haozhong/ECE9483/artifacts/qwen35_integration/gpu_clock_query.txt)
- [gpu_clock_lock_attempt.txt](/home/haozhong/ECE9483/artifacts/qwen35_integration/gpu_clock_lock_attempt.txt)

So the current best-practice method in this repo is:

- use the shared-model round-robin benchmark
- keep multiple repeated runs
- treat clock locking as optional only when permissions allow it

## Files Removed During Cleanup

The following obsolete files were removed because they were no longer part of the current benchmark path:

- `benchmark_qwen35_deltanet_integration.py`
- `artifacts/qwen35_integration/qwen35_triton_patch_results.json`
- `artifacts/qwen35_integration/qwen35_mode_compare.json`
- `artifacts/qwen35_integration/qwen35_single_user_compare.json`
- `artifacts/qwen35_integration/qwen35_single_user_compare_order_fla_first.json`
- `artifacts/qwen35_integration/qwen35_single_user_compare_order_triton_first.json`
- `artifacts/qwen35_integration/qwen35_single_user_order_bias_summary.json`
- `artifacts/qwen35_integration/profiler/triton_single_user_trace.json`

## Recommended Source of Truth Going Forward

Use these as the current source of truth:

- benchmark script:
  - [benchmark_qwen35_single_user_round_robin.py](/home/haozhong/ECE9483/benchmark_qwen35_single_user_round_robin.py)
- integration logic:
  - [qwen35_integration.py](/home/haozhong/ECE9483/triton_kernels/qwen35_integration.py)
- benchmark scheduling helpers:
  - [qwen35_single_user_benchmark.py](/home/haozhong/ECE9483/triton_kernels/qwen35_single_user_benchmark.py)
- round-robin results:
  - [qwen35_single_user_round_robin_summary.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_single_user_round_robin_summary.json)
- profiler summary:
  - [single_user_profile_compare.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/profiler/single_user_profile_compare.json)

If a future benchmark result disagrees with these conclusions, the first things to check are:

1. whether the benchmark reused one loaded model or reloaded per mode
2. whether mode order was balanced
3. whether the GPU state changed during the run
4. whether the same prompt and generation settings were used
