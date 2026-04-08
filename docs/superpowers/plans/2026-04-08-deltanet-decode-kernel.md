# DeltaNet Decode Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a decode-specialized Triton kernel for Qwen3.5 DeltaNet recurrent updates, with correctness tests and a benchmark comparing PyTorch, FLA, and the custom kernel.

**Architecture:** Keep the implementation split into a focused kernel module, a focused test module, and a standalone benchmark script. The kernel module will provide a pure PyTorch reference path, a Triton base kernel and wrapper, and an optional fused-gate kernel and wrapper so correctness and performance can be validated independently.

**Tech Stack:** Python 3.12, PyTorch CUDA, Triton 3.5, `flash-linear-attention`, `unittest`

---

### Task 1: Add Correctness Tests First

**Files:**
- Create: `triton_kernels/test_deltanet_decode.py`

- [ ] **Step 1: Write failing tests for the PyTorch reference helper**

Write tests that construct random `q`, `k`, `v`, `g`, `beta`, and `state` tensors using the real Qwen3.5 decode dimensions `HV=32`, `K=128`, `V=128`, then assert the reference output shape is `[HV, V]` and the state shape remains `[HV, K, V]`.

- [ ] **Step 2: Run the new test file and verify it fails**

Run: `~/vllm-env/bin/python -m unittest triton_kernels.test_deltanet_decode.DeltaNetDecodeReferenceTests -v`
Expected: FAIL because `triton_kernels/deltanet_decode.py` does not exist yet.

- [ ] **Step 3: Add failing tests for the Triton wrapper correctness**

Write CUDA-only tests comparing the custom Triton decode step against the reference implementation. Check output cosine similarity and max absolute error, and also compare the updated recurrent state.

- [ ] **Step 4: Run the wrapper tests and verify they fail for the expected reason**

Run: `~/vllm-env/bin/python -m unittest triton_kernels.test_deltanet_decode.DeltaNetDecodeKernelTests -v`
Expected: FAIL because the wrapper is not implemented yet.

### Task 2: Implement the Base Kernel Module

**Files:**
- Create: `triton_kernels/deltanet_decode.py`
- Test: `triton_kernels/test_deltanet_decode.py`

- [ ] **Step 1: Implement the PyTorch reference path**

Add `deltanet_decode_reference(q, k, v, g, beta, state, use_qk_l2norm=True)` with FP32 math and in-place state updates on a cloned temporary tensor so the semantics are explicit.

- [ ] **Step 2: Implement the Triton base kernel**

Add `deltanet_decode_fused_kernel` specialized for one decode step using `grid=(NV, HV)`, `BK=128`, `BV=32`, masked loads/stores, optional q/k L2 normalization, decay, delta update, and output computation.

- [ ] **Step 3: Implement the Python wrapper**

Add `deltanet_decode_step(q, k, v, g, beta, state, use_qk_l2norm=True)` with input validation, contiguity handling, output allocation, kernel launch, and in-place state update behavior.

- [ ] **Step 4: Run the reference tests**

Run: `~/vllm-env/bin/python -m unittest triton_kernels.test_deltanet_decode.DeltaNetDecodeReferenceTests -v`
Expected: PASS

- [ ] **Step 5: Run the Triton correctness tests**

Run: `~/vllm-env/bin/python -m unittest triton_kernels.test_deltanet_decode.DeltaNetDecodeKernelTests -v`
Expected: PASS on CUDA with the target error thresholds.

### Task 3: Add the Benchmark Harness

**Files:**
- Create: `benchmark_deltanet_decode.py`
- Modify: `triton_kernels/deltanet_decode.py`

- [ ] **Step 1: Inspect the FLA helper signature**

Read the local `fla` function signature and adapt shapes so the benchmark invokes the real fused recurrent path, not an approximation.

- [ ] **Step 2: Implement benchmark helpers**

Add helpers for random input generation, warmup loops, `torch.cuda.Event` timing, and formatting summary rows for PyTorch naive, FLA, base Triton kernel, and optional fused-gate Triton kernel.

- [ ] **Step 3: Run the benchmark script**

Run: `~/vllm-env/bin/python benchmark_deltanet_decode.py`
Expected: a table of `implementation`, `latency_us`, and `relative_speedup`.

### Task 4: Add the Fused-Gate Variant

**Files:**
- Modify: `triton_kernels/deltanet_decode.py`
- Modify: `triton_kernels/test_deltanet_decode.py`

- [ ] **Step 1: Add failing tests for raw `a/b` gate inputs**

Write CUDA-only tests that compare a fused-gate wrapper against the reference path after computing `g` and `beta` from `a`, `b`, `A_log`, and `dt_bias`.

- [ ] **Step 2: Run the fused-gate tests and verify they fail**

Run: `~/vllm-env/bin/python -m unittest triton_kernels.test_deltanet_decode.DeltaNetDecodeFusedGateTests -v`
Expected: FAIL because the fused-gate kernel does not exist yet.

- [ ] **Step 3: Implement the fused-gate Triton kernel and wrapper**

Add a second kernel entry point plus a wrapper that accepts raw gate inputs and computes `beta` and `g` inside the Triton kernel.

- [ ] **Step 4: Re-run the fused-gate tests**

Run: `~/vllm-env/bin/python -m unittest triton_kernels.test_deltanet_decode.DeltaNetDecodeFusedGateTests -v`
Expected: PASS if the fused-gate path is numerically stable.

### Task 5: Final Verification

**Files:**
- Modify: `triton_kernels/deltanet_decode.py`
- Modify: `triton_kernels/test_deltanet_decode.py`
- Create: `benchmark_deltanet_decode.py`

- [ ] **Step 1: Run the focused Triton test suite**

Run: `~/vllm-env/bin/python -m unittest triton_kernels.test_deltanet_decode -v`
Expected: PASS

- [ ] **Step 2: Run the repository unit tests**

Run: `~/vllm-env/bin/python -m unittest tests.test_phase1_modules -v`
Expected: PASS

- [ ] **Step 3: Re-run the benchmark and capture the latest numbers**

Run: `~/vllm-env/bin/python benchmark_deltanet_decode.py`
Expected: benchmark table with the final implementations.

### Task 6: Tune Triton Launch Parameters

**Files:**
- Modify: `triton_kernels/deltanet_decode.py`
- Modify: `triton_kernels/test_deltanet_decode.py`
- Modify: `benchmark_deltanet_decode.py`

- [ ] **Step 1: Add failing tests for configurable kernel launch parameters**

Write CUDA-only tests that run the base and fused-gate wrappers with non-default `BV`, `num_warps`, and `num_stages`, then compare them against the reference path.

- [ ] **Step 2: Run the config tests and verify they fail**

Run: `~/vllm-env/bin/python -m unittest triton_kernels.test_deltanet_decode.DeltaNetDecodeTuningConfigTests -v`
Expected: FAIL because the wrappers do not yet accept explicit tuning configs.

- [ ] **Step 3: Add a tuning config abstraction and plumb it through the wrappers**

Implement a small config object or equivalent validation helper so both wrappers can launch with alternate `BV`, `num_warps`, and `num_stages` values.

- [ ] **Step 4: Extend the benchmark script to sweep candidate configs**

Add a mode that benchmarks several candidate configs for the fused-gate kernel and reports the best one.

- [ ] **Step 5: Run the sweep and set the best observed default**

Run: `~/vllm-env/bin/python benchmark_deltanet_decode.py --warmup 100 --iters 1000 --sweep-fused`
Expected: a ranked table of fused-gate configs and a best observed configuration.
