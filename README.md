# Qwen3.5-4B INT8 Quantized CUDA Kernel

This repository contains the phase-one deliverables for a Qwen3.5-4B INT8 weight-only quantization project with CPU reference tooling and notebook-based outputs.

## What is included

- `phase1.ipynb`: a single notebook that runs through Task 1 to Task 5 and saves the generated outputs.
- `phase1_utils.py`: model loading, layer classification, benchmarking, and CSV export helpers.
- `quantize.py`: symmetric INT8 quantization, error analysis, and quantized artifact helpers.
- `cpu_reference.py`: CPU reference matmul / matvec implementations and correctness checks.
- `tests/`: unit tests for the phase-one helpers.
- `layer_list.csv`, `baseline_fp16.csv`, `baseline_results.csv`: generated results from phase one.
- `project_plan_v2.md`: the planning document used to guide the implementation.

## Notes on large files

Large model artifacts are intentionally not tracked in GitHub:

- `models/**/*.safetensors`
- `models/**/*.gguf`
- `quantized_weights.pt`

Those files stay local so the repository can remain public and lightweight. To run the notebook locally, place the Qwen3.5-4B checkpoint under `models/Qwen3.5-4B/` and use the `ECE9483 vllm-env` kernel.

## Verification

The phase-one helpers are covered by `tests/test_phase1_modules.py`, and the notebook has been executed with saved outputs.
