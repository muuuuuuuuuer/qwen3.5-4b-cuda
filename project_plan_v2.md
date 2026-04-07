# Qwen3.5-4B INT8 Quantized CUDA Kernel — 完整统筹计划 v2
# 核心卖点：针对混合架构（DeltaNet vs Full Attention vs FFN）的差异化量化分析

---

## 项目一句话

对 Qwen3.5-4B 的三种线性层（DeltaNet projection / Full Attention projection / FFN projection）分别实现 INT8 weight-only 量化 CUDA GEMV kernel，对比 FP16 原始实现和 cuBLAS INT8，分析混合架构下不同层类型的量化收益差异。

---

## 为什么这个项目独特

Qwen3.5 是第一批大规模使用 Gated DeltaNet 混合架构的生产模型。
它的 32 层里 24 层是 DeltaNet（linear attention），8 层是标准 full attention。
DeltaNet 层的 Q/K/V projection shape 和标准 attention 层完全不同。
目前没有人专门针对 DeltaNet 层做过量化 kernel 分析。
你们的独特贡献 = "首次对 DeltaNet 混合架构中不同层类型的量化 kernel 性能做系统对比"。

---

## Qwen3.5-4B 架构关键参数（from config.json）

```
hidden_size: 2560
intermediate_size: 9216
num_hidden_layers: 32
num_attention_heads: 16        # full attention 层用
num_key_value_heads: 4         # full attention 层用 (GQA)
head_dim: 256                  # full attention 层用

linear_num_key_heads: 16       # DeltaNet 层用
linear_key_head_dim: 128       # DeltaNet 层用
linear_num_value_heads: 32     # DeltaNet 层用
linear_value_head_dim: 128     # DeltaNet 层用
linear_conv_kernel_dim: 4      # DeltaNet 层的 causal conv1d

layer_types: [lin, lin, lin, full, lin, lin, lin, full, ...] × 8 组
→ 24 层 DeltaNet + 8 层 Full Attention
```

---

## 三类目标线性层（你们 kernel 要覆盖的）

### 类型 A：FFN Projection（所有 32 层都有）
- gate_proj: [9216, 2560]
- up_proj:   [9216, 2560]
- down_proj:  [2560, 9216]
- 共 32×3 = 96 个层
- 这是权重占比最大的部分

### 类型 B：Full Attention Projection（8 层：第 3,7,11,15,19,23,27,31 层）
- q_proj: [num_heads × head_dim, hidden_size] = [16×256, 2560] = [4096, 2560]
- k_proj: [num_kv_heads × head_dim, hidden_size] = [4×256, 2560] = [1024, 2560]
- v_proj: [1024, 2560]
- o_proj: [2560, 4096]
- 共 8×4 = 32 个层

### 类型 C：DeltaNet Projection（24 层：其余所有层）
- q_proj: [linear_num_key_heads × linear_key_head_dim, hidden_size] = [16×128, 2560] = [2048, 2560]
- k_proj: [16×128, 2560] = [2048, 2560]
- v_proj: [linear_num_value_heads × linear_value_head_dim, hidden_size] = [32×128, 2560] = [4096, 2560]
- o_proj: [2560, 4096]
- 还有 beta_proj、output_gate 等 DeltaNet 特有的投影
- 共 24×(至少4个) = 96+ 个层

> **注意：以上 shape 是根据 config 推算的，Task 2 会打印真实 shape 来验证。**

---

## 实验核心矩阵

对每种层类型 × 每种实现方式，测 decode 场景（M=1）的 latency：

| 层类型 | shape [N, K] | FP16 matmul | cuBLAS INT8 | 手写 INT8 GEMV (naive) | 手写 INT8 GEMV (opt) |
|--------|-------------|-------------|-------------|----------------------|---------------------|
| FFN gate_proj | [9216, 2560] | ? ms | ? ms | ? ms | ? ms |
| FFN down_proj | [2560, 9216] | ? ms | ? ms | ? ms | ? ms |
| FullAttn q_proj | [4096, 2560] | ? ms | ? ms | ? ms | ? ms |
| FullAttn k_proj | [1024, 2560] | ? ms | ? ms | ? ms | ? ms |
| DeltaNet q_proj | [2048, 2560] | ? ms | ? ms | ? ms | ? ms |
| DeltaNet v_proj | [4096, 2560] | ? ms | ? ms | ? ms | ? ms |

这张表就是你们报告的核心贡献。

---

## 环境要求

- GPU: RTX 4090 (24GB VRAM, 1 TB/s bandwidth, 82.6 TFLOPS FP16)
- Python 3.10+
- PyTorch 2.x with CUDA 12.x
- transformers: `pip install git+https://github.com/huggingface/transformers.git@main`
- accelerate
- 模型: `Qwen/Qwen3.5-4B`（已下载）

---

## 分工方案

### 你：负责 Full Attention 层 + FFN 层的 kernel
- 写 INT8 GEMV kernel (naive + optimized)
- 测 FFN 和 Full Attention projection 的 correctness + benchmark
- 做这些层的 profiling

### 你室友：负责 DeltaNet 层的 kernel
- 复用你的 kernel 代码（shape 不同但计算逻辑一样）
- 重点测 DeltaNet projection 的 correctness + benchmark
- 分析 DeltaNet 层 vs Full Attention 层的性能差异
- 做 DeltaNet 层的 profiling

### 一起做
- 公共前置（Task 1-5）
- cuBLAS INT8 baseline 接入
- 端到端集成
- 最终对比表格和报告

---

# 阶段一：公共前置（3-5 天）

## Task 1: 环境验证 + 模型加载

**文件：** `01_load_model.ipynb`

```python
# Cell 1: 安装
# pip install git+https://github.com/huggingface/transformers.git@main
# pip install accelerate

# Cell 2: 加载模型
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen3.5-4B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Cell 3: 验证推理
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Cell 4: 打印模型结构概览
print(model)
```

**产出：** 模型能加载、能推理。

---

## Task 2: 层结构分析 — 分类三种层

**文件：** `02_layer_analysis.ipynb`

这是最关键的分析步骤。你需要把模型里所有线性层按三类分好。

```python
import pandas as pd

# 从 config 读层类型
layer_types_config = model.config.text_config.layer_types
# ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention', ...]

records = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        # 解析层索引
        parts = name.split(".")
        layer_idx = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
                break
        
        # 判断层类型
        if layer_idx is not None and layer_idx < len(layer_types_config):
            attn_type = layer_types_config[layer_idx]
        else:
            attn_type = "other"  # vision encoder 等
        
        # 判断子模块类型
        if "mlp" in name:
            sub_type = "FFN"
        elif "self_attn" in name or "temporal_block" in name:
            if attn_type == "linear_attention":
                sub_type = "DeltaNet_Attn"
            elif attn_type == "full_attention":
                sub_type = "FullAttn"
            else:
                sub_type = "Other_Attn"
        else:
            sub_type = "Other"
        
        # 提取具体投影名
        proj_name = parts[-1] if parts else name  # e.g., q_proj, gate_proj, etc.
        
        records.append({
            "name": name,
            "layer_idx": layer_idx,
            "attn_type": attn_type,
            "sub_type": sub_type,
            "proj_name": proj_name,
            "out_features": module.out_features,
            "in_features": module.in_features,
            "shape": f"[{module.out_features}, {module.in_features}]",
            "params_M": module.out_features * module.in_features / 1e6
        })

df = pd.DataFrame(records)

# 打印汇总
print("=== 按类型统计 ===")
print(df.groupby(["sub_type", "proj_name", "shape"]).agg(
    count=("name", "count"),
    total_params_M=("params_M", "sum")
).to_string())

# 打印每种类型的代表层
print("\n=== FFN 代表层 ===")
print(df[df["sub_type"] == "FFN"].head(6).to_string())

print("\n=== Full Attention 代表层 ===")
print(df[df["sub_type"] == "FullAttn"].head(8).to_string())

print("\n=== DeltaNet 代表层 ===")
print(df[df["sub_type"] == "DeltaNet_Attn"].head(8).to_string())

# 保存完整列表
df.to_csv("layer_list.csv", index=False)
print(f"\n总共 {len(df)} 个线性层，已保存到 layer_list.csv")
```

**产出：** `layer_list.csv` — 所有层的完整清单，按 FFN / FullAttn / DeltaNet 分类。

---

## Task 3: FP16 Baseline

**文件：** `03_baseline.ipynb`

### 3a: 单层 benchmark（核心）

```python
import torch
import time

def benchmark_matmul_fp16(N, K, M=1, n_warmup=20, n_runs=200, device="cuda"):
    """
    测 FP16 矩阵乘 latency
    模拟 decode: x [M, K] @ W.T [K, N] → y [M, N]
    """
    W = torch.randn(N, K, dtype=torch.float16, device=device)
    x = torch.randn(M, K, dtype=torch.float16, device=device)
    
    # warmup
    for _ in range(n_warmup):
        _ = x @ W.T
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(n_runs):
        _ = x @ W.T
    end.record()
    torch.cuda.synchronize()
    
    avg_ms = start.elapsed_time(end) / n_runs
    
    # 计算理论 memory traffic（读 W + 读 x + 写 y）
    bytes_read = N * K * 2 + M * K * 2  # FP16 = 2 bytes
    bytes_write = M * N * 2
    total_bytes = bytes_read + bytes_write
    bandwidth_GBs = total_bytes / (avg_ms / 1000) / 1e9
    
    return {
        "M": M, "N": N, "K": K,
        "latency_ms": avg_ms,
        "bandwidth_GBs": bandwidth_GBs,
        "total_bytes_MB": total_bytes / 1e6
    }

# 测所有代表 shape
shapes_to_test = [
    # (名称, N, K)
    ("FFN_gate_proj",     9216, 2560),
    ("FFN_down_proj",     2560, 9216),
    ("FullAttn_q_proj",   4096, 2560),
    ("FullAttn_k_proj",   1024, 2560),
    ("DeltaNet_q_proj",   2048, 2560),
    ("DeltaNet_v_proj",   4096, 2560),
]

print("=== FP16 Baseline (M=1, decode) ===")
results = []
for name, N, K in shapes_to_test:
    r = benchmark_matmul_fp16(N, K, M=1)
    r["layer_type"] = name
    results.append(r)
    print(f"{name:25s} [{N:5d}, {K:5d}] → {r['latency_ms']:.4f} ms, "
          f"BW={r['bandwidth_GBs']:.1f} GB/s, data={r['total_bytes_MB']:.2f} MB")

# 保存
import pandas as pd
pd.DataFrame(results).to_csv("baseline_fp16.csv", index=False)
```

### 3b: 端到端 baseline（辅助，有时间再做）

```python
def measure_generate_latency(model, tokenizer, prompt, max_new_tokens=64, n_runs=3):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1)
    
    # prefill (generate 1 token)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) / n_runs * 1000
    
    # full generation
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) / n_runs * 1000
    
    decode_ms_per_token = (total_ms - prefill_ms) / max_new_tokens
    
    return {
        "prefill_ms": prefill_ms,
        "total_ms": total_ms,
        "decode_ms_per_token": decode_ms_per_token,
    }
```

### 3c: VRAM 记录

```python
torch.cuda.reset_peak_memory_stats()
# ... 跑一次推理 ...
peak_mb = torch.cuda.max_memory_allocated() / 1024**2
print(f"Peak VRAM: {peak_mb:.1f} MB")

# 权重占用估算
total_params = sum(p.numel() for p in model.parameters())
fp16_size_mb = total_params * 2 / 1024**2
int8_size_mb = total_params * 1 / 1024**2  # 大约，scale 忽略不计
print(f"FP16 weights: {fp16_size_mb:.1f} MB")
print(f"INT8 weights: {int8_size_mb:.1f} MB (saves {fp16_size_mb - int8_size_mb:.1f} MB)")
```

**产出：** `baseline_fp16.csv` — 每种层类型的 FP16 单层 latency。

---

## Task 4: 量化脚本

**文件：** `quantize.py`

和之前版本一样，但增加按层类型分组量化和误差分析：

```python
import torch
import pandas as pd

def quantize_symmetric_int8(weight_fp16):
    """Per-output-channel 对称 INT8 量化"""
    w = weight_fp16.float()
    scale = w.abs().amax(dim=1) / 127.0
    scale = scale.clamp(min=1e-8)
    qw = (w / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return qw, scale.half()

def dequantize_int8(qweight, scale):
    return qweight.float() * scale.float().unsqueeze(1)

def quantize_error(weight_fp16, qweight, scale):
    w_approx = dequantize_int8(qweight, scale)
    w = weight_fp16.float()
    abs_err = (w - w_approx).abs()
    return {
        "max_abs_err": abs_err.max().item(),
        "mean_abs_err": abs_err.mean().item(),
        "rel_err": (abs_err / (w.abs() + 1e-8)).mean().item()
    }

def quantize_all_target_layers(model, layer_list_csv="layer_list.csv"):
    """
    读取 layer_list.csv，对所有 FFN/FullAttn/DeltaNet 层量化
    返回 dict: { layer_name: {"qweight", "scale", "errors", "shape", "sub_type"} }
    """
    df = pd.read_csv(layer_list_csv)
    # 只量化这三类
    target_df = df[df["sub_type"].isin(["FFN", "FullAttn", "DeltaNet_Attn"])]
    
    name_to_module = {n: m for n, m in model.named_modules()}
    results = {}
    
    for _, row in target_df.iterrows():
        name = row["name"]
        if name not in name_to_module:
            continue
        module = name_to_module[name]
        if not isinstance(module, torch.nn.Linear):
            continue
        
        w = module.weight.data
        qw, s = quantize_symmetric_int8(w)
        errs = quantize_error(w, qw, s)
        
        results[name] = {
            "qweight": qw.cpu(),
            "scale": s.cpu(),
            "errors": errs,
            "shape": list(w.shape),
            "sub_type": row["sub_type"],
            "proj_name": row["proj_name"]
        }
        print(f"[{row['sub_type']:15s}] {name:60s} {str(list(w.shape)):20s} "
              f"max_err={errs['max_abs_err']:.6f} rel_err={errs['rel_err']:.6f}")
    
    return results

def save_quantized(results, path="quantized_weights.pt"):
    torch.save(results, path)
    print(f"Saved {len(results)} layers to {path}")

def load_quantized(path="quantized_weights.pt"):
    return torch.load(path, weights_only=False)

# === 量化误差按层类型汇总 ===
def summarize_errors_by_type(results):
    """按 sub_type 汇总量化误差"""
    from collections import defaultdict
    by_type = defaultdict(list)
    for name, info in results.items():
        by_type[info["sub_type"]].append(info["errors"])
    
    print("\n=== 量化误差按层类型汇总 ===")
    for stype, errs_list in by_type.items():
        avg_max = sum(e["max_abs_err"] for e in errs_list) / len(errs_list)
        avg_mean = sum(e["mean_abs_err"] for e in errs_list) / len(errs_list)
        avg_rel = sum(e["rel_err"] for e in errs_list) / len(errs_list)
        print(f"{stype:15s}: n={len(errs_list):3d}, avg_max_err={avg_max:.6f}, "
              f"avg_mean_err={avg_mean:.6f}, avg_rel_err={avg_rel:.6f}")
```

**产出：** `quantize.py` + `quantized_weights.pt` + 按层类型的误差分析。

---

## Task 5: CPU Reference

**文件：** `cpu_reference.py`

```python
import torch

def quantized_matmul_ref(x, qweight, scale):
    """
    x:       [M, K] float
    qweight: [N, K] int8
    scale:   [N]    float
    return:  [M, N] float32
    """
    w_approx = qweight.float() * scale.float().unsqueeze(1)
    return x.float() @ w_approx.T

def quantized_gemv_ref(x, qweight, scale):
    """decode 专用：x 是 [K] 或 [1, K]"""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return quantized_matmul_ref(x, qweight, scale).squeeze(0)

def check_correctness(y_ref, y_test, tag=""):
    yr = y_ref.float().flatten()
    yt = y_test.float().flatten()
    max_err = (yr - yt).abs().max().item()
    mean_err = (yr - yt).abs().mean().item()
    rel_err = ((yr - yt).abs() / (yr.abs() + 1e-8)).mean().item()
    cos = torch.nn.functional.cosine_similarity(yr.unsqueeze(0), yt.unsqueeze(0)).item()
    print(f"[{tag}] max={max_err:.6f} mean={mean_err:.6f} rel={rel_err:.6f} cos={cos:.8f}")
    return {"max_abs_err": max_err, "mean_abs_err": mean_err, "rel_err": rel_err, "cos_sim": cos}

def self_test():
    from quantize import quantize_symmetric_int8
    torch.manual_seed(42)
    
    # 测三种代表 shape
    test_shapes = [
        ("FFN_gate",     9216, 2560),
        ("FullAttn_q",   4096, 2560),
        ("DeltaNet_q",   2048, 2560),
    ]
    for tag, N, K in test_shapes:
        W = torch.randn(N, K, dtype=torch.float16)
        x = torch.randn(1, K, dtype=torch.float16)
        
        y_fp16 = x.float() @ W.float().T
        qw, s = quantize_symmetric_int8(W)
        y_quant = quantized_matmul_ref(x, qw, s)
        check_correctness(y_fp16, y_quant, tag=tag)
    
    print("Self test PASSED")

if __name__ == "__main__":
    self_test()
```

**产出：** `cpu_reference.py`

---

# 阶段二：CUDA Kernel 开发（5-7 天）

## Task 6: 手写 INT8 GEMV Kernel — Naive 版

**文件：** `kernels/int8_gemv.cu` + `kernels/setup.py`

### Kernel 设计

```
目标：计算 y[n] = scale[n] * Σ_k( x[k] * qweight[n, k] )

=== Naive 版 ===
- grid: (ceil(N / BLOCK_SIZE),)
- block: (BLOCK_SIZE,)   例如 256
- 每个 thread 负责一个 y[n]
- 对 K 循环，float32 累加
- 最后乘 scale[n]，写回 FP16

伪代码：
__global__ void int8_gemv_naive(
    const half* x,           // [K]
    const int8_t* qweight,   // [N, K] row-major
    const half* scale,       // [N]
    half* y,                 // [N]
    int N, int K
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        float xk = __half2float(x[k]);
        float wk = (float)qweight[n * K + k];
        acc += xk * wk;
    }
    acc *= __half2float(scale[n]);
    y[n] = __float2half(acc);
}
```

### Python 封装

```python
# kernels/setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='int8_gemv_cuda',
    ext_modules=[
        CUDAExtension('int8_gemv_cuda', ['int8_gemv.cu'])
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

### Correctness 验证（必须通过才能继续）

```python
from cpu_reference import quantized_gemv_ref, check_correctness
from quantize import load_quantized
import int8_gemv_cuda

weights = load_quantized("quantized_weights.pt")

# 对三种层类型各测一个代表
test_layers = [
    "model.model.layers.0.mlp.gate_proj",        # FFN
    "model.model.layers.3.self_attn.q_proj",      # FullAttn
    "model.model.layers.0.self_attn.q_proj",      # DeltaNet (layer 0 是 linear_attention)
]

for layer_name in test_layers:
    info = weights[layer_name]
    qw = info["qweight"].cuda()
    s = info["scale"].cuda()
    K = info["shape"][1]
    
    x = torch.randn(K, dtype=torch.float16, device="cuda")
    
    y_ref = quantized_gemv_ref(x.cpu(), qw.cpu(), s.cpu())
    y_cuda = int8_gemv_cuda.forward(x, qw, s)
    
    check_correctness(y_ref, y_cuda.cpu(), tag=f"{info['sub_type']}:{layer_name.split('.')[-1]}")
```

**产出：** 能编译、能跑、三种层类型 correctness 全通过的 naive kernel。

---

## Task 7: 手写 INT8 GEMV Kernel — Optimized 版

**文件：** `kernels/int8_gemv_opt.cu`

### 优化策略

```
=== Optimized 版 ===

优化 1: Vectorized INT8 Load
- 一次读 4 个 int8（用 int32 或 char4），减少 memory transaction
- 4090 的 L2 cache line 是 128 bytes，vectorized load 能更好利用

优化 2: Warp-level Parallel Reduction
- 一个 warp (32 threads) 协作算一个 y[n]
- 每个 thread 负责 K/32 个元素
- 用 __shfl_down_sync 做 warp reduce
- 好处：减少对 K 的串行循环

优化 3: Shared Memory for x
- 把 x[K] 加载到 shared memory
- 所有 thread 共享读 x，避免重复从 global memory 读

优化 4: Multiple output channels per warp
- 一个 block 内多个 warp，每个 warp 算一个不同的 y[n]
- 提高 occupancy 和 SM 利用率

伪代码：
__global__ void int8_gemv_opt(
    const half* x,
    const int8_t* qweight,
    const half* scale,
    half* y,
    int N, int K
) {
    extern __shared__ float sx[];  // shared memory for x
    
    // 协作加载 x 到 shared memory
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        sx[i] = __half2float(x[i]);
    }
    __syncthreads();
    
    // 每个 warp 算一个 output channel
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int n = blockIdx.x * (blockDim.x / 32) + warp_id;
    if (n >= N) return;
    
    // 每个 lane 算 K/32 个元素
    float acc = 0.0f;
    const int8_t* row = qweight + n * K;
    for (int k = lane_id; k < K; k += 32) {
        acc += sx[k] * (float)row[k];
    }
    
    // warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    
    if (lane_id == 0) {
        y[n] = __float2half(acc * __half2float(scale[n]));
    }
}
```

**产出：** 优化版 kernel + correctness 通过 + 与 naive 版对比数据。

---

## Task 8: cuBLAS INT8 Baseline

**文件：** `cublas_int8_baseline.py`

这是对照组，用 PyTorch 的 `torch._int_mm` 或直接调 cuBLAS：

```python
def cublas_int8_matmul(x_fp16, qweight_int8, scale):
    """
    用 PyTorch 内置的 INT8 matmul 作为 cuBLAS baseline
    
    torch._int_mm 要求两个输入都是 int8
    所以需要把 x 也量化（动态量化），或者用 mixed precision 方式
    
    更简单的做法：先 dequant 再用 FP16 matmul（模拟 cuBLAS 能达到的效果）
    """
    # 方法 A: dequant + FP16 matmul (作为 upper bound reference)
    w_approx = (qweight_int8.float() * scale.float().unsqueeze(1)).half()
    return x_fp16 @ w_approx.T
    
    # 方法 B: 如果 PyTorch 支持 torch._int_mm
    # x_int8 = x_fp16.to(torch.int8)  # 需要先量化 x
    # y_int32 = torch._int_mm(x_int8, qweight_int8.T)
    # y = y_int32.float() * scale_x * scale_w
```

> 注意：cuBLAS 的 INT8 GEMM 在 M=1 时未必有优势，因为 cuBLAS 是为大矩阵优化的。
> 这正是你们手写 GEMV kernel 可能胜出的地方。

**产出：** cuBLAS INT8 baseline latency 数据。

---

## Task 9: Benchmark — 三种层类型 × 四种实现

**文件：** `04_benchmark.ipynb`

```python
import pandas as pd

def benchmark_all(shapes_dict, qweights, n_warmup=20, n_runs=200):
    """
    shapes_dict: {"FFN_gate": (9216, 2560), "DeltaNet_q": (2048, 2560), ...}
    返回完整 benchmark 表
    """
    results = []
    
    for name, (N, K) in shapes_dict.items():
        x = torch.randn(1, K, dtype=torch.float16, device="cuda")
        W_fp16 = torch.randn(N, K, dtype=torch.float16, device="cuda")
        qw = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda")
        scale = torch.randn(N, dtype=torch.float16, device="cuda").abs()
        
        # (1) FP16 matmul
        fp16_ms = timed_run(lambda: x @ W_fp16.T, n_warmup, n_runs)
        
        # (2) cuBLAS INT8 (dequant + fp16)
        w_deq = (qw.float() * scale.float().unsqueeze(1)).half()
        cublas_ms = timed_run(lambda: x @ w_deq.T, n_warmup, n_runs)
        
        # (3) 手写 naive
        naive_ms = timed_run(lambda: int8_gemv_cuda.forward_naive(x.squeeze(), qw, scale), n_warmup, n_runs)
        
        # (4) 手写 opt
        opt_ms = timed_run(lambda: int8_gemv_cuda.forward_opt(x.squeeze(), qw, scale), n_warmup, n_runs)
        
        # Memory traffic 计算 (INT8 版)
        bytes_int8 = N * K * 1 + K * 2 + N * 2 + N * 2  # qw + x + scale + y
        bytes_fp16 = N * K * 2 + K * 2 + N * 2           # W + x + y
        
        results.append({
            "layer_type": name,
            "N": N, "K": K,
            "fp16_ms": fp16_ms,
            "cublas_int8_ms": cublas_ms,
            "naive_int8_ms": naive_ms,
            "opt_int8_ms": opt_ms,
            "speedup_vs_fp16": fp16_ms / opt_ms if opt_ms > 0 else 0,
            "mem_traffic_fp16_MB": bytes_fp16 / 1e6,
            "mem_traffic_int8_MB": bytes_int8 / 1e6,
            "mem_saving_pct": (1 - bytes_int8 / bytes_fp16) * 100
        })
    
    return pd.DataFrame(results)

def timed_run(fn, n_warmup, n_runs):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_runs):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_runs
```

**产出：** 完整 benchmark 表，这就是报告的核心 Table。

---

## Task 10: Profiling

**文件：** `run_profile.py`（给 Nsight Compute 用的脚本）

```bash
# 对三种代表 shape 各跑一次 profiling
ncu --set full --target-processes all \
    -o profile_ffn_gate \
    python run_profile.py --layer FFN_gate --shape 9216 2560

ncu --set full --target-processes all \
    -o profile_fullattn_q \
    python run_profile.py --layer FullAttn_q --shape 4096 2560

ncu --set full --target-processes all \
    -o profile_deltanet_q \
    python run_profile.py --layer DeltaNet_q --shape 2048 2560
```

关注的指标：
- `dram__bytes_read.sum` — 实际内存读取
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` — 带宽利用率
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` — 计算利用率
- `launch__occupancy` — occupancy

**产出：** Nsight Compute 报告 + 关键指标表格。

---

# 阶段三：集成 + 端到端（2-3 天）

## Task 11: PyTorch Module 封装 + 模型层替换

**文件：** `quant_linear.py` + `05_integration.ipynb`

```python
class QuantLinearINT8(torch.nn.Module):
    def __init__(self, qweight, scale, bias=None):
        super().__init__()
        self.register_buffer("qweight", qweight)  # [N, K] int8
        self.register_buffer("scale", scale)        # [N] fp16
        self.bias = bias
    
    def forward(self, x):
        # x: [..., K]
        orig_shape = x.shape
        x_2d = x.view(-1, x.shape[-1])  # [M, K]
        
        if x_2d.shape[0] == 1:
            # decode path: 用手写 GEMV kernel
            y = int8_gemv_cuda.forward_opt(x_2d.squeeze(0), self.qweight, self.scale)
            y = y.unsqueeze(0)
        else:
            # prefill path: dequant + matmul (用 cuBLAS)
            w = (self.qweight.float() * self.scale.float().unsqueeze(1)).half()
            y = x_2d.half() @ w.T
        
        if self.bias is not None:
            y = y + self.bias
        
        return y.view(*orig_shape[:-1], -1)
```

替换逻辑：
```python
def replace_linear_layers(model, quantized_weights):
    """替换模型中的目标线性层"""
    replaced = 0
    for name, info in quantized_weights.items():
        parts = name.split(".")
        # 导航到父模块
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr_name = parts[-1]
        
        old_module = getattr(parent, attr_name)
        new_module = QuantLinearINT8(
            info["qweight"].cuda(),
            info["scale"].cuda(),
            old_module.bias
        )
        setattr(parent, attr_name, new_module)
        replaced += 1
    
    print(f"Replaced {replaced} layers")
```

端到端验证：
```python
# 替换前
outputs_before = model.generate(**inputs, max_new_tokens=32)

# 替换
replace_linear_layers(model, quantized_weights)

# 替换后
outputs_after = model.generate(**inputs, max_new_tokens=32)

# 对比输出是否合理（不需要完全相同，但不能乱码）
print("Before:", tokenizer.decode(outputs_before[0], skip_special_tokens=True))
print("After:", tokenizer.decode(outputs_after[0], skip_special_tokens=True))
```

**产出：** 端到端可运行 + latency/VRAM 对比。

---

# 阶段四：结果整理（2-3 天）

## Task 12: 最终报告素材

### 需要的表格

1. **模型架构表**：三种层类型的数量、shape、参数量
2. **量化误差表（按层类型）**：FFN vs FullAttn vs DeltaNet 的误差对比
3. **核心 benchmark 表**：三种层类型 × 四种实现的 latency
4. **Memory traffic 分析表**：FP16 vs INT8 的数据搬运量
5. **VRAM 节省表**：FP16 权重 vs INT8 权重
6. **Profiling 指标表**：bandwidth utilization, occupancy 等

### 需要的图

1. **Latency 对比柱状图**：按层类型分组
2. **Roofline 图**：标出各层类型在 FP16 和 INT8 下的位置
3. **带宽利用率对比图**：不同 shape 下的 bandwidth utilization

---

## 文件结构

```
project/
├── 01_load_model.ipynb
├── 02_layer_analysis.ipynb
├── 03_baseline.ipynb
├── 04_benchmark.ipynb
├── 05_integration.ipynb
├── 06_results.ipynb
├── quantize.py
├── cpu_reference.py
├── cublas_int8_baseline.py
├── quant_linear.py
├── run_profile.py
├── kernels/
│   ├── int8_gemv.cu            # naive + optimized kernel
│   └── setup.py
├── layer_list.csv
├── baseline_fp16.csv
├── quantized_weights.pt
└── README.md
```

---

## 给 Codex 的执行顺序

```
Task 1 → Task 2 → Task 3 → Task 4 → Task 5
（公共前置，顺序执行）

然后并行：
  你：Task 6 → Task 7 → Task 9(FFN + FullAttn 部分) → Task 10
  室友：复用 Task 6/7 的 kernel → Task 9(DeltaNet 部分) → Task 10

最后合作：
  Task 8 → Task 11 → Task 12
```

每个 Task 交给 Codex 时，附上：
- 前置 Task 的产出文件
- 这个 Task 的完整描述
- 预期产出

---

## 你们报告的核心叙事

"Qwen3.5-4B 采用了 Gated DeltaNet + Full Attention 的混合架构，75% 的层是 DeltaNet。
我们发现这三种线性层（FFN / Full Attention / DeltaNet）在 decode 阶段（M=1）都是 memory-bound 的，
但由于 shape 不同，它们的量化收益也不同。
我们实现了 INT8 weight-only CUDA GEMV kernel，对比了手写 kernel、cuBLAS、和 FP16 baseline，
证明了 INT8 量化在 decode 阶段可以有效降低 memory traffic 并提升 throughput，
并且首次系统分析了 DeltaNet 层在量化下的性能特征。"

这句话就是你们报告的 abstract 和面试时的一句话总结。
