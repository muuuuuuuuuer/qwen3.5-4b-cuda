# Qwen3.5-4B INT8 Weight-Only Quantization Project — 统筹计划

## 项目目标

在 4090 (24GB) 上，对 Qwen3.5-4B 的 decode 阶段线性层实现 INT8 weight-only 量化 CUDA kernel，验证 correctness，做 profiling，证明 memory-bound 场景下量化可以降低 latency。

---

## 环境要求

- GPU: RTX 4090 (24GB)
- Python 3.10+
- PyTorch 2.x with CUDA
- transformers: 从 main 分支安装 (`pip install git+https://github.com/huggingface/transformers.git@main`)
- accelerate
- 模型: `Qwen/Qwen3.5-4B` (已下载)

---

## 阶段一：公共前置（两人共同，3-5天）

### Task 1: 环境验证 + 模型加载

**文件：** `01_load_model.ipynb`

**做什么：**

1. 安装依赖：
```bash
pip install git+https://github.com/huggingface/transformers.git@main
pip install accelerate
```

2. 加载模型：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-4B",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
```

3. 验证模型能做一次 forward：
```python
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**产出：** 确认模型能正常加载和推理。

---

### Task 2: 打印层结构 + 生成层列表

**文件：** `02_layer_analysis.ipynb`

**做什么：**

1. 打印所有线性层的 name、in_features、out_features：
```python
layer_info = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        layer_info.append({
            "name": name,
            "out_features": module.out_features,
            "in_features": module.in_features,
            "params": module.out_features * module.in_features
        })
        print(f"{name}: [{module.out_features}, {module.in_features}]")
```

2. 根据 config.json 中的 `layer_types` 标注每层类型：
   - 第 0,1,2,4,5,6,8,9,10,...层 → `linear_attention` (DeltaNet)
   - 第 3,7,11,15,19,23,27,31层 → `full_attention`
   - 所有层都有 FFN (gate_proj, up_proj, down_proj)

3. 生成一张汇总表，格式如下：

| 层索引 | 层类型 | 子模块名 | shape [N, K] | 参数量 | 优先级 |
|--------|--------|----------|-------------|--------|--------|
| 0 | DeltaNet | model.layers.0.mlp.gate_proj | [9216, 2560] | 23.6M | P0-优先 |
| 0 | DeltaNet | model.layers.0.mlp.up_proj | [9216, 2560] | 23.6M | P0-优先 |
| 0 | DeltaNet | model.layers.0.mlp.down_proj | [2560, 9216] | 23.6M | P0-优先 |
| ... | ... | ... | ... | ... | ... |

4. 优先级标记规则：
   - **P0-优先**：所有层的 FFN projection (gate_proj, up_proj, down_proj) → shape 统一，数量最多
   - **P1-次优**：full_attention 层的 q_proj, k_proj, v_proj, o_proj
   - **P2-暂缓**：DeltaNet 层的 attention 相关 projection
   - **不碰**：vision encoder 内部层、embedding、layernorm

5. 把层列表保存为 `layer_list.csv`

**产出：** `layer_list.csv`，所有目标线性层的完整清单。

---

### Task 3: FP16 Baseline 测试

**文件：** `03_baseline.ipynb`

**做什么：**

1. 准备固定输入：
```python
test_prompts = {
    "short_128": "Explain quantum computing in simple terms.",        # ~128 tokens
    "mid_512": "Write a detailed essay about climate change...",      # ~512 tokens  
    "long_2048": "..." # 一段很长的文本，确保 tokenize 后约 2048 tokens
}
```

2. 测 prefill latency（只生成 1 个 token）：
```python
import torch
import time

def measure_prefill(model, input_ids, n_warmup=3, n_runs=10):
    # warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model.generate(input_ids=input_ids, max_new_tokens=1)
    
    torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model.generate(input_ids=input_ids, max_new_tokens=1)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    return sum(times) / len(times)
```

3. 测 decode latency（生成 128 个 token，减去 prefill）：
```python
def measure_decode(model, input_ids, gen_tokens=128, n_warmup=3, n_runs=5):
    prefill_time = measure_prefill(model, input_ids, n_warmup, n_runs)
    
    # warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model.generate(input_ids=input_ids, max_new_tokens=gen_tokens)
    
    torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model.generate(input_ids=input_ids, max_new_tokens=gen_tokens)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    
    avg_total = sum(times) / len(times)
    decode_time = avg_total - prefill_time
    per_token = decode_time / gen_tokens
    return per_token
```

4. 测 peak VRAM：
```python
torch.cuda.reset_peak_memory_stats()
# ... 跑推理 ...
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
```

5. **单层 baseline**（这个更重要）：
```python
def benchmark_single_layer_fp16(weight, x, n_warmup=10, n_runs=100):
    """
    weight: [N, K] fp16, 某一层的权重
    x: [M, K] fp16, 输入 activation
    """
    # warmup
    for _ in range(n_warmup):
        _ = x @ weight.T
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(n_runs):
        y = x @ weight.T
    end_event.record()
    torch.cuda.synchronize()
    
    avg_ms = start_event.elapsed_time(end_event) / n_runs
    return avg_ms
```

测试不同的 M 值（decode 场景 M=1，prefill 场景 M=128/512/2048），固定用一个 FFN 层的真实权重（比如 `model.model.layers[0].mlp.gate_proj.weight`）。

6. 记录成表格：

| 场景 | M | N | K | FP16 latency (ms) | 
|------|---|---|---|-------------------|
| decode | 1 | 9216 | 2560 | ? |
| decode | 1 | 2560 | 9216 | ? |
| prefill | 128 | 9216 | 2560 | ? |
| prefill | 512 | 9216 | 2560 | ? |
| prefill | 2048 | 9216 | 2560 | ? |

**产出：** `baseline_results.csv`，FP16 单层 latency 数据 + 端到端 latency 数据。

---

### Task 4: 量化脚本

**文件：** `quantize.py`

**做什么：**

实现 per-output-channel 对称 INT8 量化。

```python
import torch

def quantize_symmetric_int8(weight_fp16):
    """
    输入: weight_fp16 [N, K] float16
    输出: qweight [N, K] int8, scale [N] float16
    
    对称量化，无 zero-point
    每个 output channel 一个 scale
    scale = max(abs(row)) / 127
    """
    weight_float = weight_fp16.float()
    scale = weight_float.abs().amax(dim=1) / 127.0  # [N]
    scale = scale.clamp(min=1e-8)
    qweight = (weight_float / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return qweight, scale.half()


def dequantize_int8(qweight, scale):
    """
    还原近似权重: W_approx = qweight * scale
    输入: qweight [N, K] int8, scale [N] float16
    输出: weight_approx [N, K] float32
    """
    return qweight.float() * scale.float().unsqueeze(1)


def quantize_error_analysis(weight_fp16, qweight, scale):
    """
    计算量化误差
    """
    weight_approx = dequantize_int8(qweight, scale)
    weight_float = weight_fp16.float()
    
    abs_err = (weight_float - weight_approx).abs()
    max_abs_err = abs_err.max().item()
    mean_abs_err = abs_err.mean().item()
    rel_err = (abs_err / (weight_float.abs() + 1e-8)).mean().item()
    
    return {
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "mean_rel_err": rel_err
    }


def quantize_model_layers(model, target_layers):
    """
    对模型中指定的层进行量化
    
    target_layers: list of layer name strings (从 layer_list.csv 中读取 P0 层)
    
    返回: dict { layer_name: {"qweight": tensor, "scale": tensor, "errors": dict} }
    """
    results = {}
    for name, module in model.named_modules():
        if name in target_layers and isinstance(module, torch.nn.Linear):
            w = module.weight.data  # [N, K]
            qw, s = quantize_symmetric_int8(w)
            errs = quantize_error_analysis(w, qw, s)
            results[name] = {
                "qweight": qw.cpu(),
                "scale": s.cpu(),
                "errors": errs,
                "shape": list(w.shape)
            }
            print(f"{name}: shape={list(w.shape)}, max_err={errs['max_abs_err']:.6f}, "
                  f"mean_err={errs['mean_abs_err']:.6f}, rel_err={errs['mean_rel_err']:.6f}")
    return results


def save_quantized(results, save_path="quantized_weights.pt"):
    """保存量化后的权重"""
    torch.save(results, save_path)
    print(f"Saved {len(results)} quantized layers to {save_path}")


def load_quantized(save_path="quantized_weights.pt"):
    """加载量化后的权重"""
    return torch.load(save_path)
```

**验证步骤：**
1. 用随机 tensor 测试 quantize → dequantize 流程
2. 对模型真实权重量化，检查误差是否合理（INT8 对称量化 rel_err 通常 < 1%）
3. 保存量化权重

**产出：** `quantize.py` + `quantized_weights.pt`

---

### Task 5: CPU Reference

**文件：** `cpu_reference.py`

**做什么：**

```python
import torch

def quantized_matmul_reference(x, qweight, scale):
    """
    CPU 上的量化矩阵乘法参考实现
    
    x:       [M, K] float，输入 activation（decode 时 M=1）
    qweight: [N, K] int8，量化权重
    scale:   [N]    float，per-channel scale
    return:  [M, N] float32
    
    计算: Y = X @ (qweight * scale)^T
    等价于: Y[m, n] = sum_k( x[m,k] * qweight[n,k] * scale[n] )
    """
    # 方法：先 dequant 再乘（简单、保证正确）
    weight_approx = qweight.float() * scale.float().unsqueeze(1)  # [N, K]
    y = x.float() @ weight_approx.T  # [M, N]
    return y


def quantized_matvec_reference(x, qweight, scale):
    """
    专门针对 decode 场景 (M=1) 的 reference
    
    x:       [K] float
    qweight: [N, K] int8
    scale:   [N] float
    return:  [N] float32
    """
    # 等价于 quantized_matmul_reference 但输入是 1D
    weight_approx = qweight.float() * scale.float().unsqueeze(1)  # [N, K]
    y = weight_approx @ x.float()  # [N]
    return y


def check_correctness(y_reference, y_test, tag=""):
    """
    对比 reference 输出和待测输出
    """
    y_ref = y_reference.float()
    y_tst = y_test.float()
    
    max_abs_err = (y_ref - y_tst).abs().max().item()
    mean_abs_err = (y_ref - y_tst).abs().mean().item()
    rel_err = ((y_ref - y_tst).abs() / (y_ref.abs() + 1e-8)).mean().item()
    
    # cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        y_ref.flatten().unsqueeze(0),
        y_tst.flatten().unsqueeze(0)
    ).item()
    
    print(f"[{tag}] max_abs={max_abs_err:.6f}, mean_abs={mean_abs_err:.6f}, "
          f"rel_err={rel_err:.6f}, cos_sim={cos_sim:.8f}")
    
    return {
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "rel_err": rel_err,
        "cos_sim": cos_sim
    }


# ============================================
# 自测：验证 reference 本身是否正确
# ============================================
def self_test():
    """
    用 FP16 原始权重做对照：
    1. FP16 直接算: Y_fp16 = X @ W^T
    2. 量化再算: Y_quant = X @ (dequant(W))^T
    3. 比较 Y_fp16 vs Y_quant → 误差应该很小（来自量化本身）
    """
    from quantize import quantize_symmetric_int8
    
    torch.manual_seed(42)
    
    # 模拟 FFN gate_proj 的 shape: [9216, 2560]
    N, K = 9216, 2560
    W = torch.randn(N, K, dtype=torch.float16)
    
    # 模拟 decode 输入 (batch=1)
    x_decode = torch.randn(1, K, dtype=torch.float16)
    
    # FP16 ground truth
    y_fp16 = (x_decode.float() @ W.float().T)
    
    # 量化 + reference
    qw, s = quantize_symmetric_int8(W)
    y_quant = quantized_matmul_reference(x_decode, qw, s)
    
    print("=== Self Test: FP16 vs Quantized Reference ===")
    check_correctness(y_fp16, y_quant, tag="decode M=1")
    
    # 模拟 prefill 输入
    x_prefill = torch.randn(512, K, dtype=torch.float16)
    y_fp16_p = (x_prefill.float() @ W.float().T)
    y_quant_p = quantized_matmul_reference(x_prefill, qw, s)
    check_correctness(y_fp16_p, y_quant_p, tag="prefill M=512")
    
    print("=== Self Test PASSED ===")


if __name__ == "__main__":
    self_test()
```

**产出：** `cpu_reference.py`，后续所有 CUDA kernel 的 correctness 都用这个来验证。

---

## 阶段二：CUDA Kernel 开发（两人分头，5-7天）

> 以下两个 Task 你和你室友各负责一个。
> 两个人都做 decode 方向（M=1 的 GEMV-like），但可以分成：
> - **你：INT8 kernel 实现 + correctness**
> - **你室友：Profiling + Roofline + 性能分析**
> 
> 或者：
> - **你：INT8 kernel**
> - **你室友：INT4 kernel**
>
> 你们自己决定。

---

### Task 6: INT8 Decode CUDA Kernel — 第一版 (Naive)

**文件：** `kernels/int8_gemv_naive.cu` + `kernels/setup.py`

**做什么：**

1. 写一个最简单的 INT8 weight-only GEMV kernel：

```
核心计算：
  y[n] = sum_k( x[k] * qweight[n, k] * scale[n] )
       = scale[n] * sum_k( x[k] * qweight[n, k] )

第一版策略：
  - 每个 thread 负责一个输出元素 y[n]
  - 对 K 维度循环累加
  - 用 float32 做累加
  - 最后乘 scale[n]
  - 写回 y[n]

输入：
  x:       [K] float16
  qweight: [N, K] int8
  scale:   [N] float16
输出：
  y:       [N] float16 或 float32

grid: (ceil(N / blockDim.x),)
block: (256,) 或 (128,)
```

2. 用 PyTorch C++ extension 封装：
```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='int8_gemv',
    ext_modules=[
        CUDAExtension('int8_gemv', [
            'int8_gemv_naive.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

3. 写 Python wrapper：
```python
def int8_gemv_forward(x, qweight, scale):
    """
    x: [K] or [1, K] float16, on CUDA
    qweight: [N, K] int8, on CUDA
    scale: [N] float16, on CUDA
    returns: [N] or [1, N] float16
    """
    return int8_gemv.forward(x, qweight, scale)
```

4. **Correctness 验证**：
```python
from cpu_reference import quantized_matvec_reference, check_correctness

# 用真实层权重测试
layer_name = "model.model.layers.0.mlp.gate_proj"
qw = quantized_weights[layer_name]["qweight"].cuda()
s = quantized_weights[layer_name]["scale"].cuda()
x = torch.randn(2560, dtype=torch.float16, device="cuda")

y_ref = quantized_matvec_reference(x.cpu(), qw.cpu(), s.cpu())
y_cuda = int8_gemv_forward(x, qw, s)

check_correctness(y_ref, y_cuda.cpu(), tag="naive kernel vs CPU ref")
```

5. 测多组 shape 的 correctness：
   - [9216, 2560] — gate_proj / up_proj
   - [2560, 9216] — down_proj
   - [4096, 2560] — attention q_proj (如果有)
   - 非整齐 shape 如 [9216, 2560] 本身就不是 2 的幂

**产出：** 能编译、能跑、correctness 通过的第一版 naive kernel。

---

### Task 7: INT8 Decode CUDA Kernel — 第二版 (Optimized)

**文件：** `kernels/int8_gemv_opt.cu`

**做什么（在 naive 版基础上优化）：**

1. **Vectorized load**：用 `int4`（即 4 个 int8 打包成 32bit）一次读 4 个权重
2. **Warp-level reduction**：一个 warp (32 threads) 协作计算一个 y[n]，每个 thread 负责 K/32 个元素，最后做 warp shuffle reduce
3. **多个 output channel per block**：一个 block 算多个 y[n]，提高 occupancy
4. **Shared memory for x**：把 x 加载到 shared memory，所有 thread 共享读取

优化目标：
- 提高 global memory throughput（接近 4090 的理论带宽 ~1 TB/s）
- 提高 occupancy
- 减少 memory transaction 次数

**产出：** 优化版 kernel + correctness 验证 + 与 naive 版性能对比。

---

### Task 8: Benchmark + Profiling

**文件：** `04_benchmark.ipynb`

**做什么：**

1. 单层 latency 对比：
```python
# 对比三种实现：
# (a) FP16 原始 matmul: y = x @ W.T
# (b) INT8 naive kernel
# (c) INT8 optimized kernel

shapes = [
    ("gate_proj", 9216, 2560),
    ("down_proj", 2560, 9216),
]

for name, N, K in shapes:
    W_fp16 = torch.randn(N, K, dtype=torch.float16, device="cuda")
    x = torch.randn(1, K, dtype=torch.float16, device="cuda")
    qw, s = quantize_symmetric_int8(W_fp16)
    qw, s = qw.cuda(), s.cuda()
    
    # benchmark each...
```

2. 记录指标：
   - Latency (ms)
   - Memory bandwidth utilization (GB/s)
   - Speedup vs FP16

3. Nsight Compute profiling（命令行）：
```bash
ncu --set full -o profile_naive python run_kernel.py --mode naive
ncu --set full -o profile_opt python run_kernel.py --mode optimized
```

4. 关注指标：
   - `dram__bytes_read.sum` — 实际读了多少字节
   - `sm__throughput.avg.pct_of_peak_sustained_elapsed` — SM 利用率
   - `dram__throughput.avg.pct_of_peak_sustained_elapsed` — 带宽利用率
   - `launch__occupancy` — occupancy

5. 画 Roofline 图：
   - x 轴: arithmetic intensity (FLOPs / bytes)
   - y 轴: performance (GFLOPS)
   - 标出 FP16、INT8 naive、INT8 opt 各自的位置
   - 4090 的 memory bandwidth roof: ~1 TB/s
   - 4090 的 compute roof: ~82.6 TFLOPS FP16

**产出：** benchmark 表格 + profiling 截图 + roofline 图

---

## 阶段三：集成 + 端到端验证（合作，2-3天）

### Task 9: PyTorch Module 封装

**文件：** `quant_linear.py`

**做什么：**

```python
import torch
import torch.nn as nn

class QuantLinearDecode(nn.Module):
    """
    替换 nn.Linear 的量化版本（decode 专用，M=1）
    """
    def __init__(self, qweight, scale, bias=None):
        super().__init__()
        self.register_buffer("qweight", qweight)  # [N, K] int8
        self.register_buffer("scale", scale)        # [N] float16
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
    
    def forward(self, x):
        # x: [batch, K]
        # 调用你的 CUDA kernel
        y = int8_gemv_forward(x.squeeze(0), self.qweight, self.scale)
        if self.bias is not None:
            y = y + self.bias
        return y.unsqueeze(0) if x.dim() == 2 else y
```

---

### Task 10: 模型层替换 + 端到端测试

**文件：** `05_integration.ipynb`

**做什么：**

1. 先替换一层试试：
```python
# 替换第 0 层的 gate_proj
original_layer = model.model.layers[0].mlp.gate_proj
qw = quantized_weights["model.model.layers.0.mlp.gate_proj"]["qweight"].cuda()
s = quantized_weights["model.model.layers.0.mlp.gate_proj"]["scale"].cuda()
bias = original_layer.bias  # 可能是 None

model.model.layers[0].mlp.gate_proj = QuantLinearDecode(qw, s, bias)
```

2. 跑一次 generate，检查输出是否合理（不崩、不乱码）

3. 逐步替换更多层（所有 P0 层），每次替换后检查输出

4. 测替换后的端到端 decode latency，和 baseline 对比

5. 测替换后的 peak VRAM，和 baseline 对比

**产出：** 端到端 latency 对比表 + VRAM 对比

---

## 阶段四：结果整理（合作，2-3天）

### Task 11: 最终结果汇总

**文件：** `06_results.ipynb` 或 直接写到报告里

**需要的表格和图：**

1. **Baseline 表**: FP16 单层 latency，不同 M 值
2. **Quantization Error 表**: 每层的 max/mean/relative error
3. **Kernel 对比表**: FP16 vs INT8-naive vs INT8-opt, latency + speedup
4. **Bandwidth 分析表**: 理论 memory traffic vs 实际 throughput
5. **VRAM 对比表**: FP16 权重 vs INT8 权重，节省了多少
6. **Roofline 图**: 标出各实现的位置
7. **（可选）端到端 latency 对比**: 替换层前后

---

## 文件结构

```
project/
├── 01_load_model.ipynb          # Task 1: 环境验证
├── 02_layer_analysis.ipynb      # Task 2: 层结构分析
├── 03_baseline.ipynb            # Task 3: FP16 baseline
├── 04_benchmark.ipynb           # Task 8: benchmark + profiling
├── 05_integration.ipynb         # Task 10: 端到端集成
├── 06_results.ipynb             # Task 11: 结果汇总
├── quantize.py                  # Task 4: 量化脚本
├── cpu_reference.py             # Task 5: CPU reference
├── quant_linear.py              # Task 9: PyTorch module
├── kernels/
│   ├── int8_gemv_naive.cu       # Task 6: naive kernel
│   ├── int8_gemv_opt.cu         # Task 7: optimized kernel
│   └── setup.py                 # 编译脚本
├── layer_list.csv               # 层列表
├── quantized_weights.pt         # 量化后的权重
├── baseline_results.csv         # baseline 数据
└── README.md                    # 项目说明
```

---

## 分工建议（你和室友）

### 公共（一起做）
- Task 1-5：公共前置，3-5 天

### 分头开发（选一种）

**方案 A：按 kernel 精度分**
- 你：Task 6+7 (INT8 kernel)
- 室友：另写一套 INT4 kernel（类似结构，打包方式不同）
- 好处：可以对比 INT8 vs INT4 的精度-性能 tradeoff

**方案 B：按职责分**
- 你：Task 6+7 (kernel 实现 + correctness)
- 室友：Task 8 (profiling + roofline + 性能分析)
- 好处：并行度高，一个人写 kernel 另一个人马上能 profile

### 合作
- Task 9-11：集成 + 报告，2-3 天

---

## 给 Codex 的执行顺序

按以下顺序逐个执行：

1. 先跑 Task 1 → 确认环境 OK
2. 跑 Task 2 → 拿到层列表
3. 跑 Task 3 → 拿到 baseline 数据
4. 跑 Task 4 → 生成量化权重
5. 跑 Task 5 → 验证 CPU reference
6. 然后 Task 6-7 是 CUDA 开发，需要人工参与较多
7. Task 8-11 依赖 kernel 完成后执行

每个 Task 都是独立的，可以单独交给 Codex。
Task 之间的依赖关系：
- Task 2 依赖 Task 1（需要模型加载）
- Task 3 依赖 Task 1
- Task 4 依赖 Task 1
- Task 5 依赖 Task 4
- Task 6 依赖 Task 4 + Task 5
- Task 7 依赖 Task 6
- Task 8 依赖 Task 6
- Task 9 依赖 Task 6
- Task 10 依赖 Task 9
- Task 11 依赖所有
