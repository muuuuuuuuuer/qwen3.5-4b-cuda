# Qwen3.5-4B 边缘推理优化 - 完整项目日志

**项目：** ECE 9483 - 受限边缘设备上的多模态大模型推理优化
**成员：** Yaoqi Xia、Hao Zhong
**目标：** 在受限的类边缘部署环境中，优化 Qwen3.5-4B 在 batch-1 单用户服务场景下的 decode 延迟
**硬件：** RTX 4090 Laptop（24 GB VRAM，576 GB/s 内存带宽）
**模型：** Qwen/Qwen3.5-4B（32 层：24 层 DeltaNet、8 层全注意力、32 个 FFN 模块）

本文档是基于截至 2026 年 4 月 11 日的工作时间线整理得到的本地项目日志，也是本仓库中新增代码注释所对应的叙事性权威记录。

---

## 阶段 0：提案细化

**我们做了什么**
- 提交了初版提案，核心关注单用户边缘推理、VRAM 分析，以及 decode 阶段的量化 CUDA kernel。
- 在收到反馈后收缩范围，将重点从宽泛的系统分析转向一到两个更有力度的 kernel 优化点。
- 最终确定了两个主要方向：decode 阶段的 INT8 线性投影加速，以及针对 DeltaNet 的专项优化。

**关键决策**
- 放弃 prefill 优化，因为该路径已经由 cuBLAS 高效覆盖，进一步取得有意义收益的空间很小。
- 采用 RTX 4090 Laptop 作为更贴近现实的边缘代理平台，而不是强行使用更小的部署设备。
- 保留 Qwen3.5-4B 作为目标模型，因为其 DeltaNet 与全注意力混合结构为项目提供了更鲜明的系统研究角度。

**项目结果**
- 本阶段没有单独的 benchmark 结果文件，主要产出是项目范围的收敛。
- 项目正式聚焦为两条主线：DeltaNet decode kernel 优化，以及 decode 阶段 INT8 线性投影优化。
- 这一收敛直接决定了后续阶段的代码、实验和结果文件组织方式。

---

## 阶段 1：共享基础设施与基线

**我们做了什么**
- 搭建了 Python、PyTorch、Transformers 和 FLA 环境。
- 验证了模型加载流程，并对 `Qwen3_5ForConditionalGeneration` 做了 smoke test。
- 统计出 248 个目标线性层，覆盖 FFN、全注意力和 DeltaNet 投影。
- 为有代表性的 decode 阶段线性层建立了 FP16 延迟与带宽基线。
- 实现了按输出通道的对称 INT8 量化、误差分析以及 CPU 参考 kernel。
- 确认了真实投影形状，包括更宽的 `q_proj` 以及 Qwen3.5 使用的融合 DeltaNet 投影。

**关键结果**
- 在目标 GPU 上，FP16 的 decode 阶段 GEMV 已经能达到大约 75% 到 90% 的可用内存带宽，这说明 decode 路径明显受内存带宽限制。
- 这支撑了我们最初的假设：仅对权重做 INT8 量化有机会通过减少数据搬运量来降低延迟。

**项目结果**
- 结构与缓存诊断产物：
  - [deltanet_layer_tensors.json](/home/haozhong/ECE9483/artifacts/deltanet_diagnostics/deltanet_layer_tensors.json)
  - [decode_hook_summary.json](/home/haozhong/ECE9483/artifacts/deltanet_diagnostics/decode_hook_summary.json)
  - [prefill_cache_summary.json](/home/haozhong/ECE9483/artifacts/deltanet_diagnostics/prefill_cache_summary.json)
  - [baseline_compare.json](/home/haozhong/ECE9483/artifacts/deltanet_diagnostics/baseline_compare.json)
- `baseline_compare.json` 显示，在接入 FLA 后，`short_128`、`mid_512`、`long_2048` 三个场景的 decode 单 token 延迟分别达到 `1.240x`、`1.301x`、`1.235x` 的提升。
- 这一阶段同时确认了 DeltaNet decode state/cache 为固定形状 `[1, 32, 128, 128]`，为后续 `T=1` 特化 kernel 提供了结构依据。

---

## 阶段 2A：DeltaNet Decode Triton Kernel

**我们做了什么**
- 研究了 `flash-linear-attention` 中的 recurrent DeltaNet kernel，并识别出其在 `T=1` decode 场景下的低效点。
- 设计了一个面向 decode 的 Triton 专用 kernel：
  - 移除了时间维循环
  - 移除了变长逻辑
  - 增大了 V 维 tile 大小以减少 program 数量
  - 在 kernel 内保留 Q/K 归一化
- 还尝试了一个 fused-gate 变体，将 sigmoid 和 softplus 逻辑也融合进 Triton kernel。

**观测结果**
- 这个面向 decode 的 Triton 专用 recurrent kernel 在微基准测试中，相比 PyTorch fallback 和通用 FLA recurrent kernel 都取得了明显提升。
- fused-gate 版本反而略慢于更简单的版本，说明额外的寄存器压力抵消了融合轻量标量计算所带来的收益。

**项目结果**
- 核心微基准结果文件：
  - [deltanet_microbenchmark_latest.txt](/home/haozhong/ECE9483/artifacts/qwen35_integration/deltanet_microbenchmark_latest.txt)
  - [deltanet_trend_summary.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/deltanet_trend_summary.json)
- kernel-only 微基准数值：
  - `pytorch_naive = 282.713 us`
  - `fla_fused_recurrent = 94.446 us`
  - `triton_decode = 28.965 us`，相对 FLA 为 `3.261x`
  - `triton_decode_fused_gates = 33.580 us`，相对 FLA 为 `2.813x`
- 结果说明 decode 专用 Triton recurrent kernel 是一个真实有效的微观优化，而 fused-gates 方案没有继续扩大收益。

---

## 阶段 2B：DeltaNet 端到端集成

**我们做了什么**
- 修改了 Qwen3.5 的 DeltaNet decode 路径，使其在层级 forward 中直接调用 Triton recurrent kernel。
- 运行了短序列和长序列 decode 长度下的端到端生成基准测试。

**观测结果**
- 微基准中的收益只转化为了较为有限的端到端提升。
- Profiling 显示，recurrent DeltaNet 状态更新只占整体 decode 延迟中的很小一部分，而线性投影才是主要开销来源。

**核心洞察**
- 一个微观上很快的 kernel，并不意味着端到端一定会有明显收益，因为 Amdahl 定律会限制总体加速空间。

**项目结果**
- 端到端结果汇总文件：
  - [deltanet_trend_summary.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/deltanet_trend_summary.json)
  - [qwen35_single_user_round_robin_summary.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_single_user_round_robin_summary.json)
- 稳定后的共享模型 round-robin 结果显示：
  - `gen=16` 时，端到端从 `555.85 ms` 降到 `518.49 ms`，约 `1.072x`
  - `gen=128` 时，端到端从 `7008.48 ms` 降到 `6584.44 ms`，约 `1.064x`
  - `gen=256` 时，端到端从 `14077.69 ms` 降到 `13076.21 ms`，约 `1.077x`
- 这些结果保持与基线相同的生成输出，是本项目中最稳健、最可复现的端到端增益来源。

---

## 阶段 3A：INT8 GEMV Triton Kernel

**我们做了什么**
- 实现了一个带按输出通道 scale 的 Triton 权重量化 INT8 GEMV kernel。
- 针对有代表性的 FFN、全注意力和 DeltaNet 形状，加入了 autotuning 和正确性测试。

**观测结果**
- 大型 decode 阶段投影通常能在微基准中展现出较强的加速效果。
- 较小的投影则经常输给 FP16，因为 kernel 启动开销占了主导。

**项目结果**
- 微基准结果文件：
  - [benchmark_int8_gemv_median.csv](/home/haozhong/ECE9483/artifacts/qwen35_integration/benchmark_int8_gemv_median.csv)
- 代表性 median 结果：
  - `FFN_gate [9216,2560]` 的 hybrid 路径相对 FP16 为 `1.756x`
  - `FFN_down [2560,9216]` 的 hybrid 路径相对 FP16 为 `1.863x`
  - `FullAttn_q [8192,2560]` 的 hybrid 路径相对 FP16 为 `1.387x`
  - `DeltaNet_qkv [8192,2560]` 的 hybrid 路径相对 FP16 为 `1.790x`
  - `FullAttn_k [1024,2560]` 只有 `0.923x`
  - `DeltaNet_z [4096,2560]` 只有 `0.477x`
- 结果说明 INT8 GEMV 只在足够大的 decode 投影上才有稳定优势。

---

## 阶段 3B：选择性阈值与混合路由

**我们做了什么**
- 增加了选择性路由，让 decode 只在足够大的层上使用 INT8。
- 将 fallback 改为使用缓存好的 FP16 权重，而不是每次调用都重新反量化。
- 通过重复试验并比较中位数而非单次噪声测量，提升了 benchmark 的稳定性。

**观测结果**
- 实际可行的阈值上移到了大约 2000 万参数。
- 这使得 FFN 和一部分大型 attention 投影成为仅有的、在 decode 阶段稳定受益于 INT8 的候选对象。

**项目结果**
- 路由与阈值结果文件：
  - [benchmark_int8_gemv_threshold.csv](/home/haozhong/ECE9483/artifacts/qwen35_integration/benchmark_int8_gemv_threshold.csv)
  - [benchmark_int8_gemv_median.csv](/home/haozhong/ECE9483/artifacts/qwen35_integration/benchmark_int8_gemv_median.csv)
- 阈值扫描显示：
  - `FFN_gate` 与 `FFN_down` 在 hybrid 路径上分别达到 `2.243x` 和 `2.097x`
  - `FullAttn_k` 在 hybrid 路径上仅 `1.06x`，收益边缘
  - `DeltaNet_z` 在 hybrid 路径上仅 `0.447x`，明显不适合走该路径
- 因此最终将 decode 路由阈值提升到 `20_000_000` 参数，并把小中型投影留在 FP16 路径。

---

## 阶段 3C：第一次端到端 INT8 集成

**我们做了什么**
- 用 `QuantLinearINT8` 替换了符合条件的 Qwen3.5 线性层。
- 对短生成和长生成长度都进行了 benchmark。

**观测结果**
- 尽管微基准表现不错，逐层替换的 INT8 方案在端到端上反而更差。
- 最终识别出两个根本原因：
  - 逐层 wrapper 带来的 Python dispatch 开销会在 decode 中不断累积。
  - 量化误差会在更长的 decode 过程中导致生成漂移。

**核心洞察**
- 仅靠 kernel 本身变快还不够，dispatch 结构同样关键。

**项目结果**
- 端到端结果文件：
  - [qwen35_int8_deployment_compare_gen16_v2.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_int8_deployment_compare_gen16_v2.json)
  - [qwen35_int8_deployment_compare_gen128_v2.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_int8_deployment_compare_gen128_v2.json)
- 结果显示逐层 INT8 集成没有转化为模型级收益：
  - `gen=16` 时，相对 FP16 的端到端速度仅为 `0.838x`，`same_generation = true`
  - `gen=128` 时，相对 FP16 的端到端速度仅为 `0.844x`，`same_generation = false`
- 这一步明确证明了“kernel 快”不等于“端到端快”，模块调度成本已经开始压过单个 kernel 的收益。

---

## 阶段 4：融合式 FFN Decode 模块

**我们做了什么**
- 将 `gate_proj` 和 `up_proj` 融合进一个用于 decode 的 Triton kernel。
- 将激活函数和逐元素合并保留在 PyTorch 中，因为这部分成本较小。
- 对 `down_proj` 继续使用 INT8 GEMV。

**观测结果**
- 融合减少了足够多的 Python 边界和 kernel 启动边界，从而收回了逐层 wrapper 带来的性能损失。
- 短生成场景下的端到端性能大致追平了 FP16。
- 当 32 个 FFN 层全部量化时，长生成的精度仍会发生漂移。

**核心洞察**
- 模块级融合比朴素的逐层量化替换策略更有效。

**项目结果**
- 结果文件：
  - [qwen35_fused_ffn_deployment_compare_gen16.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_fused_ffn_deployment_compare_gen16.json)
  - [qwen35_fused_ffn_deployment_compare_gen128.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_fused_ffn_deployment_compare_gen128.json)
- 微观收益总结：
  - 融合 `gate + up` 相比两次独立 INT8 GEMV 调用约 `1.62x`
  - 整个 FFN block 相比 FP16 MLP 微基准约 `2.13x`
- 端到端结果：
  - `gen=16` 时，`fused_ffn` 的端到端均值为 `640.07 ms`，相对 FP16 为 `1.006x`，`same_generation = true`
  - `gen=128` 时，`fused_ffn` 的端到端均值为 `4879.60 ms`，相对 FP16 为 `0.899x`，`same_generation = false`
- 结果说明融合式 FFN 的确解决了大量 dispatch 开销，但简单 INT8 量化的数值稳定性仍然限制了长输出场景。

---

## 阶段 5：层子集实验

**我们做了什么**
- 测试了部分 FFN 替换方案：前 8 层、后 8 层、前 16 层、后 16 层，以及全部 32 层。

**观测结果**
- 在短生成场景下，最佳点是仅量化后 8 个 FFN 层。
- 靠后的 FFN 层比前面的层更能容忍量化，这与文献中已知的层敏感性规律一致。
- 即使是最优子集，在更长的 decode 序列上仍然会发生发散。

**核心洞察**
- 误差落在哪些层上很重要，而不只是误差本身有多大。

**项目结果**
- 子集实验结果文件：
  - [qwen35_fused_ffn_subset_compare_gen16.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_fused_ffn_subset_compare_gen16.json)
  - [qwen35_fused_ffn_subset_compare_gen128_candidates.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_fused_ffn_subset_compare_gen128_candidates.json)
- `gen=16` 时：
  - `back_8` 相对 FP16 的端到端速度为 `1.085x`，`same_generation = true`
  - `front_16` 相对 FP16 的端到端速度为 `1.059x`，`same_generation = true`
  - `all_32` 相对 FP16 的端到端速度为 `0.997x`，`same_generation = true`
- `gen=128` 时，最有希望的候选也没有守住：
  - `back_8` 相对 FP16 的端到端速度为 `0.929x`，`same_generation = false`
  - `front_16` 相对 FP16 的端到端速度为 `0.930x`，`same_generation = false`
- 这一步最终把“最佳短输出子集是后 8 层”这个结论固定下来，也同时确认它仍不具备长输出稳定性。

---

## 阶段 6：`torch.compile` 探索

**我们做了什么**
- 测试 `torch.compile` 是否能比手工 kernel 优化更有效地去除 Python dispatch 开销。
- 首先尝试了原生 FP16。
- 然后将 compile 与自定义 Triton 路径结合，最后再扩展到多步生成。

**观测结果**
- 单步微基准的结果看起来非常理想。
- 但多步生成没能保住这些收益，因为 CUDA graph 与自定义 Triton DeltaNet kernel 的交互会带来严重退化，或者要求关闭 compile 获益所依赖的那条关键路径。

**核心洞察**
- 没有 graph break 并不等于执行计划就一定好，而真实部署栈仍然无法兼容一种切实可用的 compile 加速方案。

**项目结果**
- compile 探索结果文件：
  - [qwen35_compiled_deployment_compare_gen16.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_compiled_deployment_compare_gen16.json)
  - [qwen35_compiled_deployment_compare_gen128.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_compiled_deployment_compare_gen128.json)
- `gen=16` 时：
  - `fp16_eager` 端到端均值为 `590.97 ms`
  - `fp16_compiled` 端到端均值为 `739.47 ms`，相对 eager 仅 `0.799x`
  - `back8_compiled_deltanet` 端到端均值为 `621.08 ms`，相对 eager 仅 `0.952x`
- `gen=128` 时：
  - `fp16_eager` 端到端均值为 `4560.05 ms`
  - `fp16_compiled` 端到端均值为 `5378.67 ms`，相对 eager 仅 `0.848x`，`same_generation = true`
  - `back8_compiled_deltanet` 相对 eager 仅 `0.929x`，且 `same_generation = false`
- 基于 2026 年 4 月 10 日当时的实验，本阶段的结论是：在 `DynamicCache` 路径上，compile 在真实多步生成栈里没有带来可用收益，因此被放弃。
- 这个判断后来在阶段 8 被部分修正：问题的根因并不是 compile 整体不可用，而是 `DynamicCache` 导致 KV 地址不稳定，以及最初选取的 compile 粒度不合适。

---

## 阶段 7：DeltaNet 模块级融合收益评估

**我们做了什么**
- 估算了如果在模块层面进一步融合更多 DeltaNet 投影，可能带来的收益。

**决策**
- 暂缓这部分工作，因为预期的单 token 收益相对于总体 decode 延迟来说过小。

**项目结果**
- 本阶段没有新增独立 benchmark 或 artifact 文件。
- 该阶段的直接产出是工程决策：暂不投入更大粒度的 DeltaNet 模块级融合，因为预估收益不足以覆盖实现和验证成本。
- 这个决策也解释了为什么后续资源优先投入到了 FFN 融合、层子集筛选以及 compile 可行性验证上。

---

## 阶段 8：Static KV Cache + CUDA Graph 恢复 compile 收益

**我们做了什么**
- 为 Qwen3.5 新增了 StaticCache 兼容层，修补了混合 `linear_attention + full_attention` 结构在 `create_masks_for_generate` 上与 Hugging Face `StaticCache` 的不兼容问题。
- 将 compile 策略改成 `prefill eager -> decode compile`，并把 compile 粒度收缩到 `full_attention self_attn`，不再编译 MLP。
- 保持 DeltaNet Triton kernel 处于 eager 路径，不把它拉进 cudagraph capture。
- 在同一条 static-compile 路径上，先确立新的 Step 2 基线 `fp16_static_compiled_attn_only_deltanet`，再把 Phase 5 的 `back_8` fused FFN 接成 Step 3 候选 `back8_static_compiled_attn_only_deltanet`。
- 按 `gen=16` 和 `gen=128` 两个长度重新做了严格的 same-generation 对比，并额外运行了 `TORCH_LOGS=recompiles` smoke 检查。

**观测结果**
- StaticCache 解决了 compile 路径下 KV 地址变化的问题，使 CUDA graph 可以在 decode 多步场景中稳定工作。
- `TORCH_LOGS=recompiles` 只观察到 8 个 full-attention 层因 `self.layer_idx` 触发的已知 warmup 重编译，没有再出现此前那种 DeltaNet Triton 与 cudagraph 交互导致的秒级灾难性退化。
- MLP compile 仍然会拖慢 steady-state decode，而 `attn-only compile` 才是能保住收益的正确粒度。
- `back_8` fused FFN 在这条 static-compile 路径上仍能维持短生成精确一致，但没有超越 Step 2；在长生成上也再次失去精确一致性。

**核心洞察**
- compile 的真实失败根因是 `DynamicCache + 错误的 compile 粒度`，而不是 compile 本身对这个模型完全无效。
- 当前最强、最稳定的可落地配置已经从“仅 DeltaNet Triton”升级为“StaticCache + attn-only compile + DeltaNet Triton”。
- `back_8` fused FFN 更适合作为短生成 ablation，而不是新的默认最佳配置。

**项目结果**
- 相关结果文件：
  - [qwen35_static_compile_attn_only_back8_recompile_smoke.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_static_compile_attn_only_back8_recompile_smoke.json)
  - [qwen35_static_compile_attn_only_back8_compare_2026-04-11_gen16.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_static_compile_attn_only_back8_compare_2026-04-11_gen16.json)
  - [qwen35_static_compile_attn_only_back8_compare_2026-04-11_gen128.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_static_compile_attn_only_back8_compare_2026-04-11_gen128.json)
- `gen=16` 时，四档 ablation 梯度如下：
  - `fp16_eager = 622.02 ms`
  - `fp16_static_compiled_attn_only = 560.62 ms`，相对 eager 为 `1.110x`，`same_generation = true`
  - `fp16_static_compiled_attn_only_deltanet = 533.90 ms`，相对 eager 为 `1.165x`，`same_generation = true`
  - `back8_static_compiled_attn_only_deltanet = 554.74 ms`，相对 eager 为 `1.121x`，`same_generation = true`
- `gen=128` 时，四档 ablation 梯度如下：
  - `fp16_eager = 4436.87 ms`
  - `fp16_static_compiled_attn_only = 4398.08 ms`，相对 eager 为 `1.009x`，`same_generation = true`
  - `fp16_static_compiled_attn_only_deltanet = 3909.76 ms`，相对 eager 为 `1.135x`，`same_generation = true`
  - `back8_static_compiled_attn_only_deltanet = 3964.17 ms`，相对 eager 为 `1.119x`，`same_generation = false`
- 这一步的最终结论非常明确：Step 2 路径现在是短生成和长生成的共同最佳配置；Step 3 在短生成中保住了质量，但没有进一步提升速度，在长生成中也没有通过精度门槛。

---

## 阶段 9：MTP 与 DeltaNet Custom Op 路线落地

**我们做了什么**
- 保留阶段 8 的 `StaticCache + attn-only compile + DeltaNet Triton` 作为默认最佳路径，并新增一个 `fp16_static_compiled_attn_only_deltanet_custom_op` 实验模式，用于验证 DeltaNet Triton kernel 通过 `torch.library.custom_op` 接入 compiler 后的行为。
- 将 fused-gate DeltaNet decode kernel 包装成 `qwen35::deltanet_decode_step_fused_gates` custom op，显式声明其会原地更新 recurrent state，使 `torch.compile` 可以把它视为 opaque op，而不是 trace 进 Triton 实现内部。
- 新增 `Qwen35MTPDraftModule`，从 checkpoint 中读取被 HF 默认忽略的 `mtp.*` 权重，复用目标模型的 token embedding 和 `lm_head`，形成一个独立的 one-step MTP draft head。
- 新增 `benchmark_qwen35_mtp_self_speculation.py`，实现 greedy 单 draft token self-speculation 控制流：MTP 先给出 draft token，target model 用一次两 token decode pass 验证；接受时一次 target pass 可推进两个 token，拒绝时回滚 cache 并走 correction pass。

**观测结果**
- 本阶段先完成代码路径和单元测试闭环，还没有把 MTP 路径提升为默认最佳配置。
- custom-op 路径已作为可选 runtime mode 接入，并完成了一次 `gen=8` CUDA smoke 对照。结果显示 custom-op 模式生成一致，但在当前 `attn-only compile` 粒度下没有带来速度收益。
- MTP 路径已经能发现 `mtp_num_hidden_layers: 1` 和 checkpoint 中的 `mtp.*` 权重，并记录 attempted、accepted、rejected、acceptance_rate 与 target_passes；下一步关键指标是实际接受率和端到端速度，而不是单元测试中的控制流正确性。
- 本地重测环境使用 `/home/haozhong/vllm-env/bin/python`，而不是系统 `python` 或 Anaconda；该环境为 Python `3.12.3`、PyTorch `2.9.1+cu128`、CUDA runtime `12.8`。在 WSL2 环境中 `nvidia-smi` 可见 RTX 4090 Laptop，PyTorch 报告 `torch.cuda.is_available() = true`、`device_count = 1`，并且 CUDA tensor 分配成功。
- 使用上述 `vllm-env` 重新运行聚焦测试：`triton_kernels.test_deltanet_decode`、`triton_kernels.test_qwen35_integration`、`triton_kernels.test_benchmark_qwen35_compiled_deployment`、`triton_kernels.test_qwen35_mtp_self_speculation`、`triton_kernels.test_benchmark_qwen35_mtp_self_speculation`。结果为 `28` 个测试全部通过，没有 skip。
- custom-op smoke benchmark 命令为 `TORCH_LOGS=recompiles /home/haozhong/vllm-env/bin/python benchmark_qwen35_compiled_deployment.py --modes fp16_static_compiled_attn_only_deltanet fp16_static_compiled_attn_only_deltanet_custom_op --gen-tokens 8 --warmup-runs 1 --runs 1 --output artifacts/qwen35_integration/qwen35_custom_op_compare_gen8_vllm_env.json`。结果文件显示：非 custom-op 模式 decode 均值 `28.22 ms`、端到端 `252.87 ms`；custom-op 模式 decode 均值 `31.54 ms`、端到端 `290.56 ms`，相对非 custom-op 的 decode speedup 为 `0.895x`、端到端 speedup 为 `0.870x`，`same_generation = true`。

**核心洞察**
- StaticCache + compile 的主路径不应再回退到 `DynamicCache`；新增工作都必须作为正交实验挂在当前 Step 2 周围。
- MTP self-speculation 的第一版应该先验证接受率和 cache 控制流，再讨论是否与 StaticCache/cudagraph 完全融合。
- DeltaNet custom op 是 compiler 兼容性层，不是新的数学 kernel；当前 `attn-only compile` 路径并不会因为它自动变快，反而在 `gen=8` smoke 中有额外 dispatcher/调用开销。它的价值仍主要在未来尝试更大 compile 粒度时降低 Triton trace 和 graph capture 交互风险。

**项目结果**
- 新增代码与测试：
  - `triton_kernels/qwen35_mtp_self_speculation.py`
  - `benchmark_qwen35_mtp_self_speculation.py`
  - `triton_kernels/test_qwen35_mtp_self_speculation.py`
  - `triton_kernels/test_benchmark_qwen35_mtp_self_speculation.py`
- 扩展代码与测试：
  - `triton_kernels/deltanet_decode.py`
  - `triton_kernels/qwen35_integration.py`
  - `benchmark_qwen35_compiled_deployment.py`
  - `triton_kernels/test_deltanet_decode.py`
  - `triton_kernels/test_qwen35_integration.py`
  - `triton_kernels/test_benchmark_qwen35_compiled_deployment.py`

---

## 阶段 10：量化作为部署压缩维度

**定位调整**
- 量化不应再被叙述为与 `StaticCache + compile` 或 MTP 直接竞争的 latency 主线，而应重新定位为部署压缩维度：降低权重显存、降低带宽压力，并给嵌入式或低显存设备提供更现实的部署 trade-off。
- 阶段 3 到阶段 5 已经证明，INT8 在 HF eager + `DynamicCache` 下端到端收益边缘甚至为负；但阶段 8 已经把主路径切到 `StaticCache + attn-only compile`，因此仍然需要在当前最佳 runtime 条件下补一次 closure。
- 这条 closure 无论结果如何都有报告价值：如果延迟仍持平或略负，就是完整的 embedded trade-off 数据；如果在 static-compile 路径上反而变快，则是额外 bonus。

**Phase 10A：W8A16 Static-Compile Closure**
- 第一版量化实验应直接复用阶段 4 的 fused FFN / fused GEMV 路径，不再重复阶段 3C 的逐层 `QuantLinearINT8` wrapper。逐层替换已经被 dispatch overhead 证明不是好路径。
- 对照基线应使用当前最佳模式 `fp16_static_compiled_attn_only_deltanet`，量化模式则只作为同一 runtime 条件下的部署压缩 ablation。
- 第一版保持 DeltaNet recurrent state、StaticCache、norm 与 `lm_head` 为 FP16；只量化真正覆盖大量矩阵乘的权重路径。
- 评估指标必须绑定 quality 与 performance 两侧：PPL 或小 eval loss、greedy consistency 或 token diff rate、TTFT、decode mean、端到端延迟，以及 peak memory / weight memory。

**Phase 10A 执行结果**
- 新增 `w8a16_static_compiled_attn_only_deltanet` 模式：保持 `StaticCache + attn-only compile + DeltaNet Triton` 不变，将阶段 4 的 fused FFN / fused GEMV 扩展到全部 `32` 个 FFN 层，而不是阶段 8 的 `back_8` 子集。
- benchmark 现在额外记录 `token_diff_count` / `token_diff_rate`、generated-token NLL / PPL proxy、CUDA peak memory，以及模型参数和 buffer 的 tensor memory。这里的 NLL / PPL proxy 是 greedy 生成 token 上的 decode confidence 指标，不等同于完整语料 PPL。
- 结果文件：
  - [qwen35_phase10a_w8a16_static_compile_gen16_vllm_env.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_phase10a_w8a16_static_compile_gen16_vllm_env.json)
  - [qwen35_phase10a_w8a16_static_compile_gen128_vllm_env.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_phase10a_w8a16_static_compile_gen128_vllm_env.json)
- `gen=16` 时，当前最佳 FP16 路径 decode 均值 `33.82 ms`、端到端 `573.99 ms`、peak CUDA memory `8707.15 MiB`；全量 W8A16 fused FFN decode 均值 `40.09 ms`、端到端 `667.45 ms`、peak CUDA memory `10868.55 MiB`。W8A16 相对 FP16 当前最佳的 decode speedup 为 `0.844x`、端到端 speedup 为 `0.860x`，`same_generation = true`，`token_diff_rate = 0.0`。
- `gen=128` 时，当前最佳 FP16 路径 decode 均值 `28.75 ms`、端到端 `3702.21 ms`、peak CUDA memory `8710.65 MiB`；全量 W8A16 fused FFN decode 均值 `35.33 ms`、端到端 `4544.87 ms`、peak CUDA memory `10872.06 MiB`。W8A16 相对 FP16 当前最佳的 decode speedup 为 `0.814x`、端到端 speedup 为 `0.815x`，并且 `same_generation = false`，`token_diff_count = 4`，`token_diff_rate = 0.03125`。
- 这一结果把旧 INT8 结论迁移到了阶段 8 的最佳 runtime 条件下：全量 W8A16 FFN 在 static-compile 主路径上仍没有端到端速度收益，且长生成质量漂移仍未解决。
- 当前 fused FFN 实现为了 prefill 和 fallback 仍保留 FP16 `original_mlp`，因此它不是实际显存压缩版部署。全量 W8A16 模式额外持有约 `2160 MiB` 的 INT8 buffer，模型 tensor memory 从约 `8658 MiB` 增至约 `10819 MiB`。所以 Phase 10A 的结论不是“W8A16 已经压缩模型”，而是“在当前 fused-runtime 设计下，W8A16 不能作为默认交付路径”。

**Phase 10B：W4A16 AWQ Quality + Memory Probe**
- W4A16 与 W8A16 之间不是简单连续小步，而是 kernel 与系统边界切换：W8A16 可以复用现有 Triton INT8 GEMV；W4A16 需要 bit unpacking、group-wise scale 与 vectorized dequant，或者直接接 AWQ 官方 kernel。
- W4 第一版应先关闭 compile，用 AWQ 官方路径或现成 kernel 拿到 quality 与 memory 数字；eager latency 只作为参考，不作为淘汰主指标。
- 自动化量化工具不能无脑扫描所有 `nn.Linear`。Qwen3.5 的 DeltaNet 层中 `in_proj_qkv`、`in_proj_z`、`in_proj_a`、`in_proj_b`、`out_proj` 都是线性层，但其中 `in_proj_a` / `in_proj_b` 形状很小，例如 `[32, 2560]`，容易遇到 calibration 不稳、scale 统计失真或 kernel 不支持小 `out_features` 的问题。
- AWQ skip list 应明确维护，至少包含：`lm_head`、`in_proj_a`、`in_proj_b`、所有 `out_features < 1024` 的 projection，以及必要时其他 tiny DeltaNet gate/projection。这本身就是一个可以写进报告的 lesson learned：非标准架构里“扫所有 `nn.Linear`”不是安全默认策略。

**Phase 10C：AWQ Kernel 与 Compile 兼容性**
- 只有当 W4A16 的 quality 和 memory probe 过关后，才值得继续验证 AWQ kernel 与 `torch.compile` / cudagraph 的兼容性。
- 如果 AWQ eager kernel 破坏 graph capture，再考虑将它包装为 `torch.library.custom_op`，让 compiler 把它视作带 schema 的 opaque op。
- 这会是阶段 9 custom-op 经验的正向复用场景：DeltaNet custom-op 在当前 attn-only compile 粒度下没有 net speedup，但外部 AWQ kernel 接入 compile 路径正是 custom-op 更合适的用途。

**Phase 10D：Quantized Target + FP16 MTP Draft**
- 量化与 MTP 不应简单假设完全正交。MTP draft head 本身会影响 acceptance rate；如果 draft 也被量化，draft quality 下降可能直接抵消 speculative decoding 的收益。
- 更稳妥的组合是 target model 使用 W8A16 或 W4A16，而 MTP draft/head 保持 FP16。这样可以先让 target 变便宜，同时尽量不牺牲 draft token 的接受率。
- 这一阶段需要新增的关键指标包括 draft acceptance rate、accepted tokens per target step、rejected-step overhead，以及最终 quality drift。

---

## 阶段 11：DeltaNet Triton Kernel Autotune 与 L2 Norm 去重

**我们做了什么**
- 回到阶段 2A 的 DeltaNet decode 专用 kernel，对两个低成本方向做局部深挖：Triton autotuning，以及 q/k L2 norm 去重。
- 为 base recurrent kernel 和 fused-gate recurrent kernel 增加了 `DELTANET_AUTOTUNE_CONFIGS`，候选覆盖 `BV=16/32/64/128`、`num_warps=2/4/8`、`num_stages=1/2` 的小集合。
- 新增了 `@triton.autotune` 版本的 base / fused-gate kernel，但没有让真实 recurrent state 直接参与调优。由于该 kernel 会原地更新 `state`，直接 autotune 会让 state restore / copy 干扰计时；最终改为先在 scratch state 上选择配置，再缓存 `DeltaNetKernelConfig`，真实 decode path 仍只更新一次 state。
- 新增 `deltanet_l2_normalize_qk()`，允许在 kernel 外预先归一化 q/k，然后以 `use_qk_l2norm=False` 调用 decode kernel，从而避免同一个 head 在多个 V tile 上重复计算 L2 norm。
- 扩展 `benchmark_deltanet_decode.py`，显式区分：
  - 旧手动配置路径：`triton_decode_manual_bv32`、`triton_fused_manual_bv64`
  - 默认 autotune 路径：`triton_decode_autotune`、`triton_fused_autotune`
  - q/k 已预归一化的 kernel-only 路径：`triton_fused_prenorm_bv64`、`triton_fused_prenorm_kernel`
  - 将 PyTorch normalize 成本也计入的端到端局部路径：`triton_fused_prenorm_included`

**观测结果**
- Autotune 在这台 RTX 4090 Laptop 上能找到与手动配置相近、偶尔更快的候选，但收益受测量噪声影响，不是稳定数量级提升。
- 手动 fused-gate config 原先是 `BV=64, num_warps=8, num_stages=1`；一次 fused config sweep 中，最快候选为 `fused_bv16_w2_s2 = 27.826 us`，而旧手动 `fused_bv64_w8_s1 = 29.793 us`，kernel-only 约有 `6.6%` 潜在提升。
- 默认 autotune 的实际选择会随 benchmark 噪声波动。例如一次检查中，base 选择 `DeltaNetKernelConfig(bv=64, num_warps=2, num_stages=1)`，fused-gate 选择 `DeltaNetKernelConfig(bv=32, num_warps=4, num_stages=1)`。
- L2 norm 去重在 kernel-only 情况下收益很小，基本落在微基准噪声范围内；但如果把外部 PyTorch normalize 也算进去，会显著变慢。因此它不能作为单独 PyTorch 前处理插入 hot path，后续只有在能融合进前面的 Q/K projection、reshape 或其他已有 kernel 时才有继续价值。

**项目结果**
- 修改与验证文件：
  - `triton_kernels/deltanet_decode.py`
  - `triton_kernels/test_deltanet_decode.py`
  - `benchmark_deltanet_decode.py`
- 代表性 microbenchmark 结果之一：
  - `triton_decode_manual_bv32 = 29.552 us`
  - `triton_decode_autotune = 29.458 us`
  - `triton_fused_manual_bv64 = 33.613 us`
  - `triton_fused_autotune = 30.044 us`
  - `triton_fused_prenorm_bv64 = 29.908 us`
  - `triton_fused_prenorm_included = 156.982 us`
- 另一轮复跑显示：
  - `triton_decode_manual_bv32 = 27.030 us`
  - `triton_decode_autotune = 30.574 us`
  - `triton_fused_manual_bv64 = 28.755 us`
  - `triton_fused_autotune = 30.634 us`
  - `triton_fused_prenorm_bv64 = 29.650 us`
  - `triton_fused_prenorm_included = 170.434 us`
- 这说明当前最诚实的结论是：autotune 值得保留为跨 GPU 适配机制，但需要 repeated-median 或离线 profile 选择默认配置；L2 norm 去重只有作为融合优化才值得继续推进。
- 正确性验证通过：
  - `~/vllm-env/bin/python -m unittest triton_kernels.test_deltanet_decode triton_kernels.test_qwen35_integration -v`
  - 结果为 `18 tests OK`
  - `~/vllm-env/bin/python -m py_compile benchmark_deltanet_decode.py triton_kernels/deltanet_decode.py triton_kernels/test_deltanet_decode.py triton_kernels/qwen35_integration.py`

**核心洞察**
- 对带原地 state update 的 recurrent kernel，不能把 `@triton.autotune` 当作普通纯函数 matmul kernel 使用；调优路径必须显式隔离可变 state，否则 benchmark 过程会污染状态或引入额外 restore 成本。
- Autotune 的工程价值更多体现在跨硬件可移植性，而不是在单台 GPU 上保证每轮都超过人工配置。
- L2 norm 重复计算在设计上确实存在，但单独前置成 PyTorch op 不划算。它应该被记录为“未来 fusion opportunity”，而不是当前默认优化路径。

---

## 阶段 12：INT8 GEMV dot_m16 Tensor Core 实验

**我们做了什么**
- 针对阶段 3 的 INT8 GEMV decode kernel，新增一个实验分支 `int8_gemv_dot_m16()`，用于验证“伪造 `M=16` 触发 Tensor Core”的路线，而不是直接替换现有 `tl.sum` kernel。
- 新 kernel 固定 `BLOCK_M=16`：第 0 行加载真实 decode 向量，其余 15 行填 0；输出时只把第 0 行写回。这让 `tl.dot([16, K], [K, N])` 可以走 Tensor Core 路径，同时把 15 行无效计算的代价暴露给 benchmark。
- 在 kernel 内将 `int8` weight 先转为 `float16`，与 `float16` per-channel scale 相乘完成 SRAM 内反量化，再输入 `tl.dot`；`acc` 仍保持 `float32`，避免把累加精度一起降掉。
- 扩展 `benchmark_int8_gemv.py`，新增 `dot_m16_ms`、`dot_m16_speedup`、`dot_m16_vs_sum` 和样本列，用同一组 shape 同时比较 FP16、当前 INT8 sum kernel、dot_m16 kernel 和 hybrid 路由。

**正确性验证**
- 新增 CUDA correctness test，覆盖当前目标 shape：
  - `FFN_gate [9216,2560]`
  - `FFN_down [2560,9216]`
  - `FullAttn_q [8192,2560]`
  - `FullAttn_k [1024,2560]`
  - `DeltaNet_qkv [8192,2560]`
  - `DeltaNet_z [4096,2560]`
- 与 CPU 量化 matvec reference 对齐，所有 shape 都满足 `cos_sim > 0.999`，最大绝对误差小于 `0.125`。代表性误差范围：
  - `max_abs_err = 0.064545` 到 `0.112701`
  - `mean_abs_err = 0.011321` 到 `0.021797`
  - `cos_sim = 0.99999934` 到 `1.00000000`
- 回归命令：
  - `~/vllm-env/bin/python -m unittest triton_kernels.test_int8_gemv triton_kernels.test_benchmark_int8_gemv -v`
  - 结果为 `12 tests OK`
  - `~/vllm-env/bin/python -m py_compile triton_kernels/int8_gemv.py triton_kernels/test_int8_gemv.py benchmark_int8_gemv.py triton_kernels/test_benchmark_int8_gemv.py`

**Benchmark 结果**
- 主要复跑命令：
  - `~/vllm-env/bin/python benchmark_int8_gemv.py --warmup 100 --iters 500 --repeats 5 --output artifacts/qwen35_integration/benchmark_int8_gemv_dot_m16_phase1_repeats5.csv`
- 结果如下，`dot_m16_vs_sum > 1` 表示 dot_m16 比当前 `tl.sum` kernel 更快：

| shape | FP16 ms | INT8 sum ms | dot_m16 ms | dot_m16 vs sum | Hybrid ms | Hybrid path |
|---|---:|---:|---:|---:|---:|---|
| `FFN_gate [9216,2560]` | `0.0885` | `0.0279` | `0.0271` | `1.028x` | `0.0453` | `int8` |
| `FFN_down [2560,9216]` | `0.0795` | `0.0369` | `0.0406` | `0.909x` | `0.0480` | `int8` |
| `FullAttn_q [8192,2560]` | `0.0452` | `0.0311` | `0.0307` | `1.012x` | `0.0396` | `int8` |
| `FullAttn_k [1024,2560]` | `0.0165` | `0.0285` | `0.0293` | `0.973x` | `0.0211` | `fp16` |
| `DeltaNet_qkv [8192,2560]` | `0.0707` | `0.0286` | `0.0285` | `1.004x` | `0.0403` | `int8` |
| `DeltaNet_z [4096,2560]` | `0.0177` | `0.0267` | `0.0373` | `0.716x` | `0.0198` | `fp16` |

**结论**
- dot_m16 的正确性成立，但性能不是单调胜利。它在 `FFN_gate`、`FullAttn_q`、`DeltaNet_qkv` 上小幅领先当前 sum kernel，幅度分别约为 `2.8%`、`1.2%`、`0.4%`；在 `FFN_down`、`FullAttn_k`、`DeltaNet_z` 上反而落后，最差的 `DeltaNet_z` 只有 `0.716x`。
- 这些小胜大多不足以支撑默认替换，因为它们接近 launch、occupancy、autotune 选择和 GPU 状态带来的测量波动；而失败 shape 的回退代价更明显。
- 当前不应把 `int8_gemv_dot_m16()` 提升为默认 decode 路径。更稳妥的工程定位是：保留为 Phase 1 实验分支和后续 shape-specific routing 候选，由 benchmark 数据决定是否只在少数 shape 上启用。
- 这次实验验证了前面的物理判断：Tensor Core 路线可以消除 `tl.sum` 的规约形态，但为 `M=1` 人为扩成 `M=16` 会引入 16 倍计算压力；它只有在额外计算代价被 HBM 节省充分覆盖时才可能赢。
- 下一步如果继续推进，应优先做 Phase 2 的结构化 Split-K 实验，尤其关注 `K >= 8192` 且 `N` 也足够宽的 shape；并且把默认路由阈值从单一参数量阈值升级为按 `(N, K)` 结构选择。

---

## 阶段 13：INT8 GEMV Split-K Workspace 实验

**我们做了什么**
- 针对阶段 12 暴露出的长 K 问题，新增 `int8_gemv_splitk_workspace()` 实验分支，用 Workspace + Reduce 的双 kernel 方案探测 Split-K 上限。
- 第一阶段 kernel 按 `(N tile, split_k)` 调度，把 K 维切成 `split_k` 段，每个 block 只计算一段 K 的未缩放 FP32 partial sum，并写入 `workspace[split_k, N]`。
- 第二阶段 reduce kernel 对 `split_k` 个 FP32 partial 做规约，再乘 per-channel scale，最终 cast 成 FP16 输出。
- 这版没有接入默认 `QuantLinearINT8` 路由，也没有混入 `dot_m16`，目的是单独隔离 Split-K 对当前 `tl.sum` 路线的真实价值。
- `benchmark_int8_gemv.py` 新增 `split_k = 2/4/8` 三个候选，并输出 `best_split_k`、`best_splitk_ms`、`best_splitk_vs_sum` 以及每个 split 的样本列。

**正确性验证**
- 新增 CUDA correctness test，重点覆盖 `FFN_down [2560,9216]` 这个长 K shape，并验证 `split_k = 2/4/8` 都能对齐 CPU reference。
- 验证结果：
  - `FFN_down_splitk2`: `max_abs_err = 0.119568`，`mean_abs_err = 0.013976`，`rel = 0.000179`，`cos = 0.99999994`
  - `FFN_down_splitk4`: `max_abs_err = 0.119568`，`mean_abs_err = 0.013976`，`rel = 0.000179`，`cos = 1.00000000`
  - `FFN_down_splitk8`: `max_abs_err = 0.119568`，`mean_abs_err = 0.013976`，`rel = 0.000179`，`cos = 0.99999988`
- 回归命令：
  - `~/vllm-env/bin/python -m unittest triton_kernels.test_int8_gemv triton_kernels.test_benchmark_int8_gemv -v`
  - 结果为 `14 tests OK`
  - `~/vllm-env/bin/python -m py_compile triton_kernels/int8_gemv.py triton_kernels/test_int8_gemv.py benchmark_int8_gemv.py triton_kernels/test_benchmark_int8_gemv.py`

**Benchmark 结果**
- 主 benchmark 命令：
  - `~/vllm-env/bin/python benchmark_int8_gemv.py --warmup 100 --iters 500 --repeats 5 --output artifacts/qwen35_integration/benchmark_int8_gemv_splitk_phase2_repeats5.csv`
- `best_splitk_vs_sum > 1` 表示 Split-K workspace 比当前 INT8 sum kernel 更快：

| shape | INT8 sum ms | dot_m16 ms | best splitK ms | best K | splitK vs sum | Hybrid ms | route |
|---|---:|---:|---:|---:|---:|---:|---|
| `FFN_gate [9216,2560]` | `0.0485` | `0.0302` | `0.0553` | `2` | `0.876x` | `0.0416` | `int8` |
| `FFN_down [2560,9216]` | `0.0380` | `0.0420` | `0.0578` | `4` | `0.658x` | `0.0403` | `int8` |
| `FullAttn_q [8192,2560]` | `0.0330` | `0.0297` | `0.0484` | `2` | `0.682x` | `0.0612` | `int8` |
| `FullAttn_k [1024,2560]` | `0.0487` | `0.0288` | `0.0479` | `8` | `1.016x` | `0.0236` | `fp16` |
| `DeltaNet_qkv [8192,2560]` | `0.0285` | `0.0290` | `0.0470` | `2` | `0.607x` | `0.0403` | `int8` |
| `DeltaNet_z [4096,2560]` | `0.0367` | `0.0314` | `0.0472` | `2` | `0.777x` | `0.0411` | `fp16` |

**Partial / Reduce 拆分诊断**
- 对 `FFN_down [2560,9216]` 单独计时 partial kernel 与 reduce kernel：
  - `split_k=2`: `partial = 0.0662 ms`，`reduce = 0.0148 ms`，估算合计 `0.0810 ms`
  - `split_k=4`: `partial = 0.0549 ms`，`reduce = 0.0150 ms`，估算合计 `0.0698 ms`
  - `split_k=8`: `partial = 0.0547 ms`，`reduce = 0.0161 ms`，估算合计 `0.0708 ms`
- 这说明失败不是单纯因为第二个 reduce kernel 或 launch overhead。即使只看 partial kernel，Split-K 后的计算本体也没有快过当前 sum kernel 的 `0.0380 ms`。

**结论**
- Split-K workspace 版本正确性成立，但性能结论是否定的：在这台 RTX 4090 Laptop 上，即使对 `FFN_down K=9216` 这种最应该受益的长 K shape，Split-K 也没有带来 occupancy 收益，反而因为更多 block、workspace 写入、reduce launch 与更差的 kernel 形态显著变慢。
- 这推翻了“长 K 一定需要 Split-K”的直觉。当前 `N=2560` 时，原始 sum kernel 的 N tile 数已经足够填充 SM，瓶颈不是明显的 SM 闲置；把 K 再切开会把每个 block 变短，额外调度和写回成本超过并行度收益。
- Atomic Add 版暂时不值得继续实现。原因是 workspace 版的 partial 本体已经慢于当前 sum kernel；atomic 只能省掉 reduce kernel，却会引入 L2 atomic contention 和输出初始化/累加顺序问题，无法解决 partial 本体变慢的问题。
- 当前默认路径仍应保持原 INT8 sum kernel + 参数量阈值 fallback。后续若要继续压 `FFN_down`，更有希望的方向不是 Split-K，而是改计算内核形态：例如更低精度反量化、向量化 packed int8 load、或者重新设计 dot_m16/TC 路线的 shape-specific autotune。

---

## 阶段 14：INT8 Storage / Runtime 路由拆分与 Fallback Cache 修复

**背景**
- 阶段 3C 的 `QuantLinearINT8.from_linear()` 默认会 clone 原始 FP16 weight，并把它注册为 `fallback_weight` buffer。这样模块会同时持有 `qweight`、`scale` 和 FP16 `fallback_weight`，实际显存占用比原始 FP16 Linear 更高。
- 同时，旧集成逻辑把 `INT8_GEMV_MIN_DECODE_PARAMS = 20_000_000` 同时用于两个不同决策：
  - 是否把 Linear 替换成 INT8 storage；
  - decode 时是否走 Triton INT8 GEMV kernel。
- 这会把小于 20M 参数的 attention projection 直接排除在量化 storage 外。更合理的设计是：storage 尽量量化，runtime 再按 shape 决定走 INT8 kernel 还是 fallback。

**我们做了什么**
- `QuantLinearINT8.from_linear()` 默认不再注册 FP16 `fallback_weight`。只有显式传 `preserve_fallback_weight=True` 时才保留原始 FP16 weight buffer。
- `_fallback_weight_cache` 改为普通 Python dict，按 `(dtype, device)` key 做 lazy dequant cache，不进入 `state_dict`。
- 新增 `clear_fallback_cache()`，允许调用方在知道后续不需要 fallback weight 时主动释放进程内缓存。
- `should_use_int8_gemv_decode()` 新增 `scale` 与 `triton is None` 检查；scale 不在 CUDA 或 Triton 不可用时直接返回 `False`，平滑走 fallback。
- `configure_qwen35_int8_runtime()` 和 `apply_qwen35_int8_modules()` 新增 `storage_min_params`，默认 `0`，把 storage replacement 与 runtime `min_decode_params` 分离。
- `benchmark_qwen35_int8_deployment.py` 默认改为：
  - `--storage-min-params 0`
  - `--cache-fallback-weight`
  - `--no-preserve-fallback-weight`
- `int8_gemv_dot_m16()` 和 `int8_gemv_splitk_workspace()` 从 `__all__` 中移除，保留为内部实验/benchmark ablation；dot_m16 docstring 明确它是 FP16 Tensor Core W8A16 实验，不是真 INT8 MMA 路径。
- `QuantLinearINT8` docstring 明确当前限制：symmetric per-channel W8A16 storage，bias 在输出 dtype 下由 PyTorch fallback/外层相加，不在 Triton FP32 accumulator 中融合。

**正确性验证**
- 新增/更新测试覆盖：
  - `from_linear()` 默认不注册 `fallback_weight`，`state_dict` 中没有 FP16 fallback buffer；
  - 显式 `preserve_fallback_weight=True` 时仍可保留原始 FP16 weight；
  - fallback cache 按 `(dtype, device)` 复用，并可通过 `clear_fallback_cache()` 清空；
  - storage replacement 默认不再被 `min_decode_params` 过滤；
  - 显式 `storage_min_params` 可恢复旧式 storage 过滤；
  - `should_use_int8_gemv_decode()` 在 Triton 不可用或 scale 不在 CUDA 时返回 `False`；
  - dot_m16 / splitK 实验函数不再从 `__all__` 导出。
- 回归命令：
  - `~/vllm-env/bin/python -m unittest triton_kernels.test_int8_gemv triton_kernels.test_qwen35_int8_integration triton_kernels.test_benchmark_int8_gemv triton_kernels.test_benchmark_qwen35_int8_deployment -v`
  - 结果为 `30 tests OK`
  - `~/vllm-env/bin/python -m py_compile triton_kernels/int8_gemv.py triton_kernels/qwen35_int8_integration.py triton_kernels/test_int8_gemv.py triton_kernels/test_qwen35_int8_integration.py benchmark_int8_gemv.py benchmark_qwen35_int8_deployment.py benchmark_qwen35_int8_round_robin.py benchmark_qwen35_fused_ffn_deployment.py`

**Fallback Cache Microbenchmark**
- 刷新后的 INT8 GEMV microbenchmark：
  - `~/vllm-env/bin/python benchmark_int8_gemv.py --warmup 60 --iters 300 --repeats 3 --output artifacts/qwen35_integration/benchmark_int8_gemv_storage_runtime_cache_phase14.csv`
- 其中小 shape fallback 的 cached hybrid 结果：
  - `FullAttn_k [1024,2560]`: FP16 `0.0203 ms`，cached hybrid `0.0216 ms`
  - `DeltaNet_z [4096,2560]`: FP16 `0.0484 ms`，cached hybrid `0.0457 ms`
- 单独比较 cache 与 no-cache：
  - `FullAttn_k`: cached `0.0228 ms`，no-cache `0.0755 ms`，cache speedup `3.304x`
  - `DeltaNet_z`: cached `0.0639 ms`，no-cache `0.3460 ms`，cache speedup `5.418x`

**结论**
- 这次修复把 W8A16 的 storage 语义和 decode kernel routing 语义拆开了：默认可以转换更多 Linear 为 INT8 storage，但小 shape decode 不会被强推到 custom INT8 kernel。
- Lazy fallback cache 是拆分后的关键粘合层。没有 cache 时，小 shape 每步 dequant 会明显拖慢；有 cache 时，fallback 近似恢复为一次 dequant 后的 PyTorch `F.linear`。
- 对可逆 runtime `configure_qwen35_int8_runtime()`，模型状态中仍会保留原始模块用于恢复 FP16，这是实验/runtime toggling 的成本；真正评估部署显存时应使用 `apply_qwen35_int8_modules()` 这类不可逆替换路径，并保持 `preserve_fallback_weight=False`。
- dot_m16 和 Split-K workspace 都保留为负结果/消融实验，不再作为公开导出 API，也不进入默认路由。

---

## 阶段 15：Phase 14 Storage/Runtime INT8 端到端复测

**复测目的**
- 阶段 14 修复了两个系统边界问题：默认不再保存 FP16 `fallback_weight`，并把 INT8 storage replacement 与 runtime decode kernel routing 拆开。
- 本阶段用真实 Qwen3.5 single-user decode 复测，判断这些修复是否足以把逐层 W8A16 `QuantLinearINT8` 从端到端负收益拉回来。
- 配置：
  - `--storage-min-params 0`
  - `--min-decode-params 20000000`
  - `--cache-fallback-weight`
  - `--no-preserve-fallback-weight`
  - modes: `fp16`、`int8_ffn`、`int8_all`
- 注意：当前 `benchmark_qwen35_int8_deployment.py` 还没有记录 peak CUDA memory 字段，因此本阶段结果只覆盖 latency 与 generation drift，不能作为显存 closure。

**gen=16 结果**
- 命令：
  - `~/vllm-env/bin/python benchmark_qwen35_int8_deployment.py --modes fp16 int8_ffn int8_all --gen-tokens 16 --warmup-runs 1 --runs 3 --storage-min-params 0 --min-decode-params 20000000 --cache-fallback-weight --no-preserve-fallback-weight --output artifacts/qwen35_integration/qwen35_phase14_storage_runtime_int8_gen16.json`
- 结果：

| mode | replaced layers | decode mean ms | end-to-end mean ms | decode speedup vs FP16 | end-to-end speedup vs FP16 | generation |
|---|---:|---:|---:|---:|---:|---|
| `fp16` | `0` | `35.142372` | `577.699024` | `1.000x` | `1.000x` | reference |
| `int8_ffn` | `96` | `41.128958` | `668.901644` | `0.854444x` | `0.863653x` | same |
| `int8_all` | `248` | `48.980182` | `799.861662` | `0.717481x` | `0.722249x` | same |

**gen=128 结果**
- 命令：
  - `~/vllm-env/bin/python benchmark_qwen35_int8_deployment.py --modes fp16 int8_ffn int8_all --gen-tokens 128 --warmup-runs 1 --runs 3 --storage-min-params 0 --min-decode-params 20000000 --cache-fallback-weight --no-preserve-fallback-weight --output artifacts/qwen35_integration/qwen35_phase14_storage_runtime_int8_gen128.json`
- 结果：

| mode | replaced layers | decode mean ms | end-to-end mean ms | decode speedup vs FP16 | end-to-end speedup vs FP16 | token diff |
|---|---:|---:|---:|---:|---:|---:|
| `fp16` | `0` | `35.654206` | `4581.354173` | `1.000x` | `1.000x` | reference |
| `int8_ffn` | `96` | `41.384294` | `5308.982777` | `0.861540x` | `0.862944x` | `8/128 = 0.0625` |
| `int8_all` | `248` | `43.936034` | `5635.277895` | `0.811503x` | `0.812978x` | `78/128 = 0.609375` |

**结论**
- Phase 14 的 storage/runtime 拆分与 lazy fallback cache 修复了架构语义问题，但没有改变逐层 W8A16 `QuantLinearINT8` 的端到端速度结论：`int8_ffn` 和 `int8_all` 在短生成、长生成下都慢于 FP16。
- `int8_all` 在 `gen=16` 没有 token drift，但 `gen=128` 出现严重漂移，`78/128` tokens 不同；这说明把 attention / DeltaNet projection 一起纳入 W8A16 storage/runtime 仍然不适合作为默认质量路径。
- `int8_ffn` 的长生成 drift 较小但仍存在，`8/128` tokens 不同；速度也只有 FP16 的约 `0.86x`。因此逐层 `QuantLinearINT8` 仍只能作为 ablation，不应作为默认部署路径。
- 当前最有价值的后续问题不再是继续打磨 `QuantLinearINT8` 的单 kernel，而是补一个 memory-aware deployment benchmark：确认 `preserve_fallback_weight=False` 后真实 peak CUDA memory 是否下降。如果显存确实下降但速度慢，INT8 可作为 memory-saving mode；如果显存也没有下降，则这条逐层 W8A16 路线应关闭。

---

## 阶段 16：DeltaNet 算子改进端到端复测

**复测目的**
- 复测 DeltaNet decode 算子改进在真实 Qwen3.5 single-user decode 中的端到端收益。
- 对比模式：
  - `torch`: 原始 PyTorch recurrent / conv / chunk 路径
  - `fla`: upstream FLA fast path
  - `triton_base`: FLA conv/chunk + Triton DeltaNet decode forward，gate 在 kernel 外计算
  - `triton_fused`: FLA conv/chunk + Triton fused-gate DeltaNet decode forward
- 当前 benchmark 不记录 peak CUDA memory，因此本阶段只覆盖 latency、patch 调用计数与 generation consistency。

**gen=16 结果**
- 命令：
  - `~/vllm-env/bin/python benchmark_qwen35_single_user.py --modes torch fla triton_base triton_fused --gen-tokens 16 --warmup-runs 1 --runs 3`
- 结果保存：
  - `artifacts/qwen35_integration/qwen35_deltanet_operator_e2e_phase16_gen16.json`

| mode | decode mean ms | end-to-end mean ms | TTFT mean ms | vs FLA decode | vs FLA end-to-end | generation |
|---|---:|---:|---:|---:|---:|---|
| `torch` | `55.073203` | `1151.332930` | `325.234877` | `0.741820x` | `0.584663x` | same vs FLA |
| `fla` | `40.854407` | `673.141803` | `60.325696` | `1.000x` | `1.000x` | reference |
| `triton_base` | `35.536595` | `584.763347` | `51.714424` | `1.149643x` | `1.151135x` | same |
| `triton_fused` | `34.593565` | `573.527744` | `54.624270` | `1.180983x` | `1.173687x` | same |

**gen=128 结果**
- 命令：
  - `~/vllm-env/bin/python benchmark_qwen35_single_user.py --modes torch fla triton_base triton_fused --gen-tokens 128 --warmup-runs 1 --runs 3`
- 结果保存：
  - `artifacts/qwen35_integration/qwen35_deltanet_operator_e2e_phase16_gen128.json`

| mode | decode mean ms | end-to-end mean ms | TTFT mean ms | vs FLA decode | vs FLA end-to-end | generation vs FLA |
|---|---:|---:|---:|---:|---:|---|
| `torch` | `45.012939` | `5963.489211` | `246.845914` | `0.799713x` | `0.775314x` | `4/128` diff |
| `fla` | `35.997413` | `4623.576840` | `51.905389` | `1.000x` | `1.000x` | reference |
| `triton_base` | `36.893180` | `4741.518102` | `56.084278` | `0.975720x` | `0.975126x` | same |
| `triton_fused` | `35.093908` | `4512.314977` | `55.388692` | `1.025745x` | `1.024657x` | same |

**Patch 覆盖**
- `gen=16` 下 `triton_base` / `triton_fused` 均记录 `patch_stats = {calls: 1080, batch_items: 1080, tokens: 1080}`。
- `gen=128` 下 `triton_base` / `triton_fused` 均记录 `patch_stats = {calls: 9144, batch_items: 9144, tokens: 9144}`。
- 这说明端到端 decode 中 DeltaNet patch 确实覆盖到了所有目标 linear attention decode 调用。

**结论**
- `triton_fused` 是当前 DeltaNet 算子端到端最优路径。
- 短生成 `gen=16` 下，`triton_fused` 相对 FLA 达到 `1.180983x` decode speedup 和 `1.173687x` end-to-end speedup；相对 torch 原始路径达到 `1.592007x` decode speedup 和 `2.007458x` end-to-end speedup。
- 长生成 `gen=128` 下，`triton_fused` 相对 FLA 收益收敛为小幅正收益：`1.025745x` decode speedup 和 `1.024657x` end-to-end speedup；相对 torch 仍有 `1.282643x` decode speedup 和 `1.321603x` end-to-end speedup。
- `triton_base` 在长生成下略慢于 FLA，而 `triton_fused` 仍快于 FLA，说明 gate fusion 是端到端可见收益的关键。
- `triton_fused` 与 FLA 在 `gen=16` 和 `gen=128` 都保持 generation 一致；torch 与 FLA/Triton 在 `gen=128` 有 `4/128` token diff，应继续以 FLA 作为更合适的 fast-path reference。
- 这轮结果确认：DeltaNet Triton fused-gate 算子改进在端到端中是有效的，但长生成收益较小；后续继续优化单个 DeltaNet kernel 的边际价值低于 StaticCache/compile 这类系统层优化。

---

## 阶段 17：Head-grouped projection+conv 与 low-rank DeltaNet 融合

**目的**
- 继续追踪阶段 16 暴露的端到端瓶颈：单独优化 DeltaNet recurrent kernel 后，projection、kernel launch、Python wrapper 和中间 tensor 边界仍然占据实际 decode 开销。
- 在 `triton_lowrank_beta_gate_packed` 路径上，把 head-grouped projection+conv pack 与 grouped-QK low-rank DeltaNet decode 融合成一个 Triton kernel，消除中间 `q_raw/k_raw/v/z/a/b` 写回和第二次 kernel launch。
- 融合范围只覆盖 projection+conv 到 DeltaNet core output；`norm` 和 `out_proj` 继续保留在后续路径，因为它们需要跨 value head 的信息，尤其 `out_proj` 是跨所有 value head 的 dense projection。

**实现内容**
- 新增 fused wrapper：
  - `triton_kernels/qwen35_projection_pack.py::qwen35_grouped_projection_conv_lowrank_deltanet`
- 新增 fused Triton kernel：
  - `qwen35_grouped_projection_conv_lowrank_deltanet_kernel`
- `triton_lowrank_beta_gate_packed` runtime 已从两段式：
  - `qwen35_grouped_projection_conv_pack`
  - `deltanet_decode_step_lowrank_beta_gate_grouped_qk`
  切换为单次 fused call。
- 端到端 `generate()` 复测前发现真实 Qwen cache 中 `conv_state` 不是 contiguous tensor，而是 shape `(1, 8192, 4)`、stride `(204800, 1, 8192)`。fused kernel 已改为 stride-aware 读取和写回 `conv_state`，避免 `.contiguous()` 破坏 cache in-place 更新语义。
- fused kernel 的 ownership 仍按 head group 切分：每个 group 负责一个 raw Q/K head 和两个 value heads。当前第一版明确支持 Qwen3.5 的 `QK_REPEAT=2`，避免在第一版里为泛化 repeat factor 增加额外分支。
- 为了与“两算子路径先写中间 tensor 再读回”的语义严格一致，fused kernel 在内部对 `q/k/v/a/b` 做 hidden dtype round-trip；真实 Qwen-shaped synthetic smoke 中 core output、`z`、`conv_state`、recurrent state 均达到 `max_abs=0.0`。

**验证**
- 单元测试：
  - `/home/haozhong/vllm-env/bin/python -m unittest triton_kernels.test_deltanet_decode triton_kernels.test_qwen35_projection_pack triton_kernels.test_qwen35_integration -v`
  - 结果：`Ran 30 tests ... OK`
- 语法与 diff 检查：
  - `/home/haozhong/vllm-env/bin/python -m py_compile triton_kernels/qwen35_projection_pack.py triton_kernels/qwen35_integration.py triton_kernels/test_qwen35_projection_pack.py triton_kernels/test_qwen35_integration.py`
  - `git diff --check`
  - 结果：均通过。
- 真实 Qwen3.5 synthetic shape smoke：
  - `B=1, d_model=2560, Hk=16, Hv=32, K=V=128, rank=8`
  - 对比：fused path vs `projection_pack + grouped_lowrank_decode`
  - `core_max_abs = 0.0`
  - `z_max_abs = 0.0`
  - `conv_state_max_abs = 0.0`
  - `state_max_abs = 0.0`

**小规模 microbenchmark：与项目自研算子对比**

下面结果使用 synthetic decode workload，均包含 projection+conv 前处理。`triton_fused_scalar` 是原 scalar gate path，`triton_lowrank_beta_gate` / `triton_lowrank_grouped_qk` / `fused_pack_lowrank_grouped` 是 low-rank beta path，因此速度可以横向参考，但数学路径并非完全相同。

`B=1, d_model=256, Hk=4, Hv=8, K=V=32, rank=4`：

| mode | wall-clock median |
|---|---:|
| `torch_scalar_native` | `2684.349 us` |
| `torch_lowrank_native` | `3881.887 us` |
| `triton_fused_scalar` | `173.474 us` |
| `triton_lowrank_beta_gate` | `198.864 us` |
| `triton_lowrank_grouped_qk` | `186.252 us` |
| `fused_pack_lowrank_grouped` | `83.641 us` |

相对 `fused_pack_lowrank_grouped`：
- vs native PyTorch scalar：`32.094x`
- vs native PyTorch lowrank：`46.411x`
- vs `triton_fused_scalar`：`2.074x`
- vs `triton_lowrank_beta_gate`：`2.378x`
- vs `triton_lowrank_grouped_qk`：`2.227x`

`B=1, d_model=512, Hk=4, Hv=8, K=V=64, rank=4`：

| mode | wall-clock median |
|---|---:|
| `torch_scalar_native` | `2495.882 us` |
| `torch_lowrank_native` | `3698.343 us` |
| `triton_fused_scalar` | `152.262 us` |
| `triton_lowrank_beta_gate` | `147.312 us` |
| `triton_lowrank_grouped_qk` | `119.259 us` |
| `fused_pack_lowrank_grouped` | `89.557 us` |

相对 `fused_pack_lowrank_grouped`：
- vs native PyTorch scalar：`27.869x`
- vs native PyTorch lowrank：`41.296x`
- vs `triton_fused_scalar`：`1.700x`
- vs `triton_lowrank_beta_gate`：`1.645x`
- vs `triton_lowrank_grouped_qk`：`1.332x`

**小规模 microbenchmark：与 cuBLAS/PyTorch projection 对比**

`cublas_projection_only` 使用 PyTorch `F.linear` / GEMM 风格 projection+conv reference；`cublas_proj+...` 表示 projection 使用 cuBLAS/PyTorch path，后接对应 Triton decode kernel。

`B=1, d_model=256, Hk=4, Hv=8, K=V=32, rank=4`：

| mode | wall-clock median |
|---|---:|
| `cublas_projection_only` | `189.513 us` |
| `triton_projection_only` | `71.278 us` |
| `cublas_proj+triton_fused_scalar` | `301.898 us` |
| `cublas_proj+triton_lowrank` | `296.171 us` |
| `cublas_proj+grouped_lowrank` | `254.064 us` |
| `triton_proj+grouped_lowrank` | `116.155 us` |
| `cublas_full_lowrank_reference` | `2837.618 us` |
| `fused_pack_lowrank_grouped` | `75.854 us` |

`B=1, d_model=512, Hk=4, Hv=8, K=V=64, rank=4`：

| mode | wall-clock median |
|---|---:|
| `cublas_projection_only` | `277.126 us` |
| `triton_projection_only` | `105.278 us` |
| `cublas_proj+triton_fused_scalar` | `576.044 us` |
| `cublas_proj+triton_lowrank` | `292.424 us` |
| `cublas_proj+grouped_lowrank` | `249.945 us` |
| `triton_proj+grouped_lowrank` | `114.163 us` |
| `cublas_full_lowrank_reference` | `2915.374 us` |
| `fused_pack_lowrank_grouped` | `86.175 us` |

**纯 GPU kernel 口径的限制**
- `triton.testing.do_bench` 显示：fused packed 在小尺寸下不总是比两算子 Triton path 的 GPU kernel body 更快。
- `K=V=32` 时，`triton_lowrank_grouped_qk` 约 `12.288 us`，fused packed 约 `14.336 us`。
- `K=V=64` 时，`triton_lowrank_grouped_qk` 约 `19.456 us`，fused packed 约 `33.792 us`。
- 这说明 fused path 的收益主要来自减少 kernel launch、Python/wrapper 边界和中间 tensor materialization；它也牺牲了一部分 head/value-head 并行度。因此该路径目前更适合写成“端到端调用开销优化”，而不是“单 kernel body 全面更快”。

**完整 generate() round-robin 复测**

命令：

```bash
/home/haozhong/vllm-env/bin/python benchmark_qwen35_single_user_round_robin.py \
  --modes fla triton_fused triton_lowrank_beta_gate triton_lowrank_beta_gate_packed \
  --gen-tokens 16 --warmup-cycles 1 --runs 3

/home/haozhong/vllm-env/bin/python benchmark_qwen35_single_user_round_robin.py \
  --modes fla triton_fused triton_lowrank_beta_gate triton_lowrank_beta_gate_packed \
  --gen-tokens 128 --warmup-cycles 1 --runs 3
```

结果保存：
- `artifacts/qwen35_integration/qwen35_phase17_fused_packed_round_robin_gen16.json`
- `artifacts/qwen35_integration/qwen35_phase17_fused_packed_round_robin_gen128.json`

`gen=16`：

| mode | decode mean ms | end-to-end mean ms | TTFT mean ms | vs FLA decode | vs FLA end-to-end | generation |
|---|---:|---:|---:|---:|---:|---|
| `fla` | `34.540115` | `564.776215` | `46.674489` | `1.000x` | `1.000x` | reference |
| `triton_fused` | `30.782594` | `508.512889` | `46.773982` | `1.122066x` | `1.110643x` | same |
| `triton_lowrank_beta_gate` | `33.316382` | `545.572249` | `45.826526` | `1.036731x` | `1.035200x` | same |
| `triton_lowrank_beta_gate_packed` | `27.763459` | `471.697383` | `55.245498` | `1.244085x` | `1.197327x` | same |

`gen=16` patch 覆盖：
- `triton_fused` / `triton_lowrank_beta_gate` / `triton_lowrank_beta_gate_packed` 均为 `patch_stats = {calls: 1080, batch_items: 1080, tokens: 1080}`。

`gen=128`：

| mode | decode mean ms | end-to-end mean ms | TTFT mean ms | vs FLA decode | vs FLA end-to-end | generation |
|---|---:|---:|---:|---:|---:|---|
| `fla` | `32.534206` | `4178.137406` | `46.293271` | `1.000x` | `1.000x` | reference |
| `triton_fused` | `30.431763` | `3911.159108` | `46.325237` | `1.069087x` | `1.068261x` | same |
| `triton_lowrank_beta_gate` | `33.314277` | `4278.034601` | `47.121465` | `0.976584x` | `0.976649x` | same |
| `triton_lowrank_beta_gate_packed` | `26.995851` | `3475.201142` | `46.728076` | `1.205156x` | `1.202272x` | `4/128` diff |

`gen=128` patch 覆盖：
- `triton_fused` / `triton_lowrank_beta_gate` / `triton_lowrank_beta_gate_packed` 均为 `patch_stats = {calls: 9144, batch_items: 9144, tokens: 9144}`。

**当前结论**
- 在当前项目的自研算子集合里，`triton_lowrank_beta_gate_packed` / `fused_pack_lowrank_grouped` 已经从小规模 microbenchmark 正信号推进到完整 `generate()` round-robin 正结果。
- 它是当前项目自研 DeltaNet decode operator path 中端到端最快的版本：`gen=16` 相对 FLA decode `1.244x`、end-to-end `1.197x`；`gen=128` 相对 FLA decode `1.205x`、end-to-end `1.202x`。
- 但它还不是“质量完全闭环的默认部署路径”：`gen=128` 出现 `4/128` token diff。该差异可能来自 low-rank beta gate 路径与原 scalar gate / FLA fast path 的数学差异，也可能来自数值精度和融合顺序，需要进一步定位。
- 报告里建议表述为：这是“项目内自研算子级当前最快路径”和“端到端最快候选”，但默认可落地配置仍需把 generation consistency / token drift 解释清楚后再定。

**还缺的工作**
1. 端到端 profiler 分解：
   - fused packed kernel
   - norm / out_proj
   - remaining projection or wrapper overhead
   - Python dispatch / launch overhead
2. `gen=128` 的 `4/128` token diff 定位：
   - 对比 `triton_lowrank_beta_gate` 与 `triton_lowrank_beta_gate_packed`，确认是否是 fused packed 数值路径导致，还是 low-rank beta gate 本身与 scalar gate / FLA reference 不完全一致。
   - 如果报告需要“质量保持”表述，必须补一个 strict same-generation 或可接受误差说明。
3. 如果继续追求默认部署路径，需要把 `triton_lowrank_beta_gate_packed` 与当前系统级最佳 `fp16_static_compiled_attn_only_deltanet` 做组合或对照，确认 fused operator 收益是否能叠加到 StaticCache/compile 主路径。

---

## 阶段 18：清理退役 DeltaNet 实验算子

**目标**
- 目录里阶段性算子过多，容易把报告主线、历史 ablation 和当前可维护路径混在一起。
- 按当前结论收敛 Qwen3.5 DeltaNet runtime：保留旧的 `triton_base` / `triton_fused` 对照路径，以及新的 `triton_lowrank_beta_gate_packed` 融合算子；删除未进入主线的实验算子入口。

**保留**
- `deltanet_decode_step`：老的 base DeltaNet Triton decode kernel。
- `deltanet_decode_step_fused_gates`：老的 fused-gate DeltaNet Triton decode kernel，也是 `triton_fused` 的核心。
- `qwen35_grouped_projection_conv_lowrank_deltanet`：当前新的 head-grouped projection+conv+low-rank DeltaNet fused operator。
- `deltanet_decode_lowrank_beta_gate_reference`：仅作为 PyTorch correctness reference，服务 fused packed 算子的测试。

**删除 / 退役**
- runtime mode 中移除：
  - `triton_fused_custom_op`
  - `triton_vector_gates`
  - `triton_lowrank_beta_gate`
- kernel / wrapper 中移除：
  - content/vector gate kernel 与 wrapper
  - 独立 low-rank beta gate kernel
  - standalone grouped-QK low-rank decode kernel
  - standalone projection pack Triton kernel
  - DeltaNet `torch.library.custom_op` wrapper
- benchmark 默认 modes 收窄到当前仍支持的主线 mode，避免默认 benchmark 跑到已退役分支。

**验证**
```bash
/home/haozhong/vllm-env/bin/python -m py_compile \
  triton_kernels/deltanet_decode.py \
  triton_kernels/qwen35_projection_pack.py \
  triton_kernels/qwen35_integration.py \
  benchmark_deltanet_decode.py \
  benchmark_qwen35_single_user.py \
  benchmark_qwen35_single_user_round_robin.py \
  benchmark_qwen35_compiled_deployment.py \
  triton_kernels/test_deltanet_decode.py \
  triton_kernels/test_qwen35_projection_pack.py \
  triton_kernels/test_qwen35_integration.py \
  triton_kernels/test_benchmark_qwen35_compiled_deployment.py
```

```bash
/home/haozhong/vllm-env/bin/python -m unittest \
  triton_kernels.test_deltanet_decode \
  triton_kernels.test_qwen35_projection_pack \
  triton_kernels.test_qwen35_integration \
  triton_kernels.test_benchmark_qwen35_compiled_deployment -v
```

结果：`Ran 32 tests ... OK`。

---

## 阶段 19：GPTQ W4A16 mixed quantization 路线打通

**目标**
- 将部署压缩路线从早期自研 INT8/W8A16 ablation，推进到 vLLM 官方更接近生产路径的 GPTQ W4A16。
- 保留当前自研 DeltaNet fused packed 算子的 dense FP16 假设：linear-attention / DeltaNet 路径不量化，优先量化 MLP 与 full-attention dense Linear。

**新增**
- 新增 `quantize_gptq_w4a16.py`
  - 默认模型：`models/Qwen3.5-4B`
  - 默认输出：`models/quantized/qwen35-4b-gptq-w4a16`
  - 默认配置：GPTQ W4A16、group size `128`、`actorder=weight`、`dampening_frac=0.01`
  - 默认保持 FP16：
    - `lm_head`
    - 所有 `model.language_model.layers.*.linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}`
  - 这对应 32 层中 24 个 linear-attention 层，共 `1 + 24*5 = 121` 个 ignore modules。
- 新增 `triton_kernels/test_gptq_w4a16_quantization.py`
  - 覆盖 W4A16 参数生成
  - 覆盖 Qwen3.5 ConditionalGeneration / CausalLM module prefix 推导
  - 覆盖 mixed GPTQ artifact metadata

**环境兼容性处理**
- 当前主 `vllm-env` 需要 Transformers 5.x 才有 `transformers.models.qwen3_5`，而 `llmcompressor 0.10.0.1` 的 declared dependency 仍要求 `transformers<=4.57.6`。
- 直接安装 `llmcompressor` 会把主环境降级到 Transformers 4.57.6，导致 `qwen3_5` import 失败；因此最终采用隔离 PYTHONPATH 方式：

```bash
/home/haozhong/vllm-env/bin/python -m venv --system-site-packages .venvs/gptq-w4a16
.venvs/gptq-w4a16/bin/pip install --no-deps \
  llmcompressor \
  compressed-tensors==0.14.0.1 \
  accelerate==1.12.0 \
  datasets==4.6.0 \
  dill==0.4.0 \
  multiprocess==0.70.18
```

运行时使用：

```bash
PYTHONPATH=/home/haozhong/ECE9483/.venvs/gptq-w4a16/lib/python3.12/site-packages \
  /home/haozhong/vllm-env/bin/python quantize_gptq_w4a16.py ...
```

**Smoke 结果**
- 已跑 1-sample / 128-token GPTQ smoke：

```bash
PYTHONPATH=/home/haozhong/ECE9483/.venvs/gptq-w4a16/lib/python3.12/site-packages \
  /home/haozhong/vllm-env/bin/python quantize_gptq_w4a16.py \
  --max-calib-samples 1 \
  --max-seq-length 128 \
  --output-dir /tmp/qwen35-gptq-w4a16-smoke
```

结果：
- 完整加载 723 个权重 shard entries。
- GPTQ pipeline trace 到 `33` 个 sequential subgraphs。
- 日志显示被量化模块为 MLP 与 full-attention 的 `q_proj/k_proj/v_proj/o_proj`；DeltaNet `linear_attn.*` 投影没有进入量化列表。
- 产物写出：`/tmp/qwen35-gptq-w4a16-smoke/artifact.json`
- metadata 中记录：
  - `quantization = gptq-w4a16`
  - `backend = compressed-tensors`
  - `group_size = 128`
  - `actorder = weight`
  - `calibration_count = 1`
  - `ignored_modules_count = 121`

**注意**
- 这个 smoke 只证明 GPTQ W4A16 mixed quantization 工具链在本机可跑通，不能作为最终质量产物。
- 正式报告或部署产物应至少跑默认 `128` calibration samples，并补：
  1. vLLM 加载 smoke；
  2. 与 FP16 / FLA / fused packed 的同场 generate latency；
  3. token drift / 输出一致性记录。

---

## 阶段 20：DeltaNet fused packed 与 StaticCache + compile 合并测试

**目标**
- 验证阶段 17 的 `triton_lowrank_beta_gate_packed` 是否能叠加到阶段 8 的 StaticCache + attn-only compile 主路径。
- 不考虑量化，只比较 FP16 路径：
  - baseline：`fp16_static_compiled_attn_only_deltanet`
  - candidate：`fp16_static_compiled_attn_only_deltanet_packed`

**新增 benchmark mode**
- 在 `benchmark_qwen35_compiled_deployment.py` 中新增：
  - `fp16_static_compiled_attn_only_deltanet_packed`
- 该 mode 与当前 baseline 保持相同：
  - `use_static_cache=True`
  - `compile_decode=True`
  - `compile_after_prefill=True`
  - `compile_mlp=False`
  - `compile_self_attn=True`
  - `compile_mode=reduce-overhead`
- 唯一差异：
  - baseline DeltaNet：`triton_fused`
  - candidate DeltaNet：`triton_lowrank_beta_gate_packed`

**Benchmark**
短生成：

```bash
/home/haozhong/vllm-env/bin/python benchmark_qwen35_compiled_deployment.py \
  --modes fp16_static_compiled_attn_only_deltanet fp16_static_compiled_attn_only_deltanet_packed \
  --gen-tokens 16 --warmup-runs 1 --runs 3 \
  --output artifacts/qwen35_integration/qwen35_static_compile_packed_deltanet_gen16.json
```

结果：

| mode | decode mean | e2e mean | generation |
|---|---:|---:|---|
| old fused + StaticCache + compile | `28.329953 ms` | `475.942580 ms` | reference |
| packed + StaticCache + compile | `29.268867 ms` | `501.442333 ms` | same |

相对 old fused：
- decode speedup：`0.967921x`
- end-to-end speedup：`0.949147x`
- token diff：`0/16`

长生成：

```bash
/home/haozhong/vllm-env/bin/python benchmark_qwen35_compiled_deployment.py \
  --modes fp16_static_compiled_attn_only_deltanet fp16_static_compiled_attn_only_deltanet_packed \
  --gen-tokens 128 --warmup-runs 1 --runs 3 \
  --output artifacts/qwen35_integration/qwen35_static_compile_packed_deltanet_gen128.json
```

结果：

| mode | decode mean | e2e mean | generation |
|---|---:|---:|---|
| old fused + StaticCache + compile | `27.829245 ms` | `3584.916436 ms` | reference |
| packed + StaticCache + compile | `24.097891 ms` | `3112.092519 ms` | drift |

相对 old fused：
- decode speedup：`1.154842x`
- end-to-end speedup：`1.151931x`
- token diff：`4/128`，diff rate `0.03125`

**结论**
- fused packed DeltaNet 可以与 StaticCache + attn-only compile 合并运行；二者没有 runtime 结构冲突。
- 长生成端到端速度信号为正，约 `1.15x`。
- 但它复现了阶段 17 的质量风险：长生成存在小幅 token drift，因此当前不能直接替代 `fp16_static_compiled_attn_only_deltanet` 成为稳健默认路径。
- 报告中可以把它列为“最快自研算子组合 / speed-oriented candidate”，但 stable default 仍应保留 old fused + StaticCache + compile，除非后续把 low-rank beta gate 与 reference 完成严格数值对齐。

---

## 综合诊断：核心问题与路线优先级

**核心问题的诊断**
- 项目前半程反复遇到的不是某个 kernel 写得还不够快，而是 Amdahl 定律。阶段 2B 的 profiling 已经给出结构性信号：`aten::mm` 和底层 `gemvx` 合计占 decode CUDA 自时间约 80%，而 DeltaNet recurrent 更新只占很小一部分。因此 DeltaNet decode kernel 即使在微基准中达到 `3.261x`，端到端也只能传导成约 `1.06x` 到 `1.08x`。
- INT8 GEMV 与 DeltaNet Triton 的端到端折损，本质上是同一类问题：优化覆盖的是局部很快的一段，而真实瓶颈更多来自 HF eager decode 中高频 Python dispatch、kernel launch overhead，以及大量线性投影。`QuantLinearINT8` 逐层替换被 dispatch 开销吃掉，DeltaNet micro speedup 被 Amdahl 上限吃掉，都说明优化对象没有覆盖最慢的主路径。
- 阶段 6 的 compile spike 曾在单步路径上给出约 `1.675x` 的信号，这比继续打磨单个自定义 kernel 更直接地暴露了真正的杠杆：减少 eager dispatch 和 launch overhead。它在最初多步实验中失败，并不是因为 compile 方向错，而是 `DynamicCache` 让 KV cache 的形状和地址随 decode 步数变化，CUDA graph 无法稳定 replay；同时把 DeltaNet Triton 和 MLP 都拉入 compile 粒度，又放大了重编译与 capture 交互问题。
- 阶段 8 对这个判断做了闭环验证：StaticCache 稳定了 KV cache，`attn-only compile` 保住了 cudagraph 收益；DeltaNet Triton 继续作为 eager 侧专项优化存在，但不再被强行纳入 graph capture。当前最佳路径因此从“单个 DeltaNet kernel 更快”升级为“StaticCache + attn-only compile + DeltaNet Triton”。

**下一步建议（按性价比排序）**
1. Static KV Cache + `torch.compile` 的 FP16 纯净路径已经在阶段 8 落地，并从原先的 Future Work 升级为当前最佳交付路径。后续如果继续打磨这条线，应优先减少 full-attention 层 warmup recompiles，而不是回到 `DynamicCache` compile。
2. MTP self-speculation 已经在阶段 9 形成实验路径。下一步应在真实 GPU 上测接受率、target pass 数量和端到端速度，并决定是否值得继续做 StaticCache/cudagraph 融合。
3. Phase 10A 的 W8A16 static-compile closure 已完成。它说明全量 W8A16 fused FFN 在当前 static-compile 主路径上没有端到端收益，长生成仍会产生 token drift，并且当前 runtime 因保留 FP16 fallback 不具备显存压缩收益。因此它只能作为 ablation，不能取代 `fp16_static_compiled_attn_only_deltanet`。
4. DeltaNet Triton kernel 已经在阶段 9 注册为 `torch.library.custom_op` 实验路径；当前结果显示它不应提升为默认路径，但这条经验可以迁移到后续 AWQ 外部 kernel 的 compile 兼容性验证。

---

## 最终排行榜

| 配置 | 短生成 | 长生成 | 生成是否一致 | 状态 |
|---|---:|---:|---|---|
| FP16 基线 | 1.00x | 1.00x | 是 | 参考 |
| FP16 + DeltaNet Triton | 约 `1.07x` | 约 `1.06x` 到 `1.08x` | 是 | 已被阶段 8 超越 |
| FP16 + StaticCache + attn-only compile | `1.110x` | `1.009x` | 是 | 稳定有效 |
| FP16 + StaticCache + attn-only compile + DeltaNet Triton | `1.165x` | `1.135x` | 是 | 当前最佳 |
| `back_8` FFN + StaticCache + attn-only compile + DeltaNet Triton | `1.121x` | `1.119x` | 短输出是，长输出否 | 仅保留为 ablation |
| 全量 W8A16 FFN + StaticCache + attn-only compile + DeltaNet Triton | 相对当前最佳 `0.860x` | 相对当前最佳 `0.815x` | 短输出是，长输出否；长生成差 `4/128` tokens | 仅保留为 Phase 10A closure |
| 原始 DynamicCache compile 探索 | `0.799x` | `0.848x` | 不适用 | 已淘汰 |

**当前可落地的交付结果**
- 当前最强且最稳健的可落地配置是 `fp16_static_compiled_attn_only_deltanet`。
- `back8_static_compiled_attn_only_deltanet` 只适合作为短生成消融项保留，不应提升为默认最佳配置。
- `w8a16_static_compiled_attn_only_deltanet` 完成了量化 closure，但由于短生成负收益、长生成漂移以及当前实现没有显存压缩收益，不应作为默认部署路径。

---

## 仓库映射

这次补充的代码注释，会将仓库中的文件回映到上述各阶段：

- 阶段 1 辅助工具：`phase1_utils.py`、`quantize.py`、`cpu_reference.py`、`tests/test_phase1_modules.py`
- 阶段 2 DeltaNet kernel 与集成：`triton_kernels/deltanet_decode.py`、`triton_kernels/qwen35_integration.py`、`benchmark_deltanet_decode.py`、`benchmark_qwen35_single_user*.py`
- 阶段 3 INT8 kernel 与路由：`triton_kernels/int8_gemv.py`、`triton_kernels/qwen35_int8_integration.py`、`benchmark_int8_gemv.py`、`benchmark_qwen35_int8*.py`
- 阶段 4 FFN 融合：`triton_kernels/int8_fused_ffn.py`、`triton_kernels/qwen35_fused_ffn_integration.py`、`benchmark_qwen35_fused_ffn_deployment.py`
- 阶段 5 子集分析：`benchmark_qwen35_back8_deltanet_combo.py`
- 阶段 6 compile 探索：`triton_kernels/qwen35_compile_integration.py`、`benchmark_qwen35_compiled_deployment.py`
- 阶段 8 StaticCache + compile 恢复路径：`triton_kernels/qwen35_static_cache_integration.py`、`triton_kernels/test_qwen35_static_cache_integration.py`、`triton_kernels/test_benchmark_qwen35_compiled_deployment.py`
- 阶段 9 MTP 与 custom-op 实验路径：`triton_kernels/qwen35_mtp_self_speculation.py`、`benchmark_qwen35_mtp_self_speculation.py`、`triton_kernels/test_qwen35_mtp_self_speculation.py`、`triton_kernels/test_benchmark_qwen35_mtp_self_speculation.py`
- 阶段 10 W8A16 closure：`benchmark_qwen35_compiled_deployment.py`、`triton_kernels/qwen35_single_user_benchmark.py`、`triton_kernels/test_benchmark_qwen35_compiled_deployment.py`、`triton_kernels/test_qwen35_single_user_benchmark.py`
- Profiling 与诊断：`profile_qwen35_single_user.py`、`triton_kernels/qwen35_profiler.py`、`deltanet_diagnostics.py`

---

## 后续工作

1. 不要把 `fp16_static_compiled_attn_only_deltanet_custom_op` 提升为默认路径；它目前只保留为 compiler 兼容性 ablation。
2. 运行 `benchmark_qwen35_mtp_self_speculation.py`，优先观察 MTP acceptance rate、target_passes/token、端到端速度和生成一致性。
3. 如果 MTP 接受率足够高，再把 MTP self-speculation 与 StaticCache/cudagraph 路径进一步融合；如果接受率不足，则停止在调度层继续投入。
4. 不要把 `w8a16_static_compiled_attn_only_deltanet` 提升为默认路径；它目前只保留为量化 closure ablation。
5. 如果后续继续打磨当前最佳路径，可以针对 `self.layer_idx` 触发的 full-attention warmup 重编译做更细粒度优化，减少前几步抖动。
6. 如果继续做真正部署压缩，需要先移除或下放当前 fused FFN 中的 FP16 fallback 权重；否则 W8A16 只是在 FP16 模型旁边额外挂 INT8 buffer，不会降低显存。
7. 若进入 W4A16 AWQ probe，第一版仍应关闭 compile，并维护 skip list：`lm_head`、`in_proj_a`、`in_proj_b`、所有 `out_features < 1024` 的 projection，以及必要的 tiny DeltaNet projection。

---

## 阶段 21：GSM8K 50 题最终评估

**目标**
- 用真实长 decode 任务评估最终路径，而不是用 MMLU 这类短 decode/prefill 主导任务掩盖优化收益。
- 固定 GSM8K test split 随机 50 题，`seed=42`，8-shot CoT，greedy，`max_new_tokens=256`。
- 对比 6 条路径：
  - `torch`
  - `fla`
  - `fp16_eager`（在该脚本中明确映射到 eager `triton_fused`，对应 Phase 2）
  - `fp16_eager_packed`（只用 packed DeltaNet，不启用 StaticCache/compile）
  - `fp16_static_compiled_attn_only_deltanet`
  - `fp16_static_compiled_attn_only_deltanet_packed`

**新增脚本**
- [benchmark_qwen35_gsm8k_final.py](/home/haozhong/ECE9483/benchmark_qwen35_gsm8k_final.py)
- 新增 helper 单测：
  - [test_benchmark_qwen35_gsm8k_final.py](/home/haozhong/ECE9483/triton_kernels/test_benchmark_qwen35_gsm8k_final.py)

**命令**

```bash
/home/haozhong/vllm-env/bin/python -m py_compile benchmark_qwen35_gsm8k_final.py
/home/haozhong/vllm-env/bin/python -m unittest triton_kernels.test_benchmark_qwen35_gsm8k_final -v
/home/haozhong/vllm-env/bin/python benchmark_qwen35_gsm8k_final.py --smoke
/home/haozhong/vllm-env/bin/python benchmark_qwen35_gsm8k_final.py
/home/haozhong/vllm-env/bin/python benchmark_qwen35_gsm8k_final.py --modes fp16_eager_packed
```

**正式结果**

| Mode | Accuracy | Decode mean | Decode median | Total mean | Total median | Tokens/sec |
|---|---:|---:|---:|---:|---:|---:|
| `fp16_eager_packed` | `0.900` | `29.705 ms/tok` | `29.051 ms/tok` | `7699.681 ms` | `7539.690 ms` | `33.66` |
| `fp16_static_compiled_attn_only_deltanet_packed` | `0.900` | `33.964 ms/tok` | `33.614 ms/tok` | `8791.400 ms` | `8712.322 ms` | `29.44` |
| `fp16_eager` | `0.900` | `36.047 ms/tok` | `35.468 ms/tok` | `9316.091 ms` | `9174.480 ms` | `27.74` |
| `fla` | `0.900` | `38.831 ms/tok` | `37.935 ms/tok` | `10026.938 ms` | `9802.934 ms` | `25.75` |
| `fp16_static_compiled_attn_only_deltanet` | `0.880` | `39.141 ms/tok` | `38.553 ms/tok` | `10111.485 ms` | `9964.135 ms` | `25.55` |
| `torch` | `0.900` | `46.996 ms/tok` | `47.320 ms/tok` | `12447.569 ms` | `12554.312 ms` | `21.28` |

相对 `fp16_eager_packed`：
- 对 `torch`：decode `1.582x`，total `1.617x`
- 对 `fla`：decode `1.307x`，total `1.302x`
- 对 `fp16_eager`：decode `1.213x`，total `1.210x`
- 对 old static fused：decode `1.318x`，total `1.313x`
- 对 static packed：decode `1.143x`，total `1.142x`

**质量观察**
- `torch`、`fla`、`fp16_eager`、`fp16_eager_packed`、`static packed` 的错题集合一致：`[1309, 65, 689, 541, 255]`。
- old static fused 额外错了 `qid=407`，该题 `torch` 预测 `2000`，old static fused 预测 `4000`。
- `fp16_eager_packed` 与 torch 的最终答案没有 mismatch；raw generation 有少量文本差异，但最终数字答案保持一致。
- 本次 50 题都生成满 `256` token 并被截断，因此该 accuracy 是在固定 `max_new_tokens=256` 口径下的结果。

**Artifacts**
- [gsm8k_50_question_ids.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/gsm8k_50_question_ids.json)
- [qwen35_gsm8k_final_summary.md](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_gsm8k_final_summary.md)
- [qwen35_gsm8k_final_torch.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_gsm8k_final_torch.json)
- [qwen35_gsm8k_final_fla.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_gsm8k_final_fla.json)
- [qwen35_gsm8k_final_fp16_eager.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_gsm8k_final_fp16_eager.json)
- [qwen35_gsm8k_final_fp16_eager_packed.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_gsm8k_final_fp16_eager_packed.json)
- [qwen35_gsm8k_final_fp16_static_compiled_attn_only_deltanet.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_gsm8k_final_fp16_static_compiled_attn_only_deltanet.json)
- [qwen35_gsm8k_final_fp16_static_compiled_attn_only_deltanet_packed.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_gsm8k_final_fp16_static_compiled_attn_only_deltanet_packed.json)

**结论**
- 在这组真实 GSM8K 长 decode 任务上，`fp16_eager_packed` 是本项目自研路径里速度最好的版本，并且没有观察到最终答案 accuracy 下降。
- 这个结果说明 packed DeltaNet 与 StaticCache+compile 在当前实现里不是正向叠加；去掉 StaticCache/compile 后反而更快。最终报告里应把 `fp16_eager_packed` 作为 best self-developed kernel path，把 static packed 作为 ablation。
- `fp16_static_compiled_attn_only_deltanet` 仍是保守稳定路径的历史默认，但本次 50 题结果不支持它比 packed 更稳。
- 需要在报告中注明：GSM8K 输出全部触达 256-token 截断，因此质量结论绑定当前 generation config；若要报告绝对 GSM8K 能力，后续应额外跑更长 `max_new_tokens` 或加入任务级 stopping。

---

## 阶段 22：vLLM Qwen3Next shim 对比

**目标**
- 增加一个 vLLM baseline，用同一批 GSM8K 50 题衡量当前最佳自研路径距离主流推理引擎还有多远。
- 直接加载 `models/Qwen3.5-4B` 时，vLLM 0.15.1 不支持 `Qwen3_5ForConditionalGeneration`；因此本阶段只做 text-only `Qwen3NextForCausalLM` shim 对比。

**新增脚本**
- [prepare_qwen35_vllm_shim.py](/home/haozhong/ECE9483/scripts/prepare_qwen35_vllm_shim.py)
  - 去掉视觉/MTP 权重，只保留 language model。
  - 将 split projection 权重重新打包成 vLLM 期望的 head-grouped layout：每组 `[q_h,k_h,v_2h,v_2h+1,z_2h,z_2h+1]` 与 `[b_2h,b_2h+1,a_2h,a_2h+1]`。
  - 移除 text-only vLLM 不支持的 M-RoPE visual 字段。
- [vllm_qwen35_smoke.py](/home/haozhong/ECE9483/scripts/vllm_qwen35_smoke.py)
- [benchmark_qwen35_gsm8k_vllm.py](/home/haozhong/ECE9483/benchmark_qwen35_gsm8k_vllm.py)

**兼容性说明**
- vLLM 原生路径当前不能直接跑该 checkpoint；本结果不是“官方原生 Qwen3.5 support”，而是本项目构造的 Qwen3Next text shim。
- shim 需要 monkeypatch dense Qwen3.5 的 MoE metadata，否则 vLLM 的 Qwen3Next 类会假设存在 MoE layer。
- vLLM 默认 compile/cudagraph 在当前 shim 上不能初始化：4-token smoke probe 于 profile/torch.compile 阶段报 `expected mat1 and mat2 to have the same dtype, but got: float != c10::Half`，因此正式结果使用 `enforce_eager=True`。
- vLLM 这里只拿到端到端 wall-clock，不拆 prefill/decode；对比表统一使用每条路径 JSON 中的 total latency 计算 e2e output tok/s。

**命令**

```bash
/home/haozhong/vllm-env/bin/python -m py_compile \
  benchmark_qwen35_gsm8k_vllm.py \
  scripts/prepare_qwen35_vllm_shim.py \
  scripts/vllm_qwen35_smoke.py

VLLM_WORKER_MULTIPROC_METHOD=spawn \
  /home/haozhong/vllm-env/bin/python scripts/vllm_qwen35_smoke.py

VLLM_WORKER_MULTIPROC_METHOD=spawn \
  /home/haozhong/vllm-env/bin/python benchmark_qwen35_gsm8k_vllm.py --smoke

VLLM_WORKER_MULTIPROC_METHOD=spawn \
  /home/haozhong/vllm-env/bin/python benchmark_qwen35_gsm8k_vllm.py
```

**正式结果（按 e2e output tokens/sec 排序）**

| Mode | Accuracy | E2E output tok/s | E2E ms/output tok | Total mean | Total median |
|---|---:|---:|---:|---:|---:|
| `fp16_eager_packed` | `0.900` | `33.25` | `30.077` | `7699.681 ms` | `7539.690 ms` |
| `fp16_static_compiled_attn_only_deltanet_packed` | `0.900` | `29.12` | `34.341` | `8791.400 ms` | `8712.322 ms` |
| `vllm_qwen3next_shim_eager` | `0.900` | `28.62` | `34.940` | `8944.569 ms` | `8900.377 ms` |
| `fp16_eager` | `0.900` | `27.48` | `36.391` | `9316.091 ms` | `9174.480 ms` |
| `fla` | `0.900` | `25.53` | `39.168` | `10026.938 ms` | `9802.934 ms` |
| `fp16_static_compiled_attn_only_deltanet` | `0.880` | `25.32` | `39.498` | `10111.485 ms` | `9964.135 ms` |
| `torch` | `0.900` | `20.57` | `48.623` | `12447.569 ms` | `12554.312 ms` |

**结论**
- 当前最佳自研路径 `fp16_eager_packed` 的 e2e output tok/s 是 `33.25`，比 vLLM shim eager 的 `28.62` 高 `1.16x`。
- vLLM shim eager 比 torch native 高 `1.39x`，比 FLA 高 `1.12x`；它很强，但在这个 Qwen3.5 DeltaNet 单用户长 decode 口径下没有超过项目最佳 packed kernel。
- 由于 vLLM 结果依赖 shim、禁用 compile/cudagraph，报告中应写成“vLLM compatibility baseline”，不能写成 vLLM 对该模型的完整官方最优性能。

**Artifacts**
- [vllm_qwen3next_shim](/home/haozhong/ECE9483/artifacts/qwen35_integration/vllm_qwen3next_shim)
- [qwen35_gsm8k_final_vllm_qwen3next_shim_eager.json](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_gsm8k_final_vllm_qwen3next_shim_eager.json)
- [qwen35_gsm8k_vllm_compare.md](/home/haozhong/ECE9483/artifacts/qwen35_integration/qwen35_gsm8k_vllm_compare.md)
