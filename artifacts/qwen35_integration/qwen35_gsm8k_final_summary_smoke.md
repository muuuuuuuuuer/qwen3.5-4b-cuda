# GSM8K 3-Question Final Benchmark

Seed: 42
Generation: greedy, max_new_tokens=256, 8-shot CoT prompt

| Mode | Accuracy | Decode mean (ms/tok) | Decode median (ms/tok) | Total mean (ms) | Total median (ms) | Tokens/sec |
|---|---:|---:|---:|---:|---:|---:|
| fp16_static_compiled_attn_only_deltanet_packed | 0.667 | 29.870 | 31.416 | 7749.586 | 8132.965 | 33.48 |
| fp16_eager | 0.667 | 32.177 | 31.075 | 8326.020 | 8039.614 | 31.08 |
| fp16_static_compiled_attn_only_deltanet | 0.667 | 33.653 | 35.229 | 8705.194 | 9101.047 | 29.72 |
| fla | 0.667 | 33.957 | 34.082 | 8778.936 | 8806.621 | 29.45 |
| torch | 0.667 | 47.448 | 47.136 | 12491.075 | 12350.199 | 21.08 |
| fp8_ffn | 0.000 | 374.909 | 375.170 | 96622.297 | 96762.016 | 2.67 |
