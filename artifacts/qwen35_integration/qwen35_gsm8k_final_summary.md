# GSM8K 50-Question Final Benchmark

Seed: 42
Generation: greedy, max_new_tokens=256, 8-shot CoT prompt

| Mode | Accuracy | Decode mean (ms/tok) | Decode median (ms/tok) | Total mean (ms) | Total median (ms) | Tokens/sec |
|---|---:|---:|---:|---:|---:|---:|
| fp16_eager_packed | 0.900 | 29.705 | 29.051 | 7699.681 | 7539.690 | 33.66 |
| fp16_static_compiled_attn_only_deltanet_packed | 0.900 | 33.964 | 33.614 | 8791.400 | 8712.322 | 29.44 |
| fp16_eager | 0.900 | 36.047 | 35.468 | 9316.091 | 9174.480 | 27.74 |
| fla | 0.900 | 38.831 | 37.935 | 10026.938 | 9802.934 | 25.75 |
| fp16_static_compiled_attn_only_deltanet | 0.880 | 39.141 | 38.553 | 10111.485 | 9964.135 | 25.55 |
| torch | 0.900 | 46.996 | 47.320 | 12447.569 | 12554.312 | 21.28 |
