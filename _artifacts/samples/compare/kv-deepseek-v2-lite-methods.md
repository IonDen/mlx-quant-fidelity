# Quant comparison (kv) vs `mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx`

| target | cost | KL mean | KL p99 | flip | bundled KL | resident +/token | verdict | frontier |
|---|---|---|---|---|---|---|---|---|
| `4:64` | 62.2 KB | 0.2760 | 2.4097 | 0.2237 | 0.2763 | 0 B | bad | ✓ |
| `affine:8:4` | 89.9 KB | 0.0199 | 0.2136 | 0.0606 | — | 221.2 KB | bad | ✓ |
| `8:64` | 117.5 KB | 0.0055 | 0.0618 | 0.0304 | 0.0055 | 0 B | marginal | ✓ |


> Note: measured KV bytes/token 146880 differs from the analytic 117504 for method 'stock' (ranking uses the analytic figure; a scale/bias dtype other than fp16/bf16 is the usual cause).

> Note: measured KV bytes/token 77760 differs from the analytic 62208 for method 'stock' (ranking uses the analytic figure; a scale/bias dtype other than fp16/bf16 is the usual cause).

> Note: measured KV bytes/token 119232 differs from the analytic 89856 for method 'affine' (ranking uses the analytic figure; a scale/bias dtype other than fp16/bf16 is the usual cause).

> Note: affine: per-tensor asymmetric bits have no shipped runtime — upstream mlx-lm's QuantizedKVCache is symmetric; the only known implementation is an idle mlx-lm fork. The probe dequantizes on fetch and rides standard SDPA, so this drift is the quantizer round-trip only, for a hypothetical deployment.

_ranked on quantizer-only drift; stock rows carry their bundled deployment drift alongside._
