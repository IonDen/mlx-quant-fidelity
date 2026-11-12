# Quant comparison (kv) vs `mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx`

| target | cost | KL mean | KL p99 | flip | bundled KL | resident +/token | verdict | frontier |
|---|---|---|---|---|---|---|---|---|
| `4:64` | 77.8 KB | 0.2760 | 2.4097 | 0.2237 | 0.2763 | 0 B | bad | ✓ |
| `affine:8:4` | 119.2 KB | 0.0199 | 0.2136 | 0.0606 | — | 276.5 KB | bad | ✓ |
| `8:64` | 146.9 KB | 0.0055 | 0.0618 | 0.0304 | 0.0055 | 0 B | marginal | ✓ |


> Note: affine: per-tensor asymmetric bits have no shipped runtime — upstream mlx-lm's QuantizedKVCache is symmetric; the only known implementation is an idle mlx-lm fork. The probe dequantizes on fetch and rides standard SDPA, so this drift is the quantizer round-trip only, for a hypothetical deployment.

_ranked on quantizer-only drift; stock rows carry their bundled deployment drift alongside._
