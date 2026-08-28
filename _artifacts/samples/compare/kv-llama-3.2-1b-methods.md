# Quant comparison (kv) vs `mlx-community/Llama-3.2-1B-Instruct-4bit`

| target | cost | KL mean | KL p99 | flip | bundled KL | resident +/token | verdict | frontier |
|---|---|---|---|---|---|---|---|---|
| `turboquant:3` | 8.2 KB | 0.4229 | 2.3559 | 0.3259 | — | 65.5 KB | bad | ✓ |
| `4:64` | 9.2 KB | 0.1485 | 0.9571 | 0.2056 | 0.1477 | 0 B | bad | ✗ dominated by `turboquant:4` |
| `turboquant:4` | 9.2 KB | 0.0825 | 0.5663 | 0.1582 | — | 65.5 KB | bad | ✓ |
| `affine:8:2` | 11.3 KB | 0.2180 | 1.3105 | 0.2391 | — | 32.8 KB | bad | ✗ dominated by `turboquant:4` |
| `affine:8:4` | 13.3 KB | 0.0120 | 0.0818 | 0.0628 | — | 32.8 KB | bad | ✓ |
| `8:64` | 17.4 KB | 0.0004 | 0.0028 | 0.0121 | 0.0004 | 0 B | marginal | ✓ |
| `turboquant-vonly:3` | 36.9 KB | 0.0297 | 0.2025 | 0.0974 | — | 32.8 KB | bad | ✗ dominated by `affine:8:4` |
| `turboquant-vonly:4` | 37.4 KB | 0.0078 | 0.0520 | 0.0508 | — | 32.8 KB | bad | ✗ dominated by `8:64` |


> Note: turboquant: the quantized run dequantizes on fetch and rides standard SDPA in prefill (the port's fused kernel is decode-only and not exercised), so drift measures the quantizer round-trip only, while stock bundles quantizer + quantized-attention numerics; uniform-bit cache at the port's default seed — its V-only configuration is measured separately (`turboquant-vonly`); its layer-adaptive configuration is not — and its `make_adaptive_cache` silently ignores the documented `k_bits`/`v_bits` parameters at the pinned commit; resident memory in this path is the stored bytes plus two full-precision working copies — roughly 2.3x an fp16 cache on Llama-3.2-1B geometry, derived from the port's retained dequantization buffers — so peak memory does not show the compression.

> Note: turboquant-vonly: K stays fp16 and rides standard SDPA; only V is quantized-and-dequantized on fetch, so drift measures the V quantizer round-trip only, at the port's default seed.

> Note: turboquant-vonly: the pinned port stores an unused fp16 copy of V in its inner KVCache, so stored bytes EXCEED a plain fp16 cache — the V-only value at this commit is V-compression quality, not memory.

> Note: affine: per-tensor asymmetric bits have no shipped runtime — upstream mlx-lm's QuantizedKVCache is symmetric; the only known implementation is an idle mlx-lm fork. The probe dequantizes on fetch and rides standard SDPA, so this drift is the quantizer round-trip only, for a hypothetical deployment.

_ranked on quantizer-only drift; stock rows carry their bundled deployment drift alongside._
