# KV-fidelity: `mlx-community/Llama-3.2-1B-Instruct-4bit` @ 4-bit (group 64)

**Verdict:** bad · **mode:** deployment (quantize_start=256)

| metric | value |
|---|---|
| KL mean | 0.1078 nats |
| KL median | 0.0690 nats |
| KL p99 | 0.6229 nats |
| KL max | 2.2559 nats |
| flip rate | 0.1608 |
| perplexity Δ | +1.8508 (14.477 → 16.328) |

Measured on **wikitext-2-raw/test**, 1020 positions across 4 chunks of length 512 (tokenizer `mlx-community/Llama-3.2-1B-Instruct-4bit`). Fidelity is corpus- and context-length-specific; short-prose temp-0 drift under-predicts long-context/code degradation.

_mlx 0.31.2, mlx-lm 0.31.3, model rev `None`, peak 1.93 GB._

> **Deployment mode:** metrics and the 1020 positions cover only the post-boundary quantized region; the first 256 positions per window are full-precision and excluded. Per-token drift ≈ stress (see docs/measurement-principles.md).
