# KV-fidelity: `mlx-community/Llama-3.2-1B-Instruct-4bit` @ 4-bit (group 64)

**Verdict:** bad · **mode:** deployment (quantize_start=256)

| metric | value |
|---|---|
| KL mean | 0.1208 nats |
| KL median | 0.0774 nats |
| KL p99 | 0.7729 nats |
| KL max | 4.2177 nats |
| flip rate | 0.1856 |
| perplexity Δ | +2.7693 (17.818 → 20.588) |

Measured on **wikitext-2-raw/test**, 25500 positions across 100 chunks of length 512 (tokenizer `mlx-community/Llama-3.2-1B-Instruct-4bit`). Fidelity is corpus- and context-length-specific; short-prose temp-0 drift under-predicts long-context/code degradation.

_mlx 0.31.2, mlx-lm 0.31.3, model rev `None`, peak 1.93 GB, on Apple M1 Max, 32 GB._

> **Deployment mode:** metrics and the 25500 positions cover only the post-boundary region. The first 256 positions per window were computed with a full-precision cache and are excluded; at the boundary, their stored K/V entries are converted too. Compare drift with stress only in a matched measurement (see docs/measurement-principles.md).
