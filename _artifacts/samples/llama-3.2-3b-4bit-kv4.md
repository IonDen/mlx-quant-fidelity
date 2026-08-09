# KV-fidelity: `mlx-community/Llama-3.2-3B-Instruct-4bit` @ 4-bit (group 64)

**Verdict:** bad · **mode:** stress (quantize_start=0)

| metric | value |
|---|---|
| KL mean | 0.0514 nats |
| KL median | 0.0291 nats |
| KL p99 | 0.4019 nats |
| KL max | 6.4716 nats |
| flip rate | 0.1125 |
| perplexity Δ | +0.9105 (17.722 → 18.633) |

Measured on **wikitext-2-raw/test**, 51100 positions across 100 chunks of length 512 (tokenizer `mlx-community/Llama-3.2-3B-Instruct-4bit`). Fidelity is corpus- and context-length-specific; short-prose temp-0 drift under-predicts long-context/code degradation.

_mlx 0.31.2, mlx-lm 0.31.3, model rev `None`, peak 3.72 GB, on Apple M1 Max, 32 GB._

**Drift by position depth** (stress mode; positions pooled across chunks):

| positions | KL mean | KL p99 |
|---|---|---|
| 0-62 | 0.0612 | 0.4694 |
| 63-126 | 0.0526 | 0.3939 |
| 127-190 | 0.0518 | 0.3908 |
| 191-254 | 0.0505 | 0.4119 |
| 255-318 | 0.0510 | 0.4768 |
| 319-382 | 0.0502 | 0.3835 |
| 383-446 | 0.0468 | 0.3206 |
| 447-510 | 0.0472 | 0.3208 |
