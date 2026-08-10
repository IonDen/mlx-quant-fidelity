# KV-fidelity: `mlx-community/Llama-3.2-1B-Instruct-4bit` @ 8-bit (group 64)

**Verdict:** marginal · **mode:** stress (quantize_start=0)

| metric | value |
|---|---|
| KL mean | 0.0004 nats |
| KL median | 0.0003 nats |
| KL p99 | 0.0029 nats |
| KL max | 0.0466 nats |
| flip rate | 0.0126 |
| perplexity Δ | +0.0152 (22.941 → 22.956) |

Measured on **wikitext-2-raw/test**, 51100 positions across 100 chunks of length 512 (tokenizer `mlx-community/Llama-3.2-1B-Instruct-4bit`). Fidelity is corpus- and context-length-specific; short-prose temp-0 drift under-predicts long-context/code degradation.

_mlx 0.31.2, mlx-lm 0.31.3, model rev `08231374eeacb049a0eade7922910865b8fce912`, peak 2.56 GB, on Apple M1 Max, 32 GB._

**Drift by position depth** (stress mode; positions pooled across chunks):

| positions | KL mean | KL p99 |
|---|---|---|
| 0-62 | 0.0005 | 0.0032 |
| 63-126 | 0.0004 | 0.0031 |
| 127-190 | 0.0004 | 0.0027 |
| 191-254 | 0.0004 | 0.0027 |
| 255-318 | 0.0004 | 0.0028 |
| 319-382 | 0.0004 | 0.0028 |
| 383-446 | 0.0004 | 0.0029 |
| 447-510 | 0.0004 | 0.0026 |
