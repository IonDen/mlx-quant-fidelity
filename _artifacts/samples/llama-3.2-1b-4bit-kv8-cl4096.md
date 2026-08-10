# KV-fidelity: `mlx-community/Llama-3.2-1B-Instruct-4bit` @ 8-bit (group 64)

**Verdict:** marginal · **mode:** stress (quantize_start=0)

| metric | value |
|---|---|
| KL mean | 0.0004 nats |
| KL median | 0.0003 nats |
| KL p99 | 0.0029 nats |
| KL max | 0.0619 nats |
| flip rate | 0.0118 |
| perplexity Δ | +0.0115 (15.441 → 15.452) |

Measured on **wikitext-2-raw/test**, 49140 positions across 12 chunks of length 4096 (tokenizer `mlx-community/Llama-3.2-1B-Instruct-4bit`). Fidelity is corpus- and context-length-specific; short-prose temp-0 drift under-predicts long-context/code degradation.

_mlx 0.31.2, mlx-lm 0.31.3, model rev `08231374eeacb049a0eade7922910865b8fce912`, peak 14.56 GB, on Apple M1 Max, 32 GB._

> Note: chunk_length=4096: paired fp32 logits peak ≈ 13.7 GiB per chunk; see docs/measurement-principles.md (Drift by position depth) for measured ceilings.

**Drift by position depth** (stress mode; positions pooled across chunks):

| positions | KL mean | KL p99 |
|---|---|---|
| 0-510 | 0.0004 | 0.0031 |
| 511-1022 | 0.0004 | 0.0026 |
| 1023-1534 | 0.0004 | 0.0028 |
| 1535-2046 | 0.0004 | 0.0024 |
| 2047-2558 | 0.0004 | 0.0027 |
| 2559-3070 | 0.0004 | 0.0029 |
| 3071-3582 | 0.0004 | 0.0034 |
| 3583-4094 | 0.0004 | 0.0034 |
