# KV-fidelity: `mlx-community/Qwen2.5-7B-Instruct-4bit` @ 8-bit (group 64)

**Verdict:** marginal · **mode:** stress (quantize_start=0)

| metric | value |
|---|---|
| KL mean | 0.0094 nats |
| KL median | 0.0014 nats |
| KL p99 | 0.1426 nats |
| KL max | 13.0970 nats |
| flip rate | 0.0317 |
| perplexity Δ | +0.0635 (10.793 → 10.856) |

Measured on **wikitext-2-raw/test**, 51100 positions across 100 chunks of length 512 (tokenizer `mlx-community/Qwen2.5-7B-Instruct-4bit`). Fidelity is corpus- and context-length-specific; short-prose temp-0 drift under-predicts long-context/code degradation.

_mlx 0.31.2, mlx-lm 0.31.3, model rev `None`, peak 6.35 GB, on Apple M1 Max, 32 GB._

**Drift by position depth** (stress mode; positions pooled across chunks):

| positions | KL mean | KL p99 |
|---|---|---|
| 0-62 | 0.0200 | 0.2909 |
| 63-126 | 0.0100 | 0.1413 |
| 127-190 | 0.0080 | 0.0841 |
| 191-254 | 0.0089 | 0.1019 |
| 255-318 | 0.0055 | 0.0803 |
| 319-382 | 0.0061 | 0.0865 |
| 383-446 | 0.0088 | 0.1636 |
| 447-510 | 0.0080 | 0.1753 |
