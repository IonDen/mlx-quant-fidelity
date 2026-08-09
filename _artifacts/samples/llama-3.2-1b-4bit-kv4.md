# KV-fidelity: `mlx-community/Llama-3.2-1B-Instruct-4bit` @ 4-bit (group 64)

**Verdict:** bad · **mode:** stress (quantize_start=0)

| metric | value |
|---|---|
| KL mean | 0.1477 nats |
| KL median | 0.0969 nats |
| KL p99 | 0.9225 nats |
| KL max | 6.2491 nats |
| flip rate | 0.2048 |
| perplexity Δ | +4.1211 (22.941 → 27.062) |

Measured on **wikitext-2-raw/test**, 51100 positions across 100 chunks of length 512 (tokenizer `mlx-community/Llama-3.2-1B-Instruct-4bit`). Fidelity is corpus- and context-length-specific; short-prose temp-0 drift under-predicts long-context/code degradation.

_mlx 0.31.2, mlx-lm 0.31.3, model rev `None`, peak 2.55 GB, on Apple M1 Max, 32 GB._

**Drift by position depth** (stress mode; positions pooled across chunks):

| positions | KL mean | KL p99 |
|---|---|---|
| 0-62 | 0.1542 | 0.9356 |
| 63-126 | 0.1385 | 0.9023 |
| 127-190 | 0.1425 | 0.8714 |
| 191-254 | 0.1534 | 1.0120 |
| 255-318 | 0.1508 | 0.9535 |
| 319-382 | 0.1491 | 0.9284 |
| 383-446 | 0.1470 | 0.8646 |
| 447-510 | 0.1467 | 0.8652 |
