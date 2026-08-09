# KV-fidelity: `mlx-community/Llama-3.2-1B-Instruct-4bit` @ 4-bit (group 64)

**Verdict:** bad · **mode:** stress (quantize_start=0)

| metric | value |
|---|---|
| KL mean | 0.1507 nats |
| KL median | 0.1015 nats |
| KL p99 | 0.9482 nats |
| KL max | 6.4244 nats |
| flip rate | 0.1998 |
| perplexity Δ | +3.0694 (15.441 → 18.510) |

Measured on **wikitext-2-raw/test**, 49140 positions across 12 chunks of length 4096 (tokenizer `mlx-community/Llama-3.2-1B-Instruct-4bit`). Fidelity is corpus- and context-length-specific; short-prose temp-0 drift under-predicts long-context/code degradation.

_mlx 0.31.2, mlx-lm 0.31.3, model rev `None`, peak 14.52 GB, on Apple M1 Max, 32 GB._

> Note: chunk_length=4096: paired fp32 logits peak ≈ 13.7 GiB per chunk; see docs/measurement-principles.md (Drift by position depth) for measured ceilings.

**Drift by position depth** (stress mode; positions pooled across chunks):

| positions | KL mean | KL p99 |
|---|---|---|
| 0-510 | 0.1485 | 0.9470 |
| 511-1022 | 0.1455 | 0.8659 |
| 1023-1534 | 0.1534 | 0.9329 |
| 1535-2046 | 0.1479 | 0.9568 |
| 2047-2558 | 0.1439 | 0.9048 |
| 2559-3070 | 0.1572 | 0.9835 |
| 3071-3582 | 0.1537 | 0.9757 |
| 3583-4094 | 0.1554 | 1.0237 |
