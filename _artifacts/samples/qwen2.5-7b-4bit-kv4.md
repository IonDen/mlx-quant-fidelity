# KV-fidelity: `mlx-community/Qwen2.5-7B-Instruct-4bit` @ 4-bit (group 64)

**Verdict:** bad · **mode:** stress (quantize_start=0)

| metric | value |
|---|---|
| KL mean | 9.3584 nats |
| KL median | 8.8432 nats |
| KL p99 | 19.3773 nats |
| KL max | 27.0106 nats |
| flip rate | 0.9922 |
| perplexity Δ | +82664.9420 (10.793 → 82675.735) |

Measured on **wikitext-2-raw/test**, 51100 positions across 100 chunks of length 512 (tokenizer `mlx-community/Qwen2.5-7B-Instruct-4bit`). Fidelity is corpus- and context-length-specific; short-prose temp-0 drift under-predicts long-context/code degradation.

_mlx 0.31.2, mlx-lm 0.31.3, model rev `c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed`, peak 6.50 GB, on Apple M1 Max, 32 GB._

**Drift by position depth** (stress mode; positions pooled across chunks):

| positions | KL mean | KL p99 |
|---|---|---|
| 0-62 | 7.8575 | 17.6311 |
| 63-126 | 9.1598 | 18.7703 |
| 127-190 | 9.5194 | 19.9929 |
| 191-254 | 9.5590 | 19.3937 |
| 255-318 | 9.6563 | 19.7247 |
| 319-382 | 9.6897 | 19.7881 |
| 383-446 | 9.6453 | 19.5661 |
| 447-510 | 9.7569 | 19.5720 |
