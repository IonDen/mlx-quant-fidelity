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

_mlx 0.31.2, mlx-lm 0.31.3, model rev `c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed`, peak 6.35 GB._