# Weight-fidelity: `mlx-community/Llama-3.2-1B-Instruct-8bit` @ 8-bit (group 64) vs `mlx-community/Llama-3.2-1B-Instruct-bf16`

**Verdict:** good (provisional tiers — WikiText-2, not validated against downstream accuracy)

| metric | value |
|---|---|
| KL mean | 0.0010 nats |
| KL median | 0.0008 nats |
| KL p99 | 0.0052 nats |
| KL max | 0.1254 nats |
| flip rate | 0.0228 |
| perplexity Δ | -0.0134 (19.431 → 19.417) |

Measured on **wikitext-2-raw/test**, 51100 positions across 100 chunks of length 512 (tokenizer `mlx-community/Llama-3.2-1B-Instruct-bf16`). Fidelity is corpus- and context-length-specific; short-prose temp-0 drift under-predicts long-context/code degradation.

_mlx 0.31.2, mlx-lm 0.31.3, quant 1313157436 B, reference 2471645521 B, peak 5.62 GB._

> Note: Assumes both repos share a tokenizer; only vocab_size (and bos/eos ids) were checked — a divergent token map would be measured as quant drift.