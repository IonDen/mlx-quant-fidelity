# Weight-fidelity: `mlx-community/Llama-3.2-1B-Instruct-4bit` @ 4-bit (group 64) vs `mlx-community/Llama-3.2-1B-Instruct-bf16`

**Verdict:** marginal (provisional tiers — WikiText-2, not validated against downstream accuracy)

| metric | value |
|---|---|
| KL mean | 0.1581 nats |
| KL median | 0.1147 nats |
| KL p99 | 0.9213 nats |
| KL max | 8.9108 nats |
| flip rate | 0.2138 |
| perplexity Δ | +3.5105 (19.431 → 22.941) |

Measured on **wikitext-2-raw/test**, 51100 positions across 100 chunks of length 512 (tokenizer `mlx-community/Llama-3.2-1B-Instruct-bf16`). Fidelity is corpus- and context-length-specific; short-prose temp-0 drift under-predicts long-context/code degradation.

_mlx 0.31.2, mlx-lm 0.31.3, quant 695283921 B, reference 2471645521 B, peak 5.00 GB._

> Note: Assumes both repos share a tokenizer; only vocab_size (and bos/eos ids) were checked — a divergent token map would be measured as quant drift.