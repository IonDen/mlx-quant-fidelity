# Weight-fidelity: `mlx-community/Llama-3.2-3B-Instruct-4bit` @ 4-bit (group 64) vs `mlx-community/Llama-3.2-3B-Instruct-bf16`

**Verdict:** marginal (provisional tiers — WikiText-2, not validated against downstream accuracy)

| metric | value |
|---|---|
| KL mean | 0.0851 nats |
| KL median | 0.0567 nats |
| KL p99 | 0.5624 nats |
| KL max | 6.3282 nats |
| flip rate | 0.1506 |
| perplexity Δ | +1.4479 (16.274 → 17.722) |

Measured on **wikitext-2-raw/test**, 51100 positions across 100 chunks of length 512 (tokenizer `mlx-community/Llama-3.2-3B-Instruct-bf16`). Fidelity is corpus- and context-length-specific; short-prose temp-0 drift under-predicts long-context/code degradation.

_mlx 0.31.2, mlx-lm 0.31.3, quant 1807496278 B, reference 6425528971 B, peak 10.07 GB._

> Note: Assumes both repos share a tokenizer; only vocab_size (and bos/eos ids) were checked — a divergent token map would be measured as quant drift.