# Weight-fidelity: `mlx-community/Qwen2.5-7B-Instruct-4bit` @ 4-bit (group 64) vs `mlx-community/Qwen2.5-7B-Instruct-8bit`

**Verdict:** marginal (provisional tiers — WikiText-2, not validated against downstream accuracy)

> Reference is itself 8-bit — drift is relative to an already-quantized reference, **not full precision**.

| metric | value |
|---|---|
| KL mean | 0.1089 nats |
| KL median | 0.0711 nats |
| KL p99 | 0.8205 nats |
| KL max | 6.3589 nats |
| flip rate | 0.1639 |
| perplexity Δ | +0.9011 (9.892 → 10.793) |

Measured on **wikitext-2-raw/test**, 51100 positions across 100 chunks of length 512 (tokenizer `mlx-community/Qwen2.5-7B-Instruct-8bit`). Fidelity is corpus- and context-length-specific; short-prose temp-0 drift under-predicts long-context/code degradation.

_mlx 0.31.2, mlx-lm 0.31.3, quant 4284346255 B, reference 8091987725 B, peak 14.40 GB._

> Note: Assumes both repos share a tokenizer; only vocab_size (and bos/eos ids) were checked — a divergent token map would be measured as quant drift.

> Note: reference is itself 8-bit, not full precision; drift is relative to it.