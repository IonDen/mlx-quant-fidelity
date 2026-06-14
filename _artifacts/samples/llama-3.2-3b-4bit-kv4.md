# KV-fidelity: `mlx-community/Llama-3.2-3B-Instruct-4bit` @ 4-bit (group 64)

**Verdict:** bad · **mode:** stress (quantize_start=0)

| metric | value |
|---|---|
| KL mean | 0.0514 nats |
| KL median | 0.0291 nats |
| KL p99 | 0.4019 nats |
| KL max | 6.4716 nats |
| flip rate | 0.1125 |
| perplexity Δ | +0.9105 (17.722 → 18.633) |

Measured on **wikitext-2-raw/test**, 51100 positions across 100 chunks of length 512 (tokenizer `mlx-community/Llama-3.2-3B-Instruct-4bit`). Fidelity is corpus- and context-length-specific; short-prose temp-0 drift under-predicts long-context/code degradation.

_mlx 0.31.2, mlx-lm 0.31.3, model rev `7f0dc925e0d0afb0322d96f9255cfddf2ba5636e`, peak 3.59 GB._