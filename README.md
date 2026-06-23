# mlx-quant-fidelity

<p align="center">
  <img src="docs/assets/mlx-quant-fidelity-logo.png" alt="mlx-quant-fidelity" width="100%">
</p>

[![PyPI version](https://img.shields.io/pypi/v/mlx-quant-fidelity.svg)](https://pypi.org/project/mlx-quant-fidelity/)
[![Python versions](https://img.shields.io/pypi/pyversions/mlx-quant-fidelity.svg)](https://pypi.org/project/mlx-quant-fidelity/)
[![License: Apache-2.0](https://img.shields.io/pypi/l/mlx-quant-fidelity.svg)](https://github.com/IonDen/mlx-quant-fidelity/blob/main/LICENSE)

Measure how much quality a quantization costs on Apple Silicon. `mlx-quant-fidelity` scores a quantized model against a higher-precision reference on the same corpus and reports the drift as numbers you can act on: KL divergence, top-token flip rate, perplexity delta. It measures both **KV-cache quantization** and **weight quantization**. No more choosing a bit-width by file size.

The CUDA/GGUF world has had this for years: llama.cpp's `--kl-divergence-base`, EleutherAI's `lm-evaluation-harness`. MLX had nothing. This is the MLX version, and it covers the KV-cache and attention angle those tools skip.

## Install

```bash
pip install mlx-quant-fidelity
```

Apple Silicon (MLX), Python 3.11+.

## Use it

```bash
mlx-quant-fidelity kv mlx-community/Llama-3.2-3B-Instruct-4bit --kv-bits 8
```

Prints a Markdown report. Add `--format json` for JSON, `--kv-bits 4`, `--kv-group-size 64`, or `--max-chunks N` to bound the corpus.

```python
from mlx_quant_fidelity import measure_kv_fidelity

report = measure_kv_fidelity("mlx-community/Llama-3.2-3B-Instruct-4bit", kv_bits=8)
print(report.kl.mean, report.flip_rate, report.verdict)
```

Or measure **weight** quantization — a quantized repo against a higher-precision reference:

```bash
mlx-quant-fidelity weights mlx-community/Llama-3.2-3B-Instruct-4bit --reference mlx-community/Llama-3.2-3B-Instruct-bf16
```

```python
from mlx_quant_fidelity import measure_weight_fidelity

# measure_weight_fidelity(quantized_repo, reference_repo)
report = measure_weight_fidelity(
    "mlx-community/Llama-3.2-3B-Instruct-4bit",  # quantized
    "mlx-community/Llama-3.2-3B-Instruct-bf16",  # reference
)
print(report.kl.mean, report.flip_rate, report.verdict)
```

## What a report looks like

```markdown
# KV-fidelity: `mlx-community/Llama-3.2-3B-Instruct-4bit` @ 8-bit (group 64)

**Verdict:** good · **mode:** stress (quantize_start=0)

| metric | value |
|---|---|
| KL mean | 0.0002 nats |
| KL median | 0.0001 nats |
| KL p99 | 0.0015 nats |
| KL max | 0.1129 nats |
| flip rate | 0.0065 |
| perplexity Δ | +0.0054 (17.722 → 17.728) |

Measured on **wikitext-2-raw/test**, 51100 positions across 100 chunks of length 512 ...
```

## How much does KV quantization cost?

M1 Max, WikiText-2 test (100 chunks of 512 tokens), stress mode (quantize from token 0). Reproduce any row with `mlx-quant-fidelity kv <model> --kv-bits <bits> --max-chunks 100`; the full committed reports are under [`_artifacts/samples/`](_artifacts/samples).

| Model | KV bits | KL mean (nats) | flip rate | verdict |
|---|---|---|---|---|
| Llama-3.2-1B | 4 | 0.148 | 0.20 | bad |
| Llama-3.2-1B | 8 | 0.0004 | 0.013 | marginal |
| Llama-3.2-3B | 4 | 0.051 | 0.11 | bad |
| Llama-3.2-3B | 8 | 0.0002 | 0.007 | good |
| Qwen2.5-7B | 4 | 9.36 | 0.99 | bad |
| Qwen2.5-7B | 8 | 0.009 | 0.032 | marginal |

8-bit KV is near-lossless on all three models. 4-bit is another matter, and Qwen2.5-7B at 4-bit in stress mode falls apart: nearly every token flips. That is the attention sink at work: stress mode quantizes the cache from token 0, including the first tokens attention leans on most, and Qwen2.5 does not tolerate it. mlx-lm's own default keeps the first 5000 tokens full-precision for exactly this reason. Run the tool first and you see it coming.

## How much does weight quantization cost?

Same corpus and recipe, but the comparison is now a quantized model repo against a higher-precision reference repo. Reproduce any row with `mlx-quant-fidelity weights <quant> --reference <reference> --max-chunks 100`; the committed reports are under [`_artifacts/samples/weights/`](_artifacts/samples/weights).

| Model | quant | reference | KL mean (nats) | flip rate | perplexity Δ | verdict |
|---|---|---|---|---|---|---|
| Llama-3.2-1B | 4-bit | bf16 | 0.158 | 0.21 | +3.5 | marginal |
| Llama-3.2-1B | 8-bit | bf16 | 0.001 | 0.023 | −0.01 | good |
| Llama-3.2-3B | 4-bit | bf16 | 0.085 | 0.15 | +1.4 | marginal |
| Llama-3.2-3B | 8-bit | bf16 | 0.0009 | 0.021 | 0.00 | good |
| Qwen2.5-7B | 4-bit | 8-bit | 0.109 | 0.16 | +0.9 | marginal |

8-bit weights are near-lossless: about 2% of top tokens flip and perplexity barely moves. 4-bit is a real trade: 15 to 21% of top tokens flip and perplexity climbs a point or more, worst on the small 1B model. The Qwen row compares 4-bit against 8-bit rather than bf16, so its drift is relative to an already-quantized reference, not full precision; the report records that the reference is 8-bit and says so in plain text. The verdict tiers are provisional, anchored to these q8 and q4 reference points on short prose rather than to downstream task accuracy.

Unlike the KV probe, both runs use standard attention, so the drift is the deployed quantized model's weight-quant cost with no quantized-attention kernel folded in. It does still include the quantized-matmul kernel's numerics, which is exactly what you run when you load the model.

## Comparing quantizations

`compare` ranks a set of quantizations on a memory-normalized Pareto frontier: quality (mean KL divergence) on one axis, memory cost on the other. It identifies any configuration that is both worse quality and more expensive than another option on the list — those are dominated and you would never choose them.

```bash
# rank weight quantizations against a bf16 reference
mlx-quant-fidelity compare weights q4 q6 q8 --reference fp16

# rank KV configs on a single model
mlx-quant-fidelity compare kv <model> --configs 4:32,4:64,8:64
```

Add `--max-kld 0.05` to get the cheapest configuration whose mean KLD stays under a threshold, or `--min-tier good` to get the cheapest one that passes the good-tier verdict. [docs/ranking-principles.md](docs/ranking-principles.md) explains how each axis is computed, what Pareto domination means in practice, and where the ranking has limits.

## How it works

Teacher-forced scoring, not generation. For each fixed-length corpus chunk the model runs twice on the *same* tokens — once with a full-precision KV cache, once with a quantized one — and the two next-token distributions are compared position by position. Generation would let the runs diverge in their own inputs the moment quantization changed a sampled token, turning the measurement into trajectory drift instead of cache cost. Logits collapse to per-position scalars inside the chunk loop and are released before the next chunk, so a long corpus never holds full distributions in memory.

Two modes:

- **stress** (`--quantize-start 0`, the default): quantize from token 0. The harsh, apples-to-apples quantizer test.
- **deployment** (`quantize_start > 0`): what mlx-lm users actually run, with the first N tokens kept full-precision. Not available yet — see the [roadmap](ROADMAP.md).

A run that returns exactly zero drift raises instead of reporting a silent "perfect fidelity." That almost always means quantization never engaged, not that it was free.

The weight probe works the same way with two models instead of two caches: a quantized repo and a reference repo, scored on the same corpus tokens. A compatibility gate refuses a mismatched pair before loading, and a memory pre-flight refuses a pair too large for the device rather than risking a kernel panic.

See [docs/measurement-principles.md](docs/measurement-principles.md) for the zero-probability policy, the exact-zero guard, and how perplexity delta relates to mean KLD.

## What the numbers don't say

- A fidelity number is **corpus- and context-length-specific**. WikiText-2 at temperature 0 measures short-prose distributional drift; the paper this builds on, *Accuracy Is Not All You Need*, shows that under-predicts task-specific and long-context degradation. Every report records the corpus and the token count so the number is never read as a bare score.
- Perplexity delta is reported for continuity with llama.cpp. It is related to but distinct from mean KLD — it scores the realized next token and can diverge from full-vocabulary drift — so it is not independent corroboration.
- The measured drift bundles the quantizer's error with the quantized-attention kernel's numerics. That is the real end-to-end cost; a quantizer-only control is on the roadmap.

## Status

0.3.1, released on PyPI as `mlx-quant-fidelity` — hardens `compare` error handling and adds a methodology document (`docs/measurement-principles.md`). 0.3.0 added the `compare` command for memory-normalized Pareto ranking of KV-cache and weight quantizations. Downstream-task accuracy, deployment mode, and more are on the [roadmap](ROADMAP.md).

## License

[Apache-2.0](LICENSE).
