# mlx-quant-fidelity

<p align="center">
  <img src="https://raw.githubusercontent.com/IonDen/mlx-quant-fidelity/main/docs/assets/mlx-quant-fidelity-logo.png" alt="mlx-quant-fidelity" width="100%">
</p>

[![PyPI version](https://img.shields.io/pypi/v/mlx-quant-fidelity.svg)](https://pypi.org/project/mlx-quant-fidelity/)
[![Python versions](https://img.shields.io/pypi/pyversions/mlx-quant-fidelity.svg)](https://pypi.org/project/mlx-quant-fidelity/)
[![License: Apache-2.0](https://img.shields.io/pypi/l/mlx-quant-fidelity.svg)](https://github.com/IonDen/mlx-quant-fidelity/blob/main/LICENSE)

You install a 4-bit model. It loads, it answers, the prose reads fine. Nothing in the logs suggests otherwise.

On Qwen2.5-7B with a 4-bit KV cache active from the first token, 99% of next-token choices come out different from the same model running a full-precision cache. Quantizing from token zero is the harshest way to measure, and it is not what you get by default: mlx-lm's own generate command leaves the cache unquantized until token 5000. The text is still fluent English — that is exactly the problem. A quantization failure does not announce itself, and file size tells you nothing about it.

`mlx-quant-fidelity` runs the same text through your model twice, once quantized and once not, and reports how far apart the two ended up: KL divergence, top-token flip rate, perplexity delta. It covers both **KV-cache quantization** and **weight quantization**.

The CUDA and GGUF world has had this for years — llama.cpp's `--kl-divergence-base`, EleutherAI's `lm-evaluation-harness`. MLX had nothing, and neither of those covers the KV-cache and attention angle.

## Try it in one command

```bash
pip install mlx-quant-fidelity
mlx-quant-fidelity kv mlx-community/Llama-3.2-3B-Instruct-4bit --kv-bits 8
```

```
# KV-fidelity: `mlx-community/Llama-3.2-3B-Instruct-4bit` @ 8-bit (group 64)

**Verdict:** good · **mode:** stress (quantize_start=0)

| metric | value |
|---|---|
| KL mean | 0.0002 nats |
| KL p99 | 0.0015 nats |
| flip rate | 0.0065 |
| perplexity Δ | +0.0054 (17.722 → 17.728) |
```

That model at 8-bit KV is safe to ship on this corpus. Apple Silicon, Python 3.11+.

## Does this apply to you?

Precision gets lost in two places — on disk and in the running cache — and each needs its own command.

![Diagram: a running quantized model loses precision in two places. Weights on disk were quantized once, before download, and are measured by `mlx-quant-fidelity weights`, which scores a quantized repo against a higher-precision repo. The KV cache is quantized continuously at run time as the cache grows with every token, and is measured by `mlx-quant-fidelity kv`, which scores a full-precision cache against a quantized one. Both feed `mlx-quant-fidelity compare`, which ranks configurations by quality per byte](https://raw.githubusercontent.com/IonDen/mlx-quant-fidelity/main/docs/assets/diagrams/coverage-map.svg)

**"I'm about to run a quantized model and I want to know what I gave up."**

```bash
mlx-quant-fidelity kv mlx-community/Llama-3.2-3B-Instruct-4bit --kv-bits 4
mlx-quant-fidelity weights mlx-community/Llama-3.2-3B-Instruct-4bit --reference mlx-community/Llama-3.2-3B-Instruct-bf16
```

**"I need this model to fit in my RAM and I don't know which setting to cut."**

```bash
mlx-quant-fidelity compare kv <model> --sweep --max-kv-bytes-per-token 200
```

Builds the whole bits-by-group-size grid from the model's `config.json`, drops anything over your memory budget, and ranks what's left by quality per byte. No weights downloaded to do it.

**"I publish quantized models and I want to show they're good."**

```bash
mlx-quant-fidelity kv <model> --kv-bits 8 --format badge
```

Prints one shields.io line for your model card. Green, yellow, or red, with the bit width, corpus, chunk length, and mode baked into the message, so two badges are never confused.

## Badge output

`--format badge` replaces the whole report with one line:

```
![KV fidelity](https://img.shields.io/badge/KV_fidelity-good_%C2%B7_8--bit_%C2%B7_wikitext--2--raw%2F512_%C2%B7_stress-brightgreen)
```

Green for `good`, yellow for `marginal`, red for `bad`. Threshold values and the color map are in [docs/threshold-policy.md](docs/threshold-policy.md).

## What it found

![Chart with two panels showing mean KL divergence on a logarithmic axis, lower being better. Green, amber and red bands mark the good, marginal and bad mean-KLD thresholds. A bar's length is its mean KL divergence alone, while a bar's color is its overall verdict, which also weighs the p99 tail and the top-token flip rate, so a bar's color and the band it lands in do not always agree. Left panel, KV-cache quantization, six bars, bands at 0.01 and 0.10. At 4 bits, Llama-3.2-1B at 0.148 and Qwen2.5-7B at 9.36 are both in the red band and colored red, while Llama-3.2-3B at 0.051 falls inside the amber band but is colored red anyway, because its flip rate of 0.11 is past the marginal ceiling of 0.05. At 8 bits all three bars are short and land in the green band below 0.01, but only one of them is green: Llama-3.2-3B at 0.0002 is a good verdict, Llama-3.2-1B at 0.00043 is amber for a marginal verdict on a flip rate of 0.013 against the good ceiling of 0.01, and Qwen2.5-7B at 0.0094 is amber on both a p99 of 0.14 against a ceiling of 0.10 and a flip rate of 0.032. The Qwen bar also reaches almost to the 0.01 band boundary rather than sitting deep inside it. Right panel, weight quantization, five bars, bands at 0.01 and 0.20, where color and band agree throughout. All three 4-bit bars are amber for a marginal verdict and sit inside the amber band: Llama-3.2-1B at 0.158, Llama-3.2-3B at 0.085, and Qwen2.5-7B at 0.109, the last measured against an 8-bit reference rather than bf16. Both 8-bit bars are green for a good verdict in the green band: Llama-3.2-1B at 0.0010 and Llama-3.2-3B at 0.00091](https://raw.githubusercontent.com/IonDen/mlx-quant-fidelity/main/docs/assets/charts/fidelity-cliff.svg)

Eight-bit costs little everywhere we measured it. Four-bit is a real trade, and on one checkpoint it collapses.

Bar length is mean KL divergence and nothing else. Bar color is the overall verdict, which also weighs the p99 tail and the top-token flip rate, so two of the three 8-bit KV bars sit in the green band and are still amber, and the 4-bit Llama-3.2-3B bar is red inside the amber band. [docs/threshold-policy.md](docs/threshold-policy.md) lists the ceilings. The right panel gets its own section further down.

KV cache, M1 Max, WikiText-2 test (100 chunks of 512 tokens), stress mode (quantize from token 0). Reproduce any row with `mlx-quant-fidelity kv <model> --kv-bits <bits> --max-chunks 100`; the full committed reports are under [`_artifacts/samples/`](_artifacts/samples).

| Model | KV bits | KL mean (nats) | flip rate | verdict |
|---|---|---|---|---|
| Llama-3.2-1B | 4 | 0.148 | 0.20 | bad |
| Llama-3.2-1B | 8 | 0.0004 | 0.013 | marginal |
| Llama-3.2-3B | 4 | 0.051 | 0.11 | bad |
| Llama-3.2-3B | 8 | 0.0002 | 0.007 | good |
| Qwen2.5-7B | 4 | 9.36 | 0.99 | bad |
| Qwen2.5-7B | 8 | 0.009 | 0.032 | marginal |

8-bit KV costs little on all three models, though only Llama-3.2-3B clears the good tier outright. 4-bit is another matter, and Qwen2.5-7B at 4-bit in stress mode falls apart: nearly every token flips. This measurement establishes a checkpoint-specific failure, not its cause. mlx-lm's own generate command leaves the cache unquantized until token 5000, so those positions are computed while attention uses a full-precision cache. At the boundary, however, mlx-lm converts the entire stored prefix too. The Python API defaults differently: pass `kv_bits` to `mlx_lm.generate` and quantization starts at token 0 unless you also set `quantized_kv_start`. Run the tool first and you see the fidelity risk before deployment.

## Does drift change with position depth?

`--chunk-length 4096` widens the window and adds a table breaking mean and p99 KLD down by position depth within a chunk. Llama-3.2-1B at 4-bit KV, M1 Max, WikiText-2 test (12 chunks of 4096 tokens, the same ~50k-token corpus coverage as the 512-token samples above):

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

On this model and corpus, drift at position 4000 looks about the same as drift at position 60 — quantization cost isn't building up across the window at these lengths. That's a narrower claim than it might sound: 4096 tokens is short next to the context lengths where other work has found KV-quantization drift growing with depth. [docs/measurement-principles.md](docs/measurement-principles.md#drift-by-position-depth) covers the measured memory cost of longer windows and why the comparison to longer-context findings elsewhere isn't apples to apples. The full report, including the 8-bit KV counterpart, is under [`_artifacts/samples/`](_artifacts/samples) (`llama-3.2-1b-4bit-kv4-cl4096.md`, `llama-3.2-1b-4bit-kv8-cl4096.md`).

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

![Diagram explaining domination: configuration A drifts less, with a lower mean KL divergence, and costs less, with fewer cache bytes per token. Configuration B loses on both counts, so B is dominated — worse on quality and more expensive, meaning no memory budget would make it the right pick. Ranking reports domination so options can be discarded outright instead of weighed by hand](https://raw.githubusercontent.com/IonDen/mlx-quant-fidelity/main/docs/assets/diagrams/pareto-domination.svg)

```bash
# rank weight quantizations against a bf16 reference
mlx-quant-fidelity compare weights q4 q6 q8 --reference fp16

# rank KV configs on a single model
mlx-quant-fidelity compare kv <model> --configs 4:32,4:64,8:64

# or auto-generate the grid from the model's config.json instead of listing configs by hand
mlx-quant-fidelity compare kv <model> --sweep --max-kv-bytes-per-token 200
```

Add `--max-kld 0.05` to get the cheapest configuration whose mean KLD stays under a threshold, or `--min-tier good` to get the cheapest one that passes the good-tier verdict. `--sweep` builds the (bits × group-size) grid from the model's config alone, no weight download needed, and drops any combination that would crash the upstream KV cache implementation; `--max-kv-bytes-per-token` narrows that grid to configurations under a memory budget. Either way, skipped configurations are listed in the report rather than silently dropped. [docs/ranking-principles.md](docs/ranking-principles.md) explains how each axis is computed, what Pareto domination means in practice, and where the ranking has limits.

## How it works

Teacher-forced scoring, not generation. For each fixed-length corpus chunk the model runs twice on the *same* tokens — once with a full-precision KV cache, once with a quantized one — and the two next-token distributions are compared position by position. Generation would let the runs diverge in their own inputs the moment quantization changed a sampled token, turning the measurement into trajectory drift instead of cache cost. Logits collapse to per-position scalars inside the chunk loop and are released before the next chunk, so a long corpus never holds full distributions in memory.

![Diagram of the paired teacher-forced scoring loop: a fixed corpus chunk feeds a reference run with a full-precision KV cache and a quantized run with a quantized KV cache; both produce vocabulary-wide fp32 logits that are reduced inside the loop to per-position scalars for KL divergence, top-token flip and target negative log-likelihood; the scalars are evaluated, the per-chunk caches are dropped and the cache pool cleared, and the loop advances, so no vocabulary-wide tensor outlives its chunk](https://raw.githubusercontent.com/IonDen/mlx-quant-fidelity/main/docs/papers/diagrams/paired-scoring-chunk-loop.svg)

Every report records which of two modes produced it.

![Diagram comparing the two measurement modes. In stress mode with quantize-start 0, the default, every position runs against a quantized cache from the very first token, and every position is scored. In deployment mode with quantize-start 5000, positions 0 through 4999 are computed against a full-precision cache; at position 5000 the entire stored cache converts to quantized and no full-precision prefix is kept; positions 5000 and beyond run against the quantized cache, and only the post-boundary positions are scored](https://raw.githubusercontent.com/IonDen/mlx-quant-fidelity/main/docs/assets/diagrams/stress-vs-deployment.svg)

Stress mode (`--quantize-start 0`, the default) quantizes from token 0 — the harsh, apples-to-apples quantizer test. Deployment mode (`--quantize-start N`) computes the first N positions with a full-precision cache, then converts the entire stored cache and scores only the post-boundary region. That matches mlx-lm's `--quantized-kv-start` conversion behavior, which does not preserve a full-precision prefix in storage. [docs/measurement-principles.md](docs/measurement-principles.md) explains why deployment and stress drift need a matched comparison and why neither is a long-context deployment average.

A run that returns exactly zero drift raises instead of reporting a silent "perfect fidelity." That almost always means quantization never engaged, not that it was free.

The weight probe works the same way with two models instead of two caches: a quantized repo and a reference repo, scored on the same corpus tokens. A compatibility gate refuses a mismatched pair before loading, and a memory pre-flight refuses a pair too large for the device rather than risking a kernel panic.

See [docs/measurement-principles.md](docs/measurement-principles.md) for the zero-probability policy, the exact-zero guard, and how perplexity delta relates to mean KLD.

## What the numbers don't say

- A fidelity number is **corpus- and context-length-specific**. WikiText-2 at temperature 0 measures short-prose distributional drift; the paper this builds on, *Accuracy Is Not All You Need*, shows that under-predicts task-specific and long-context degradation. Every report records the corpus and the token count so the number is never read as a bare score.
- Perplexity delta is reported for continuity with llama.cpp. It is related to but distinct from mean KLD — it scores the realized next token and can diverge from full-vocabulary drift — so it is not independent corroboration.
- The measured drift bundles the quantizer's error with the quantized-attention kernel's numerics. That is the real end-to-end cost; a quantizer-only control is on the roadmap.

## Research notes

- [Low-bit KV caches on MLX: what exists and what is missing](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/papers/low-bit-kv-caches-on-mlx-what-exists-and-what-is-missing.md) — surveys mlx-lm's shipped cache, the measured 8-bit and 4-bit fidelity cost, KIVI/KVQuant-style alternatives, and the remaining MLX layout and kernel gaps.
- [How to measure what quantization actually costs](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/papers/how-to-measure-what-quantization-actually-costs.md) — the methods companion: teacher-forced paired scoring, streaming full-vocabulary KL on a 32 GB machine, the guards that keep a harness from passing by doing nothing, and where the verdict thresholds honestly come from.

## Status

0.5.0, released on PyPI as `mlx-quant-fidelity` — adds depth-resolved KV drift over a configurable `--chunk-length`, an auto-generated `compare kv --sweep`, and device provenance in every report. 0.4.0 added deployment mode (`--quantize-start`) and a shareable fidelity badge (`--format badge`). 0.3.x added the `compare` command for memory-normalized Pareto ranking of KV-cache and weight quantizations. Downstream-task accuracy and more are on the [roadmap](ROADMAP.md).

## License

[Apache-2.0](LICENSE).

## Sister projects

Other MLX libraries for Apple Silicon:

- [mlx-taef](https://github.com/IonDen/mlx-taef) — tiny autoencoders for fast diffusion-latent previews and low-memory decode (FLUX / SD).
- [mlx-teacache](https://github.com/IonDen/mlx-teacache) — TeaCache residual caching to skip redundant FLUX denoising steps.
- [mlx-model-doctor](https://github.com/IonDen/mlx-model-doctor) — validate an MLX / Hugging Face model repo before you load it (config, tokenizer, safetensors, memory).
