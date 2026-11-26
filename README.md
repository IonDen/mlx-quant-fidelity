# mlx-quant-fidelity

<p align="center">
  <img src="https://raw.githubusercontent.com/IonDen/mlx-quant-fidelity/main/docs/assets/mlx-quant-fidelity-logo.png" alt="mlx-quant-fidelity" width="100%">
</p>

[![PyPI version](https://img.shields.io/pypi/v/mlx-quant-fidelity.svg)](https://pypi.org/project/mlx-quant-fidelity/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://pypi.org/project/mlx-quant-fidelity/)
[![License: Apache-2.0](https://img.shields.io/pypi/l/mlx-quant-fidelity.svg)](https://github.com/IonDen/mlx-quant-fidelity/blob/main/LICENSE)

A quantized model (its numbers stored in fewer bits) still loads and answers, but its answers may have changed, and nothing tells you by how much. `mlx-quant-fidelity` measures that change for MLX models on Apple Silicon.

You need this if you run or publish quantized MLX models and want to pick a bit width by measured quality.

Use it to:

- Compare a quantized KV cache (what the model remembers of the text) with a full-precision one.
- Compare a quantized model's weights with a higher-precision copy.
- Get KL divergence (how far apart the two predictions are), flip rate (how often the top token changes), perplexity change and a verdict.
- Find the quantization with the best quality for its memory, and the KV-cache settings that fit a per-token limit you set.
- Print a badge for a model's Hugging Face page (its model card).

## Try it

It needs Apple Silicon and Python 3.11+.

```bash
pip install mlx-quant-fidelity
mlx-quant-fidelity kv mlx-community/Llama-3.2-3B-Instruct-4bit --kv-bits 8 --max-chunks 100
```

```
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

Measured on **wikitext-2-raw/test**, 51100 positions across 100 chunks of length 512 (tokenizer `mlx-community/Llama-3.2-3B-Instruct-4bit`).

...
```

That model at 8-bit KV gets a good verdict on this corpus. Each chunk scores 511 positions, one fewer than its 512 tokens. The first run downloads about 1.8 GB (1.7 GiB) of weights plus the corpus.

### Common options

- `--kv-bits` / `--kv-group-size`: the KV configuration to score, default `4` / `64`.
- `--max-chunks N`: score only the first N corpus chunks. The figures in this README use `--max-chunks 100` unless the text says otherwise; leave it off and the run covers the whole WikiText-2 test split.
- `--chunk-length N`: the scoring window, default 512, hard ceiling 4096.
- `--quantize-start N`: `0` for stress mode, the default; any N above 0 for deployment mode.
- `--allow-custom-code`: let a repo whose `config.json` names a `model_file` run that file. mlx-lm executes it on load, so the tool refuses such a repo unless you pass this flag. Use it only for repos you trust.

The other options (`--kv-method`, `--control`, `--format`, pinned revisions, `--worker-timeout`) are in [docs/cli-reference.md](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/cli-reference.md#common-options). The `compare` options (`--configs`, `--sweep`, `--max-kv-bytes-per-token`, `--max-kld`, `--min-tier`) are described under [Comparing quantizations by quality per byte](#comparing-quantizations-by-quality-per-byte) and in [docs/ranking-principles.md](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/ranking-principles.md).

A wider window costs memory rather than time. The 4096-token run on Llama-3.2-1B at 4-bit KV (below) peaks at 13.53 GiB, so it will not fit a 16 GB machine. [docs/measurement-principles.md](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/measurement-principles.md#drift-by-position-depth) lists the measured peak for every window length and explains the pre-flight that refuses a window too large for your device.

## Does this apply to you?

Precision is lost in two places, in the weights on disk and in the running cache, and each has its own command.

![Diagram: a running quantized model loses precision in two places. Weights on disk were quantized once, before download, and are measured by `mlx-quant-fidelity weights`, which scores a quantized repo against a higher-precision repo. The KV cache is quantized continuously at run time as the cache grows with every token, and is measured by `mlx-quant-fidelity kv`, which scores a full-precision cache against a quantized one. Both feed `mlx-quant-fidelity compare`, which ranks configurations by quality per byte](https://raw.githubusercontent.com/IonDen/mlx-quant-fidelity/main/docs/assets/diagrams/coverage-map.svg)

**"I'm about to run a quantized model and I want to know what I gave up."**

```bash
mlx-quant-fidelity kv mlx-community/Llama-3.2-3B-Instruct-4bit --kv-bits 4
mlx-quant-fidelity weights mlx-community/Llama-3.2-3B-Instruct-4bit --reference mlx-community/Llama-3.2-3B-Instruct-bf16
```

**"I need this model to fit in my RAM and I don't know which setting to cut."**

```bash
mlx-quant-fidelity compare kv mlx-community/Llama-3.2-1B-Instruct-4bit --sweep --max-kv-bytes-per-token 10000
```

The budget is in bytes per token across all layers. Building the bits-by-group-size grid reads only the model's `config.json`; the command then drops anything over your budget, downloads the model and ranks the rest by quality per byte.

**"I publish quantized models and I want to show they're good."**

```bash
mlx-quant-fidelity kv <model> --kv-bits 8 --format badge
```

This prints one shields.io line for your model card.

## A fidelity badge for your model card

`--format badge` replaces the whole report with one line:

```
![KV fidelity](https://img.shields.io/badge/KV_fidelity-good_%C2%B7_8--bit_%C2%B7_wikitext--2--raw%2F512_%C2%B7_stress-brightgreen)
```

Green for `good`, yellow for `marginal`, red for `bad`. The bit width, corpus, chunk length and mode are in the message, so two badges for the same model at different settings look different. Threshold values and the color map are in [docs/threshold-policy.md](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/threshold-policy.md). A hybrid model measured on only some of its layers carries `partial 4/26 layers` in the badge message, so the badge can't be read as covering the whole cache.

## How much do KV-cache and weight quantization cost?

![Chart of mean KL divergence on a logarithmic axis, lower being better, for KV-cache quantization (left panel, six bars) and weight quantization (right panel, five bars). The 8-bit bars are short for every model; the 4-bit KV bars are long, with Qwen2.5-7B at 9.36. A bar's length is its mean KL divergence alone, and its color is the overall verdict](https://raw.githubusercontent.com/IonDen/mlx-quant-fidelity/main/docs/assets/charts/fidelity-cliff.svg)

Eight-bit costs little everywhere we measured it. Four-bit is a real trade, and on one checkpoint it collapses.

Bar length is mean KL divergence alone. Bar color is the overall verdict, which also weighs the p99 tail and the top-token flip rate, so a short bar can still be amber, and the 4-bit Llama-3.2-3B bar is red inside the amber band. [docs/threshold-policy.md](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/threshold-policy.md) lists the ceilings. The right panel gets its own section below.

KV cache, M1 Max, WikiText-2 test (100 chunks of 512 tokens), stress mode (quantize from token 0). Reproduce any row with `mlx-quant-fidelity kv <model> --kv-bits <bits> --max-chunks 100`; the full committed reports are under [`_artifacts/samples/`](https://github.com/IonDen/mlx-quant-fidelity/blob/main/_artifacts/samples).

| Model | KV bits | KL mean (nats) | flip rate | verdict |
|---|---|---|---|---|
| Llama-3.2-1B | 4 | 0.148 | 0.20 | bad |
| Llama-3.2-1B | 8 | 0.0004 | 0.013 | marginal |
| Llama-3.2-3B | 4 | 0.051 | 0.11 | bad |
| Llama-3.2-3B | 8 | 0.0002 | 0.007 | good |
| Qwen2.5-7B | 4 | 9.36 | 0.99 | bad |
| Qwen2.5-7B | 8 | 0.009 | 0.032 | marginal |

8-bit KV costs little on all three models, though only Llama-3.2-3B gets a good verdict. At 4 bits, Qwen2.5-7B in stress mode falls apart: its flip rate is 0.99, so nearly every top token changes. The measurement shows the failure on this checkpoint, not its cause.

Stress mode is harsher than mlx-lm's default generate command. In mlx-lm 0.31.3, the `mlx_lm.generate` command with `--kv-bits` keeps the first 5,000 tokens in full precision by default (`--quantized-kv-start`), then converts the whole stored prefix at the boundary. The Python function `generate_step` defaults `quantized_kv_start` to 0, so quantization starts at token 0 there. Running this tool first shows the fidelity risk before you deploy.

## Does drift change with position depth?

On Llama-3.2-1B at 4-bit KV, M1 Max, WikiText-2 test (`--chunk-length 4096 --max-chunks 12`, so 12 chunks of 4096 tokens), drift at position 4000 looks about the same as drift at position 60. Quantization cost does not build up across the window at these lengths. That is a narrow claim: 4096 tokens is short next to the context lengths where other work has found KV-quantization drift growing with depth. The eight-row table, the memory cost of longer windows and the comparison with longer-context findings are in [docs/measurement-principles.md](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/measurement-principles.md#drift-by-position-depth).

## How much does weight quantization cost?

Same corpus and recipe, but the comparison is now a quantized model repo against a higher-precision reference repo. Reproduce any row with `mlx-quant-fidelity weights <quant> --reference <reference> --max-chunks 100`; the committed reports are under [`_artifacts/samples/weights/`](https://github.com/IonDen/mlx-quant-fidelity/blob/main/_artifacts/samples/weights). These five rows were recorded with the 0.5.0 report format and are kept as recorded.

| Model | quant | reference | KL mean (nats) | flip rate | perplexity Δ | verdict |
|---|---|---|---|---|---|---|
| Llama-3.2-1B | 4-bit | bf16 | 0.158 | 0.21 | +3.5 | marginal |
| Llama-3.2-1B | 8-bit | bf16 | 0.001 | 0.023 | −0.01 | good |
| Llama-3.2-3B | 4-bit | bf16 | 0.085 | 0.15 | +1.4 | marginal |
| Llama-3.2-3B | 8-bit | bf16 | 0.0009 | 0.021 | 0.00 | good |
| Qwen2.5-7B | 4-bit | 8-bit | 0.109 | 0.16 | +0.9 | marginal |

8-bit weights are near-lossless: about 2% of top tokens flip and perplexity barely moves. Across the three 4-bit rows, 4-bit flips 15 to 21% of top tokens and perplexity climbs by 0.9 to 3.5 perplexity units, worst on the smallest model here. The Qwen row compares 4-bit against 8-bit rather than bf16, so its drift is relative to an already-quantized reference, not full precision. The verdict tiers are provisional, anchored to these q8 and q4 reference points on short prose rather than to downstream task accuracy.

### One model, five quantizations

Qwen3-0.6B has a bf16 reference on the Hub. `compare weights` scores five of its quantized repos against it (Apple M1 Max, WikiText-2 test, 100 chunks of 512 tokens, peak 3.60 GiB across the five runs). The 8-bit repo is the only one rated good (mean KL 0.0040), the 6-bit repo is marginal (0.0213), and the 4-bit repos run from 0.1608 (`-4bit-AWQ`, marginal) to 0.3539 (`-4bit-DWQ`, bad). The plain `-4bit` and `-4bit-DWQ` repos are the same size to within 36 bytes, and `-4bit-DWQ` has the higher mean KL here, 0.3539 against 0.2538, though it has the lower flip rate, p99 tail and perplexity change. The full table, the command with its pinned revisions and the reading are in [docs/weight-ladders.md](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/weight-ladders.md#one-model-five-quantizations).

### The same ladder at 4B

Qwen3-4B-Instruct-2507 has a bf16 reference and five quantized repos, scored the same way (peak 13.50 GiB across the five runs). Again only the 8-bit repo is rated good (mean KL 0.0017). Here `-4bit-DWQ-2510` has lower drift than the plain `-4bit` repo, 0.0638 against 0.1106, the other way round from 0.6B. The table and command are in [docs/weight-ladders.md](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/weight-ladders.md#the-same-ladder-at-4b).

## Comparing quantizations by quality per byte

`compare` ranks a set of quantizations on a memory-normalized Pareto frontier: quality (mean KL divergence) on one axis, memory cost on the other. A configuration that another option matches or beats on both axes, and beats on at least one, is dominated, and you would never choose it.

![Diagram explaining domination: configuration A is no worse than B on quality, its mean KL divergence being no higher, and no worse on cost, its cache bytes per token being no higher. B wins on neither axis while A beats it on at least one, so B is dominated and no memory budget would make it the right pick. Ranking reports domination so options can be discarded outright instead of weighed by hand](https://raw.githubusercontent.com/IonDen/mlx-quant-fidelity/main/docs/assets/diagrams/pareto-domination.svg)

```bash
# rank weight quantizations against a bf16 reference
mlx-quant-fidelity compare weights q4 q6 q8 --reference fp16

# rank KV configs on a single model
mlx-quant-fidelity compare kv <model> --configs 4:32,4:64,8:64

# or auto-generate the grid from the model's config.json instead of listing configs by hand
mlx-quant-fidelity compare kv mlx-community/Llama-3.2-1B-Instruct-4bit --sweep --max-kv-bytes-per-token 10000
```

Add `--max-kld 0.05` to get the cheapest configuration whose mean KL stays under a threshold, or `--min-tier good` for the cheapest one with a good verdict. `--sweep` builds the (bits × group-size) grid from the model's config alone, with no weight download, and `--max-kv-bytes-per-token` keeps only configurations under a memory budget, counted in bytes per token across all layers; skipped configurations are listed in the report. [docs/ranking-principles.md](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/ranking-principles.md) explains how each axis is computed and where the ranking has limits.

### Measuring a third-party KV cache (TurboQuant-MLX)

`--kv-method turboquant` measures the [TurboQuant-MLX](https://github.com/arozanov/turboquant-mlx) uniform-bit cache on the same yardstick, and `compare kv` ranks it against the stock configurations. Install the pinned port first; the PyPI package named `turboquant-mlx` is unrelated. The tool checks the install record before importing the port and accepts only an https git install of this repository (a different commit is measured with a note in the report); a PyPI, editable, SSH or fork install is refused on purpose:

```bash
pip install "turboquant-mlx @ git+https://github.com/arozanov/turboquant-mlx@6e928d715595dee9f6b6cc3968baa44e1f408d28"
mlx-quant-fidelity compare kv mlx-community/Llama-3.2-1B-Instruct-4bit --configs 8:64,4:64,turboquant:4,turboquant:3,turboquant-vonly:4,turboquant-vonly:3,affine:8:4,affine:8:2
```

With uv, `uv sync --group turboquant` installs the same pin.

On Llama-3.2-1B (M1 Max, WikiText-2 test, 100 chunks of 512 tokens, stress mode), `turboquant:4` has a mean KL of 0.0825 against 0.1485 for stock `4:64` at the same stored cost of 9.2 KB per token, so `4:64` is dominated; both are rated bad. Stored cost leaves out working memory: `turboquant:4` also holds about 65.5 KB per token of full-precision working buffers while it runs, and stock holds none. Every row's mean KL is quantizer-only. The full table, how to read its `cost` and `resident +/token` columns, and what the pinned port leaves unmeasured are in [docs/third-party-kv-cache.md](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/third-party-kv-cache.md#measuring-a-third-party-kv-cache-turboquant-mlx).

## How it works: teacher-forced paired scoring

Teacher-forced scoring, not generation. For each fixed-length corpus chunk the model runs twice on the *same* tokens, once with a full-precision KV cache and once with a quantized one, and the two next-token distributions are compared position by position. Generation would let the runs diverge in their own inputs the moment quantization changed a sampled token, so the measurement would show trajectory drift instead of cache cost. Logits collapse to per-position scalars inside the chunk loop and are released before the next chunk, so a long corpus never holds full distributions in memory.

![Diagram of the paired teacher-forced scoring loop: a fixed corpus chunk feeds a reference run with a full-precision KV cache and a quantized run with a quantized KV cache; both produce vocabulary-wide fp32 logits that are reduced inside the loop to per-position scalars for KL divergence, top-token flip and target negative log-likelihood; the scalars are evaluated, the per-chunk caches are dropped and the cache pool cleared, and the loop advances, so no vocabulary-wide tensor outlives its chunk](https://raw.githubusercontent.com/IonDen/mlx-quant-fidelity/main/docs/papers/diagrams/paired-scoring-chunk-loop.svg)

Every report records which of two modes produced it.

![Diagram comparing the two measurement modes. In stress mode with quantize-start 0, the default, every position runs against a quantized cache from the very first token, and every position is scored. In deployment mode with quantize-start 5000, positions 0 through 4999 are computed against a full-precision cache; at position 5000 the entire stored cache converts to quantized and no full-precision prefix is kept; positions 5000 and beyond run against the quantized cache, and only the post-boundary positions are scored](https://raw.githubusercontent.com/IonDen/mlx-quant-fidelity/main/docs/assets/diagrams/stress-vs-deployment.svg)

Stress mode (`--quantize-start 0`, the default) quantizes from token 0, the harsh, apples-to-apples quantizer test. Deployment mode (`--quantize-start N`) computes the first N positions with a full-precision cache, then converts the entire stored cache and scores only the post-boundary region. That matches mlx-lm's `--quantized-kv-start` conversion behavior, which does not preserve a full-precision prefix in storage. [docs/measurement-principles.md](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/measurement-principles.md) explains why deployment and stress drift need a matched comparison and why neither is a long-context deployment average.

A run that returns exactly zero drift raises instead of reporting a silent "perfect fidelity." That almost always means quantization never engaged, not that it was free.

The weight probe works the same way with two models instead of two caches: a quantized repo and a reference repo, scored on the same corpus tokens. It refuses a mismatched or oversized pair before either model loads. The zero-probability policy, how perplexity delta relates to mean KL, and why the weight probe's drift still includes the quantized-matmul kernel's numerics are in [docs/measurement-principles.md](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/measurement-principles.md).

## What a fidelity number can't tell you

- A fidelity number is specific to its corpus and context length. WikiText-2 at temperature 0 measures short-prose distributional drift, and the paper this builds on, *Accuracy Is Not All You Need*, shows that such a score under-predicts task-specific and long-context degradation. Every report records the corpus and the token count so the number is never read as a bare score.
- Perplexity delta is reported for continuity with llama.cpp. It is related to but distinct from mean KL: it scores the realized next token and can diverge from full-vocabulary drift, so it is not independent corroboration.
- The KV probe's deployed-path drift bundles the quantizer's error with the quantized-attention kernel's numerics, which is the real end-to-end cost. `kv --control` (stock only) measures the quantizer alone, and `compare kv` ranks on that number; see [docs/measurement-principles.md](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/measurement-principles.md#decomposing-bundled-and-quantizer-only-drift).
- Models whose cache holds something other than plain keys and values need their own reading: [MLA models](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/measurement-principles.md#mla-models-and-what-their-cache-stores) store a compressed latent, and [hybrid models](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/measurement-principles.md#partial-coverage-of-hybrid-models) are measured on only some of their layers.

## Python API

Each command has a function behind it that returns the same report object the CLI renders.

```python
from mlx_quant_fidelity import measure_kv_fidelity

report = measure_kv_fidelity("mlx-community/Llama-3.2-3B-Instruct-4bit", kv_bits=8)
print(report.kl.mean, report.flip_rate, report.verdict)
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

Pass a method object to measure a cache other than mlx-lm's stock one. TurboQuant needs the pinned install described under [Measuring a third-party KV cache](#measuring-a-third-party-kv-cache-turboquant-mlx); `AffineKVMethod` needs nothing extra.

```python
from mlx_quant_fidelity import AffineKVMethod, TurboQuantKVMethod, measure_kv_fidelity

report = measure_kv_fidelity(
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    method=TurboQuantKVMethod(bits=4, seed=42),  # or AffineKVMethod(k_bits=8, v_bits=4)
)
print(report.kv_method, report.kl.mean, report.drift_footing)
```

`compare_kv_fidelity` and `compare_weight_fidelity` back the two `compare` subcommands and return a `ComparisonReport`. None of the Python functions print progress unless you pass a `progress` callback, and none of them start the memory watchdog described below.

## Exit codes and safety

| code | meaning |
|---|---|
| 0 | the report was printed |
| 1 | `compare` measured no row at all (the report is still printed), or an unexpected internal error |
| 2 | a user error: a bad option, a missing or gated repo, no network, a refused configuration, or a refused measurement (a NaN metric, an exact-zero result) |
| 3 | the memory watchdog stopped the run |

Every command reads the model's `config.json` before loading any weights. mlx-lm runs the Python file a repo names under `model_file` when it loads the model, so a repo whose config names one is refused unless you pass `--allow-custom-code`. Use the flag only for repos you trust. The model is then loaded at exactly the commit whose `config.json` was checked. Invalid method parameters, a missing TurboQuant install and a TurboQuant method that does not fit the model's cache layout are also caught here, before a download starts.

`--allow-custom-code` also covers a tokenizer that needs its own code. On transformers 5.4.0 or later the refusal (exit code 2) comes before any weight download. On an older version it comes at the model load, after the weights have downloaded and loaded and before any code from the repo has run. [docs/cli-reference.md](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/cli-reference.md#tokenizer-code-check) has the rest.

`compare` records a target it cannot read as a failed row and carries on with the rest. A shared reference it cannot read (`compare weights`), or an unreadable model (`compare kv`), is exit 2, since no row could be measured.

Every entry point, the Python API included, caps MLX's wired memory and its buffer cache (4 GiB) before loading. The CLI also runs a watchdog that exits with code 3 once MLX's active plus cached memory passes the device's memory size less 4 GiB; the Python API starts no watchdog. That ceiling is only a backstop. The pre-flight gates described in [docs/measurement-principles.md](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/measurement-principles.md#memory-guards) are what should refuse an oversized run before it starts. Long runs print progress to stderr; standard output carries only the report.

## Further reading

- [Low-bit KV caches on MLX: what exists and what is missing](https://ineshin.space/papers/low-bit-kv-caches-on-mlx-what-exists-and-what-is-missing/): mlx-lm's shipped cache, the measured 8-bit and 4-bit cost, and the gaps that remain.
- [How to measure what quantization actually costs](https://ineshin.space/papers/how-to-measure-what-quantization-actually-costs/): the method behind this tool.
- [What an MLX quantized repository does and does not tell you](https://ineshin.space/papers/what-an-mlx-quantized-repository-does-and-does-not-tell-you/): the plain and DWQ 4-bit Qwen3-0.6B repos have byte-identical configs, yet their fidelity metrics disagree about which is better.
- More writing at [ineshin.space](https://ineshin.space).

## Status

0.12.1, released on PyPI as `mlx-quant-fidelity`, is a bug-fix release. A run that cannot work, because a TurboQuant method does not fit the model or a tokenizer needs its own code, is now refused before the weights download (the tokenizer check needs transformers 5.4.0 or later). Every release is in [CHANGELOG.md](https://github.com/IonDen/mlx-quant-fidelity/blob/main/CHANGELOG.md), and what comes next is in [ROADMAP.md](https://github.com/IonDen/mlx-quant-fidelity/blob/main/ROADMAP.md).

Questions and shared results go to [GitHub Discussions](https://github.com/IonDen/mlx-quant-fidelity/discussions), bugs to [Issues](https://github.com/IonDen/mlx-quant-fidelity/issues).

## License

[Apache-2.0](https://github.com/IonDen/mlx-quant-fidelity/blob/main/LICENSE).

## Sister projects

Other MLX libraries for Apple Silicon:

- [mlx-taef](https://github.com/IonDen/mlx-taef): tiny autoencoders for fast diffusion-latent previews and low-memory decode (FLUX / SD).
- [mlx-teacache](https://github.com/IonDen/mlx-teacache): TeaCache residual caching to skip redundant FLUX denoising steps.
- [mlx-model-doctor](https://github.com/IonDen/mlx-model-doctor): validate an MLX / Hugging Face model repo before you load it.
