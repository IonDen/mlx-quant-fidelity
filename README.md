# mlx-quant-fidelity

Measure how much quality a quantization costs on Apple Silicon. `mlx-quant-fidelity` scores a quantized KV cache against full precision on the same model and reports the drift as numbers you can act on: KL divergence, top-token flip rate, perplexity delta. No more choosing a bit-width by file size.

The CUDA/GGUF world has had this for years: llama.cpp's `--kl-divergence-base`, EleutherAI's `lm-evaluation-harness`. MLX had nothing. This is the MLX version, and it covers the KV-cache and attention angle those tools skip.

Version 0.1.0 measures **KV-cache quantization**. Weight-quantization fidelity is next; see the [roadmap](ROADMAP.md).

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

8-bit KV is near-lossless on all three models. 4-bit is another matter, and Qwen2.5-7B at 4-bit in stress mode falls apart: nearly every token flips. That is the attention sink at work: stress mode quantizes the cache from token 0, including the first tokens attention leans on most, and Qwen2.5 does not tolerate it. mlx-lm's own default keeps the first 5000 tokens full-precision for exactly this reason. The point of the tool is that you can see this for your model before you pick a bit-width.

## How it works

Teacher-forced scoring, not generation. For each fixed-length corpus chunk the model runs twice on the *same* tokens — once with a full-precision KV cache, once with a quantized one — and the two next-token distributions are compared position by position. Generation would let the runs diverge in their own inputs the moment quantization changed a sampled token, turning the measurement into trajectory drift instead of cache cost. Logits collapse to per-position scalars inside the chunk loop and are released before the next chunk, so a long corpus never holds full distributions in memory.

Two modes:

- **stress** (`--quantize-start 0`, the default): quantize from token 0. The harsh, apples-to-apples quantizer test.
- **deployment** (`quantize_start > 0`): what mlx-lm users actually run, with the first N tokens kept full-precision. Planned for 0.2.0.

A run that returns exactly zero drift raises instead of reporting a silent "perfect fidelity." That almost always means quantization never engaged, not that it was free.

## What the numbers don't say

- A fidelity number is **corpus- and context-length-specific**. WikiText-2 at temperature 0 measures short-prose distributional drift; the paper this builds on, *Accuracy Is Not All You Need*, shows that under-predicts task-specific and long-context degradation. Every report records the corpus and the token count so the number is never read as a bare score.
- Perplexity delta is reported for continuity with llama.cpp. It correlates with mean KLD by construction, so treat it as a familiar restatement, not independent corroboration.
- The measured drift bundles the quantizer's error with the quantized-attention kernel's numerics. That is the real end-to-end cost; a quantizer-only control is on the roadmap.

## Status

0.1.0, released on PyPI as `mlx-quant-fidelity` — the KV-cache fidelity probe (CLI + Python API, JSON and Markdown reports). Weight-quantization fidelity, downstream-task accuracy, and memory-normalized method ranking are on the [roadmap](ROADMAP.md).

## License

[Apache-2.0](LICENSE).
