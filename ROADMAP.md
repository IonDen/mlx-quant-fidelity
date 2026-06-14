# Roadmap

`mlx-quant-fidelity` measures how much quality a quantization costs on Apple Silicon / MLX.

## Released

- **0.1.0** — KV-cache quantization fidelity: KL divergence, top-token flip rate, perplexity delta; CLI and Python API; JSON/Markdown reports; WikiText-2 corpus.

## Next

- **Weight-quantization fidelity** — the same drift metrics for q4/q6/q8 weight quantization, with reference-logit handling that respects the full-vocab KLD requirement (stream or recompute rather than persist truncated top-k).
- **Deployment mode** — `quantize_start > 0`, matching what mlx-lm users run, with the first N tokens kept full-precision.
- **Downstream-task accuracy** — EleutherAI's lm-evaluation-harness (MMLU, ARC, HellaSwag, GSM8K, HumanEval) alongside distributional drift.
- **Method comparison** — rank quantization methods on one yardstick, memory-normalized (quality vs KV bytes per token) as a Pareto view, never a raw-metric sort.
- **Quantizer-only control** — an optional dequantize-then-standard-attention path that separates quantizer error from quantized-kernel numerics.
- **Wider attention coverage** — MLA and sliding-window caches beyond flag-don't-crash.
- **Fidelity badge** — a model-card badge driven by an explicit, documented threshold policy.
