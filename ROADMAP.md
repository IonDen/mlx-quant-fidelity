# Roadmap

`mlx-quant-fidelity` measures how much quality a quantization costs on Apple Silicon / MLX.

## Released

- **0.3.1** — Hardens `compare` error handling (consistent package-rooted errors; the CLI no longer masks unexpected errors), isolates cached partials with an invalid stored verdict, and adds a `measurement-principles.md` methodology document.
- **0.3.0** — `compare` command: ranks weight-quant repos or KV-cache configs on a memory-normalized Pareto frontier (quality per byte), flags dominated configurations, and supports budget-filter shortcuts.
- **0.2.0** — Weight-quantization fidelity: a quantized model repo scored against a higher-precision reference on the same corpus, with a compatibility gate and a memory pre-flight for the two-model load. Also fixes the KV cache group-size / head-dimension gate.
- **0.1.0** — KV-cache quantization fidelity: KL divergence, top-token flip rate, perplexity delta; CLI and Python API; JSON/Markdown reports; WikiText-2 corpus.

## Next

- **Deployment mode** — `quantize_start > 0`, matching what mlx-lm users run, with the first N tokens kept full-precision.
- **Downstream-task accuracy** — EleutherAI's lm-evaluation-harness (MMLU, ARC, HellaSwag, GSM8K, HumanEval) alongside distributional drift.
- **Quantizer-only control** — an optional dequantize-then-standard-attention path that separates quantizer error from quantized-kernel numerics.
- **Wider attention coverage** — MLA and sliding-window caches beyond flag-don't-crash.
- **Fidelity badge** — a model-card badge driven by an explicit, documented threshold policy.
