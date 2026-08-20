# Roadmap

`mlx-quant-fidelity` measures how much quality a quantization costs on Apple Silicon / MLX.

## Released

- **0.5.1** — Documentation release. Rewritten README built around the three questions the tool answers, diagrams for the coverage map, the two measurement modes, and Pareto domination, and a fidelity chart rendered from the committed sample reports. No change to measurement behavior.
- **0.5.0** — Depth-resolved KV drift: stress-mode reports over a fixed-window corpus break KLD down by position depth, with a configurable `--chunk-length` (up to 4096) to widen the window. Adds `compare kv --sweep` to auto-generate a config grid from the model's `config.json`, a `--max-kv-bytes-per-token` budget filter, a rejection gate for `(head_dim, bits)` combinations that crash the upstream KV cache, device provenance in every report, and hardening for a malformed cached comparison partial.
- **0.4.0** — Deployment mode (`quantize_start > 0`: the first N positions are computed with a full-precision cache, then the whole stored cache converts; metrics cover the post-boundary region) and a shareable model-card fidelity badge (`--format badge`) driven by the documented threshold policy; also isolates a structurally malformed cached comparison partial.
- **0.3.1** — Hardens `compare` error handling (consistent package-rooted errors; the CLI no longer masks unexpected errors), isolates cached partials with an invalid stored verdict, and adds a `measurement-principles.md` methodology document.
- **0.3.0** — `compare` command: ranks weight-quant repos or KV-cache configs on a memory-normalized Pareto frontier (quality per byte), flags dominated configurations, and supports budget-filter shortcuts.
- **0.2.0** — Weight-quantization fidelity: a quantized model repo scored against a higher-precision reference on the same corpus, with a compatibility gate and a memory pre-flight for the two-model load. Also fixes the KV cache group-size / head-dimension gate.
- **0.1.0** — KV-cache quantization fidelity: KL divergence, top-token flip rate, perplexity delta; CLI and Python API; JSON/Markdown reports; WikiText-2 corpus.

## Next

- **Downstream-task accuracy** — EleutherAI's lm-evaluation-harness (MMLU, ARC, HellaSwag, GSM8K, HumanEval) alongside distributional drift.
- **Quantizer-only control** — an optional dequantize-then-standard-attention path that separates quantizer error from quantized-kernel numerics.
- **Wider attention coverage** — MLA and sliding-window caches beyond flag-don't-crash.
