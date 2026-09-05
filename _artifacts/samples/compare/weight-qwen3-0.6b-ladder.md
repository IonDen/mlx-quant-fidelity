# Quant comparison (weight) vs `mlx-community/Qwen3-0.6B-bf16`

| target | cost | KL mean | KL p99 | flip | bits/wt | verdict | frontier |
|---|---|---|---|---|---|---|---|
| `mlx-community/Qwen3-0.6B-4bit-DWQ` | 335.5 MB | 0.3539 | 1.4570 | 0.2289 | 4.50 | bad | ✓ |
| `mlx-community/Qwen3-0.6B-4bit` | 335.5 MB | 0.2538 | 1.5364 | 0.2586 | 4.50 | bad | ✓ |
| `mlx-community/Qwen3-0.6B-4bit-AWQ` | 345.2 MB | 0.1608 | 1.1002 | 0.2117 | 4.63 | marginal | ✓ |
| `mlx-community/Qwen3-0.6B-6bit` | 484.4 MB | 0.0213 | 0.1324 | 0.0781 | 6.50 | marginal | ✓ |
| `mlx-community/Qwen3-0.6B-8bit` | 633.4 MB | 0.0040 | 0.0198 | 0.0331 | 8.50 | good | ✓ |


> Note: Assumes both repos share a tokenizer; only vocab_size (and bos/eos ids) were checked — a divergent token map would be measured as quant drift.

> Note: Quantization method is not recorded in the repo config — mlx-lm's DWQ, AWQ and dynamic quantizers write a quantization block that records only geometry (bits, group sizes, per-module overrides), never the recipe. This report describes the geometry of the loaded model, not the method that produced it.

> Weight compare reloads the reference once per target — N targets ≈ Nx a single `weights` run. Fidelity is corpus- and context-length-specific.
