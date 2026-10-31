# Quant comparison (weight) vs `mlx-community/Qwen3-4B-Instruct-2507-bf16`

| target | cost | KL mean | KL p99 | flip | bits/wt | verdict | frontier |
|---|---|---|---|---|---|---|---|
| `mlx-community/Qwen3-4B-Instruct-2507-4bit` | 2.26 GB | 0.1106 | 0.7045 | 0.1717 | 4.50 | marginal | ✗ dominated by `mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510` |
| `mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510` | 2.26 GB | 0.0638 | 0.5719 | 0.1181 | 4.50 | marginal | ✓ |
| `mlx-community/Qwen3-4B-Instruct-2507-DDWQ` | 2.51 GB | 0.0401 | 0.3425 | 0.0920 | 5.00 | marginal | ✓ |
| `mlx-community/Qwen3-4B-Instruct-2507-5.0bpw-dynamic` | 2.52 GB | 0.0478 | 0.3558 | 0.1021 | 5.00 | marginal | ✗ dominated by `mlx-community/Qwen3-4B-Instruct-2507-DDWQ` |
| `mlx-community/Qwen3-4B-Instruct-2507-8bit` | 4.27 GB | 0.0017 | 0.0102 | 0.0209 | 8.50 | good | ✓ |


> Note: Assumes both repos share a tokenizer; only vocab_size (and bos/eos ids) were checked — a divergent token map would be measured as quant drift.

> Note: The recipe that produced this quantization is not recorded in a form this tool can check: mlx-lm's DWQ, AWQ and dynamic quantizers write a quantization block that records only geometry (bits, group sizes, per-module overrides), never the recipe. This report describes the geometry of the loaded model, not the method that produced it.

> Weight compare reloads the reference once per target — N targets ≈ Nx a single `weights` run. Fidelity is corpus- and context-length-specific.
