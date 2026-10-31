# Quant comparison (weight) vs `mlx-community/Qwen2.5-0.5B-Instruct-bf16`

| target | cost | KL mean | KL p99 | flip | bits/wt | verdict | frontier |
|---|---|---|---|---|---|---|---|
| `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | 278.1 MB | 0.2236 | 1.2122 | 0.2503 | 4.50 | bad | ✓ |
| `mlx-community/Qwen2.5-0.5B-Instruct-8bit` | 525.0 MB | 0.0021 | 0.0111 | 0.0269 | 8.50 | good | ✓ |


> Note: Assumes both repos share a tokenizer; only vocab_size (and bos/eos ids) were checked — a divergent token map would be measured as quant drift.

> Note: The recipe that produced this quantization is not recorded in a form this tool can check: mlx-lm's DWQ, AWQ and dynamic quantizers write a quantization block that records only geometry (bits, group sizes, per-module overrides), never the recipe. This report describes the geometry of the loaded model, not the method that produced it.

> Weight compare reloads the reference once per target — N targets ≈ Nx a single `weights` run. Fidelity is corpus- and context-length-specific.
