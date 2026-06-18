# Quant comparison (weight) vs `mlx-community/Qwen2.5-0.5B-Instruct-bf16`

| target | cost | KL mean | KL p99 | flip | verdict | frontier |
|---|---|---|---|---|---|---|
| `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | 278.1 MB | 0.2249 | 1.3324 | 0.2559 | bad | ✓ |
| `mlx-community/Qwen2.5-0.5B-Instruct-8bit` | 525.0 MB | 0.0024 | 0.0154 | 0.0254 | good | ✓ |


> Weight compare reloads the reference once per target — N targets ≈ Nx a single `weights` run. Fidelity is corpus- and context-length-specific.
