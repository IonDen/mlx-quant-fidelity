# Quant comparison (weight) vs `mlx-community/Qwen2.5-0.5B-Instruct-bf16`

| target | cost | KL mean | KL p99 | flip | verdict | frontier |
|---|---|---|---|---|---|---|
| `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | 278.1 MB | 0.2236 | 1.2122 | 0.2503 | bad | ✓ |
| `mlx-community/Qwen2.5-0.5B-Instruct-8bit` | 525.0 MB | 0.0021 | 0.0111 | 0.0269 | good | ✓ |


> Weight compare reloads the reference once per target — N targets ≈ Nx a single `weights` run. Fidelity is corpus- and context-length-specific.
