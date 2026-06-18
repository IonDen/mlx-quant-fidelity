# Quant comparison (kv) vs `mlx-community/Qwen2.5-0.5B-Instruct-4bit`

| target | cost | KL mean | KL p99 | flip | verdict | frontier |
|---|---|---|---|---|---|---|
| `4:64` | 3.5 KB | 2.5568 | 8.6525 | 0.7003 | bad | ✓ |
| `4:32` | 3.8 KB | 1.3413 | 6.9018 | 0.5232 | bad | ✓ |
| `8:64` | 6.5 KB | 0.0304 | 0.2671 | 0.0883 | bad | ✓ |

