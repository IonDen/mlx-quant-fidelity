# Quant comparison (kv) vs `mlx-community/Qwen2.5-0.5B-Instruct-4bit`

| target | cost | KL mean | KL p99 | flip | verdict | frontier |
|---|---|---|---|---|---|---|
| `4:64` | 3.5 KB | 2.0531 | 7.7054 | 0.6250 | bad | ✓ |
| `4:32` | 3.8 KB | 1.0966 | 5.8339 | 0.4702 | bad | ✓ |
| `8:64` | 6.5 KB | 0.0336 | 0.3313 | 0.0862 | bad | ✓ |

