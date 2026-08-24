# Quant comparison (kv) vs `mlx-community/Llama-3.2-1B-Instruct-4bit`

| target | cost | KL mean | KL p99 | flip | verdict | frontier |
|---|---|---|---|---|---|---|
| `turboquant:3` | 8.2 KB | 0.4229 | 2.3559 | 0.3259 | bad | ✓ |
| `4:64` | 9.2 KB | 0.1477 | 0.9225 | 0.2048 | bad | ✗ dominated by `turboquant:4` |
| `turboquant:4` | 9.2 KB | 0.0825 | 0.5663 | 0.1582 | bad | ✓ |
| `8:64` | 17.4 KB | 0.0004 | 0.0029 | 0.0126 | marginal | ✓ |

