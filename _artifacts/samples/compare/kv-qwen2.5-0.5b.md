# Quant comparison (kv) vs `mlx-community/Qwen2.5-0.5B-Instruct-4bit`

| target | cost | KL mean | KL p99 | flip | bundled KL | resident +/token | verdict | frontier |
|---|---|---|---|---|---|---|---|---|
| `4:64` | 3.5 KB | 2.0549 | 7.6896 | 0.6254 | 2.0531 | 0 B | bad | ✓ |
| `4:32` | 3.8 KB | 1.0898 | 5.8498 | 0.4695 | 1.0966 | 0 B | bad | ✓ |
| `8:64` | 6.5 KB | 0.0330 | 0.3277 | 0.0862 | 0.0336 | 0 B | bad | ✓ |


_ranked on quantizer-only drift; stock rows carry their bundled deployment drift alongside._
