# Weight-quantization ladders

This page holds two comparisons of several quantized repos of one model against a single reference: Qwen3-0.6B and Qwen3-4B-Instruct-2507. Each has its table, its command and the conditions it was measured under. The [README](../README.md) has the short version.

## One model, five quantizations

Qwen3-0.6B has a bf16 reference on the Hub, and `compare weights` scores five of its quantized repos against that one reference:

| target | cost | KL mean | KL p99 | flip | bits/wt | verdict | frontier |
|---|---|---|---|---|---|---|---|
| `mlx-community/Qwen3-0.6B-4bit-DWQ` | 335.5 MB | 0.3539 | 1.4570 | 0.2289 | 4.50 | bad | ✓ |
| `mlx-community/Qwen3-0.6B-4bit` | 335.5 MB | 0.2538 | 1.5364 | 0.2586 | 4.50 | bad | ✓ |
| `mlx-community/Qwen3-0.6B-4bit-AWQ` | 345.2 MB | 0.1608 | 1.1002 | 0.2117 | 4.63 | marginal | ✓ |
| `mlx-community/Qwen3-0.6B-6bit` | 484.4 MB | 0.0213 | 0.1324 | 0.0781 | 6.50 | marginal | ✓ |
| `mlx-community/Qwen3-0.6B-8bit` | 633.4 MB | 0.0040 | 0.0198 | 0.0331 | 8.50 | good | ✓ |

```bash
mlx-quant-fidelity compare weights \
  mlx-community/Qwen3-0.6B-4bit@73e3e38d981303bc594367cd910ea6eb48349da8 \
  mlx-community/Qwen3-0.6B-4bit-DWQ@e630d870397d5a2d95fe0c9075c6f499fc0fc5c8 \
  mlx-community/Qwen3-0.6B-4bit-AWQ@3c064b3401d4a7d355262a1d518faa823a4d8f11 \
  mlx-community/Qwen3-0.6B-6bit@45d962b21b1e813c3e9a7f3505391e72e8daba1e \
  mlx-community/Qwen3-0.6B-8bit@11de96878523501bcaa86104e3c186de07ff9068 \
  --reference mlx-community/Qwen3-0.6B-bf16@42096995f6402fde107068cf530136fe64b604f8 \
  --max-chunks 100
```

Apple M1 Max, WikiText-2 test, 100 chunks of 512 tokens, peaking at 3.60 GiB across the five runs. The committed report is [`_artifacts/samples/compare/weight-qwen3-0.6b-ladder.md`](../_artifacts/samples/compare/weight-qwen3-0.6b-ladder.md).

The plain `-4bit` repo and the `-4bit-DWQ` repo are the same size to within 36 bytes, and the metrics disagree about which of them is better. The `-4bit-DWQ` repo has the higher mean KL on this model and this corpus, 0.3539 against 0.2538, and the lower flip rate and p99 tail, with a smaller perplexity delta: +3.59 against +4.51. Ranking scores quality on mean KL alone, so on that axis the `-4bit-DWQ` repo is the worse of the two, and neither row is dominated: the `-4bit-DWQ` row stays on the frontier only on its 36-byte cost edge.

Of the three 4-bit repos, `-4bit-AWQ` is the best on every quality column and on perplexity delta (+3.23), and it stores 4.63 [bits per weight](measurement-principles.md#what-the-weight-report-can-and-cannot-say-about-the-method) rather than 4.50; its geometry has one module at group 32 where the other 196 use 64. Above 4 bits the drift falls away quickly: 6-bit is marginal and 8-bit good on this model and this corpus. The report does not know which of these repos was produced with learned scales or activation-aware scaling — mlx-lm records only the geometry — so the rows are labeled by repo, not by method.

## The same ladder at 4B

Qwen3-4B-Instruct-2507 has a bf16 reference and five quantized repos beside it, scored the same way:

| target | cost | KL mean | KL p99 | flip | bits/wt | verdict | frontier |
|---|---|---|---|---|---|---|---|
| `mlx-community/Qwen3-4B-Instruct-2507-4bit` | 2.26 GB | 0.1106 | 0.7045 | 0.1717 | 4.50 | marginal | ✗ dominated by `mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510` |
| `mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510` | 2.26 GB | 0.0638 | 0.5719 | 0.1181 | 4.50 | marginal | ✓ |
| `mlx-community/Qwen3-4B-Instruct-2507-DDWQ` | 2.51 GB | 0.0401 | 0.3425 | 0.0920 | 5.00 | marginal | ✓ |
| `mlx-community/Qwen3-4B-Instruct-2507-5.0bpw-dynamic` | 2.52 GB | 0.0478 | 0.3558 | 0.1021 | 5.00 | marginal | ✗ dominated by `mlx-community/Qwen3-4B-Instruct-2507-DDWQ` |
| `mlx-community/Qwen3-4B-Instruct-2507-8bit` | 4.27 GB | 0.0017 | 0.0102 | 0.0209 | 8.50 | good | ✓ |

```bash
mlx-quant-fidelity compare weights \
  mlx-community/Qwen3-4B-Instruct-2507-4bit@50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b \
  mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510@c073725c8ac051eabad9d64f4dcd3019d1072559 \
  mlx-community/Qwen3-4B-Instruct-2507-DDWQ@88033de44951ebedb96e0adb68cc037443aab93a \
  mlx-community/Qwen3-4B-Instruct-2507-5.0bpw-dynamic@6bd1669a2ed4f8f47a7a1fa6038c14fb2bbea608 \
  mlx-community/Qwen3-4B-Instruct-2507-8bit@0e42af58449718de7931ee04f28191fbe6c43a56 \
  --reference mlx-community/Qwen3-4B-Instruct-2507-bf16@f9e77d4283734966e9cd641bf35547f0cff5d427 \
  --max-chunks 100
```

Apple M1 Max, WikiText-2 test, 100 chunks of 512 tokens, peaking at 13.50 GiB across the five runs. The committed report is [`_artifacts/samples/compare/weight-qwen3-4b-2507-ladder.md`](../_artifacts/samples/compare/weight-qwen3-4b-2507-ladder.md).

The two uniform 4-bit repos are the same size, and here the `-4bit-DWQ-2510` repo has the lower drift: mean KL 0.0638 against 0.1106 for `-4bit`, with a lower flip rate (0.1181 against 0.1717) and p99 tail. The `-4bit` row is dominated by it. At 0.6B the same pair came out the other way round.

The `-DDWQ` and `-5.0bpw-dynamic` repos are mixed-precision, so they are reported by measured geometry, not by a method name: the `-DDWQ` repo holds 165 modules at 4 bits and 88 at 6 bits, the `-5.0bpw-dynamic` repo 109 at 4 bits and 144 at 5 bits, and both come to 5.00 bits per weight. At that size the `-DDWQ` repo has the lower mean KL (0.0401 against 0.0478) and is also slightly smaller (2.51 GB against 2.52 GB), so the `-5.0bpw-dynamic` repo is dominated by it. The frontier is the `-4bit-DWQ-2510`, `-DDWQ` and `-8bit` repos. The 8-bit repo is the only one rated good, at a mean KL of 0.0017. As with the 0.6B ladder, the report records geometry only, so repo names are the repos' own claims and not something the tool checked.
