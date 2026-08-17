# Fidelity verdict thresholds

Your report came back `marginal`. Here's where that line was drawn.

`mlx-quant-fidelity` assigns each measurement a verdict of `good`, `marginal`, or `bad`. The verdict is driven by three calibrated thresholds — mean KLD, p99 KLD, and flip rate — and all three must pass for a tier to apply.

## What the verdicts mean

A `good` verdict means the quantization is near-lossless on this corpus. The distribution barely shifts and nearly all top-token choices match the reference. The model as seen by the decoder is essentially unchanged.

A `marginal` verdict means the quantization is measurable but contained. Mean drift is noticeable and some top tokens change, but the model remains usable for most tasks. Whether that is acceptable depends on the downstream task.

A `bad` verdict means the quantization causes substantial drift: the mean KLD is high, the tail is severe, or a large fraction of top tokens disagree with the reference. The model's output on this corpus diverges meaningfully from the unquantized version.

Verdicts are measurement-specific. A model that scores `good` on 8-bit KV may score `bad` on 4-bit. The same configuration can score differently on a different architecture.

## Why all three metrics must pass

The verdict requires mean KLD, p99 KLD, and flip rate to all clear their tier ceilings. Any one can flag a problem the others miss.

Mean KLD measures average full-vocabulary drift across all positions. It catches systematic shifts but gets diluted when problems are concentrated at specific positions. A healthy mean does not rule out a bad tail.

KLD p99 is the 99th-percentile per-position KLD. It catches whether bad positions are rare outliers or a real pattern. A high p99 relative to the mean means a small fraction of positions are taking a large hit — which matters when those positions happen to be important: sentence boundaries, infrequent tokens, long-range dependencies.

Flip rate is top-token disagreement: the fraction of positions that choose a different top token under quantization. It can flag problems KLD misses when full-vocab drift is low but concentrated at the argmax. It is also the most directly interpretable metric — those positions produce different outputs.

The logic from `policy.py`:

```python
if kl_mean <= g["kl_mean"] and kl_p99 <= g["kl_p99"] and flip_rate <= g["flip_rate"]:
    return "good"
if kl_mean <= m["kl_mean"] and kl_p99 <= m["kl_p99"] and flip_rate <= m["flip_rate"]:
    return "marginal"
return "bad"
```

Exceed any single threshold and the tier drops. A measurement with KL mean 0.008 and flip rate 0.02 is `marginal`, not `good`, because flip rate exceeds the `good` ceiling of 0.01.

## KV-cache tiers

These thresholds apply to `mlx-quant-fidelity kv`. Calibrated for KV-cache quantization on attention-based models and unchanged since 0.1.0.

| tier | KL mean (nats) | KL p99 (nats) | flip rate |
|---|---|---|---|
| good | ≤ 0.01 | ≤ 0.10 | ≤ 0.01 |
| marginal | ≤ 0.10 | ≤ 1.00 | ≤ 0.05 |
| bad | above marginal on any metric | | |

## Weight-quantization tiers (provisional)

These thresholds apply to `mlx-quant-fidelity weights`. **They are provisional**: calibrated from a small set of Llama-3.2 and Qwen2.5 measurements on WikiText-2 short prose, not validated against downstream task accuracy. Weight quantization produces larger distributional shifts than KV quantization, so the ceilings are wider.

| tier | KL mean (nats) | KL p99 (nats) | flip rate |
|---|---|---|---|
| good | ≤ 0.01 | ≤ 0.10 | ≤ 0.05 |
| marginal | ≤ 0.20 | ≤ 1.50 | ≤ 0.25 |
| bad | above marginal on any metric | | |

The word `provisional` appears in every weight-quantization verdict and badge message. These thresholds will be revised as more models and downstream-accuracy data accumulate.

## Badge colors

`--format badge` outputs a shields.io Markdown image line. The color maps directly to the verdict:

| verdict | shields.io color |
|---|---|
| good | brightgreen |
| marginal | yellow |
| bad | red |

The badge message includes the verdict, bit width, corpus, chunk length, and mode (`stress` or `deployment`), so two badges from the same model at different configurations are distinguishable at a glance.

## References

- `src/mlx_quant_fidelity/policy.py`: `_KV_TIERS`, `_WEIGHT_TIERS_v0_2_0`, `verdict_for` — the authoritative threshold values and the three-metric AND logic.
- `src/mlx_quant_fidelity/badge.py`: `_COLORS`, `render_badge_markdown` — the verdict-to-color map and badge renderer.
- [docs/measurement-principles.md](measurement-principles.md) — how each metric is computed.
- [docs/ranking-principles.md](ranking-principles.md) — how verdicts interact with `compare --min-tier`.
