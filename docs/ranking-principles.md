# How to read the Pareto

`mlx-quant-fidelity compare` ranks a set of quantizations on a Pareto frontier: quality on one axis, memory cost on the other. This document explains what that means, how each axis is computed, and where the approach falls short.

## The question

When you have several quantizations of the same model — q4, q6, and q8 weights, or 4-bit and 8-bit KV caches with different group sizes — you want to know which gives the best quality for the fewest bytes.

Raw metric sorting does not answer that. Sorting by KL divergence alone picks the best quality regardless of cost, which tells you nothing you didn't already know (more bits = lower KLD). Sorting by size alone ignores quality. Neither tells you whether q6 is a meaningful step up from q4 or barely different.

The useful question: are there any quantizations on the list that are both worse quality and more expensive than another option on the same list? Those are strictly dominated — you would never choose them. The remaining candidates form the Pareto frontier, where every point gives up something compared to its neighbors.

## The quality axis

Quality is mean full-vocab KL divergence in nats, computed as `KL(P_full ‖ Q_quant)` — the reference distribution leads. This follows the llama.cpp `--kl-divergence-base` convention.

The KLD is fp32, full-vocabulary (not truncated to top-k), computed over fixed-length corpus chunks at temperature 0. Each position produces one KLD scalar; the reported mean averages across all positions in the run.

The tool also reports KL p99 and KL max for every target, so you can see whether a quantization has occasional bad positions even when its mean looks fine. Ranking and domination are decided on mean KLD only, so a target with a favorable mean and an ugly tail will not be penalized in its Pareto placement. If the tail matters for your use case, read the p99 column directly.

## The cost axis

For weight quantization, cost is total on-disk model bytes as reported by the model repository.

For KV quantization, cost is KV-cache bytes per token:

```
2 · n_layers · n_kv_heads · head_dim · (bits/8 + 4/group_size)
```

The `4/group_size` term is the per-group overhead: a fp16 scale (2 bytes) and a fp16 bias (2 bytes), amortized over `group_size` elements. This formula applies to the bf16/fp16 KV path used by MLX's `QuantizedKVCache` on Apple Silicon. An fp32 KV cache would carry `8/group_size` bytes of overhead per element, but that path is not currently supported.

The formula is exact for the supported path. Group overhead matters for ranking: a smaller group size improves accuracy but increases bytes per token, so two configurations at the same bit width can land at different positions on the cost axis. Ignoring group size would misrank them.

Memory normalization is what makes the comparison meaningful. Without it, comparing q4 and q8 on quality alone is tautological — q8 wins by definition because the cost difference is not part of the comparison. The question is whether q8's quality gain is worth the extra bytes.

## Pareto efficiency

Target A dominates target B when A is no worse than B on both axes and strictly better on at least one. The Pareto frontier is the set of non-dominated targets. Dominated targets appear in the comparison report flagged with a dominator's label.

A worked example: suppose you compare four weight quantizations.

| target | KL mean | model size |
|---|---|---|
| q8 | 0.01 | 8.0 GB |
| q6 | 0.04 | 6.2 GB |
| q4 | 0.09 | 4.2 GB |
| q4-bad | 0.20 | 4.3 GB |

q4-bad has worse quality than q4 (KLD 0.20 vs 0.09) and costs more bytes (4.3 vs 4.2 GB). q4 dominates q4-bad: you would never choose q4-bad. The frontier is {q4, q6, q8}. Each is the cheapest way to achieve at least that quality level; none dominates another.

The tool does not pick a single "knee" from the frontier automatically. Where the knee sits depends on how you weight quality against cost, and choosing that weighting is a value judgment the tool deliberately leaves to you.

## Budget-driven selection

Pass a budget constraint if you want a recommendation:

- `--max-kld X` picks the cheapest frontier target whose mean KLD is at or below X.
- `--min-tier good|marginal|bad` picks the cheapest frontier target whose verdict clears the requested tier.

In both cases the pick is restricted to the Pareto frontier. A dominated target is never recommended even if it clears the budget.

For `--max-kld`, this restriction costs nothing: if a dominated target clears the KLD threshold, its dominator has equal or lower mean KLD and is both on the frontier and cheaper. So the frontier always contains the best `--max-kld` pick.

For `--min-tier`, the situation is different. Tier qualification uses the full verdict — mean KLD, p99, and flip rate together — while domination is decided on mean KLD alone. A dominated target can pass the tier check while its frontier dominator fails it (because the dominator has a worse p99 or flip rate despite its mean KLD being no worse). When that happens, the tool returns no pick rather than recommending a dominated target. The comparison table shows tiers for all targets so you can decide.

To pick a threshold: start by reading the comparison table. If all frontier targets have acceptable quality, take the cheapest one. To put a number on "acceptable," use `--max-kld` with a threshold from the reference runs in the README — for example, 0.01 nats is roughly in the range of 8-bit weight quantization on Llama-3 models.

## Interpretation and limits

Fidelity numbers are corpus- and context-length-specific. The measurements use WikiText-2 short prose at temperature 0. The paper this work builds on, *Accuracy Is Not All You Need* (arXiv:2407.09141), documents that short-prose distributional drift under-predicts degradation on long-context tasks and code. Every report records the corpus and token count so the number is not read as a bare score without context.

Perplexity delta is a related but distinct signal. Mean KL measures full-vocabulary drift
weighted by the reference distribution at each position; perplexity delta measures how the
quantized distribution scores the realized next corpus tokens (-log q(target)). On a well-fit
model the two usually move together, but perplexity delta is not an independent full-vocab
confirmation of KL and is not algebraically identical to it — they can diverge when the
reference does not place its mass on the observed token. Perplexity delta is reported for
continuity with llama.cpp.

In `compare kv`, the quantized run uses `mx.fast.quantized_scaled_dot_product_attention` while the reference uses standard SDPA. The measured drift bundles the quantizer's numerical error with the quantized-attention kernel's numerics. That is the real end-to-end cost of deploying the model — the right thing to measure for a deployment decision — but the report says so explicitly rather than attributing everything to the quantizer alone.

The comparison is only meaningful when all targets derive from the same base model. The tool checks basic architecture and vocabulary compatibility for weight targets, but it does not verify that all quantizations share the same original weights.

Domination and frontier placement are decided on mean KLD alone. A ranking that incorporates p99, flip rate, and perplexity delta as separate objectives is a possible future direction. For now, use the table columns directly if specific statistics matter.

`compare weights` reloads the reference model once per target. N targets take roughly N times as long as a single `weights` run. `compare kv` loads the model once and runs all KV configurations in the same process, so its overhead per configuration is much smaller.

## References

- llama.cpp `llama-perplexity --kl-divergence-base` — the KLD direction convention this tool follows.
- *Accuracy Is Not All You Need* — arXiv:2407.09141. Documents the gap between distributional fidelity and downstream task accuracy.
- [Pareto efficiency](https://en.wikipedia.org/wiki/Pareto_efficiency) — the Wikipedia article is a reasonable starting point for the underlying concept.
