# How fidelity is measured

`mlx-quant-fidelity` measures the gap between a quantized model and its reference by running both on the same tokens and comparing their next-token distributions position by position. What follows explains the mechanics, the implementation choices, and where the numbers stop being useful.

## Teacher-forced paired scoring, not generation

For each fixed-length corpus chunk the probe calls the model twice on identical input tokens — once with a full-precision cache, once with a quantized one — and compares the output distributions directly.

`_score_chunk` in `probes/kv.py`:

```python
inp = ids[None, :-1]
targets = ids[1:]
ref_logits = model(inp, cache=ref_cache)[0].astype(mx.float32)
quant_logits = model(inp, cache=quant_cache)[0].astype(mx.float32)
```

Both passes receive the same `inp`; only the cache differs. This is teacher-forced scoring: the corpus tokens drive both runs, so the two passes see the same context at every position.

Generation would break this. The moment quantization changes a sampled token, the next pass receives different input, and the divergence compounds. What you end up measuring is trajectory drift, not cache cost. The llama.cpp `--kl-divergence-base` flag works the same way: it scores a forward pass over fixed text, not a generation.

Logits are large. A single position over a 128k-token vocabulary is half a megabyte in fp32. The chunk loop collapses logits to per-position scalars (`kl_divergence`, `top_token_flips`, `token_nll`) and calls `mx.eval` before moving on, letting the vocab-wide tensors go out of scope. Accumulating them across a corpus would require roughly 125 GB — four times the machine. Streaming is not an optimization; it is what makes the measurement possible.

The KV probe ships only in stress mode (`quantize_start=0`): quantization begins at token 0, so both caches start empty and the probe measures pure quantizer cost from the first position. Deployment mode (`quantize_start > 0`) is not implemented.

## Full-vocab KL divergence and the tail

The KLD at each position is `KL(P_full ‖ Q_quant)` — the reference distribution leads. This follows the llama.cpp convention. The implementation in `metrics/kl.py`:

```python
log_p = nn.log_softmax(ref_logits.astype(mx.float32), axis=-1)
log_q = nn.log_softmax(quant_logits.astype(mx.float32), axis=-1)
p = mx.exp(log_p)
terms = mx.where(p > 0, p * (log_p - log_q), mx.array(0.0, dtype=mx.float32))
return terms.sum(axis=-1)
```

The zero-probability policy is explicit: where `P = 0`, the term is zero (`0 · log 0 := 0` by convention). There is no epsilon smoothing. If `P > 0` and `Q` approaches zero, the term approaches positive infinity — the honest cost of the reference assigning probability to a token the quantized model nearly rules out. In practice, softmax over real logits is always strictly positive, so finite results are the norm.

KLD is computed over the full vocabulary, not truncated to the reference's top-k. Top-k storage is fine for flip rate and top-k overlap, but truncating for KLD discards the tail and systematically underreports the divergence.

The report includes mean, median, p99, and max. The tail is where quantization tends to hurt quietly: a model with a healthy mean KLD can still have occasional positions where it is badly wrong. Ranking and domination in `compare` use mean KLD only; read the p99 column directly if the tail matters for your use case.

## Determinism at temperature 0

The probe runs at temperature 0. There is no sampler RNG: `mlx-lm`'s `make_sampler` returns argmax when `temp=0`, and the `mx.random` samplers only run when `temp > 0`. Seeding a random state does nothing here.

The actual determinism levers are fp32 logits and per-chunk graph materialization. Flip rate is computed on fp32 logits before argmax (`metrics/flip.py`):

```python
ref_top = mx.argmax(ref_logits.astype(mx.float32), axis=-1)
quant_top = mx.argmax(quant_logits.astype(mx.float32), axis=-1)
```

Using fp16 logits would cause near-ties to resolve differently across runs due to rounding. The `mx.eval` call after each chunk bounds the lazy graph and ensures the computation is materialized before the cache is reset and cleared, preventing graph state from leaking between chunks.

## The exact-zero guard

A run that returns exactly zero KLD and exactly zero flip rate raises `ExactZeroError` rather than silently reporting perfect fidelity. The check is in `probes/_paired.py`:

```python
def _check_exact_zero(*, kl_mean: float, flip_rate: float, context: str) -> None:
    if kl_mean == 0.0 and flip_rate == 0.0:
        raise ExactZeroError(
            f"KLD and flip were exactly 0 — {context}. "
            "This is never a silent 'perfect fidelity'."
        )
```

The guard triggers when both metrics are exactly zero, not just near-zero. When this happens it almost always means quantization never engaged — a cache that was bypassed or never engaged, or a configuration error. Reporting zero as a valid fidelity score in these cases would be wrong.

## What the numbers don't say

KLD measures how much the quantized distribution differs from the reference on the evaluation corpus at temperature 0. It does not measure downstream task accuracy. The paper this tool builds on, *Accuracy Is Not All You Need* (arXiv:2407.09141), shows that short-prose distributional drift underestimates degradation on long-context tasks and code. Every report records the corpus, token count, and mode so the number carries its own qualification.

Perplexity delta is a related but distinct signal: it scores the realized next corpus token (`token_nll = -log softmax(logits)[target]`), while mean KLD measures full-vocabulary drift. They correlate but can diverge when the reference distribution doesn't concentrate on the observed token. See [docs/ranking-principles.md](ranking-principles.md) for how perplexity delta interacts with ranking.

In `compare kv`, the quantized run uses `mx.fast.quantized_scaled_dot_product_attention` while the reference uses standard SDPA. The measured drift bundles the quantizer's numerical error with the quantized-attention kernel's numerics. That is the real end-to-end cost of running the model, but the report says so rather than attributing everything to the quantizer alone. See [docs/ranking-principles.md](ranking-principles.md) for how this interacts with Pareto ranking.

## References

- `probes/kv.py`: `_score_chunk`, `score_kv_config` — teacher-forced paired KV scoring.
- `metrics/kl.py`: `kl_divergence` — full-vocab KLD, fp32, `log_softmax` + `mx.where` zero-guard.
- `metrics/flip.py`: `top_token_flips` — fp32 argmax disagreement.
- `metrics/perplexity.py`: `token_nll` — `-log softmax(logits)[target]`, fp32.
- `probes/_paired.py`: `_check_exact_zero` — `ExactZeroError` on exact-zero KLD and flip.
- llama.cpp `llama-perplexity --kl-divergence-base` — the KLD direction convention this tool follows.
- *Accuracy Is Not All You Need* — arXiv:2407.09141.
