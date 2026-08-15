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

Both passes receive the same `inp`. This is teacher-forced scoring: the corpus tokens drive both runs, so the two passes see the same context at every position. In the KV probe, the measured intervention also includes mlx-lm's different quantized-attention path; the section on limitations below describes that bundle.

Generation would break this. The moment quantization changes a sampled token, the next pass receives different input, and the divergence compounds. What you end up measuring is trajectory drift, not cache cost. The llama.cpp `--kl-divergence-base` flag works the same way: it scores a forward pass over fixed text, not a generation.

Logits are large. A single position over a 128k-token vocabulary is half a megabyte in fp32. The chunk loop collapses logits to per-position scalars (`kl_divergence`, `top_token_flips`, `token_nll`) and calls `mx.eval` before moving on, letting the vocab-wide tensors go out of scope. Accumulating them across a corpus would require roughly 125 GB — four times the machine. Holding full next-token distributions for the corpus would require roughly 125 GB in fp32 — four times the machine — so each position is reduced to scalars as the probe goes, keeping memory flat across the corpus.

Two modes are available. Stress mode (`quantize_start=0`, the default): quantization begins at token 0 and both caches start empty, so the probe measures pure quantizer cost from the first position. Deployment mode (`quantize_start > 0`): the first N positions are computed with a full-precision cache, then the entire stored cache converts at position N. The section at the end of this document covers what each mode measures and where the numbers stop being comparable.

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

KLD is computed over the full vocabulary, not truncated to the reference's top-k. Top-k storage is fine for flip rate and top-k overlap, but truncation does not preserve full-vocabulary `KL(P_ref || Q_quant)`. Truncating outright and renormalizing the retained probabilities are two different approximations, and neither has a guaranteed bias direction.

The report includes mean, median, p99, and max. The tail is where quantization tends to hurt quietly: a model with a healthy mean KLD can still have occasional positions where it is badly wrong. Ranking and domination in `compare` use mean KLD only; read the p99 column directly if the tail matters for your use case.

## Reproducibility controls

The probe is teacher-forced and never calls a sampler. Corpus tokens drive both forward passes, so sampler seeds and temperature do not control the measurement.

The metrics widen logits to fp32 before KL arithmetic and the argmax comparison (`metrics/flip.py`):

```python
ref_top = mx.argmax(ref_logits.astype(mx.float32), axis=-1)
quant_top = mx.argmax(quant_logits.astype(mx.float32), axis=-1)
```

Widening stabilizes metric arithmetic, but it cannot recover precision lost while the model computed the logits or make an existing finite-valued argmax more deterministic. The `mx.eval` call after each chunk bounds the lazy graph and materializes the reduced outputs before the per-chunk caches are dropped. That is a memory-lifetime boundary, not a cross-run determinism guarantee. Reports record the model revisions and software versions; bitwise reproducibility across releases or devices is not promised.

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

The guard triggers when both metrics are exactly zero, not just near-zero. That outcome is indistinguishable from a bypass without further diagnosis — for example, a cache that was never quantized or a configuration dead-end. Reporting it as a valid fidelity score would therefore be unsafe.

## What the numbers don't say

KLD measures how much the quantized distribution differs from the reference on the evaluation corpus. It does not measure downstream task accuracy. *Accuracy Is Not All You Need* (arXiv:2407.09141) shows that aggregate benchmark accuracy can hide answer flips and reports worse MT-Bench results for compressed models on a free-form generative task. It does not evaluate long-context or code workloads, so the short WikiText-2 windows here provide no evidence about either domain. Every report records the corpus, token count, and mode so the number carries its own qualification.

Perplexity delta is a related but distinct signal: it scores the realized next corpus token (`token_nll = -log softmax(logits)[target]`), while mean KLD measures full-vocabulary drift. They correlate but can diverge when the reference distribution doesn't concentrate on the observed token. See [docs/ranking-principles.md](ranking-principles.md) for how perplexity delta interacts with ranking.

In the mlx-lm 0.31.x path used by the current reports, the quantized run composes attention from two `mx.quantized_matmul` calls against packed Keys and Values. The reference uses standard SDPA. The measured drift bundles the quantizer's numerical error with the different attention path's numerics. That is the real end-to-end cost of running the model, but the report says so rather than attributing everything to the quantizer alone. See [docs/ranking-principles.md](ranking-principles.md) for how this interacts with Pareto ranking.

## Stress vs deployment — what each number measures

`--quantize-start N` computes the first N prediction positions while both runs use full-precision caches. At the boundary, the quantized run converts its entire stored cache, including the prefix, in one step. Later entries are quantized as they are written. The reported metrics exclude the first N positions because their logits were computed before conversion and match the reference.

Post-boundary positions therefore read through a quantized prefix as well as quantized new entries. That storage state resembles stress mode, but the computation history differs: deployment mode produced the prefix activations while attention still used a full-precision cache. The committed deployment sample covers 1,020 positions across four chunks, while the stress samples cover 51,100 positions across 100 chunks. They are not a matched comparison and do not establish equal drift.

Deployment mode exercises the `to_quantized` conversion path that mlx-lm uses in practice. Stress mode quantizes from an empty cache and never touches the path that converts existing full-precision entries. A quantizer that behaves differently on pre-filled data is therefore invisible to stress mode. Deployment mode also preserves the exact pre-boundary computation history, even though it does not preserve the prefix's storage type after conversion.

What these numbers do not cover: the probe's 512-token chunk window is not a long document. mlx-lm's default delays conversion until 5000 tokens, but it then quantizes those stored prefix entries too. The probe measures post-boundary cost one chunk at a time. Deployment numbers are a per-chunk proxy, not a real-deployment average over long-form generation.

## Drift by position depth

Every stress-mode report over a fixed-window corpus also buckets its per-position KLD by depth. The scored positions in each chunk split into eight equal-width ranges (`metrics/depth.py::bucket_by_depth`), and each range's mean and p99 KLD are pooled across every scored chunk. A model whose quantizer cost grows the further it reads into a window shows a rising KL mean from the first bucket to the last; a model whose cost doesn't depend on depth shows a flat table.

The table only appears in stress mode, and only when every scored chunk carries the same number of positions. Deployment mode's post-boundary region isn't a fixed width across configurations, and pooling an unequal-length corpus by position index would conflate depth with wherever a chunk happens to end. When the scored chunks don't match in length, the probe drops the table and adds a report warning that points back to this section instead of publishing a number that misrepresents depth.

Depth buckets get more informative as the window grows, so `kv` and `compare kv` take a `--chunk-length` option (default 512, capped at 4096). Paired fp32 logits for one chunk scale with window length times vocabulary size, so a longer window costs proportionally more memory — both runs hold a full logits tensor at once before it collapses to per-position scalars. Measured peaks on Llama-3.2-1B-4bit at kv4, group size 64 (Apple M1 Max, 32 GB; reproducer: `scripts/spike_long_window_memory.py`):

| chunk length | peak memory |
|---|---|
| 512 | 2.27 GiB |
| 1024 | 3.86 GiB |
| 2048 | 7.09 GiB |
| 4096 | 13.53 GiB |

4096 is a hard ceiling — `chunk_length` above it raises before a model loads. Below the ceiling, the CLI adds a warning once the estimated logits footprint for the chosen window passes 4 GiB, using a slope calibrated against the measurements above.

Those measurements come from a model with a vocabulary of about 128,000 tokens, and because the footprint scales with vocabulary as well as window length, a model with twice that vocabulary needs roughly twice the memory at the same window. The 4096 ceiling on its own is therefore not a sufficient guard. Before scoring a window, the probe also compares its estimated logits footprint against the memory cap it installs on the current device, and refuses any window that would claim more than a conservative fraction of that cap instead of warning once the memory is already committed. On a large-vocabulary model that refusal can land on a window the table above shows as comfortable; lower `--chunk-length` until it fits. The fraction is deliberately cautious, because the estimate covers the paired logits alone while a real run also holds the model weights and the cache beside them.

On the models sampled so far, at windows up to 4096 tokens on WikiText-2, the depth curve is close to flat: 4-bit and 8-bit KV cost about the same near position 60 as near position 4000. That is not evidence that KV quantization is depth-insensitive in general. It's what these specific checkpoints show, on this corpus, at these window lengths — and short-prose windows this size are still well short of the context lengths where other work has found the effect. Depth sensitivity has been reported at longer context elsewhere: one study finds KV-quantization drift widening past roughly 4k tokens of context (arXiv:2607.05399), and another traces error accumulation over long reasoning-style generations (arXiv:2606.03458).

Neither external result is directly comparable to the table above. Both study generation over spans longer than this probe's 4096-token ceiling, and this probe's depth table comes from teacher-forced windows: it measures attention over a longer *quantized cache*, not the compounding error of a model sampling its own tokens over many steps. A flat curve at 4096 says the quantizer's cost doesn't grow with attention span in that range on these checkpoints — it says nothing about what happens once a model is generating against its own quantized history for tens of thousands of tokens.

## References

- `probes/kv.py`: `_score_chunk`, `score_kv_config` — teacher-forced paired KV scoring.
- `metrics/kl.py`: `kl_divergence` — full-vocab KLD, fp32, `log_softmax` + `mx.where` zero-guard.
- `metrics/flip.py`: `top_token_flips` — fp32 argmax disagreement.
- `metrics/perplexity.py`: `token_nll` — `-log softmax(logits)[target]`, fp32.
- `probes/_paired.py`: `_check_exact_zero` — `ExactZeroError` on exact-zero KLD and flip.
- `metrics/depth.py`: `bucket_by_depth` — equal-width depth buckets pooled across scored chunks.
- `scripts/spike_long_window_memory.py` — the chunk-length memory measurement above.
- llama.cpp `llama-perplexity --kl-divergence-base` — the KLD direction convention this tool follows.
- *Accuracy Is Not All You Need* — arXiv:2407.09141.
- KV-quantization drift at longer context — arXiv:2607.05399.
- Error accumulation over long reasoning generations — arXiv:2606.03458.
