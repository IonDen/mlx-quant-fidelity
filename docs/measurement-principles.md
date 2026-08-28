# How fidelity is measured

You have a drift number and you want to know exactly what was run to produce it.

`mlx-quant-fidelity` measures the gap between a quantized model and its reference by running both on the same tokens and comparing their next-token distributions position by position.

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

Logits are large. A single position over a 128k-token vocabulary is half a megabyte in fp32, and holding full next-token distributions for the whole corpus would require roughly 125 GB — four times the machine. The chunk loop collapses logits to per-position scalars (`kl_divergence`, `top_token_flips`, `token_nll`) and calls `mx.eval` before moving on, letting the vocab-wide tensors go out of scope, so memory stays flat across the corpus.

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

Post-boundary positions therefore read through a quantized prefix as well as quantized new entries. That storage state resembles stress mode, but the computation history differs: deployment mode produced the prefix activations while attention still used a full-precision cache. The committed deployment sample and the stress samples run the same corpus geometry — 100 chunks of 512 tokens — but they do not score the same positions. At `quantize_start=256` the deployment run reports 25,500 positions, the tail of each window, while stress reports all 51,100. The deployment average is therefore taken over positions that sit deeper in the window than stress's, on top of the differing computation history above, so reading the two numbers side by side does not establish equal drift.

For the stock cache, deployment mode exercises the `to_quantized` conversion path that mlx-lm uses in practice; for a third-party cache the stored prefix is replayed into a fresh cache, which no shipped deployment does, so those deployment numbers describe a hypothetical deployment. Stress mode quantizes from an empty cache and never touches the path that converts existing full-precision entries. A quantizer that behaves differently on pre-filled data is therefore invisible to stress mode. Deployment mode also preserves the exact pre-boundary computation history, even though it does not preserve the prefix's storage type after conversion.

What these numbers do not cover: the probe's 512-token chunk window is not a long document. mlx-lm's own generate command delays conversion until token 5000, but it then quantizes those stored prefix entries too. The probe measures post-boundary cost one chunk at a time. Deployment numbers are a per-chunk proxy, not a real-deployment average over long-form generation.

## Measuring a third-party cache

The KV probe scores any cache implementation that satisfies its method protocol, not only mlx-lm's `QuantizedKVCache`. What differs between methods is the attention path each one rides, and that difference is exactly what the drift number bundles:

| method | attention path | what the drift measures |
|---|---|---|
| stock | two `mx.quantized_matmul` calls against the packed K/V, then a precise softmax | quantizer error plus the quantized-attention kernel's own numerics (see "What the numbers don't say" above) |
| turboquant | dequantizes the cache on fetch and runs standard SDPA, the same kernel the reference run uses | quantizer error alone |
| turboquant-vonly | K stays in a plain fp16 cache and rides standard SDPA untouched; only V dequantizes on fetch through the port's V-only cache, also onto standard SDPA | quantizer error alone, V-side only — K is never quantized |
| affine | dequantizes on fetch (`mx.quantize`/`mx.dequantize`, independent K and V bit widths) and rides standard SDPA — no shipped cache runs this layout | quantizer error alone, for a per-side K/V bit width that exists nowhere else in this table |

Dequantizing on fetch has a memory cost the stored-bytes column doesn't show. TurboQuant's working set is four step-256-padded buffers shaped `(n_layers, n_kv_heads, window, head_dim)`, not two: the prefill forward's transient full `all_k` and `all_v` dequantizations (freed once that forward returns) plus the two working copies (`_k_deq_buf`, `_v_deq_buf`) the port keeps resident after every `update_and_fetch`. Derived from those four buffers on Llama-3.2-1B geometry, that puts TurboQuant's resident memory at roughly 2.3× an fp16 cache. `turboquant-vonly` carries only the V-side half of that working set — K stays a plain fp16 cache riding standard SDPA untouched, so only V's prefill transient and its single retained `_v_deq_buf` count. `affine` carries none of it: its per-fetch dequantization covers the exact window rather than a step-256-padded buffer, and the transient is freed immediately after each fetch, so its working set scales with chunk length alone.

The chunk-length memory ceiling was originally validated on the stock cache alone. A method name outside `RECEIPTED_METHODS` in `probes/kv.py` still narrows that margin, and the report warns when it runs above the 512-token default; the next section records which method names now carry a measured long-window receipt of their own.

Deployment mode for a third-party cache can't reuse mlx-lm's in-place `to_quantized` conversion, since that method doesn't exist on a third-party cache. Instead, each layer's full-precision state is sliced to `offset` (the stored tokens, never the step-padded buffer mlx-lm allocates ahead of use), replayed through a fresh quantized cache with a single `update_and_fetch`, and then trimmed with `trim(0)` to drop the dequantized working buffers the port retains after that call.

Every report carries four fields identifying what produced it: `kv_method` (`"stock"`, `"turboquant"`, `"turboquant-vonly"`, or `"affine"`), `kv_method_params` (the method's own parameters — `bits`/`group_size` for stock, `bits`/`seed` for TurboQuant, `v_bits`/`seed` for TurboQuant V-only, `k_bits`/`v_bits`/`group_size` for affine), `kv_method_provenance` (package name, installed version, and, for a git-pinned port, the installed and pinned commit hashes), and `measured_kv_bytes_per_token` (the stored bytes per token actually measured at runtime, alongside the analytic cost `compare` ranks on). A method may also carry its own seeded rotation: TurboQuant seeds its Key rotation from `seed` and its Value rotation from `seed + 1`, both recorded in `kv_method_provenance`.

## Decomposing bundled and quantizer-only drift

Every method above dequantizes on fetch except stock, so its own reported number is already the quantizer's cost alone — there is nothing bundled to separate. Stock is different: its deployed path bundles quantizer error with the numerics of its own two-`mx.quantized_matmul` attention. `kv --control` runs a third forward per chunk to separate them: the same corpus tokens, through a cache that dequantizes on fetch and rides standard SDPA at the method's own bits and group size. That control cache is exactly `AffineKVMethod(k_bits=bits, v_bits=bits, group_size=group_size)` — the class `compare kv` ranks `affine:k:v` configurations with, called here with matching K and V bit widths. The report then carries two lanes: `kl` (the deployed path) and `control_kl` (the quantizer-only path), plus a `drift_footing` field that names which lane the report's own `verdict` grades — always `"bundled"` for a stock report, control lane or not.

`compare kv` ranks every target on the quantizer-only number. A method that already dequantizes on fetch ranks on its own `kl`; stock ranks on its control lane's `kl` instead of its bundled one. The comparison table still shows stock's bundled number in a separate `bundled KL` column, so the deployed-path cost isn't hidden — it's just not what decides the Pareto frontier or a `--max-kld`/`--min-tier` pick.

On the committed sample, stock `4:64` measures a quantizer-only mean KL of 0.1485 and a bundled mean KL of 0.1477 — the bundled path landing slightly *lower*. That is not evidence the quantized-attention kernel improves fidelity, and the gap is not a kernel-purity score in general. Both numbers are read off the model's final logits after all 16 of Llama-3.2-1B's layers, and each lane is an independent 16-layer forward: the bundled lane's every attention call ran the two-quantized-matmul composition, the control lane's every attention call ran standard SDPA on dequantized tensors. A per-layer numerical difference between the two paths does not survive to the final logits unchanged — softmax, residual adds, and RMSNorm at every later layer can amplify it, damp it, or flip its sign before it reaches a scored position. Subtracting the two end-to-end KL numbers mixes that compounding in with whatever the attention kernel itself contributed, so a small measured gap says the two paths land close together at this configuration and model scale — not that the kernel's own contribution is small, negative, or absent in general. A different model, bit width, or corpus could compound the same per-layer difference into a larger gap, or a reversed one.

## Drift by position depth

Every stress-mode report over a fixed-window corpus also buckets its per-position KLD by depth. The scored positions in each chunk split into eight equal-width ranges (`metrics/depth.py::bucket_by_depth`), and each range's mean and p99 KLD are pooled across every scored chunk. A model whose quantizer cost grows the further it reads into a window shows a rising KL mean from the first bucket to the last; a model whose cost doesn't depend on depth shows a flat table.

The table only appears in stress mode, and only when every scored chunk carries the same number of positions. Deployment mode's post-boundary region isn't a fixed width across configurations, and pooling an unequal-length corpus by position index would conflate depth with wherever a chunk happens to end. When the scored chunks don't match in length, the probe drops the table and adds a report warning that points back to this section instead of publishing a number that misrepresents depth.

Depth buckets get more informative as the window grows, so `kv` and `compare kv` take a `--chunk-length` option (default 512, capped at 4096). Paired fp32 logits for one chunk scale with window length times vocabulary size, so a longer window costs proportionally more memory — both runs hold a full logits tensor at once before it collapses to per-position scalars. Measured peaks on Llama-3.2-1B-4bit at kv4, group size 64 (Apple M1 Max, 32 GB; reproducer: `scripts/spike_long_window_memory.py`):

| chunk length | peak memory | source |
|---|---|---|
| 512 | 2.27 GiB | committed sample report |
| 1024 | 3.86 GiB | reproducer run, no committed artifact |
| 2048 | 7.09 GiB | reproducer run, no committed artifact |
| 4096 | 13.53 GiB | committed sample report |

The two confirmed rows are the `peak_memory_bytes` recorded in `_artifacts/samples/llama-3.2-1b-4bit-kv4.json` and `llama-3.2-1b-4bit-kv4-cl4096.json`. The 1024 and 2048 rows were measured by the reproducer above but its output directory is not committed, so they cannot be checked against the repository.

That table covered the stock cache only. The same spike script takes a `--method` flag (`stock`, `turboquant`, `turboquant-vonly`, `affine`, `stock-control` — stock run with the quantizer-only control forward), and each lane writes its own resumable artifacts under `_artifacts/spike_long_window/<lane>/`, keeping the `stock` lane's flat legacy path so the baseline record above stays valid. None of the following numbers is a committed sample; all four lanes are reproducer-only runs, on the same model, revision, and hardware as the table above (2026-08-26):

| lane (config) | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| turboquant (`turboquant:4`) | 2.27 GiB | 3.90 GiB | 7.15 GiB | 13.65 GiB |
| turboquant-vonly (`turboquant-vonly:4`) | 2.40 GiB | 3.91 GiB | 7.17 GiB | 13.70 GiB |
| affine (`affine:8:4`) | 2.27 GiB | 3.87 GiB | 7.09 GiB | 13.54 GiB |
| stock + `--control` (`4:64`) | 2.38 GiB | 3.86 GiB | 7.09 GiB | 13.53 GiB |

Every cell above finished at `rc=0` — no watchdog abort, no wall-clock timeout, and no pre-flight gate refusal, at any length on any lane. Each lane's 4096 peak also lands under the gate's own per-lane estimate: turboquant's gate estimate at 4096, for example, is about 13.67 GiB of paired logits plus roughly 0.25 GiB of method working set, against a measured 13.65 GiB. `MAX_CHUNK_LENGTH = 4096` therefore re-validates, at this vocabulary, for every method name in `RECEIPTED_METHODS` — `stock`, `turboquant`, `turboquant-vonly`, and `affine` — and for `stock` run with `--control`. The same caveat as the table above still applies: these peaks scale with vocabulary, so a larger-vocabulary model needs its own measurement before the ceiling can be trusted at 4096.

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
- `probes/kv_methods.py`: `StockKVMethod`, `TurboQuantKVMethod`, `TurboQuantVOnlyKVMethod`, `AffineKVMethod` — the cache-method protocol, provenance, and prefix-replay logic behind "Measuring a third-party cache"; `StockKVMethod.control_method` — the quantizer-only control lane behind "Decomposing bundled and quantizer-only drift."
- `scripts/spike_long_window_memory.py` — the chunk-length memory measurement above, run per `--method` lane.
- llama.cpp `llama-perplexity --kl-divergence-base` — the KLD direction convention this tool follows.
- *Accuracy Is Not All You Need* — arXiv:2407.09141.
- KV-quantization drift at longer context — arXiv:2607.05399.
- Error accumulation over long reasoning generations — arXiv:2606.03458.
