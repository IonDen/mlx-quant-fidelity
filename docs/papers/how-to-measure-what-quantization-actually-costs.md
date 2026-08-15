# How to measure what quantization actually costs

*A research memo from `mlx-quant-fidelity`: the design of an honest quantization-fidelity
measurement, and the specific ways the obvious harness lies*

Ask what 4-bit quantization costs a model and the obvious measurement plan writes itself:
generate some text with and without quantization, compare, seed everything for reproducibility,
keep the logits around for analysis, and report an average score. Every step of that plan hides
a failure mode. Some bury damage; others measure the wrong quantity or let the harness pass
without exercising the feature. Any of them can convert an open question into false confidence.

This memo is the methods companion to a survey of
[low-bit KV caches on MLX](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/papers/low-bit-kv-caches-on-mlx-what-exists-and-what-is-missing.md).
That memo covers what exists and what it costs; this one covers how the costs were measured and
why the harness looks the way it does. The mechanics of the shipped tool are documented in
[measurement principles](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/measurement-principles.md)
and [ranking principles](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/ranking-principles.md);
this memo builds on that tour rather than repeating it. Its subject is the failure catalogue: for each design
choice, the concrete way a naive harness fails and the committed evidence that the corresponding
guard earns its place.

The memo is documentary. Every reported measurement result comes from a committed, linkable report in the
[`mlx-quant-fidelity`](https://github.com/IonDen/mlx-quant-fidelity) repository at v0.4.0;
nothing was rerun for this write-up. The reports record the software versions (mlx 0.31.2,
mlx-lm 0.31.3); the hardware (an M1 Max with 32 GB unified memory) is stated by the author
rather than captured in the report files. Links into this repository's source and evidence are
pinned to the public v0.4.0 tag. Product documentation and companion memos point at their home
on each repository's `main` branch.

None of the individual ingredients is novel — teacher-forced scoring and
KL-divergence-against-a-reference are established practice in
[llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md) and
the evaluation literature. The contribution is the MLX-side design and the catalogue of ways
the measurement breaks.

## 1. Score the corpus, not a generation

The intuitive experiment — generate with the quantized model, generate with the reference,
compare the outputs — fails in two opposite directions at once.

In one direction it over-reports, noisily. The first position where quantization changes a
sampled token changes the *input* to every later position. The two runs now walk different
texts, and the comparison measures compounding trajectory drift, not the per-position cost of
the quantization. The number depends on where the first flip happened to land, which makes it
unstable across prompts and useless for comparing configurations.

In the other direction it under-reports, silently. A popular variant of the generation test
checks whether greedy decode reproduces the reference's text exactly. That oracle is binary per
position and says nothing about margin: the argmax token can stay identical while the
probability mass underneath it moves substantially. A configuration can pass a greedy-equality
check on one prompt and still be measurably degraded at almost every position.

Teacher-forced paired scoring avoids both failures
([`_score_chunk`, probes/kv.py](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/src/mlx_quant_fidelity/probes/kv.py#L93)):
for each fixed-length corpus chunk the model runs twice on *identical* tokens — once with a
full-precision KV cache, once with a quantized one — and the two next-token distributions are
compared position by position. The corpus drives both runs, so the input tokens and trajectory
cannot diverge.
llama.cpp's `--kl-divergence-base` made the same choice for weight quantization: it scores a
forward pass over fixed text against saved reference logits, not one generation against
another. The measured intervention is still a bundle: in the KV probe it includes the
different attention path described in section 6.

## 2. Reduce as you go, or the harness cannot run at all

Comparing full distributions position by position has a memory problem. One position over a
128k-token vocabulary is half a megabyte in fp32. The WikiText-2 test split is on the order of
245,000 positions, so holding the corpus's distributions for analysis is roughly **125 GB** —
about four times the 32 GB machine the committed reports were produced on. Even the capped runs
in this repository (100 chunks, 51,100 scored positions) would need about 26 GB for one model's
distributions at Llama-3.2's 128,256-token vocabulary, on a machine that also holds a 7B
model's weights; Qwen2.5's 151,936-token vocabulary pushes the same run past 31 GB. "Store the logits, then
analyze" is not a design option; it decides whether the harness can run at all.

Two working answers exist. llama.cpp saves the reference logits to disk once and streams the
quantized run against the file. This tool runs both passes in lockstep and never persists
distributions at all: inside the chunk loop, vocab-wide logits are collapsed to per-position
scalars — KL divergence, a top-token flip flag, the target token's negative log-likelihood
([`_reduce_pair`, probes/_paired.py](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/src/mlx_quant_fidelity/probes/_paired.py#L28))
— and the logits leave scope before the next chunk begins.

On MLX the streaming discipline has a second, framework-specific job. MLX is lazily evaluated:
appending un-evaluated arrays to a Python list keeps every vocab-wide intermediate alive in one
growing computation graph, which reproduces the 26 GB problem in lazy form. What actually bounds memory is the per-chunk sequence: `mx.eval` on the chunk's scalars, then
dropping the per-chunk caches, then `mx.clear_cache()`. The committed reports carry the receipts: the Qwen2.5-7B KV runs peak at **6.35 GB**,
and the largest committed measurement — a 7B two-model weight comparison — peaks at
**14.40 GB**. The tool also installs device-derived MLX wired and memory caps before any model
load, at the CLI entry point and inside each Python-API probe. These caps reduce memory pressure;
they are not a hard process-local allocation ceiling. The weight probe adds a separate byte
preflight because two live models can exceed the device working set despite the caps.

![Paired teacher-forced scoring loop: a fixed corpus chunk feeds a reference run with a full-precision KV cache and a quantized run with a quantized KV cache; both produce vocab-wide fp32 logits that are reduced inside the loop to per-position scalars (KL divergence, top-token flip, target negative log-likelihood); the scalars are evaluated with mx.eval, the per-chunk caches are dropped and the cache pool cleared, and the loop advances to the next chunk, so no vocab-wide tensor outlives its chunk](https://raw.githubusercontent.com/IonDen/mlx-quant-fidelity/main/docs/papers/diagrams/paired-scoring-chunk-loop.svg)

*Figure 1. Conceptual flow of the paired teacher-forced probe: one fixed corpus chunk drives a
reference run and a quantized run, their vocab-wide fp32 logits are reduced to per-position
scalars inside the loop, and the chunk's caches and logits are dropped before the next chunk
begins. The figure shows the shipped mechanism's structure; the measured memory evidence is the
6.35 GB and 14.40 GB peaks reported above. The editable
[PlantUML source](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/papers/diagrams/paired-scoring-chunk-loop.puml)
is published with the SVG.*

## 3. Keep the whole vocabulary, and the whole tail

Two popular shortcuts each hide damage.

The first is truncating to the reference's top-k tokens before computing divergence. Top-k
storage is fine for flip rate or top-k overlap, but it does not preserve the stated
`KL(P_ref || Q_quant)`. Truncating outright and renormalizing the retained probabilities are two
different approximations, and neither has a guaranteed bias direction. The shipped metric avoids
that ambiguity by using the full vocabulary in fp32, with an explicit zero-probability policy
([`kl_divergence`, metrics/kl.py](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/src/mlx_quant_fidelity/metrics/kl.py#L7)):
where the reference assigns zero, the term is zero by the `0 · log 0 := 0` convention. Forward
KL directly penalizes the quantized model for suppressing tokens that retain reference
probability; extra quantized mass in a reference-zero tail matters only indirectly through
normalization. There is no epsilon smoothing, which would cap the penalty as `Q_quant`
approaches zero for an event supported by the reference.

The second shortcut is reporting only the mean. Averages are where quantization damage hides.
The committed
[Qwen2.5-7B 8-bit KV report](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/_artifacts/samples/qwen2.5-7b-4bit-kv8.md)
has a mean KL of **0.0094** nats — comfortably "near-lossless" — while its p99 is **0.14** and
its worst position reaches **13.1** nats. A mean-only report would call that run clean; the
tail says a small fraction of positions are being hit hard. The report does not classify those
positions, so their downstream importance remains unknown. Every report therefore carries mean,
median, p99, and max, and the verdict machinery (section 8) refuses to grade on the mean alone.

## 4. Reproducibility is not a seed

A reproducibility ritual common in evaluation scripts is to seed Python, NumPy, and the
framework RNG. This probe never samples: corpus tokens drive the forward passes directly, so
sampler seeds cannot affect the measurement. Temperature is not a control here either. The
probe's fixed inputs come from the corpus, not from mlx-lm's generation sampler.

The logits are widened to fp32 before KL arithmetic and the argmax comparison
([`top_token_flips`, metrics/flip.py](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/src/mlx_quant_fidelity/metrics/flip.py#L6)).
Widening stabilizes the metric arithmetic, but it cannot recover precision lost while the model
computed the logits, and it does not make an existing finite-valued argmax more deterministic.
Likewise, `mx.eval` is a graph-lifetime and memory boundary, not a determinism guarantee. The
harness controls the corpus tokens, model revisions, and software versions; it does not claim
bitwise reproducibility across releases or devices.

## 5. A measurement must not be able to pass by doing nothing

The most dangerous failure mode a fidelity harness has is silent success: a configuration where
quantization never actually engaged, scored as perfect fidelity. This is not hypothetical. In
mlx-lm 0.31.3, requesting `--kv-bits` on a model whose cache classes lack a conversion path is
a silent no-op — batch, chunked, and composite caches skip quantization with no warning
([generate.py](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/generate.py#L299)). A
naive harness pointed at such a path would run the "quantized" and reference passes on
byte-identical caches and report zero drift — the best possible score, for a measurement that
never happened.

The shipped design refuses this outcome at three layers:

- **Capability is probed by doing, not by looking.** Before scoring, every layer cache's
  `to_quantized` is actually called on the cheap empty cache
  ([`_cache_is_quantizable`, probes/kv.py](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/src/mlx_quant_fidelity/probes/kv.py#L63)).
  An attribute check is not enough: mlx-lm's sliding-window caches *have* the method and raise
  `NotImplementedError` when it runs. A model whose cache cannot quantize produces a clear
  error, never a clean-looking zero.
- **An exact-zero result raises.** A run whose KL divergence and flip rate are both exactly
  zero raises `ExactZeroError` rather than reporting perfect fidelity
  ([`_check_exact_zero`, probes/_paired.py](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/src/mlx_quant_fidelity/probes/_paired.py#L63)),
  because that outcome is indistinguishable from a bypass without further diagnosis. The same
  principle covers configuration dead-ends: a deployment boundary longer than every corpus chunk means
  zero positions were ever quantized, and the run raises `QuantizeStartError` instead of
  averaging over nothing.
- **The test suite proves the metrics can detect damage.** The committed oracle tests
  ([tests/probes/test_kv_oracles.py](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/tests/probes/test_kv_oracles.py))
  pin both directions: a full-precision-vs-full-precision paired run must produce *exactly*
  zero KL, a deliberately corrupted cache injected as the "quantized" path must push KL above a
  floor, and the same corruption applied to both paths must return the measurement to exactly
  zero. These tests show that the metric responds to a known perturbation and returns zero for
  identical paths. They do not prove the converse that every nonzero result certifies end-to-end
  cache engagement; the capability and exact-zero checks are complementary guards against that
  failure.

The failure family is general. A related memo in this series documents a residual-caching
mechanism whose gate
[never engaged on short distilled schedules](https://github.com/IonDen/mlx-teacache/blob/main/docs/papers/why-teacache-does-not-engage-on-short-distilled-schedules.md)
while the surrounding pipeline looked healthy. Any instrument whose "everything is fine"
output is indistinguishable from its "I did nothing" output will eventually report the latter
as the former.

## 6. Say what you actually compared

A fidelity number is a comparison against a reference, and the reference is a choice the report
must surface rather than bury.

Sometimes the reference is not full precision. The committed
[Qwen2.5-7B weight report](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/_artifacts/samples/weights/qwen2.5-7b-4bit-vs-8bit.md)
scores a 4-bit checkpoint against the *8-bit* checkpoint of the same model, and says so in a
call-out line: the measured drift is relative to an already-quantized reference, not to bf16.
Likewise, all of the committed KV-cache runs use 4-bit-weight community checkpoints, with no
fp16-weight control run, so a weight-quantization interaction cannot be excluded from the KV
numbers. Neither fact invalidates the measurements; both change what the numbers mean, which
is why the report format carries them instead of leaving them to the reader's assumptions.

The measured quantity is also a bundle, and the honest move is to name the bundle rather than
claim an isolation that was never performed. In the KV probe, the quantized run rides mlx-lm's
quantized-attention composition — two `mx.quantized_matmul` calls around a softmax
([base.py](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/base.py#L64)) —
while the reference run rides the fused standard SDPA. The reported drift therefore bundles
the quantizer's rounding error with the numerics of a different attention code path. That
bundle is precisely the end-to-end cost a user pays when they turn the flag on, so it is the
right thing to report for a deployment decision — but attributing all of it to "the quantizer"
would be wrong, and the documentation says so. (The weight probe has no such asymmetry: both
models run standard attention, so its bundle is the quantized weights plus the quantized-matmul
kernels they execute — which is, again, exactly what a user deploys.)

## 7. Comparing is a different problem from scoring

Scoring one configuration honestly does not yet answer the question users actually have: which
of several quantizations should I run? Sorting a comparison table by KL divergence answers only
the quality side and will often favor higher bit widths; sorting by size ignores quality. The
missing ingredient is cost normalization.

The `compare` command ranks configurations on a two-axis Pareto frontier: mean KL divergence
against memory cost
([`dominates`, ranking.py](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/src/mlx_quant_fidelity/ranking.py#L21)).
For KV configurations the cost axis is bytes per token,
`2 · n_layers · n_kv_heads · head_dim · (bits/8 + 4/group_size)`, where the `4/group_size` term
is the per-group fp16 scale and bias. That overhead term is not pedantry: group size is a real
quality lever, so two configurations at the same bit width sit at different costs, and a
ranking that ignored the overhead would misrank them. The committed
[Qwen2.5-0.5B KV comparison](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/_artifacts/samples/compare/kv-qwen2.5-0.5b.md)
shows the lever priced out: halving the group size from 64 to 32 at 4-bit roughly halves the
mean KL (**2.56 → 1.34** nats) for **11%** more cache bytes (4.5 → 5.0 bits per element by the
formula above). Against the table's 8-bit configuration (group 64), the halving narrows the
gap from **84×** to **44×**. All three configurations sit on the frontier (none is
strictly worse on both axes). In this one Qwen2.5-0.5B comparison, covering 4,088 positions
across eight chunks and three configurations, bit width moved mean KL much more than halving
the group size. Other models, corpora, and configuration grids may behave differently.

Two honesty rules keep the ranking from overclaiming. Domination is decided on mean KL only —
the tail is deliberately not a ranking axis, so a configuration with a good mean and an ugly
p99 keeps its frontier seat, and the table prints the p99 column for the reader to weigh. And
when a budget filter (`--min-tier`) would be satisfied only by a dominated configuration, the
tool returns *no* recommendation rather than a plausible-but-dominated one — the same
preference for "no answer" over "wrong answer" as the exact-zero guard in section 5.

## 8. Where the verdict lines come from

Each report carries a `good` / `marginal` / `bad` verdict, and a verdict needs thresholds. The
defensible part of the design is the AND rule: all three of mean KL, p99 KL, and flip rate must
clear a tier's ceilings
([`verdict_for`, policy.py](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/src/mlx_quant_fidelity/policy.py#L33);
values in the
[threshold policy](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/docs/threshold-policy.md)).
The committed reports show why one metric cannot stand in for the others. The
[Llama-3.2-1B 8-bit run](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/_artifacts/samples/llama-3.2-1b-4bit-kv8.md)
has a mean KL of **0.0004** nats — more than twenty times under the `good` ceiling — yet grades
`marginal` because **1.26%** of positions flip their top token. The Qwen2.5-7B 8-bit run passes
the mean ceiling and fails both the p99 and flip ceilings. Grading on mean KL alone would have
called both runs `good`.

The threshold *values* deserve a plainer statement than tools usually give: they are policy,
not measurement. The KV tiers were chosen by judgment at the tool's first release —
deliberately conservative — and have not changed since. The weight tiers were anchored to the
tool's own early sample runs (8-bit landing `good`, 4-bit `marginal` on short prose), which is
circular in the way any small-sample anchoring is. Human-readable weight reports and badges
label the verdict **provisional**; raw JSON exposes the verdict value without that annotation,
so machine consumers must consult the published policy.

No task-validated fidelity thresholds have been published for MLX, and the committed evidence here — a handful of (model, bit-width) points —
demonstrates tail-vs-mean and checkpoint-dependent failure but cannot defend "0.01 nats" as a
boundary. Establishing one would take a calibration study against downstream task accuracy
that has not been run. Until then the verdicts make reports *auditable* — the lines are
published, versioned, and applied uniformly — without pretending the lines are truths. The
badge output keeps the same discipline: it names the corpus, context length, and mode alongside
the verdict, so it cannot compress into an unqualified "fidelity: 0.98".

## 9. What the numbers still do not say

The guards above make the measurement honest about what it measures. They do not extend what it
measures, and four boundaries are worth stating as bluntly as the failure modes were.

Fidelity is corpus- and context-length-specific. These runs are WikiText-2 short prose, scored
teacher-forced in 512-token windows. *Accuracy Is Not All You Need*
([arXiv:2407.09141](https://arxiv.org/abs/2407.09141)) shows that aggregate benchmark accuracy
can hide answer flips and reports worse MT-Bench results for compressed models on a free-form
generative task. It does not evaluate long-context or code workloads. These 512-token windows
therefore provide no evidence about either domain, and they never reach the cache sizes where
quantization is most tempting.

Perplexity delta is reported for continuity with llama.cpp, not as independent confirmation.
It scores the realized corpus token while mean KL weighs the full vocabulary; they usually move
together and can diverge when the reference does not concentrate its mass on the observed
token. Two agreeing columns in the same report are two views of one run, not replication.

A fidelity score is not a quality score. Nothing here evaluates downstream task accuracy —
that is what benchmark harnesses like
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) exist for. A
`good` verdict means the quantized model's next-token behavior tracks its own reference on
this corpus; whether either model is any good at a task is out of scope by design.

Deployment-mode numbers are a per-chunk proxy, not a long-form average. The committed
[deployment sample](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/_artifacts/samples/kv-deployment.md)
(Llama-3.2-1B, 4-bit, boundary at 256) covers **1,020** positions against the stress reports'
**51,100** — direction-consistent with stress mode, but 50× smaller and not a matched
comparison. Deployment mode also adds an ingredient to the measured bundle that stress mode
lacks: the probe re-enters the model at the conversion boundary, so the split forward itself —
not only the attention-path swap — is part of what it measures. Real deployments convert at
mlx-lm's CLI default boundary of 5,000 tokens (the Python API defaults to 0) inside much longer
contexts than a 512-token window can represent.

## 10. Lessons

**Fix the text, compare the distributions.** Generation-based checks fail in both directions —
compounding drift over-reports, greedy-equality oracles under-report. Teacher-forced scoring on
a fixed corpus controls the input tokens and trajectory at every position.

**A harness that can pass by doing nothing eventually will.** The silent no-op paths in
upstream tooling are real, so the guard has to be structural: probe capability by calling it,
raise on exact zero, and keep tests that prove the metric rises when the input is damaged.

**The mean is where damage hides.** Full-vocabulary KL with no epsilon, reported with its tail.
A run can sit more than twenty times under the mean ceiling and still change one top token in
eighty.

**A seed cannot control a sampler the probe never calls.** Reproducibility comes from fixed corpus
tokens and recorded model and software revisions. Fp32 metric arithmetic and per-chunk
materialization serve numerical and memory roles, not a cross-device determinism guarantee.

**Name the reference and the bundle.** Drift against an 8-bit reference, on 4-bit-weight
checkpoints, through a different attention path — each is fine when stated and misleading when
implied away.

**Normalize by memory before comparing, and keep policy separate from measurement.** A
comparison without a cost axis is a tautology, and a verdict line is a published convention,
not a law of nature. The tool's job is to make both auditable.

## References and source notes

- `mlx-quant-fidelity` at the v0.4.0 tag (matches the installed version used for the committed
  reports):
  [`probes/kv.py` (`_score_chunk`, `_cache_is_quantizable`)](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/src/mlx_quant_fidelity/probes/kv.py),
  [`probes/_paired.py` (`_reduce_pair`, `_check_exact_zero`)](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/src/mlx_quant_fidelity/probes/_paired.py),
  [`metrics/kl.py`](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/src/mlx_quant_fidelity/metrics/kl.py),
  [`metrics/flip.py`](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/src/mlx_quant_fidelity/metrics/flip.py),
  [`policy.py`](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/src/mlx_quant_fidelity/policy.py),
  [`ranking.py`](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/src/mlx_quant_fidelity/ranking.py),
  [`tests/probes/test_kv_oracles.py`](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/tests/probes/test_kv_oracles.py).
- Committed evidence:
  [sample fidelity reports](https://github.com/IonDen/mlx-quant-fidelity/tree/v0.4.0/_artifacts/samples)
  (stress and deployment KV runs, weight runs, comparisons).
- Public methodology docs this memo deliberately does not restate:
  [measurement principles](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/measurement-principles.md),
  [ranking principles](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/ranking-principles.md),
  [threshold policy](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/threshold-policy.md).
- mlx-lm at the v0.31.3 tag:
  [`generate.py` (silent-skip cache conversion)](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/generate.py#L299),
  [`models/base.py` (two-matmul quantized attention)](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/base.py#L64).
- Prior art: [llama.cpp `llama-perplexity` (`--kl-divergence-base`)](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md);
  [*Accuracy Is Not All You Need*, arXiv:2407.09141](https://arxiv.org/abs/2407.09141);
  [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
- Companion memos:
  [Low-bit KV caches on MLX: what exists and what is missing](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/papers/low-bit-kv-caches-on-mlx-what-exists-and-what-is-missing.md);
  [Why the TeaCache Gate Did Not Engage on Short Distilled FLUX Schedules](https://github.com/IonDen/mlx-teacache/blob/main/docs/papers/why-teacache-does-not-engage-on-short-distilled-schedules.md).

---

*Prepared 2026-07-25. Last updated 2026-08-15. Denis Ineshin.*
