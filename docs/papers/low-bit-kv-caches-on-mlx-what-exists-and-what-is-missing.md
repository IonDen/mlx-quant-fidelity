# Low-bit KV caches on MLX: what exists and what is missing

*A research memo from `mlx-quant-fidelity`: the shipped state of KV-cache quantization
on Apple Silicon, what it measurably costs, and where the real gaps are*

Low-bit KV caches already ship on Apple Silicon. [mlx-lm](https://github.com/ml-explore/mlx-lm)
provides a drop-in, calibration-free `QuantizedKVCache` behind three generation CLI flags. It
works on standard attention paths when the cache type supports conversion. New proposals should
improve this path rather than rebuild it.

The existing feature still leaves important gaps. It applies the same per-token, group-scalar
affine quantization to Keys and Values. KIVI and KVQuant found that Keys and Values need different
treatment, while this repository's committed reports show that mlx-lm's scheme costs little at
8-bit, degrades quality at 4-bit, and fails badly at 4-bit on at least one popular checkpoint.
Performance is not automatic either. An upstream codebook-prefill proof of concept reported
cosine similarity 1.0 against its dequantized reference but lost badly on wall clock. Its author
eventually recommended the scalar format that mlx-lm already uses.

This memo covers mlx-lm's current implementation and measured fidelity cost, then compares it
with KIVI, KVQuant, other inference stacks, and public Metal-kernel work.

The memo is documentary. At drafting time (2026-07-25), the upstream API and repository states
were checked against installed mlx 0.31.2, mlx-lm 0.31.3, and the public repositories. mlx-lm
source links are pinned to the v0.31.3 tag.

The fidelity numbers come from committed, public reports in this repository. Those reports
record the software versions. The author identifies the hardware as an M1 Max with 32 GB of
unified memory, but the report files do not capture it. Nothing was rerun for this write-up.
Portable memory and quality claims remain separate from hardware-bound throughput claims.

## 1. What mlx-lm ships today

The shipped cache
([`QuantizedKVCache`, mlx-lm v0.31.3](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/cache.py#L232))
quantizes both Keys and Values with `mx.quantize(x, group_size, bits)`, defaulting to 64-element
groups at 8 bits.
[`mx.quantize`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html)
groups along the last axis. For a KV tensor, that axis is one head's `head_dim`. Each token's
per-head Key and Value vectors are sliced into 64-element groups. Each group gets an affine
scale and bias from its own minimum and maximum, so `head_dim` must be divisible by the group
size. In the literature's terms, this is per-token group-scalar quantization. It applies the
same scheme to K and V after RoPE and requires no calibration.

A full `KVCache` converts through `to_quantized(group_size=64, bits=4)`. The method returns a new
`QuantizedKVCache` containing a quantized copy of everything stored so far. Its 4-bit default
differs from the constructor's 8-bit default. The generation CLI exposes the path as `--kv-bits`,
`--kv-group-size`, and `--quantized-kv-start`
([generate.py](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/generate.py)).

The quantize-start flag does not preserve a permanently full-precision prefix. In the CLI it
defaults to 5,000 (the Python `generate_step` API defaults to 0).
Generation uses a full-precision cache until the first conversion check after the cache reaches
or exceeds that threshold. Token-by-token decode can hit the boundary exactly; chunked prefill
can overshoot it. At that check the *entire stored cache* is quantized in one shot, including the
prefix; everything after is quantized as it is written. The stored prefix does not
remain full precision. Its computation history is still different from quantize-from-zero,
because those prefix activations were produced while attention used a full-precision cache.
Deferring conversion delays the memory win and preserves an exact pre-boundary history, but it
does not by itself establish how post-boundary drift compares with stress mode.

MLX itself has no built-in quantized attention. The `mx.fast`
namespace in mlx 0.31.2 contains no quantized scaled-dot-product-attention op (the state of the
upstream request is covered in section 5). Instead, mlx-lm composes quantized attention in
Python
([`quantized_scaled_dot_product_attention`, models/base.py](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/base.py#L64)):
one `mx.quantized_matmul` for the scores against the packed Keys, a softmax, and a second
`mx.quantized_matmul` against the packed Values. `mx.quantized_matmul` reads the packed layout
directly, so no full-precision copy of the cache is ever materialized. The dispatch is a
duck-type check: any cache carrying a `bits` attribute routes to the two-matmul path,
everything else to `mx.fast.scaled_dot_product_attention`. The composition materializes the
full score matrix rather than streaming it, and it raises on attention sinks.

Four coverage limits were verified at the pinned tag:

- Sliding-window caches cannot quantize. `RotatingKVCache.to_quantized` raises
  `NotImplementedError` ("RotatingKVCache Quantization NYI"), and the batch variant matches.
- Some caches skip quantization silently. The CLI's conversion helper only converts caches
  that define `to_quantized`. The batch, chunked, and composite cache classes do not define it.
  On those paths, `--kv-bits` is a no-op with no warning and no memory saving.
- MLA models crash. On DeepSeek-V3-architecture models, quantized caches break at the
  source level: the attention code calls `.swapaxes` on what `update_and_fetch` returns
  ([`models/deepseek_v3.py`](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/deepseek_v3.py#L147)),
  and a `QuantizedKVCache` returns packed tuples, producing
  `AttributeError: 'list' object has no attribute 'swapaxes'`
  ([mlx-lm #1082](https://github.com/ml-explore/mlx-lm/issues/1082), open at drafting time).
  The architecture family behind several flagship 2026 models has no working quantized-KV path
  on mlx-lm.
- Low bit-widths are mechanically reachable, with one exception. `mx.quantize` accepts 2, 3,
  4, 5, 6, and 8 bits on mlx 0.31.2. The CLI's `to_quantized` conversion path works at all six
  widths. A directly constructed `QuantizedKVCache`, however, hits a packing-arithmetic shape
  bug at 5 or 6 bits on mlx-lm 0.31.3. The defaults and documentation steer users to 8 and 4
  bits. The next two sections explain why lower widths are poor choices for this scheme.

Calibration-free 8- and 4-bit affine KV quantization ships for standard attention
paths on MLX. Fidelity and architecture coverage remain open. A new design needs to improve on
this implementation rather than recreate it.

## 2. What the shipped scheme costs

This repository's committed sample reports use paired teacher-forced scoring to evaluate the
shipped cache. Each pair uses the same model and the same wikitext-2-raw/test tokens. One run
uses a full-precision cache and the other uses the quantized cache. The reports compare them at
each position using full-vocabulary KL divergence, top-token flip rate, and perplexity delta.
The public [measurement principles](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/docs/measurement-principles.md)
describe the method.
The reports below are stress-mode: quantization from token 0, so every scored position sits
behind a quantized cache. A committed
[deployment-mode sample](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/_artifacts/samples/kv-deployment.md)
exercises delayed conversion and scores only the post-boundary region. It covers 1,020 positions
across four chunks, so it is not a matched comparison with the 100-chunk stress reports and does
not establish that the two modes have equal drift. Its footer calls the first 256 positions
"full-precision." That label describes their pre-boundary computation and their exclusion from
the metrics. It does not describe their storage type after the boundary converts their K/V.

Each report in the table covers 51,100 positions across 100 chunks of length 512. All use
4-bit-weight community checkpoints with mlx 0.31.2 and mlx-lm 0.31.3. Each model cell links to
its committed report.

| Model (4-bit weights) | KV bits | KL mean (nats) | flip rate | perplexity Δ |
|---|---|---|---|---|
| [Llama-3.2-1B-Instruct](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/_artifacts/samples/llama-3.2-1b-4bit-kv8.md) | 8 | 0.0004 | 1.3% | +0.015 |
| [Llama-3.2-1B-Instruct](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/_artifacts/samples/llama-3.2-1b-4bit-kv4.md) | 4 | 0.1477 | 20.5% | +4.12 |
| [Llama-3.2-3B-Instruct](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/_artifacts/samples/llama-3.2-3b-4bit-kv8.md) | 8 | 0.0002 | 0.65% | +0.005 |
| [Llama-3.2-3B-Instruct](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/_artifacts/samples/llama-3.2-3b-4bit-kv4.md) | 4 | 0.0514 | 11.3% | +0.91 |
| [Qwen2.5-7B-Instruct](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/_artifacts/samples/qwen2.5-7b-4bit-kv8.md) | 8 | 0.0094 | 3.2% | +0.064 |
| [Qwen2.5-7B-Instruct](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/_artifacts/samples/qwen2.5-7b-4bit-kv4.md) | 4 | **9.3584** | **99.2%** | **+82,665** |

### 8-bit is near-lossless in the mean

All three checkpoints sit at or under a 0.01-nat mean KL and low single-digit flip rates at 8
bits. The tail is not
uniformly benign: Qwen2.5-7B's worst position reaches **13.1 nats** even at 8-bit, and the
tool's own verdicts on two of the three 8-bit runs are *marginal* rather than *good* because
the flip rate clears its strictest threshold. Averages are where quantization damage hides,
which is why the reports carry p99 and max.

### 4-bit per-token quantization is lossy where it works

Both Llamas lose real probability mass at 4 bits: the 3B model flips its top token at 11.3% of
positions and gives up
0.91 perplexity points; the 1B model flips 20.5% and gives up 4.12. Whether that is acceptable
depends on the workload, which is the reason the tool reports numbers instead of a blanket
verdict (the thresholds behind its good/marginal/bad labels are documented in the
[threshold policy](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/docs/threshold-policy.md)).

### 4-bit failure depends on the checkpoint, not model size

`mlx-community/Qwen2.5-7B-Instruct-4bit` fails at kv4: a 9.4-nat mean KL, a 99.2% flip rate, and
perplexity rising from 10.8 to roughly 82,676 (a **+82,665** delta). That is not degradation;
the model's output distribution no longer resembles its full-cache self at almost any position.
A larger model did not mean more robustness. The 7B Qwen collapses where the 1B and 3B Llamas
merely degrade. A committed comparison on Qwen2.5-0.5B, also a 4-bit-weight checkpoint, shows
the same family pattern at kv4
([report](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/_artifacts/samples/compare/kv-qwen2.5-0.5b.md)).
That comparison also prices the group-size lever: halving the group size from 64 to 32 at 4
bits roughly halves the KL (2.56 to 1.34 nats) for 11% more cache bytes (4.5 to 5.0 bits per
element), and still lands about 44× worse than 8-bit. Group size is a real but weak knob; it
does not buy back the failure.

These results have three limits.

First, fidelity depends on the corpus and context length. Short-prose measurements may
understate the risk for long contexts or code, so these figures are not ceilings.

Second, the measured drift includes both quantizer error and the numerics of the attention path.
The quantized run uses the two-matmul composition from section 1, while the reference uses fused
SDPA. That combination is the end-to-end cost paid by the user, so the method does not attribute
all drift to the quantizer.

Third, every scored model is a 4-bit-weight community checkpoint, and there is no fp16-weight
control run. A weight-quantization interaction cannot be excluded. The collapse is established
for this checkpoint under this cache, not for the architecture as a whole.

## 3. What KIVI and KVQuant found, and why it does not drop in

KIVI and KVQuant independently reached the same empirical result. KIVI
([arXiv:2402.02750](https://arxiv.org/abs/2402.02750), ICML 2024) found a small number of fixed
channels with large-magnitude outliers in Key caches. Value caches did not show the same pattern.
KIVI therefore quantizes Keys per channel and Values per token. Its tuning-free scales come from
the live cache's minimum and maximum, with no calibration set.

KVQuant ([arXiv:2401.18079](https://arxiv.org/abs/2401.18079), NeurIPS 2024) also starts with
per-channel Key quantization. It quantizes Keys *before* rotary position embedding because RoPE
spreads the outlier channels across dimensions. Its full method also uses sensitivity-weighted
non-uniform datatypes and dense-and-sparse outlier isolation. Both require offline calibration
data.

KVQuant reports **under 0.1 perplexity degradation at 3-bit** on Wikitext-2 and C4 across the
Llama and Mistral families. KIVI runs **2-bit** caches without tuning. On its serving setup, KIVI
reports 2.6× less peak memory *including model weights*. That reduction enabled batches up to 4×
larger and **2.35 to 3.47× throughput** on CUDA workloads.

The peak-memory and throughput multipliers belong to that serving stack, including its batch
sizes, weights, and fused CUDA kernels. They do not port to MLX. The quality findings and the
bits-per-element arithmetic in section 4 do.

The Qwen2.5-7B collapse in section 2 is consistent with this picture. Per-token group-scalar
quantization must stretch each group's 16 levels across any Key outlier that lands in it. The
committed measurement establishes *that* the shipped scheme collapses on this checkpoint at 4
bits. It does not establish that Key outlier channels caused the collapse. That attribution
would require a discriminating experiment, such as quantizing K and V at different widths.
This repository has not run one.

One community project has tested a related split on the same checkpoint.
[turboquant-mlx](https://github.com/arozanov/turboquant-mlx)'s README reports that K
quantization "destroys greedy decode" at 4 bits and below, including mlx-lm's native
`kv_bits=4`, while V survives 3-bit. It also reports that a mixed K-at-8/V-at-4 cache decodes
identically to baseline on `mlx-community/Qwen2.5-7B-Instruct-4bit`. This is a
repository-reported result based on greedy-text equality rather than a distribution metric. A
K-vs-V bit-width split also
does not isolate per-channel outliers. Still, the result points in the same direction as the
literature: on this checkpoint, the Keys carry the damage.

The asymmetric scheme does not drop into mlx-lm because of the decode path. MLX can already
express per-channel grouping by storing K transposed so that the token axis is the last grouped
axis, and `mx.quantized_matmul` consumes that layout directly. What breaks is decode. The quantization
group now runs along the axis that grows one token at a time. A group cannot be finalized until
`group_size` tokens have arrived. Before then, each append would rewrite a partially filled
packed word. KIVI's answer is a residual buffer: recent tokens stay full-precision, Keys
quantize in closed blocks once the buffer fills, and attention runs over a quantized bulk plus a
full-precision tail.

That residual machinery is the main engineering work in a KIVI-style cache. It also requires
layout bookkeeping that mlx-lm's attention code does not perform. HF Transformers implements
this design on the PyTorch side (section 4). **No per-channel-K path has been
found in the surveyed public MLX implementations as of 2026-07-25, and none has been implemented
or validated here**. Everything in this section is literature plus layout analysis, and any port
would need its own fidelity measurements before claiming the papers' numbers.

![Conceptual comparison of low-bit KV-cache decode layouts: mlx-lm groups each token along the head dimension, so it can quantize and append the complete Key and Value groups once; a KIVI-style asymmetric layout groups Keys across tokens, so recent Keys remain in a full-precision residual buffer until a complete block can move into the quantized bulk, while Values still follow a per-token path](https://raw.githubusercontent.com/IonDen/mlx-quant-fidelity/main/docs/papers/diagrams/per-token-vs-per-channel-kv-cache.svg)

*Figure 1. Conceptual decode and storage flow for the shipped per-token layout and a KIVI-style
asymmetric layout. The figure explains why the latter needs a residual buffer. It does not report
an implemented MLX path, a speedup, or a fidelity result. The editable
[PlantUML source](https://github.com/IonDen/mlx-quant-fidelity/blob/main/docs/papers/diagrams/per-token-vs-per-channel-kv-cache.puml)
is published with the SVG.*

## 4. What other stacks ship

### Hugging Face Transformers

Hugging Face Transformers ships a
[`QuantizedCache`](https://huggingface.co/docs/transformers/main/en/kv_cache) with two backends:
quanto at int2/int4, which is the default, and HQQ at int2/int4/int8. Its controls follow KIVI's
shape. A quantization-axis parameter selects the layout, with axis 0 recommended for quanto and
axis 1 for HQQ. A residual-length parameter controls how many recent tokens remain full precision.

The documentation is direct about the trade: "Quantizing the cache can harm latency if the context length is
short and there is enough GPU memory available for generation without enabling cache
quantization." The same trade needs measurement on MLX. One packaged MLX project, mlx-qsdpa,
reports that its M2 Ultra implementation is faster when it dequantizes and uses fp16 SDPA below
16K, then switches to its fused kernel at longer contexts (section 5). That crossover belongs to
that device, shape, format, query length, and kernel. Cache-byte savings are portable; the
direction of the latency change is not.

### vLLM

vLLM quantizes KV to
[FP8](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) (e4m3/e5m2),
calibration-free by default with all scales set to 1.0, and with FlashAttention 3 it computes
attention in the FP8 domain directly. That throughput path depends on the supported CUDA or ROCm
kernels. MLX 0.31.2 exposes FP8 and MX block formats as *storage* (`mx.quantize` modes `mxfp8`,
`mxfp4`, `nvfp4`, with matching `mx.quantized_matmul` support), but ships no built-in FP8-domain
attention path. Low-bit KV reduces memory and can extend context when cache capacity is the
binding constraint, but it does not inherit CUDA serving throughput.

Bit-width sets the cache's byte budget roughly linearly (a 64-element group's fp16 scale and bias
add half a bit per element), so 4-bit is roughly a 3.6× cache reduction against fp16 and 3-bit
roughly 4.6×. Those are KV-only capacity factors; usable context also depends on model weights,
temporary allocations, and other runtime headroom.

## 5. Metal kernels

At long context, decode attention spends much of its time reading the cache. CUDA implementations
fuse dequantization into attention: they read packed bytes, dequantize in registers,
and avoid a full-precision K/V buffer. MLX's shipped two-matmul path already reads the packed
layout through `mx.quantized_matmul`, so its memory saving does not require a custom kernel. By
contrast, dequantizing the whole cache before fp16 SDPA creates a full-precision copy. The public
record below reports a 31 GB activation spike at 128K context on a 31B model.

### Why the codebook proof of concept lost

The public record is [mlx #3404](https://github.com/ml-explore/mlx/issues/3404). Its author built a
fused dequant-in-attention proof of concept through MLX's Python `mx.fast.metal_kernel` API for
*TurboQuant codebook* quantization (a Hadamard rotation plus codebook lookup, not the affine
scalar format used above).

The proof of concept reported cosine similarity 1.0 against its dequantized reference, but lost
on wall clock to plain dequantize-plus-native-SDPA. It was **17.9×** slower at 2K tokens and
**3.4×** slower at 32K and 64K. This was a single-layer *prefill* microbenchmark with 128 queries,
GQA 4:1, on an M5 Max with 128 GB. The issue body first attributed the long-context gap to
Python-to-Metal dispatch overhead.

One day later, the author reported results from the same kernels built as a native C++ extension.
The diagnosis changed: codebook lookup was 3 to 5× slower than scalar dequantization on the GPU.
Per-element indexed loads defeated SIMD coalescing even with a 16-entry codebook in L1. The
author noted that llama.cpp and vLLM avoid this cost by using scalar formats.

The revised recommendation kept the rotation as an option but stored values in `mx.quantize`'s
scalar format. It also used the existing `mx.quantized_matmul` infrastructure, whose read path
the author described as already running at native speed. The format remained the bottleneck
after the dispatch layer changed, and the recommended layout is the one mlx-lm ships.

### Why generic quantized SDPA is still unmerged

A maintainer declined a TurboQuant-specific built-in in favor of generic quantized SDPA and
pointed to [mlx PR #3026](https://github.com/ml-explore/mlx/pull/3026). That PR has been open since January
2026. Its current branch supports affine 4/6/8-bit at group size 32 plus MX formats. A later
community benchmark on an M3 Pro measured single-token, Llama-shaped affine-4-bit decode at 1.8×
the fp16 baseline at 32K, but the same report measured only 0.6× for a Gemma-shaped case at 32K.
The result is shape-specific, not a general speedup. In April a maintainer cited limited review
capacity; as of 2026-07-25 the PR is still open and conflicts with `main`, so review and
integration both remain visible obstacles. Anyone planning kernel work here should read that PR
first. It is the most decision-relevant object in this section.

### What the community kernels report

Two community projects also publish kernels and measurements. Their claims are repository-reported
and were not independently verified for this memo.
[mlx-qsdpa](https://github.com/Thump604/mlx-qsdpa) is a packaged fused decode kernel for a single
query position. It reads mlx-lm's 4/8-bit affine packed layout and dequantizes inline in one
dispatch. It uses a single pass up to 4,096 tokens and two-pass split-K beyond that point. The
project reports 1.7× over mlx-lm's two-call path at 128K context with grouped-query attention.
Its dispatcher routes contexts under 16K to plain dequantize-plus-fp16-SDPA, which it reports as
the faster option.

mlx-qsdpa uses the same Python `mx.fast.metal_kernel` API as the #3404 proof of concept. If its
numbers hold, the earlier result does not show that the Python API itself cannot win. Read with
the revised #3404 diagnosis, the deciding variables appear to be the quantization format, the
regime (prefill versus decode), and the baseline, rather than the dispatch layer alone.

[turboquant-mlx](https://github.com/arozanov/turboquant-mlx)
pursues the codebook route (Walsh-Hadamard rotation plus Lloyd-Max codebooks) with fused
kernels. Its README's results table on Qwen2.5-7B at 32K reports a recommended mixed
K-at-8/V-at-4 configuration at **18% lower** active memory and **25.84 versus 35.75 tokens/s**,
about 72% of fp16 decode speed. Greedy output was identical to baseline. The repository blurb's
"4.6× compression at 98% FP16 speed" describes a different configuration and does not appear
in the README's results table. The table is the source used here.

Neither community project publishes a distribution-level quality measurement. turboquant-mlx
verifies that greedy decode reproduces the baseline's text; that is a meaningful
check and a weak oracle, binary over one decoding mode and silent about how much probability
mass moved underneath the argmax. This repository measures that hidden difference with KL, flip
rate, and perplexity across more than fifty thousand scored positions. The same method can be
used for any cache implementation that plugs into mlx-lm, including custom kernels.

## 6. What remains missing

1. A quality-preserving low-bit scheme on MLX. The literature's per-channel-K core
   (tuning-free in KIVI's form, with the residual buffer as the decode mechanism) reports
   near-lossless 3-bit and usable 2-bit on CUDA-side evaluations. This survey found no public
   per-channel-K implementation or MLX measurement as of 2026-07-25. Nearby work includes
   turboquant-mlx's mixed K8/V4 bit-width split, reported with a greedy-text oracle. MLX also
   exposes MX block formats, but mlx-lm's cache does not wire them up because `QuantizedKVCache`
   hardcodes affine mode. These formats are another candidate to evaluate. The shipped per-token
   scheme is measured here to degrade at 4-bit and to collapse outright on one popular
   checkpoint. Lower widths with the same layout are very unlikely to improve on it, though
   nothing here measures them.
2. MLA coverage. DeepSeek-V3-family models crash with quantized caches on mlx-lm
   ([#1082](https://github.com/ml-explore/mlx-lm/issues/1082) plus the source-level `.swapaxes`
   call in section 1, both current at drafting time). The family has no working
   `QuantizedKVCache` path in mlx-lm v0.31.3.
3. Coverage edges in upstream mlx-lm. Rotating (sliding-window) caches raise on
   `to_quantized`; batch, chunked, and composite caches skip quantization silently; the
   quantized attention path rejects attention sinks. (mlx-qsdpa ships repo-reported quantized
   rotating-cache classes, so the sliding-window gap is upstream-specific.)
4. Fidelity numbers for the community kernels. Both fused-kernel projects report speed and
   text-equality; neither reports KL, flip-rate, or perplexity-delta measurements. Until
   someone runs them, their speed columns have no measured quality column next to them.
5. A merged fused path. The generic quantized SDPA exists as an open, unmerged PR
   ([#3026](https://github.com/ml-explore/mlx/pull/3026)) with shape-dependent long-context
   results, an earlier review-capacity concern, and a branch that currently conflicts with
   `main`; one community decode kernel reports 1.7× at 128K through the Python kernel API.
   Whether and when a fused path lands in MLX core, and how it performs across regimes, remains
   publicly unsettled.

This memo implements nothing. It reports no per-channel-K result on MLX and found none in the
surveyed public projects as of 2026-07-25. CUDA throughput and system peak-memory multipliers do
not port to Metal. Key outliers remain a hypothesis for the Qwen2.5-7B collapse, consistent with
the literature and one repository-reported K-vs-V split but not demonstrated here.

## 7. Claims and status

| Claim | Status | Source |
|---|---|---|
| mlx-lm ships `QuantizedKVCache` (per-token group-scalar, K and V symmetric, defaults 64/8, CLI-integrated; quantize-start converts the whole stored prefix at the boundary) | verified against installed mlx-lm 0.31.3 and the pinned public tag | [cache.py](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/cache.py#L232), [generate.py](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/generate.py) |
| mlx core has no built-in quantized SDPA; mlx-lm composes it from two `mx.quantized_matmul` calls reading the packed layout | verified against installed mlx 0.31.2 (`mx.fast` contents) and the pinned tag | [base.py](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/base.py#L64) |
| Rotating caches raise on `to_quantized`; batch/chunked/composite caches skip silently; MLA models crash; quantized path rejects sinks | verified at the pinned tag; issue open at drafting time | [RotatingKVCache](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/cache.py#L551), [BatchRotatingKVCache](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/cache.py#L1327), [generate.py](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/generate.py#L299), [deepseek_v3.py L147](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/deepseek_v3.py#L147), [mlx-lm #1082](https://github.com/ml-explore/mlx-lm/issues/1082) |
| kv8 near-lossless in the mean on three checkpoints; kv4 lossy on Llama-3.2 1B/3B; kv4 collapse on the Qwen2.5-7B checkpoint (KL 9.36, flip 99.2%, ppl → ~82,676) | committed measurements, stress mode, single corpus and context length, 4-bit-weight checkpoints without an fp16-weight control | [sample reports](https://github.com/IonDen/mlx-quant-fidelity/tree/v0.4.0/_artifacts/samples) |
| Group size 64→32 at kv4 halves KL for 11% more bytes, stays ~44× worse than kv8 (Qwen2.5-0.5B) | committed comparison report | [compare report](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/_artifacts/samples/compare/kv-qwen2.5-0.5b.md) |
| KIVI: per-channel-K / per-token-V, tuning-free 2-bit; KVQuant: per-channel + pre-RoPE Keys, <0.1 ppl at 3-bit; KIVI's 2.6× peak memory includes model weights (system measurement) | verified from the papers' abstracts at drafting time | [arXiv:2402.02750](https://arxiv.org/abs/2402.02750), [arXiv:2401.18079](https://arxiv.org/abs/2401.18079) |
| Fused codebook PoC: reported cosine similarity 1.0 against its dequantized reference, 17.9× (2K) to 3.4× (32K to 64K) slower in a single-layer prefill microbench; author's revised diagnosis attributes the bottleneck to codebook lookup (3 to 5× vs scalar dequant, C++ included) and recommends the scalar `mx.quantize` format | reported in the public issue record by its author (initial attribution superseded by his follow-up); not independently reproduced here | [mlx #3404](https://github.com/ml-explore/mlx/issues/3404) |
| TurboQuant-specific built-in declined in favor of generic quantized SDPA first; [PR #3026](https://github.com/ml-explore/mlx/pull/3026) is open with affine 4/6/8 at group size 32 plus MX formats, shape-dependent community benchmarks, and a current conflict with `main` | verified from the public issue/PR records on 2026-07-25 | [mlx #3404](https://github.com/ml-explore/mlx/issues/3404), [PR #3026](https://github.com/ml-explore/mlx/pull/3026) |
| turboquant-mlx README table: K8+V4 at −18% memory, ~72% fp16 decode speed, identical greedy text (Qwen2.5-7B, 32K); mlx-qsdpa: 1.7× at 128K decode via `mx.fast.metal_kernel`, <16K routed to dequant+SDPA | repo-reported; no independent verification and no published distribution-level quality numbers | [turboquant-mlx](https://github.com/arozanov/turboquant-mlx), [mlx-qsdpa](https://github.com/Thump604/mlx-qsdpa) |
| Whether Key outlier channels caused the Qwen2.5-7B kv4 collapse | hypothesis, consistent with the literature and one repo-reported K-vs-V split on the same checkpoint; no discriminating experiment run here | sections 2 and 3 |

## 8. Practical lessons

Checking mlx-lm first changes the work. A plan written without reading its source could rebuild
`QuantizedKVCache` and mistake it for a new feature.

The incumbent also needs measurement before replacement. "4-bit KV quantization degrades
quality" is too vague to guide a design. The committed reports are more useful: 8-bit costs
little in the mean but receives a *marginal* verdict on two of three checkpoints. At 4-bit, two
checkpoints degrade and one fails. A demo that checks only greedy output could miss that
difference.

KIVI's quality result can motivate work on MLX. Its peak-memory and throughput numbers cannot be
copied across platforms because they include its serving stack. Bits per element is the portable
part; system throughput and peak memory need new measurements.

The kernel record also warns against blaming the launch mechanism too early. The initial report
blamed Python dispatch. The same author's C++ port found a larger cost in the codebook's
per-element indexed loads and recommended the scalar format already supported by MLX. Baseline,
format, and dispatch need separate measurements.

In short, low-bit cache work needs both speed and distribution-level quality evidence. The
community kernels currently publish compression and throughput without the latter. This
repository already measures KL divergence, flip rate, and perplexity delta for implementations
that can plug into mlx-lm.

## References and source notes

- mlx-lm at the v0.31.3 tag (matches the installed version used for verification):
  [`models/cache.py` (`QuantizedKVCache`, `to_quantized`, rotating-cache NYI)](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/cache.py#L232),
  [`models/base.py` (two-matmul quantized attention, sinks rejection, dispatch)](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/base.py#L64),
  [`generate.py` (`--kv-bits`, `--kv-group-size`, `--quantized-kv-start`)](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/generate.py),
  [`models/deepseek_v3.py` (the MLA `.swapaxes` call site)](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/deepseek_v3.py#L147).
- MLX: [`mx.quantize` documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html);
  [issue #3404 (fused-PoC timings, the author's revised codebook diagnosis, the maintainer's direction)](https://github.com/ml-explore/mlx/issues/3404);
  [PR #3026, "Quantized SDPA" (open)](https://github.com/ml-explore/mlx/pull/3026).
- mlx-lm issue tracker: [#1082 (MLA / DeepSeek-V3 crash with quantized caches, open)](https://github.com/ml-explore/mlx-lm/issues/1082).
- `mlx-quant-fidelity` committed evidence:
  [sample fidelity reports](https://github.com/IonDen/mlx-quant-fidelity/tree/v0.4.0/_artifacts/samples),
  [measurement principles](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/docs/measurement-principles.md),
  [threshold policy](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/docs/threshold-policy.md),
  [README](https://github.com/IonDen/mlx-quant-fidelity/blob/v0.4.0/README.md).
- Papers: [KIVI, arXiv:2402.02750](https://arxiv.org/abs/2402.02750) (ICML 2024);
  [KVQuant, arXiv:2401.18079](https://arxiv.org/abs/2401.18079) (NeurIPS 2024).
- Other stacks: [HF Transformers cache strategies](https://huggingface.co/docs/transformers/main/en/kv_cache)
  (`QuantizedCache`, axis and residual-length knobs, short-context latency warning);
  [vLLM quantized KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
  (FP8, calibration-free default, FA3 FP8-domain attention).
- Community Metal kernels (claims repo-reported):
  [turboquant-mlx](https://github.com/arozanov/turboquant-mlx),
  [mlx-qsdpa](https://github.com/Thump604/mlx-qsdpa).

---

*Prepared 2026-07-25. Last updated 2026-07-25. Denis Ineshin.*
