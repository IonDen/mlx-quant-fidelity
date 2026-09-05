# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2026-09-05

The weights probe reports the quantization geometry it measured on the loaded model, and repos published as DWQ and AWQ quantizations are ranked on one yardstick.

### Added

- Four `WeightFidelityReport` fields, read off the loaded model rather than its `config.json`: `quant_geometry` (each distinct bits/group-size combination and how many modules use it), `quant_n_full_precision` (quantizable modules left at full precision), `quant_bits_per_weight` (effective precision, using the definition `mlx_lm.convert` prints), and `quant_precision` (`uniform` or `mixed`).
- A `bits/wt` column in `compare weights`, and each target's own report notes under a weight comparison, where previously only a KV comparison carried them.
- `repo@revision` on `weights` and `compare weights`, pinning a Hub revision inline; an inline pin wins over the flag. `kv` and `compare kv` keep `--model-revision`.
- `--quant-revision` and `--reference-revision` on `weights`, matching what `compare weights` already accepted.
- A committed `compare weights` sample ranking five Qwen3-0.6B repos against the bf16 reference: two 4-bit repos of the same size, one published as a DWQ quantization and one plain, a third published as AWQ, and the 6-bit and 8-bit rungs (`_artifacts/samples/compare/weight-qwen3-0.6b-ladder.{json,md}`).
- `pytest --hide-port`: the default suite can now be run with the TurboQuant port masked, matching CI.

### Changed

- A weight report's headline carries the measured precision instead of the declared nominal: `@ 4-bit (group 64, 4.50 bits/weight)` for a uniform model, and a `mixed 4/5-bit` form when more than one bit width is in use. A report written before 0.8.0 keeps its original headline byte-for-byte.
- The weight badge labels a report spanning more than one bit width by the widths it found (`mixed 4/5-bit` for a model holding both) rather than by the declared nominal.
- Every new weight report carries a standing note that the method producing the quantization is not recorded anywhere the tool can read.
- The committed Qwen2.5-0.5B `compare weights` sample was regenerated with pinned revisions.

### Notes

- A repo published as a DWQ quantization is indistinguishable from a plain quantized repo by anything the tool can read. mlx-lm's DWQ, AWQ and dynamic quantizers all write a `quantization` block recording geometry alone — bits, group sizes, per-module overrides — and never the recipe, so a repo id ending in `-DWQ` is a claim the tool does not check. The comparison rows therefore carry repo ids (`docs/measurement-principles.md`).

## [0.7.0] - 2026-08-27

`compare kv` now ranks every method on the same footing — quantizer error alone — instead of whichever number a method happened to report natively. Two more KV methods join the comparison.

### Added

- `--kv-method affine:k_bits:v_bits[:group_size]`: an independent per-side K/V bit width, measured quantizer-only. No shipped cache runs this layout; the only known implementation is an idle mlx-lm fork.
- `--kv-method turboquant-vonly:v_bits[:seed]`: TurboQuant-MLX's V-only cache. K stays fp16; only V quantizes.
- `kv --control` (stock only): runs a third, quantizer-only forward — dequantize on fetch, standard SDPA, the method's own bits and group size — alongside the deployed path, so a stock report carries both numbers.
- `--model-revision` on `kv` and `compare kv`, pinning the HuggingFace revision the way `weights`/`compare weights` already did; part of `compare kv`'s resume identity.
- Four `FidelityReport` fields: `drift_footing`, `control_kl`, `control_flip_rate`, `working_set_bytes_per_token`. `compare kv`'s Markdown table gains a `bundled KL` column (stock's native deployed-path number, shown only where the control lane ran) and a `resident +/token` column (a method's fetch-path working-set overhead beyond its stored bytes).
- `AffineKVMethod` and `TurboQuantVOnlyKVMethod`, exported from the package root alongside `StockKVMethod` and `TurboQuantKVMethod`.
- Per-lane long-window memory receipts for `turboquant`, `turboquant-vonly`, `affine`, and stock run with `--control`, re-validating `MAX_CHUNK_LENGTH = 4096` for each at this vocabulary (`docs/measurement-principles.md`).

### Changed

- **`compare kv` ranks on quantizer-only drift for every target.** 0.6.0 ranked stock's bundled number against TurboQuant's already-quantizer-only one — different quantities compared on one Pareto axis. `compare kv` now runs stock's control lane automatically and ranks on it; the table still shows stock's bundled number in the `bundled KL` column rather than hiding it. On the committed Llama-3.2-1B sample, re-footing moved stock `4:64`'s ranked number by about half a percent (0.1477 bundled to 0.1485 quantizer-only) and left the ranking outcome unchanged — `4:64` is still dominated by `turboquant:4`. That gap is not a kernel-purity score: `docs/measurement-principles.md` explains why.
- The KV comparison partial schema is bumped to 4 (the ranking footing is now part of a partial's identity); partials from 0.6.x recompute. The weight comparison partial schema is bumped to 2 (the identity is now revision-aware); partials from 0.5.x recompute once.
- `badge_for_report`'s KV bit-width label now goes through `method_bits_text` instead of interpolating `kv_bits` directly, fixing a `None`-bit badge title for an adapter method (`k8v4-bit` for `affine`, `v3-bit` for `turboquant-vonly`).
- A new committed sample (`_artifacts/samples/compare/kv-llama-3.2-1b-methods.{json,md}`) replaces 0.6.0's: regenerated on the quantizer-only footing, with `affine:8:4`, `affine:8:2`, `turboquant-vonly:3`, and `turboquant-vonly:4` added to the ranked set.
- The chunk-length pre-flight gate now factors both the scored method's own working-set bytes and a running `--control` lane's working-set bytes into its estimate, and raises the more specific `LogitsBudgetError` (a `CorpusError` subclass) rather than a generic one.

### Notes

- The pinned TurboQuant-MLX port's V-only cache stores an unused fp16 copy of V in its inner cache alongside K, so its stored bytes exceed a plain fp16 cache's even though V itself compresses — the method's value at this commit is V-compression quality, not memory (`docs/ranking-principles.md` covers the stored-vs-resident distinction this surfaces).
- The gap between a stock report's bundled and quantizer-only numbers is not a kernel-purity score: both are read off the model's final logits after many layers of compounding, not one attributable difference (`docs/measurement-principles.md`).

## [0.6.0] - 2026-08-24

The KV probe measures any per-layer cache implementation, and the first third-party cache — TurboQuant-MLX — is ranked against mlx-lm's stock cache on one memory-normalized yardstick.

### Added

- `--kv-method turboquant` on `kv`, and `turboquant:<bits>[:<seed>]` entries in `compare kv --configs`, for the TurboQuant-MLX uniform-bit cache (git-pinned manual install; see the README).
- A committed `compare kv` sample ranking stock 8-bit and 4-bit against TurboQuant 4-bit and 3-bit on Llama-3.2-1B.
- Four report fields: `kv_method`, `kv_method_params`, `kv_method_provenance`, and `measured_kv_bytes_per_token` (stored bytes after the first scored window; tested to equal the analytic cost for every shipped method).
- A Python `KVCacheMethod` protocol with `StockKVMethod` and `TurboQuantKVMethod`, exported from the package root.
- The release build now runs on every pull request, so a distribution that would fail at tag time fails on the PR.

### Changed

- `kv_bits` and `kv_group_size` in `FidelityReport` are optional (a method without a group size reports `null`), and the report carries two dictionary fields, so report objects are no longer hashable. Stock reports keep every value and their Markdown unchanged; older sample JSON files predate the four new fields and load with defaults.
- `compare kv` partials carry the method in their identity (schema 3); partials from 0.5.x are recomputed.

### Notes

- Through a teacher-forced pass the TurboQuant cache dequantizes on fetch and runs standard attention, so its drift is the quantizer alone while stock's also includes the quantized attention path; every TurboQuant report says so. Only the uniform-bit cache is measured, and its resident memory in this path is roughly 2.3× an fp16 cache (derived from its retained dequantization buffers).

## [0.5.1] - 2026-08-18

Documentation release: a rewritten README, diagrams for the measurement model, and a fidelity chart rendered from the committed sample reports.

### Added

- Diagrams for where precision is lost, how stress and deployment measurement differ, and what it means for one configuration to dominate another.
- A fidelity chart rendered from the committed sample reports by `scripts/plot_fidelity_cliff.py`, so every plotted number is reproducible from committed data.

### Changed

- Rewrote the README around the three questions the tool answers: what a quantization you are about to run costs, which configuration fits a memory budget, and what to publish alongside a quantized model.
- Refreshed package metadata and promoted the development status to Beta.

No change to measurement behavior. Reports produced by 0.5.1 are identical to 0.5.0.

## [0.5.0] - 2026-08-10

Adds depth-resolved KV drift over a configurable window, an auto-generated `compare kv` sweep, and device provenance in every report.

### Added

- Stress-mode reports over a fixed-window corpus now carry a "Drift by position depth" table: per-position KLD pooled across chunks into eight equal-width depth buckets. The table is stress-only and requires every scored chunk to be the same length; an unequal-length corpus gets a report warning instead of a misleading number. On the sampled models, the curve is close to flat out to a 4096-token window — the KV quantizer's cost isn't building up across the window at these lengths, though that is narrower than "KV quantization is depth-insensitive": other work has found drift widening at longer context and over long reasoning-style generation (see the citations in `docs/measurement-principles.md`). That document also covers the buckets themselves and the measured memory cost of longer windows.
- `--chunk-length` on `kv` and `compare kv` (default 512, hard ceiling 4096) widens the scoring window so the depth table has more positions to work with. Paired fp32 logits scale with window length times vocabulary size, so the CLI warns once the estimated footprint for a chosen window passes 4 GiB, calibrated against measured peaks on Llama-3.2-1B-4bit (2.27 GiB at 512 tokens, up to 13.53 GiB at the 4096 ceiling; reproducer: `scripts/spike_long_window_memory.py`). Those peaks come from a 128k-token vocabulary and a larger-vocabulary model needs proportionally more at the same window, so a run whose estimated footprint would claim more than a conservative fraction of the installed memory cap is refused up front rather than warned about — lower `--chunk-length` until it fits.
- `compare kv --sweep` auto-generates the (bits × group-size) grid from the model's `config.json` alone, no weight download required. `--max-kv-bytes-per-token N` filters the grid to a memory budget. Configurations skipped for either reason are listed in the report rather than silently dropped.
- Reports now record the measuring device (for example "Apple M1 Max, 32 GB") in both the Markdown footer and the JSON output.

### Fixed

- `(head_dim, bits)` combinations that crash upstream mlx-lm's `QuantizedKVCache` on the very first cache append (bits=6 at head_dim 128 or 96, a packed-width pre-allocation bug) are now rejected up front with a clear, package-rooted error instead of a raw upstream crash. `compare kv --sweep` routes these combinations to its skipped-configuration list rather than emitting a failed row.
- A cached comparison partial whose top-level JSON is not an object is isolated instead of crashing a resumed run, extending the same isolation the 0.4.0 malformed-report fix added for structurally invalid report bodies.

### Changed

- The KV comparison partial's on-disk schema now records the scoring window. Partials written by 0.4.0 or earlier don't carry it and are treated as absent: a `compare kv` run resumed against them recomputes those configurations rather than reusing a partial from a different window. Weight-comparison partials are unaffected.

## [0.4.0]

Adds deployment-mode KV fidelity and a shareable fidelity badge, and isolates a malformed cached comparison partial.

### Added

- Deployment mode for the KV probe: `mlx-quant-fidelity kv <model> --quantize-start N` (and `compare kv ... --quantize-start N`) keeps the first N positions of each window full-precision and quantizes the rest, matching how mlx-lm's `--quantized-kv-start` runs. Reported metrics cover only the post-boundary region, so the number is the per-quantized-token cost — close to stress mode, because a post-boundary token attends to a fully quantized cache. `docs/measurement-principles.md` explains what deployment mode does and does not measure.
- Fidelity badge: `--format badge` on `kv` and `weights` emits a shields.io model-card badge whose message carries the verdict, bit-width, corpus, context length, and mode, so it cannot read as a bare "fidelity: 0.98". `badge_color`, `badge_for_report`, and `render_badge_markdown` are exported.
- `docs/threshold-policy.md` documents what good, marginal, and bad mean, why the mean, the tail, and the flip rate must all pass, and how the KV and provisional weight tiers were set.

### Fixed

- A cached comparison partial whose report body is structurally malformed — a missing or wrong-typed field — is isolated as `CorruptPartial` and skipped instead of aborting a resumed run with an uncaught error. This extends the 0.3.1 fix, which covered only an invalid stored verdict.

### Changed

- `fidelity_report_from_dict` and `weight_report_from_dict` raise a package-rooted `ReportSchemaError` on a malformed report dict instead of a bare `KeyError` or `AssertionError`.

## [0.3.1]

Hardens `compare` error handling, isolates cached partials with an invalid stored verdict, and adds a methodology document.

### Fixed

- `compare` now raises `CompareConfigError` (a package-rooted error, still catchable as `ValueError`) for invalid configurations such as fewer than two targets or duplicate repo IDs, instead of a bare `ValueError`. The CLI no longer catches unexpected errors as configuration mistakes.
- A persisted comparison partial whose stored verdict is invalid is isolated as `CorruptPartial` and skipped rather than aborting the rest of a resumed run.
- `docs/ranking-principles.md` now describes perplexity delta correctly: it is a related but distinct signal from mean KL divergence, not an algebraically identical restatement — it scores the realized next token and can diverge from mean KL when the reference does not concentrate on the observed token.

### Added

- `docs/measurement-principles.md` explains what the probe actually measures: teacher-forced paired scoring, why the corpus and chunk length matter, and what the numbers do and do not say about real-world inference quality.

## [0.3.0]

Adds a `compare` command that ranks a set of quantizations on a memory-normalized Pareto frontier — quality per byte, not a raw-metric sort.

### Added

- `compare weights <repos...> --reference <ref>` — scores N weight-quant repos against a shared reference, ranks them by quality per byte, and flags any configuration that is both worse quality and more expensive than another on the list.
- `compare kv <model> --configs b:g,b:g,...` — ranks N KV-cache `(bits, group_size)` configurations on one model, loading the model once, on quality per KV-byte-per-token.
- Budget filter `--max-kld X` returns the cheapest configuration whose mean KLD stays under threshold; `--min-tier good|marginal|bad` returns the cheapest that meets the verdict tier.
- Failed or unrankable configurations are reported in isolation rather than aborting the run.
- `compare_weight_fidelity`, `compare_kv_fidelity`, and `ComparisonReport` Python API.
- `docs/ranking-principles.md` explains how each axis is computed, what Pareto domination means in practice, and where the ranking has limits.

### Fixed

- KV probe applies `max_chunks` to a caller-provided corpus, matching the weight probe.

## [0.2.0]

Adds weight-quantization fidelity and hardens the KV probe.

### Added

- Weight-quantization fidelity probe: teacher-forced paired scoring of a quantized model repo against a higher-precision reference repo on the same corpus tokens. Both runs use standard attention, so the drift is the deployed quantized model's weight-quant cost with no quantized-attention kernel folded in.
- `mlx-quant-fidelity weights <quant> --reference <reference>` CLI and `measure_weight_fidelity(...)` Python API, with a weight report that records both repos, the declared quant bits and group size, whether the reference is itself quantized, and on-disk model bytes.
- Compatibility gate that rejects an incomparable pair — different architecture or vocabulary, or a quant repo that declares no quantization — before loading either model.
- Memory pre-flight that refuses a model pair too large for the device's working set rather than risking a kernel panic, since both full models stay resident for the whole run.
- Provisional weight verdict tiers, calibrated against 8-bit and 4-bit reference quants on WikiText-2.
- Committed weight sample reports for Llama-3.2-1B and Llama-3.2-3B (4-bit and 8-bit vs bf16) and Qwen2.5-7B (4-bit vs 8-bit).

### Fixed

- KV probe validates the cache group size against the model's head dimension after load, raising a clear error instead of a raw MLX failure mid-run.

### Changed

- Shared paired-scoring helpers moved to an internal module so the KV and weight probes reuse the same streaming reduction and exact-zero guard.

## [0.1.0]

First release. Measures how much quality a KV-cache quantization costs on MLX.

### Added

- KV-cache fidelity probe: teacher-forced paired scoring of a quantized KV cache against full precision on the same model, on identical corpus tokens. Per-chunk reduction to scalars keeps a long corpus from holding full distributions in memory.
- Metrics in fp32: full-vocab KL divergence (mean, median, p99, max), top-token flip rate, and perplexity delta.
- `mlx-quant-fidelity kv <model>` CLI and `measure_kv_fidelity(...)` Python API, with JSON and Markdown reports that record the corpus provenance, model revision, and quantize-start mode.
- Stress mode (quantize from token 0) with an exact-zero guard that refuses to report a silent "perfect fidelity" when quantization never engaged.
- Cache-capability gate that flags models whose KV cache cannot be quantized instead of crashing mid-run.
- WikiText-2 test-split loader with a pinned, reproducible chunking contract.
- Device-derived wired-memory cap installed before any model load, on both the CLI and test paths.
- Committed sample reports for Llama-3.2-1B, Llama-3.2-3B, and Qwen2.5-7B at 4-bit and 8-bit KV.
