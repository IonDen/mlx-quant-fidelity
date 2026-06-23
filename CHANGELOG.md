# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
