# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
