# Third-party KV caches

This page shows how to score the TurboQuant-MLX cache with `mlx-quant-fidelity`, with the measured comparison on Llama-3.2-1B and how to read it. The [README](../README.md) has the short version.

## Measuring a third-party KV cache (TurboQuant-MLX)

The KV probe is not tied to mlx-lm's cache. `--kv-method turboquant` measures the
[TurboQuant-MLX](https://github.com/arozanov/turboquant-mlx) uniform-bit cache on the same
paired, teacher-forced, full-vocabulary yardstick, and `compare kv` ranks it against the stock
configurations memory-normalized. Install the pinned port first — the PyPI package named
`turboquant-mlx` is unrelated. The tool checks the install record before importing the port and accepts
only an https git install of this repository (a different commit is measured with a note in the
report); a PyPI, editable, SSH or fork install is refused on purpose:

```bash
pip install "turboquant-mlx @ git+https://github.com/arozanov/turboquant-mlx@6e928d715595dee9f6b6cc3968baa44e1f408d28"
mlx-quant-fidelity compare kv mlx-community/Llama-3.2-1B-Instruct-4bit --configs 8:64,4:64,turboquant:4,turboquant:3,turboquant-vonly:4,turboquant-vonly:3,affine:8:4,affine:8:2
```

With uv, `uv sync --group turboquant` installs the same pin.

```
# Quant comparison (kv) vs `mlx-community/Llama-3.2-1B-Instruct-4bit`

| target | cost | KL mean | KL p99 | flip | bundled KL | resident +/token | verdict | frontier |
|---|---|---|---|---|---|---|---|---|
| `turboquant:3` | 8.2 KB | 0.4229 | 2.3559 | 0.3259 | — | 65.5 KB | bad | ✓ |
| `4:64` | 9.2 KB | 0.1485 | 0.9571 | 0.2056 | 0.1477 | 0 B | bad | ✗ dominated by `turboquant:4` |
| `turboquant:4` | 9.2 KB | 0.0825 | 0.5663 | 0.1582 | — | 65.5 KB | bad | ✓ |
| `affine:8:2` | 11.3 KB | 0.2180 | 1.3105 | 0.2391 | — | 32.8 KB | bad | ✗ dominated by `turboquant:4` |
| `affine:8:4` | 13.3 KB | 0.0120 | 0.0818 | 0.0628 | — | 32.8 KB | bad | ✓ |
| `8:64` | 17.4 KB | 0.0004 | 0.0028 | 0.0121 | 0.0004 | 0 B | marginal | ✓ |
| `turboquant-vonly:3` | 36.9 KB | 0.0297 | 0.2025 | 0.0974 | — | 32.8 KB | bad | ✗ dominated by `affine:8:4` |
| `turboquant-vonly:4` | 37.4 KB | 0.0078 | 0.0520 | 0.0508 | — | 32.8 KB | bad | ✗ dominated by `8:64` |
```

Read this table with three things in mind. The `KL mean` column is quantizer-only for every row:
`turboquant`, `turboquant-vonly`, and `affine` dequantize on fetch and ride standard SDPA by
construction, and `4:64`/`8:64` ran a `--control` forward alongside their deployed path so stock
ranks on the same footing. Stock's own deployed-path number — mlx-lm's two-quantized-matmul
attention — is the `bundled KL` column instead; on this sample it barely differs from the
quantizer-only column (0.1477 vs 0.1485 for `4:64`) — the two numbers land within about half a
percent of each other at this configuration and scale, not evidence that the kernel's own
contribution is small or zero in general (see
[docs/measurement-principles.md](measurement-principles.md#decomposing-bundled-and-quantizer-only-drift)).
And the `cost` column is stored bytes, not the memory a method's fetch path needs while it runs:
`turboquant` and its V-only variant keep full-precision dequantization buffers beyond what they
store, shown in `resident +/token` — `turboquant-vonly`'s pinned port even stores an unused fp16
copy of V, so its stored bytes exceed a plain fp16 cache's even though V itself compresses (see
[docs/ranking-principles.md](ranking-principles.md#stored-bytes-vs-resident-memory)). The
uniform-bit cache and its V-only variant are measured at the port's default seed; its asymmetric
and layer-adaptive configurations are not, and its `make_adaptive_cache` silently ignores the
documented `k_bits`/`v_bits` parameters at the pinned commit. `affine:k:v` measures an independent
per-side bit width that no shipped cache runs. Sample captured on Apple M1 Max, 32 GB, revision
`08231374…`, 100 chunks of 512 tokens, stress mode.
