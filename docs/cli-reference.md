# Command-line options

This page lists the options the `kv`, `weights` and `compare` commands share, and how the tokenizer-code check works. The [README](../README.md) shows the commands in use.

## Common options

- `--kv-bits` / `--kv-group-size` — the KV configuration to score, default `4` / `64`. The `4:32,4:64` shorthand in `compare kv --configs` is `bits:group_size`.
- `--kv-method` — `stock` (default) reads `--kv-bits`/`--kv-group-size` directly. `turboquant`, `turboquant-vonly`, and `affine` also take a colon-delimited spec string in place of a bare name: `turboquant:bits[:seed]`, `turboquant-vonly:v_bits[:seed]` (K stays fp16; only V quantizes), and `affine:k_bits:v_bits[:group_size]` (an independent per-side K/V bit width no shipped cache runs — see [docs/measurement-principles.md](measurement-principles.md#measuring-a-third-party-cache)). `compare kv --configs` mixes any of these into one list, e.g. `8:64,turboquant:4,affine:8:4`. `--kv-seed` sets the TurboQuant rotation seed (default 42, must be ≥ 1) for the flag-based `--kv-method turboquant` and `--kv-method turboquant-vonly` forms.
- `--control` — stock only. Runs a third, quantizer-only forward (dequantize on fetch, standard SDPA, the same bits and group size) alongside the deployed path, so a stock report carries both numbers. `compare kv` always ranks every method on the quantizer-only number and runs this control lane for stock automatically — see [docs/ranking-principles.md](ranking-principles.md#ranking-footing-quantizer-only-every-method-always).
- `--max-chunks N` — score only the first N corpus chunks. The figures in the README use `--max-chunks 100`, except the 4096-token depth run (`--max-chunks 12`), described in [measurement-principles.md](measurement-principles.md#drift-by-position-depth); leave it off and the run covers the whole WikiText-2 test split.
- `--chunk-length N` — the scoring window, default 512, hard ceiling 4096.
- `--quantize-start N` — `0` for stress mode, the default; any N above 0 for deployment mode.
- `--format json|md|badge` — `md` by default. `json` is the machine-readable form the reports under [`_artifacts/samples/`](../_artifacts/samples) are written in, and `badge` works on `weights` as well as `kv`. The two `compare` subcommands take `json` and `md` only.
- `repo@revision` on `weights` and `compare weights` pins a Hugging Face revision inline; an inline pin wins over `--quant-revision` / `--reference-revision`. `kv` and `compare kv` keep `--model-revision`.
- `--allow-custom-code` — let a repo whose `config.json` names a `model_file` run that file. mlx-lm executes it on load, so the tool refuses such a repo unless you pass this flag. Only use it for repos you trust.
- `--worker-timeout SECONDS` — `compare weights` only: the wall-clock limit for each target's worker, default 21600 (six hours; `0` turns it off). A worker that runs over it becomes a failed row; raise the limit for full-corpus runs on large models.

## Tokenizer code check

`--allow-custom-code` also decides whether a tokenizer's own code may run. That refusal (exit code 2) comes before any weight download: the tool reads `config.json` and `tokenizer_config.json` first. This costs at most one extra small request, unless you pass the flag or point at a local directory. The check asks the installed transformers whether the tokenizer needs the repo's code, and it can answer on transformers 5.4.0 or later. On an older version the file is still downloaded, but the refusal comes at the model load, after the weights have downloaded and loaded and before any code from the repo has run. A tokenizer the check cannot classify is refused at load time too.
