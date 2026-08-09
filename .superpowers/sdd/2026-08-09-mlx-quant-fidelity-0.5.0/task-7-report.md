# Task 7 report (code-only scope: brief Steps 1 and 4)

## Scope actually done

Per dispatch instructions, this covered **Steps 1 and 4 of the brief only** — writing the spike
script and the gated slow test. Steps 2/3/5 (actually running the spike and the slow test) are
explicitly deferred to the controller's main-thread session; nothing in this dispatch loaded a
model, hit the network, or ran the script/test.

## Files created

- `scripts/spike_long_window_memory.py` — subprocess-per-length orchestrator (512/1024/2048/4096
  on `mlx-community/Llama-3.2-1B-Instruct-4bit`, pinned to `MODEL_REVISION =
  "08231374eeacb049a0eade7922910865b8fce912"`), resumable via `_artifacts/spike_long_window/<length>.json`,
  active-memory watchdog + wall-clock backstop per the brief's embedded `WORKER` script.
- `tests/probes/test_kv_longwindow_slow.py` — `@pytest.mark.slow` + `@pytest.mark.network`,
  pins the `chunk_length=2048` lane: `report.corpus.chunk_length == 2048`, `kl_by_depth` has 8
  buckets, `peak_memory_bytes` under `mx.device_info()["max_recommended_working_set_size"]`. Same
  pinned `_REVISION`.

## Deviations from the brief's literal code

- **`_run_worker` extraction.** The brief's orchestrator has the `try/except
  subprocess.TimeoutExpired` block lexically inside the `for length in LENGTHS:` loop, which
  ruff's `PERF203` (try-except-in-loop) flags under this repo's strict `select` (this project's
  ruff config is stricter than the brief's originating sibling repos — no `scripts/` per-file-ignore
  exists here, unlike `mlx-taef`/`mlx-model-doctor`). Moved the subprocess call + exception
  handling into a `_run_worker(length) -> tuple[int, str, str]` helper, called from the loop.
  Behavior is unchanged; only the lexical nesting moved.
- **Removed the redundant `if exc.stderr else ""` guard** on the `TimeoutExpired.stderr` decode
  — `(raw_stderr or "")` already covers the `None` case, so the extra check was dead logic once
  written functionally. Replaced with an `isinstance(raw_stderr, bytes)` check so mypy-style
  correctness holds whether `subprocess` hands back bytes or `str` for `text=True` runs that hit
  the timeout path (undocumented CPython-version-dependent behavior).
- Added a longer module docstring (usage line, resumability note) to satisfy this repo's `D100`
  requirement (public-module docstring) — the brief's original one-liner is folded in as the
  first sentence.
- `MODEL_REVISION` / `_REVISION` set to the value supplied by the controller,
  `"08231374eeacb049a0eade7922910865b8fce912"`, replacing the brief's placeholder in both files.

No other logic changes — corpus/model args, `LENGTHS`, `WALL_TIMEOUT_S`, `OUT` path, and the
embedded `WORKER` payload script match the brief verbatim (the `WORKER` string is opaque to
ruff/mypy since it's a string literal executed via `python -c`, not parsed as this file's AST).

## Gate results (all run, none of them execute the spike or the slow test's body)

- `uv run ruff check scripts/spike_long_window_memory.py tests/probes/test_kv_longwindow_slow.py`
  → `All checks passed!`
- `uv run ruff format --check scripts/spike_long_window_memory.py tests/probes/test_kv_longwindow_slow.py`
  → `2 files already formatted`
- `uv run mypy` → `Success: no issues found in 26 source files` (config `files =
  ["src/mlx_quant_fidelity"]` — `scripts/` and `tests/` are out of mypy's scope in this repo, confirmed
  by inspecting `pyproject.toml`; kept the script fully annotated regardless).
- `uv run pytest -q` (default, no gate flags) → `314 passed, 9 skipped in 0.40s`. The new
  `test_kv_longwindow_slow.py::test_long_window_stress_run_fits_and_buckets` is among the skips:
  `SKIPPED [1] tests/probes/test_kv_longwindow_slow.py:20: requires --run-slow (real-model probe integration)`.
- `uv run pytest --cov=mlx_quant_fidelity --cov-report=term-missing --cov-fail-under=85 -q` →
  `Required test coverage of 85% reached. Total coverage: 99.54%`, `314 passed, 9 skipped` — the
  coverage gate is unaffected (new files are a script and a gated test; neither is instrumented
  source).

### Collection proof

```
$ uv run pytest --co -q tests/probes/test_kv_longwindow_slow.py
tests/probes/test_kv_longwindow_slow.py::test_long_window_stress_run_fits_and_buckets
1 test collected in 0.08s

$ uv run pytest --co -q tests/probes/test_kv_longwindow_slow.py --run-slow --run-network
tests/probes/test_kv_longwindow_slow.py::test_long_window_stress_run_fits_and_buckets
1 test collected in 0.08s
```

Both collect the same single test; neither invocation executed it (`--co` is collect-only).

## Commit

`test: pin long-window memory behavior; add the spike reproducer` — SHA filled in by the
`bash-git-ops` agent that performed the commit; see the parent session's final report for the
exact hash.

## Not done in this dispatch (by design)

- Did not run `scripts/spike_long_window_memory.py` (Step 2).
- Did not run the slow test with `--run-slow --run-network` (Step 5).
- Did not write `docs/superpowers/spikes/2026-08-09-mlx-quant-fidelity-long-window-memory.md`
  (Step 3) — that file records measured results, which don't exist yet.
- Did not touch `MAX_CHUNK_LENGTH` in `src/mlx_quant_fidelity/probes/kv.py` — the brief's decision
  gate ("update the constant in the same commit as the spike results if it must drop below 4096")
  depends on the actual measured peak-memory numbers from Step 2, which have not been produced.

## Concerns for the controller

- The MODEL_REVISION value was supplied by the controller and used verbatim; not independently
  re-verified against the Hub in this dispatch (no network access permitted here).
- `MAX_CHUNK_LENGTH = 4096` in `src/mlx_quant_fidelity/probes/kv.py` remains unchanged (still the
  provisional ceiling from Task 6) pending the real spike run.
