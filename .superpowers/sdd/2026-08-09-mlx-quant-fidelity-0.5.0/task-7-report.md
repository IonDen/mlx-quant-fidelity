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

## Follow-up 2: task-review findings (resume-skip bug + gitignore)

Task review returned one Important finding and one same-area Minor on
`scripts/spike_long_window_memory.py`. Both fixed in this follow-up.

**IMPORTANT — resume-skip didn't distinguish success from failure.** The original `main()` loop
did `if out.exists(): skip`, so a length whose artifact recorded a watchdog abort (`returncode
3`) or a wall-clock timeout (`returncode -1`) would be skipped on every subsequent run, printing
the same `"skip {length} (exists)"` a clean lane gets — silently treating a failed run as done
forever.

Fix: added `_recorded_returncode(path: Path) -> int | None`, which reads and parses the existing
artifact and returns its `returncode` field, or `None` if the file is missing, unreadable, not
valid JSON, not a dict, or has a non-int `returncode` (defensive guard: a corrupt/unreadable
artifact counts as failed, so the lane re-runs). `_lane_is_complete(path: Path) -> bool` wraps
that as `_recorded_returncode(path) == 0` — the pure predicate the review suggested extracting.
`main()` now skips a length only when `recorded_rc == 0` (printing `"skip {length} (exists,
rc=0)"`), and otherwise — if an artifact exists but isn't a recorded success — prints the
recorded rc before re-running and overwriting it (e.g. `"4096: existing artifact recorded rc=3
(not a success) -> re-running"`).

**MINOR — added `/_artifacts/spike_long_window/` to `.gitignore`**, mirroring the existing
`/_artifacts/compare/` entry, so the run artifacts can't be accidentally staged by `git add -A`.

**TDD sequence:** wrote `tests/test_spike_long_window_resume.py` first (six cases: missing
artifact, `rc=0` success, `rc=3` watchdog abort, `rc=-1` wall timeout, corrupt/non-JSON content,
JSON missing the `returncode` field — all asserting `_lane_is_complete`), confirmed RED
(`ImportError: cannot import name '_lane_is_complete'`), then implemented the two helpers and
wired them into `main()`, confirmed GREEN (`6 passed`). The script has no coverage requirement
(`[tool.coverage.run].source = ["mlx_quant_fidelity"]` doesn't include `scripts/`), so this stays
a small, proportionate offline unit test rather than a full harness.

**Verification (no heavy runs, no model load):**
- The new pytest cases exercise the predicate directly against synthetic tmp_path artifacts.
- A throwaway `python3 -c` check confirmed the same behavior end-to-end
  (`_lane_is_complete` on a synthetic `rc=3` artifact → `False`; on `rc=0` → `True`;
  `_recorded_returncode` on a missing path → `None`).
- Read (not modified) the four real files in `_artifacts/spike_long_window/` and confirmed all
  four (`512.json`, `1024.json`, `2048.json`, `4096.json`) still record `returncode: 0` —
  untouched, per the coordinator's instruction not to delete or regenerate them.
- `git check-ignore -v _artifacts/spike_long_window/512.json` confirms the new `.gitignore` line
  matches.

**Gate results (this follow-up):**
- `uv run ruff check scripts/spike_long_window_memory.py tests/test_spike_long_window_resume.py`
  → `All checks passed!`
- `uv run ruff format --check scripts/spike_long_window_memory.py
  tests/test_spike_long_window_resume.py` → `2 files already formatted` (one auto-reformat
  applied via `ruff format` before this, matching CI's formatter)
- `uv run mypy` → `Success: no issues found in 26 source files` (`scripts/` still out of mypy's
  configured scope; kept the new helpers fully annotated regardless)
- `uv run pytest -q` (offline/default lane) → `320 passed, 9 skipped in 0.41s` (314 → 320: the
  six new resume-predicate tests; same skip set as before)
- `uv run pytest --cov=mlx_quant_fidelity --cov-report=term-missing --cov-fail-under=85 -q` →
  `Required test coverage of 85% reached. Total coverage: 99.54%` (unchanged — new files aren't
  instrumented source)

**Commit:** plain message `Fix resume-skip to check returncode, not just file existence` — SHA
recorded by the `bash-git-ops` agent that performed the commit; see the parent session's final
report for the exact hash.

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

## Follow-up: measured-peak reconciliation (post-spike)

After the controller ran the spike on the main session (all four lanes clean; measured peaks:
512→2,435,507,673 B, 1024→4,149,720,041 B, 2048→7,607,472,105 B, 4096→14,522,976,233 B ≈ 13.5
GiB, comfortably under the wired cap), two follow-up edits landed in this dispatch:

1. **`MAX_CHUNK_LENGTH` comment** in `src/mlx_quant_fidelity/probes/kv.py` — rewritten from
   "provisional pending the spike" to record the 2026-08-09 validation and the measured ~13.5
   GiB peak at chunk_length=4096. The constant itself (`4096`) is unchanged.

2. **Memory-warning multiplier calibrated from 4x to 7x** in `score_kv_config` (same file). The
   measured slope between the 2048 and 4096 lanes is `(14522976233 - 7607472105) / 2048 ≈
   3.377e6 B/token ≈ 6.58` `[positions, vocab]` fp32-array-equivalents per window (vocab=128256
   for Llama-3.2-1B, one array = `vocab * 4` bytes = 513,024 B). The old 4x estimate
   under-predicted the measured total peak by ~1.7x. Rounded 6.58 up to 7 so the estimate keeps
   erring high (a safety warning must err high, not low); the 4 GiB threshold and the warning
   message text are otherwise unchanged. Comment rewritten to state the calibration source and
   note that the slope excludes the model-weight/KV-cache intercept, so small models fire the
   warning slightly early — the right direction for a safety warning.

**TDD sequence followed:** updated `tests/probes/test_kv_fakeforward.py::test_large_window_emits_memory_warning`
first to expect `expected_gib = 7 * (4096 - 1) * 151_936 * 4 / 1024**3` (≈16.2 GiB); ran it alone
and confirmed RED against the still-4x implementation (`assert False`); then changed the
multiplier in `probes/kv.py` from 4 to 7 in both the threshold check and the warning-message
f-string; reran the same test and confirmed GREEN.

**Gate results (this follow-up):**
- `uv run ruff check src/mlx_quant_fidelity/probes/kv.py tests/probes/test_kv_fakeforward.py` →
  `All checks passed!`
- `uv run ruff format --check src/mlx_quant_fidelity/probes/kv.py tests/probes/test_kv_fakeforward.py`
  → `2 files already formatted`
- `uv run mypy` → `Success: no issues found in 26 source files`
- `uv run pytest -q` (offline/default lane) → `314 passed, 9 skipped in 0.40s` (same skip set as
  before — no lane changed shape)
- `uv run pytest --cov=mlx_quant_fidelity --cov-report=term-missing --cov-fail-under=85 -q` →
  `Required test coverage of 85% reached. Total coverage: 99.54%`

**Commit:** `Calibrate the long-window memory estimate against measured peaks` — SHA recorded by
the `bash-git-ops` agent that performed the commit; see the parent session's final report for the
exact hash.
