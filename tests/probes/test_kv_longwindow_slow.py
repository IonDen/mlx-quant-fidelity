"""Pins the 2048-token long-window KV-probe lane: fits under the wired cap, buckets present.

``@pytest.mark.slow`` and ``@pytest.mark.network`` — downloads and loads a real model. Skipped
by default; run with ``--run-slow --run-network`` on the MAIN THREAD only.
DO NOT run in CI or in a subagent.

Same pinned revision as ``scripts/spike_long_window_memory.py``, the reproducer this test
pins one lane of.
"""

import pytest

from mlx_quant_fidelity.probes.kv import measure_kv_fidelity

_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
# Current `main` commit SHA, resolved 2026-08-09 (`hf repo info mlx-community/Llama-3.2-1B-Instruct-4bit`).
_REVISION = "08231374eeacb049a0eade7922910865b8fce912"


@pytest.mark.slow
@pytest.mark.network
def test_long_window_stress_run_fits_and_buckets() -> None:
    """chunk_length=2048 stays under the INSTALLED wired cap and reports depth buckets.

    The bound is ``compute_safe_caps_gb()[0]`` (the cap the probe actually installs, 20 GiB on
    a 32 GB M1 Max), not the device's ``max_recommended_working_set_size`` (~26.8 GiB) — a run
    peaking at 24 GiB would pass the latter while blowing the cap that is really in force.
    The recorded measured peak for this lane is 7.09 GiB, well under either; 13.53 GiB is
    the 4096-token lane, not this one.
    """
    report = measure_kv_fidelity(
        _MODEL,
        kv_bits=4,
        kv_group_size=64,
        chunk_length=2048,
        max_chunks=2,
        model_revision=_REVISION,
    )
    assert report.corpus.chunk_length == 2048
    assert report.kl_by_depth is not None
    assert len(report.kl_by_depth) == 8

    from mlx_quant_fidelity._memory_caps import compute_safe_caps_gb

    wired_gb, _ = compute_safe_caps_gb()
    assert wired_gb, (
        "device reports no working-set size — the wired cap under test is not installed"
    )
    assert report.peak_memory_bytes < wired_gb * 1024**3
