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
    """chunk_length=2048 stays under the device wired cap and reports depth buckets."""
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

    import mlx.core as mx

    cap = int(mx.device_info()["max_recommended_working_set_size"])
    assert report.peak_memory_bytes < cap
