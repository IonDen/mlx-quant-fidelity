"""Real-model run of the stock KV probe on an MLA model that caches expanded K/V (0054).

``@pytest.mark.slow``; run with ``--run-slow`` on the main thread (loads a 16B-parameter MoE).
``mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx`` is ``deepseek_v2``: mlx-lm 0.31.3 rebuilds keys
and values from the compressed latent first and stores the per-head tensors (K 192, V 128 wide)
in a plain ``KVCache``, so the stock quantized cache should run on it like any other model.

Offline only: ``HF_HUB_OFFLINE`` is on for the test, so the pinned snapshot and the pinned
WikiText-2 parquet must already be cached; a missing one fails instead of downloading.

The committed measurement (``_artifacts/samples/compare/kv-deepseek-v2-lite-methods.json``,
2026-10-24, M1 Max 32 GB, mlx 0.31.2, mlx-lm 0.31.3) is stock 4:64, stress mode, 100 chunks of
512 tokens: mean KL 0.2763, flip rate 0.224, peak 10.80 GB. This test scores 2 chunks, not the
committed 100, so it pins no KL value: it checks that the probe runs end to end and that the
quantized cache engaged.
"""

import math

import mlx.core as mx
import pytest

from mlx_quant_fidelity.probes.kv import measure_kv_fidelity

pytestmark = pytest.mark.slow

MLA_MODEL = "mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx"
MLA_REVISION = "a33a8ad64477b9ca7a65b6dc1d8c06c997f87216"


@pytest.fixture
def hub_offline(monkeypatch):
    """Hub offline mode for this test only, so an uncached snapshot fails instead of downloading.

    huggingface_hub 1.19 reads the env var once at import into `constants.HF_HUB_OFFLINE`, and
    `is_offline_mode()` returns that attribute at call time; both are set.
    """
    import huggingface_hub.constants as hub_constants

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)


def test_mla_expanded_kv_runs_stock_probe_end_to_end(hub_offline):
    """Reds if the cache-capability refusal fires on an MLA model whose cache holds expanded K/V
    (a `deepseek_v2` check keyed on the family, not on what the attention can run), if its layers
    are counted as skipped (report flagged partial), or if the quantized cache never reaches the
    forward or poisons it (mean KL exactly 0, or NaN/inf from the 192/128 K/V width split)."""
    mx.reset_peak_memory()
    report = measure_kv_fidelity(
        MLA_MODEL,
        model_revision=MLA_REVISION,
        kv_bits=4,
        kv_group_size=64,
        max_chunks=2,
    )
    assert report.cache_supported is True
    assert report.kv_partial is False
    assert math.isfinite(report.kl.mean)
    assert report.kl.mean > 0.0
    # A non-zero flip rate is expected (committed 0.224 at 100 chunks); only its range is checked.
    assert 0.0 <= report.flip_rate <= 1.0
    # committed peak 10.80 GB; reds if the run stops fitting the device's working set
    assert report.peak_memory_bytes < int(mx.device_info()["max_recommended_working_set_size"])
