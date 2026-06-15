"""Real-model oracle tests for the weight-quant probe (--run-slow + --run-network, MAIN thread only).

Requires a real model load (mlx-community Llama-3.2-1B). Skipped by default. The corpus is
built in-memory (no WikiText download), but the MODEL must be downloaded from the Hub on a cold
machine, so a cold run requires both --run-slow and --run-network. A warm/cached run where the
model snapshot is already on disk effectively only needs --run-slow. DO NOT run in CI or in a
subagent.
"""

import mlx.core as mx
import pytest

from mlx_quant_fidelity.corpora.provenance import Corpus, CorpusProvenance
from mlx_quant_fidelity.probes.weights import measure_weight_fidelity

REF = "mlx-community/Llama-3.2-1B-Instruct-bf16"
QUANT = "mlx-community/Llama-3.2-1B-Instruct-4bit"


def _tiny_corpus(tok, *, chunk_length: int = 64, n_chunks: int = 2) -> Corpus:
    """In-memory corpus so the oracle stays slow-only (no --run-network needed)."""
    ids = tok.encode("Teacher forcing keeps both runs on identical tokens. " * 40)
    chunks = tuple(
        mx.array(ids[i * chunk_length : (i + 1) * chunk_length]) for i in range(n_chunks)
    )
    prov = CorpusProvenance(
        "in-memory",
        "test",
        REF,
        chunk_length,
        chunk_length,
        "none",
        "drop",
        "raw",
        chunk_length * n_chunks,
    )
    return Corpus(chunks=chunks, provenance=prov)


@pytest.mark.slow
@pytest.mark.network
def test_real_pair_produces_plausible_drift_and_pins_shape():
    """4-bit vs bf16 on a real 1B model: plausible drift, the [0] batch-of-1 shape contract,
    correct bits, and peak memory under the device working-set cap.

    The corpus is built in-memory (no WikiText download needed), but loading the model repos
    from the Hub on a cold machine requires network access. A warm/cached run (snapshots already
    on disk) effectively only needs --run-slow.
    """
    from mlx_lm import load  # type: ignore[import]

    ref_model, tok = load(REF)  # type: ignore[misc]
    ids = mx.array(tok.encode("Teacher forcing keeps both runs on identical tokens." * 5)[:128])
    logits = ref_model(ids[None, :-1])
    # pin the batch-of-1 [0]-strip contract that CI cannot cover (model returns [1, L-1, vocab])
    assert logits.shape == (1, ids.shape[0] - 1, logits.shape[-1])
    del ref_model
    mx.clear_cache()

    report = measure_weight_fidelity(QUANT, REF, corpus=_tiny_corpus(tok))
    assert 0.0 < report.kl.mean < 5.0  # 4-bit weights: small-but-nonzero drift on short prose
    assert report.quant_bits == 4
    assert report.reference_bits is None  # bf16 reference is full precision
    assert report.peak_memory_bytes < int(mx.device_info()["max_recommended_working_set_size"])
