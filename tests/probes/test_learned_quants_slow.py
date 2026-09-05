"""Real-repo verification of the weights probe on learned quants (--run-slow --run-network; MAIN thread only).

Pinned mlx-community Qwen3-0.6B repos. Tolerances are derived from the storage formula
(4 + 2*16/64 = 4.5 bits/weight at group 64) and the 2026-09-05 smoke values, not padded.
"""

import mlx.core as mx
import pytest

from mlx_quant_fidelity.corpora.provenance import Corpus, CorpusProvenance
from mlx_quant_fidelity.probes.weights import (
    METHOD_NOT_RECORDED_WARNING,
    bits_per_weight,
    measure_weight_fidelity,
)

REF = "mlx-community/Qwen3-0.6B-bf16"
REF_REV = "42096995f6402fde107068cf530136fe64b604f8"
DWQ = ("mlx-community/Qwen3-0.6B-4bit-DWQ", "e630d870397d5a2d95fe0c9075c6f499fc0fc5c8")
AWQ = ("mlx-community/Qwen3-0.6B-4bit-AWQ", "3c064b3401d4a7d355262a1d518faa823a4d8f11")


def _tiny_corpus(tok, *, chunk_length: int = 64, n_chunks: int = 2) -> Corpus:
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


def _tokenizer_only(repo: str, revision: str):
    """Load for the tokenizer, then release the model before the probe loads its own pair."""
    from mlx_lm import load

    model, tok = load(repo, revision=revision)
    del model
    mx.clear_cache()
    return tok


@pytest.mark.slow
@pytest.mark.network
def test_dwq_repo_measures_as_uniform_4bit_indistinguishable_from_plain():
    """Reds if the gate rejects a DWQ repo, or the geometry/bits-per-weight disagree with the
    formula (DWQ leaves no config trace; it must read exactly like plain 4-bit)."""
    tok = _tokenizer_only(REF, REF_REV)
    mx.reset_peak_memory()
    report = measure_weight_fidelity(
        DWQ[0], REF, corpus=_tiny_corpus(tok), quant_revision=DWQ[1], reference_revision=REF_REV
    )
    assert report.quant_geometry == ((4, 64, 197),)
    assert report.quant_precision == "uniform"
    assert report.quant_n_full_precision == 0
    assert 4.49 < report.quant_bits_per_weight < 4.51
    assert METHOD_NOT_RECORDED_WARNING in report.warnings
    assert 0.0 < report.kl.mean < 5.0


@pytest.mark.slow
@pytest.mark.network
def test_awq_repo_measures_mixed_group_sizes_and_matches_mlx_lm_bits_per_weight():
    """Reds if per-module group sizes are lost (the embedding is group 32), if our bits/weight
    diverges from mlx-lm's own definition on a real model, or if bits/weight and on-disk bytes
    stop agreeing (a wrong safetensors glob, or a numerator drift)."""
    from mlx_lm import load
    from mlx_lm.utils import compute_bits_per_weight, get_total_parameters

    model, tok = load(AWQ[0], revision=AWQ[1])
    assert abs(bits_per_weight(model) - compute_bits_per_weight(model)) < 1e-6
    params = get_total_parameters(model)
    del model
    mx.clear_cache()
    mx.reset_peak_memory()
    report = measure_weight_fidelity(
        AWQ[0], REF, corpus=_tiny_corpus(tok), quant_revision=AWQ[1], reference_revision=REF_REV
    )
    assert report.quant_geometry == ((4, 32, 1), (4, 64, 196))
    assert report.quant_precision == "uniform"
    assert report.quant_n_full_precision == 0
    assert 4.62 < report.quant_bits_per_weight < 4.64
    assert report.quant_model_bytes is not None
    # bits/weight is a per-parameter view of the same on-disk bytes (safetensors headers aside)
    assert (
        abs(report.quant_bits_per_weight * params / 8 - report.quant_model_bytes)
        / report.quant_model_bytes
        < 0.01
    )
