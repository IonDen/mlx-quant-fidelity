import dataclasses

import pytest
from tests.factories import (
    make_corpus,
    make_fid_report,
    make_provenance,
    make_ranked_weight_report,
    make_weight_report,
)


def test_factory_override_reaches_the_built_object():
    # bug caught: a factory that ignores its keyword overrides (tests would measure defaults)
    assert make_fid_report(verdict="bad").verdict == "bad"
    assert make_weight_report(quant_bits=8).quant_bits == 8
    assert make_provenance(n_tokens=7).n_tokens == 7


def test_factory_rejects_unknown_field():
    # bug caught: a typo'd override being swallowed instead of failing the test that typed it
    with pytest.raises(TypeError):
        make_fid_report(verdikt="bad")


def test_factories_are_keyword_only():
    # bug caught: positional construction creeping back (a swapped str|None type-checks)
    with pytest.raises(TypeError):
        make_fid_report("m")  # type: ignore[misc]


def test_make_corpus_provenance_matches_chunks():
    corpus = make_corpus(n_chunks=3, chunk_length=4)
    assert len(corpus.chunks) == 3
    assert corpus.provenance.n_tokens == 12
    assert corpus.provenance.chunk_length == 4
    assert dataclasses.is_dataclass(corpus.provenance)


def test_ranked_weight_report_carries_label_kl_and_cost():
    # bug caught: the ranking builder dropping its positional inputs (every row would tie)
    rep = make_ranked_weight_report("q4", 0.09, 4200)
    assert rep.quant_model_id == "q4"
    assert rep.kl.mean == 0.09
    assert rep.quant_model_bytes == 4200
