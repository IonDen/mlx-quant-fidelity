import mlx.core as mx
import pytest

from mlx_quant_fidelity.corpora.provenance import CorpusProvenance
from mlx_quant_fidelity.corpora.wikitext import chunk_token_ids
from mlx_quant_fidelity.errors import CorpusError

# ---------------------------------------------------------------------------
# Pure chunking contract — golden fixture tests (non-overlapping, drop vs keep)
# ---------------------------------------------------------------------------


def test_chunking_non_overlapping_drops_short_final():
    ids = list(range(10))
    chunks = chunk_token_ids(ids, chunk_length=4, drop_final_partial=True)
    assert [c.tolist() for c in chunks] == [[0, 1, 2, 3], [4, 5, 6, 7]]  # last 2 dropped


def test_chunking_keeps_final_when_requested():
    ids = list(range(10))
    chunks = chunk_token_ids(ids, chunk_length=4, drop_final_partial=False)
    assert chunks[-1].tolist() == [8, 9]


def test_chunk_is_mx_array():
    chunks = chunk_token_ids([1, 2, 3, 4], chunk_length=2, drop_final_partial=True)
    assert isinstance(chunks[0], mx.array)


def test_exact_length_no_partial():
    # 8 tokens / chunk_length=4 -> exactly 2 full chunks, nothing to drop
    chunks = chunk_token_ids(list(range(8)), chunk_length=4, drop_final_partial=True)
    assert len(chunks) == 2
    assert [c.tolist() for c in chunks] == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_chunks_are_non_overlapping():
    # Stride == chunk_length: consecutive chunks must not share indices
    ids = list(range(12))
    chunks = chunk_token_ids(ids, chunk_length=4, drop_final_partial=True)
    all_ids = [v for c in chunks for v in c.tolist()]
    # No duplicates — non-overlapping means each token appears at most once
    assert len(all_ids) == len(set(all_ids))


def test_invalid_chunk_length_raises():
    with pytest.raises(CorpusError, match="chunk_length must be positive"):
        chunk_token_ids([1, 2, 3], chunk_length=0, drop_final_partial=True)


# ---------------------------------------------------------------------------
# Network-gated test: live Hub fetch + full load_wikitext2 pipeline
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_load_wikitext2_live_provenance_and_chunks():
    """Call load_wikitext2 against the live Hub and assert provenance + chunks are correct.

    Requires a real tokenizer (mlx-lm PreTrainedTokenizerFast) and network access.
    Gated by @pytest.mark.network — skipped in default CI; opt in with --run-network.
    """
    from mlx_lm import load as mlx_load

    from mlx_quant_fidelity.corpora.wikitext import load_wikitext2

    # Use a small model just for its tokenizer — avoid loading weights
    _model, tokenizer = mlx_load("mlx-community/Llama-3.2-1B-Instruct-4bit")

    corpus = load_wikitext2(tokenizer, chunk_length=512, drop_final_partial=True, max_chunks=5)

    # Provenance fields are populated
    prov = corpus.provenance
    assert isinstance(prov, CorpusProvenance)
    assert prov.name == "wikitext-2-raw"
    assert prov.split == "test"
    assert prov.chunk_length == 512
    assert prov.stride == 512
    assert prov.bos_policy == "none"
    assert prov.final_chunk_policy == "drop"
    assert prov.normalization == "raw"
    assert prov.tokenizer_id != ""

    # Chunks are non-empty mx.arrays of the right size
    assert len(corpus.chunks) > 0
    assert len(corpus.chunks) <= 5
    for chunk in corpus.chunks:
        assert isinstance(chunk, mx.array)
        assert int(chunk.size) == 512

    # n_tokens is sum of chunk sizes (post-max_chunks), not the full corpus length
    assert prov.n_tokens == sum(int(c.size) for c in corpus.chunks)
    assert prov.n_tokens == len(corpus.chunks) * 512
