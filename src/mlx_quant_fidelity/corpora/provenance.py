"""Frozen provenance carriers so a report records exactly how a number was produced."""

from dataclasses import dataclass, replace

import mlx.core as mx


@dataclass(frozen=True, slots=True)
class CorpusProvenance:
    """Everything needed to reproduce a corpus tokenization + chunking."""

    name: str
    split: str
    tokenizer_id: str
    chunk_length: int
    stride: int
    bos_policy: str
    final_chunk_policy: str
    normalization: str
    n_tokens: int
    dataset_revision: str | None = None


@dataclass(frozen=True, slots=True)
class Corpus:
    """Tokenized corpus: a list of fixed-length token-id chunks + its provenance."""

    chunks: tuple[mx.array, ...]
    provenance: CorpusProvenance


def scored_provenance(corpus: Corpus, n_scored_chunks: int) -> CorpusProvenance:
    """The corpus provenance, with ``n_tokens`` recounted when only a prefix of chunks is scored.

    ``max_chunks`` scores a prefix of a caller's corpus; the report must not claim the whole
    corpus's token count. Returns the original object untouched when nothing was sliced.
    """
    if n_scored_chunks >= len(corpus.chunks):
        return corpus.provenance
    scored = corpus.chunks[:n_scored_chunks]
    return replace(corpus.provenance, n_tokens=sum(int(c.size) for c in scored))
