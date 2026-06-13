"""Frozen provenance carriers so a report records exactly how a number was produced."""

from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class Corpus:
    """Tokenized corpus: a list of fixed-length token-id chunks + its provenance."""

    chunks: tuple[mx.array, ...]
    provenance: CorpusProvenance
