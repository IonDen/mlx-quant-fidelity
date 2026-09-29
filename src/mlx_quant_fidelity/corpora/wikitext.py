"""WikiText-2 test-split loader + pinned chunking contract.

Comparability stance (0.1.0): INTERNALLY-CONSISTENT (cross-MLX-method), not
llama.cpp-comparable — non-overlapping fixed-length chunks, BOS prepended once
per chunk is OFF by default, final partial chunk dropped. Documented here and
carried in CorpusProvenance so reports never imply cross-tool comparability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import mlx.core as mx

from mlx_quant_fidelity.corpora.provenance import Corpus, CorpusProvenance
from mlx_quant_fidelity.errors import CorpusError

if TYPE_CHECKING:
    from collections.abc import Sequence

_WIKITEXT2_REPO = "Salesforce/wikitext"
# Commit of Salesforce/wikitext the corpus is fetched at (unchanged since 2024-01-04); a moving
# branch would let the scored text change under an unchanged report.
_WIKITEXT2_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
_WIKITEXT2_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"


class _TokenizerProtocol(Protocol):  # pragma: no cover - structural type, never instantiated
    """Minimal protocol for tokenizers accepted by :func:`load_wikitext2`."""

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        """Encode text to a list of integer token IDs."""
        ...

    @property
    def name_or_path(self) -> str:
        """Identifier for the tokenizer (model name or local path)."""
        ...


def chunk_token_ids(
    token_ids: Sequence[int], *, chunk_length: int, drop_final_partial: bool
) -> list[mx.array]:
    """Split a flat token-id sequence into fixed-length, non-overlapping chunks.

    Args:
        token_ids: Flat sequence of integer token IDs.
        chunk_length: Number of tokens per chunk. Must be positive.
        drop_final_partial: If True, discard the final chunk when it is shorter
            than ``chunk_length``. If False, keep it as-is.

    Returns:
        List of ``mx.array`` chunks, each of dtype int32 (or int64 depending on
        MLX default for integer arrays).

    Raises:
        CorpusError: If ``chunk_length`` is not positive.
    """
    if chunk_length <= 0:
        raise CorpusError("chunk_length must be positive")
    chunks: list[mx.array] = []
    for start in range(0, len(token_ids), chunk_length):
        window = token_ids[start : start + chunk_length]
        if len(window) < chunk_length and drop_final_partial:
            break
        chunks.append(mx.array(list(window)))
    return chunks


def _bos_policy(tokenizer: _TokenizerProtocol) -> str:
    """``"first-chunk"`` when the tokenizer's default encode prepends a special token, else ``"none"``.

    The corpus is encoded once with the tokenizer's defaults and then chunked, so a tokenizer
    that adds BOS (Llama-3) puts it at the start of chunk 0 only. Detected by comparing the
    default encoding of a probe string with the ``add_special_tokens=False`` one; a tokenizer
    whose ``encode`` does not accept the keyword is recorded as ``"none"``.
    """
    try:
        with_special = tokenizer.encode("a")
        without_special = tokenizer.encode("a", add_special_tokens=False)
    except TypeError:
        return "none"
    return "first-chunk" if len(with_special) > len(without_special) else "none"


def load_wikitext2(
    tokenizer: _TokenizerProtocol,
    *,
    chunk_length: int = 512,
    drop_final_partial: bool = True,
    max_chunks: int | None = None,
    tokenizer_id: str | None = None,
) -> Corpus:
    """Load WikiText-2 test split, tokenize, and chunk.

    Requires network access (raw HuggingFace Hub fetch). Covered by a
    ``@pytest.mark.network`` test; excluded from default-CI coverage.

    Args:
        tokenizer: Any tokenizer with an ``encode(text: str) -> list[int]`` method
            and a ``name_or_path`` attribute (e.g. a HuggingFace
            ``PreTrainedTokenizerFast``).
        chunk_length: Token IDs per chunk. Default 512.
        drop_final_partial: Drop the last chunk if it is shorter than
            ``chunk_length``. Default True (0.1.0 comparability stance).
        max_chunks: If set, keep only the first ``max_chunks`` chunks. The
            ``n_tokens`` in the provenance is computed from the capped set.
        tokenizer_id: Identifier recorded in provenance. Defaults to
            ``tokenizer.name_or_path``, which for an mlx-lm tokenizer is a local
            cache path — pass the model repo id to keep reports portable.

    Returns:
        A :class:`Corpus` with provenance recording the exact tokenization +
        chunking recipe used.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - hard dep, defensive
        raise CorpusError("huggingface-hub is required to load WikiText-2") from exc

    path = hf_hub_download(
        _WIKITEXT2_REPO, _WIKITEXT2_FILE, repo_type="dataset", revision=_WIKITEXT2_REVISION
    )
    text = _read_parquet_text(path)
    token_ids = tokenizer.encode(text)  # tokenizer defaults; see _bos_policy
    chunks = chunk_token_ids(
        token_ids, chunk_length=chunk_length, drop_final_partial=drop_final_partial
    )
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    provenance = CorpusProvenance(
        name="wikitext-2-raw",
        split="test",
        tokenizer_id=tokenizer_id if tokenizer_id is not None else tokenizer.name_or_path,
        chunk_length=chunk_length,
        stride=chunk_length,
        bos_policy=_bos_policy(tokenizer),
        final_chunk_policy="drop" if drop_final_partial else "keep",
        normalization="raw",
        # n_tokens is computed from the FINAL (post-max_chunks) chunks only
        n_tokens=sum(int(c.size) for c in chunks),
        dataset_revision=_WIKITEXT2_REVISION,
    )
    return Corpus(chunks=tuple(chunks), provenance=provenance)


def _read_parquet_text(path: str) -> str:  # pragma: no cover
    """Read the ``text`` column from a WikiText-2 parquet file, joining rows with blank lines."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)  # type: ignore[no-untyped-call]
    return "\n\n".join(str(v) for v in table.column("text").to_pylist())
