"""Shared, keyword-only builders for the frozen report and corpus dataclasses.

Defaults match the long-standing ``_fake_report`` / ``_weight_report`` helpers, so a test that
overrides one field states exactly what it cares about. Tests import from here, never from
another test module.
"""

from typing import Any

import mlx.core as mx

from mlx_quant_fidelity.corpora.provenance import Corpus, CorpusProvenance
from mlx_quant_fidelity.metrics import ScalarSummary
from mlx_quant_fidelity.report import FidelityReport, WeightFidelityReport


def make_provenance(**kw: Any) -> CorpusProvenance:
    """A ``CorpusProvenance`` (wikitext-2-raw test split, 512x512, 100 tokens) with overrides."""
    base: dict[str, Any] = {
        "name": "wikitext-2-raw",
        "split": "test",
        "tokenizer_id": "tok",
        "chunk_length": 512,
        "stride": 512,
        "bos_policy": "none",
        "final_chunk_policy": "drop",
        "normalization": "raw",
        "n_tokens": 100,
    }
    base.update(kw)
    return CorpusProvenance(**base)


def make_fid_report(**kw: Any) -> FidelityReport:
    """A stress-mode stock 4-bit ``FidelityReport`` (verdict ``marginal``) with overrides."""
    base: dict[str, Any] = {
        "model_id": "m",
        "model_revision": None,
        "kv_bits": 4,
        "kv_group_size": 64,
        "quantize_start": 0,
        "quantize_mode": "stress",
        "kl": ScalarSummary(0.02, 0.01, 0.2, 1.5),
        "flip_rate": 0.03,
        "perplexity_ref": 10.0,
        "perplexity_quant": 10.4,
        "perplexity_delta": 0.4,
        "n_positions": 100,
        "n_chunks": 1,
        "corpus": make_provenance(),
        "mlx_version": "0.21",
        "mlx_lm_version": "0.31.3",
        "peak_memory_bytes": 1,
        "cache_supported": True,
        "verdict": "marginal",
        "warnings": (),
    }
    base.update(kw)
    return FidelityReport(**base)


def make_weight_report(**kw: Any) -> WeightFidelityReport:
    """A 4-bit affine ``WeightFidelityReport`` (verdict ``marginal``) with overrides."""
    base: dict[str, Any] = {
        "quant_model_id": "org/m-4bit",
        "quant_revision": None,
        "reference_model_id": "org/m-bf16",
        "reference_revision": None,
        "quant_bits": 4,
        "quant_group_size": 64,
        "quant_mode": "affine",
        "per_layer": False,
        "reference_bits": None,
        "kl": ScalarSummary(0.06, 0.03, 0.4, 2.0),
        "flip_rate": 0.03,
        "perplexity_ref": 10.0,
        "perplexity_quant": 10.6,
        "perplexity_delta": 0.6,
        "n_positions": 1000,
        "n_chunks": 2,
        "corpus": make_provenance(tokenizer_id="org/m-bf16", n_tokens=1024),
        "mlx_version": "0.21",
        "mlx_lm_version": "0.31.3",
        "peak_memory_bytes": 18_000_000_000,
        "quant_model_bytes": 4_000_000_000,
        "reference_model_bytes": 14_000_000_000,
        "verdict": "marginal",
        "warnings": (),
    }
    base.update(kw)
    return WeightFidelityReport(**base)


def make_ranked_weight_report(label: str, kl_mean: float, cost: int) -> WeightFidelityReport:
    """A weight report for ranking tests: flat KL summary at ``kl_mean``, ``cost`` quant bytes."""
    return make_weight_report(
        quant_model_id=label,
        reference_model_id="ref",
        kl=ScalarSummary(kl_mean, kl_mean, kl_mean, kl_mean),
        flip_rate=0.02,
        perplexity_quant=10.1,
        perplexity_delta=0.1,
        n_positions=10,
        n_chunks=2,
        corpus=make_provenance(tokenizer_id="ref", n_tokens=10),
        peak_memory_bytes=1,
        quant_model_bytes=cost,
        reference_model_bytes=8000,
        verdict="good",
        warnings=(),
    )


def make_corpus(*, n_chunks: int = 2, chunk_length: int = 8, **prov_kw: Any) -> Corpus:
    """A tiny ``Corpus`` of ``n_chunks`` sequential-id chunks; provenance fields overridable."""
    chunks = tuple(
        mx.arange(i * chunk_length, (i + 1) * chunk_length, dtype=mx.int32) for i in range(n_chunks)
    )
    prov_kw.setdefault("chunk_length", chunk_length)
    prov_kw.setdefault("stride", chunk_length)
    prov_kw.setdefault("n_tokens", n_chunks * chunk_length)
    return Corpus(chunks=chunks, provenance=make_provenance(**prov_kw))


__all__ = [
    "make_corpus",
    "make_fid_report",
    "make_provenance",
    "make_ranked_weight_report",
    "make_weight_report",
]
