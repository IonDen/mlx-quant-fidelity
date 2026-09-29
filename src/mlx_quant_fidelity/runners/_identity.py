"""Resume identities for `compare` partials, built in exactly one place.

The orchestrator (deciding whether a partial can be resumed) and the weight worker (writing the
partial) both call these builders, so the two can never drift apart. Pure: stdlib only.
"""

import importlib.metadata
import json
import os
import tempfile
from pathlib import Path

# Bump the relevant constant when that mode's partial format or cost formula changes, so only
# that mode's old partials are rejected. The two modes' partials are independent.
KV_PARTIAL_SCHEMA_VERSION = 4
WEIGHT_PARTIAL_SCHEMA_VERSION = 3  # bumped: identity now carries mlx/mlx-lm versions

# The footing every `compare kv` row is ranked on, regardless of a method's native footing.
# Recorded in the run identity so a change to what it means invalidates old partials.
RANKED_FOOTING = "quantizer_only"


def package_versions() -> dict[str, str]:
    """The installed mlx and mlx-lm versions a measurement was produced under."""
    return {
        "mlx_version": importlib.metadata.version("mlx"),
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
    }


def weight_run_identity(
    *,
    quant: str,
    reference: str,
    max_chunks: int | None,
    quant_revision: str | None,
    reference_revision: str | None,
    allow_custom_code: bool = False,
) -> dict[str, object]:
    """Identity of one weight-compare target; a partial resumes only on an exact match."""
    return {
        "mode": "weight",
        "quant": quant,
        "reference": reference,
        "max_chunks": max_chunks,
        "schema_version": WEIGHT_PARTIAL_SCHEMA_VERSION,
        "quant_revision": quant_revision,
        "reference_revision": reference_revision,
        "allow_custom_code": allow_custom_code,
        **package_versions(),
    }


def kv_run_identity(
    *,
    model_id: str,
    model_revision: str | None,
    method_name: str,
    params: dict[str, object],
    method_provenance: dict[str, str],
    quantize_start: int,
    max_chunks: int | None,
    chunk_length: int,
) -> dict[str, object]:
    """Identity of one KV-compare method run; a partial resumes only on an exact match."""
    return {
        "mode": "kv",
        "model_id": model_id,
        "model_revision": model_revision,
        "method": method_name,
        "params": dict(params),
        "method_provenance": method_provenance,
        **package_versions(),
        "quantize_start": quantize_start,
        "max_chunks": max_chunks,
        "chunk_length": chunk_length,
        "schema_version": KV_PARTIAL_SCHEMA_VERSION,
        "ranked_footing": RANKED_FOOTING,
    }


def write_json_atomic(path: Path, payload: object) -> None:
    """Write JSON so a reader never sees a truncated file: temp file in the same dir + replace."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh)
        Path(tmp_name).replace(path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
