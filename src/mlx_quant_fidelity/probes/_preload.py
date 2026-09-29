"""Pre-load checks: read a model's config.json and refuse untrusted code before any `load`.

mlx-lm's loader executes the Python file a config names under `model_file`. Every measurement
entry point therefore reads `config.json` first (a few KB, no weights) and refuses such a repo
unless the caller opted in. The same config feeds the weights probe's pre-load comparability
gate. Keep this module free of MLX imports: it must run before anything touches the GPU.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from mlx_quant_fidelity.errors import ModelNotAccessibleError, UntrustedModelCodeError


@dataclass(frozen=True, slots=True)
class PreloadResult:
    """What the pre-load check inspected: the parsed config and the commit it came from.

    ``revision`` is the resolved commit hash for a Hub id (the ``snapshots/<sha>`` directory
    ``hf_hub_download`` returned) and ``None`` for a local directory. Every ``load`` must use
    it, so the code that is loaded is exactly the code that was checked.
    """

    config: dict[str, object]
    revision: str | None

    def load_revision(self, user_revision: str | None) -> str | None:
        """The revision to hand to ``load``: the checked commit, else what the user pinned."""
        return self.revision if self.revision is not None else user_revision


def _error_summary(exc: BaseException) -> str:
    """The line of an error worth showing: its first, or the `for url:` line of a bare 404."""
    lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
    if not lines:
        return type(exc).__name__
    if "Request ID" in lines[0]:
        for line in lines[1:]:
            if "for url:" in line:
                return line
    return lines[0]


def _fetch_config(model: str, revision: str | None) -> tuple[dict[str, object], str | None]:
    from huggingface_hub import errors as hub_errors

    path = Path(model)
    resolved: str | None = None
    try:
        if path.is_dir():
            config_path = path / "config.json"
        else:
            from huggingface_hub import hf_hub_download

            config_path = Path(hf_hub_download(model, "config.json", revision=revision))
            # The cache layout is .../snapshots/<commit>/config.json.
            if config_path.parent.parent.name == "snapshots":
                resolved = config_path.parent.name
        with config_path.open() as fh:
            text = fh.read()
    except (
        hub_errors.RepositoryNotFoundError,
        hub_errors.GatedRepoError,
        hub_errors.RevisionNotFoundError,
        hub_errors.EntryNotFoundError,
        hub_errors.LocalEntryNotFoundError,
        hub_errors.HFValidationError,
        hub_errors.HfHubHTTPError,
        OSError,
        UnicodeDecodeError,
    ) as exc:
        raise ModelNotAccessibleError(
            f"could not read config.json for {model!r}: {_error_summary(exc)} "
            "(typo, gated repo, or offline?)"
        ) from exc
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        config = None
    if not isinstance(config, dict):
        raise ModelNotAccessibleError(
            f"config.json for {model!r} is not a JSON object; it cannot be checked or loaded."
        )
    return config, resolved


def read_model_config(model: str, revision: str | None = None) -> dict[str, object]:
    """Return a model's `config.json` without downloading weights.

    A local directory is read from disk and never sent to the Hub. Anything else is fetched
    with ``hf_hub_download``, which falls back to the local cache on a connection error by
    itself — so ``local_files_only`` is deliberately not passed (offline use with a warm cache
    keeps working).
    """
    return _fetch_config(model, revision)[0]


def refuse_custom_code(config: dict[str, object], *, model: str, allow_custom_code: bool) -> None:
    """Raise UntrustedModelCodeError when the config names a `model_file` and no opt-in was given."""
    model_file = config.get("model_file")
    if model_file is None or allow_custom_code:
        return
    raise UntrustedModelCodeError(
        f"{model} ships its own model code (config.json model_file={model_file!r}); "
        "mlx-lm would execute it on load. Re-run with --allow-custom-code "
        "(Python: allow_custom_code=True) only if you trust this repo."
    )


def preload_check(model: str, revision: str | None, *, allow_custom_code: bool) -> PreloadResult:
    """Read the config and refuse custom code; return the config and the commit checked."""
    config, resolved = _fetch_config(model, revision)
    refuse_custom_code(config, model=model, allow_custom_code=allow_custom_code)
    return PreloadResult(config=config, revision=resolved)
