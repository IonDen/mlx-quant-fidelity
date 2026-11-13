"""Pre-load checks: read a model's config.json and refuse untrusted code before any `load`.

mlx-lm's loader executes the Python file a config names under `model_file`. Every measurement
entry point therefore reads `config.json` first (a few KB, no weights) and refuses such a repo
unless the caller opted in. It also reads `tokenizer_config.json` and refuses, before any weight
download, a tokenizer transformers would only load as remote code. The same config feeds the
weights probe's pre-load comparability gate. Keep this module free of MLX imports: it must run
before anything touches the GPU.
"""

import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _transport_errors() -> tuple[type[Exception], ...]:
    """The base errors of httpx2 and httpx, each when it is installed; nothing when neither is.

    huggingface_hub 2.x speaks httpx2 and 1.x httpx (the two packages' errors are unrelated);
    each falls back to the cache on a connect error or a timeout, and any other transport error
    (a dropped read, a protocol error) escapes. huggingface_hub 0.x is requests-based and
    installs neither.
    """
    errors: list[type[Exception]] = []
    for name in ("httpx2", "httpx"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        errors.append(module.HTTPError)
    return tuple(errors)


def _fetch_config(model: str, revision: str | None) -> tuple[dict[str, object], str | None]:
    from huggingface_hub import errors as hub_errors

    unreadable: tuple[type[Exception], ...] = (
        hub_errors.RepositoryNotFoundError,
        hub_errors.GatedRepoError,
        hub_errors.RevisionNotFoundError,
        hub_errors.EntryNotFoundError,
        hub_errors.LocalEntryNotFoundError,
        hub_errors.HFValidationError,
        hub_errors.HfHubHTTPError,
        *_transport_errors(),
        OSError,
        UnicodeDecodeError,
    )
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
    except unreadable as exc:
        raise ModelNotAccessibleError(
            f"could not read config.json for {model!r}: {_error_summary(exc)} "
            "(typo, gated repo, or offline?)"
        ) from exc
    try:
        config = json.loads(text)
    except (json.JSONDecodeError, RecursionError):  # nesting past the interpreter's limit
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


_TOKENIZER_CONFIG = "tokenizer_config.json"

# Tokenizer classes mlx-lm registers with AutoTokenizer on import (mlx_lm/tokenizer_utils.py);
# a real load sees them as local classes.
_MLX_LM_TOKENIZER_CLASSES = frozenset({"NewlineTokenizer"})


# Largest tokenizer_config.json the pre-load check parses; a larger one is left to the load.
_TOKENIZER_CONFIG_MAX_BYTES = 64 * 1024 * 1024


def _fetch_tokenizer_config(model: str, revision: str | None) -> object:
    """The parsed ``tokenizer_config.json``; ``{}`` or ``None`` when there is none to read.

    ``{}`` is for a local directory without the file. ``None`` is for any source that cannot be
    read: a Hub repo without the file (the download raises), a fetch or transport error, bad
    JSON, nesting past the recursion limit, a file over 64 MiB. Both lead to no refusal:
    transformers reads a missing file as ``{}``, and an unreadable one is left to the load.
    """
    path = Path(model)
    try:
        if path.is_dir():
            tok_path = path / _TOKENIZER_CONFIG
            if not tok_path.is_file():
                return {}
        else:
            from huggingface_hub import hf_hub_download

            tok_path = Path(hf_hub_download(model, _TOKENIZER_CONFIG, revision=revision))
        if tok_path.stat().st_size > _TOKENIZER_CONFIG_MAX_BYTES:
            return None
        return json.loads(tok_path.read_text(encoding="utf-8"))
    except Exception:  # unreadable means defer to the load: this check never refuses on a guess
        return None


def _has_local_tokenizer(model_type: str, class_name: str | None) -> bool:
    """Mirror transformers 5.12's ``has_local_code`` (plus its wrong-hub-class override).

    Reads transformers' own registries, so the answer moves with the installed version.
    """
    import importlib

    auto = importlib.import_module("transformers.models.auto.tokenization_auto")
    config_auto = importlib.import_module("transformers.models.auto.configuration_auto")
    if model_type in auto.MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS:
        return True  # transformers drops the hub's auto_map for these model types
    if (
        model_type in config_auto.CONFIG_MAPPING
        and config_auto.CONFIG_MAPPING[model_type] in auto.TOKENIZER_MAPPING
    ):
        return True
    if class_name is None:
        return False
    name = class_name.removesuffix("Fast")
    if name in _MLX_LM_TOKENIZER_CLASSES:
        return True
    from_name = auto.tokenizer_class_from_name
    return from_name(name) is not None or from_name(name + "Fast") is not None


def _tokenizer_needs_remote_code(tokenizer_config: object, config: dict[str, object]) -> bool:
    """True only where AutoTokenizer with ``trust_remote_code=False`` refuses for certain.

    That is an ``AutoTokenizer`` entry in ``tokenizer_config.json``'s ``auto_map`` (a dict, or
    the legacy ``[slow, fast]`` list) and no local tokenizer: neither the config's model type nor
    the named ``tokenizer_class`` is one transformers or mlx-lm provides. transformers 5.12
    takes remote tokenizer code from ``tokenizer_config.json`` only, never from ``config.json``.
    Every shape it would not refuse cleanly passes, and the load decides.
    """
    if not isinstance(tokenizer_config, dict) or "auto_map" not in tokenizer_config:
        return False
    auto_map = tokenizer_config["auto_map"]
    entry = auto_map.get("AutoTokenizer") if isinstance(auto_map, dict) else auto_map
    if not isinstance(entry, list) or len(entry) < 2:
        return False
    class_ref = entry[1] if entry[1] is not None else entry[0]
    class_name = tokenizer_config.get("tokenizer_class")
    model_type = config.get("model_type")
    if not isinstance(class_ref, str) or not isinstance(model_type, str):
        return False  # no model_type: AutoConfig guesses one from the path
    if class_name is not None and not isinstance(class_name, str):
        return False
    try:
        return not _has_local_tokenizer(model_type, class_name)
    except Exception:  # an import or registry failure is no proof of remote code
        return False


def _tokenizer_refusal(model: str) -> str:
    return (
        f"{model} ships its own tokenizer code, which transformers would run on load. "
        "Re-run with --allow-custom-code (Python: allow_custom_code=True) only if you "
        "trust this repo."
    )


def preload_check(
    model: str, revision: str | None, *, allow_custom_code: bool, check_tokenizer: bool = True
) -> PreloadResult:
    """Read the configs and refuse custom code; return the config and the commit checked.

    Fetches ``config.json`` and, without the opt-in, ``tokenizer_config.json`` at the same
    commit: nothing else, so a refusal comes before any weight download. A caller that never
    loads a tokenizer passes ``check_tokenizer=False`` and only ``config.json`` is fetched.
    """
    config, resolved = _fetch_config(model, revision)
    refuse_custom_code(config, model=model, allow_custom_code=allow_custom_code)
    if check_tokenizer and not allow_custom_code:
        tokenizer_config = _fetch_tokenizer_config(
            model, resolved if resolved is not None else revision
        )
        if _tokenizer_needs_remote_code(tokenizer_config, config):
            raise UntrustedModelCodeError(_tokenizer_refusal(model))
    return PreloadResult(config=config, revision=resolved)


# Both refusals transformers raises (as ValueError) for remote code without trust_remote_code:
# the explicit `trust_remote_code=False` path and the prompt-timeout / no-TTY path.
_REMOTE_CODE_REFUSAL = "trust_remote_code=True"


def load_checked(
    load: Callable[..., Any], model: str, *, allow_custom_code: bool, **kwargs: Any
) -> Any:
    """Call mlx-lm's ``load`` with the tokenizer's remote code governed by ``allow_custom_code``.

    Passes ``tokenizer_config={"trust_remote_code": allow_custom_code}`` so transformers never
    prompts, and turns its refusal into UntrustedModelCodeError when no opt-in was given.
    """
    try:
        return load(model, tokenizer_config={"trust_remote_code": allow_custom_code}, **kwargs)
    except ValueError as exc:
        if allow_custom_code or _REMOTE_CODE_REFUSAL not in str(exc):
            raise
        raise UntrustedModelCodeError(_tokenizer_refusal(model)) from exc
