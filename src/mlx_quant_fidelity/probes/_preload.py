"""Pre-load checks: read a model's config.json and refuse untrusted code before any `load`.

mlx-lm's loader executes the Python file a config names under `model_file`. Every measurement
entry point therefore reads `config.json` first (a few KB, no weights) and refuses such a repo
unless the caller opted in. The same config feeds the weights probe's pre-load comparability
gate. Keep this module free of MLX imports: it must run before anything touches the GPU.
"""

import json
from pathlib import Path

from mlx_quant_fidelity.errors import UntrustedModelCodeError


def read_model_config(model: str, revision: str | None = None) -> dict[str, object]:
    """Return a model's `config.json` without downloading weights.

    A local directory is read from disk and never sent to the Hub. Anything else is fetched
    with ``hf_hub_download``, which falls back to the local cache on a connection error by
    itself — so ``local_files_only`` is deliberately not passed (offline use with a warm cache
    keeps working).
    """
    path = Path(model)
    if path.is_dir():
        config_path = path / "config.json"
    else:
        from huggingface_hub import hf_hub_download

        config_path = Path(hf_hub_download(model, "config.json", revision=revision))
    with config_path.open() as fh:
        config: dict[str, object] = json.load(fh)
    return config


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


def preload_check(
    model: str, revision: str | None, *, allow_custom_code: bool
) -> dict[str, object]:
    """Read the config and refuse custom code; return the config for later gates."""
    config = read_model_config(model, revision)
    refuse_custom_code(config, model=model, allow_custom_code=allow_custom_code)
    return config
