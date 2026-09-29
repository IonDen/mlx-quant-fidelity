import json

import pytest

from mlx_quant_fidelity.errors import UntrustedModelCodeError
from mlx_quant_fidelity.probes import _preload


def test_refuses_repo_with_model_file():
    """Bug: a repo whose config.json names a model_file (mlx-lm executes it on load) loads
    without any opt-in."""
    with pytest.raises(UntrustedModelCodeError, match="--allow-custom-code") as exc:
        _preload.refuse_custom_code(
            {"model_file": "m.py"}, model="evil/repo", allow_custom_code=False
        )
    assert "evil/repo" in str(exc.value)
    assert "m.py" in str(exc.value)


def test_allows_model_file_when_opted_in():
    _preload.refuse_custom_code({"model_file": "m.py"}, model="r", allow_custom_code=True)


def test_plain_config_passes():
    _preload.refuse_custom_code({"model_type": "llama"}, model="r", allow_custom_code=False)


def test_read_model_config_local_dir(tmp_path, monkeypatch):
    """Bug: a local model directory is sent to the Hub as a repo id."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama", "vocab_size": 7}))
    import huggingface_hub

    def boom(*a, **k):
        raise AssertionError("hub must not be called for a local directory")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    assert _preload.read_model_config(str(tmp_path)) == {"model_type": "llama", "vocab_size": 7}


def test_read_model_config_passes_revision(tmp_path, monkeypatch):
    """Bug: the pinned revision is dropped, so the gate reads a different config than load()."""
    import huggingface_hub

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"vocab_size": 3}))
    seen = {}

    def fake(repo, filename, **kwargs):
        seen.update(repo=repo, filename=filename, **kwargs)
        return str(cfg)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    _preload.read_model_config("org/name", "abc123")
    assert seen == {"repo": "org/name", "filename": "config.json", "revision": "abc123"}


def test_read_model_config_falls_back_to_cache_offline(tmp_path, monkeypatch):
    """Bug: forcing local_files_only (or otherwise bypassing hf_hub_download's own cache
    fallback) breaks offline use with a warm cache. We must not pass local_files_only."""
    import huggingface_hub

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"vocab_size": 3}))
    seen = {}

    def fake(repo, filename, **kwargs):
        seen.update(kwargs)
        return str(cfg)  # what hf_hub_download returns from its cache on a connection error

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    assert _preload.read_model_config("org/name") == {"vocab_size": 3}
    assert "local_files_only" not in seen


def test_preload_check_returns_config_and_refuses(monkeypatch):
    monkeypatch.setattr(_preload, "read_model_config", lambda m, r=None: {"model_file": "x.py"})
    with pytest.raises(UntrustedModelCodeError):
        _preload.preload_check("r", None, allow_custom_code=False)
    cfg = _preload.preload_check("r", None, allow_custom_code=True)
    assert cfg == {"model_file": "x.py"}
