import json

import pytest

from mlx_quant_fidelity.errors import UntrustedModelCodeError
from mlx_quant_fidelity.probes import _preload


@pytest.fixture(autouse=True)
def _pinned_device(monkeypatch):
    """Hermetic device for the kv/weights load tests below (see test_kv_fakeforward)."""
    import mlx.core as mx

    from mlx_quant_fidelity.probes import kv as kv_mod

    monkeypatch.setattr(kv_mod, "_max_working_set_bytes", lambda: 26_800_603_136)
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 26_800_603_136}
    )


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


def test_read_model_config_reads_whatever_path_hf_hub_download_returns_without_local_files_only(
    tmp_path, monkeypatch
):
    """Bug: forcing local_files_only (or otherwise bypassing hf_hub_download's own cache
    fallback) breaks offline use with a warm cache. We must not pass local_files_only, and we
    read the file hf_hub_download hands back (its cache path when the connection fails)."""
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
    monkeypatch.setattr(_preload, "_fetch_config", lambda m, r=None: ({"model_file": "x.py"}, None))
    with pytest.raises(UntrustedModelCodeError):
        _preload.preload_check("r", None, allow_custom_code=False)
    cfg = _preload.preload_check("r", None, allow_custom_code=True)
    assert cfg.config == {"model_file": "x.py"}


def _hub_returning_snapshot(monkeypatch, tmp_path, sha="abc123", config=None):
    """Fake hf_hub_download returning the real cache layout `.../snapshots/<sha>/config.json`."""
    import huggingface_hub

    snap = tmp_path / "models--org--name" / "snapshots" / sha
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps(config or {"vocab_size": 3}))
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda *a, **k: str(snap / "config.json")
    )


def test_preload_check_returns_the_commit_it_checked_for_a_hub_id(tmp_path, monkeypatch):
    """Bug: the check reads `main` at time T1 and load() re-resolves `main` at T2, so a repo
    that changes in between (adds a model_file) is loaded unchecked."""
    _hub_returning_snapshot(monkeypatch, tmp_path, "abc123")
    result = _preload.preload_check("org/name", None, allow_custom_code=False)
    assert result.revision == "abc123"
    assert result.config == {"vocab_size": 3}


def test_preload_check_local_dir_has_no_revision(tmp_path):
    """Bug: a local path gets a revision (a Hub concept) passed to load()."""
    (tmp_path / "config.json").write_text(json.dumps({"vocab_size": 3}))
    result = _preload.preload_check(str(tmp_path), None, allow_custom_code=False)
    assert result.revision is None
    assert result.config == {"vocab_size": 3}


def test_kv_measure_loads_exactly_the_checked_commit(tmp_path, monkeypatch):
    """Bug: measure_kv_fidelity loads the user revision (None -> main) instead of the commit
    the pre-load check inspected."""
    import mlx_lm

    from mlx_quant_fidelity.probes import kv as kv_mod

    _hub_returning_snapshot(monkeypatch, tmp_path, "abc123")
    seen = {}

    class _StopError(Exception):
        pass

    def fake_load(repo, **kw):
        seen.update(kw)
        raise _StopError

    monkeypatch.setattr(mlx_lm, "load", fake_load)
    from tests.factories import make_corpus

    with pytest.raises(_StopError):
        kv_mod.measure_kv_fidelity("org/name", corpus=make_corpus())
    assert seen["revision"] == "abc123"


def test_weights_measure_loads_exactly_the_checked_commits(tmp_path, monkeypatch):
    """Bug: the weights probe loads either repo at a re-resolved revision."""
    import huggingface_hub
    import mlx_lm
    from tests.factories import make_corpus

    from mlx_quant_fidelity.probes import weights as w

    def fake_dl(repo, filename, **kw):
        snap = tmp_path / repo.replace("/", "--") / "snapshots" / f"sha-{repo.split('/')[-1]}"
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "config.json").write_text(
            json.dumps({"model_type": "llama", "vocab_size": 3, "quantization": {"bits": 4}})
        )
        return str(snap / "config.json")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)
    monkeypatch.setattr(w, "_resolve_weight_bytes", lambda *a, **k: 1000)
    monkeypatch.setattr(w, "_hub_weight_bytes", lambda *a, **k: 1000)
    loads = {}

    class _StopError(Exception):
        pass

    def fake_load(repo, **kw):
        loads[repo] = kw.get("revision")
        if len(loads) == 2:
            raise _StopError
        return object(), object(), {"model_type": "llama", "vocab_size": 3}

    monkeypatch.setattr(mlx_lm, "load", fake_load)
    with pytest.raises(_StopError):
        w.measure_weight_fidelity(
            "org/quant",
            "org/ref",
            corpus=make_corpus(n_chunks=2, chunk_length=4, tokenizer_id="org/m-bf16"),
        )
    assert loads == {"org/ref": "sha-ref", "org/quant": "sha-quant"}


def test_read_model_config_wraps_os_errors_naming_the_model(tmp_path):
    """Bug: a local dir without config.json escapes as a bare FileNotFoundError with no model id."""
    from mlx_quant_fidelity.errors import ModelNotAccessibleError

    with pytest.raises(ModelNotAccessibleError, match=r"config\.json for") as exc:
        _preload.read_model_config(str(tmp_path))
    assert str(tmp_path) in str(exc.value)


@pytest.mark.parametrize("body", ["{not json", "[1, 2]", '"a string"'])
def test_malformed_config_is_model_not_accessible(tmp_path, body):
    """Bug: a config.json that is not a JSON object escaped as JSONDecodeError / AttributeError,
    which the CLI reports as an internal error (exit 1) instead of a user error naming the repo."""
    from mlx_quant_fidelity.errors import ModelNotAccessibleError

    (tmp_path / "config.json").write_text(body)
    with pytest.raises(ModelNotAccessibleError, match="JSON object"):
        _preload.preload_check(str(tmp_path), None, allow_custom_code=False)


def test_non_utf8_config_is_a_model_not_accessible_error(tmp_path):
    """Bug: a config.json that is not valid UTF-8 escapes as a raw UnicodeDecodeError (an
    internal error, exit 1) instead of the package-rooted user error."""
    from mlx_quant_fidelity.errors import ModelNotAccessibleError

    (tmp_path / "config.json").write_bytes(b'{"model_type": "\xff\xfe"}')
    with pytest.raises(ModelNotAccessibleError):
        _preload.read_model_config(str(tmp_path))


# ---------------------------------------------------------------------------
# Tokenizer remote code (transformers' trust_remote_code), at every load site
# ---------------------------------------------------------------------------


def _remote_code_error(kind: str) -> ValueError:
    """The ValueError the installed transformers raises for a tokenizer that needs remote code.

    ``refused``: ``trust_remote_code=False`` was passed (what the probes pass without the opt-in).
    ``timeout``: nothing was passed and the interactive prompt timed out or had no TTY (what
    mlx-lm's bare ``load`` hits).
    """
    from transformers import dynamic_module_utils as dmu

    try:
        if kind == "refused":
            dmu.resolve_trust_remote_code(
                False, "org/name", has_local_code=False, has_remote_code=True
            )
        else:
            dmu._raise_timeout_error(None, None)
    except ValueError as exc:
        return exc
    raise AssertionError("transformers did not refuse remote code")  # pragma: no cover


class _StopError(Exception):
    pass


def _tokenizer_gated_load(seen: dict[str, object], *, kind: str = "refused"):
    """A fake mlx_lm.load that behaves like transformers' AutoTokenizer for a remote-code repo:
    it refuses unless ``tokenizer_config`` trusts remote code, else stops the run."""

    def fake_load(repo, **kw):
        seen.update(kw)
        if (kw.get("tokenizer_config") or {}).get("trust_remote_code") is not True:
            raise _remote_code_error(kind)
        raise _StopError

    return fake_load


def _stub_kv_before_load(monkeypatch):
    from mlx_quant_fidelity.probes import kv as kv_mod

    monkeypatch.setattr(kv_mod, "preload_check", lambda *a, **k: _preload.PreloadResult({}, None))
    monkeypatch.setattr(kv_mod, "install_memory_caps", lambda: (0, 0))


def _stub_weights_before_load(monkeypatch):
    from mlx_quant_fidelity.probes import weights as w

    cfg = {"model_type": "llama", "vocab_size": 3, "quantization": {"bits": 4, "group_size": 64}}
    monkeypatch.setattr(w, "preload_check", lambda *a, **k: _preload.PreloadResult(cfg, None))
    monkeypatch.setattr(w, "install_memory_caps", lambda: (0, 0))
    monkeypatch.setattr(w, "_resolve_weight_bytes", lambda *a, **k: 1000)
    monkeypatch.setattr(w, "_hub_weight_bytes", lambda *a, **k: 1000)


def _run_site(site: str, *, allow_custom_code: bool) -> None:
    from tests.factories import make_corpus

    from mlx_quant_fidelity.probes import kv as kv_mod
    from mlx_quant_fidelity.probes import weights as w
    from mlx_quant_fidelity.runners import compare as cmp

    corpus = make_corpus(n_chunks=2, chunk_length=4, tokenizer_id="org/m-bf16")
    if site == "kv":
        kv_mod.measure_kv_fidelity("org/name", corpus=corpus, allow_custom_code=allow_custom_code)
    elif site == "weights":
        w.measure_weight_fidelity(
            "org/quant", "org/ref", corpus=corpus, allow_custom_code=allow_custom_code
        )
    else:
        cmp._load_model("org/name", "rev", allow_custom_code=allow_custom_code)


_SITES = ["kv", "weights", "compare-kv"]


@pytest.mark.parametrize("site", _SITES)
@pytest.mark.parametrize("kind", ["refused", "timeout"])
def test_tokenizer_remote_code_without_opt_in_is_untrusted_model_code(monkeypatch, site, kind):
    """Bug: a repo whose tokenizer needs remote code escapes the load as a bare transformers
    ValueError (the CLI's internal error, exit 1) instead of the opt-in refusal (exit 2)."""
    import mlx_lm

    _stub_kv_before_load(monkeypatch)
    _stub_weights_before_load(monkeypatch)
    seen: dict[str, object] = {}
    monkeypatch.setattr(mlx_lm, "load", _tokenizer_gated_load(seen, kind=kind))
    with pytest.raises(UntrustedModelCodeError, match="--allow-custom-code") as exc:
        _run_site(site, allow_custom_code=False)
    assert "tokenizer" in str(exc.value)
    assert isinstance(exc.value.__cause__, ValueError)


@pytest.mark.parametrize("site", _SITES)
@pytest.mark.parametrize("allow", [False, True])
def test_every_load_site_passes_the_opt_in_to_the_tokenizer(monkeypatch, site, allow):
    """Bug: a load site calls mlx_lm.load without tokenizer_config, so --allow-custom-code never
    reaches AutoTokenizer (opt-in ignored) and, without it, transformers prompts on a TTY."""
    import mlx_lm

    _stub_kv_before_load(monkeypatch)
    _stub_weights_before_load(monkeypatch)
    received: list[object] = []

    def recording_load(repo, **kw):
        received.append(kw.get("tokenizer_config", "<not passed>"))
        if repo == "org/ref":  # the weights probe loads the reference first, then the quant
            return object(), object(), {"model_type": "llama", "vocab_size": 3}
        raise _StopError

    monkeypatch.setattr(mlx_lm, "load", recording_load)
    with pytest.raises(_StopError):
        _run_site(site, allow_custom_code=allow)
    expected_loads = 2 if site == "weights" else 1
    assert received == [{"trust_remote_code": allow}] * expected_loads


def test_opted_in_tokenizer_code_loads(monkeypatch):
    """With the opt-in the remote-code tokenizer loads: the gated fake gets past AutoTokenizer
    (and stops the run there) instead of refusing."""
    import mlx_lm

    from mlx_quant_fidelity.runners import compare as cmp

    seen: dict[str, object] = {}
    monkeypatch.setattr(mlx_lm, "load", _tokenizer_gated_load(seen))
    with pytest.raises(_StopError):
        cmp._load_model("org/name", "rev", allow_custom_code=True)


def test_other_load_value_errors_are_not_reported_as_untrusted_code(monkeypatch):
    """Bug: the refusal mapping catches every ValueError from load, so an unrelated load failure
    tells the user to re-run with --allow-custom-code."""
    import mlx_lm

    from mlx_quant_fidelity.runners import compare as cmp

    def bad_load(repo, **kw):
        raise ValueError("Model type foo not supported.")

    monkeypatch.setattr(mlx_lm, "load", bad_load)
    with pytest.raises(ValueError, match="foo not supported") as exc:
        cmp._load_model("org/name", "rev", allow_custom_code=False)
    assert not isinstance(exc.value, UntrustedModelCodeError)


def test_opted_in_remote_code_error_is_not_reworded_as_a_missing_opt_in(monkeypatch):
    """Bug: with --allow-custom-code already passed, a remote-code error from transformers is
    still reported as "re-run with --allow-custom-code", which the user already did."""
    import mlx_lm

    from mlx_quant_fidelity.runners import compare as cmp

    def still_refusing(repo, **kw):
        raise _remote_code_error("refused")

    monkeypatch.setattr(mlx_lm, "load", still_refusing)
    with pytest.raises(ValueError, match="trust_remote_code=True") as exc:
        cmp._load_model("org/name", "rev", allow_custom_code=True)
    assert not isinstance(exc.value, UntrustedModelCodeError)
