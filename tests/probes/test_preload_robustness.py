"""The pre-load paths on odd or hostile input: they defer to the load or fail with a named error.

Covers the config fetch, the tokenizer-config fetch, the pre-load ModelArgs build, the
TurboQuant refusal text, and the ``check_tokenizer`` switch. No network: every Hub call is a stub.
"""

import copy
import json
import sys
import types
from pathlib import Path

import httpx
import huggingface_hub
import mlx_lm.utils
import pytest

from mlx_quant_fidelity.costs import KVLayout
from mlx_quant_fidelity.errors import CacheNotQuantizableError, ModelNotAccessibleError
from mlx_quant_fidelity.probes import _preload
from mlx_quant_fidelity.probes import kv as kv_mod
from mlx_quant_fidelity.probes._preload import PreloadResult, preload_check
from mlx_quant_fidelity.probes.kv_methods import _gate_turboquant_layout, parse_method_spec

LLAMA_CONFIG = {"model_type": "llama", "hidden_size": 2048, "num_attention_heads": 32}
REMOTE_TOKENIZER = {"auto_map": {"AutoTokenizer": ["tok.RemoteTok", None]}}
DEEP = 100_000  # far past the interpreter's recursion limit
MIB = 1024 * 1024


def _model_dir(tmp_path: Path, *, config: object, tokenizer_text: str | None = None) -> Path:
    (tmp_path / "config.json").write_text(json.dumps(config))
    if tokenizer_text is not None:
        (tmp_path / "tokenizer_config.json").write_text(tokenizer_text)
    return tmp_path


def _padded_json(obj: object, size: int) -> str:
    text = json.dumps(obj)
    return text + " " * (size - len(text))


def _json_raising_recursion_error(monkeypatch) -> list[str]:
    """Replace the module's ``json`` with one whose ``loads`` raises RecursionError.

    Whether deep nesting raises depends on the interpreter (3.13 does, 3.14 parses it), so the
    error is injected. Returns the text of every injected raise.
    """
    raised: list[str] = []

    def _loads(text, *args, **kwargs):
        raised.append(text)
        raise RecursionError("maximum recursion depth exceeded while decoding a JSON array")

    fake = types.SimpleNamespace(loads=_loads, JSONDecodeError=json.JSONDecodeError)
    monkeypatch.setattr(_preload, "json", fake)
    return raised


# ── tokenizer_config.json: unreadable or oversized means defer to the load ──────────────────
def test_a_recursion_error_parsing_the_tokenizer_config_is_unreadable(monkeypatch, tmp_path):
    # bug caught: RecursionError (not a ValueError) escaping the tokenizer-config fetch
    model = _model_dir(tmp_path, config=LLAMA_CONFIG, tokenizer_text=json.dumps(REMOTE_TOKENIZER))
    raised = _json_raising_recursion_error(monkeypatch)
    assert _preload._fetch_tokenizer_config(str(model), None) is None
    assert raised == [json.dumps(REMOTE_TOKENIZER)]  # the parse reached the injected error


def test_deeply_nested_tokenizer_config_is_not_refused(tmp_path):
    # bug caught: an unreadable tokenizer config aborting the run instead of deferring to the load
    # (on any interpreter: deep nesting either raises RecursionError or parses to a list)
    model = _model_dir(tmp_path, config=LLAMA_CONFIG, tokenizer_text="[" * DEEP + "]" * DEEP)
    result = preload_check(str(model), None, allow_custom_code=False)
    assert result == PreloadResult(config=LLAMA_CONFIG, revision=None)


def test_transport_error_on_the_tokenizer_fetch_is_unreadable(monkeypatch):
    # bug caught: an httpx transport error escaping the narrow except list
    def _download(repo_id, filename, *, revision=None):
        raise httpx.ConnectError("connection reset")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    assert _preload._fetch_tokenizer_config("org/model", None) is None


def test_tokenizer_config_of_exactly_64_mib_is_read(tmp_path):
    # the boundary: a file of 64 MiB is still read (only a larger one is skipped)
    text = _padded_json(REMOTE_TOKENIZER, 64 * MIB)
    model = _model_dir(tmp_path, config=LLAMA_CONFIG, tokenizer_text=text)
    assert _preload._fetch_tokenizer_config(str(model), None) == REMOTE_TOKENIZER


def test_tokenizer_config_over_64_mib_is_not_read(tmp_path):
    # bug caught: no size cap, so a hostile repo makes the pre-load step parse any amount of JSON
    text = _padded_json(REMOTE_TOKENIZER, 64 * MIB + 1)
    model = _model_dir(tmp_path, config=LLAMA_CONFIG, tokenizer_text=text)
    assert _preload._fetch_tokenizer_config(str(model), None) is None


# ── config.json: a named error instead of a raw exception ────────────────────────────────────
def test_a_recursion_error_parsing_the_config_is_not_a_json_object(monkeypatch, tmp_path):
    # bug caught: RecursionError from json.loads escaping raw instead of the named error
    model = _model_dir(tmp_path, config=LLAMA_CONFIG)
    raised = _json_raising_recursion_error(monkeypatch)
    with pytest.raises(ModelNotAccessibleError) as info:
        preload_check(str(model), None, allow_custom_code=False)
    assert str(info.value) == (
        f"config.json for {str(model)!r} is not a JSON object; it cannot be checked or loaded."
    )
    assert raised == [json.dumps(LLAMA_CONFIG)]  # the parse reached the injected error


def test_deeply_nested_config_is_not_a_json_object(tmp_path):
    # bug caught: a deeply nested config.json escaping raw (as RecursionError) or passing as a
    # config; on any interpreter it is either unparsable or a list, and a list is not an object
    (tmp_path / "config.json").write_text("[" * DEEP + "]" * DEEP)
    with pytest.raises(ModelNotAccessibleError) as info:
        preload_check(str(tmp_path), None, allow_custom_code=False)
    assert str(info.value) == (
        f"config.json for {str(tmp_path)!r} is not a JSON object; it cannot be checked or loaded."
    )


def test_transport_error_on_the_config_fetch_is_not_accessible(monkeypatch):
    # bug caught: an httpx transport error from hf_hub_download escaping as a raw traceback
    def _download(repo_id, filename, *, revision=None):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    with pytest.raises(ModelNotAccessibleError) as info:
        preload_check("org/model", None, allow_custom_code=False)
    assert str(info.value) == (
        "could not read config.json for 'org/model': connection refused "
        "(typo, gated repo, or offline?)"
    )


# ── check_tokenizer=False: config.json only ─────────────────────────────────────────────────
def _recording_download(monkeypatch, tmp_path):
    model = _model_dir(tmp_path, config=LLAMA_CONFIG, tokenizer_text=json.dumps({}))
    fetched: list[str] = []

    def _download(repo_id, filename, *, revision=None):
        fetched.append(filename)
        return str(model / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    return fetched


def test_check_tokenizer_false_fetches_only_config_json(monkeypatch, tmp_path):
    # bug caught: the keyword ignored, so a weights-only caller still fetches the tokenizer config
    fetched = _recording_download(monkeypatch, tmp_path)
    preload_check("org/model", None, allow_custom_code=False, check_tokenizer=False)
    assert fetched == ["config.json"]


def test_check_tokenizer_defaults_to_on(monkeypatch, tmp_path):
    # bug caught: the default flipped, so every measurement entry point skips the tokenizer check
    fetched = _recording_download(monkeypatch, tmp_path)
    preload_check("org/model", None, allow_custom_code=False)
    assert fetched == ["config.json", "tokenizer_config.json"]


# ── the pre-load ModelArgs build ─────────────────────────────────────────────────────────────
DSV2 = {
    "model_type": "deepseek_v2",
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rope_scaling": {"type": "yarn", "factor": 40},
}


@pytest.mark.parametrize("spec", ["4:64", "stock:8:64", "affine:8:4"])
def test_methods_other_than_turboquant_build_no_args(monkeypatch, spec):
    # bug caught: the pre-load gate importing a model module for stock and affine runs, which
    # it can never refuse
    def _fail(config):
        raise AssertionError("_preload_model_args must not run for a non-TurboQuant method")

    monkeypatch.setattr(kv_mod, "_preload_model_args", _fail)
    kv_mod._gate_layout_before_load(parse_method_spec(spec), DSV2)


def test_system_exit_in_a_model_module_import_gives_none(monkeypatch):
    # bug caught: catching Exception only, so a model module calling sys.exit ends the run
    def _exit(config):
        raise SystemExit(3)

    monkeypatch.setattr(mlx_lm.utils, "_get_classes", _exit)
    assert kv_mod._preload_model_args(DSV2) is None


def test_the_callers_config_is_not_mutated(monkeypatch):
    # bug caught: a shallow copy, so a lookup or from_dict editing a nested dict changes the
    # config the caller goes on to use
    class _Args:
        @classmethod
        def from_dict(cls, params):
            params["rope_scaling"]["factor"] = 1
            return cls()

    def _lookup(config):
        config["rope_scaling"]["type"] = "linear"
        return object, _Args

    monkeypatch.setattr(mlx_lm.utils, "_get_classes", _lookup)
    # A private copy, so a regression mutates only this test's config, never the module's DSV2.
    config = copy.deepcopy(DSV2)
    pristine = copy.deepcopy(DSV2)
    kv_mod._preload_model_args(config)
    assert config == pristine


# ── TurboQuant refusal text ──────────────────────────────────────────────────────────────────
def _refusal(layout: KVLayout, model_type: str) -> str:
    with pytest.raises(CacheNotQuantizableError) as info:
        _gate_turboquant_layout(parse_method_spec("turboquant:4"), layout, model_type=model_type)
    return str(info.value)


def test_a_model_type_with_a_newline_cannot_break_the_message():
    # bug caught: the config's model_type printed raw, so a hostile name forges message lines
    message = _refusal(KVLayout(k_dim=192, v_dim=128, modelled=True), "x\nFAKE: ok")
    assert "\n" not in message
    assert "model 'x\\nFAKE: ok' has key width=192" in message


def test_unreadable_layout_message_for_a_plain_model_type():
    # the refusal text for a normal name, byte for byte as released
    assert _refusal(KVLayout(k_dim=None, v_dim=None, modelled=False), "minicpm3") == (
        "TurboQuant-MLX kernels require a known power-of-two head_dim <= 256; model 'minicpm3' "
        "has a KV cache whose widths cannot be read from its config. This is a limitation of the "
        "port's kernels, not of the model."
    )


def test_unreadable_layout_message_with_a_newline_in_the_model_type():
    # bug caught: the second refusal message still printing the name raw
    message = _refusal(KVLayout(k_dim=None, v_dim=None, modelled=False), "x\nFAKE: ok")
    assert "\n" not in message


# ── the scheduled newest-deps job that re-runs these decisions ──────────────────────────────
NEWEST_DEPS = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "newest-deps.yml"


CHECKOUT = "actions/checkout@11d5960a326750d5838078e36cf38b85af677262"
SETUP_UV = "astral-sh/setup-uv@d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86"


def _newest_deps() -> tuple[dict, dict, list[dict]]:
    """The parsed workflow: its document, its triggers, and every step of every job."""
    import yaml

    doc = yaml.safe_load(NEWEST_DEPS.read_text())
    triggers = doc.get("on", doc.get(True))  # YAML 1.1 reads a bare `on` key as True
    steps = [step for job in doc["jobs"].values() for step in job["steps"]]
    return doc, triggers, steps


def _step_using(steps: list[dict], action: str) -> dict:
    (step,) = [s for s in steps if s.get("uses") == action]
    return step


def test_newest_deps_job_runs_weekly_and_on_demand():
    # bug caught: a job with no schedule never runs, so drift is still caught only at the lock
    _doc, triggers, _steps = _newest_deps()
    assert set(triggers) == {"schedule", "workflow_dispatch"}
    assert len(triggers["schedule"]) == 1


def test_newest_deps_job_upgrades_both_packages_and_tests_offline():
    # bug caught: the job resolving only one of the two packages, dropping a pre-load test file,
    # or testing against the Hub (the offline switch only in a comment, or on another step)
    _doc, _triggers, steps = _newest_deps()
    runs = [s.get("run", "") for s in steps]
    assert "uv lock --upgrade-package transformers --upgrade-package mlx-lm" in runs
    assert "uv sync --locked --group dev" in runs
    (pytest_step,) = [s for s in steps if "pytest" in s.get("run", "")]
    assert pytest_step["run"] == (
        "uv run pytest -q tests/probes/test_preload_tokenizer.py "
        "tests/probes/test_kv_preload_layout.py tests/probes/test_loaded_args.py"
    )
    assert pytest_step["env"] == {"HF_HUB_OFFLINE": "1"}


def test_newest_deps_checkout_keeps_no_credentials():
    # bug caught: the token left in .git/config for every later step that runs new package code
    assert _step_using(_newest_deps()[2], CHECKOUT)["with"]["persist-credentials"] is False


def test_newest_deps_does_not_share_the_uv_cache():
    # bug caught: a job that installs unreviewed newest packages saves the Actions cache that
    # the regular CI jobs restore
    assert _step_using(_newest_deps()[2], SETUP_UV)["with"]["enable-cache"] is False


def test_newest_deps_uses_only_the_two_pinned_actions():
    # bug caught: an added action (a push or release action) or an unpinned tag
    assert sorted(s["uses"] for s in _newest_deps()[2] if "uses" in s) == [CHECKOUT, SETUP_UV]


def test_newest_deps_job_never_writes_back():
    # bug caught: the upgraded lock committed or pushed from the job, or more than read access
    doc, _triggers, steps = _newest_deps()
    assert doc["permissions"] == {"contents": "read"}
    assert [name for name, job in doc["jobs"].items() if "permissions" in job] == []
    assert not [s for s in steps if "git push" in s.get("run", "") or "gh " in s.get("run", "")]


# ── follow-up: an environment without httpx; the remaining raw model_type warnings ──────────
def test_config_fetch_works_without_httpx(monkeypatch, tmp_path):
    # bug caught: an unguarded `import httpx` in the config fetch; huggingface-hub 0.x is
    # requests-based and does not install httpx, so every run there would fail at the import
    model = _model_dir(tmp_path, config=LLAMA_CONFIG)
    monkeypatch.setitem(sys.modules, "httpx", None)  # makes `import httpx` raise ImportError
    assert _preload._fetch_config(str(model), None) == (LLAMA_CONFIG, None)


def _fake_httpx2(monkeypatch) -> type[Exception]:
    """Install a stand-in ``httpx2`` (huggingface_hub 2.0's client) and return its ReadError.

    Its HTTPError is unrelated to httpx's, as in the real package, so only an except list that
    names httpx2's own base catches it.
    """

    class HTTPError(Exception):
        pass

    class ReadError(HTTPError):
        pass

    module = types.ModuleType("httpx2")
    module.HTTPError = HTTPError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx2", module)
    return ReadError


@pytest.mark.parametrize("httpx_installed", [True, False], ids=["with-httpx", "httpx2-only"])
def test_an_httpx2_transport_error_on_the_config_fetch_is_not_accessible(
    monkeypatch, httpx_installed
):
    # bug caught: only httpx's HTTPError in the except list, so on huggingface_hub 2.0 (httpx2)
    # a dropped read escapes preload_check as a raw traceback
    read_error = _fake_httpx2(monkeypatch)
    if not httpx_installed:
        monkeypatch.setitem(sys.modules, "httpx", None)

    def _download(repo_id, filename, *, revision=None):
        raise read_error("dropped read")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    with pytest.raises(ModelNotAccessibleError) as info:
        preload_check("org/model", None, allow_custom_code=False)
    assert str(info.value) == (
        "could not read config.json for 'org/model': dropped read (typo, gated repo, or offline?)"
    )


@pytest.mark.parametrize("spec", ["4:64", "affine:8:4"])
def test_unverified_head_dim_warning_cannot_be_broken_by_the_model_type(spec):
    # bug caught: the config's model_type printed raw in the "unverified" warning
    warnings = parse_method_spec(spec).check(head_dim=None, model_type="x\nFAKE: ok")
    assert warnings
    assert not [w for w in warnings if "\n" in w]
