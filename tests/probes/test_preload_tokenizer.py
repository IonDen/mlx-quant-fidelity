"""The pre-load check refuses a tokenizer that needs remote code before any weight download.

The decision mirrors transformers' ``AutoTokenizer.from_pretrained`` with
``trust_remote_code=False`` (what mlx-lm's ``load`` passes when no opt-in was given): refuse only
when ``tokenizer_config.json`` maps ``AutoTokenizer`` to repo code and transformers has no local
tokenizer for the repo. The committed fixtures are trimmed copies of ``config.json`` and
``tokenizer_config.json`` of every Hub repo the project names, at a pinned revision.
"""

import json
import warnings
from pathlib import Path

import pytest

from mlx_quant_fidelity.errors import UntrustedModelCodeError
from mlx_quant_fidelity.probes import _preload

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "tokenizer_preload"
_REAL = sorted(FIXTURES.glob("*.json"))
_SHA = "0123456789abcdef0123456789abcdef01234567"

# Moonlight's real tokenizer_config.json: its tokenizer lives in the repo (tokenization_moonshot.py).
_REMOTE_ONLY = {
    "tokenizer_class": "TikTokenTokenizer",
    "auto_map": {"AutoTokenizer": ["tokenization_moonshot.TikTokenTokenizer", None]},
}
# A model type transformers does not know, so no config-level tokenizer applies.
_UNKNOWN = {"model_type": "moonlight_custom"}


@pytest.fixture(autouse=True)
def _pinned_device(monkeypatch):
    """Hermetic device for the probe call sites below (see test_preload)."""
    import mlx.core as mx

    from mlx_quant_fidelity.probes import kv as kv_mod

    monkeypatch.setattr(kv_mod, "_max_working_set_bytes", lambda: 26_800_603_136)
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 26_800_603_136}
    )


def _write_repo(root: Path, config: dict, tokenizer_config: object | None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config))
    if tokenizer_config is not None:
        text = (
            tokenizer_config if isinstance(tokenizer_config, str) else json.dumps(tokenizer_config)
        )
        (root / "tokenizer_config.json").write_text(text)
    return root


def _serve_hub(monkeypatch, tmp_path, config, tokenizer_config, *, sha=_SHA, fail=None):
    """Fake hf_hub_download over a real cache layout; records every (repo, filename, revision).

    A missing file raises the Hub's 404 entry error; ``fail`` is raised for tokenizer_config.json.
    """
    import httpx
    import huggingface_hub
    from huggingface_hub.errors import RemoteEntryNotFoundError

    requested: list[tuple[str, str, str | None]] = []

    def fake(repo, filename, *, revision=None, **_kw):
        requested.append((repo, filename, revision))
        snap = _write_repo(
            tmp_path / f"models--{repo.replace('/', '--')}" / "snapshots" / sha,
            config,
            tokenizer_config,
        )
        if filename == "tokenizer_config.json" and fail is not None:
            raise fail
        path = snap / filename
        if not path.exists():
            response = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co/x"))
            raise RemoteEntryNotFoundError(f"404 for {filename}", response=response)
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    return requested


_Failure = tuple[type[Exception], str]

# A failure transformers 5.12 raises only after it decided the tokenizer is local: the fixtures
# omit the vocab files, so building the chosen local tokenizer fails. Accepted for every case.
_PAST_THE_DECISION: tuple[_Failure, ...] = (
    (ValueError, "Couldn't instantiate the backend tokenizer"),
)
# A tokenizer_class transformers does not know, read once it found no remote code, leaves it
# calling ``None.from_pretrained``. Accepted only for the case that is known to reach it.
_NONE_FROM_PRETRAINED: _Failure = (
    AttributeError,
    "'NoneType' object has no attribute 'from_pretrained'",
)


def _transformers_decision(repo_dir: Path, *, also_passed: tuple[_Failure, ...] = ()) -> str:
    """The oracle: what real AutoTokenizer.from_pretrained(trust_remote_code=False) does.

    mlx-lm calls it on the downloaded directory, after importing its tokenizer_utils (which
    registers mlx-lm's own tokenizer class). ``"refused"`` is the remote-code refusal;
    ``"passed"`` is a load, the backend failure past the remote-code decision, or one of the
    case's own ``also_passed`` failures; anything else is ``"other: <error>"``, an early failure
    that says nothing about remote code.
    """
    import mlx_lm.tokenizer_utils  # noqa: F401  (registers NewlineTokenizer, as in a real load)
    from transformers import AutoTokenizer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            AutoTokenizer.from_pretrained(str(repo_dir), trust_remote_code=False)
        except Exception as exc:
            if isinstance(exc, ValueError) and "trust_remote_code=True" in str(exc):
                return "refused"
            past = _PAST_THE_DECISION + also_passed
            if any(isinstance(exc, kind) and text in str(exc) for kind, text in past):
                return "passed"
            return f"other: {type(exc).__name__}: {str(exc)[:200]}"
    return "passed"


def _load_fixture(path: Path) -> dict:
    return json.loads(path.read_text())


# --- 1. every repo the project names still passes -------------------------------------------


def test_the_fixture_set_covers_the_named_repos():
    """Bug: the fixture directory is emptied or trimmed, so the parametrized check proves nothing."""
    names = {p.stem for p in _REAL}
    assert len(names) == 28
    for must in (
        "mlx-community__Moonlight-16B-A3B-Instruct-4-bit",
        "mlx-community__Llama-3.2-1B-Instruct-4bit",
        "mlx-community__Qwen3-4B-Instruct-2507-DDWQ",
        "mlx-community__gemma-3-1b-it-4bit",
        "Qwen__Qwen3-0.6B",
    ):
        assert must in names
    for path in _REAL:
        src = _load_fixture(path)["_source"]
        assert path.stem == src["repo"].replace("/", "__")
        assert len(src["revision"]) == 40


@pytest.mark.parametrize("fixture", _REAL, ids=[p.stem for p in _REAL])
def test_every_repo_that_loads_today_passes_the_check(monkeypatch, tmp_path, fixture):
    """Bug: refusing on `auto_map` alone (Moonlight carries one, yet transformers loads its
    tokenizer locally for a deepseek_v3 config), or treating a built-in class as remote."""
    data = _load_fixture(fixture)
    repo = data["_source"]["repo"]
    requested = _serve_hub(
        monkeypatch, tmp_path, data["config.json"], data.get("tokenizer_config.json")
    )
    result = _preload.preload_check(repo, None, allow_custom_code=False)
    assert result.revision == _SHA
    assert [f for _, f, _ in requested] == ["config.json", "tokenizer_config.json"]


# --- the decision, against real transformers --------------------------------------------------

_CASES = [
    # (id, config.json, tokenizer_config.json or None when absent, refused)
    ("remote-only, unknown model type", _UNKNOWN, _REMOTE_ONLY, True),
    (
        "remote, no tokenizer_class",
        _UNKNOWN,
        {"auto_map": {"AutoTokenizer": ["t.Slow", "t.Fast"]}},
        True,
    ),
    (
        "auto_map + built-in class",
        _UNKNOWN,
        {"tokenizer_class": "Qwen2Tokenizer", "auto_map": {"AutoTokenizer": ["t.A", None]}},
        False,
    ),
    (
        "auto_map + built-in Fast class",
        _UNKNOWN,
        {"tokenizer_class": "LlamaTokenizerFast", "auto_map": {"AutoTokenizer": [None, "t.F"]}},
        False,
    ),
    (
        "legacy [slow, fast] list + built-in class",
        _UNKNOWN,
        {"tokenizer_class": "LlamaTokenizer", "auto_map": ["t.Slow", "t.Fast"]},
        False,
    ),
    (
        "legacy [slow, fast] list, no local class",
        _UNKNOWN,
        {"auto_map": ["t.Slow", "t.Fast"]},
        True,
    ),
    # transformers prefers the fast entry, falling back to the slow one only when fast is None.
    (
        "remote fast entry only, no class",
        _UNKNOWN,
        {"auto_map": {"AutoTokenizer": [None, "t.F"]}},
        True,
    ),
    ("remote class, config type with a tokenizer", {"model_type": "qwen3"}, _REMOTE_ONLY, False),
    # transformers 5.12 maps no tokenizer to the llama config type: only the class name decides.
    ("remote class, llama config", {"model_type": "llama"}, _REMOTE_ONLY, True),
    ("remote class, config type without a tokenizer", {"model_type": "resnet"}, _REMOTE_ONLY, True),
    (
        "remote class, deepseek_v3 config (Moonlight)",
        {"model_type": "deepseek_v3"},
        _REMOTE_ONLY,
        False,
    ),
    ("auto_map without AutoTokenizer", _UNKNOWN, {"auto_map": {"AutoConfig": "c.C"}}, False),
    ("no auto_map", _UNKNOWN, {"tokenizer_class": "TikTokenTokenizer"}, False),
    (
        "no tokenizer_config.json, remote AutoTokenizer in config.json",
        {**_UNKNOWN, "auto_map": {"AutoTokenizer": ["t.A", None]}},
        None,
        False,
    ),
    (
        "no tokenizer_config.json, tokenizer_class in config.json",
        {**_UNKNOWN, "tokenizer_class": "TikTokenTokenizer"},
        None,
        False,
    ),
]


# The failures past the decision that only some cases reach, by case id; every other case
# accepts only _PAST_THE_DECISION, so a new failure elsewhere shows as "other: ...".
_CASE_ALSO_PASSED: dict[str, tuple[_Failure, ...]] = {
    "no tokenizer_config.json, tokenizer_class in config.json": (_NONE_FROM_PRETRAINED,),
}


@pytest.mark.parametrize(
    ("case", "config", "tokenizer_config", "refused"),
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_the_check_refuses_exactly_where_transformers_does(
    tmp_path, case, config, tokenizer_config, refused
):
    """Bug: any drift from transformers' has_remote_code / has_local_code rule; each case also
    asks real transformers 5.12, so the expected value is upstream's, not this module's."""
    repo = _write_repo(tmp_path / "repo", config, tokenizer_config)
    decision = _transformers_decision(repo, also_passed=_CASE_ALSO_PASSED.get(case, ()))
    assert decision == ("refused" if refused else "passed")
    if refused:
        with pytest.raises(UntrustedModelCodeError, match="--allow-custom-code"):
            _preload.preload_check(str(repo), None, allow_custom_code=False)
    else:
        _preload.preload_check(str(repo), None, allow_custom_code=False)


@pytest.mark.parametrize("fixture", _REAL, ids=[p.stem for p in _REAL])
def test_transformers_itself_loads_every_named_repo_without_remote_code(tmp_path, fixture):
    """Bug: a fixture that really needs remote code sits in the 'loads today' list."""
    data = _load_fixture(fixture)
    repo = _write_repo(tmp_path / "repo", data["config.json"], data.get("tokenizer_config.json"))
    assert _transformers_decision(repo) == "passed"


# --- 2. a remote-code tokenizer is refused, unless opted in -----------------------------------


def test_remote_tokenizer_is_refused_before_any_weight_fetch(monkeypatch, tmp_path):
    """Bug: the check reads config.json only, so the refusal waits for the full download."""
    requested = _serve_hub(monkeypatch, tmp_path, _UNKNOWN, _REMOTE_ONLY)
    with pytest.raises(UntrustedModelCodeError, match="--allow-custom-code") as exc:
        _preload.preload_check("org/moon", "main", allow_custom_code=False)
    assert "org/moon" in str(exc.value)
    assert "tokenizer" in str(exc.value)
    assert requested == [
        ("org/moon", "config.json", "main"),
        ("org/moon", "tokenizer_config.json", _SHA),
    ]


def test_remote_tokenizer_passes_with_the_opt_in(monkeypatch, tmp_path):
    """Bug: allow_custom_code=True still refuses (or fetches the tokenizer config for nothing)."""
    requested = _serve_hub(monkeypatch, tmp_path, _UNKNOWN, _REMOTE_ONLY)
    result = _preload.preload_check("org/moon", None, allow_custom_code=True)
    assert result.revision == _SHA
    assert [f for _, f, _ in requested] == ["config.json"]


def test_refusal_message_is_the_load_time_message(monkeypatch, tmp_path):
    """Bug: the early refusal words the opt-in differently from the load-time one."""
    _serve_hub(monkeypatch, tmp_path, _UNKNOWN, _REMOTE_ONLY)
    with pytest.raises(UntrustedModelCodeError) as early:
        _preload.preload_check("org/moon", None, allow_custom_code=False)

    def refusing_load(*_a, **_k):
        raise ValueError("Please pass the argument `trust_remote_code=True` to allow custom code")

    with pytest.raises(UntrustedModelCodeError) as late:
        _preload.load_checked(refusing_load, "org/moon", allow_custom_code=False)
    assert str(early.value) == str(late.value)


def test_local_directory_is_checked_from_disk(monkeypatch, tmp_path):
    """Bug: a local directory's tokenizer_config.json is ignored, or sent to the Hub."""
    import huggingface_hub

    def boom(*_a, **_k):
        raise AssertionError("hub must not be called for a local directory")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    repo = _write_repo(tmp_path / "local", _UNKNOWN, _REMOTE_ONLY)
    with pytest.raises(UntrustedModelCodeError):
        _preload.preload_check(str(repo), None, allow_custom_code=False)
    assert _preload.preload_check(str(repo), None, allow_custom_code=True).revision is None


@pytest.mark.parametrize("site", ["kv", "weights", "compare-kv", "compare-weights"])
def test_every_call_site_refuses_before_any_load(monkeypatch, tmp_path, site):
    """Bug: a call site bypasses preload_check, so the load downloads the weights first."""
    import mlx_lm
    from tests.factories import make_corpus

    from mlx_quant_fidelity.probes import kv as kv_mod
    from mlx_quant_fidelity.probes import weights as w
    from mlx_quant_fidelity.runners import compare as cmp

    loads: list[str] = []
    monkeypatch.setattr(mlx_lm, "load", lambda repo, **_k: loads.append(repo))
    monkeypatch.setattr(cmp, "_load_model", lambda repo, *_a, **_k: loads.append(repo))
    for mod in (kv_mod, w, cmp):
        monkeypatch.setattr(mod, "install_memory_caps", lambda: (0, 0))
    monkeypatch.setattr(w, "_resolve_weight_bytes", lambda *a, **k: 1000)
    monkeypatch.setattr(w, "_hub_weight_bytes", lambda *a, **k: 1000)
    requested = _serve_hub(monkeypatch, tmp_path, {**_UNKNOWN, "vocab_size": 3}, _REMOTE_ONLY)
    corpus = make_corpus(n_chunks=2, chunk_length=4, tokenizer_id="org/m-bf16")
    art = tmp_path / "art"
    run = {
        "kv": lambda: kv_mod.measure_kv_fidelity(
            "org/name", corpus=corpus, allow_custom_code=False
        ),
        "weights": lambda: w.measure_weight_fidelity(
            "org/q", "org/ref", corpus=corpus, allow_custom_code=False
        ),
        "compare-kv": lambda: cmp.compare_kv_fidelity(
            "org/name", [(4, 64), (8, 64)], artifacts_dir=art
        ),
        "compare-weights": lambda: cmp.compare_weight_fidelity(
            ["org/q4", "org/q8"], "org/ref", artifacts_dir=art
        ),
    }[site]
    with pytest.raises(UntrustedModelCodeError, match="tokenizer"):
        run()
    assert loads == []
    assert {f for _, f, _ in requested} == {"config.json", "tokenizer_config.json"}


# --- 3/4. pass on anything uncertain ----------------------------------------------------------


@pytest.mark.parametrize(
    "fail",
    [
        pytest.param(OSError("connection reset"), id="connection-error"),
        pytest.param("missing", id="no tokenizer_config.json on the Hub"),
    ],
)
def test_tokenizer_config_fetch_errors_pass(monkeypatch, tmp_path, fail):
    """Bug: a failed or missing tokenizer_config.json fetch aborts the run or refuses on a guess."""
    if fail == "missing":
        _serve_hub(monkeypatch, tmp_path, _UNKNOWN, None)
    else:
        _serve_hub(monkeypatch, tmp_path, _UNKNOWN, _REMOTE_ONLY, fail=fail)
    assert _preload.preload_check("org/moon", None, allow_custom_code=False).revision == _SHA


@pytest.mark.parametrize(
    ("config", "tokenizer_config"),
    [
        pytest.param(_UNKNOWN, '{"auto_map": {"AutoTokenizer": ["t.A", nu', id="truncated JSON"),
        pytest.param(_UNKNOWN, ["not", "an", "object"], id="JSON list"),
        pytest.param(_UNKNOWN, {"auto_map": "t.A"}, id="auto_map is a string"),
        pytest.param(_UNKNOWN, {"auto_map": {"AutoTokenizer": "t.A"}}, id="AutoTokenizer string"),
        pytest.param(_UNKNOWN, {"auto_map": {"AutoTokenizer": [None, None]}}, id="both None"),
        pytest.param(_UNKNOWN, {"auto_map": ["t.A"]}, id="one-element list"),
        pytest.param(
            _UNKNOWN,
            {"tokenizer_class": 5, "auto_map": {"AutoTokenizer": ["t.A", None]}},
            id="non-string class",
        ),
        pytest.param(
            {"model_type": "internlm2"}, _REMOTE_ONLY, id="unknown type transformers skips"
        ),
    ],
)
def test_shapes_that_cannot_be_classified_pass(tmp_path, config, tokenizer_config):
    """Bug: an odd or broken tokenizer_config.json becomes a refusal; load_checked decides."""
    repo = _write_repo(tmp_path / "repo", config, tokenizer_config)
    _preload.preload_check(str(repo), None, allow_custom_code=False)


def test_the_mlx_lm_registered_classes_are_all_known():
    """Bug: mlx-lm registers a new tokenizer class and the check refuses repos that use it."""
    import mlx_lm.tokenizer_utils  # noqa: F401
    from transformers.models.auto import tokenization_auto

    registered = {
        name
        for name, cls in tokenization_auto.REGISTERED_TOKENIZER_CLASSES.items()
        if cls.__module__.startswith("mlx_lm.")
    }
    assert registered
    assert registered <= _preload._MLX_LM_TOKENIZER_CLASSES


def test_no_model_type_defers_to_the_load_although_transformers_refuses(tmp_path):
    """Bug: the deliberate defer is lost, so a config without model_type is refused on a guess.

    transformers does refuse this repo: with no model_type, AutoConfig guesses one from the
    path, which the pre-load check cannot know. The check passes on purpose and leaves the
    refusal to load_checked, so the oracle's answer here is recorded, not mirrored.
    """
    repo = _write_repo(tmp_path / "repo", {}, _REMOTE_ONLY)
    assert _transformers_decision(repo) == "refused"
    _preload.preload_check(str(repo), None, allow_custom_code=False)


def _is_mlx_lm_newline_tokenizer(obj: object) -> bool:
    """True for mlx-lm's NewlineTokenizer, as its class, its name, or a (slow, fast) pair."""
    from mlx_lm.tokenizer_utils import NewlineTokenizer

    if isinstance(obj, tuple):
        return any(_is_mlx_lm_newline_tokenizer(item) for item in obj)
    return obj is NewlineTokenizer or obj == "NewlineTokenizer"


def _unregister_mlx_lm_tokenizer(monkeypatch) -> None:
    """Undo mlx-lm's ``AutoTokenizer.register`` for this test, under either key shape.

    mlx-lm 0.31 registers with the name string as the key; 0.32 with the class itself.
    """
    from transformers.models.auto import tokenization_auto

    registered = tokenization_auto.REGISTERED_TOKENIZER_CLASSES
    monkeypatch.delitem(registered, "NewlineTokenizer", raising=False)
    extra = tokenization_auto.TOKENIZER_MAPPING._extra_content
    for key, value in list(extra.items()):
        if _is_mlx_lm_newline_tokenizer(key) or _is_mlx_lm_newline_tokenizer(value):
            monkeypatch.delitem(extra, key)


def _rekey_mlx_lm_registration_by_class(monkeypatch) -> None:
    """Re-key the installed registration the way mlx-lm 0.32 makes it: the class, not its name."""
    from mlx_lm.tokenizer_utils import NewlineTokenizer
    from transformers.models.auto import tokenization_auto

    extra = tokenization_auto.TOKENIZER_MAPPING._extra_content
    for key, value in list(extra.items()):
        if key == "NewlineTokenizer":
            monkeypatch.delitem(extra, key)
            monkeypatch.setitem(extra, NewlineTokenizer, value)
    assert NewlineTokenizer in extra


@pytest.mark.parametrize("key_shape", ["as-installed", "class-key"])
@pytest.mark.parametrize("class_name", ["NewlineTokenizer", "NewlineTokenizerFast"])
def test_an_mlx_lm_tokenizer_passes_without_transformers_knowing_it(
    monkeypatch, tmp_path, class_name, key_shape
):
    """Bug: _MLX_LM_TOKENIZER_CLASSES is emptied, or the "Fast" suffix is no longer stripped.

    mlx-lm registers NewlineTokenizer with AutoTokenizer only when mlx_lm.tokenizer_utils is
    imported, which the pre-load check never does. With that registration removed, transformers
    refuses the repo, and the check must still pass it, as a real load (which imports mlx-lm)
    would. "class-key" re-keys the registration the way mlx-lm 0.32 makes it.
    """
    import mlx_lm.tokenizer_utils  # noqa: F401

    if key_shape == "class-key":
        _rekey_mlx_lm_registration_by_class(monkeypatch)
    repo = _write_repo(
        tmp_path / "repo",
        _UNKNOWN,
        {"tokenizer_class": class_name, "auto_map": {"AutoTokenizer": ["t.A", None]}},
    )
    assert _transformers_decision(repo) == "passed"
    _unregister_mlx_lm_tokenizer(monkeypatch)
    assert _transformers_decision(repo) == "refused"
    _preload.preload_check(str(repo), None, allow_custom_code=False)


def test_a_registry_import_failure_passes(monkeypatch, tmp_path):
    """Bug: an ImportError from transformers' registries escapes preload_check (or refuses),
    when it is no proof of remote code."""
    import importlib

    real_import = importlib.import_module

    def failing_import(name, *args, **kwargs):
        if name.startswith("transformers.models.auto"):
            raise ImportError(f"no module {name}")
        return real_import(name, *args, **kwargs)

    repo = _write_repo(tmp_path / "repo", _UNKNOWN, _REMOTE_ONLY)
    monkeypatch.setattr(importlib, "import_module", failing_import)
    _preload.preload_check(str(repo), None, allow_custom_code=False)


# --- the fixtures stay re-checkable ------------------------------------------------------------

# The keys the tokenizer decision reads; anything else (a chat template, a vocabulary) is noise.
_KEPT_KEYS = {
    "_source": {"repo", "revision", "sha256"},
    "config.json": {"model_type", "tokenizer_class", "auto_map"},
    "tokenizer_config.json": {"tokenizer_class", "auto_map"},
}


@pytest.mark.parametrize("fixture", _REAL, ids=[p.stem for p in _REAL])
def test_every_fixture_holds_only_whitelisted_keys_and_a_sha256_shaped_digest_per_file(fixture):
    """Bug: a fixture keeps an untrimmed key, or loses the digest of an original file. Only the
    digest's shape is checked here; whether it matches the Hub file needs the download."""
    data = _load_fixture(fixture)
    assert set(data) <= set(_KEPT_KEYS)
    for section, keys in _KEPT_KEYS.items():
        if section in data:
            assert set(data[section]) <= keys, section
    digests = data["_source"]["sha256"]
    assert set(digests) == {"config.json", "tokenizer_config.json"} & set(data)
    for digest in digests.values():
        assert len(digest) == 64
        assert set(digest) <= set("0123456789abcdef")


def test_the_trim_script_keeps_only_whitelisted_keys_and_hashes_the_originals(tmp_path):
    """Bug: the trim script keeps a chat template, drops auto_map, or hashes the trimmed text
    instead of the original bytes."""
    import hashlib
    import subprocess
    import sys

    originals = tmp_path / "dl" / "org__m"
    originals.mkdir(parents=True)
    # Laid out as Hub files are (indented, trailing newline), so re-serialised JSON hashes apart.
    config_bytes = (
        b'{\n  "model_type": "qwen3",\n  "vocab_size": 7,\n  "auto_map": {"AutoConfig": "c.C"}\n}\n'
    )
    tok_bytes = b'{\n  "tokenizer_class": "Qwen2Tokenizer",\n  "chat_template": "{{ x }}"\n}\n'
    (originals / "config.json").write_bytes(config_bytes)
    (originals / "tokenizer_config.json").write_bytes(tok_bytes)
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    pin = {"repo": "org/m", "revision": "a" * 40}
    (fixtures / "org__m.json").write_text(json.dumps({"_source": pin}))

    subprocess.run(
        [sys.executable, str(FIXTURES / "trim_fixtures.py"), str(tmp_path / "dl"), str(fixtures)],
        check=True,
        capture_output=True,
    )
    assert json.loads((fixtures / "org__m.json").read_text()) == {
        "_source": {
            **pin,
            "sha256": {
                "config.json": hashlib.sha256(config_bytes).hexdigest(),
                "tokenizer_config.json": hashlib.sha256(tok_bytes).hexdigest(),
            },
        },
        "config.json": {"model_type": "qwen3", "auto_map": {"AutoConfig": "c.C"}},
        "tokenizer_config.json": {"tokenizer_class": "Qwen2Tokenizer"},
    }


def _run_trim_on_one_repo(tmp_path: Path, files: dict[str, bytes], fetched: str | None):
    """Run the trim script over one pinned fixture and its originals; the fixture's text before."""
    import subprocess
    import sys

    originals = tmp_path / "dl" / "org__m"
    originals.mkdir(parents=True)
    for name, raw in files.items():
        (originals / name).write_bytes(raw)
    if fetched is not None:
        metadata = originals / ".cache" / "huggingface" / "download"
        metadata.mkdir(parents=True)
        (metadata / "config.json.metadata").write_text(f"{fetched}\netag\n1.0\n")
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    fixture = fixtures / "org__m.json"
    before = json.dumps({"_source": {"repo": "org/m", "revision": "a" * 40}})
    fixture.write_text(before)
    done = subprocess.run(
        [sys.executable, str(FIXTURES / "trim_fixtures.py"), str(tmp_path / "dl"), str(fixtures)],
        check=False,
        capture_output=True,
        text=True,
    )
    return done, fixture.read_text() == before


_BOTH = {"config.json": b'{"model_type": "qwen3"}\n', "tokenizer_config.json": b"{}\n"}


def test_the_trim_script_refuses_a_download_at_another_revision(tmp_path):
    """Bug: the revision check is dropped, so a fixture is rebuilt from files of another
    revision and pins hashes its revision never had."""
    done, unchanged = _run_trim_on_one_repo(tmp_path, _BOTH, fetched="b" * 40)
    assert done.returncode == 1
    assert f"fetched {'b' * 40}, fixture pins {'a' * 40}" in done.stderr
    assert unchanged


@pytest.mark.parametrize("missing", ["config.json", "tokenizer_config.json"])
def test_the_trim_script_refuses_a_missing_original(tmp_path, missing):
    """Bug: a missing original is skipped, so the fixture silently loses that file's section."""
    files = {name: raw for name, raw in _BOTH.items() if name != missing}
    done, unchanged = _run_trim_on_one_repo(tmp_path, files, fetched="a" * 40)
    assert done.returncode == 1
    assert f"missing {missing}" in done.stderr
    assert unchanged


# --- no test reaches the real Hub ---------------------------------------------------------------


def test_the_hub_guard_exempts_exactly_the_network_and_slow_lanes(pytestconfig):
    """Bug: the guard skips an ordinary test (a live Hub call goes unnoticed), or blocks the
    opt-in lanes that are meant to reach the Hub."""
    # The conftest pytest loaded, not `import tests.conftest`: a second copy would register a
    # second exit hook and turn a red session's exit code into 0.
    conftest = Path(__file__).resolve().parent.parent / "conftest.py"
    (loaded,) = [
        plugin
        for plugin in pytestconfig.pluginmanager.get_plugins()
        if getattr(plugin, "__file__", None) and Path(plugin.__file__).resolve() == conftest
    ]
    _hub_guard_exempt = loaded._hub_guard_exempt

    assert _hub_guard_exempt(["network"]) is True
    assert _hub_guard_exempt(["slow", "parametrize"]) is True
    assert _hub_guard_exempt([]) is False
    assert _hub_guard_exempt(["parametrize", "filterwarnings"]) is False


def test_an_unstubbed_hub_download_escapes_the_code_that_catches_exception(request):
    """Bug: the guard raises an Exception (AssertionError included), which the tokenizer fetch's
    ``except Exception`` swallows, so the live call passes silently; or the guard is not
    installed at all. An invalid repo id keeps a broken guard offline."""
    recorded = request.getfixturevalue("_no_real_hub")
    with pytest.raises(BaseException, match="real Hub call in a test") as caught:
        _preload._fetch_tokenizer_config("not a valid//repo id", None)
    assert not isinstance(caught.value, Exception)
    assert len(recorded) == 1
    recorded.clear()  # this call was the point of the test; the teardown check stays quiet


# Every route to the Hub the guard must close, whatever imported the module first. The required
# seven are always checked; the optional two only when the installed transformers binds them.
_REQUIRED_HUB_CALLS = [
    ("huggingface_hub", "hf_hub_download"),
    ("huggingface_hub", "snapshot_download"),
    ("huggingface_hub.file_download", "hf_hub_download"),
    ("huggingface_hub._snapshot_download", "hf_hub_download"),
    ("huggingface_hub._snapshot_download", "snapshot_download"),
    ("huggingface_hub.repocard", "hf_hub_download"),
    ("mlx_lm.utils", "snapshot_download"),
]
_OPTIONAL_HUB_CALLS = [
    ("transformers.utils.hub", "hf_hub_download"),
    ("transformers.utils.hub", "snapshot_download"),
]
_GUARDED_HUB_CALLS = _REQUIRED_HUB_CALLS + _OPTIONAL_HUB_CALLS


@pytest.mark.parametrize(
    ("module_name", "attr"), _GUARDED_HUB_CALLS, ids=[".".join(c) for c in _GUARDED_HUB_CALLS]
)
def test_every_hub_entry_point_is_refused(request, module_name, attr):
    """Bug: a name a download goes through (a module's own import-time copy included) is left
    unpatched, so a test reaching it makes a live Hub call."""
    import importlib

    recorded = request.getfixturevalue("_no_real_hub")
    module = importlib.import_module(module_name)
    if (module_name, attr) in _OPTIONAL_HUB_CALLS and not hasattr(module, attr):
        pytest.skip(f"this transformers release does not bind {module_name}.{attr}")
    entry = getattr(module, attr)
    with pytest.raises(BaseException, match="real Hub call in a test") as caught:
        entry("not a valid//repo id", "config.json")
    assert not isinstance(caught.value, Exception)
    assert recorded == [f"{module_name}.{attr}('not a valid//repo id', 'config.json')"]
    recorded.clear()


def test_a_hub_metadata_call_is_refused(request):
    """Bug: HfApi metadata calls are not guarded, so the weight-size lookup (HfApi().model_info)
    reaches the Hub from a test that forgot to stub it."""
    import huggingface_hub

    recorded = request.getfixturevalue("_no_real_hub")
    with pytest.raises(BaseException, match="real Hub call in a test") as caught:
        huggingface_hub.HfApi().model_info("not a valid//repo id")
    assert not isinstance(caught.value, Exception)
    assert len(recorded) == 1
    assert recorded[0].startswith("huggingface_hub.HfApi.model_info(")
    recorded.clear()
