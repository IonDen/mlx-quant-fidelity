"""The TurboQuant layout gate, run from config.json before any download or load.

The pre-load decision must equal the post-load one. The post-load side here is a real model:
mlx-lm's own ``load_model`` builds it from the same config.json in a temporary directory (lazy,
no weights, nothing evaluated), and the unchanged post-load gate judges that model.
"""

import json
from pathlib import Path

import mlx.core as mx
import mlx_lm
import mlx_lm.utils
import pytest
from tests.factories import make_corpus, make_fid_report
from tests.probes.fake_turboquant import install_fake_port

from mlx_quant_fidelity.errors import CacheNotQuantizableError
from mlx_quant_fidelity.probes import kv as kv_mod
from mlx_quant_fidelity.probes._preload import PreloadResult
from mlx_quant_fidelity.probes.kv_methods import _gate_turboquant_layout, parse_method_spec
from mlx_quant_fidelity.runners import compare as cmp

# DeepSeek-V2-Lite's attention geometry: cached K = 128 + 64 = 192 (not a power of two), V = 128.
# One layer and few experts: the gate reads attention widths only, and the build stays small.
DSV2_LITE = {
    "model_type": "deepseek_v2",
    "vocab_size": 102400,
    "hidden_size": 2048,
    "intermediate_size": 10944,
    "moe_intermediate_size": 1408,
    "num_hidden_layers": 1,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "n_shared_experts": 1,
    "n_routed_experts": 2,
    "num_experts_per_tok": 1,
    "first_k_dense_replace": 1,
    "kv_lora_rank": 512,
    "q_lora_rank": None,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 163840,
    "rope_theta": 10000,
    "rope_scaling": {
        "type": "yarn",
        "factor": 40,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 0.707,
        "mscale_all_dim": 0.707,
    },
}
DSV2_MISSING_ROPE = {k: v for k, v in DSV2_LITE.items() if k != "qk_rope_head_dim"}

LLAMA = {
    "model_type": "llama",
    "hidden_size": 2048,
    "num_hidden_layers": 1,
    "intermediate_size": 8192,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-5,
    "vocab_size": 128256,
    "rope_theta": 500000.0,
    "rope_scaling": None,
}
LLAMA_STRAY_MLA = {
    **LLAMA,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
}
QWEN2 = {
    "model_type": "qwen2",
    "hidden_size": 1536,
    "num_hidden_layers": 1,
    "intermediate_size": 8960,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "vocab_size": 151936,
    "rope_theta": 1000000.0,
}
QWEN3 = {
    "model_type": "qwen3",
    "hidden_size": 2048,
    "num_hidden_layers": 1,
    "intermediate_size": 6144,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-6,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
    "rope_theta": 1000000.0,
    "head_dim": 128,
    "tie_word_embeddings": True,
}
GEMMA3_TEXT = {
    "model_type": "gemma3_text",
    "hidden_size": 1152,
    "head_dim": 256,
    "num_hidden_layers": 1,
}
DSV3 = {
    **DSV2_LITE,
    "model_type": "deepseek_v3",
    "q_lora_rank": 1536,
    "n_group": 1,
    "topk_group": 1,
}
KIMI_K2 = {**DSV3, "model_type": "kimi_k2"}  # mlx-lm remaps kimi_k2 to deepseek_v3
MINICPM3 = {
    "model_type": "minicpm3",
    "hidden_size": 2560,
    "dim_model_base": 256,
    "num_hidden_layers": 1,
    "intermediate_size": 6400,
    "num_attention_heads": 40,
    "rms_norm_eps": 1e-5,
    "vocab_size": 73448,
    "num_key_value_heads": 40,
    "q_lora_rank": 768,
    "qk_nope_head_dim": 64,
    "qk_rope_head_dim": 32,
    "kv_lora_rank": 256,
    "scale_depth": 1.4,
    "scale_emb": 12,
    "max_position_embeddings": 32768,
    "rope_scaling": {
        "type": "longrope",
        "long_factor": 1.0,
        "short_factor": 1.0,
        "original_max_position_embeddings": 32768,
    },
}
# Every geometry field at its ModelArgs default; mlx-lm keeps these args as `model.config`.
YOUTU_LLM = {"model_type": "youtu_llm", "num_hidden_layers": 1}

TABLE = {
    "llama": LLAMA,
    "llama_stray_mla": LLAMA_STRAY_MLA,
    "qwen2": QWEN2,
    "qwen3": QWEN3,
    "gemma3_text": GEMMA3_TEXT,
    "deepseek_v2_full": DSV2_LITE,
    "deepseek_v2_missing_rope": DSV2_MISSING_ROPE,
    "deepseek_v3": DSV3,
    "kimi_k2": KIMI_K2,
    "minicpm3": MINICPM3,
    "youtu_llm": YOUTU_LLM,
}
METHODS = ("turboquant:4", "turboquant-vonly:4")


def _load(directory: Path, config: dict) -> object:
    """mlx-lm's real model for ``config``: lazy, no weights, nothing evaluated."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    model, _config = mlx_lm.utils.load_model(directory, lazy=True, strict=False)
    return model


def _decide(gate) -> str | None:
    """None when the gate lets the method through, else the refusal message."""
    try:
        gate()
    except CacheNotQuantizableError as exc:
        return str(exc)
    return None


def _post_load(spec: str, config: dict, directory: Path) -> str | None:
    """The post-load gate on the model mlx-lm's loader builds from ``config``."""
    model = _load(directory, config)
    model_type = str(getattr(kv_mod._loaded_args(model), "model_type", "unknown"))
    method = parse_method_spec(spec)
    return _decide(
        lambda: _gate_turboquant_layout(method, kv_mod._kv_layout(model), model_type=model_type)
    )


def _pre_load(spec: str, config: dict) -> str | None:
    method = parse_method_spec(spec)
    return _decide(lambda: kv_mod._gate_layout_before_load(method, config))


# ── The equivalence table: pre-load decision == post-load decision on a real model ──────────
@pytest.mark.parametrize("name", sorted(TABLE))
def test_every_table_config_builds_through_mlx_lm(name, tmp_path):
    # guards the oracle: a row mlx-lm cannot build would compare against nothing real
    model = _load(tmp_path, TABLE[name])
    assert kv_mod._loaded_args(model) is not None


@pytest.mark.parametrize("spec", METHODS)
@pytest.mark.parametrize("name", sorted(TABLE))
def test_pre_load_decision_equals_post_load_decision(name, spec, tmp_path):
    # bug caught: the pre-load side reading config.json with its own field reader (e.g. the
    # compare cost reader) instead of mlx-lm's ModelArgs.from_dict, so a stray MLA field on a
    # llama config refuses before the load while the loaded model passes
    assert _pre_load(spec, TABLE[name]) == _post_load(spec, TABLE[name], tmp_path)


def test_table_holds_both_refusals_and_passes(tmp_path):
    # guards the table above against going vacuous (all-pass or all-refuse)
    decisions = {
        (n, s): _post_load(s, c, tmp_path / f"{n}-{s}") for n, c in TABLE.items() for s in METHODS
    }
    assert decisions[("deepseek_v2_full", "turboquant:4")] is not None
    assert decisions[("deepseek_v2_full", "turboquant-vonly:4")] is None
    assert decisions[("llama_stray_mla", "turboquant:4")] is None
    assert decisions[("youtu_llm", "turboquant:4")] is not None
    assert decisions[("youtu_llm", "turboquant-vonly:4")] is None


def test_building_the_table_allocates_little(tmp_path):
    # bug caught: an eager build (lazy=False, or an mx.eval of the parameters) allocating the
    # weights of every row. Measured: 248 bytes of new peak for the whole table; the smallest
    # weight matrix in it (512 x 2048 fp32) is 4 MiB, an embedding is hundreds of MiB.
    active_before = mx.get_active_memory()
    mx.reset_peak_memory()
    for name, config in TABLE.items():
        _load(tmp_path / name, config)
    assert mx.get_peak_memory() - active_before < 1024**2


# ── Single pre-load decisions ───────────────────────────────────────────────────────────────
def test_k192_refuses_with_the_key_width_in_the_message():
    # bug caught: the pre-load side gating on V only (or not gating at all)
    msg = _pre_load("turboquant:4", DSV2_LITE)
    assert msg is not None
    assert "model 'deepseek_v2' has key width=192" in msg


def test_missing_rope_field_decides_as_at_load(tmp_path):
    # bug caught: a raw-config reader calling the layout unreadable, while mlx-lm's default
    # (qk_rope_head_dim=64) gives the loaded model K 192 and a different refusal message
    assert _pre_load("turboquant:4", DSV2_MISSING_ROPE) == _post_load(
        "turboquant:4", DSV2_MISSING_ROPE, tmp_path
    )


def test_llama_with_stray_kv_lora_rank_passes():
    # bug caught: the stray MLA field read from the raw dict marks the layout unmodelled
    assert _pre_load("turboquant:4", LLAMA_STRAY_MLA) is None


def test_unknown_model_type_passes():
    # bug caught: mlx-lm's "Model type ... not supported" ValueError escaping as a refusal
    assert _pre_load("turboquant:4", {**DSV2_LITE, "model_type": "not_a_real_arch"}) is None


def test_config_without_model_type_passes():
    # bug caught: KeyError from the lookup escaping before the load
    assert _pre_load("turboquant:4", {"hidden_size": 192}) is None


def test_lookup_that_raises_passes(monkeypatch):
    # bug caught: catching only ValueError, so any other lookup failure aborts the run
    def _boom(config):
        raise RuntimeError("lookup broke")

    monkeypatch.setattr(mlx_lm.utils, "_get_classes", _boom)
    assert _pre_load("turboquant:4", DSV2_LITE) is None


def test_renamed_private_lookup_passes(monkeypatch):
    # bug caught: an mlx-lm release renaming _get_classes turns every TurboQuant run into a crash
    monkeypatch.delattr(mlx_lm.utils, "_get_classes")
    assert _pre_load("turboquant:4", DSV2_LITE) is None


def test_from_dict_that_raises_passes(monkeypatch):
    # bug caught: a from_dict error (e.g. a __post_init__ check) escaping as a refusal
    class _Args:
        @classmethod
        def from_dict(cls, params):
            raise TypeError("bad config")

    monkeypatch.setattr(mlx_lm.utils, "_get_classes", lambda config: (object, _Args))
    assert _pre_load("turboquant:4", DSV2_LITE) is None


def test_layout_reader_that_raises_passes(monkeypatch):
    # bug caught: an args object the layout reader chokes on aborting the run before the load
    def _boom(model):
        raise TypeError("odd args")

    monkeypatch.setattr(kv_mod, "_kv_layout", _boom)
    assert _pre_load("turboquant:4", DSV2_LITE) is None


def test_custom_model_file_is_left_to_the_load():
    # bug caught: judging a repo that ships its own model code by mlx-lm's built-in class
    assert _pre_load("turboquant:4", {**DSV2_LITE, "model_file": "modeling.py"}) is None


def test_stock_method_is_never_refused():
    # bug caught: the gate's TurboQuant-only check dropped, refusing stock on an MLA layout
    assert _pre_load("4:64", DSV2_LITE) is None


# ── measure_kv_fidelity: the refusal comes before the caps and the load ─────────────────────
class _LoadReachedError(Exception):
    """Raised by the fake load so a test can see the run reached it."""


def _forbid(what):
    def _fail(*_a, **_k):
        raise AssertionError(f"{what} must not run")

    return _fail


def _measure_with_config(monkeypatch, config, spec, *, load, caps):
    install_fake_port(monkeypatch)
    monkeypatch.setattr(kv_mod, "preload_check", lambda *a, **k: PreloadResult(config, None))
    monkeypatch.setattr(kv_mod, "install_memory_caps", caps)
    monkeypatch.setattr(mlx_lm, "load", load)
    return kv_mod.measure_kv_fidelity(
        "deepseek-ai/DeepSeek-V2-Lite", method=parse_method_spec(spec), corpus=make_corpus()
    )


def _post_load_score_message(spec: str, config: dict, directory: Path) -> str:
    """What the real post-load path (score_kv_config) raises on mlx-lm's model for ``config``."""
    model = _load(directory, config)
    with pytest.raises(CacheNotQuantizableError) as info:
        kv_mod.score_kv_config(model, make_corpus(), model_id="m", method=parse_method_spec(spec))
    return str(info.value)


def test_score_kv_config_refuses_youtu_llm_on_its_key_width(tmp_path):
    # bug caught: score_kv_config reading `model.args` only, so a loaded youtu_llm (args kept as
    # `model.config`) is called unreadable instead of refused on K = 128 + 64
    message = _post_load_score_message("turboquant:4", YOUTU_LLM, tmp_path)
    assert message == (
        "TurboQuant-MLX kernels require a known power-of-two head_dim <= 256; model 'youtu_llm' "
        "has key width=192. This is a limitation of the port's kernels, not of the model."
    )


def test_measure_refuses_k192_turboquant_before_caps_and_load(monkeypatch, tmp_path):
    # bug caught: the refusal still coming from score_kv_config, i.e. after the weight download
    with pytest.raises(CacheNotQuantizableError) as info:
        _measure_with_config(
            monkeypatch,
            DSV2_LITE,
            "turboquant:4",
            load=_forbid("load"),
            caps=_forbid("install_memory_caps"),
        )
    assert str(info.value) == _post_load_score_message("turboquant:4", DSV2_LITE, tmp_path)


def test_measure_vonly_on_k192_reaches_the_load(monkeypatch):
    # bug caught: the pre-load gate refusing V-only on the K width V-only never quantizes
    caps_calls: list[str] = []

    def _load(*_a, **_k):
        raise _LoadReachedError

    with pytest.raises(_LoadReachedError):
        _measure_with_config(
            monkeypatch,
            DSV2_LITE,
            "turboquant-vonly:4",
            load=_load,
            caps=lambda: caps_calls.append("caps") or (0, 0),
        )
    assert caps_calls == ["caps"]


def test_measure_unknown_model_type_reaches_the_load(monkeypatch):
    # bug caught: a pre-load lookup failure refusing instead of deferring to the load
    def _load(*_a, **_k):
        raise _LoadReachedError

    with pytest.raises(_LoadReachedError):
        _measure_with_config(
            monkeypatch,
            {**DSV2_LITE, "model_type": "not_a_real_arch"},
            "turboquant:4",
            load=_load,
            caps=lambda: (0, 0),
        )


# ── compare kv: a refused row reads the same before and after the load ─────────────────────
def _patch_compare(monkeypatch, config, *, load_model):
    install_fake_port(monkeypatch)
    monkeypatch.setattr(cmp, "install_memory_caps", lambda: (0, 0))
    monkeypatch.setattr(
        cmp,
        "preload_check",
        lambda model, revision, *, allow_custom_code: PreloadResult(config, None),
    )
    monkeypatch.setattr(cmp, "_load_model", load_model)
    monkeypatch.setattr(cmp, "_kv_dims", lambda model: (27, 16, 192))
    monkeypatch.setattr(
        cmp,
        "_load_corpus_for_kv",
        lambda tokenizer, model_id, max_chunks, *, chunk_length: make_corpus(),
    )
    scored: list[str] = []
    real_score = kv_mod.score_kv_config

    def fake_score(model, corpus, *, method, **kw):
        scored.append(method.label)
        if method.label.startswith("turboquant"):
            return real_score(model, corpus, method=method, **kw)  # the real post-load gate
        return make_fid_report()

    monkeypatch.setattr(cmp, "score_kv_config", fake_score)
    return scored


def _dsv2_loader(directory: Path):
    def _load_model(model_id, revision, **_):
        return _load(directory, DSV2_LITE), object()

    return _load_model


def test_compare_refused_row_equals_the_post_load_row(monkeypatch, tmp_path):
    # bug caught: the pre-load refusal writing a different envelope (or none) than the post-load
    # except branch, so the report and resume see a different row
    specs = ["turboquant:4", "4:64"]
    # Post-load reference: an empty pre-load config defers everything to the loaded model.
    _patch_compare(monkeypatch, {}, load_model=_dsv2_loader(tmp_path / "model"))
    ref = cmp.compare_kv_fidelity(
        "m", [parse_method_spec(s) for s in specs], artifacts_dir=tmp_path / "post"
    )
    scored = _patch_compare(monkeypatch, DSV2_LITE, load_model=_dsv2_loader(tmp_path / "model"))
    report = cmp.compare_kv_fidelity(
        "m", [parse_method_spec(s) for s in specs], artifacts_dir=tmp_path / "pre"
    )
    assert scored == ["4:64"]  # the refused method is not scored again after the load
    rows = {r.label: r for r in report.results}
    ref_rows = {r.label: r for r in ref.results}
    assert rows["turboquant:4"].error_type == ref_rows["turboquant:4"].error_type
    assert rows["turboquant:4"].message == ref_rows["turboquant:4"].message
    assert rows["turboquant:4"].error_type == "CacheNotQuantizableError"
    assert "key width=192" in rows["turboquant:4"].message
    assert rows["4:64"].report is not None
    name = cmp._kv_partial_filename(parse_method_spec("turboquant:4"))
    assert json.loads((tmp_path / "pre" / name).read_text()) == json.loads(
        (tmp_path / "post" / name).read_text()
    )


def test_compare_never_loads_when_every_method_is_refused(monkeypatch, tmp_path):
    # bug caught: the model still downloaded and loaded when nothing is left to score
    _patch_compare(monkeypatch, DSV2_LITE, load_model=_forbid("_load_model"))
    report = cmp.compare_kv_fidelity(
        "m",
        [parse_method_spec("turboquant:4"), parse_method_spec("turboquant:3")],
        artifacts_dir=tmp_path,
    )
    assert {r.label: r.error_type for r in report.results} == {
        "turboquant:4": "CacheNotQuantizableError",
        "turboquant:3": "CacheNotQuantizableError",
    }
