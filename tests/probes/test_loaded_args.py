"""The post-load geometry readers on models mlx-lm really builds.

Most mlx-lm models keep their ``ModelArgs`` as ``model.args``; youtu_llm (and a few others) keep
them as ``model.config``. The readers must see the geometry either way. Each model here is built
by mlx-lm's own ``load_model`` from a config.json alone: lazy, no weights, nothing evaluated.
"""

import json
import types
from pathlib import Path

import mlx_lm.utils

from mlx_quant_fidelity.costs import KVLayout
from mlx_quant_fidelity.probes import kv as kv_mod
from mlx_quant_fidelity.runners import compare as cmp

# youtu_llm with every geometry field at its ModelArgs default (MLA: K 128 + 64, V 128),
# one layer so the lazy build stays small.
YOUTU_LLM = {"model_type": "youtu_llm", "num_hidden_layers": 1}


def _load_lazily(tmp_path: Path, config: dict) -> object:
    (tmp_path / "config.json").write_text(json.dumps(config))
    model, _config = mlx_lm.utils.load_model(tmp_path, lazy=True, strict=False)
    return model


def test_youtu_llm_keeps_its_args_as_config(tmp_path):
    # guards the premise of this file: if mlx-lm moves youtu_llm to `.args`, the tests below
    # no longer exercise the `.config` path
    model = _load_lazily(tmp_path, YOUTU_LLM)
    assert getattr(model, "args", None) is None
    assert model.config.model_type == "youtu_llm"


def test_kv_layout_reads_youtu_llm_geometry(tmp_path):
    # bug caught: the reader looking only at `model.args`, so youtu_llm's MLA widths are unseen
    model = _load_lazily(tmp_path, YOUTU_LLM)
    assert kv_mod._kv_layout(model) == KVLayout(k_dim=192, v_dim=128, modelled=True)


def test_kv_head_dim_on_youtu_llm_is_the_key_width(tmp_path):
    # bug caught: `_kv_head_dim` falling back to None (no args) instead of K = 128 + 64
    model = _load_lazily(tmp_path, YOUTU_LLM)
    assert kv_mod._kv_head_dim(model) == 192


def test_compare_kv_dims_reads_youtu_llm_geometry(tmp_path):
    # bug caught: compare's `_kv_dims` reading `model.args` for the KV head count
    model = _load_lazily(tmp_path, YOUTU_LLM)
    assert cmp._kv_dims(model) == (1, 16, 192)


def test_a_config_that_is_not_mlx_lm_args_is_ignored():
    # bug caught: the accessor taking any `.config` (a transformers config, a dict) as geometry
    config = types.SimpleNamespace(
        model_type="deepseek_v2",
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
    )
    model = types.SimpleNamespace(config=config)
    assert kv_mod._kv_layout(model) == KVLayout(k_dim=None, v_dim=None, modelled=True)
    assert kv_mod._kv_head_dim(model) is None


def test_args_win_over_config(tmp_path):
    # bug caught: reading `.config` first, so a model holding both reports the wrong geometry
    youtu = _load_lazily(tmp_path, YOUTU_LLM)
    args = types.SimpleNamespace(model_type="llama", hidden_size=2048, num_attention_heads=32)
    model = types.SimpleNamespace(args=args, config=youtu.config)
    assert kv_mod._kv_head_dim(model) == 64


# plamo2 caches K and V at `hidden_size_per_head`, which need not be hidden // heads: here
# 64 // 2 = 32 against a real width of 16. Small mamba sizes keep the lazy build tiny.
PLAMO2 = {
    "model_type": "plamo2",
    "hidden_size": 64,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "hidden_size_per_head": 16,
    "num_hidden_layers": 2,
    "intermediate_size": 64,
    "vocab_size": 32,
    "mamba_num_heads": 2,
    "mamba_d_state": 8,
    "mamba_chunk_size": 8,
}


def test_kv_head_dim_on_plamo2_is_its_per_head_width(tmp_path):
    # bug caught: `_kv_head_dim` ignoring plamo2's `hidden_size_per_head` and gating the group
    # size on hidden // heads (32) instead of the cached width (16)
    model = _load_lazily(tmp_path, PLAMO2)
    assert kv_mod._kv_head_dim(model) == 16


def test_head_dim_wins_over_hidden_size_per_head():
    # bug caught: `hidden_size_per_head` read before an explicit `head_dim`
    args = types.SimpleNamespace(
        model_type="llama",
        head_dim=128,
        hidden_size_per_head=16,
        hidden_size=2048,
        num_attention_heads=32,
    )
    assert kv_mod._kv_head_dim(types.SimpleNamespace(args=args)) == 128
