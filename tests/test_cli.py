import json

import pytest

from mlx_quant_fidelity import cli
from mlx_quant_fidelity.corpora.provenance import CorpusProvenance
from mlx_quant_fidelity.errors import QuantFidelityError
from mlx_quant_fidelity.metrics import ScalarSummary
from mlx_quant_fidelity.probes.kv_methods import (
    AffineKVMethod,
    StockKVMethod,
    TurboQuantKVMethod,
    TurboQuantVOnlyKVMethod,
)
from mlx_quant_fidelity.report import FidelityReport, WeightFidelityReport


def _fake_report() -> FidelityReport:
    return FidelityReport(
        "m",
        None,
        4,
        64,
        0,
        "stress",
        ScalarSummary(0.02, 0.01, 0.2, 1.5),
        0.03,
        10.0,
        10.4,
        0.4,
        100,
        1,
        CorpusProvenance("wikitext-2-raw", "test", "tok", 512, 512, "none", "drop", "raw", 100),
        "0.21",
        "0.31.3",
        1,
        True,
        "marginal",
        (),
    )


def test_cli_kv_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "measure_kv_fidelity", lambda *a, **k: _fake_report())
    rc = cli.main(["kv", "mlx-community/x", "--kv-bits", "4", "--format", "json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "marginal"


def test_cli_installs_caps_before_measure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "install_memory_caps", lambda: calls.append("caps") or (20, 22))
    monkeypatch.setattr(
        cli, "measure_kv_fidelity", lambda *a, **k: calls.append("measure") or _fake_report()
    )
    cli.main(["kv", "mlx-community/x"])
    assert calls == ["caps", "measure"]  # caps FIRST


def test_cli_reports_domain_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _raise(*_a: object, **_k: object) -> FidelityReport:
        raise QuantFidelityError("deployment mode not supported")

    monkeypatch.setattr(cli, "measure_kv_fidelity", _raise)
    rc = cli.main(["kv", "m"])
    assert rc == 2
    assert "deployment mode not supported" in capsys.readouterr().err


def _weight_report() -> WeightFidelityReport:
    return WeightFidelityReport(
        quant_model_id="org/m-4bit",
        quant_revision=None,
        reference_model_id="org/m-bf16",
        reference_revision=None,
        quant_bits=4,
        quant_group_size=64,
        quant_mode="affine",
        per_layer=False,
        reference_bits=None,
        kl=ScalarSummary(0.06, 0.03, 0.4, 2.0),
        flip_rate=0.03,
        perplexity_ref=10.0,
        perplexity_quant=10.6,
        perplexity_delta=0.6,
        n_positions=1000,
        n_chunks=2,
        corpus=CorpusProvenance(
            "wikitext-2-raw", "test", "org/m-bf16", 512, 512, "none", "drop", "raw", 1024
        ),
        mlx_version="0.21",
        mlx_lm_version="0.31.3",
        peak_memory_bytes=18_000_000_000,
        quant_model_bytes=4_000_000_000,
        reference_model_bytes=14_000_000_000,
        verdict="marginal",
        warnings=(),
    )


def test_weights_subcommand_dispatches_and_renders_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def fake_measure(quant: str, reference: str, **kw: object) -> WeightFidelityReport:
        captured["args"] = (quant, reference, kw)
        return _weight_report()

    monkeypatch.setattr(cli, "measure_weight_fidelity", fake_measure)
    rc = cli.main(["weights", "org/m-4bit", "--reference", "org/m-bf16", "--format", "json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["quant_model_id"] == "org/m-4bit"
    args = captured["args"]
    assert isinstance(args, tuple)
    assert args[0] == "org/m-4bit"
    assert args[1] == "org/m-bf16"
    assert captured["args"][2]["max_chunks"] is None


def test_weights_subcommand_reports_domain_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from mlx_quant_fidelity.errors import ModelMismatchError

    def boom(*a: object, **k: object) -> WeightFidelityReport:
        raise ModelMismatchError("vocab_size mismatch")

    monkeypatch.setattr(cli, "measure_weight_fidelity", boom)
    rc = cli.main(["weights", "q", "--reference", "r"])
    assert rc == 2
    assert "vocab_size mismatch" in capsys.readouterr().err


def test_weights_subcommand_forwards_max_chunks(monkeypatch, capsys):
    captured = {}

    def fake_measure(quant, reference, **kw):
        captured["kw"] = kw
        return _weight_report()

    monkeypatch.setattr(cli, "measure_weight_fidelity", fake_measure)
    rc = cli.main(["weights", "q", "--reference", "r", "--max-chunks", "3"])
    assert rc == 0
    assert captured["kw"]["max_chunks"] == 3


def test_weights_subcommand_forwards_revision_flags(monkeypatch, capsys):
    captured = {}

    def fake_measure(quant, reference, **kw):
        captured["kw"] = kw
        return _weight_report()

    monkeypatch.setattr(cli, "measure_weight_fidelity", fake_measure)
    rc = cli.main(
        ["weights", "q", "--reference", "r", "--quant-revision", "A", "--reference-revision", "B"]
    )
    assert rc == 0
    assert (captured["kw"]["quant_revision"], captured["kw"]["reference_revision"]) == ("A", "B")


def test_weights_subcommand_inline_revisions_win_over_flags(monkeypatch, capsys):
    """Reds if `repo@rev` is passed through unsplit or a flag overrides the inline pin on
    either the quant or the reference side."""
    captured = {}

    def fake_measure(quant, reference, **kw):
        captured["args"] = (quant, reference, kw)
        return _weight_report()

    monkeypatch.setattr(cli, "measure_weight_fidelity", fake_measure)
    rc = cli.main(
        [
            "weights",
            "org/q@inline",
            "--reference",
            "org/r@rinline",
            "--quant-revision",
            "flag",
            "--reference-revision",
            "rflag",
        ]
    )
    assert rc == 0
    quant, reference, kw = captured["args"]
    assert (quant, kw["quant_revision"]) == ("org/q", "inline")
    assert (reference, kw["reference_revision"]) == ("org/r", "rinline")


def test_weights_subcommand_malformed_inline_revision_is_a_usage_error(monkeypatch, capsys):
    """Reds if a malformed pin reaches the probe (a real Hub load would follow)."""

    def boom(*a, **k):
        raise AssertionError("measure must not be called")

    monkeypatch.setattr(cli, "measure_weight_fidelity", boom)
    rc = cli.main(["weights", "org/q@", "--reference", "r"])
    assert rc == 2
    assert "malformed target" in capsys.readouterr().err


def test_cli_weights_installs_caps_before_measure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "install_memory_caps", lambda: calls.append("caps") or (20, 22))
    monkeypatch.setattr(
        cli, "measure_weight_fidelity", lambda *a, **k: calls.append("measure") or _weight_report()
    )
    rc = cli.main(["weights", "q", "--reference", "r"])
    assert calls == ["caps", "measure"]
    assert rc == 0


def test_cli_kv_badge_format(monkeypatch, capsys):
    monkeypatch.setattr(cli, "measure_kv_fidelity", lambda *a, **k: _fake_report())
    rc = cli.main(["kv", "mlx-community/x", "--format", "badge"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("![")
    assert "img.shields.io/badge/" in out


def test_cli_weights_badge_format(monkeypatch, capsys):
    monkeypatch.setattr(cli, "measure_weight_fidelity", lambda *a, **k: _weight_report())
    rc = cli.main(["weights", "q", "--reference", "r", "--format", "badge"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("![")
    assert "img.shields.io/badge/" in out


def test_cli_compare_rejects_badge_format():
    with pytest.raises(SystemExit) as exc:  # argparse rejects an invalid --format choice
        cli.main(["compare", "kv", "m", "--configs", "4:64,8:64", "--format", "badge"])
    assert exc.value.code == 2  # argparse usage error exit code


# ── regression: --chunk-length CLI plumbing ───────────────────────────────────


def test_kv_cli_passes_chunk_length(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_measure(model: str, **kw: object) -> FidelityReport:
        captured["kw"] = kw
        return _fake_report()

    monkeypatch.setattr(cli, "measure_kv_fidelity", fake_measure)
    rc = cli.main(["kv", "m", "--chunk-length", "1024"])
    assert rc == 0
    assert captured["kw"]["chunk_length"] == 1024


def test_kv_cli_chunk_length_defaults_to_512(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_measure(model: str, **kw: object) -> FidelityReport:
        captured["kw"] = kw
        return _fake_report()

    monkeypatch.setattr(cli, "measure_kv_fidelity", fake_measure)
    rc = cli.main(["kv", "m"])
    assert rc == 0
    assert captured["kw"]["chunk_length"] == 512


def _fake_comparison_report() -> object:
    from mlx_quant_fidelity.report import ComparisonReport

    return ComparisonReport(
        mode="kv",
        reference=None,
        model="m",
        corpus=None,
        quantize_start=0,
        quantize_mode="stress",
        budget=None,
        results=(),
        frontier=(),
        dominated=(),
        budget_pick=None,
        mlx_version="0.21",
        mlx_lm_version="0.31.3",
    )


def test_compare_kv_cli_passes_chunk_length(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_compare(model: str, configs: object, **kw: object) -> object:
        captured["kw"] = kw
        return _fake_comparison_report()

    monkeypatch.setattr(cli, "compare_kv_fidelity", fake_compare)
    rc = cli.main(
        [
            "compare",
            "kv",
            "m",
            "--configs",
            "4:64,8:64",
            "--chunk-length",
            "1024",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    assert captured["kw"]["chunk_length"] == 1024


def test_kv_cli_defaults_to_stock_method(monkeypatch):
    seen = {}

    def fake_measure(model, *, method, **kw):
        seen["method"] = method
        return _fake_report()

    monkeypatch.setattr(cli, "measure_kv_fidelity", fake_measure)
    assert cli.main(["kv", "org/m"]) == 0
    assert seen["method"] == StockKVMethod(bits=4, group_size=64)


def test_kv_cli_turboquant_method_and_seed(monkeypatch):
    seen = {}

    def fake_measure(model, *, method, **kw):
        seen["method"] = method
        return _fake_report()

    monkeypatch.setattr(cli, "measure_kv_fidelity", fake_measure)
    assert (
        cli.main(["kv", "org/m", "--kv-method", "turboquant", "--kv-bits", "3", "--kv-seed", "7"])
        == 0
    )
    assert seen["method"] == TurboQuantKVMethod(bits=3, seed=7)


def test_kv_cli_rejects_group_size_with_turboquant(capsys):
    assert cli.main(["kv", "org/m", "--kv-method", "turboquant", "--kv-group-size", "64"]) == 2
    assert "--kv-group-size" in capsys.readouterr().err


def test_kv_cli_rejects_seed_with_stock(capsys):
    assert cli.main(["kv", "org/m", "--kv-seed", "7"]) == 2
    assert "--kv-seed" in capsys.readouterr().err


def test_kv_cli_rejects_non_positive_seed(monkeypatch, capsys):
    def _unexpected(*_a, **_k):
        raise AssertionError("measure_kv_fidelity must not be reached")

    monkeypatch.setattr(cli, "measure_kv_fidelity", _unexpected)
    rc = cli.main(["kv", "org/m", "--kv-method", "turboquant", "--kv-seed", "0"])
    assert rc == 2
    assert "seed" in capsys.readouterr().err


def test_parse_kv_configs_accepts_method_specs():
    assert cli._parse_kv_configs("4:64,turboquant:3") == [
        StockKVMethod(bits=4, group_size=64),
        TurboQuantKVMethod(bits=3),
    ]


# ── regression: spec-string --kv-method, --control, --model-revision ──────────


def test_kv_cli_kv_method_spec_string_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    """RED if '--kv-method' does not route a colon-bearing value through parse_method_spec.

    'affine' has no flag-based (name) form (see the affine-by-name test below), so a spec
    string is the only way to reach it from the CLI -- this proves that path is wired.
    """
    seen = {}

    def fake_measure(model, *, method, **kw):
        seen["method"] = method
        return _fake_report()

    monkeypatch.setattr(cli, "measure_kv_fidelity", fake_measure)
    assert cli.main(["kv", "org/m", "--kv-method", "affine:8:4"]) == 0
    assert seen["method"] == AffineKVMethod(k_bits=8, v_bits=4)


def test_kv_cli_spec_string_conflicts_with_kv_bits(capsys: pytest.CaptureFixture[str]) -> None:
    """RED if a spec-string --kv-method silently ignores a simultaneous --kv-bits instead of
    erroring -- the two forms must not both apply to the same invocation.
    """
    rc = cli.main(["kv", "org/m", "--kv-method", "turboquant:3", "--kv-bits", "4"])
    assert rc == 2
    assert "--kv-bits" in capsys.readouterr().err


def test_kv_cli_spec_string_conflicts_with_kv_group_size(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """RED if a spec-string --kv-method silently ignores a simultaneous --kv-group-size."""
    rc = cli.main(["kv", "org/m", "--kv-method", "4:64", "--kv-group-size", "32"])
    assert rc == 2
    assert "--kv-group-size" in capsys.readouterr().err


def test_kv_cli_spec_string_conflicts_with_kv_seed(capsys: pytest.CaptureFixture[str]) -> None:
    """RED if a spec-string --kv-method silently ignores a simultaneous --kv-seed."""
    rc = cli.main(["kv", "org/m", "--kv-method", "turboquant:3", "--kv-seed", "9"])
    assert rc == 2
    assert "--kv-seed" in capsys.readouterr().err


def test_kv_cli_affine_by_name_errors_with_spec_hint(capsys: pytest.CaptureFixture[str]) -> None:
    """RED if bare '--kv-method affine' silently builds a method instead of erroring: affine
    has no flag-based form and the error must point at the spec grammar (e.g. 'affine:8:4').
    """
    rc = cli.main(["kv", "org/m", "--kv-method", "affine"])
    assert rc == 2
    assert "affine:8:4" in capsys.readouterr().err


def test_kv_cli_turboquant_vonly_by_name_uses_kv_bits_as_v_bits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED if the turboquant-vonly name path ignores --kv-bits or misreads it as k_bits."""
    seen = {}

    def fake_measure(model, *, method, **kw):
        seen["method"] = method
        return _fake_report()

    monkeypatch.setattr(cli, "measure_kv_fidelity", fake_measure)
    rc = cli.main(["kv", "org/m", "--kv-method", "turboquant-vonly", "--kv-bits", "3"])
    assert rc == 0
    assert seen["method"] == TurboQuantVOnlyKVMethod(v_bits=3)


def test_kv_cli_turboquant_vonly_by_name_requires_kv_bits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """RED if turboquant-vonly's name path defaults v_bits instead of requiring --kv-bits."""
    rc = cli.main(["kv", "org/m", "--kv-method", "turboquant-vonly"])
    assert rc == 2
    assert "--kv-bits" in capsys.readouterr().err


def test_kv_cli_turboquant_vonly_by_name_rejects_kv_group_size(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """RED if the turboquant-vonly name path silently accepts --kv-group-size instead of
    erroring -- turboquant-vonly has no group_size parameter (it takes v_bits + seed), so
    a --kv-group-size passed alongside it must be rejected the same way --kv-group-size is
    rejected for the plain turboquant name form.
    """
    rc = cli.main(
        [
            "kv",
            "org/m",
            "--kv-method",
            "turboquant-vonly",
            "--kv-bits",
            "3",
            "--kv-group-size",
            "32",
        ]
    )
    assert rc == 2
    assert "--kv-group-size" in capsys.readouterr().err


def test_kv_cli_unknown_method_name_lists_known_methods(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """RED if an unrecognized --kv-method name is silently accepted or the error omits the
    known-methods list a user needs to fix the invocation (choices= was dropped, so this
    validation must now happen in _resolve_kv_method itself).
    """
    rc = cli.main(["kv", "org/m", "--kv-method", "nonexistent"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "stock" in err
    assert "turboquant" in err


def test_kv_defaults_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """RED if moving --kv-bits/--kv-group-size to default=None leaks None into StockKVMethod
    instead of applying the 0.5.x legacy defaults (4:64) when the user passes neither flag.
    """
    seen = {}

    def fake_measure(model, *, method, **kw):
        seen["method"] = method
        return _fake_report()

    monkeypatch.setattr(cli, "measure_kv_fidelity", fake_measure)
    assert cli.main(["kv", "org/m"]) == 0
    assert seen["method"] == StockKVMethod(bits=4, group_size=64)


def test_kv_cli_control_flag_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """RED if --control is not threaded through main() into measure_kv_fidelity's control kwarg."""
    captured: dict[str, object] = {}

    def fake_measure(model, **kw):
        captured["kw"] = kw
        return _fake_report()

    monkeypatch.setattr(cli, "measure_kv_fidelity", fake_measure)
    assert cli.main(["kv", "org/m", "--control"]) == 0
    assert captured["kw"]["control"] is True


def test_kv_cli_control_flag_defaults_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """RED if omitting --control does not forward control=False (e.g. a stray default=True)."""
    captured: dict[str, object] = {}

    def fake_measure(model, **kw):
        captured["kw"] = kw
        return _fake_report()

    monkeypatch.setattr(cli, "measure_kv_fidelity", fake_measure)
    assert cli.main(["kv", "org/m"]) == 0
    assert captured["kw"]["control"] is False


def test_kv_cli_model_revision_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """RED if `kv` has no --model-revision flag, or it is not forwarded to measure_kv_fidelity."""
    captured: dict[str, object] = {}

    def fake_measure(model, **kw):
        captured["kw"] = kw
        return _fake_report()

    monkeypatch.setattr(cli, "measure_kv_fidelity", fake_measure)
    assert cli.main(["kv", "org/m", "--model-revision", "abc123"]) == 0
    assert captured["kw"]["model_revision"] == "abc123"
