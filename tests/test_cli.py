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


def _hub_response(status: int):
    import httpx

    return httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/x"))


@pytest.mark.parametrize(
    "argv", [["kv", "no-such/model"], ["weights", "no-such/model", "--reference", "r"]]
)
def test_repo_not_found_is_a_user_error(monkeypatch, capsys, argv):
    """Reds if a mistyped repo id prints 'internal error' / crashes instead of exit 2."""
    from huggingface_hub.errors import RepositoryNotFoundError

    def _raise(*_a, **_k):
        raise RepositoryNotFoundError("404 Client Error", response=_hub_response(404))

    monkeypatch.setattr(cli, "measure_kv_fidelity", _raise)
    monkeypatch.setattr(cli, "measure_weight_fidelity", _raise)
    assert cli.main(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: 404 Client Error")
    assert "typo, gated repo, or offline?" in err
    assert "internal error" not in err


@pytest.mark.parametrize("exc_name", ["HFValidationError", "LocalEntryNotFoundError"])
def test_kv_hub_validation_and_offline_are_user_errors(monkeypatch, capsys, exc_name):
    import huggingface_hub.errors as hub_errors

    def _raise(*_a, **_k):
        raise getattr(hub_errors, exc_name)("bad id\nsecond line")

    monkeypatch.setattr(cli, "measure_kv_fidelity", _raise)
    assert cli.main(["kv", "bad id"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: bad id (typo")
    assert "second line" not in err  # first line only


def test_missing_local_path_is_a_user_error(monkeypatch, capsys):
    def _raise(*_a, **_k):
        raise FileNotFoundError("config.json not found")

    monkeypatch.setattr(cli, "measure_kv_fidelity", _raise)
    assert cli.main(["kv", "/no/such/dir"]) == 2
    assert capsys.readouterr().err.startswith("error: config.json not found (typo")


def test_unrelated_exception_still_propagates(monkeypatch):
    """The boundary is a whitelist: a genuine bug must not be dressed up as a user error."""

    def _raise(*_a, **_k):
        raise RuntimeError("real bug")

    monkeypatch.setattr(cli, "measure_kv_fidelity", _raise)
    with pytest.raises(RuntimeError, match="real bug"):
        cli.main(["kv", "m"])


def test_kv_cli_reports_chunk_progress_on_stderr(monkeypatch, capsys):
    """Reds if `kv` prints nothing until the report, or leaks progress onto stdout."""

    def fake(model, *, progress=None, **kw):
        assert progress is not None
        progress("chunk 1/1")
        return _fake_report()

    monkeypatch.setattr(cli, "measure_kv_fidelity", fake)
    assert cli.main(["kv", "m", "--format", "json"]) == 0
    captured = capsys.readouterr()
    assert "chunk 1/1" in captured.err
    assert json.loads(captured.out)["verdict"] == "marginal"  # stdout is the pure report


def test_weights_cli_reports_chunk_progress_on_stderr(monkeypatch, capsys):
    def fake(quant, reference, *, progress=None, **kw):
        assert progress is not None
        progress("chunk 1/1")
        return _weight_report()

    monkeypatch.setattr(cli, "measure_weight_fidelity", fake)
    assert cli.main(["weights", "q", "--reference", "r", "--format", "json"]) == 0
    captured = capsys.readouterr()
    assert "chunk 1/1" in captured.err
    json.loads(captured.out)


def _all_actions(parser):
    import argparse

    yield from parser._actions
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                yield from _all_actions(sub)


def test_every_cli_option_has_help(monkeypatch):
    """Reds if any option or positional lacks help= (a bare `--flag` in --help explains nothing)."""
    import argparse

    captured: dict[str, argparse.ArgumentParser] = {}

    def spy(self, args=None, namespace=None):
        captured.setdefault("root", self)
        raise SystemExit(0)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", spy)
    with pytest.raises(SystemExit):
        cli.main(["kv", "m"])
    missing = [
        a.option_strings or a.dest
        for a in _all_actions(captured["root"])
        if not isinstance(
            a, (argparse._HelpAction, argparse._VersionAction, argparse._SubParsersAction)
        )
        and not a.help
    ]
    assert missing == []


def test_compare_kv_menu_text_and_quant_revision_help(capsys):
    with pytest.raises(SystemExit):
        cli.main(["compare", "--help"])
    assert "rank N KV-cache methods/configs on one model" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["weights", "--help"])
    assert "without an inline @revision" in " ".join(capsys.readouterr().out.split())


def _load_must_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    import mlx_lm

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("mlx_lm.load ran for a request that should be refused up front")

    monkeypatch.setattr(mlx_lm, "load", _boom)

    # Nor may the pre-load config fetch or any Hub download: the method parameters are checked
    # model-free, before either.
    import huggingface_hub

    from mlx_quant_fidelity.probes import kv as kv_probe

    def _no_fetch(*_a: object, **_k: object) -> None:
        raise AssertionError("the Hub was contacted for a request that should be refused up front")

    monkeypatch.setattr(kv_probe, "preload_check", _no_fetch)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _no_fetch)


@pytest.mark.parametrize(
    ("flags", "rejected"),
    [
        (["--kv-group-size", "0"], "0"),
        (["--kv-group-size", "16"], "16"),
        (["--kv-bits", "5"], "5"),
    ],
)
def test_cli_kv_bad_params_exit_2_without_loading(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
    rejected: str,
) -> None:
    """Reds if bad bits / group size only surface after the config fetch or model load (or as a
    traceback), or the error does not name the rejected value."""
    _load_must_not_run(monkeypatch)
    assert cli.main(["kv", "org/never-fetched", *flags]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert rejected in err


def test_cli_kv_turboquant_unavailable_exits_2_without_loading(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    _load_must_not_run(monkeypatch)
    monkeypatch.setitem(sys.modules, "turboquant_mlx", None)
    rc = cli.main(["kv", "org/never-fetched", "--kv-method", "turboquant", "--kv-bits", "4"])
    assert rc == 2
    assert "turboquant" in capsys.readouterr().err


def test_kv_method_spec_error_does_not_mention_configs_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reds if `kv --kv-method affine:9:4` blames a `--configs` flag the command lacks."""
    assert cli.main(["kv", "m", "--kv-method", "affine:9:4"]) == 2
    err = capsys.readouterr().err
    assert "--configs" not in err
    assert "--kv-method" in err


def test_compare_configs_error_names_the_configs_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reds if the CLI drops the `--configs entry` prefix the parser no longer supplies."""
    assert cli.main(["compare", "kv", "m", "--configs", "4:64,affine:9:4"]) == 2
    assert "--configs entry" in capsys.readouterr().err


_HUB_404 = (
    "404 Client Error. (Request ID: Root=1-abc)\n\n"
    "Repository Not Found for url: https://huggingface.co/bad/ref/resolve/main/config.json.\n"
    "Please make sure you specified the correct `repo_id`."
)


def test_weights_missing_reference_names_the_reference_not_the_quant(monkeypatch, capsys, tmp_path):
    """Bug: the CLI blamed the FIRST model id ('good/q') for a missing REFERENCE repo."""
    import huggingface_hub
    from huggingface_hub.errors import RepositoryNotFoundError

    good = tmp_path / "config.json"
    good.write_text(json.dumps({"model_type": "llama", "vocab_size": 3}))

    def fake_dl(repo, filename, **kw):
        if repo == "bad/ref":
            raise RepositoryNotFoundError(_HUB_404, response=_hub_response(404))
        return str(good)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)
    assert cli.main(["weights", "good/q", "--reference", "bad/ref"]) == 2
    err = capsys.readouterr().err
    assert "bad/ref" in err
    assert "good/q" not in err
    assert "Repository Not Found for url" in err  # not just the "404 Client Error." first line
    assert "Request ID" not in err
    assert "internal error" not in err


def test_unattributable_load_failure_is_reported_neutrally(monkeypatch, capsys):
    """Bug: a failure raised inside the probe (which repo is unknown) named one model id."""
    from huggingface_hub.errors import RepositoryNotFoundError

    def _raise(*_a, **_k):
        raise RepositoryNotFoundError("404 Client Error", response=_hub_response(404))

    monkeypatch.setattr(cli, "measure_weight_fidelity", _raise)
    assert cli.main(["weights", "good/q", "--reference", "bad/ref"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: 404 Client Error (typo, gated repo, or offline?)")
    assert "good/q" not in err


@pytest.mark.parametrize(
    ("argv", "prefix"),
    [
        (["kv", "m", "--kv-group-size", "48"], "error: --kv-group-size: "),
        (["kv", "m", "--kv-method", "turboquant", "--kv-bits", "9"], "error: --kv-method: "),
        (["kv", "m", "--kv-method", "nonexistent"], "error: --kv-method: "),
    ],
)
def test_kv_flag_path_errors_name_the_offending_flag_first(capsys, argv, prefix):
    """Bug: a rejected flag value surfaces as a bare library message ('unsupported group_size=48')
    with no hint which command-line flag to change."""
    assert cli.main(argv) == 2
    assert capsys.readouterr().err.startswith(prefix)
