import dataclasses

from tests.test_compare_report import _wreport  # reuse the WeightFidelityReport builder

from mlx_quant_fidelity.ranking import RankPoint
from mlx_quant_fidelity.report import ComparisonTargetResult
from mlx_quant_fidelity.runners.compare import assemble_comparison_report


def _ok(label, kl_mean, cost, verdict="good"):
    rep = _wreport(label, kl_mean, cost)
    rep = dataclasses.replace(rep, verdict=verdict)  # frozen+slots: replace via dataclasses.replace
    return ComparisonTargetResult(
        label, "ok", rep, RankPoint(label, kl_mean, cost), None, None, None
    )


def _failed(label):
    return ComparisonTargetResult(label, "failed", None, None, None, "ModelMismatchError", "bad")


def _assemble(results, **kw):
    defaults: dict[str, object] = {
        "mode": "weight",
        "reference": "ref",
        "model": None,
        "corpus": None,
        "quantize_start": None,
        "quantize_mode": None,
        "max_kld": None,
        "min_tier": None,
        "mlx_version": "0.21",
        "mlx_lm_version": "0.31.3",
    }
    defaults.update(kw)
    return assemble_comparison_report(results, **defaults)


def test_frontier_and_dominated_computed():
    r = _assemble(
        [
            _ok("q8", 0.01, 8000),
            _ok("q6", 0.04, 6200),
            _ok("q4", 0.09, 4200),
            _ok("q4-bad", 0.20, 4300),
        ]
    )
    assert set(r.frontier) == {"q4", "q6", "q8"}
    assert dict(r.dominated) == {"q4-bad": "q4"}


def test_failed_target_excluded_not_aborting():
    r = _assemble([_ok("q8", 0.01, 8000), _failed("q2")])
    labels = {res.label for res in r.results}
    assert labels == {"q8", "q2"}  # failed target still present
    assert "q2" not in r.frontier


def test_budget_pick_min_tier():
    r = _assemble(
        [_ok("q8", 0.01, 8000, "good"), _ok("q4", 0.09, 4200, "marginal")], min_tier="good"
    )
    assert r.budget_pick == "q8"  # q4 is marginal -> excluded from the budget
    assert r.budget == "--min-tier good"


def test_budget_none_when_no_constraint():
    r = _assemble([_ok("q8", 0.01, 8000)])
    assert r.budget is None
    assert r.budget_pick is None


def test_budget_pick_max_kld():
    # Covers the max_kld branch of _budget_label; q8 clears 0.05, q4 does not.
    r = _assemble([_ok("q8", 0.01, 8000, "good"), _ok("q4", 0.09, 4200, "good")], max_kld=0.05)
    assert r.budget_pick == "q8"
    assert r.budget == "--max-kld 0.05"


def _unrankable(label: str, kl_mean: float) -> ComparisonTargetResult:
    """status=ok, report present, point=None (cost unavailable) — not rankable."""
    return ComparisonTargetResult(
        label, "ok", _wreport(label, kl_mean, 0), None, "cost unavailable", None, None
    )


def test_unrankable_kept_but_excluded_from_frontier_and_pick():
    # q-nobytes' tiny kl_mean (0.001) would clear max_kld, but point=None must keep it
    # out of the frontier and out of any recommendation (audit #3).
    r = _assemble([_ok("q8", 0.04, 8000), _unrankable("q-nobytes", 0.001)], max_kld=0.05)
    assert {res.label for res in r.results} == {"q8", "q-nobytes"}  # kept in results
    assert "q-nobytes" not in r.frontier
    assert r.budget_pick == "q8"


def test_budget_pick_none_when_nothing_qualifies():
    r = _assemble([_ok("q4", 0.09, 4200, "marginal")], min_tier="good")
    assert r.budget == "--min-tier good"
    assert r.budget_pick is None


def test_budget_pick_both_constraints_anded():
    # q4 clears max_kld (0.04 <= 0.05) but its marginal verdict fails min_tier good -> excluded.
    r = _assemble(
        [_ok("q8", 0.01, 8000, "good"), _ok("q4", 0.04, 4200, "marginal")],
        max_kld=0.05,
        min_tier="good",
    )
    assert r.budget == "--max-kld 0.05 --min-tier good"
    assert r.budget_pick == "q8"
