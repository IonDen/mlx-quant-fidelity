from mlx_quant_fidelity.ranking import (
    RankPoint,
    budget_pick,
    dominated_by,
    dominates,
    pareto_frontier,
)

Q8 = RankPoint("q8", quality=0.01, cost_bytes=8_000)
Q6 = RankPoint("q6", quality=0.04, cost_bytes=6_200)
Q4 = RankPoint("q4", quality=0.09, cost_bytes=4_200)
Q4_BAD = RankPoint("q4-bad", quality=0.20, cost_bytes=4_300)  # dominated by q4


def test_dominates_worse_and_bigger():
    assert dominates(Q4, Q4_BAD) is True
    assert dominates(Q4_BAD, Q4) is False


def test_dominates_is_false_for_identical_axes():
    a = RankPoint("a", 0.05, 5_000)
    b = RankPoint("b", 0.05, 5_000)  # same cost, same quality -> neither dominates
    assert dominates(a, b) is False
    assert dominates(b, a) is False


def test_dominates_equal_cost_lower_quality_wins():
    # audit #4: equal cost, lower quality -> dominates
    a = RankPoint("a", 0.05, 5_000)
    b = RankPoint("b", 0.09, 5_000)
    assert dominates(a, b) is True
    assert dominates(b, a) is False


def test_pareto_frontier_excludes_dominated():
    assert pareto_frontier([Q8, Q6, Q4, Q4_BAD]) == ["q4", "q6", "q8"]  # sorted by cost


def test_pareto_frontier_keeps_equal_cost_different_quality_only_the_better():
    a = RankPoint("a", 0.05, 5_000)
    b = RankPoint("b", 0.09, 5_000)  # dominated (audit #4)
    assert pareto_frontier([a, b]) == ["a"]


def test_dominated_by_names_the_dominator():
    assert dominated_by([Q8, Q6, Q4, Q4_BAD]) == {"q4-bad": "q4"}


def test_budget_pick_cheapest_qualifying():
    pts = [Q8, Q6, Q4]
    assert budget_pick(pts, qualifying={"q6", "q8"}) == "q6"  # cheapest of the qualifiers


def test_budget_pick_none_when_nothing_qualifies():
    assert budget_pick([Q8, Q6, Q4], qualifying=set()) is None


def test_budget_pick_determinism_under_shuffle():
    import random

    pts = [Q8, Q6, Q4, Q4_BAD]
    shuffled = pts[:]
    random.Random(0).shuffle(shuffled)
    assert budget_pick(shuffled, qualifying={"q4", "q6", "q8"}) == budget_pick(
        pts, qualifying={"q4", "q6", "q8"}
    ) == "q4"


def test_naive_sort_would_fail():
    # A raw-KLD sort would put q4 (lowest cost, highest KLD) anywhere; the frontier must
    # still exclude only the truly-dominated q4-bad, keeping all three real tradeoffs.
    frontier = pareto_frontier([Q8, Q6, Q4, Q4_BAD])
    assert "q4-bad" not in frontier
    assert set(frontier) == {"q4", "q6", "q8"}
