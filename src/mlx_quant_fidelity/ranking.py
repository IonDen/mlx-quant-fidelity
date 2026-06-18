"""Pure Pareto/ranking core for `compare` — both axes minimized (quality, cost_bytes).

The heart of method ranking. It knows nothing about KLD, models, or bytes-as-disk: adapters
map domain metrics onto `RankPoint` where lower is better on each axis. The single comparison
rule lives in `dominates`; everything else defers to it (DRY). Generalizing to vector quality
later changes only `dominates` (Open/Closed). See docs/ranking-principles.md.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RankPoint:
    """One ranked target. Lower `quality` (mean KLD) and lower `cost_bytes` are both better."""

    label: str
    quality: float
    cost_bytes: int


def dominates(a: RankPoint, b: RankPoint) -> bool:
    """True if `a` Pareto-dominates `b`: no worse on either axis, strictly better on one.

    See docs/ranking-principles.md (Pareto efficiency).

    >>> dominates(RankPoint("q4", 0.09, 4200), RankPoint("q4-bad", 0.20, 4300))
    True
    >>> dominates(RankPoint("q4-bad", 0.20, 4300), RankPoint("q4", 0.09, 4200))
    False
    """
    no_worse = a.cost_bytes <= b.cost_bytes and a.quality <= b.quality
    strictly_better = a.cost_bytes < b.cost_bytes or a.quality < b.quality
    return no_worse and strictly_better


def _sort_key(p: RankPoint) -> tuple[int, float, str]:
    return (p.cost_bytes, p.quality, p.label)


def pareto_frontier(points: list[RankPoint]) -> list[str]:
    """Labels of the non-dominated points, deterministically ordered by cost, quality, label."""
    ordered = sorted(points, key=_sort_key)
    return [p.label for p in ordered if not any(dominates(o, p) for o in points)]


def dominated_by(points: list[RankPoint]) -> dict[str, str]:
    """Map each dominated label -> a dominator's label (the cheapest/best, deterministically)."""
    result: dict[str, str] = {}
    for p in points:
        dominators = sorted((o for o in points if dominates(o, p)), key=_sort_key)
        if dominators:
            result[p.label] = dominators[0].label
    return result


def budget_pick(points: list[RankPoint], *, qualifying: set[str]) -> str | None:
    """Cheapest frontier point whose label is in `qualifying`; None if none qualify.

    The contract is the cheapest *frontier* point that qualifies — a qualifying label that
    is off the frontier is never returned. For a `--max-kld` budget this is also the cheapest
    qualifying point overall (a dominated qualifier's dominator has <= mean KLD, so it clears
    the same threshold). For a `--min-tier` budget that need not hold: tier qualification uses
    the full verdict (mean, p99, flip) while domination is on mean KLD alone, so a dominated
    point can qualify while its frontier dominator does not — then the result is None.
    See docs/ranking-principles.md.
    """
    frontier = set(pareto_frontier(points))
    eligible = [p for p in points if p.label in frontier and p.label in qualifying]
    if not eligible:
        return None
    return min(eligible, key=_sort_key).label
