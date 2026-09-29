import doctest

from mlx_quant_fidelity import ranking


def test_ranking_doctests_run_and_pass():
    # bug caught: the worked examples in ranking.py rotting unnoticed (pytest never collected
    # them). `attempted > 0` proves the examples were actually found, not vacuously passed.
    result = doctest.testmod(ranking)
    assert result.attempted > 0
    assert result.failed == 0
