import math

import mlx.core as mx

from mlx_quant_fidelity.metrics import perplexity, perplexity_delta, token_nll


def _logits(prob_rows):
    return mx.log(mx.array(prob_rows, dtype=mx.float32))


def test_token_nll_matches_neg_log_prob():
    # target index 0; p(target) = [0.5, 0.25, 0.8]; nll = -ln(p)
    logits = _logits([[0.5, 0.5], [0.25, 0.75], [0.8, 0.2]])
    nll = token_nll(logits, mx.array([0, 0, 0]))
    expected = [-math.log(0.5), -math.log(0.25), -math.log(0.8)]
    for got, exp in zip([float(x) for x in nll], expected, strict=True):
        assert math.isclose(got, exp, abs_tol=1e-5)


def test_token_nll_gathers_the_per_position_target():
    # Mixed targets [0, 1, 0] exercise the take_along_axis gather: a SUT hardcoded to index 0
    # (e.g. `log_probs[:, 0]`) mis-scores position 1 (-ln 0.75 vs -ln 0.25) and goes red.
    logits = _logits([[0.5, 0.5], [0.25, 0.75], [0.8, 0.2]])
    nll = token_nll(logits, mx.array([0, 1, 0]))
    expected = [-math.log(0.5), -math.log(0.75), -math.log(0.8)]
    for got, exp in zip([float(x) for x in nll], expected, strict=True):
        assert math.isclose(got, exp, abs_tol=1e-5)


def test_perplexity_is_exp_mean_nll():
    # mean nll = (ln2 + ln4 + ln1.25)/3 -> ppl = exp(0.767528) = 2.15443
    logits = _logits([[0.5, 0.5], [0.25, 0.75], [0.8, 0.2]])
    nll = token_nll(logits, mx.array([0, 0, 0]))
    assert math.isclose(perplexity(nll), 2.15443, abs_tol=1e-3)


def test_perplexity_delta_positive_when_quant_more_spread():
    # ref p(target)=0.5 -> ppl 2.0 ; quant p(target)=0.25 -> ppl 4.0 ; delta = +2.0
    ref = token_nll(_logits([[0.5, 0.5]] * 3), mx.array([0, 0, 0]))
    quant = token_nll(_logits([[0.25, 0.75]] * 3), mx.array([0, 0, 0]))
    assert math.isclose(perplexity_delta(ref, quant), 2.0, abs_tol=1e-3)
