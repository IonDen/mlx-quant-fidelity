import re

import pytest
from tests.test_cli import _fake_report, _weight_report

from mlx_quant_fidelity.badge import badge_color, badge_for_report, render_badge_markdown


def test_badge_color_table():
    assert badge_color("good") == "brightgreen"
    assert badge_color("marginal") == "yellow"
    assert badge_color("bad") == "red"


def test_badge_color_rejects_unknown():
    with pytest.raises(ValueError, match="unknown verdict"):
        badge_color("nonsense")


def test_badge_message_carries_corpus_length_mode_not_bare_number():
    fields = badge_for_report(_fake_report())  # stress, marginal, 4-bit, wikitext-2-raw/512
    msg = fields["message"]
    assert any(v in msg for v in ("good", "marginal", "bad"))
    assert not re.search(r"fidelity:\s*[0-9]", msg)
    assert "wikitext-2-raw" in msg
    assert "512" in msg
    assert "stress" in msg
    assert fields["color"] == "yellow"


def test_weight_badge_carries_provisional_caveat():
    fields = badge_for_report(_weight_report())
    assert "provisional" in fields["message"]
    assert fields["label"] == "Weight fidelity"


def test_render_badge_markdown_escapes_shields_separators():
    md = render_badge_markdown(_fake_report())
    assert md.startswith("![")
    assert "img.shields.io/badge/" in md
    assert "4--bit" in md
    assert "wikitext--2--raw" in md
