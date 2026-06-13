from conftest import GATED_MARKERS, _markers_to_skip  # pytest prepend mode adds tests/ to sys.path


def test_all_markers_skipped_when_no_flags():
    skipped = dict(_markers_to_skip(enabled_flags=set()))
    assert set(skipped) == {marker for marker, _flag, _description in GATED_MARKERS}


def test_enabled_flag_unskips_its_marker():
    skipped = dict(_markers_to_skip(enabled_flags={"--run-slow"}))
    assert "slow" not in skipped
    assert "network" in skipped
