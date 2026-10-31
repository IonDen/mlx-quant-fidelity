"""Offline unit test for the MLA latent-cache spike script's frame path.

``scripts/spike_mla_latent_cache.py`` writes the innermost traceback frame of a failing
condition into a committed sample. A machine's absolute path (home folder, workspace, virtual
environment) must not reach that file. No model, no network: importing the script only pulls in
stdlib, mlx and package modules.
"""

from pathlib import Path

import scripts.spike_mla_latent_cache as spike

ROOT = Path("/home/someone/work/repo")


def test_site_packages_path_keeps_only_the_part_after_site_packages():
    # The virtual environment sits inside the repo: site-packages must win over repo-relative.
    venv_file = "/home/someone/work/repo/.venv/lib/python3.12/site-packages/mlx_lm/models/ds.py"
    assert spike._portable_path(venv_file, repo_root=ROOT) == "mlx_lm/models/ds.py"


def test_nested_site_packages_cuts_at_the_last_one():
    nested = "/opt/site-packages/vendored/site-packages/pkg/mod.py"
    assert spike._portable_path(nested, repo_root=ROOT) == "pkg/mod.py"


def test_repo_file_is_relative_to_the_repo_root():
    repo_file = "/home/someone/work/repo/scripts/spike.py"
    assert spike._portable_path(repo_file, repo_root=ROOT) == "scripts/spike.py"


def test_sibling_folder_sharing_the_root_prefix_is_not_inside_the_repo():
    sibling = "/home/someone/work/repo-other/tool.py"
    assert spike._portable_path(sibling, repo_root=ROOT) == "tool.py"


def test_file_outside_the_repo_and_site_packages_keeps_its_base_name():
    outside = "/usr/lib/python3.12/json/decoder.py"
    assert spike._portable_path(outside, repo_root=ROOT) == "decoder.py"


def _boom() -> None:
    raise AttributeError("'tuple' object has no attribute 'swapaxes'")


def test_innermost_frame_of_a_repo_file_carries_no_absolute_prefix():
    try:
        _boom()
    except AttributeError as exc:
        frame = spike._innermost_frame(exc)
    assert frame is not None
    path, _, rest = frame.partition(":")
    assert path == "tests/test_spike_mla_latent_cache.py"
    assert rest.endswith(" in _boom")


def test_innermost_frame_of_an_exception_never_raised_is_none():
    assert spike._innermost_frame(AttributeError("never raised")) is None
