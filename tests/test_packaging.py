"""Packaging and CI-configuration guards (pure file parsing — no build, no network)."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))


def _pyproject() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_sdist_excludes_local_agent_files() -> None:
    # A local `uv build` reads .gitignore but not .git/info/exclude, so agent notes and
    # local settings would ship in the sdist unless excluded here explicitly.
    build = _pyproject()["tool"]["hatch"]["build"]["targets"]["sdist"]  # type: ignore[index]
    exclude = set(build["exclude"])
    for entry in (
        ".claude/",
        "AGENTS.md",
        "CLAUDE.md",
        "_spike/",
        "_artifacts/compare/",
        "_artifacts/spike_long_window/",
        "coverage.json",
    ):
        assert entry in exclude, entry


def test_workflows_exist() -> None:
    names = {p.name for p in WORKFLOWS}
    assert {"ci.yml", "release.yml"} <= names


def test_every_workflow_action_is_pinned_to_a_full_commit_sha() -> None:
    # A movable tag lets a retagged or compromised action run with the release job's
    # id-token / contents:write grants; a 40-hex commit SHA cannot move.
    uses = [
        (wf.name, m.group(1))
        for wf in WORKFLOWS
        for m in re.finditer(r"^\s*-?\s*uses:\s*(\S+)", wf.read_text(), re.MULTILINE)
    ]
    assert uses, "no `uses:` lines found"
    unpinned = [
        (wf, ref) for wf, ref in uses if not re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", ref)
    ]
    assert unpinned == []


def test_ci_installs_from_the_lockfile() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    syncs = re.findall(r"uv sync[^\n]*", ci)
    assert syncs, "ci.yml runs no `uv sync`"
    assert all("--locked" in s for s in syncs), syncs


def test_workflows_default_to_read_only_token() -> None:
    for wf in WORKFLOWS:
        text = wf.read_text()
        assert re.search(r"^permissions:\s*\n\s+contents:\s*read", text, re.MULTILINE), wf.name


def test_ci_tests_every_declared_python_version() -> None:
    classifiers = _pyproject()["project"]["classifiers"]  # type: ignore[index]
    declared = {
        c.rsplit(":: ", 1)[1]
        for c in classifiers
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c)
    }
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    matrix = re.search(r"python-version:\s*\[([^\]]*)\]", ci)
    assert matrix is not None, "ci.yml has no python-version matrix"
    tested = {v.strip().strip("\"'") for v in matrix.group(1).split(",")}
    assert declared == {"3.11", "3.12", "3.13"}
    assert tested == declared


def test_pyarrow_floor_excludes_cve_2023_47248() -> None:
    # pyarrow 14.0.0 and older unpickle attacker-controlled data (CVE-2023-47248); the floor
    # must start at the first fixed release, 14.0.1.
    from packaging.requirements import Requirement

    deps = _pyproject()["project"]["dependencies"]  # type: ignore[index]
    (pyarrow,) = [Requirement(d) for d in deps if Requirement(d).name == "pyarrow"]
    assert "14.0.0" not in pyarrow.specifier
    assert "14.0.1" in pyarrow.specifier
