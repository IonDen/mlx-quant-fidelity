"""Packaging and CI-configuration guards (pure file parsing — no build, no network)."""

import re
import tomllib
from pathlib import Path

import pytest

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


# --- the built-sdist check (scripts/check_sdist.py, run on the real tarball in CI) ----------


def _check_sdist():  # type: ignore[no-untyped-def]
    """Import the script by path; it is not an installed package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_sdist", ROOT / "scripts" / "check_sdist.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tarball(path: Path, names: list[str]) -> Path:
    """A gzip tarball whose members carry exactly ``names`` (empty files)."""
    import io
    import tarfile

    with tarfile.open(path, "w:gz") as tar:
        for name in names:
            tar.addfile(tarfile.TarInfo(name), io.BytesIO(b""))
    return path


_CLEAN = [
    "pkg-0.0/PKG-INFO",
    "pkg-0.0/pyproject.toml",
    "pkg-0.0/src/pkg/__init__.py",
    "pkg-0.0/_artifacts/samples/kv.json",
    "pkg-0.0/docs/measurement-principles.md",
]


def test_check_sdist_fails_a_tarball_carrying_local_agent_settings(tmp_path: Path) -> None:
    """Bug: hatch ignores an exclude pattern and `.claude/settings.local.json` ships, yet the
    check exits 0."""
    tarball = _tarball(
        tmp_path / "pkg-0.0.tar.gz", [*_CLEAN, "pkg-0.0/.claude/settings.local.json"]
    )
    assert _check_sdist().main([str(tarball)]) != 0


def test_check_sdist_fails_a_tarball_carrying_working_state_docs(tmp_path: Path) -> None:
    """Bug: hatch ignores the `docs/backlog/` exclude and the backlog ships, yet the check
    exits 0."""
    tarball = _tarball(tmp_path / "pkg-0.0.tar.gz", [*_CLEAN, "pkg-0.0/docs/backlog/item.md"])
    assert _check_sdist().main([str(tarball)]) != 0


def test_check_sdist_passes_a_clean_tarball(tmp_path: Path) -> None:
    """Bug: the check flags ordinary sdist content (committed samples, docs), so CI is red on
    every PR and the step gets switched off."""
    assert _check_sdist().main([str(_tarball(tmp_path / "pkg-0.0.tar.gz", _CLEAN))]) == 0


@pytest.mark.parametrize(
    "member",
    [
        "pkg-0.0/.claude/settings.local.json",
        "pkg-0.0/CLAUDE.md",
        "pkg-0.0/AGENTS.md",
        "pkg-0.0/_spike/notes.md",
        "pkg-0.0/_artifacts/compare/kv/partial.json",
        "pkg-0.0/src/pkg/CLAUDE.md",
        "pkg-0.0/docs/superpowers/plans/plan.md",
        "pkg-0.0/docs/backlog/item.md",
        "pkg-0.0/.codegraph/codegraph.db",
        "pkg-0.0/src/pkg/.codegraph/codegraph.db",
    ],
)
def test_forbidden_members_names_each_local_only_path(member: str) -> None:
    """Bug: one of the excluded patterns is missing from the check (or is matched only at the
    sdist root), so that file could ship unnoticed."""
    assert _check_sdist().forbidden_members([*_CLEAN, member]) == [member]


@pytest.mark.parametrize(
    "member",
    [
        "pkg-0.0/_artifacts/compare.md",
        "pkg-0.0/docs/_artifacts/compare/x.json",
        "pkg-0.0/src/pkg/claude_md.py",
        "pkg-0.0/src/docs/backlog/x.md",
        "pkg-0.0/docs/backlog.md",
    ],
)
def test_forbidden_members_ignores_near_miss_names(member: str) -> None:
    """Bug: the match is a substring test, so a file that only resembles an excluded path (or
    `_artifacts/compare/` below the root, which the anchored exclude does not cover) is
    flagged."""
    assert _check_sdist().forbidden_members([*_CLEAN, member]) == []


def test_check_sdist_fails_when_no_tarball_is_found(tmp_path: Path) -> None:
    """Bug: an unmatched `dist/*.tar.gz` glob reaches the script as a literal path (or as no
    argument) and the check passes without reading any sdist."""
    module = _check_sdist()
    assert module.main([str(tmp_path / "dist" / "*.tar.gz")]) != 0
    assert module.main([]) != 0


def test_ci_build_job_runs_the_sdist_check_on_the_built_tarball() -> None:
    """Bug: the check exists but the CI build job never runs it, or runs it before the build."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    build = ci.split("\n  build:\n", 1)[1]
    build_step = build.index("hynek/build-and-inspect-python-package@")
    check_step = build.index("scripts/check_sdist.py")
    assert check_step > build_step
    assert re.search(r"scripts/check_sdist\.py \S*\*\.tar\.gz", build), "not run on the tarball"
