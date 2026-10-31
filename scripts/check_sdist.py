"""Fail when a built sdist carries local-only working files.

The sdist excludes in pyproject.toml are the first line; this reads the tarball that was
actually built, so a pattern hatch ignores still fails the build. Standard library only.

    python scripts/check_sdist.py dist/*.tar.gz
"""

import sys
import tarfile
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath

# Matched at any depth, like the unanchored gitignore-style excludes in pyproject.toml.
FORBIDDEN_DIRS = frozenset({".claude", ".codegraph", "_spike"})
FORBIDDEN_FILES = frozenset({"CLAUDE.md", "AGENTS.md"})
# Matched at the sdist root only, like the anchored excludes (`_artifacts/compare/`, ...).
FORBIDDEN_ROOT_PREFIXES = (
    ("_artifacts", "compare"),
    ("docs", "superpowers"),
    ("docs", "backlog"),
)


def _is_forbidden(name: str) -> bool:
    parts = PurePosixPath(name).parts[1:]  # drop the `<name>-<version>/` top directory
    if not parts:
        return False
    if any(part in FORBIDDEN_DIRS for part in parts):
        return True
    if parts[-1] in FORBIDDEN_FILES:
        return True
    return any(parts[: len(prefix)] == prefix for prefix in FORBIDDEN_ROOT_PREFIXES)


def forbidden_members(names: Iterable[str]) -> list[str]:
    """The member names, in order, that are local-only working files."""
    return [name for name in names if _is_forbidden(name)]


def main(argv: Sequence[str]) -> int:
    """Check each sdist path; 0 when all are clean, 1 on a forbidden member, 2 on bad input."""
    if not argv:
        print("usage: check_sdist.py <sdist.tar.gz>...", file=sys.stderr)
        return 2
    status = 0
    for arg in argv:
        path = Path(arg)
        if not path.is_file():
            print(f"{arg}: no such sdist (did the build run?)", file=sys.stderr)
            return 2
        with tarfile.open(path, "r:*") as tar:
            names = tar.getnames()
        bad = forbidden_members(names)
        for name in bad:
            print(f"{path.name}: local-only file in sdist: {name}", file=sys.stderr)
        if bad:
            status = 1
        else:
            print(f"{path.name}: {len(names)} members, no local-only files")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
