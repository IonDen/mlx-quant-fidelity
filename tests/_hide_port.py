"""Mask the git-pinned turboquant_mlx port so the default suite runs the way CI does."""

import importlib.metadata
from collections.abc import Callable, MutableMapping

PORT_MODULE = "turboquant_mlx"
HIDDEN_DISTRIBUTIONS = frozenset({"turboquant-mlx", "turboquant_mlx"})


def _canonical(name: str) -> str:
    return name.lower().replace("_", "-")


def mask_port(modules: MutableMapping[str, object], name: str = PORT_MODULE) -> None:
    """Evict `name` and its submodules, then park `None` so `import name` raises
    ModuleNotFoundError (a subclass of the ImportError the adapters catch)."""
    for key in [k for k in modules if k == name or k.startswith(name + ".")]:
        del modules[key]
    modules[name] = None


def hiding_distribution(
    real: Callable[[str], object], hidden: frozenset[str] = HIDDEN_DISTRIBUTIONS
) -> Callable[[str], object]:
    """Wrap importlib.metadata.distribution so the hidden names look uninstalled.
    `version()` resolves `distribution` from the same module globals, so one wrapper covers both."""
    hidden_canonical = {_canonical(h) for h in hidden}

    def _distribution(name: str) -> object:
        if _canonical(name) in hidden_canonical:
            raise importlib.metadata.PackageNotFoundError(name)
        return real(name)

    return _distribution


def flag_conflict(*, hide_port: bool, run_slow: bool) -> str | None:
    """`--hide-port` is a default-lane flag; the slow lane needs the real port."""
    if hide_port and run_slow:
        return "--hide-port cannot be combined with --run-slow: the slow lane needs the real port"
    return None
