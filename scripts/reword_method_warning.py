"""Replace the earlier method-not-recorded wording in the committed samples with the current one.

Every `_artifacts/samples/**/*.json` whose parsed values include the earlier sentence is
rewritten with exactly that string value replaced, in the samples' own JSON formatting, and its
paired `.md` is re-rendered from the rewritten JSON. Nothing is re-measured. Standard library
and the package only.

    python scripts/reword_method_warning.py --out DIR     # write the rewritten twins under DIR
    python scripts/reword_method_warning.py --verify DIR  # twins equal originals but for the note
    python scripts/reword_method_warning.py --apply       # rewrite the samples in place
    python scripts/reword_method_warning.py --check       # exit 1 if any sample would change
"""

import argparse
import json
import sys
from pathlib import Path

from mlx_quant_fidelity.report import (
    METHOD_NOT_RECORDED_WARNING,
    comparison_report_from_dict,
    fidelity_report_from_dict,
    render_comparison_markdown,
    render_markdown,
    render_weight_markdown,
    weight_report_from_dict,
)

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "_artifacts" / "samples"
OLD = (
    "Quantization method is not recorded in the repo config — mlx-lm's DWQ, AWQ and dynamic "
    "quantizers write a quantization block that records only geometry (bits, group sizes, "
    "per-module overrides), never the recipe. This report describes the geometry of the loaded "
    "model, not the method that produced it."
)
NEW = METHOD_NOT_RECORDED_WARNING


def _dump(obj: object) -> str:
    """The samples' JSON formatting: `render_json` / `render_comparison_json` plus a newline."""
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def _replace(obj: object) -> tuple[object, int]:
    """`obj` with every string value equal to OLD replaced by NEW, and the replacement count."""
    if obj == OLD:
        return NEW, 1
    if isinstance(obj, list):
        pairs = [_replace(x) for x in obj]
        return [x for x, _ in pairs], sum(n for _, n in pairs)
    if isinstance(obj, dict):
        items = {k: _replace(v) for k, v in obj.items()}
        return {k: x for k, (x, _) in items.items()}, sum(n for _, n in items.values())
    return obj, 0


def _render_markdown(obj: dict[str, object]) -> str:
    """The paired `.md` text for a sample JSON, through the package's own renderers."""
    if "results" in obj:
        return render_comparison_markdown(comparison_report_from_dict(obj)) + "\n"
    if "quant_model_id" in obj:
        return render_weight_markdown(weight_report_from_dict(obj)) + "\n"
    return render_markdown(fidelity_report_from_dict(obj)) + "\n"


def plan(samples: Path) -> dict[Path, str]:
    """Each sample file the rewrite changes, mapped to its new text."""
    out: dict[Path, str] = {}
    for path in sorted(samples.rglob("*.json")):
        text = path.read_text()
        obj = json.loads(text)
        new_obj, n = _replace(obj)
        if n == 0:
            continue
        if _dump(obj) != text:
            sys.exit(f"refusing {path}: its formatting does not round-trip through json.dumps")
        out[path] = _dump(new_obj)
        md = path.with_suffix(".md")
        if md.exists():
            assert isinstance(new_obj, dict)
            out[md] = _render_markdown(new_obj)
    stray = [p for p in sorted(samples.rglob("*.md")) if OLD in p.read_text() and p not in out]
    if stray:
        sys.exit("earlier wording in a .md with no paired JSON: " + ", ".join(map(str, stray)))
    return out


def _forms(sentence: str) -> set[str]:
    """`sentence` raw (Markdown) and JSON-escaped (the samples' ASCII JSON); one form if equal."""
    return {sentence, json.dumps(sentence)[1:-1]}


def _strip(text: str, sentence: str) -> str:
    """`text` without any form of `sentence`."""
    for form in _forms(sentence):
        text = text.replace(form, "")
    return text


def _count(text: str, sentence: str) -> int:
    return sum(text.count(form) for form in _forms(sentence))


def write_out(out_dir: Path) -> int:
    """Write the rewritten twins under `out_dir` (outside the repo), samples untouched."""
    out_dir = out_dir.resolve()
    if out_dir.is_relative_to(REPO):
        sys.exit(f"--out must be outside the repo: {out_dir}")
    for path, text in plan(SAMPLES).items():
        twin = out_dir / path.relative_to(REPO)
        twin.parent.mkdir(parents=True, exist_ok=True)
        twin.write_text(text)
        print(f"wrote {twin}")
    return 0


def verify(out_dir: Path) -> int:
    """Each twin equals its original once both notes are removed; the counts match."""
    twins = sorted(p for p in out_dir.resolve().rglob("*") if p.is_file())
    if not twins:
        sys.exit(f"no rewritten files under {out_dir}")
    failed = 0
    for twin in twins:
        rel = twin.relative_to(out_dir.resolve())
        original, rewritten = (REPO / rel).read_text(), twin.read_text()
        n_old, n_new = _count(original, OLD), _count(rewritten, NEW)
        clean = (
            _strip(original, OLD) == _strip(rewritten, NEW)
            and n_old == n_new > 0
            and _count(original, NEW) == 0
            and _count(rewritten, OLD) == 0
        )
        failed += not clean
        rest = "rest identical" if clean else "DIFFERS beyond the note"
        print(f"{'ok  ' if clean else 'FAIL'} {rel}: {n_old} old -> {n_new} new, {rest}")
    print(f"{len(twins) - failed}/{len(twins)} files verified")
    return 1 if failed else 0


def apply() -> int:
    """Rewrite the samples in place."""
    for path, text in plan(SAMPLES).items():
        path.write_text(text)
        print(f"rewrote {path.relative_to(REPO)}")
    return 0


def check() -> int:
    """Exit 1 if any sample file would change."""
    changed = [p for p, text in plan(SAMPLES).items() if p.read_text() != text]
    for path in changed:
        print(f"would change {path.relative_to(REPO)}")
    print(f"{len(changed)} sample file(s) would change")
    return 1 if changed else 0


def main(argv: list[str] | None = None) -> int:
    """Parse one mode flag and run it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=Path, metavar="DIR")
    mode.add_argument("--verify", type=Path, metavar="DIR")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.out is not None:
        return write_out(args.out)
    if args.verify is not None:
        return verify(args.verify)
    return apply() if args.apply else check()


if __name__ == "__main__":
    raise SystemExit(main())
