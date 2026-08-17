import ast
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    """Import the script by path; it is not an installed package."""
    spec = importlib.util.spec_from_file_location(
        "plot_fidelity_cliff", REPO_ROOT / "scripts" / "plot_fidelity_cliff.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_module_scope_imports(node: ast.AST) -> list[ast.Import | ast.ImportFrom]:
    # Recurses into compound statements (If/Try/For/While/With, ...) so an import
    # nested one level down at module scope is still caught, but does not recurse
    # into FunctionDef/AsyncFunctionDef/ClassDef bodies — imports genuinely scoped
    # inside a function (e.g. render()) are not module-scope imports.
    found: list[ast.Import | ast.ImportFrom] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            found.append(child)
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        found.extend(_find_module_scope_imports(child))
    return found


def test_matplotlib_is_not_imported_at_module_scope():
    # render() imports matplotlib lazily so the collectors stay usable without the
    # dev-only `docs` group. This asserts the source, not the runtime: once the docs
    # group is installed, a runtime check would pass even if the import moved up.
    source = (REPO_ROOT / "scripts" / "plot_fidelity_cliff.py").read_text()
    tree = ast.parse(source)
    offenders = []
    for node in _find_module_scope_imports(tree):
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if a.name.startswith("matplotlib")]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("matplotlib"):
            offenders.append(node.module)
    assert offenders == [], f"matplotlib imported at module scope: {offenders}"


def test_collect_kv_points_reads_committed_samples(tmp_path):
    module = _load_module()
    sample = {
        "model_id": "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "kv_bits": 4,
        "kl": {"mean": 0.148, "median": 0.1, "p99": 0.9, "max": 5.0},
        "flip_rate": 0.2,
        "verdict": "bad",
    }
    (tmp_path / "llama-3.2-1b-4bit-kv4.json").write_text(json.dumps(sample))

    points = module.collect_kv_points(tmp_path)

    assert len(points) == 1
    assert points[0].bits == 4
    assert points[0].kl_mean == 0.148
    assert points[0].verdict == "bad"
    assert "Llama-3.2-1B" in points[0].label


def test_collect_kv_points_ignores_long_window_variants(tmp_path):
    # The cl4096 samples measure a different chunk length; mixing them into a
    # chart captioned "512-token chunks" would misattribute the numbers.
    module = _load_module()
    base = {"model_id": "m", "kv_bits": 4, "kl": {"mean": 0.1}, "flip_rate": 0.2, "verdict": "bad"}
    (tmp_path / "llama-3.2-1b-4bit-kv4.json").write_text(json.dumps(base))
    (tmp_path / "llama-3.2-1b-4bit-kv4-cl4096.json").write_text(json.dumps(base))

    points = module.collect_kv_points(tmp_path)

    assert len(points) == 1


def test_collect_kv_points_sorts_by_bits_then_label(tmp_path):
    # Filenames are alphabetically the OPPOSITE of the intended bits order (the
    # bits=4 fixture sorts last, bits=8 sorts first), so only the collector's own
    # `sorted(points, key=lambda p: (p.bits, p.label))` can produce [4, 8] — the
    # directory-glob's alphabetical file order alone would yield [8, 4].
    module = _load_module()
    for bits, name in ((8, "a"), (4, "z")):
        payload = {
            "model_id": f"mlx-community/{name}",
            "kv_bits": bits,
            "kl": {"mean": 0.01},
            "flip_rate": 0.0,
            "verdict": "good",
        }
        (tmp_path / f"{name}-kv{bits}.json").write_text(json.dumps(payload))

    points = module.collect_kv_points(tmp_path)

    assert [p.bits for p in points] == [4, 8]


def test_collect_weight_points_labels_the_reference(tmp_path):
    # The Qwen row compares 4-bit against 8-bit, not bf16. A chart that hides
    # the reference would overstate the drift as if it were versus full precision.
    module = _load_module()
    sample = {
        "quant_model_id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "quant_bits": 4,
        "reference_model_id": "mlx-community/Qwen2.5-7B-Instruct-8bit",
        "reference_bits": 8,
        "kl": {"mean": 0.109},
        "flip_rate": 0.16,
        "verdict": "marginal",
    }
    (tmp_path / "qwen2.5-7b-4bit-vs-8bit.json").write_text(json.dumps(sample))

    points = module.collect_weight_points(tmp_path)

    assert len(points) == 1
    assert "vs 8-bit" in points[0].label


def test_real_committed_samples_are_readable():
    module = _load_module()
    kv = module.collect_kv_points(REPO_ROOT / "_artifacts" / "samples")
    weights = module.collect_weight_points(REPO_ROOT / "_artifacts" / "samples" / "weights")

    assert len(kv) == 6
    assert len(weights) == 5
    assert all(p.kl_mean > 0 for p in kv + weights)

    # Spot-check one specific committed file's literal mean-KLD field end to end, so
    # a field mix-up (e.g. reading p99 instead of mean) is caught independently of
    # the synthetic fixtures above, which all use single-field payloads. The
    # expected value is read from the JSON itself, not hardcoded, so it cannot rot.
    known_path = REPO_ROOT / "_artifacts" / "samples" / "llama-3.2-1b-4bit-kv4.json"
    known_data = json.loads(known_path.read_text())
    expected_label = (
        f"{module._short_model_name(known_data['model_id'])} · {known_data['kv_bits']}-bit KV"
    )
    matching = [p for p in kv if p.label == expected_label]
    assert len(matching) == 1
    assert matching[0].kl_mean == known_data["kl"]["mean"]


def test_committed_chart_carries_no_wall_clock_date():
    # The chart's provenance claim is that re-running the script reproduces the
    # committed SVG. A rendered-at timestamp silently breaks that, so assert the
    # committed artifact is free of one. Pure file read: no matplotlib needed, so
    # this runs in the default lane without the `docs` group.
    svg = (REPO_ROOT / "docs" / "assets" / "charts" / "fidelity-cliff.svg").read_text()
    assert "<dc:date>" not in svg
