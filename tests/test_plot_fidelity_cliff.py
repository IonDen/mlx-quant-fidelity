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


def test_matplotlib_is_not_imported_at_module_scope():
    # render() imports matplotlib lazily so the collectors stay usable without the
    # dev-only `docs` group. This asserts the source, not the runtime: once the docs
    # group is installed, a runtime check would pass even if the import moved up.
    source = (REPO_ROOT / "scripts" / "plot_fidelity_cliff.py").read_text()
    tree = ast.parse(source)
    offenders = []
    for node in tree.body:  # module scope only, not function bodies
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
    module = _load_module()
    for bits, name in ((8, "b"), (4, "a")):
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
