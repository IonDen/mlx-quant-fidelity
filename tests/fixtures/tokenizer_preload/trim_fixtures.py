"""Rebuild the tokenizer pre-load fixtures from the original Hub files, keeping whitelisted keys.

Each fixture is a trimmed copy of one repo's ``config.json`` and ``tokenizer_config.json`` at a
pinned revision. ``_source`` records the repo, the revision and the sha256 of each original
file, so anyone can fetch the originals again and check the trim:

    hf download <repo> config.json tokenizer_config.json --revision <revision> \\
        --local-dir <download_root>/<repo with "/" replaced by "__">

    python tests/fixtures/tokenizer_preload/trim_fixtures.py <download_root> [<fixture_dir>]

The repo and revision come from the existing fixture of the same name; the script refuses a
download that `hf download` recorded at another revision, and a download missing either file,
before it rewrites the fixture. Only the keys the
tokenizer decision reads are kept: no chat templates, no vocabularies.
"""

import hashlib
import json
import sys
from pathlib import Path

CONFIG_KEYS = ("model_type", "tokenizer_class", "auto_map")
TOKENIZER_KEYS = ("tokenizer_class", "auto_map")
FILES = {"config.json": CONFIG_KEYS, "tokenizer_config.json": TOKENIZER_KEYS}


def trim(source: dict[str, str], originals: Path) -> dict[str, object]:
    """The fixture for one repo: ``_source`` with sha256 of each original, then trimmed files.

    The revision check is best-effort: it reads only ``config.json``'s download metadata, and
    only when ``hf download --local-dir`` left one; a hand-copied download is not checked. The
    sha256 values in ``_source`` let anyone confirm the originals afterwards.
    """
    metadata = originals / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if metadata.is_file():
        fetched = metadata.read_text().splitlines()[0].strip()
        if fetched != source["revision"]:
            raise SystemExit(f"{originals}: fetched {fetched}, fixture pins {source['revision']}")
    digests: dict[str, str] = {}
    fixture: dict[str, object] = {}
    for name, keys in FILES.items():
        path = originals / name
        if not path.is_file():
            raise SystemExit(f"{originals}: missing {name}; download it at {source['revision']}")
        raw = path.read_bytes()
        digests[name] = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        fixture[name] = {k: data[k] for k in keys if k in data}
    return {
        "_source": {"repo": source["repo"], "revision": source["revision"], "sha256": digests},
        **fixture,
    }


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(__doc__, file=sys.stderr)
        return 2
    root = Path(argv[1])
    fixture_dir = Path(argv[2]) if len(argv) == 3 else Path(__file__).resolve().parent
    # Every trim first, then every write: a refusal on one repo leaves all fixtures unchanged.
    trimmed = []
    for path in sorted(fixture_dir.glob("*.json")):
        source = json.loads(path.read_text())["_source"]
        trimmed.append((path, trim(source, root / path.stem)))
    for path, fixture in trimmed:
        path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
        print(path.name, json.dumps(fixture["_source"]["sha256"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
