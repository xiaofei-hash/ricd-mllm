"""Verify the frozen experiment sources against the manifest."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "llava_amber"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads((RESULT_DIR / "manifest.json").read_text("utf-8"))
    expected = dict(manifest["source_sha256"])

    failures = []
    for relative_path, expected_hash in expected.items():
        path = ROOT / relative_path
        actual_hash = sha256(path)
        matches = actual_hash == expected_hash
        print(f"{'OK' if matches else 'MISMATCH'}  {relative_path}")
        if not matches:
            failures.append((relative_path, expected_hash, actual_hash))

    if failures:
        for relative_path, expected_hash, actual_hash in failures:
            print(f"  expected {expected_hash}\n  actual   {actual_hash}\n  file     {relative_path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
