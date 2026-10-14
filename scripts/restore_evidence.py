"""Verify and recover archived benchmarks, logs and optional profiler timelines."""

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


ARCHIVE_ROOT = Path(__file__).resolve().parents[1] / "evidence/migration_20261014"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--include-profiler", action="store_true", help="Also restore large CSV/trace exports")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT,
                        help="Directory containing a manifest and its archive parts")
    args = parser.parse_args()
    archive_root = args.archive_root.resolve()
    if not args.verify_only and args.destination is None:
        parser.error("--destination is required for extraction")
    manifest = json.loads((archive_root / "manifest.json").read_text(encoding="utf-8"))
    entries = [
        item for item in manifest["archives"]
        if args.include_profiler or item["file"] == "benchmark-records.tar.gz"
    ]
    for item in entries:
        combined_hash = hashlib.sha256()
        total_bytes = 0
        for part in item["parts"]:
            path = (archive_root / part["file"]).resolve()
            if not path.is_relative_to(archive_root):
                raise RuntimeError(f"Unsafe archive part: {part['file']}")
            if path.stat().st_size != part["bytes"] or digest(path) != part["sha256"]:
                raise RuntimeError(f"Archive part failed validation: {part['file']}")
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    combined_hash.update(block)
                    total_bytes += len(block)
        if total_bytes != item["bytes"] or combined_hash.hexdigest() != item["sha256"]:
            raise RuntimeError(f"Combined archive failed validation: {item['file']}")
        print(f"Verified {item['file']}: {total_bytes} bytes", flush=True)

    if args.verify_only:
        return
    destination = args.destination.resolve()
    # Check every output before creating files; refuse to overwrite user results.
    for item in entries:
        for member in item["members"]:
            path = (destination / member["path"]).resolve()
            if not path.is_relative_to(destination) or path.exists():
                raise RuntimeError(f"Unsafe or existing extraction target: {path}")
    destination.mkdir(parents=True, exist_ok=True)
    for item in entries:
        # The temporary concatenation stays on the selected system disk.
        with tempfile.TemporaryFile(dir=destination) as combined:
            for part in item["parts"]:
                with (archive_root / part["file"]).open("rb") as stream:
                    shutil.copyfileobj(stream, combined)
            combined.seek(0)
            expected = {m["path"]: m["bytes"] for m in item["members"]}
            seen = set()
            with tarfile.open(fileobj=combined, mode="r|gz") as archive:
                for member in archive:
                    name = PurePosixPath(member.name)
                    path = (destination / member.name).resolve()
                    if (not member.isfile() or name.is_absolute() or ".." in name.parts
                            or not path.is_relative_to(destination) or member.name not in expected
                            or member.size != expected[member.name] or member.name in seen):
                        raise RuntimeError(f"Unexpected archive member: {member.name}")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, path.open("xb") as target:
                        shutil.copyfileobj(source, target)
                    seen.add(member.name)
            if seen != set(expected):
                raise RuntimeError(f"Archive members differ from manifest: {item['file']}")
        print(f"Restored {len(seen)} files from {item['file']}", flush=True)


if __name__ == "__main__":
    main()
