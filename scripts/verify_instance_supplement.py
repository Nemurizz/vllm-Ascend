"""Verify per-file hashes and optionally restore a local instance supplement.

Zstandard archives require zstandard==0.23.0; gzip archives use the standard
library. Verification streams archive members and works with Windows deep paths
without extracting them. Extract to a new directory on the replacement Linux VM.
"""

import argparse
import contextlib
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@contextlib.contextmanager
def members(path):
    if path.name.endswith(".zst"):
        try:
            import zstandard
        except ImportError as exc:
            raise RuntimeError(
                "Install zstandard==0.23.0 to verify the native profiler archive"
            ) from exc
        with (
            path.open("rb") as source,
            zstandard.ZstdDecompressor().stream_reader(source) as reader,
            tarfile.open(fileobj=reader, mode="r|") as archive,
        ):
            yield archive
    else:
        with tarfile.open(path, mode="r|gz") as archive:
            yield archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    base = args.manifest.resolve().parent
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    destination = args.destination.resolve() if args.destination else None
    if destination and destination.exists():
        raise FileExistsError("Choose a new restoration directory: " + str(destination))
    # Check archive integrity and every destination before creating files.
    for entry in manifest["archives"]:
        path = (base / entry["file"]).resolve()
        if not path.is_relative_to(base):
            raise ValueError("Archive path leaves the backup directory")
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError("Archive size or SHA256 mismatch: " + entry["file"])
        for member in entry["members"]:
            name = PurePosixPath(member["path"])
            if name.is_absolute() or ".." in name.parts or "\\" in member["path"]:
                raise ValueError("Unsafe archive member path")
            if destination and not (destination / str(name)).resolve().is_relative_to(
                destination
            ):
                raise ValueError("Member path leaves the restoration directory")
        print("Archive verified:", entry["file"], flush=True)
    result = {"status": "verified", "archives": [], "extracted": bool(destination)}
    for entry in manifest["archives"]:
        expected = {item["path"]: item for item in entry["members"]}
        if len(expected) != len(entry["members"]):
            raise ValueError("Duplicate manifest member")
        seen = set()
        with members(base / entry["file"]) as archive:
            for member in archive:
                if (
                    not member.isfile()
                    or member.name in seen
                    or member.name not in expected
                    or member.size != expected[member.name]["bytes"]
                ):
                    raise ValueError("Unexpected archive member: " + member.name)
                h = hashlib.sha256()
                with archive.extractfile(member) as source:
                    target = None
                    try:
                        if destination:
                            path = destination / member.name
                            path.parent.mkdir(parents=True, exist_ok=True)
                            target = path.open("xb")
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            h.update(block)
                            if target:
                                target.write(block)
                    finally:
                        if target:
                            target.close()
                if h.hexdigest() != expected[member.name]["sha256"]:
                    raise ValueError("Member SHA256 mismatch: " + member.name)
                seen.add(member.name)
        if seen != set(expected):
            raise ValueError("Archive member set differs from the manifest")
        result["archives"].append(
            {"file": entry["file"], "members_verified": len(seen)}
        )
        print("Members verified:", entry["file"], len(seen), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("SUPPLEMENT VERIFIED", flush=True)


if __name__ == "__main__":
    main()
