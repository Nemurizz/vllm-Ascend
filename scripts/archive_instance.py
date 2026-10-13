"""Archive benchmark records and selected profiler exports before replacing a VM."""

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path


def split_archive(target: Path, limit: int = 40 * 1024 * 1024) -> list[dict]:
    """Produce uploadable pieces without changing the archive's contents."""
    if target.stat().st_size <= limit:
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        return [{"file": target.name, "bytes": target.stat().st_size, "sha256": digest}]
    parts = []
    with target.open("rb") as source:
        for index in range(1000):
            block = source.read(limit)
            if not block:
                break
            part = target.with_name(f"{target.name}.part{index:03d}")
            part.write_bytes(block)
            parts.append({"file": part.name, "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
        else:
            raise RuntimeError("Archive exceeds 1000 parts")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    records = [
        p for folder in [root / "results", root / "logs"]
        for p in sorted(folder.rglob("*"))
        if p.is_file() and not p.is_symlink()
    ]
    # Never archive authentication tokens or PEM private keys from textual logs.
    secret = re.compile(rb"github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]+|-----BEGIN .*PRIVATE KEY-----")
    for path in records:
        if secret.search(path.read_bytes()):
            raise RuntimeError(f"Sensitive record requires redaction: {path.relative_to(root)}")

    groups = {"benchmark-records": records}
    for name in ["runtime_graph_probe1", "runtime_eager_probe_eager"]:
        folder = root / "profiles" / name
        exports = [
            p for p in sorted(folder.rglob("*"))
            if p.is_file() and not p.is_symlink()
            and p.parent.name == "ASCEND_PROFILER_OUTPUT"
            and (p.suffix == ".csv" or p.name == "trace_view.json")
        ]
        if exports:
            groups[name] = exports

    manifest = {"format_version": 1, "archive_kind": "selected_exports", "archives": []}
    for name, files in groups.items():
        target = output / f"{name}.tar.gz"
        print(f"Archiving {name}: {len(files)} files", flush=True)
        with tarfile.open(target, "w:gz", compresslevel=6) as archive:
            for path in files:
                archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        item = {
            "file": target.name,
            "bytes": target.stat().st_size,
            "sha256": digest.hexdigest(),
            "parts": split_archive(target),
            "members": [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size} for p in files],
        }
        manifest["archives"].append(item)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {target.name}: {item['bytes']} bytes", flush=True)


if __name__ == "__main__":
    main()
