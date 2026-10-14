"""Download official ModelScope files by per-file revision and verify SHA256.

Use a saved repo/files API response as --manifest, and place weights on the data disk.
The manifest fixes each file's revision, size and digest; it is not a model-wide tag.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
from urllib.parse import quote

import requests


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    files = [f for f in manifest["Data"]["Files"] if f["Type"] == "blob" and
             (f["Path"].endswith((".safetensors", ".json", ".jinja", ".txt")) or
              f["Path"] in ("LICENSE", "README.md"))]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    for file in files:
        name = PurePosixPath(file["Path"])
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"Unsafe path: {name}")
        target = args.output / str(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == file["Size"] and digest(target) == file["Sha256"]:
            print("Verified cached", name, flush=True)
            continue
        url = f"https://modelscope.cn/models/{args.model}/resolve/{file['Revision']}/{quote(str(name))}"
        part = target.with_name(target.name + ".part")
        for attempt in range(5):
            try:
                offset = part.stat().st_size if part.exists() else 0
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with requests.get(url, headers=headers, timeout=(30, 120), stream=True) as response:
                    response.raise_for_status()
                    mode = "ab" if offset and response.status_code == 206 else "wb"
                    with part.open(mode) as stream:
                        for chunk in response.iter_content(8 * 1024 * 1024):
                            stream.write(chunk)
                if part.stat().st_size != file["Size"] or digest(part) != file["Sha256"]:
                    part.unlink()
                    raise ValueError(f"Size/digest mismatch: {name}")
                part.replace(target)
                print("Verified downloaded", name, file["Size"], flush=True)
                break
            except Exception as error:
                print("Retry", name, attempt + 1, repr(error), flush=True)
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 16))
    print("ALL FILES VERIFIED", flush=True)


if __name__ == "__main__":
    main()
