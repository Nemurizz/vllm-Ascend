"""Restore pinned upstream sources and the exact research Git commits.

Only Git and Python's standard library are required. This script does not
install packages or touch the installed NPU environment.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    # vLLM has long kernel-config filenames; avoid Windows' legacy path limit.
    platform_flags = ["-c", "core.longpaths=true"] if os.name == "nt" else []
    result = subprocess.run(
        ["git", *platform_flags, "-C", str(repo), *args],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if check and result.returncode:
        raise RuntimeError(f"Git operation failed in {repo}: {args}\n{result.stderr}")
    return result


def restore(name: str, pin: dict, destination: Path, issue: str, source: str | None) -> dict:
    bundle = PROJECT_ROOT / pin["bundle"]
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    if digest != pin["bundle_sha256"]:
        raise RuntimeError(f"Bundle checksum mismatch: {name}")

    repo = destination / name
    if repo.exists() and not (repo / ".git").is_dir():
        if any(repo.iterdir()):
            raise RuntimeError(f"Destination is not an empty Git checkout: {repo}")
    repo.mkdir(parents=True, exist_ok=True)
    if not (repo / ".git").is_dir():
        git(repo, "init")
    if git(repo, "status", "--porcelain").stdout.strip():
        raise RuntimeError(f"Checkout has local changes; preserve them first: {repo}")

    base = pin["base"]
    if git(repo, "cat-file", "-e", f"{base}^{{commit}}", check=False).returncode:
        urls = [source] if source else [pin["upstream"], pin["mirror"]]
        for url in urls:
            print(f"Fetching {name} base {base} from {url}", flush=True)
            fetched = git(repo, "fetch", "--depth=1", "--no-tags", url, base, check=False)
            if fetched.returncode == 0:
                break
            print(fetched.stderr, flush=True)
        else:
            raise RuntimeError(f"Cannot fetch pinned base for {name}")

    base_ref = "refs/heads/research-base"
    current_base = git(repo, "rev-parse", "--verify", base_ref, check=False)
    if current_base.returncode == 0 and current_base.stdout.strip() != base:
        raise RuntimeError(f"Existing research-base does not match the manifest: {repo}")
    if current_base.returncode:
        git(repo, "update-ref", base_ref, base)

    git(repo, "bundle", "verify", str(bundle))
    # Import the exact commit objects without resetting any existing branch.
    git(repo, "bundle", "unbundle", str(bundle))
    for fixed in pin["issues"].values():
        branch_ref = f"refs/heads/{fixed['branch']}"
        existing = git(repo, "rev-parse", "--verify", branch_ref, check=False)
        if existing.returncode == 0:
            if existing.stdout.strip() != fixed["commit"]:
                raise RuntimeError(f"Existing branch differs; will not overwrite {branch_ref}")
        else:
            git(repo, "update-ref", branch_ref, fixed["commit"])

    selected = pin["issues"][issue]
    git(repo, "checkout", selected["branch"])
    actual = git(repo, "rev-parse", "HEAD").stdout.strip()
    if actual != selected["commit"]:
        raise RuntimeError(f"Restored HEAD does not match for {name}")
    git(repo, "diff", "--check", base, actual)
    if git(repo, "status", "--porcelain").stdout.strip():
        raise RuntimeError(f"Restored checkout is unexpectedly dirty: {repo}")
    print(f"Restored {name}: {actual} ({selected['branch']})", flush=True)
    return {"path": str(repo), "base": base, "head": actual, "branch": selected["branch"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--issue", choices=["16637", "15920"], default="16637")
    parser.add_argument("--source-vllm", help="Override the upstream source URL/path")
    parser.add_argument("--source-vllm-ascend", help="Override the Ascend source URL/path")
    args = parser.parse_args()
    manifest = json.loads((PROJECT_ROOT / "code/source-bundles.json").read_text(encoding="utf-8"))
    destination = args.destination.resolve()
    overrides = {"vllm": args.source_vllm, "vllm-ascend": args.source_vllm_ascend}
    results = {
        name: restore(name, pin, destination, args.issue, overrides[name])
        for name, pin in manifest["repositories"].items()
    }
    output = destination / "restore-result.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verification record: {output}")


if __name__ == "__main__":
    main()
