"""Fetch pinned public source files from manifest.json with hash verification.

Datasets belong outside the source checkout. No extraction is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def fetch(destination: Path, include: set[str]) -> None:
    manifest = json.loads(Path(__file__).with_name("manifest.json").read_text(encoding="utf-8"))
    available = {entry["dataset"] for entry in manifest["files"]}
    if missing := include - available:
        raise ValueError(f"Dataset not yet pinned in verified manifest: {sorted(missing)}")
    for entry in manifest["files"]:
        if entry["dataset"] not in include:
            continue
        target = destination / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".part")
            request = urllib.request.Request(entry["url"], headers={"User-Agent": "commutator-rs-data-fetch/1"})
            sha256 = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(request, timeout=90) as source, temporary.open("wb") as output:
                while block := source.read(1024 * 1024):
                    size += len(block)
                    if size > entry["bytes"]:
                        raise ValueError(f"Source larger than manifest: {target}")
                    sha256.update(block)
                    output.write(block)
            if size != entry["bytes"] or sha256.hexdigest() != entry["sha256"]:
                raise ValueError(f"Source size/hash mismatch; partial file retained: {temporary}")
            temporary.replace(target)
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if target.stat().st_size != entry["bytes"] or digest != entry["sha256"]:
            raise ValueError(f"Existing file does not match manifest: {target}")
        print(f"verified {entry['path']} {entry['bytes']} bytes {digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--dataset", action="append", choices=["te", "wte2", "phasecraft"], required=True)
    args = parser.parse_args()
    fetch(args.destination, set(args.dataset))
