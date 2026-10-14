"""Acquire only licensed Phasecraft Hamiltonian JSON members using ZIP ranges.

ZIP CRC32, uncompressed sizes, JSON parsing and per-member SHA-256 are checked.
The whole-archive MD5 is advertised provenance, not verified by selective reads.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile

URL = "https://zenodo.org/records/8370650/files/towards_quantum_simulation.zip?download=1"
SIZE = 92023575
MATERIALS = {"SrVO3", "GaAs", "H3S", "Si", "Li2CuO2"}
NAMES = {"interactions.json", "map_majorana_to_mode.json", "map_mode_to_group.json",
         "map_group_to_position.json", "map_indices_to_mode.json", "stage_data.json"}


class RangeReader(io.RawIOBase):
    def __init__(self):
        self.position = 0
        self.cache = {}
        self.ranges = []

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else SIZE + offset
        if self.position < 0 or self.position > SIZE:
            raise ValueError("Seek outside pinned archive")
        return self.position

    def read(self, count=-1):
        count = SIZE - self.position if count < 0 else min(count, SIZE - self.position)
        if count > 2 * 1024 * 1024:
            raise ValueError("Unexpected large selective ZIP read")
        result = bytearray()
        while count:
            start = self.position // 65536 * 65536
            end = min(start + 65536, SIZE) - 1
            if start not in self.cache:
                request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}":
                        raise ValueError("Server returned an unexpected byte range")
                    data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise ValueError("Wrong range length")
                self.cache[start] = data
                self.ranges.append({"start": start, "end": end, "sha256": hashlib.sha256(data).hexdigest()})
            block = self.cache[start][self.position - start:self.position - start + count]
            result.extend(block)
            count -= len(block)
            self.position += len(block)
        return bytes(result)


def fetch(destination: Path):
    pinned_path = Path(__file__).with_name("phasecraft-selective-manifest.json")
    pinned = json.loads(pinned_path.read_text(encoding="utf-8")) if pinned_path.exists() else None
    expected = {item["path"]: item for item in pinned["files"]} if pinned else {}
    destination.mkdir(parents=True, exist_ok=True)
    reader = RangeReader()
    manifest = {"doi": "10.5281/zenodo.8370650", "license": "CC-BY-4.0", "url": URL,
                "archive_bytes": SIZE, "advertised_archive_md5": "33cf4fb7b6d435814112fa75ee9b19a1",
                "whole_archive_md5_verified": False, "files": []}
    with zipfile.ZipFile(reader) as archive:
        readme_info = archive.getinfo("datashare/README.md")
        if readme_info.file_size > 65536:
            raise ValueError("Unexpected archive README size")
        readme = archive.read(readme_info)
        if pinned and hashlib.sha256(readme).hexdigest() != pinned["readme"]["sha256"]:
            raise ValueError("README differs from pinned provenance")
        readme_path = destination / "README-source.md"
        if readme_path.exists() and readme_path.read_bytes() != readme:
            raise FileExistsError("Refusing to overwrite different archive README")
        readme_path.write_bytes(readme)
        manifest["readme"] = {"zip_member": readme_info.filename, "bytes": len(readme),
                              "sha256": hashlib.sha256(readme).hexdigest(),
                              "zip_crc32": f"{readme_info.CRC:08x}", "zip_crc32_verified": True}
        selected = []
        for item in archive.infolist():
            parts = PurePosixPath(item.filename).parts
            if len(parts) == 5 and parts[0:2] == ("datashare", "towards_quantum_simulation_data") and parts[2] in MATERIALS and parts[3] == "hamiltonian" and parts[4] in NAMES:
                if item.file_size > 2 * 1024 * 1024:
                    raise ValueError("Unexpected material member size")
                selected.append((item, parts[2], parts[4]))
        if len(selected) != len(MATERIALS) * len(NAMES):
            raise ValueError("Archive does not contain the expected thirty JSON members")
        for item, material, name in selected:
            data = archive.read(item)  # zipfile verifies the member's CRC32.
            if len(data) != item.file_size:
                raise ValueError("Wrong member size")
            json.loads(data)
            path = destination / material / name
            digest = hashlib.sha256(data).hexdigest()
            if expected and (expected[f"{material}/{name}"]["sha256"] != digest or expected[f"{material}/{name}"]["bytes"] != len(data)):
                raise ValueError("Material member differs from pinned provenance")
            if path.exists() and path.read_bytes() != data:
                raise FileExistsError(f"Refusing to overwrite different data: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            manifest["files"].append({"path": f"{material}/{name}", "zip_member": item.filename,
                                      "bytes": len(data), "sha256": digest, "zip_crc32": f"{item.CRC:08x}",
                                      "zip_crc32_verified": True})
            print(f"verified {material}/{name} {len(data)} {digest}", flush=True)
    manifest["ranges"] = reader.ranges
    manifest["downloaded_range_bytes"] = sum(x["end"] - x["start"] + 1 for x in reader.ranges)
    (destination / "selective-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = fetch(args.destination)
    print(f"Acquired {len(result['files'])} JSON members with {result['downloaded_range_bytes']} range bytes.")
