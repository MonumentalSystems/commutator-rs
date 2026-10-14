"""Losslessly split a validated Wannier90 _tb.dat into _hr.dat and _r.dat.

No matrix values, indices, degeneracies or spin/orbital ordering are changed.
Centres come from the origin diagonal of the supplied position matrix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile

from inspect_wannier import inspect, nonempty, real


def validate_win(text: str, result: dict) -> list[str]:
    lines = [re.split(r"[!#]", line, maxsplit=1)[0].strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    def block(name):
        starts = [i for i, line in enumerate(lines) if line.lower() == "begin " + name]
        ends = [i for i, line in enumerate(lines) if line.lower() == "end " + name]
        if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
            raise ValueError(f"Expected one complete {name} block in .win")
        values = lines[starts[0] + 1:ends[0]]
        if values and values[0].lower() in ("ang", "angstrom", "bohr"):
            if values[0].lower() == "bohr":
                raise ValueError("Bohr .win blocks unsupported; convert explicitly before splitting")
            values = values[1:]
        return values

    lattice = [[real(x) for x in line.split()] for line in block("unit_cell_cart")]
    if len(lattice) != 3 or any(len(row) != 3 for row in lattice):
        raise ValueError("Invalid .win lattice")
    if any(abs(a - b) > 1e-7 for row_a, row_b in zip(lattice, result["lattice_angstrom"])
           for a, b in zip(row_a, row_b)):
        raise ValueError(".win lattice does not match source _tb.dat")
    for line in lines:
        if match := re.fullmatch(r"num_wann\s*=\s*(\d+)", line, re.I):
            if int(match[1]) != result["num_wann"]:
                raise ValueError(".win num_wann does not match source _tb.dat")
        if match := re.fullmatch(r"spinors\s*=\s*(\S+)", line, re.I):
            spinors = match[1].lower().strip(".")
            if spinors not in ("true", "false", "t", "f"):
                raise ValueError("Invalid spinors setting")
            if spinors in ("true", "t") and result["num_wann"] % 2:
                raise ValueError("Spinor input has odd num_wann")
    atoms = block("atoms_cart")
    if not atoms:
        raise ValueError("No atoms in .win")
    for line in atoms:
        fields = line.split()
        if len(fields) != 4 or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", fields[0]):
            raise ValueError("Expected explicit Cartesian atom coordinates in .win")
        list(map(real, fields[1:]))
    return atoms


def split(source: Path, output: Path, seed: str, win: Path) -> dict:
    if not seed or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in seed):
        raise ValueError("Seed must be a simple file stem")
    result = inspect(source, "tb")
    win_text = win.read_text(encoding="utf-8")
    atoms = validate_win(win_text, result)
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing split output")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".wannier-split-", dir=output.parent) as temporary:
        stage = Path(temporary)
        result = write_split(source, stage, seed, win, result, atoms, output)
        stage.rename(output)
    return result


def write_split(source, stage, seed, win, result, atoms, output):
    targets = [stage / (seed + suffix) for suffix in ("_hr.dat", "_r.dat", ".win", "_centres.xyz", "-split.json")]
    with source.open(encoding="utf-8") as stream:
        header = stream.readline().rstrip()
        for _ in range(3):
            nonempty(stream)
        n, nr = int(nonempty(stream)[0]), int(nonempty(stream)[0])
        degeneracies = []
        while len(degeneracies) < nr:
            degeneracies.extend(nonempty(stream))
        for section, destination in enumerate(targets[:2]):
            with destination.open("w", encoding="utf-8", newline="\n") as out:
                out.write(header + "\n" + str(n) + "\n" + str(nr) + "\n")
                if section == 0:
                    for start in range(0, nr, 15):
                        out.write(" ".join(degeneracies[start:start + 15]) + "\n")
                for _ in range(nr):
                    r = nonempty(stream)
                    for _ in range(n * n):
                        out.write(" ".join(r + nonempty(stream)) + "\n")
    shutil.copyfile(win, targets[2])
    centres = result["wannier_centres_angstrom"]
    targets[3].write_text(str(n + len(atoms)) + "\nDerived from original position-matrix origin diagonal; unchanged basis order\n" +
                          "".join("X " + " ".join(format(x, ".17g") for x in row) + "\n" for row in centres) +
                          "\n".join(atoms) + "\n", encoding="utf-8")
    result["operation"] = "format split only; original basis, all H/r values and global degeneracy retained"
    result["outputs"] = [str((output / path.name).resolve()) for path in targets[:-1]]
    targets[4].write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--seed", required=True)
    p.add_argument("--win", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(split(a.source, a.output, a.seed, a.win), indent=2))
