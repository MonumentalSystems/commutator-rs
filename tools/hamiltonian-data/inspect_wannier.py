"""Inspect public Wannier90 data without computing bands or silently dropping r.

Python standard library only. Formats follow the official Wannier90 file guide:
https://wannier90.readthedocs.io/en/latest/user_guide/wannier90/files/
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
from typing import TextIO


def nonempty(stream: TextIO) -> list[str]:
    for line in stream:
        if line.strip():
            return line.split()
    raise ValueError("Unexpected end of file")


def real(token: str) -> float:
    value = float(token.replace("D", "E").replace("d", "e"))
    if not math.isfinite(value):
        raise ValueError("Non-finite matrix value")
    return value


def read_section(stream: TextIO, n: int, nr: int, components: int, tb: bool):
    matrices = {}
    order = []
    for _ in range(nr):
        r = tuple(map(int, nonempty(stream))) if tb else None
        if tb and len(r) != 3:
            raise ValueError("Expected three-component R vector")
        values = array("d", [0.0]) * (n * n * components * 2)
        seen = bytearray(n * n)
        for _ in range(n * n):
            fields = nonempty(stream)
            offset = 0 if tb else 3
            if len(fields) != offset + 2 + components * 2:
                raise ValueError("Wrong matrix record width")
            if not tb:
                current_r = tuple(map(int, fields[:3]))
                if r is None:
                    r = current_r
                elif current_r != r:
                    raise ValueError("R changed inside a matrix block")
            i, j = map(int, fields[offset:offset + 2])
            if not (1 <= i <= n and 1 <= j <= n):
                raise ValueError("Orbital index out of range")
            idx = (i - 1) * n + j - 1
            if seen[idx]:
                raise ValueError("Duplicate orbital pair")
            seen[idx] = 1
            start = idx * components * 2
            values[start:start + components * 2] = array("d", map(real, fields[offset + 2:]))
        if r in matrices:
            raise ValueError("Duplicate R block")
        matrices[r] = values
        order.append(r)
    return matrices, order


def hermitian_residual(matrices, n: int, components: int) -> float:
    maximum = 0.0
    for r, values in matrices.items():
        opposite = matrices.get(tuple(-x for x in r))
        if opposite is None:
            raise ValueError(f"Missing Hermitian partner for R={r}")
        for i in range(n):
            for j in range(n):
                for c in range(components):
                    a = ((i * n + j) * components + c) * 2
                    b = ((j * n + i) * components + c) * 2
                    maximum = max(maximum, math.hypot(values[a] - opposite[b], values[a + 1] + opposite[b + 1]))
    return maximum


def inspect(path: Path, kind: str) -> dict:
    with path.open(encoding="utf-8") as stream:
        header = stream.readline().strip()
        lattice = [[real(x) for x in nonempty(stream)] for _ in range(3)] if kind == "tb" else None
        if lattice is not None and any(len(row) != 3 for row in lattice):
            raise ValueError("Invalid lattice")
        n, nr = int(nonempty(stream)[0]), int(nonempty(stream)[0])
        if n <= 0 or nr <= 0 or n * n * nr > 20_000_000:
            raise ValueError("Invalid or unexpectedly large matrix dimensions")
        degeneracies = []
        while len(degeneracies) < nr:
            degeneracies.extend(map(int, nonempty(stream)))
        if len(degeneracies) != nr or min(degeneracies) <= 0:
            raise ValueError("Invalid degeneracy table")
        ham, order = read_section(stream, n, nr, 1, kind == "tb")
        h_error = hermitian_residual(ham, n, 1)
        del ham
        r_error = None
        centres = None
        if kind == "tb":
            position, r_order = read_section(stream, n, nr, 3, True)
            if order != r_order:
                raise ValueError("Hamiltonian and position R orders differ")
            r_error = hermitian_residual(position, n, 3)
            origin = position.get((0, 0, 0))
            if origin is None:
                raise ValueError("Position matrix has no origin block")
            centres = [[origin[((i * n + i) * 3 + c) * 2] for c in range(3)] for i in range(n)]
        if any(line.strip() for line in stream):
            raise ValueError("Unexpected trailing nonblank records")
    with path.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "file": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256,
        "format": kind, "header": header, "num_wann": n, "num_r": nr,
        "hamiltonian_records": n * n * nr, "position_records": n * n * nr if kind == "tb" else 0,
        "lattice_angstrom": lattice, "energy_unit": "eV", "position_unit": "angstrom" if kind == "tb" else None,
        "degeneracy_min": min(degeneracies), "degeneracy_max": max(degeneracies),
        "sum_inverse_degeneracy": sum(1 / x for x in degeneracies),
        "hamiltonian_hermiticity_max_abs_ev": h_error,
        "position_hermiticity_max_abs_angstrom": r_error,
        "wannier_centres_angstrom": centres,
        "status": "structurally_valid; no band, Berry, symmetry, or convergence certification",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--format", choices=["hr", "tb"], required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(inspect(args.file, args.format), indent=2)
    if args.output:
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)
