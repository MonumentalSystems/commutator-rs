# Public Hamiltonian data for response validation

Inspected on 2026-10-14. Source files stay outside Git in the sibling
`../hamiltonians/` directory. Their pinned URLs, sizes and SHA-256 digests are
in [`tools/hamiltonian-data/manifest.json`](../tools/hamiltonian-data/manifest.json).
These are independent material inputs, not recovered Hamiltonians from the
specific 2026 Te experiment or 2019 bilayer WTe2 experiment.

| Input | Acquired content | Suitable use and boundary |
| --- | --- | --- |
| Trigonal Te, WannierBerri tutorial | `Te_tb.dat`, `Te.win`, POSCAR, INCAR, KPOINTS; full Hamiltonian **and position matrices** | Best initial material Berry-response input; spin-orbit coupling, known handed structure. Source is explicitly unsymmetrized. |
| Bulk WTe2, WannierTools example | SOC Hamiltonian archive, `.win`, `.wout`, POSCAR, `wt.in` | Bulk band/interpolation control; **not Ma et al.'s bilayer**. No full position matrix supplied in this example. |
| TaIrTe4, Xi et al. 2025 | Previously acquired experimental/calculated spreadsheets | Useful regression/response-data check; the inspected Dryad release is not a complete Wannier Hamiltonian/position-matrix package. No exact paper Hamiltonian located in this bounded search. |
| Phasecraft material models | Thirty Hamiltonian JSON files acquired and verified from licensed Zenodo archive | Correlated/Majorana model inputs for five materials; separate problem from reproducing Te BCD. See below. |

## Trigonal Te

Primary source: [WannierBerri symmetrization tutorial](https://tutorial.wannier-berri.org/tutorials/5_symmetrization/tutorial_symmetrization-solution.html).
Repository pin: [`efe56e5b312a903bdbf06dcfc7b5fb8bb44c2afb`](https://github.com/wannier-berri/WannierBerri-tutorial/tree/efe56e5b312a903bdbf06dcfc7b5fb8bb44c2afb/tutorials/5_symmetrization/Te_data).

The 17,305,196-byte `Te_tb.dat` has SHA-256
`5a9f95f14535dfa894901c8ae2ca86897b5a9e3ad6787ef46cc4b12282c3bacf`.
It contains 24 spinor Wannier functions, 195 R vectors and 112,320 matrix records
per operator. Its 24-dimensional basis spans s and p projections on three Te
atoms, with two spin channels; it has not been downfolded to a smaller model.
The projection declaration lists `Te:s` followed by `Te:p`. The tutorial
explicitly says old VASP wrote spin channels in two blocks and calls
`spin_block2interlace()` before symmetry operations. The split files retain the
original block ordering; neither H nor r is reordered independently.

Lattice vectors in angstrom are `(4.457,0,0)`, `(-2.2285,3.8598752,0)`,
`(0,0,5.9581176)`. Fractional Te positions are `(0.274,0.274,0)`,
`(0.726,0,1/3)`, `(0,0.726,2/3)`. The tutorial identifies space group
P3_121, number 152. Preserve that structural identity rather than guessing an
experimental LH/RH label. The opposite enantiomorph and any axis reversal need
an explicit, consistent transformation of lattice, atoms, H and r.

Companion inputs specify `spinors=true`, `LSORBIT=T`, hybrid-functional flags,
and an original 6x6x4 mesh. They are provenance inputs, not evidence that this
task reran DFT. Global degeneracies range 1-4; their inverse sum is 144, matching
that mesh. We found no `wsvec.dat` or standalone centres file in the tutorial
folder. The `_tb.dat` itself supplies all three position matrices, including
off-diagonal/intercell elements. Missing `BB/CC/SS` operators preclude treating
this as a complete orbital-magnetization or spin-response dataset.

Our structural audit verifies all orbital pairs and R blocks, finite values,
and Hamiltonian R/-R adjoint residual approximately 1e-15 eV. The raw position
matrix has a maximum corresponding residual **0.142823489 angstrom**. This is a
reported diagnostic, not a corrected dataset or a proof of unusability;
finite-difference position connections require care. The source tutorial
explicitly studies symmetry restoration of H and position matrices. Compare
raw, Hermitian/symmetry-restored, and independently computed responses before
making material-level accuracy claims. Do not silently repair the input.

No repository license was detected at the pinned tutorial revision. Downloaded
inputs and derived matrices remain outside Git; public availability is not an
explicit redistribution license. Proprietary VASP POTCAR data were not fetched.

For importers accepting separate files, `split_tb.py` produces `Te_hr.dat`,
`Te_r.dat`, `Te.win`, and `Te_centres.xyz` under `../hamiltonians/te/split/`.
Every H/r value token, index and degeneracy survives the format conversion.
Wannier centres are the original r(R=0) diagonal, not substituted atomic sites.

## Bulk WTe2

Source: [WannierTools WTe2 example](https://github.com/quanshengwu/wannier_tools/tree/362171ce35450118828c6960a882e3a6d08c73f5/examples/WTe2),
pinned commit `362171ce35450118828c6960a882e3a6d08c73f5`.
The archive is 19,106,014 bytes, SHA-256
`48b20e62ece962bb842aa0163dbaaba0958c0ffb731e2194b99e602a000a942e`.
Its `wannier90_hr.dat` member is 145,153,657 bytes, SHA-256
`f491a3412c7864d210ac89b9dde684c721fc272ce8957fd486f29f90f2a7eaea`.

This is a three-dimensional bulk cell with 4 W and 8 Te atoms, lattice
`diag(3.477,6.249,14.018)` angstrom, 96 spinor Wannier functions, 315 R vectors,
and SOC enabled. The basis uses W s+d and Te p projections with two spin
channels. `wt.in` supplies a reference Fermi energy 7.3147 eV; that is an input
convention, not an absolute vacuum alignment. Its repository is GPL-3.0; no
separate data license was identified in this example.
All 2,903,040 H records were inspected; Hermiticity residual is exactly zero at
stored precision, and the inverse-degeneracy sum is 192, matching the 8x6x4 mesh.

The example includes Wannier centres in `.wout` but no complete `_r.dat` or
`_tb.dat` position operator, nor a `wsvec.dat`. An H-only tight-binding Berry
calculation is possible under an explicitly declared position approximation;
it is not equivalent to full ab-initio Wannier Berry interpolation. Cutting a
bilayer from this bulk model does not reconstruct the field, stacking,
relaxation or electrostatics of the Ma experiment.

## Phasecraft catalogue and actual coefficients

The [Materials Modeling Quantum Complexity Database](https://materials.phasecraft.io/)
currently lists 46 materials and structural, Wannier, Hamiltonian-term-count
and quantum-circuit resource metadata. The inspected landing and material
pages expose no coefficient download. The [methodology page](https://materials.phasecraft.io/about/)
links the underlying paper, [Clinton et al., Nature Communications 15, 211
(2024)](https://www.nature.com/articles/s41467-023-43479-6), whose data statement
points to **actual material Hamiltonians**:
[Zenodo 8370650](https://zenodo.org/records/8370650), DOI
`10.5281/zenodo.8370650`, license CC-BY-4.0.

That release describes five material models: SrVO3, GaAs, H3S, Si and Li2CuO2.
It supplies Majorana-monomial coefficients (`interactions.json`),
Majorana-to-mode and mode-to-site maps, site Cartesian coordinates, stage
metadata, encodings and compiled quantum circuits. This is especially relevant
to commutator-rs's Majorana/lattice infrastructure. It is a truncated interacting
Hamiltonian representation; it does not substitute for Bloch Hamiltonians plus
position matrices in a full material BCD calculation. The five-model research
release should not be described as a download of all 46 catalogue entries.

The advertised ZIP is 92,023,575 bytes with repository MD5
`33cf4fb7b6d435814112fa75ee9b19a1`. We acquired only thirty Hamiltonian JSON
members plus the README using verified HTTP byte ranges. Each member passes
ZIP CRC32, size, JSON parsing and SHA-256 recording. The whole-archive MD5 has
**not** been verified. Byte-range and member hashes are recorded in
[`phasecraft-selective-manifest.json`](../tools/hamiltonian-data/phasecraft-selective-manifest.json).
Files reside at `../hamiltonians/phasecraft/selected/`.

`interactions.json` is a JSON object whose keys are tuple strings, e.g.
`"(0, 1)"`, and whose values are complex-number strings, e.g.
`"59.92469766313952j"`. The constant key is `"()"`. Observed indices are
zero-based, distinct and ascending. All stored terms pass the algebraic
Hermiticity phase check for self-adjoint, mutually anticommuting Majoranas:
`c = conj(c) * (-1)^(d*(d-1)/2)`. This check does not establish operator
normalization or the sign of the creation/annihilation-to-Majorana map.

| Material | Constant / quadratic / quartic terms | Mode-map entries | Site-map entries |
| --- | --- | --- | --- |
| SrVO3 | 1 / 162 / 231 | 114 | 19 |
| GaAs | 1 / 608 / 532 | 440 | 55 |
| H3S | 1 / 2338 / 29 | 910 | 65 |
| Si | 1 / 752 / 464 | 440 | 55 |
| Li2CuO2 | 1 / 786 / 75 | 506 | 23 |

These are central-site/neighborhood model records, with a `privileged_group`
and interaction motifs in stage metadata; do not mistake their map size for
the paper's tiled simulation size. For example, SrVO3 has 114 mapped modes in
this neighborhood, whereas the catalogue's 3x3x3 tiled example has 162 modes.
Translation, multiplicities and boundary conditions require explicit
reconstruction before a many-body run. The README and inspected JSON do not
explicitly declare coefficient energy units, `gamma^2`, or the phase convention
defining each Majorana. Those remain unresolved import requirements. No claim
of immediate compatibility with `majorana-fermions` is made.

## Reproduction tools

Python 3.11+ standard library only. Run from the repository root, using your
Python executable:

```powershell
python tools/hamiltonian-data/fetch.py ../hamiltonians --dataset te --dataset wte2
python tools/hamiltonian-data/fetch_phasecraft.py ../hamiltonians/phasecraft/selected
python tools/hamiltonian-data/inspect_wannier.py ../hamiltonians/te/Te_tb.dat --format tb
python tools/hamiltonian-data/split_tb.py ../hamiltonians/te/Te_tb.dat ../hamiltonians/te/split --seed Te --win ../hamiltonians/te/Te.win
python -m unittest discover -s tools/hamiltonian-data -v
```

The inspector follows the [official Wannier90 file formats](https://wannier90.readthedocs.io/en/latest/user_guide/wannier90/files/).
It does not certify band accuracy, Berry curvature, crystal symmetry, or
integration convergence. Do not omit Wigner-Seitz degeneracies when Fourier
transforming; if a future input supplies per-orbital `wsvec` translations, they
need their own explicit handling. A full position matrix must be transformed
together with H under every basis change.
