# quantum-tomography

Small checked reference and interoperability tools for local-Pauli state
tomography and Pauli-transfer-matrix channel diagnostics. It provides complete
measurement plans, trace-one state inversion, Euclidean PSD trace-one
projection, an informationally complete product-state process design, process
linear inversion into the Pauli-transfer representation, normalized Choi
conversion, and TP/unital/CP checks.
It also includes a focused PTM gate-set model, sequence probabilities,
multinomial likelihood/deviance records, probability-preserving similarity
gauge transforms, and a deterministic checked reconstruction routine for
explicitly selected GST coordinates.

This crate intentionally does not duplicate qtool's optimization-based FISTA
quantum process tomography. Dense storage and the dependency-light Hermitian
Jacobi eigensolver target small systems, validation, and interchange boundaries.
Production reconstruction should use specialized maximum-likelihood or convex
solvers and preserve these conventions at the boundary. The internal complex
Hermitian Jacobi solver uses a caller-controlled, scale-relative convergence
threshold and reports non-convergence.
Process inversion accepts exactly `4^n` linearly independent normalized probe
states; `product_probe_expectations` supplies the minimal tensor-product design
from `|0>`, `|1>`, `|+>`, and `|+i>`.

`reconstruct_gate_set` provides a dependency-light reference GST optimizer:
central finite differences, deterministic backtracking, explicit convergence
controls, likelihood/deviance diagnostics, and a recorded stopping reason.
Because experimental GST is similarity-gauge ambiguous, callers must select an
identifiable `GstParameter` coordinate chart (and may first use
`gauge_transform`); unselected coordinates remain fixed. Optional quadratic
anchoring is an explicit prior, not information learned from data. The routine
does not enforce CPTP constraints or replace mature production GST optimizers;
it is intended for small synthetic problems, interoperability checks, and
injected-optimizer baselines.

The Pauli ordering is lexicographic `I,X,Y,Z` per qubit, with the rightmost
qubit varying fastest. PTMs use `R_ab = Tr[P_a E(P_b)]/d`; normalized Choi
matrices have trace one for trace-preserving maps.

## License

MIT.
