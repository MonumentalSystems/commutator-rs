# quantum-tomography

Small checked reference and interoperability tools for local-Pauli state
tomography and Pauli-transfer-matrix channel diagnostics. It provides complete
measurement plans, trace-one state inversion, Euclidean PSD trace-one
projection, an informationally complete product-state process design, process
linear inversion into the Pauli-transfer representation, normalized Choi
conversion, and TP/unital/CP checks.
It also includes a focused PTM gate-set model, sequence probabilities,
multinomial likelihood/deviance records, and probability-preserving similarity
gauge transforms.

This crate intentionally does not duplicate qtool's optimization-based FISTA
quantum process tomography. Dense storage and the dependency-light Hermitian
Jacobi eigensolver target small systems, validation, and interchange boundaries.
Production reconstruction should use specialized maximum-likelihood or convex
solvers and preserve these conventions at the boundary. The internal complex
Hermitian Jacobi solver uses a caller-controlled, scale-relative convergence
threshold and reports non-convergence.
Process inversion accepts exactly `4^n` linearly independent normalized probe
states; `product_probe_expectations` supplies the minimal tensor-product design
from `|0>`, `|1>`, `|+>`, and `|+i>`. Gate-set parameter optimization is
deliberately injected by callers; this crate supplies the checked model and
objective diagnostics, not a turnkey nonlinear optimizer.

The Pauli ordering is lexicographic `I,X,Y,Z` per qubit, with the rightmost
qubit varying fastest. PTMs use `R_ab = Tr[P_a E(P_b)]/d`; normalized Choi
matrices have trace one for trace-preserving maps.

## License

MIT.
