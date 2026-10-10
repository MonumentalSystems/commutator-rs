//! Solver-agnostic VCA and DMFT embedding foundations.
//!
//! The crate deliberately stops at the embedding boundary: interacting
//! reference and impurity problems are supplied through traits, while this
//! crate checks conventions, performs Dyson equations and lattice embedding,
//! and records numerical convergence evidence.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod dmft;
mod error;
mod matrix;
mod model;
mod vca;

pub use cluster_green::{ClusterGreenGrid, DenseMatrix, EmbeddedGreenGrid, GreenPoint};
pub use dmft::{
    bethe_dmft, dmft_weiss_from_local, hybridization_from_weiss, linear_mix_grid, BetheLattice,
    DmftConfig, DmftHistory, DmftIteration, DmftOutcome, HybridizationGrid, ImpurityProblem,
    ImpuritySolution, ImpuritySolver, MatrixFrequencyGrid, WeissFieldGrid,
};
pub use error::{EmbeddingError, Result};
pub use model::{HubbardModel, OneBodyTerm, ReferenceSolution, ReferenceSolver, ReferenceSystem};
pub use num_complex::Complex64;
pub use vca::{
    cpt_embed, embed_self_energy, find_stationary_point, stationarity_diagnostics,
    LogDetBranchPolicy, MomentumPoint, PotthoffEvaluation, PotthoffFunctional, SearchConfig,
    StationarityReport, StationarityStep,
};
