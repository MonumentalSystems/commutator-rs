use crate::matrix::{
    check_square, matrix_add, matrix_max_difference, matrix_scale, shifted_identity_minus,
    validate_frequencies, validate_retarded_causality,
};
use crate::{ClusterGreenGrid, Complex64, DenseMatrix, EmbeddingError, HubbardModel, Result};

/// A checked complex-matrix grid on ordered retarded frequencies.
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixFrequencyGrid {
    frequencies: Vec<Complex64>,
    matrices: Vec<DenseMatrix>,
    sites: usize,
}

impl MatrixFrequencyGrid {
    /// Construct a grid, checking frequency order, broadening, dimensions, and
    /// matrix finiteness (the latter is enforced by `DenseMatrix`).
    pub fn try_new(frequencies: Vec<Complex64>, matrices: Vec<DenseMatrix>) -> Result<Self> {
        validate_frequencies(&frequencies)?;
        if matrices.len() != frequencies.len() {
            return Err(EmbeddingError::Length {
                name: "matrix frequency grid",
                expected: frequencies.len(),
                actual: matrices.len(),
            });
        }
        let sites = matrices
            .first()
            .ok_or(EmbeddingError::Empty("matrix frequency grid"))?
            .rows();
        if sites == 0 {
            return Err(EmbeddingError::Empty("matrix frequency-grid basis"));
        }
        for matrix in &matrices {
            check_square(matrix, sites, "matrix frequency grid")?;
        }
        Ok(Self {
            frequencies,
            matrices,
            sites,
        })
    }

    /// Build a general matrix grid from a checked cluster Green grid.
    pub fn from_cluster_green(grid: &ClusterGreenGrid) -> Self {
        Self {
            frequencies: grid.points().iter().map(|point| point.frequency).collect(),
            matrices: grid
                .points()
                .iter()
                .map(|point| point.green.clone())
                .collect(),
            sites: grid.sites(),
        }
    }

    /// Return ordered retarded frequencies.
    pub fn frequencies(&self) -> &[Complex64] {
        &self.frequencies
    }

    /// Return matrices aligned with `frequencies`.
    pub fn matrices(&self) -> &[DenseMatrix] {
        &self.matrices
    }

    /// Return the basis dimension.
    pub const fn sites(&self) -> usize {
        self.sites
    }

    fn ensure_compatible(&self, other: &Self, name: &'static str) -> Result<()> {
        if self.sites != other.sites {
            return Err(EmbeddingError::Dimensions {
                name,
                expected: (self.sites, self.sites),
                actual: (other.sites, other.sites),
            });
        }
        if self.frequencies != other.frequencies {
            return Err(EmbeddingError::InvalidFrequencyGrid);
        }
        Ok(())
    }
}

/// A causal retarded DMFT Weiss Green-function grid `G_0`.
#[derive(Debug, Clone, PartialEq)]
pub struct WeissFieldGrid(MatrixFrequencyGrid);

impl WeissFieldGrid {
    /// Construct and causality-check a Weiss field.
    pub fn try_new(grid: MatrixFrequencyGrid, tolerance: f64) -> Result<Self> {
        validate_retarded_causality(grid.matrices(), "Weiss field", tolerance)?;
        Ok(Self(grid))
    }

    /// Return the underlying checked grid.
    pub const fn grid(&self) -> &MatrixFrequencyGrid {
        &self.0
    }
}

/// A causal retarded hybridization grid `Delta`.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridizationGrid(MatrixFrequencyGrid);

impl HybridizationGrid {
    /// Construct and causality-check a hybridization.
    pub fn try_new(grid: MatrixFrequencyGrid, tolerance: f64) -> Result<Self> {
        validate_retarded_causality(grid.matrices(), "hybridization", tolerance)?;
        Ok(Self(grid))
    }

    /// Return the underlying checked grid.
    pub const fn grid(&self) -> &MatrixFrequencyGrid {
        &self.0
    }
}

/// Input passed to an injected DMFT impurity solver.
#[derive(Debug, Clone, PartialEq)]
pub struct ImpurityProblem {
    /// Current causal Weiss Green function.
    pub weiss: WeissFieldGrid,
    /// Onsite interactions in orbital order.
    pub onsite_interaction: Vec<f64>,
    /// Chemical potential used by the embedding convention.
    pub chemical_potential: f64,
}

/// Output returned by an injected DMFT impurity solver.
#[derive(Debug, Clone, PartialEq)]
pub struct ImpuritySolution {
    green: ClusterGreenGrid,
    self_energy: MatrixFrequencyGrid,
}

impl ImpuritySolution {
    /// Construct an impurity solution with matching causal retarded Green and
    /// self-energy grids.
    pub fn try_new(
        green: ClusterGreenGrid,
        self_energy: MatrixFrequencyGrid,
        tolerance: f64,
    ) -> Result<Self> {
        let green_grid = MatrixFrequencyGrid::from_cluster_green(&green);
        green_grid.ensure_compatible(&self_energy, "impurity self-energy")?;
        validate_retarded_causality(self_energy.matrices(), "impurity self-energy", tolerance)?;
        Ok(Self { green, self_energy })
    }

    /// Return the interacting local Green-function grid.
    pub const fn green(&self) -> &ClusterGreenGrid {
        &self.green
    }

    /// Return the impurity self-energy grid.
    pub const fn self_energy(&self) -> &MatrixFrequencyGrid {
        &self.self_energy
    }
}

/// Injected impurity solver used by the DMFT loop.
pub trait ImpuritySolver {
    /// Solve one impurity problem. The implementation owns bath fitting and
    /// many-body approximation choices.
    fn solve(&mut self, problem: &ImpurityProblem) -> Result<ImpuritySolution>;
}

impl<F> ImpuritySolver for F
where
    F: FnMut(&ImpurityProblem) -> Result<ImpuritySolution>,
{
    fn solve(&mut self, problem: &ImpurityProblem) -> Result<ImpuritySolution> {
        self(problem)
    }
}

/// Compute the DMFT Dyson update `G_0 = [G_loc^-1 + Sigma]^-1`.
pub fn dmft_weiss_from_local(
    local_green: &ClusterGreenGrid,
    self_energy: &MatrixFrequencyGrid,
    causality_tolerance: f64,
) -> Result<WeissFieldGrid> {
    let local = MatrixFrequencyGrid::from_cluster_green(local_green);
    local.ensure_compatible(self_energy, "local Green/self-energy grid")?;
    let matrices = local
        .matrices()
        .iter()
        .zip(self_energy.matrices())
        .map(|(green, sigma)| {
            matrix_add(&green.inverse()?, sigma)?
                .inverse()
                .map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    WeissFieldGrid::try_new(
        MatrixFrequencyGrid::try_new(local.frequencies().to_vec(), matrices)?,
        causality_tolerance,
    )
}

/// Recover `Delta = (z + mu)I - h_imp - G_0^-1` and check retarded causality.
pub fn hybridization_from_weiss(
    weiss: &WeissFieldGrid,
    impurity_hopping: &DenseMatrix,
    chemical_potential: f64,
    causality_tolerance: f64,
) -> Result<HybridizationGrid> {
    check_square(impurity_hopping, weiss.grid().sites(), "impurity hopping")?;
    if !impurity_hopping.is_hermitian(1.0e-12)? {
        return Err(EmbeddingError::NonHermitian("impurity hopping"));
    }
    let matrices = weiss
        .grid()
        .frequencies()
        .iter()
        .zip(weiss.grid().matrices())
        .map(|(frequency, value)| {
            shifted_identity_minus(*frequency, chemical_potential, &[impurity_hopping])?
                .subtract(&value.inverse()?)
                .map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    HybridizationGrid::try_new(
        MatrixFrequencyGrid::try_new(weiss.grid().frequencies().to_vec(), matrices)?,
        causality_tolerance,
    )
}

/// Linearly mix two compatible Weiss grids as
/// `(1 - alpha) old + alpha new`, with `alpha in (0, 1]`.
pub fn linear_mix_grid(
    old: &WeissFieldGrid,
    new: &WeissFieldGrid,
    alpha: f64,
    causality_tolerance: f64,
) -> Result<WeissFieldGrid> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha > 1.0 {
        return Err(EmbeddingError::InvalidValue("DMFT mixing coefficient"));
    }
    old.grid()
        .ensure_compatible(new.grid(), "mixed Weiss grids")?;
    let matrices = old
        .grid()
        .matrices()
        .iter()
        .zip(new.grid().matrices())
        .map(|(old_value, new_value)| {
            matrix_add(
                &matrix_scale(old_value, 1.0 - alpha)?,
                &matrix_scale(new_value, alpha)?,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    WeissFieldGrid::try_new(
        MatrixFrequencyGrid::try_new(old.grid().frequencies().to_vec(), matrices)?,
        causality_tolerance,
    )
}

/// Infinite-coordination Bethe-lattice self-consistency parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BetheLattice {
    hopping: f64,
}

impl BetheLattice {
    /// Construct from the scaled nearest-neighbor hopping `t`, for which
    /// `Delta(z) = t^2 G_loc(z)` and the noninteracting half-bandwidth is `2t`.
    pub fn try_new(hopping: f64) -> Result<Self> {
        if !hopping.is_finite() || hopping <= 0.0 {
            return Err(EmbeddingError::InvalidValue("Bethe hopping"));
        }
        Ok(Self { hopping })
    }

    /// Return the scaled hopping.
    pub const fn hopping(&self) -> f64 {
        self.hopping
    }

    fn target_weiss(
        &self,
        local_green: &ClusterGreenGrid,
        impurity_hopping: &DenseMatrix,
        chemical_potential: f64,
        tolerance: f64,
    ) -> Result<WeissFieldGrid> {
        check_square(impurity_hopping, local_green.sites(), "impurity hopping")?;
        let mut matrices = Vec::with_capacity(local_green.points().len());
        for point in local_green.points() {
            let hybridization = matrix_scale(&point.green, self.hopping * self.hopping)?;
            let inverse = shifted_identity_minus(
                point.frequency,
                chemical_potential,
                &[impurity_hopping, &hybridization],
            )?;
            matrices.push(inverse.inverse()?);
        }
        WeissFieldGrid::try_new(
            MatrixFrequencyGrid::try_new(
                local_green
                    .points()
                    .iter()
                    .map(|point| point.frequency)
                    .collect(),
                matrices,
            )?,
            tolerance,
        )
    }
}

/// Configuration for a Bethe DMFT fixed-point iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DmftConfig {
    /// Maximum impurity-solver calls.
    pub maximum_iterations: usize,
    /// New-Weiss mixing coefficient in `(0, 1]`.
    pub mixing: f64,
    /// Maximum matrix-element residual required for convergence.
    pub convergence_tolerance: f64,
    /// Scale-aware causality tolerance.
    pub causality_tolerance: f64,
}

impl Default for DmftConfig {
    fn default() -> Self {
        Self {
            maximum_iterations: 100,
            mixing: 0.5,
            convergence_tolerance: 1.0e-8,
            causality_tolerance: 1.0e-12,
        }
    }
}

/// One recorded DMFT fixed-point iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DmftIteration {
    /// Zero-based iteration number.
    pub iteration: usize,
    /// Maximum absolute element of `G_0,target - G_0,current` before mixing.
    pub residual: f64,
}

/// Ordered DMFT convergence evidence.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DmftHistory {
    iterations: Vec<DmftIteration>,
}

impl DmftHistory {
    /// Return all recorded iterations.
    pub fn iterations(&self) -> &[DmftIteration] {
        &self.iterations
    }
}

/// Final state of a Bethe DMFT run.
#[derive(Debug, Clone, PartialEq)]
pub struct DmftOutcome {
    /// Last mixed Weiss field.
    pub weiss: WeissFieldGrid,
    /// Last impurity solution.
    pub impurity: ImpuritySolution,
    /// Whether the requested residual tolerance was reached.
    pub converged: bool,
    /// Full convergence history.
    pub history: DmftHistory,
}

/// Run solver-injected Bethe-lattice DMFT with causal linear Weiss mixing.
pub fn bethe_dmft<S: ImpuritySolver + ?Sized>(
    solver: &mut S,
    impurity_model: &HubbardModel,
    lattice: BetheLattice,
    initial_weiss: WeissFieldGrid,
    config: DmftConfig,
) -> Result<DmftOutcome> {
    validate_config(config)?;
    if initial_weiss.grid().sites() != impurity_model.sites() {
        return Err(EmbeddingError::Dimensions {
            name: "initial Weiss/impurity basis",
            expected: (impurity_model.sites(), impurity_model.sites()),
            actual: (initial_weiss.grid().sites(), initial_weiss.grid().sites()),
        });
    }
    let mut weiss = initial_weiss;
    let mut history = DmftHistory::default();
    let mut last_solution = None;
    for iteration in 0..config.maximum_iterations {
        let problem = ImpurityProblem {
            weiss: weiss.clone(),
            onsite_interaction: impurity_model.interaction().to_vec(),
            chemical_potential: impurity_model.chemical_potential(),
        };
        let solution = solver.solve(&problem)?;
        let target = lattice.target_weiss(
            solution.green(),
            impurity_model.hopping(),
            impurity_model.chemical_potential(),
            config.causality_tolerance,
        )?;
        weiss
            .grid()
            .ensure_compatible(target.grid(), "Bethe target Weiss")?;
        let residual = weiss
            .grid()
            .matrices()
            .iter()
            .zip(target.grid().matrices())
            .map(|(current, target)| matrix_max_difference(current, target))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .fold(0.0_f64, f64::max);
        history.iterations.push(DmftIteration {
            iteration,
            residual,
        });
        last_solution = Some(solution);
        if residual <= config.convergence_tolerance {
            return Ok(DmftOutcome {
                weiss,
                impurity: last_solution.expect("solution assigned"),
                converged: true,
                history,
            });
        }
        weiss = linear_mix_grid(&weiss, &target, config.mixing, config.causality_tolerance)?;
    }
    Ok(DmftOutcome {
        weiss,
        impurity: last_solution.expect("positive maximum iterations"),
        converged: false,
        history,
    })
}

fn validate_config(config: DmftConfig) -> Result<()> {
    if config.maximum_iterations == 0
        || !config.mixing.is_finite()
        || config.mixing <= 0.0
        || config.mixing > 1.0
        || !config.convergence_tolerance.is_finite()
        || config.convergence_tolerance <= 0.0
        || !config.causality_tolerance.is_finite()
        || config.causality_tolerance <= 0.0
    {
        Err(EmbeddingError::InvalidValue("DMFT configuration"))
    } else {
        Ok(())
    }
}
