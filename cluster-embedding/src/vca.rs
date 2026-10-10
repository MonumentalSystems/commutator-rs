use crate::matrix::{
    negated_log_determinant, shifted_identity_minus, unwrap_phase, validate_frequencies,
};
use crate::{
    ClusterGreenGrid, Complex64, DenseMatrix, EmbeddedGreenGrid, EmbeddingError, GreenPoint,
    ReferenceSolution, ReferenceSolver, ReferenceSystem, Result,
};

const HERMITIAN_TOLERANCE: f64 = 1.0e-12;

/// Selects how complex log-determinant phases are treated along frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogDetBranchPolicy {
    /// Use the principal phase `(-pi, pi]` independently at every point.
    Principal,
    /// Add multiples of `2 pi` to minimize phase jumps along each ordered
    /// frequency path. The first point remains on the principal branch.
    ContinuousFrequency,
}

/// One explicitly weighted lattice momentum and its one-body Hamiltonian.
#[derive(Debug, Clone, PartialEq)]
pub struct MomentumPoint {
    hopping: DenseMatrix,
    weight: f64,
}

impl MomentumPoint {
    /// Construct a momentum quadrature point. Weights must be finite and
    /// nonnegative; the full set must sum to one.
    pub fn try_new(hopping: DenseMatrix, weight: f64) -> Result<Self> {
        if hopping.rows() == 0 || hopping.rows() != hopping.columns() {
            return Err(EmbeddingError::Dimensions {
                name: "momentum hopping",
                expected: (hopping.rows(), hopping.rows()),
                actual: (hopping.rows(), hopping.columns()),
            });
        }
        if !hopping.is_hermitian(HERMITIAN_TOLERANCE)? {
            return Err(EmbeddingError::NonHermitian("momentum hopping"));
        }
        if !weight.is_finite() || weight < 0.0 {
            return Err(EmbeddingError::InvalidValue("momentum weight"));
        }
        Ok(Self { hopping, weight })
    }

    /// Return the lattice one-body Hamiltonian at this momentum.
    pub const fn hopping(&self) -> &DenseMatrix {
        &self.hopping
    }

    /// Return the normalized Brillouin-zone weight.
    pub const fn weight(&self) -> f64 {
        self.weight
    }
}

/// A Potthoff self-energy-functional quadrature problem.
#[derive(Debug, Clone, PartialEq)]
pub struct PotthoffFunctional {
    chemical_potential: f64,
    momenta: Vec<MomentumPoint>,
    frequency_weights: Vec<f64>,
    branch_policy: LogDetBranchPolicy,
}

impl PotthoffFunctional {
    /// Construct a checked functional quadrature.
    ///
    /// Momentum weights must sum to one to make the subtraction of the single
    /// reference-cluster trace-log unambiguous. Frequency weights are used
    /// verbatim and may be signed for contour quadratures.
    pub fn try_new(
        chemical_potential: f64,
        momenta: Vec<MomentumPoint>,
        frequency_weights: Vec<f64>,
        branch_policy: LogDetBranchPolicy,
    ) -> Result<Self> {
        if !chemical_potential.is_finite() {
            return Err(EmbeddingError::NonFinite("lattice chemical potential"));
        }
        if momenta.is_empty() {
            return Err(EmbeddingError::Empty("momentum quadrature"));
        }
        if frequency_weights.is_empty() {
            return Err(EmbeddingError::Empty("frequency quadrature"));
        }
        if !frequency_weights.iter().all(|weight| weight.is_finite()) {
            return Err(EmbeddingError::NonFinite("frequency quadrature weights"));
        }
        let sites = momenta[0].hopping.rows();
        if momenta
            .iter()
            .any(|point| point.hopping.rows() != sites || point.hopping.columns() != sites)
        {
            return Err(EmbeddingError::Dimensions {
                name: "momentum hopping",
                expected: (sites, sites),
                actual: momenta
                    .iter()
                    .find(|point| point.hopping.rows() != sites || point.hopping.columns() != sites)
                    .map(|point| (point.hopping.rows(), point.hopping.columns()))
                    .unwrap_or((sites, sites)),
            });
        }
        let momentum_sum: f64 = momenta.iter().map(|point| point.weight).sum();
        if (momentum_sum - 1.0).abs() > 1.0e-10 {
            return Err(EmbeddingError::InvalidValue(
                "momentum weights must sum to one",
            ));
        }
        Ok(Self {
            chemical_potential,
            momenta,
            frequency_weights,
            branch_policy,
        })
    }

    /// Evaluate `Omega' + Tr ln(-G) - Tr ln(-G')` on the explicit quadrature.
    pub fn evaluate(&self, solution: &ReferenceSolution) -> Result<PotthoffEvaluation> {
        let points = solution.green().points();
        if self.frequency_weights.len() != points.len() {
            return Err(EmbeddingError::Length {
                name: "frequency quadrature weights",
                expected: points.len(),
                actual: self.frequency_weights.len(),
            });
        }
        if solution.green().sites() != self.momenta[0].hopping.rows() {
            return Err(EmbeddingError::Dimensions {
                name: "lattice/reference site basis",
                expected: (solution.green().sites(), solution.green().sites()),
                actual: (
                    self.momenta[0].hopping.rows(),
                    self.momenta[0].hopping.columns(),
                ),
            });
        }
        let frequencies: Vec<_> = points.iter().map(|point| point.frequency).collect();
        let reference_logs =
            path_log_determinants(points.iter().map(|point| &point.green), self.branch_policy)?;
        let mut lattice_logs = vec![Complex64::new(0.0, 0.0); points.len()];
        for momentum in &self.momenta {
            let embedded = embed_self_energy(
                &frequencies,
                solution.self_energy(),
                momentum.hopping(),
                self.chemical_potential,
            )?;
            let logs = path_log_determinants(
                embedded.points().iter().map(|point| &point.green),
                self.branch_policy,
            )?;
            for (total, value) in lattice_logs.iter_mut().zip(logs) {
                *total += value * momentum.weight;
            }
        }
        let correction: Complex64 = self
            .frequency_weights
            .iter()
            .zip(lattice_logs.iter().zip(reference_logs))
            .map(|(weight, (lattice, reference))| (*lattice - reference) * *weight)
            .sum();
        Ok(PotthoffEvaluation {
            value: solution.grand_potential() + correction.re,
            imaginary_residual: correction.im,
        })
    }
}

/// Result of a complex trace-log Potthoff evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PotthoffEvaluation {
    /// Real functional value used for stationarity.
    pub value: f64,
    /// Integrated imaginary trace-log remainder, useful for detecting branch
    /// and quadrature problems.
    pub imaginary_residual: f64,
}

/// Apply the direct CPT embedding supplied by `cluster-green`.
pub fn cpt_embed(
    cluster: &ClusterGreenGrid,
    intercluster_hopping: &DenseMatrix,
) -> Result<EmbeddedGreenGrid> {
    cluster.cpt_at(intercluster_hopping).map_err(Into::into)
}

/// Embed a retarded self-energy into a momentum-resolved lattice Green grid.
///
/// Computes `G(k,z) = [(z + mu)I - h(k) - Sigma(z)]^-1`.
pub fn embed_self_energy(
    frequencies: &[Complex64],
    self_energy: &[DenseMatrix],
    hopping: &DenseMatrix,
    chemical_potential: f64,
) -> Result<ClusterGreenGrid> {
    validate_frequencies(frequencies)?;
    if self_energy.len() != frequencies.len() {
        return Err(EmbeddingError::Length {
            name: "self-energy frequency grid",
            expected: frequencies.len(),
            actual: self_energy.len(),
        });
    }
    let sites = hopping.rows();
    if sites == 0 || hopping.columns() != sites {
        return Err(EmbeddingError::Dimensions {
            name: "lattice hopping",
            expected: (sites, sites),
            actual: (hopping.rows(), hopping.columns()),
        });
    }
    if !hopping.is_hermitian(HERMITIAN_TOLERANCE)? {
        return Err(EmbeddingError::NonHermitian("lattice hopping"));
    }
    let points = frequencies
        .iter()
        .zip(self_energy)
        .map(|(frequency, sigma)| {
            let inverse =
                shifted_identity_minus(*frequency, chemical_potential, &[hopping, sigma])?;
            Ok(GreenPoint::new(*frequency, inverse.inverse()?))
        })
        .collect::<Result<Vec<_>>>()?;
    ClusterGreenGrid::try_new(sites, points).map_err(Into::into)
}

fn path_log_determinants<'a>(
    matrices: impl Iterator<Item = &'a DenseMatrix>,
    policy: LogDetBranchPolicy,
) -> Result<Vec<Complex64>> {
    let mut previous = None;
    let mut output = Vec::new();
    for matrix in matrices {
        let mut value = negated_log_determinant(matrix)?;
        if policy == LogDetBranchPolicy::ContinuousFrequency {
            value.im = unwrap_phase(value.im, previous);
        }
        previous = Some(value.im);
        output.push(value);
    }
    Ok(output)
}

/// Central finite-difference evidence for a VCA stationary point.
#[derive(Debug, Clone, PartialEq)]
pub struct StationarityReport {
    /// Functional at the undisplaced point.
    pub evaluation: PotthoffEvaluation,
    /// Parameter names in reference-system order.
    pub parameter_names: Vec<String>,
    /// Central first derivatives.
    pub gradient: Vec<f64>,
    /// Diagonal central second derivatives.
    pub diagonal_curvature: Vec<f64>,
    /// Maximum absolute gradient component.
    pub maximum_gradient: f64,
    /// Whether `maximum_gradient <= requested tolerance`.
    pub stationary: bool,
}

/// Evaluate central finite-difference stationarity diagnostics.
pub fn stationarity_diagnostics<S: ReferenceSolver + ?Sized>(
    solver: &mut S,
    system: &ReferenceSystem,
    functional: &PotthoffFunctional,
    gradient_tolerance: f64,
) -> Result<StationarityReport> {
    if !gradient_tolerance.is_finite() || gradient_tolerance <= 0.0 {
        return Err(EmbeddingError::InvalidValue("gradient tolerance"));
    }
    let center = functional.evaluate(&solver.solve(system)?)?;
    let mut names = Vec::with_capacity(system.terms().len());
    let mut gradient = Vec::with_capacity(system.terms().len());
    let mut curvature = Vec::with_capacity(system.terms().len());
    for (index, term) in system.terms().iter().enumerate() {
        let step = term.difference_step();
        let plus = system.with_parameter(index, term.value() + step)?;
        let minus = system.with_parameter(index, term.value() - step)?;
        let plus_value = functional.evaluate(&solver.solve(&plus)?)?.value;
        let minus_value = functional.evaluate(&solver.solve(&minus)?)?.value;
        names.push(term.name().to_owned());
        gradient.push((plus_value - minus_value) / (2.0 * step));
        curvature.push((plus_value - 2.0 * center.value + minus_value) / (step * step));
    }
    let maximum_gradient = gradient.iter().map(|value| value.abs()).fold(0.0, f64::max);
    Ok(StationarityReport {
        evaluation: center,
        parameter_names: names,
        gradient,
        diagonal_curvature: curvature,
        maximum_gradient,
        stationary: maximum_gradient <= gradient_tolerance,
    })
}

/// Configuration for conservative diagonal-Newton stationarity search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchConfig {
    /// Maximum optimizer iterations.
    pub maximum_iterations: usize,
    /// Required maximum absolute gradient.
    pub gradient_tolerance: f64,
    /// Maximum absolute change of any coefficient per iteration.
    pub trust_radius: f64,
    /// Curvature magnitude below which a gradient step is used.
    pub minimum_curvature: f64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            maximum_iterations: 32,
            gradient_tolerance: 1.0e-7,
            trust_radius: 0.25,
            minimum_curvature: 1.0e-10,
        }
    }
}

/// One recorded VCA search iteration.
#[derive(Debug, Clone, PartialEq)]
pub struct StationarityStep {
    /// Zero-based iteration number.
    pub iteration: usize,
    /// Coefficients before the proposed update.
    pub parameters: Vec<f64>,
    /// Functional value at those coefficients.
    pub functional_value: f64,
    /// Maximum absolute gradient at those coefficients.
    pub maximum_gradient: f64,
}

/// Search for a VCA stationary point with bounded diagonal-Newton updates.
///
/// VCA extrema may be saddles, so no minimum-only line search is applied. The
/// returned report always describes the returned system, including when the
/// maximum iteration count is reached.
pub fn find_stationary_point<S: ReferenceSolver + ?Sized>(
    solver: &mut S,
    initial: &ReferenceSystem,
    functional: &PotthoffFunctional,
    config: SearchConfig,
) -> Result<(ReferenceSystem, StationarityReport, Vec<StationarityStep>)> {
    if config.maximum_iterations == 0
        || !config.gradient_tolerance.is_finite()
        || config.gradient_tolerance <= 0.0
        || !config.trust_radius.is_finite()
        || config.trust_radius <= 0.0
        || !config.minimum_curvature.is_finite()
        || config.minimum_curvature <= 0.0
    {
        return Err(EmbeddingError::InvalidValue(
            "stationarity search configuration",
        ));
    }
    let mut system = initial.clone();
    let mut history = Vec::new();
    for iteration in 0..config.maximum_iterations {
        let report =
            stationarity_diagnostics(solver, &system, functional, config.gradient_tolerance)?;
        history.push(StationarityStep {
            iteration,
            parameters: system.terms().iter().map(|term| term.value()).collect(),
            functional_value: report.evaluation.value,
            maximum_gradient: report.maximum_gradient,
        });
        if report.stationary {
            return Ok((system, report, history));
        }
        let values: Vec<_> = system
            .terms()
            .iter()
            .zip(report.gradient.iter().zip(&report.diagonal_curvature))
            .map(|(term, (gradient, curvature))| {
                let raw = if curvature.abs() >= config.minimum_curvature {
                    -gradient / curvature
                } else {
                    -gradient.signum() * config.trust_radius
                };
                term.value() + raw.clamp(-config.trust_radius, config.trust_radius)
            })
            .collect();
        system = system.with_parameters(&values)?;
    }
    let report = stationarity_diagnostics(solver, &system, functional, config.gradient_tolerance)?;
    Ok((system, report, history))
}
