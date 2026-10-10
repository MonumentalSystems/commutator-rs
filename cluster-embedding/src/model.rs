use crate::matrix::{check_square, matrix_add, matrix_scale, validate_retarded_causality};
use crate::{ClusterGreenGrid, DenseMatrix, EmbeddingError, Result};

const HERMITIAN_TOLERANCE: f64 = 1.0e-12;

/// A checked single-band Hubbard model in an explicit site/orbital basis.
///
/// `hopping` is the full one-body Hamiltonian matrix (including any onsite
/// energies but excluding the chemical potential), and `interaction[i]` is
/// the onsite coefficient of `n_(i,up) n_(i,down)`. Attractive and repulsive
/// finite interactions are both accepted.
#[derive(Debug, Clone, PartialEq)]
pub struct HubbardModel {
    hopping: DenseMatrix,
    interaction: Vec<f64>,
    chemical_potential: f64,
}

impl HubbardModel {
    /// Construct a Hubbard model after checking dimensions, Hermiticity, and
    /// finite couplings.
    pub fn try_new(
        hopping: DenseMatrix,
        interaction: Vec<f64>,
        chemical_potential: f64,
    ) -> Result<Self> {
        if hopping.rows() == 0 || hopping.rows() != hopping.columns() {
            return Err(EmbeddingError::Dimensions {
                name: "Hubbard hopping",
                expected: (hopping.rows(), hopping.rows()),
                actual: (hopping.rows(), hopping.columns()),
            });
        }
        if !hopping.is_hermitian(HERMITIAN_TOLERANCE)? {
            return Err(EmbeddingError::NonHermitian("Hubbard hopping"));
        }
        if interaction.len() != hopping.rows() {
            return Err(EmbeddingError::Length {
                name: "onsite Hubbard interactions",
                expected: hopping.rows(),
                actual: interaction.len(),
            });
        }
        if !chemical_potential.is_finite() || !interaction.iter().all(|value| value.is_finite()) {
            return Err(EmbeddingError::NonFinite("Hubbard parameters"));
        }
        Ok(Self {
            hopping,
            interaction,
            chemical_potential,
        })
    }

    /// Return the number of sites/orbitals.
    pub fn sites(&self) -> usize {
        self.interaction.len()
    }

    /// Return the one-body Hamiltonian matrix.
    pub const fn hopping(&self) -> &DenseMatrix {
        &self.hopping
    }

    /// Return onsite Hubbard interactions in site order.
    pub fn interaction(&self) -> &[f64] {
        &self.interaction
    }

    /// Return the chemical potential in `G^-1 = (z + mu)I - h - Sigma`.
    pub const fn chemical_potential(&self) -> f64 {
        self.chemical_potential
    }
}

/// One Hermitian variational term `lambda * O` in a reference system.
#[derive(Debug, Clone, PartialEq)]
pub struct OneBodyTerm {
    name: String,
    operator: DenseMatrix,
    value: f64,
    difference_step: f64,
}

impl OneBodyTerm {
    /// Construct a checked variational term.
    pub fn try_new(
        name: impl Into<String>,
        operator: DenseMatrix,
        value: f64,
        difference_step: f64,
    ) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(EmbeddingError::Empty("variational parameter name"));
        }
        if operator.rows() == 0 || operator.rows() != operator.columns() {
            return Err(EmbeddingError::Dimensions {
                name: "variational operator",
                expected: (operator.rows(), operator.rows()),
                actual: (operator.rows(), operator.columns()),
            });
        }
        if !operator.is_hermitian(HERMITIAN_TOLERANCE)? {
            return Err(EmbeddingError::NonHermitian("variational operator"));
        }
        if !value.is_finite() || !difference_step.is_finite() || difference_step <= 0.0 {
            return Err(EmbeddingError::InvalidValue(
                "variational value or finite-difference step",
            ));
        }
        Ok(Self {
            name,
            operator,
            value,
            difference_step,
        })
    }

    /// Return the stable human-readable parameter name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the Hermitian one-body operator.
    pub const fn operator(&self) -> &DenseMatrix {
        &self.operator
    }

    /// Return the current coefficient.
    pub const fn value(&self) -> f64 {
        self.value
    }

    /// Return the central finite-difference displacement.
    pub const fn difference_step(&self) -> f64 {
        self.difference_step
    }
}

/// A Hubbard reference system plus explicit variational one-body terms.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceSystem {
    model: HubbardModel,
    terms: Vec<OneBodyTerm>,
}

impl ReferenceSystem {
    /// Construct a reference system, rejecting dimension mismatches and
    /// duplicate parameter names.
    pub fn try_new(model: HubbardModel, terms: Vec<OneBodyTerm>) -> Result<Self> {
        for (index, term) in terms.iter().enumerate() {
            check_square(term.operator(), model.sites(), "variational operator")?;
            if terms[..index]
                .iter()
                .any(|other| other.name() == term.name())
            {
                return Err(EmbeddingError::InvalidValue(
                    "duplicate variational parameter name",
                ));
            }
        }
        Ok(Self { model, terms })
    }

    /// Return the unvaried reference Hubbard model.
    pub const fn model(&self) -> &HubbardModel {
        &self.model
    }

    /// Return variational terms in optimizer order.
    pub fn terms(&self) -> &[OneBodyTerm] {
        &self.terms
    }

    /// Return the effective reference one-body Hamiltonian
    /// `h' = h + sum_i lambda_i O_i`.
    pub fn effective_hopping(&self) -> Result<DenseMatrix> {
        let mut hopping = self.model.hopping().clone();
        for term in &self.terms {
            hopping = matrix_add(&hopping, &matrix_scale(term.operator(), term.value())?)?;
        }
        Ok(hopping)
    }

    /// Clone the system with one variational coefficient displaced.
    pub fn with_parameter(&self, index: usize, value: f64) -> Result<Self> {
        if !value.is_finite() {
            return Err(EmbeddingError::NonFinite("variational parameter"));
        }
        let mut varied = self.clone();
        let length = varied.terms.len();
        let term = varied.terms.get_mut(index).ok_or(EmbeddingError::Length {
            name: "variational parameter index",
            expected: length,
            actual: index.saturating_add(1),
        })?;
        term.value = value;
        Ok(varied)
    }

    /// Clone the system with all coefficients replaced in optimizer order.
    pub fn with_parameters(&self, values: &[f64]) -> Result<Self> {
        if values.len() != self.terms.len() {
            return Err(EmbeddingError::Length {
                name: "variational parameter values",
                expected: self.terms.len(),
                actual: values.len(),
            });
        }
        if !values.iter().all(|value| value.is_finite()) {
            return Err(EmbeddingError::NonFinite("variational parameter values"));
        }
        let mut varied = self.clone();
        for (term, value) in varied.terms.iter_mut().zip(values) {
            term.value = *value;
        }
        Ok(varied)
    }
}

/// Output supplied by an external interacting reference-system solver.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceSolution {
    green: ClusterGreenGrid,
    self_energy: Vec<DenseMatrix>,
    grand_potential: f64,
}

impl ReferenceSolution {
    /// Construct a solution, checking dimensions, length, finiteness, and
    /// retarded self-energy causality.
    pub fn try_new(
        green: ClusterGreenGrid,
        self_energy: Vec<DenseMatrix>,
        grand_potential: f64,
    ) -> Result<Self> {
        if self_energy.len() != green.points().len() {
            return Err(EmbeddingError::Length {
                name: "reference self-energy grid",
                expected: green.points().len(),
                actual: self_energy.len(),
            });
        }
        for matrix in &self_energy {
            check_square(matrix, green.sites(), "reference self-energy")?;
        }
        validate_retarded_causality(&self_energy, "reference self-energy", 1.0e-12)?;
        if !grand_potential.is_finite() {
            return Err(EmbeddingError::NonFinite("reference grand potential"));
        }
        Ok(Self {
            green,
            self_energy,
            grand_potential,
        })
    }

    /// Return the externally solved cluster Green-function grid.
    pub const fn green(&self) -> &ClusterGreenGrid {
        &self.green
    }

    /// Return the retarded self-energy matrices aligned with the Green grid.
    pub fn self_energy(&self) -> &[DenseMatrix] {
        &self.self_energy
    }

    /// Return the reference-system grand potential.
    pub const fn grand_potential(&self) -> f64 {
        self.grand_potential
    }
}

/// Injected interacting reference-system solver used by VCA.
///
/// Implementations may wrap ED, DMRG, QMC, remote services, or stored data.
pub trait ReferenceSolver {
    /// Solve one checked reference system on the implementation's documented
    /// frequency grid.
    fn solve(&mut self, system: &ReferenceSystem) -> Result<ReferenceSolution>;
}

impl<F> ReferenceSolver for F
where
    F: FnMut(&ReferenceSystem) -> Result<ReferenceSolution>,
{
    fn solve(&mut self, system: &ReferenceSystem) -> Result<ReferenceSolution> {
        self(system)
    }
}
