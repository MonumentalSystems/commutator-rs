//! Hamiltonian model and state types.

use crate::{Result, SpinLatticeError, Vec3};

/// Absolute tolerance used to validate unit-length spins.
pub const SPIN_NORM_TOLERANCE: f64 = 1e-10;

/// Linear distance-dependent Heisenberg exchange.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearExchange {
    reference: f64,
    derivative: f64,
}

impl LinearExchange {
    /// Creates `J(r) = reference + derivative * (r - r0)`.
    pub const fn new(reference: f64, derivative: f64) -> Self {
        Self {
            reference,
            derivative,
        }
    }

    /// Returns the exchange at the equilibrium bond length.
    pub const fn reference(self) -> f64 {
        self.reference
    }

    /// Returns the explicit derivative `dJ/dr`.
    pub const fn derivative(self) -> f64 {
        self.derivative
    }

    /// Evaluates the exchange at a distance relative to an equilibrium length.
    pub fn at_distance(self, distance: f64, equilibrium_length: f64) -> f64 {
        self.derivative
            .mul_add(distance - equilibrium_length, self.reference)
    }
}

/// A graph bond with harmonic and magnetoelastic interactions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bond {
    first: usize,
    second: usize,
    equilibrium: Vec3,
    spring_constant: f64,
    exchange: LinearExchange,
}

impl Bond {
    /// Creates a graph bond.
    ///
    /// `equilibrium` points from `first` to `second`; its norm is the spring
    /// rest length and exchange reference distance. Validation occurs when
    /// the bond is added to a [`SpinLatticeModel`].
    pub const fn new(
        first: usize,
        second: usize,
        equilibrium: Vec3,
        spring_constant: f64,
        exchange: LinearExchange,
    ) -> Self {
        Self {
            first,
            second,
            equilibrium,
            spring_constant,
            exchange,
        }
    }

    /// Returns the first atom index.
    pub const fn first(self) -> usize {
        self.first
    }

    /// Returns the second atom index.
    pub const fn second(self) -> usize {
        self.second
    }

    /// Returns the equilibrium vector from the first atom to the second.
    pub const fn equilibrium(self) -> Vec3 {
        self.equilibrium
    }

    /// Returns the harmonic spring constant.
    pub const fn spring_constant(self) -> f64 {
        self.spring_constant
    }

    /// Returns the distance-dependent exchange law.
    pub const fn exchange(self) -> LinearExchange {
        self.exchange
    }
}

/// Dynamical atom displacements, momenta, and unit spins.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinLatticeState {
    pub(crate) displacements: Vec<Vec3>,
    pub(crate) momenta: Vec<Vec3>,
    pub(crate) spins: Vec<Vec3>,
}

impl SpinLatticeState {
    /// Creates and validates a state for `model`.
    pub fn try_new(
        model: &SpinLatticeModel,
        displacements: Vec<Vec3>,
        momenta: Vec<Vec3>,
        spins: Vec<Vec3>,
    ) -> Result<Self> {
        let state = Self {
            displacements,
            momenta,
            spins,
        };
        model.validate_state(&state)?;
        Ok(state)
    }

    /// Creates a state with zero displacements and momenta.
    pub fn stationary(model: &SpinLatticeModel, spins: Vec<Vec3>) -> Result<Self> {
        let zeros = vec![Vec3::ZERO; model.atom_count()];
        Self::try_new(model, zeros.clone(), zeros, spins)
    }

    /// Returns all atom displacements from equilibrium coordinates.
    pub fn displacements(&self) -> &[Vec3] {
        &self.displacements
    }

    /// Returns all atom momenta.
    pub fn momenta(&self) -> &[Vec3] {
        &self.momenta
    }

    /// Returns all unit spin vectors.
    pub fn spins(&self) -> &[Vec3] {
        &self.spins
    }

    /// Sets one finite displacement.
    pub fn set_displacement(&mut self, atom: usize, displacement: Vec3) -> Result<()> {
        self.check_atom(atom)?;
        if !displacement.is_finite() {
            return Err(SpinLatticeError::NonFinite {
                context: "displacement",
                index: atom,
            });
        }
        self.displacements[atom] = displacement;
        Ok(())
    }

    /// Sets one finite momentum.
    pub fn set_momentum(&mut self, atom: usize, momentum: Vec3) -> Result<()> {
        self.check_atom(atom)?;
        if !momentum.is_finite() {
            return Err(SpinLatticeError::NonFinite {
                context: "momentum",
                index: atom,
            });
        }
        self.momenta[atom] = momentum;
        Ok(())
    }

    /// Sets one finite unit spin.
    pub fn set_spin(&mut self, atom: usize, spin: Vec3) -> Result<()> {
        self.check_atom(atom)?;
        validate_spin(atom, spin)?;
        self.spins[atom] = spin;
        Ok(())
    }

    fn check_atom(&self, atom: usize) -> Result<()> {
        if atom < self.spins.len() {
            Ok(())
        } else {
            Err(SpinLatticeError::AtomOutOfBounds {
                atom,
                atom_count: self.spins.len(),
            })
        }
    }
}

/// A checked atom graph and its masses.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinLatticeModel {
    masses: Vec<f64>,
    bonds: Vec<Bond>,
}

impl SpinLatticeModel {
    /// Creates a checked model from atom masses and graph bonds.
    pub fn try_new(masses: Vec<f64>, bonds: Vec<Bond>) -> Result<Self> {
        if masses.is_empty() {
            return Err(SpinLatticeError::EmptySystem);
        }
        for (atom, &mass) in masses.iter().enumerate() {
            if !mass.is_finite() || mass <= 0.0 {
                return Err(SpinLatticeError::InvalidMass { atom });
            }
        }
        for (index, bond) in bonds.iter().enumerate() {
            if bond.first >= masses.len() {
                return Err(SpinLatticeError::AtomOutOfBounds {
                    atom: bond.first,
                    atom_count: masses.len(),
                });
            }
            if bond.second >= masses.len() {
                return Err(SpinLatticeError::AtomOutOfBounds {
                    atom: bond.second,
                    atom_count: masses.len(),
                });
            }
            if bond.first == bond.second {
                return Err(SpinLatticeError::SelfBond {
                    bond: index,
                    atom: bond.first,
                });
            }
            if !bond.equilibrium.is_finite() {
                return Err(SpinLatticeError::NonFinite {
                    context: "bond equilibrium vector",
                    index,
                });
            }
            if bond.equilibrium.norm() == 0.0 {
                return Err(SpinLatticeError::DegenerateEquilibriumBond { bond: index });
            }
            if !bond.spring_constant.is_finite() || bond.spring_constant < 0.0 {
                return Err(SpinLatticeError::InvalidSpringConstant { bond: index });
            }
            if !bond.exchange.reference.is_finite() || !bond.exchange.derivative.is_finite() {
                return Err(SpinLatticeError::NonFinite {
                    context: "bond exchange",
                    index,
                });
            }
        }
        Ok(Self { masses, bonds })
    }

    /// Returns the number of atoms.
    pub fn atom_count(&self) -> usize {
        self.masses.len()
    }

    /// Returns atom masses.
    pub fn masses(&self) -> &[f64] {
        &self.masses
    }

    /// Returns graph bonds.
    pub fn bonds(&self) -> &[Bond] {
        &self.bonds
    }

    pub(crate) fn validate_state(&self, state: &SpinLatticeState) -> Result<()> {
        validate_length(
            "displacements",
            self.atom_count(),
            state.displacements.len(),
        )?;
        validate_length("momenta", self.atom_count(), state.momenta.len())?;
        validate_length("spins", self.atom_count(), state.spins.len())?;
        for (atom, displacement) in state.displacements.iter().enumerate() {
            if !displacement.is_finite() {
                return Err(SpinLatticeError::NonFinite {
                    context: "displacement",
                    index: atom,
                });
            }
        }
        for (atom, momentum) in state.momenta.iter().enumerate() {
            if !momentum.is_finite() {
                return Err(SpinLatticeError::NonFinite {
                    context: "momentum",
                    index: atom,
                });
            }
        }
        for (atom, &spin) in state.spins.iter().enumerate() {
            validate_spin(atom, spin)?;
        }
        Ok(())
    }
}

fn validate_length(context: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(SpinLatticeError::DimensionMismatch {
            context,
            expected,
            actual,
        })
    }
}

fn validate_spin(atom: usize, spin: Vec3) -> Result<()> {
    let norm = spin.norm();
    if !spin.is_finite() || !norm.is_finite() || (norm - 1.0).abs() > SPIN_NORM_TOLERANCE {
        Err(SpinLatticeError::InvalidSpin { atom, norm })
    } else {
        Ok(())
    }
}
