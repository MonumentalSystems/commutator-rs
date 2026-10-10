//! Shared Hamiltonian evaluation.

use crate::{Result, SpinLatticeError, SpinLatticeModel, SpinLatticeState, Vec3};

/// Hamiltonian energy components.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Energy {
    kinetic: f64,
    elastic: f64,
    exchange: f64,
}

impl Energy {
    /// Returns lattice kinetic energy.
    pub const fn kinetic(self) -> f64 {
        self.kinetic
    }

    /// Returns harmonic spring energy.
    pub const fn elastic(self) -> f64 {
        self.elastic
    }

    /// Returns Heisenberg exchange energy.
    pub const fn exchange(self) -> f64 {
        self.exchange
    }

    /// Returns the total Hamiltonian energy.
    pub fn total(self) -> f64 {
        self.kinetic + self.elastic + self.exchange
    }
}

/// Forces, effective spin fields, and energy from one Hamiltonian evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    forces: Vec<Vec3>,
    effective_fields: Vec<Vec3>,
    energy: Energy,
}

impl Evaluation {
    /// Returns lattice forces `-dH/du_i`.
    pub fn forces(&self) -> &[Vec3] {
        &self.forces
    }

    /// Returns spin fields `-dH/ds_i`.
    pub fn effective_fields(&self) -> &[Vec3] {
        &self.effective_fields
    }

    /// Returns Hamiltonian energy components.
    pub const fn energy(&self) -> Energy {
        self.energy
    }
}

impl SpinLatticeModel {
    /// Evaluates energy, lattice forces, and effective spin fields together.
    pub fn evaluate(&self, state: &SpinLatticeState) -> Result<Evaluation> {
        self.validate_state(state)?;
        let mut forces = vec![Vec3::ZERO; self.atom_count()];
        let mut effective_fields = vec![Vec3::ZERO; self.atom_count()];
        let mut energy = Energy::default();

        for (atom, momentum) in state.momenta.iter().enumerate() {
            energy.kinetic += momentum.norm_squared() / (2.0 * self.masses()[atom]);
        }

        for (index, bond) in self.bonds().iter().enumerate() {
            let first = bond.first();
            let second = bond.second();
            let current =
                bond.equilibrium() + state.displacements[second] - state.displacements[first];
            let distance = current.norm();
            if !distance.is_finite() || distance == 0.0 {
                return Err(SpinLatticeError::DegenerateCurrentBond { bond: index });
            }
            let direction = current / distance;
            let equilibrium_length = bond.equilibrium().norm();
            let extension = distance - equilibrium_length;
            let spin_dot = state.spins[first].dot(state.spins[second]);
            let exchange = bond.exchange().at_distance(distance, equilibrium_length);

            energy.elastic += 0.5 * bond.spring_constant() * extension * extension;
            energy.exchange -= exchange * spin_dot;

            // dH/dr combines the harmonic and exchange contributions. Since
            // r points first -> second, F_first = +(dH/dr) r_hat.
            let radial_derivative =
                bond.spring_constant() * extension - bond.exchange().derivative() * spin_dot;
            let first_force = direction * radial_derivative;
            forces[first] += first_force;
            forces[second] -= first_force;

            effective_fields[first] += state.spins[second] * exchange;
            effective_fields[second] += state.spins[first] * exchange;
        }

        Ok(Evaluation {
            forces,
            effective_fields,
            energy,
        })
    }

    /// Returns the mean spin vector.
    pub fn magnetization(&self, state: &SpinLatticeState) -> Result<Vec3> {
        self.validate_state(state)?;
        let sum = state
            .spins
            .iter()
            .copied()
            .fold(Vec3::ZERO, |accumulator, spin| accumulator + spin);
        Ok(sum / self.atom_count() as f64)
    }

    /// Returns the largest absolute deviation of a spin norm from one.
    pub fn max_spin_norm_error(&self, state: &SpinLatticeState) -> Result<f64> {
        self.validate_state(state)?;
        Ok(state
            .spins
            .iter()
            .map(|spin| (spin.norm() - 1.0).abs())
            .fold(0.0_f64, f64::max))
    }
}
