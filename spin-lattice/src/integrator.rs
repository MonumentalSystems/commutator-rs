//! Deterministic lattice and spin integration steps.

use crate::{Result, SpinLatticeError, SpinLatticeModel, SpinLatticeState};

impl SpinLatticeModel {
    /// Advances lattice displacements and momenta by velocity Verlet.
    ///
    /// Spins remain fixed during this substep. The update is transactional:
    /// if the proposed state collapses a bond or becomes non-finite, `state`
    /// is left unchanged.
    pub fn step_velocity_verlet(&self, state: &mut SpinLatticeState, time_step: f64) -> Result<()> {
        validate_time_step(time_step)?;
        let initial = self.evaluate(state)?;
        let mut candidate = state.clone();
        for atom in 0..self.atom_count() {
            candidate.momenta[atom] += initial.forces()[atom] * (0.5 * time_step);
            candidate.displacements[atom] +=
                candidate.momenta[atom] * (time_step / self.masses()[atom]);
        }
        let final_evaluation = self.evaluate(&candidate)?;
        for atom in 0..self.atom_count() {
            candidate.momenta[atom] += final_evaluation.forces()[atom] * (0.5 * time_step);
        }
        self.validate_state(&candidate)?;
        *state = candidate;
        Ok(())
    }

    /// Advances every spin by exact precession in its frozen effective field.
    ///
    /// This integrates `ds_i/dt = gamma B_i x s_i`, with all fields evaluated
    /// before any spin is changed. Rodrigues rotations and final roundoff
    /// normalization preserve each spin norm.
    pub fn step_spin_precession(
        &self,
        state: &mut SpinLatticeState,
        time_step: f64,
        gamma: f64,
    ) -> Result<()> {
        validate_time_step(time_step)?;
        if !gamma.is_finite() {
            return Err(SpinLatticeError::NonFinite {
                context: "gyromagnetic ratio",
                index: 0,
            });
        }
        let evaluation = self.evaluate(state)?;
        let mut candidate = state.clone();
        for atom in 0..self.atom_count() {
            let field = evaluation.effective_fields()[atom];
            let magnitude = field.norm();
            if magnitude == 0.0 {
                continue;
            }
            let angle = gamma * magnitude * time_step;
            if !angle.is_finite() {
                return Err(SpinLatticeError::RotationOverflow { atom });
            }
            let axis = field / magnitude;
            let spin = state.spins[atom];
            let (sine, cosine) = angle.sin_cos();
            let rotated =
                spin * cosine + axis.cross(spin) * sine + axis * (axis.dot(spin) * (1.0 - cosine));
            candidate.spins[atom] = rotated.normalized().ok_or(SpinLatticeError::InvalidSpin {
                atom,
                norm: rotated.norm(),
            })?;
        }
        self.validate_state(&candidate)?;
        *state = candidate;
        Ok(())
    }

    /// Advances coupled dynamics with spin/lattice/spin Strang ordering.
    ///
    /// A half spin-precession step surrounds one full velocity-Verlet lattice
    /// step. The complete operation is transactional.
    pub fn step_coupled(
        &self,
        state: &mut SpinLatticeState,
        time_step: f64,
        gamma: f64,
    ) -> Result<()> {
        validate_time_step(time_step)?;
        let mut candidate = state.clone();
        self.step_spin_precession(&mut candidate, 0.5 * time_step, gamma)?;
        self.step_velocity_verlet(&mut candidate, time_step)?;
        self.step_spin_precession(&mut candidate, 0.5 * time_step, gamma)?;
        *state = candidate;
        Ok(())
    }
}

fn validate_time_step(time_step: f64) -> Result<()> {
    if time_step.is_finite() && time_step > 0.0 {
        Ok(())
    } else {
        Err(SpinLatticeError::InvalidTimeStep)
    }
}
