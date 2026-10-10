//! Lumped Josephson-junction energy and current-phase relation.

use crate::{DynamicsError, Result};

/// A sinusoidal Josephson junction with an optional intrinsic phase offset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JosephsonJunction {
    coupling_energy: f64,
    charge_over_action: f64,
    phase_offset: f64,
}

impl JosephsonJunction {
    /// Creates a junction with energy `-E_J cos(delta - phase_offset)`.
    ///
    /// `charge_over_action` converts `-dE/dA` to current. In SI it is
    /// `2e/hbar`; set it to one for dimensionless simulation units.
    pub fn try_new(
        coupling_energy: f64,
        charge_over_action: f64,
        phase_offset: f64,
    ) -> Result<Self> {
        if !coupling_energy.is_finite() || coupling_energy < 0.0 {
            return Err(DynamicsError::InvalidNonnegativeParameter {
                parameter: "Josephson coupling energy",
            });
        }
        if !charge_over_action.is_finite() || charge_over_action <= 0.0 {
            return Err(DynamicsError::InvalidPositiveParameter {
                parameter: "charge_over_action",
            });
        }
        if !phase_offset.is_finite() {
            return Err(DynamicsError::NonFinite {
                context: "Josephson phase offset",
                index: 0,
            });
        }
        Ok(Self {
            coupling_energy,
            charge_over_action,
            phase_offset,
        })
    }

    /// Returns the junction coupling energy.
    pub fn coupling_energy(&self) -> f64 {
        self.coupling_energy
    }

    /// Returns the phase offset of a zero- or pi-like junction.
    pub fn phase_offset(&self) -> f64 {
        self.phase_offset
    }

    /// Evaluates `-E_J cos(delta - phase_offset)`.
    pub fn energy(&self, gauge_invariant_phase_difference: f64) -> Result<f64> {
        self.check_phase(gauge_invariant_phase_difference)?;
        Ok(-self.coupling_energy * (gauge_invariant_phase_difference - self.phase_offset).cos())
    }

    /// Evaluates `I_c sin(delta - phase_offset)`, where
    /// `I_c = charge_over_action * E_J`.
    pub fn current(&self, gauge_invariant_phase_difference: f64) -> Result<f64> {
        self.check_phase(gauge_invariant_phase_difference)?;
        Ok(self.critical_current() * (gauge_invariant_phase_difference - self.phase_offset).sin())
    }

    /// Returns `charge_over_action * E_J`.
    pub fn critical_current(&self) -> f64 {
        self.charge_over_action * self.coupling_energy
    }

    fn check_phase(&self, phase: f64) -> Result<()> {
        if !phase.is_finite() {
            return Err(DynamicsError::NonFinite {
                context: "Josephson phase difference",
                index: 0,
            });
        }
        Ok(())
    }
}
