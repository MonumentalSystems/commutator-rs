//! Deterministic dissipative TDGL integration.

use crate::{DynamicsError, OrderParameter, Result, TdglModel};

/// Diagnostics for one accepted TDGL step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepReport {
    /// Accepted time step after backtracking.
    pub dt: f64,
    /// Free energy before the step.
    pub energy_before: f64,
    /// Free energy after the step.
    pub energy_after: f64,
    /// Number of rejected larger trial steps.
    pub backtracks: usize,
    /// Largest complex gradient magnitude before the step.
    pub max_gradient: f64,
}

/// Energy-monotone reference integrator for dissipative TDGL flow.
///
/// A trial applies forward Euler to `d psi_i/dt = -Gamma_i dF/dpsi_i*`.
/// The step is halved until its free energy is non-increasing within the
/// configured roundoff tolerance. A failed step is transactional: the caller's
/// state is not modified.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TdglIntegrator {
    initial_dt: f64,
    minimum_dt: f64,
    max_backtracks: usize,
    energy_tolerance: f64,
}

impl TdglIntegrator {
    /// Creates a checked backtracking integrator.
    ///
    /// `energy_tolerance` is a nonnegative relative allowance used only for
    /// floating-point roundoff: accepted energy is at most
    /// `F_old + tolerance * max(1, |F_old|)`.
    pub fn try_new(
        initial_dt: f64,
        minimum_dt: f64,
        max_backtracks: usize,
        energy_tolerance: f64,
    ) -> Result<Self> {
        for (parameter, value) in [("initial_dt", initial_dt), ("minimum_dt", minimum_dt)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(DynamicsError::InvalidPositiveParameter { parameter });
            }
        }
        if minimum_dt > initial_dt {
            return Err(DynamicsError::InvalidPositiveParameter {
                parameter: "minimum_dt (must not exceed initial_dt)",
            });
        }
        if !energy_tolerance.is_finite() || energy_tolerance < 0.0 {
            return Err(DynamicsError::InvalidNonnegativeParameter {
                parameter: "energy_tolerance",
            });
        }
        Ok(Self {
            initial_dt,
            minimum_dt,
            max_backtracks,
            energy_tolerance,
        })
    }

    /// Creates conservative defaults suitable for a reference calculation.
    pub fn reference(initial_dt: f64) -> Result<Self> {
        Self::try_new(
            initial_dt,
            initial_dt * 2.0_f64.powi(-30),
            30,
            16.0 * f64::EPSILON,
        )
    }

    /// Advances one accepted step, backtracking when required.
    pub fn step(&self, model: &TdglModel, state: &mut OrderParameter) -> Result<StepReport> {
        let energy_before = model.free_energy(state)?;
        let gradient = model.gradient(state)?;
        let max_gradient = gradient
            .iter()
            .map(|value| value.norm())
            .fold(0.0, f64::max);
        let threshold = energy_before + self.energy_tolerance * energy_before.abs().max(1.0);
        let mut dt = self.initial_dt;
        let mut rejected = 0;

        loop {
            let trial_values = state
                .as_slice()
                .iter()
                .zip(&gradient)
                .zip(model.sites())
                .map(|((&value, &derivative), site)| value - dt * site.mobility() * derivative)
                .collect::<Vec<_>>();

            if let Ok(trial) = OrderParameter::try_new(model, trial_values) {
                if let Ok(energy_after) = model.free_energy(&trial) {
                    if energy_after <= threshold {
                        state.replace(trial.as_slice().to_vec());
                        return Ok(StepReport {
                            dt,
                            energy_before,
                            energy_after,
                            backtracks: rejected,
                            max_gradient,
                        });
                    }
                }
            }

            if rejected == self.max_backtracks || dt * 0.5 < self.minimum_dt {
                return Err(DynamicsError::NoDescentStep {
                    attempted_dt: dt,
                    backtracks: rejected,
                });
            }
            dt *= 0.5;
            rejected += 1;
        }
    }
}
