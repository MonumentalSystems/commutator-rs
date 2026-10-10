//! Checked mean-field Kuramoto dynamics on the circle `S¹`.
//!
//! This module implements the uniformly coupled, all-to-all model
//!
//! `d(theta_i)/dt = omega_i + (K/N) * sum_j sin(theta_j - theta_i)`.
//!
//! Phases are stored canonically in `[-pi, pi)`. The implementation uses the
//! mathematically equivalent mean-phasor form, making each Euler step `O(N)`.
//! It does not define graph coupling, noise, adaptive coupling, or a stability
//! guarantee for explicit Euler integration.

use crate::error::{expect_len, finite_scalar, finite_slice};
use crate::{DynamicsError, Result};

const UNDEFINED_MEAN_TOLERANCE: f32 = 8.0 * f32::EPSILON;
const PI_F64: f64 = core::f32::consts::PI as f64;
const TAU_F64: f64 = core::f32::consts::TAU as f64;

/// A finite angle in radians, canonically represented in `[-pi, pi)`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Phase(f32);

impl Phase {
    /// The additive identity phase.
    pub const ZERO: Self = Self(0.0);

    /// Construct a phase from a finite angle in radians.
    ///
    /// Any whole turns are discarded. In particular, both `pi` and `-pi`
    /// use the canonical representation `-pi`.
    ///
    /// # Errors
    ///
    /// Returns [`DynamicsError::NonFinite`] for NaN or either infinity.
    pub fn from_radians(radians: f32) -> Result<Self> {
        finite_scalar("phase", radians)?;
        Self::from_radians_f64(f64::from(radians))
    }

    /// Return the canonical angle in radians.
    pub const fn radians(self) -> f32 {
        self.0
    }

    /// Return the unit-complex embedding `[cos(theta), sin(theta)]`.
    pub fn unit_complex(self) -> [f32; 2] {
        let (sine, cosine) = self.0.sin_cos();
        [cosine, sine]
    }

    /// Return pairwise phase coherence `cos(other - self)` in `[-1, 1]`.
    pub fn coherence(self, other: Self) -> f32 {
        let left = self.unit_complex();
        let right = other.unit_complex();
        (left[0] * right[0] + left[1] * right[1]).clamp(-1.0, 1.0)
    }

    fn from_radians_f64(radians: f64) -> Result<Self> {
        if !radians.is_finite() {
            return Err(DynamicsError::NonFinite("phase"));
        }
        let positive = radians.rem_euclid(TAU_F64);
        let canonical = if positive >= PI_F64 {
            positive - TAU_F64
        } else {
            positive
        };
        let mut canonical = canonical as f32;
        if canonical >= core::f32::consts::PI {
            canonical = -core::f32::consts::PI;
        }
        if canonical == 0.0 {
            canonical = 0.0;
        }
        Ok(Self(canonical))
    }
}

impl TryFrom<f32> for Phase {
    type Error = DynamicsError;

    fn try_from(value: f32) -> Result<Self> {
        Self::from_radians(value)
    }
}

impl From<Phase> for f32 {
    fn from(value: Phase) -> Self {
        value.0
    }
}

/// Circular mean phasor and Kuramoto coherence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrderParameter {
    mean_phasor: [f32; 2],
    coherence: f32,
    mean_phase: Option<Phase>,
}

impl OrderParameter {
    /// Return `[mean(cos(theta)), mean(sin(theta))]`.
    pub const fn mean_phasor(self) -> [f32; 2] {
        self.mean_phasor
    }

    /// Return the phasor magnitude `r` in `[0, 1]`.
    pub const fn coherence(self) -> f32 {
        self.coherence
    }

    /// Return the mean phase when its direction is numerically defined.
    ///
    /// This returns `None` when coherence is no greater than
    /// `8 * f32::EPSILON`. Mean direction is increasingly ill-conditioned as
    /// coherence approaches that threshold.
    pub const fn mean_phase(self) -> Option<Phase> {
        self.mean_phase
    }
}

/// Compute the circular order parameter of a nonempty phase slice.
///
/// Public `f32` unit-complex values are accumulated in `f64` in slice order,
/// then exposed as `f32`. This improves summation accuracy but does not promise
/// bit-identical transcendental results across platforms.
///
/// # Errors
///
/// Returns [`DynamicsError::Empty`] when `phases` is empty.
pub fn order_parameter(phases: &[Phase]) -> Result<OrderParameter> {
    if phases.is_empty() {
        return Err(DynamicsError::Empty("Kuramoto phases"));
    }
    let [mean_cosine, mean_sine] = mean_phasor_f64(phases);
    let coherence = mean_cosine.hypot(mean_sine).clamp(0.0, 1.0) as f32;
    let mean_phase = if coherence <= UNDEFINED_MEAN_TOLERANCE {
        None
    } else {
        Some(Phase::from_radians_f64(mean_sine.atan2(mean_cosine))?)
    };
    Ok(OrderParameter {
        mean_phasor: [mean_cosine as f32, mean_sine as f32],
        coherence,
        mean_phase,
    })
}

/// Uniform all-to-all Kuramoto parameters with synchronous Euler stepping.
///
/// `coupling` has inverse-time units and may be positive (attractive), zero,
/// or negative (repulsive). `time_step` uses the reciprocal units of natural
/// frequency and must be nonnegative.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeanFieldKuramoto {
    coupling: f32,
    time_step: f32,
}

impl MeanFieldKuramoto {
    /// Construct checked mean-field Kuramoto parameters.
    ///
    /// # Errors
    ///
    /// Returns [`DynamicsError::NonFinite`] if either value is non-finite, or
    /// [`DynamicsError::InvalidDomain`] when `time_step` is negative.
    pub fn try_new(coupling: f32, time_step: f32) -> Result<Self> {
        finite_scalar("Kuramoto coupling", coupling)?;
        finite_scalar("Kuramoto time step", time_step)?;
        if time_step < 0.0 {
            return Err(DynamicsError::InvalidDomain("Kuramoto time step"));
        }
        Ok(Self {
            coupling,
            time_step,
        })
    }

    /// Return the signed coupling strength.
    pub const fn coupling(self) -> f32 {
        self.coupling
    }

    /// Return the explicit Euler time step.
    pub const fn time_step(self) -> f32 {
        self.time_step
    }

    /// Advance a complete, uniformly coupled population by one Euler step.
    ///
    /// `natural_frequencies[i]` belongs to `phases[i]` and is measured in
    /// radians per time unit.
    ///
    /// The update is synchronous: every new phase is computed from the same
    /// old-state snapshot. Natural-frequency drift is applied even when the
    /// coupling is zero or the population contains one oscillator. Updated
    /// phases are wrapped to `[-pi, pi)`, so completed rotation counts are not
    /// retained.
    ///
    /// Validation and evaluation are transactional. On error, `phases` is
    /// unchanged. A zero time step is an exact no-op after validation.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty population, a frequency-length mismatch,
    /// a non-finite natural frequency, or a non-finite derived update.
    pub fn step_euler(self, phases: &mut [Phase], natural_frequencies: &[f32]) -> Result<()> {
        if phases.is_empty() {
            return Err(DynamicsError::Empty("Kuramoto phases"));
        }
        expect_len(
            "Kuramoto natural frequencies",
            natural_frequencies.len(),
            phases.len(),
        )?;
        finite_slice("Kuramoto natural frequencies", natural_frequencies)?;
        if self.time_step == 0.0 {
            return Ok(());
        }

        let [mean_cosine, mean_sine] = mean_phasor_f64(phases);
        let coupling = f64::from(self.coupling);
        let time_step = f64::from(self.time_step);
        let mut next = Vec::with_capacity(phases.len());
        for (&phase, &natural_frequency) in phases.iter().zip(natural_frequencies) {
            let [cosine, sine] = phase.unit_complex();
            let cosine = f64::from(cosine);
            let sine = f64::from(sine);
            let interaction = coupling * (mean_sine * cosine - mean_cosine * sine);
            let velocity = f64::from(natural_frequency) + interaction;
            let updated = f64::from(phase.0) + time_step * velocity;
            if !velocity.is_finite() || !updated.is_finite() {
                return Err(DynamicsError::NonFinite("Kuramoto Euler update"));
            }
            next.push(Phase::from_radians_f64(updated)?);
        }
        phases.copy_from_slice(&next);
        Ok(())
    }
}

fn mean_phasor_f64(phases: &[Phase]) -> [f64; 2] {
    let mut cosine_sum = 0.0f64;
    let mut sine_sum = 0.0f64;
    for phase in phases {
        let [cosine, sine] = phase.unit_complex();
        cosine_sum += f64::from(cosine);
        sine_sum += f64::from(sine);
    }
    let count = phases.len() as f64;
    [cosine_sum / count, sine_sum / count]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phase(radians: f32) -> Phase {
        Phase::from_radians(radians).unwrap()
    }

    fn assert_phase_close(actual: Phase, expected: Phase, tolerance: f32) {
        let distance = phase(actual.radians() - expected.radians()).radians().abs();
        assert!(
            distance <= tolerance,
            "expected {:?}, found {:?}",
            expected,
            actual
        );
    }

    #[test]
    fn phase_has_a_stable_canonical_branch() {
        let pi = core::f32::consts::PI;
        let tau = core::f32::consts::TAU;
        assert_eq!(phase(0.0), Phase::ZERO);
        assert_eq!(phase(tau), Phase::ZERO);
        assert_eq!(phase(-tau), Phase::ZERO);
        assert_eq!(phase(-tau).radians().to_bits(), 0.0_f32.to_bits());
        assert_eq!(phase(pi).radians(), -pi);
        assert_eq!(phase(-pi).radians(), -pi);
        assert_eq!(
            Phase::from_radians_f64(PI_F64 - 1.0e-12).unwrap().radians(),
            -pi
        );
        assert!((phase(tau + 0.25).radians() - 0.25).abs() < 1.0e-6);
        assert!(Phase::from_radians(f32::NAN).is_err());
        assert!(Phase::from_radians(f32::INFINITY).is_err());
        assert!(Phase::from_radians(f32::NEG_INFINITY).is_err());
    }

    #[test]
    fn unit_complex_and_pair_coherence_are_geometric() {
        let quarter = phase(core::f32::consts::FRAC_PI_2);
        let [cosine, sine] = quarter.unit_complex();
        assert!(cosine.abs() < 1.0e-6);
        assert!((sine - 1.0).abs() < 1.0e-6);
        assert!((cosine * cosine + sine * sine - 1.0).abs() < 1.0e-6);
        assert!((Phase::ZERO.coherence(Phase::ZERO) - 1.0).abs() < 1.0e-6);
        assert!((Phase::ZERO.coherence(phase(core::f32::consts::PI)) + 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn order_parameter_covers_aligned_balanced_and_singleton_states() {
        let aligned = order_parameter(&[phase(0.7); 4]).unwrap();
        assert!((aligned.coherence() - 1.0).abs() < 1.0e-6);
        assert_phase_close(aligned.mean_phase().unwrap(), phase(0.7), 1.0e-6);

        let balanced = order_parameter(&[
            Phase::ZERO,
            phase(core::f32::consts::FRAC_PI_2),
            phase(core::f32::consts::PI),
            phase(3.0 * core::f32::consts::FRAC_PI_2),
        ])
        .unwrap();
        assert!(balanced.coherence() <= UNDEFINED_MEAN_TOLERANCE);
        assert_eq!(balanced.mean_phase(), None);

        let singleton = order_parameter(&[phase(-2.0)]).unwrap();
        assert!((singleton.coherence() - 1.0).abs() < 1.0e-6);
        assert_phase_close(singleton.mean_phase().unwrap(), phase(-2.0), 1.0e-6);
        assert!(order_parameter(&[]).is_err());
    }

    #[test]
    fn order_parameter_is_rotation_and_turn_invariant() {
        let base = [phase(-0.7), phase(0.2), phase(1.1)];
        let shifted = [phase(-0.3), phase(0.6), phase(1.5)];
        let turns = [
            phase(-0.7 + core::f32::consts::TAU),
            phase(0.2 - core::f32::consts::TAU),
            phase(1.1 + 2.0 * core::f32::consts::TAU),
        ];
        let base_order = order_parameter(&base).unwrap();
        let shifted_order = order_parameter(&shifted).unwrap();
        let turns_order = order_parameter(&turns).unwrap();
        assert!((base_order.coherence() - shifted_order.coherence()).abs() < 1.0e-6);
        assert!((base_order.coherence() - turns_order.coherence()).abs() < 1.0e-6);
        assert_phase_close(
            shifted_order.mean_phase().unwrap(),
            phase(base_order.mean_phase().unwrap().radians() + 0.4),
            1.0e-6,
        );
    }

    #[test]
    fn two_oscillator_step_freezes_k_over_n_and_synchronous_semantics() {
        let model = MeanFieldKuramoto::try_new(2.0, 0.1).unwrap();
        let mut phases = [Phase::ZERO, phase(core::f32::consts::FRAC_PI_2)];
        model.step_euler(&mut phases, &[0.0, 0.0]).unwrap();
        assert_phase_close(phases[0], phase(0.1), 1.0e-6);
        assert_phase_close(phases[1], phase(core::f32::consts::FRAC_PI_2 - 0.1), 1.0e-6);
    }

    #[test]
    fn zero_coupling_and_singletons_still_follow_natural_frequency() {
        let uncoupled = MeanFieldKuramoto::try_new(0.0, 0.25).unwrap();
        let mut phases = [phase(0.1), phase(-0.3)];
        uncoupled.step_euler(&mut phases, &[2.0, -1.0]).unwrap();
        assert_phase_close(phases[0], phase(0.6), 1.0e-6);
        assert_phase_close(phases[1], phase(-0.55), 1.0e-6);

        let coupled = MeanFieldKuramoto::try_new(100.0, 0.25).unwrap();
        let mut singleton = [phase(0.4)];
        coupled.step_euler(&mut singleton, &[0.8]).unwrap();
        assert_phase_close(singleton[0], phase(0.6), 1.0e-6);
    }

    #[test]
    fn branch_crossings_wrap_without_losing_circle_state() {
        let model = MeanFieldKuramoto::try_new(0.0, 1.0).unwrap();
        let mut phases = [phase(core::f32::consts::PI - 0.05)];
        model.step_euler(&mut phases, &[0.1]).unwrap();
        assert_phase_close(phases[0], phase(-core::f32::consts::PI + 0.05), 1.0e-6);
    }

    #[test]
    fn attractive_and_repulsive_coupling_move_coherence_oppositely() {
        let initial = [phase(-0.5), phase(0.5)];
        let initial_r = order_parameter(&initial).unwrap().coherence();
        let mut attractive = initial;
        let mut repulsive = initial;
        MeanFieldKuramoto::try_new(1.0, 0.1)
            .unwrap()
            .step_euler(&mut attractive, &[0.0; 2])
            .unwrap();
        MeanFieldKuramoto::try_new(-1.0, 0.1)
            .unwrap()
            .step_euler(&mut repulsive, &[0.0; 2])
            .unwrap();
        assert!(order_parameter(&attractive).unwrap().coherence() > initial_r);
        assert!(order_parameter(&repulsive).unwrap().coherence() < initial_r);
    }

    #[test]
    fn mean_field_step_matches_direct_complete_graph_reference() {
        let model = MeanFieldKuramoto::try_new(-0.7, 0.03).unwrap();
        let old = [phase(-2.1), phase(-0.2), phase(0.8), phase(2.4)];
        let frequencies = [-0.3_f32, 0.0, 0.2, 0.5];
        let mut actual = old;
        model.step_euler(&mut actual, &frequencies).unwrap();

        for i in 0..old.len() {
            let theta_i = f64::from(old[i].radians());
            let coupling_sum: f64 = old
                .iter()
                .map(|phase| (f64::from(phase.radians()) - theta_i).sin())
                .sum();
            let velocity = f64::from(frequencies[i]) - 0.7 * coupling_sum / old.len() as f64;
            let expected = Phase::from_radians_f64(theta_i + 0.03 * velocity).unwrap();
            assert_phase_close(actual[i], expected, 1.0e-6);
        }
    }

    #[test]
    fn step_preserves_complete_graph_symmetries() {
        let model = MeanFieldKuramoto::try_new(0.7, 0.01).unwrap();
        let old = [phase(-1.1), phase(-0.2), phase(0.4), phase(1.3)];
        let frequencies = [-0.2_f32, 0.1, 0.3, -0.1];

        let mut base = old;
        model.step_euler(&mut base, &frequencies).unwrap();

        let shift = 0.6;
        let mut shifted = old.map(|value| phase(value.radians() + shift));
        model.step_euler(&mut shifted, &frequencies).unwrap();
        for (actual, unshifted) in shifted.into_iter().zip(base) {
            assert_phase_close(actual, phase(unshifted.radians() + shift), 2.0e-6);
        }

        let permutation = [2usize, 0, 3, 1];
        let mut permuted = permutation.map(|index| old[index]);
        let permuted_frequencies = permutation.map(|index| frequencies[index]);
        model
            .step_euler(&mut permuted, &permuted_frequencies)
            .unwrap();
        for (position, index) in permutation.into_iter().enumerate() {
            assert_phase_close(permuted[position], base[index], 2.0e-6);
        }

        let interaction_increment_sum: f32 = base
            .iter()
            .zip(old)
            .zip(frequencies)
            .map(|((&new, old), omega)| (new.radians() - old.radians()) - model.time_step() * omega)
            .sum();
        assert!(interaction_increment_sum.abs() < 2.0e-6);
    }

    #[test]
    fn equal_frequency_population_converges_for_a_stable_step() {
        let model = MeanFieldKuramoto::try_new(4.0, 0.02).unwrap();
        let mut phases = [phase(-1.0), phase(-0.3), phase(0.4), phase(1.2)];
        for _ in 0..200 {
            model.step_euler(&mut phases, &[0.3; 4]).unwrap();
        }
        assert!(order_parameter(&phases).unwrap().coherence() > 0.999);
    }

    #[test]
    fn validation_is_transactional_and_zero_step_is_exact() {
        assert!(MeanFieldKuramoto::try_new(f32::NAN, 0.1).is_err());
        assert!(MeanFieldKuramoto::try_new(1.0, -0.1).is_err());
        let model = MeanFieldKuramoto::try_new(1.0, 0.1).unwrap();
        let mut phases = [phase(0.2), phase(0.7)];
        let before = phases;
        assert!(model.step_euler(&mut phases, &[0.0]).is_err());
        assert_eq!(phases, before);
        assert!(model
            .step_euler(&mut phases, &[0.0, f32::INFINITY])
            .is_err());
        assert_eq!(phases, before);
        assert!(model.step_euler(&mut [], &[]).is_err());

        let zero_step = MeanFieldKuramoto::try_new(1.0, 0.0).unwrap();
        zero_step.step_euler(&mut phases, &[10.0, -10.0]).unwrap();
        assert_eq!(phases, before);
    }
}
