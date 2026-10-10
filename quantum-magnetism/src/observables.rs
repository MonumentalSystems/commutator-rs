use crate::model::spin_sign;
use crate::{Complex64, MagnetismError, Result, SpinModel};

/// Cartesian spin component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpinAxis {
    /// `S^x = sigma_x / 2`.
    X,
    /// `S^y = sigma_y / 2`.
    Y,
    /// `S^z = sigma_z / 2`.
    Z,
}

fn inner(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left.conj() * *right)
        .sum()
}

fn spin_action(axis: SpinAxis, basis: usize, site: usize) -> (usize, Complex64) {
    match axis {
        SpinAxis::X => (basis ^ (1usize << site), Complex64::from(0.5)),
        SpinAxis::Y => (
            basis ^ (1usize << site),
            Complex64::new(0.0, -0.5 * spin_sign(basis, site)),
        ),
        SpinAxis::Z => (basis, Complex64::from(0.5 * spin_sign(basis, site))),
    }
}

fn ensure_site(model: &SpinModel, site: usize) -> Result<()> {
    if site < model.spins() {
        Ok(())
    } else {
        Err(MagnetismError::SiteOutOfBounds {
            site,
            spins: model.spins(),
        })
    }
}

impl SpinModel {
    /// Return the normalized expectation value `<H>`.
    pub fn energy(&self, state: &[Complex64]) -> Result<f64> {
        let norm = self.validate_state("state", state)?;
        let applied = self.applied(state)?;
        let value = inner(state, &applied) / norm;
        let scale = 1.0 + value.re.abs();
        if value.im.abs() > 256.0 * f64::EPSILON * scale {
            return Err(MagnetismError::NumericalFailure(
                "Hamiltonian expectation was not real",
            ));
        }
        Ok(value.re)
    }

    /// Return `<S_i^axis>` for a normalized or unnormalized pure state.
    pub fn local_magnetization(
        &self,
        state: &[Complex64],
        site: usize,
        axis: SpinAxis,
    ) -> Result<f64> {
        ensure_site(self, site)?;
        let norm = self.validate_state("state", state)?;
        let mut value = Complex64::new(0.0, 0.0);
        for (basis, amplitude) in state.iter().copied().enumerate() {
            let (target, coefficient) = spin_action(axis, basis, site);
            value += state[target].conj() * coefficient * amplitude;
        }
        value /= norm;
        if value.im.abs() > 256.0 * f64::EPSILON * (1.0 + value.re.abs()) {
            return Err(MagnetismError::NumericalFailure(
                "spin expectation was not real",
            ));
        }
        Ok(value.re)
    }

    /// Return the total magnetization `[<sum Sx>, <sum Sy>, <sum Sz>]`.
    pub fn total_magnetization(&self, state: &[Complex64]) -> Result<[f64; 3]> {
        self.validate_state("state", state)?;
        let mut total = [0.0; 3];
        for site in 0..self.spins() {
            total[0] += self.local_magnetization(state, site, SpinAxis::X)?;
            total[1] += self.local_magnetization(state, site, SpinAxis::Y)?;
            total[2] += self.local_magnetization(state, site, SpinAxis::Z)?;
        }
        Ok(total)
    }

    /// Return the ordered correlation `<S_left^a S_right^b>`.
    ///
    /// The right operator acts first. The result may be complex when both
    /// operators act on the same site with different axes.
    pub fn spin_correlation(
        &self,
        state: &[Complex64],
        left: usize,
        left_axis: SpinAxis,
        right: usize,
        right_axis: SpinAxis,
    ) -> Result<Complex64> {
        ensure_site(self, left)?;
        ensure_site(self, right)?;
        let norm = self.validate_state("state", state)?;
        let mut value = Complex64::new(0.0, 0.0);
        for (basis, amplitude) in state.iter().copied().enumerate() {
            let (middle, right_coefficient) = spin_action(right_axis, basis, right);
            let (target, left_coefficient) = spin_action(left_axis, middle, left);
            value += state[target].conj() * left_coefficient * right_coefficient * amplitude;
        }
        Ok(value / norm)
    }

    /// Return the equal-time static structure factor `S_axis(q)`.
    ///
    /// `positions` supplies one three-dimensional coordinate per site and
    /// `wavevector` is in reciprocal coordinate units. The implementation
    /// evaluates `||sum_j exp(i q·r_j) S_j^axis |state>||^2 / (N ||state||^2)`,
    /// which is nonnegative up to floating-point roundoff.
    pub fn static_structure_factor(
        &self,
        state: &[Complex64],
        positions: &[[f64; 3]],
        wavevector: [f64; 3],
        axis: SpinAxis,
    ) -> Result<f64> {
        let norm = self.validate_state("state", state)?;
        if positions.len() != self.spins() {
            return Err(MagnetismError::Shape {
                name: "site positions",
                expected: self.spins(),
                actual: positions.len(),
            });
        }
        if !positions
            .iter()
            .flatten()
            .chain(&wavevector)
            .all(|value| value.is_finite())
        {
            return Err(MagnetismError::NonFinite("site positions and wavevector"));
        }

        let mut transformed = vec![Complex64::new(0.0, 0.0); self.hilbert_dimension()];
        for (site, position) in positions.iter().enumerate() {
            let phase = position
                .iter()
                .zip(wavevector)
                .map(|(position, wavevector)| position * wavevector)
                .sum::<f64>();
            let phase = Complex64::new(phase.cos(), phase.sin());
            for (basis, amplitude) in state.iter().copied().enumerate() {
                let (target, coefficient) = spin_action(axis, basis, site);
                transformed[target] += phase * coefficient * amplitude;
            }
        }
        let spectral_weight: f64 = transformed.iter().map(|value| value.norm_sqr()).sum();
        let result = spectral_weight / (self.spins() as f64 * norm);
        if result.is_finite() {
            Ok(result.max(0.0))
        } else {
            Err(MagnetismError::NumericalFailure(
                "structure factor overflowed",
            ))
        }
    }
}
