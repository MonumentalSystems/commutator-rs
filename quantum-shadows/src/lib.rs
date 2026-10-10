//! Local random-Pauli classical-shadow estimators.
//!
//! Measurement generation is intentionally external: callers retain control
//! of hardware, random seeds, and experimental provenance. This crate checks
//! the resulting records and supplies deterministic statistical reductions.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use core::fmt;

/// Largest Pauli weight whose inverse-channel factor `3^weight` is finite in
/// `f64`.
pub const MAX_FINITE_PAULI_WEIGHT: usize = 646;

/// A Pauli factor in a tensor-product observable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pauli {
    /// Identity.
    I,
    /// Pauli X.
    X,
    /// Pauli Y.
    Y,
    /// Pauli Z.
    Z,
}

/// A single-qubit measurement basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MeasurementBasis {
    /// Pauli-X eigenbasis.
    X,
    /// Pauli-Y eigenbasis.
    Y,
    /// Pauli-Z eigenbasis.
    Z,
}

impl MeasurementBasis {
    fn matches(self, pauli: Pauli) -> bool {
        matches!(
            (self, pauli),
            (Self::X, Pauli::X) | (Self::Y, Pauli::Y) | (Self::Z, Pauli::Z)
        )
    }
}

/// Errors returned by checked shadow estimators.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShadowError {
    /// At least one qubit is required.
    ZeroQubits,
    /// A vector has the wrong number of qubit entries.
    QubitShape {
        /// Expected number of entries.
        expected: usize,
        /// Actual number of entries.
        actual: usize,
    },
    /// A measurement outcome was not `-1` or `+1`.
    InvalidOutcome {
        /// Qubit containing the invalid outcome.
        qubit: usize,
        /// Supplied value.
        value: i8,
    },
    /// No measurement snapshots were supplied.
    EmptyDataset,
    /// A requested group count is zero or exceeds the sample count.
    InvalidGroupCount {
        /// Requested number of groups.
        groups: usize,
        /// Available number of samples.
        samples: usize,
    },
    /// Confidence failure probability must lie strictly between zero and one.
    InvalidFailureProbability,
    /// An observable coefficient was NaN or infinite.
    NonFiniteCoefficient,
    /// At least one observable term is required.
    EmptyObservable,
    /// A Pauli string's inverse-channel factor would overflow `f64`.
    PauliWeightOverflow {
        /// Requested non-identity Pauli weight.
        weight: usize,
        /// Largest weight with a finite `3^weight` factor.
        maximum: usize,
    },
    /// A finite-input statistical reduction overflowed or became non-finite.
    NumericalOverflow {
        /// Operation in which the non-finite value arose.
        context: &'static str,
    },
}

impl fmt::Display for ShadowError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ShadowError {}

/// Result type for shadow estimation.
pub type Result<T> = core::result::Result<T, ShadowError>;

/// A checked tensor product of Pauli operators.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PauliString {
    factors: Vec<Pauli>,
    weight: usize,
}

impl PauliString {
    /// Constructs a nonempty Pauli string.
    pub fn try_new(factors: Vec<Pauli>) -> Result<Self> {
        if factors.is_empty() {
            return Err(ShadowError::ZeroQubits);
        }
        let weight = factors.iter().filter(|&&factor| factor != Pauli::I).count();
        if weight > MAX_FINITE_PAULI_WEIGHT {
            return Err(ShadowError::PauliWeightOverflow {
                weight,
                maximum: MAX_FINITE_PAULI_WEIGHT,
            });
        }
        Ok(Self { factors, weight })
    }

    /// Returns the ordered local factors.
    #[must_use]
    pub fn factors(&self) -> &[Pauli] {
        &self.factors
    }

    /// Returns the number of non-identity factors.
    #[must_use]
    pub const fn weight(&self) -> usize {
        self.weight
    }

    fn bound(&self) -> Result<f64> {
        checked_finite(
            3.0_f64.powi(self.weight as i32),
            "Pauli inverse-channel bound",
        )
    }
}

/// One simultaneous local-Pauli measurement record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    bases: Vec<MeasurementBasis>,
    outcomes: Vec<i8>,
}

impl Snapshot {
    /// Constructs a checked snapshot with outcomes encoded as `-1` or `+1`.
    pub fn try_new(bases: Vec<MeasurementBasis>, outcomes: Vec<i8>) -> Result<Self> {
        if bases.is_empty() {
            return Err(ShadowError::ZeroQubits);
        }
        if outcomes.len() != bases.len() {
            return Err(ShadowError::QubitShape {
                expected: bases.len(),
                actual: outcomes.len(),
            });
        }
        for (qubit, &value) in outcomes.iter().enumerate() {
            if value != -1 && value != 1 {
                return Err(ShadowError::InvalidOutcome { qubit, value });
            }
        }
        Ok(Self { bases, outcomes })
    }

    /// Returns the measurement bases in qubit order.
    #[must_use]
    pub fn bases(&self) -> &[MeasurementBasis] {
        &self.bases
    }

    /// Returns the `-1`/`+1` outcomes in qubit order.
    #[must_use]
    pub fn outcomes(&self) -> &[i8] {
        &self.outcomes
    }
}

/// A linear combination of Pauli strings.
#[derive(Clone, Debug, PartialEq)]
pub struct PauliObservable {
    terms: Vec<(f64, PauliString)>,
    qubits: usize,
}

impl PauliObservable {
    /// Constructs a checked observable. Terms with zero coefficients are kept
    /// so serialized experiment definitions can preserve their exact layout.
    pub fn try_new(terms: Vec<(f64, PauliString)>) -> Result<Self> {
        let Some((_, first)) = terms.first() else {
            return Err(ShadowError::EmptyObservable);
        };
        let qubits = first.factors.len();
        for (coefficient, string) in &terms {
            if !coefficient.is_finite() {
                return Err(ShadowError::NonFiniteCoefficient);
            }
            if string.factors.len() != qubits {
                return Err(ShadowError::QubitShape {
                    expected: qubits,
                    actual: string.factors.len(),
                });
            }
        }
        Ok(Self { terms, qubits })
    }

    /// Returns the terms in stable input order.
    #[must_use]
    pub fn terms(&self) -> &[(f64, PauliString)] {
        &self.terms
    }

    fn bound(&self) -> Result<f64> {
        let mut bound = 0.0;
        for (coefficient, string) in &self.terms {
            let term = checked_mul(
                coefficient.abs(),
                string.bound()?,
                "linear-observable Hoeffding bound",
            )?;
            bound = checked_add(bound, term, "linear-observable Hoeffding bound")?;
        }
        Ok(bound)
    }
}

/// A validated collection of local-Pauli measurement records.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShadowDataset {
    qubits: usize,
    snapshots: Vec<Snapshot>,
}

impl ShadowDataset {
    /// Constructs a checked dataset.
    pub fn try_new(qubits: usize, snapshots: Vec<Snapshot>) -> Result<Self> {
        if qubits == 0 {
            return Err(ShadowError::ZeroQubits);
        }
        if snapshots.is_empty() {
            return Err(ShadowError::EmptyDataset);
        }
        for snapshot in &snapshots {
            if snapshot.bases.len() != qubits {
                return Err(ShadowError::QubitShape {
                    expected: qubits,
                    actual: snapshot.bases.len(),
                });
            }
        }
        Ok(Self { qubits, snapshots })
    }

    /// Returns the number of qubits.
    #[must_use]
    pub const fn qubits(&self) -> usize {
        self.qubits
    }

    /// Returns the number of measurement records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Returns whether this dataset is empty. Checked datasets are never empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Returns all validated snapshots.
    #[must_use]
    pub fn snapshots(&self) -> &[Snapshot] {
        &self.snapshots
    }

    /// Returns one inverse-channel single-snapshot estimator.
    pub fn snapshot_estimate(&self, index: usize, observable: &PauliString) -> Result<f64> {
        self.check_string(observable)?;
        let snapshot = self.snapshots.get(index).ok_or(ShadowError::QubitShape {
            expected: self.snapshots.len(),
            actual: index.saturating_add(1),
        })?;
        string_snapshot_estimate(snapshot, observable)
    }

    /// Returns the ordinary sample-mean estimate of a Pauli string.
    ///
    /// Returns [`ShadowError::NumericalOverflow`] if finite snapshot estimates
    /// cannot be accumulated into a finite `f64` result.
    pub fn mean(&self, observable: &PauliString) -> Result<f64> {
        self.check_string(observable)?;
        let mut sum = 0.0;
        for snapshot in &self.snapshots {
            sum = checked_add(
                sum,
                string_snapshot_estimate(snapshot, observable)?,
                "Pauli sample mean",
            )?;
        }
        checked_finite(sum / self.snapshots.len() as f64, "Pauli sample mean")
    }

    /// Returns the estimated standard error of the sample mean.
    ///
    /// A one-snapshot dataset has no sample variance and returns `None`.
    pub fn standard_error(&self, observable: &PauliString) -> Result<Option<f64>> {
        self.check_string(observable)?;
        if self.snapshots.len() < 2 {
            return Ok(None);
        }
        let mean = self.mean(observable)?;
        let mut squared_deviations = 0.0;
        for snapshot in &self.snapshots {
            let residual = checked_finite(
                string_snapshot_estimate(snapshot, observable)? - mean,
                "Pauli standard error",
            )?;
            let square = checked_mul(residual, residual, "Pauli standard error")?;
            squared_deviations = checked_add(squared_deviations, square, "Pauli standard error")?;
        }
        let sample_variance = squared_deviations / (self.snapshots.len() - 1) as f64;
        checked_finite(sample_variance, "Pauli standard error")?;
        let standard_error = (sample_variance / self.snapshots.len() as f64).sqrt();
        Ok(Some(checked_finite(
            standard_error,
            "Pauli standard error",
        )?))
    }

    /// Returns a deterministic median-of-means estimate.
    ///
    /// Contiguous groups differ in length by at most one. For an even number of
    /// groups the two central group means are averaged.
    pub fn median_of_means(&self, observable: &PauliString, groups: usize) -> Result<f64> {
        self.check_string(observable)?;
        validate_groups(groups, self.snapshots.len())?;
        let mut means = Vec::with_capacity(groups);
        let base = self.snapshots.len() / groups;
        let remainder = self.snapshots.len() % groups;
        let mut start = 0;
        for group in 0..groups {
            let length = base + usize::from(group < remainder);
            let end = start + length;
            let mut sum = 0.0;
            for snapshot in &self.snapshots[start..end] {
                sum = checked_add(
                    sum,
                    string_snapshot_estimate(snapshot, observable)?,
                    "median-of-means group",
                )?;
            }
            means.push(checked_finite(
                sum / length as f64,
                "median-of-means group",
            )?);
            start = end;
        }
        means.sort_by(f64::total_cmp);
        if groups % 2 == 1 {
            Ok(means[groups / 2])
        } else {
            let middle = checked_add(
                means[groups / 2 - 1],
                means[groups / 2],
                "median-of-means midpoint",
            )?;
            checked_finite(0.5 * middle, "median-of-means midpoint")
        }
    }

    /// Returns the sample-mean estimate of a linear Pauli observable.
    ///
    /// Non-finite term products or accumulation are reported as
    /// [`ShadowError::NumericalOverflow`].
    pub fn observable_mean(&self, observable: &PauliObservable) -> Result<f64> {
        self.check_observable(observable)?;
        let mut sum = 0.0;
        for snapshot in &self.snapshots {
            sum = checked_add(
                sum,
                observable_snapshot_estimate(snapshot, observable)?,
                "linear-observable sample mean",
            )?;
        }
        checked_finite(
            sum / self.snapshots.len() as f64,
            "linear-observable sample mean",
        )
    }

    /// Returns a conservative two-sided Hoeffding radius for a Pauli string.
    ///
    /// With failure probability `delta`, the population mean differs from the
    /// sample mean by at most the returned radius under independent sampling.
    pub fn hoeffding_radius(&self, observable: &PauliString, delta: f64) -> Result<f64> {
        self.check_string(observable)?;
        validate_delta(delta)?;
        hoeffding_radius(observable.bound()?, delta, self.snapshots.len())
    }

    /// Returns the corresponding conservative radius for a linear observable.
    pub fn observable_hoeffding_radius(
        &self,
        observable: &PauliObservable,
        delta: f64,
    ) -> Result<f64> {
        self.check_observable(observable)?;
        validate_delta(delta)?;
        hoeffding_radius(observable.bound()?, delta, self.snapshots.len())
    }

    fn check_string(&self, observable: &PauliString) -> Result<()> {
        if observable.factors.len() != self.qubits {
            Err(ShadowError::QubitShape {
                expected: self.qubits,
                actual: observable.factors.len(),
            })
        } else {
            Ok(())
        }
    }

    fn check_observable(&self, observable: &PauliObservable) -> Result<()> {
        if observable.qubits != self.qubits {
            Err(ShadowError::QubitShape {
                expected: self.qubits,
                actual: observable.qubits,
            })
        } else {
            Ok(())
        }
    }
}

fn string_snapshot_estimate(snapshot: &Snapshot, observable: &PauliString) -> Result<f64> {
    let mut estimate = 1.0;
    for ((basis, outcome), factor) in snapshot
        .bases
        .iter()
        .zip(&snapshot.outcomes)
        .zip(&observable.factors)
    {
        if *factor == Pauli::I {
            continue;
        }
        if !basis.matches(*factor) {
            return Ok(0.0);
        }
        estimate = checked_mul(
            estimate,
            3.0 * f64::from(*outcome),
            "Pauli snapshot estimator",
        )?;
    }
    Ok(estimate)
}

fn observable_snapshot_estimate(snapshot: &Snapshot, observable: &PauliObservable) -> Result<f64> {
    let mut estimate = 0.0;
    for (coefficient, string) in &observable.terms {
        let term = checked_mul(
            *coefficient,
            string_snapshot_estimate(snapshot, string)?,
            "linear-observable snapshot estimator",
        )?;
        estimate = checked_add(estimate, term, "linear-observable snapshot estimator")?;
    }
    Ok(estimate)
}

fn hoeffding_radius(bound: f64, delta: f64, samples: usize) -> Result<f64> {
    // `ln(2 / delta)` is evaluated as `ln(2) - ln(delta)` so a valid
    // subnormal delta cannot overflow during the intermediate division.
    let logarithm = core::f64::consts::LN_2 - delta.ln();
    let scale = (2.0 * logarithm / samples as f64).sqrt();
    checked_mul(bound, scale, "Hoeffding radius")
}

fn checked_add(left: f64, right: f64, context: &'static str) -> Result<f64> {
    checked_finite(left + right, context)
}

fn checked_mul(left: f64, right: f64, context: &'static str) -> Result<f64> {
    checked_finite(left * right, context)
}

fn checked_finite(value: f64, context: &'static str) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ShadowError::NumericalOverflow { context })
    }
}

fn validate_groups(groups: usize, samples: usize) -> Result<()> {
    if groups == 0 || groups > samples {
        Err(ShadowError::InvalidGroupCount { groups, samples })
    } else {
        Ok(())
    }
}

fn validate_delta(delta: f64) -> Result<()> {
    if delta.is_finite() && delta > 0.0 && delta < 1.0 {
        Ok(())
    } else {
        Err(ShadowError::InvalidFailureProbability)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn z_snapshot(outcome: i8) -> Snapshot {
        Snapshot::try_new(vec![MeasurementBasis::Z], vec![outcome]).unwrap()
    }

    #[test]
    fn inverse_channel_estimator_matches_local_pauli_rule() {
        let dataset = ShadowDataset::try_new(
            2,
            vec![
                Snapshot::try_new(vec![MeasurementBasis::X, MeasurementBasis::Z], vec![-1, 1])
                    .unwrap(),
            ],
        )
        .unwrap();
        let xz = PauliString::try_new(vec![Pauli::X, Pauli::Z]).unwrap();
        let xx = PauliString::try_new(vec![Pauli::X, Pauli::X]).unwrap();
        let identity = PauliString::try_new(vec![Pauli::I, Pauli::I]).unwrap();
        assert_eq!(dataset.mean(&xz).unwrap(), -9.0);
        assert_eq!(dataset.mean(&xx).unwrap(), 0.0);
        assert_eq!(dataset.mean(&identity).unwrap(), 1.0);
    }

    #[test]
    fn uniform_basis_average_is_unbiased_for_z_eigenstate() {
        let dataset = ShadowDataset::try_new(
            1,
            vec![
                Snapshot::try_new(vec![MeasurementBasis::X], vec![1]).unwrap(),
                Snapshot::try_new(vec![MeasurementBasis::Y], vec![-1]).unwrap(),
                z_snapshot(1),
            ],
        )
        .unwrap();
        let z = PauliString::try_new(vec![Pauli::Z]).unwrap();
        assert_eq!(dataset.mean(&z).unwrap(), 1.0);
    }

    #[test]
    fn median_of_means_is_robust_to_one_group_outlier() {
        let dataset = ShadowDataset::try_new(
            1,
            vec![
                z_snapshot(1),
                z_snapshot(1),
                z_snapshot(1),
                z_snapshot(1),
                z_snapshot(-1),
            ],
        )
        .unwrap();
        let z = PauliString::try_new(vec![Pauli::Z]).unwrap();
        assert_eq!(dataset.median_of_means(&z, 5).unwrap(), 3.0);
        assert_eq!(dataset.mean(&z).unwrap(), 1.8);
    }

    #[test]
    fn standard_error_and_bounds_are_explicit() {
        let dataset = ShadowDataset::try_new(1, vec![z_snapshot(1), z_snapshot(-1)]).unwrap();
        let z = PauliString::try_new(vec![Pauli::Z]).unwrap();
        assert_eq!(dataset.standard_error(&z).unwrap(), Some(3.0));
        let radius = dataset.hoeffding_radius(&z, 0.05).unwrap();
        assert!(radius.is_finite() && radius > 0.0);
        assert_eq!(
            dataset.hoeffding_radius(&z, 1.0),
            Err(ShadowError::InvalidFailureProbability)
        );
    }

    #[test]
    fn linear_observable_combines_snapshot_estimates() {
        let dataset = ShadowDataset::try_new(1, vec![z_snapshot(1)]).unwrap();
        let identity = PauliString::try_new(vec![Pauli::I]).unwrap();
        let z = PauliString::try_new(vec![Pauli::Z]).unwrap();
        let observable = PauliObservable::try_new(vec![(2.0, identity), (-0.5, z)]).unwrap();
        assert_eq!(dataset.observable_mean(&observable).unwrap(), 0.5);
        assert!(
            dataset
                .observable_hoeffding_radius(&observable, 0.1)
                .unwrap()
                > 0.0
        );
    }

    #[test]
    fn high_weight_and_non_finite_reductions_are_rejected() {
        assert_eq!(
            PauliString::try_new(vec![Pauli::X; MAX_FINITE_PAULI_WEIGHT + 1]),
            Err(ShadowError::PauliWeightOverflow {
                weight: MAX_FINITE_PAULI_WEIGHT + 1,
                maximum: MAX_FINITE_PAULI_WEIGHT,
            })
        );

        let boundary = PauliString::try_new(vec![Pauli::X; MAX_FINITE_PAULI_WEIGHT]).unwrap();
        let snapshot = Snapshot::try_new(
            vec![MeasurementBasis::X; MAX_FINITE_PAULI_WEIGHT],
            vec![1; MAX_FINITE_PAULI_WEIGHT],
        )
        .unwrap();
        let boundary_dataset =
            ShadowDataset::try_new(MAX_FINITE_PAULI_WEIGHT, vec![snapshot]).unwrap();
        assert!(boundary_dataset.mean(&boundary).unwrap().is_finite());
        assert!(matches!(
            boundary_dataset.hoeffding_radius(&boundary, 0.5),
            Err(ShadowError::NumericalOverflow {
                context: "Hoeffding radius"
            })
        ));

        let one_qubit = ShadowDataset::try_new(
            1,
            vec![Snapshot::try_new(vec![MeasurementBasis::X], vec![1]).unwrap()],
        )
        .unwrap();
        let x = PauliString::try_new(vec![Pauli::X]).unwrap();
        let overflowing = PauliObservable::try_new(vec![(f64::MAX, x)]).unwrap();
        assert!(matches!(
            one_qubit.observable_mean(&overflowing),
            Err(ShadowError::NumericalOverflow { .. })
        ));
        assert!(matches!(
            one_qubit.observable_hoeffding_radius(&overflowing, 0.5),
            Err(ShadowError::NumericalOverflow { .. })
        ));
    }

    #[test]
    fn validation_rejects_malformed_records_and_queries() {
        assert_eq!(
            Snapshot::try_new(vec![MeasurementBasis::X], vec![0]),
            Err(ShadowError::InvalidOutcome { qubit: 0, value: 0 })
        );
        let dataset = ShadowDataset::try_new(1, vec![z_snapshot(1)]).unwrap();
        let two_qubit = PauliString::try_new(vec![Pauli::Z, Pauli::Z]).unwrap();
        assert!(matches!(
            dataset.mean(&two_qubit),
            Err(ShadowError::QubitShape { .. })
        ));
        let one_qubit = PauliString::try_new(vec![Pauli::Z]).unwrap();
        assert_eq!(
            dataset.median_of_means(&one_qubit, 2),
            Err(ShadowError::InvalidGroupCount {
                groups: 2,
                samples: 1,
            })
        );
    }
}
