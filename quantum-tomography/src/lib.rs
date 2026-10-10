//! Checked, dependency-light reference tools for state and channel tomography.
//!
//! This crate provides auditable linear inversion and interoperability
//! diagnostics. It does not duplicate optimization-based QPT such as qtool's
//! FISTA implementation.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use core::fmt;
pub use open_quantum_systems::{Complex64, Operator};

/// Tomography errors.
#[derive(Clone, Debug, PartialEq)]
pub enum TomographyError {
    /// Qubit count is zero or too large for checked allocation.
    InvalidQubits,
    /// Input length differs from the required length.
    Shape {
        /// Required number of values.
        expected: usize,
        /// Supplied number of values.
        actual: usize,
    },
    /// Input contains NaN or infinity.
    NonFinite,
    /// Identity Pauli expectation is not one within tolerance.
    InvalidNormalization(f64),
    /// Tolerance is not finite and positive.
    InvalidTolerance,
    /// Underlying operator construction failed.
    Operator(String),
    /// A named GST gate was not present in the model.
    UnknownGate(String),
    /// A GST outcome vector was not a finite normalized probability vector.
    InvalidProbabilities,
    /// Summing multinomial counts overflowed `u64`.
    CountOverflow,
    /// A gauge transformation was singular at its own numeric scale.
    SingularGaugeTransform,
    /// The supplied process-tomography probes are not informationally complete.
    SingularProcessDesign,
    /// A Hermitian eigenvalue iteration did not converge.
    EigensolverDidNotConverge,
}
impl fmt::Display for TomographyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for TomographyError {}
impl From<open_quantum_systems::QuantumError> for TomographyError {
    fn from(e: open_quantum_systems::QuantumError) -> Self {
        Self::Operator(e.to_string())
    }
}
/// Result type for tomography operations.
pub type Result<T> = core::result::Result<T, TomographyError>;

/// A local Pauli measurement axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

/// Complete tensor-product local-Pauli measurement plan (`3^n` settings).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalPauliPlan {
    qubits: usize,
    settings: Vec<Vec<Pauli>>,
}
impl LocalPauliPlan {
    /// Constructs all X/Y/Z settings in lexicographic tensor order.
    pub fn complete(qubits: usize) -> Result<Self> {
        if qubits == 0 || qubits >= usize::BITS as usize {
            return Err(TomographyError::InvalidQubits);
        }
        let count = 3usize
            .checked_pow(qubits as u32)
            .ok_or(TomographyError::InvalidQubits)?;
        let mut settings = Vec::with_capacity(count);
        for mut code in 0..count {
            let mut s = vec![Pauli::X; qubits];
            for q in (0..qubits).rev() {
                s[q] = [Pauli::X, Pauli::Y, Pauli::Z][code % 3];
                code /= 3;
            }
            settings.push(s);
        }
        Ok(Self { qubits, settings })
    }
    /// Number of qubits.
    pub const fn qubits(&self) -> usize {
        self.qubits
    }
    /// Measurement settings.
    pub fn settings(&self) -> &[Vec<Pauli>] {
        &self.settings
    }
}

fn count(qubits: usize) -> Result<(usize, usize)> {
    if qubits == 0 {
        return Err(TomographyError::InvalidQubits);
    };
    let d = 1usize
        .checked_shl(qubits as u32)
        .ok_or(TomographyError::InvalidQubits)?;
    let p = 4usize
        .checked_pow(qubits as u32)
        .ok_or(TomographyError::InvalidQubits)?;
    Ok((d, p))
}
fn pauli_strings(n: usize) -> Vec<Vec<Pauli>> {
    let p = 4usize.pow(n as u32);
    (0..p)
        .map(|mut c| {
            let mut s = vec![Pauli::I; n];
            for q in (0..n).rev() {
                s[q] = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z][c % 4];
                c /= 4;
            }
            s
        })
        .collect()
}
fn kron_pauli(s: &[Pauli]) -> Vec<Complex64> {
    let mut a = vec![1.0.into()];
    let mut d = 1;
    for p in s {
        let m = match p {
            Pauli::I => [1.0.into(), 0.0.into(), 0.0.into(), 1.0.into()],
            Pauli::X => [0.0.into(), 1.0.into(), 1.0.into(), 0.0.into()],
            Pauli::Y => [
                0.0.into(),
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0),
                0.0.into(),
            ],
            Pauli::Z => [1.0.into(), 0.0.into(), 0.0.into(), (-1.0).into()],
        };
        let mut b = vec![0.0.into(); 4 * d * d];
        for i in 0..d {
            for j in 0..d {
                for x in 0..2 {
                    for y in 0..2 {
                        b[(2 * i + x) * (2 * d) + 2 * j + y] = a[i * d + j] * m[2 * x + y];
                    }
                }
            }
        }
        a = b;
        d *= 2;
    }
    a
}

/// Linear-inverts all `4^n` Pauli expectations, identity first.
///
/// The identity expectation is validated against one and then pinned to one,
/// so the returned Hermitian estimate has unit trace even when the supplied
/// normalization differs within `tolerance`.
pub fn linear_inversion(qubits: usize, expectations: &[f64], tolerance: f64) -> Result<Operator> {
    let (d, p) = count(qubits)?;
    if expectations.len() != p {
        return Err(TomographyError::Shape {
            expected: p,
            actual: expectations.len(),
        });
    }
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(TomographyError::InvalidTolerance);
    }
    if expectations.iter().any(|x| !x.is_finite()) {
        return Err(TomographyError::NonFinite);
    }
    if (expectations[0] - 1.0).abs() > tolerance {
        return Err(TomographyError::InvalidNormalization(expectations[0]));
    }
    let mut v = vec![0.0.into(); d * d];
    for (index, (e, s)) in expectations.iter().zip(pauli_strings(qubits)).enumerate() {
        let expectation = if index == 0 { 1.0 } else { *e };
        for (i, x) in kron_pauli(&s).into_iter().enumerate() {
            v[i] += x * (expectation / d as f64);
        }
    }
    Ok(Operator::try_new(d, v)?)
}

/// Returns a minimal informationally complete product-state probe design.
///
/// Each qubit is prepared as one of `|0>`, `|1>`, `|+>`, or `|+i>`. Every
/// returned row contains the resulting `4^n` Pauli expectations in the same
/// lexicographic `I,X,Y,Z` order accepted by [`linear_inversion`]. The design
/// has exactly `4^n` probes and spans the real Hermitian operator space.
pub fn product_probe_expectations(qubits: usize) -> Result<Vec<Vec<f64>>> {
    let (_, pauli_count) = count(qubits)?;
    let local = [
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, -1.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
    ];
    let mut probes = Vec::new();
    probes
        .try_reserve_exact(pauli_count)
        .map_err(|_| TomographyError::InvalidQubits)?;
    for mut preparation_code in 0..pauli_count {
        let mut preparations = vec![0usize; qubits];
        for qubit in (0..qubits).rev() {
            preparations[qubit] = preparation_code % 4;
            preparation_code /= 4;
        }
        let mut row = Vec::new();
        row.try_reserve_exact(pauli_count)
            .map_err(|_| TomographyError::InvalidQubits)?;
        for paulis in pauli_strings(qubits) {
            let expectation = paulis
                .iter()
                .zip(&preparations)
                .map(|(pauli, &preparation)| {
                    let component = match pauli {
                        Pauli::I => 0,
                        Pauli::X => 1,
                        Pauli::Y => 2,
                        Pauli::Z => 3,
                    };
                    local[preparation][component]
                })
                .product();
            row.push(expectation);
        }
        probes.push(row);
    }
    Ok(probes)
}

/// Linear-inverts an informationally complete quantum process into a PTM.
///
/// `input_expectations[k]` and `output_expectations[k]` are the measured Pauli
/// expectation vectors before and after the same process for probe `k`. This
/// reference inversion requires exactly `4^n` linearly independent probes;
/// [`product_probe_expectations`] supplies a minimal checked design. Both
/// input and output states must be normalized. The returned transfer matrix
/// obeys `r_out = R r_in` in lexicographic `I,X,Y,Z` tensor order.
pub fn process_linear_inversion(
    qubits: usize,
    input_expectations: &[Vec<f64>],
    output_expectations: &[Vec<f64>],
    tolerance: f64,
) -> Result<PauliTransferMatrix> {
    check_tol(tolerance)?;
    let (_, pauli_count) = count(qubits)?;
    if input_expectations.len() != pauli_count {
        return Err(TomographyError::Shape {
            expected: pauli_count,
            actual: input_expectations.len(),
        });
    }
    if output_expectations.len() != pauli_count {
        return Err(TomographyError::Shape {
            expected: pauli_count,
            actual: output_expectations.len(),
        });
    }
    for expectations in input_expectations.iter().chain(output_expectations) {
        if expectations.len() != pauli_count {
            return Err(TomographyError::Shape {
                expected: pauli_count,
                actual: expectations.len(),
            });
        }
        if expectations.iter().any(|value| !value.is_finite()) {
            return Err(TomographyError::NonFinite);
        }
        if (expectations[0] - 1.0).abs() > tolerance {
            return Err(TomographyError::InvalidNormalization(expectations[0]));
        }
    }
    let matrix_len = pauli_count
        .checked_mul(pauli_count)
        .ok_or(TomographyError::InvalidQubits)?;
    let mut inputs = vec![0.0; matrix_len];
    let mut outputs = vec![0.0; matrix_len];
    for probe in 0..pauli_count {
        for component in 0..pauli_count {
            inputs[component * pauli_count + probe] = input_expectations[probe][component];
            outputs[component * pauli_count + probe] = output_expectations[probe][component];
        }
    }
    let inverse = invert_real_with_threshold(
        &inputs,
        pauli_count,
        tolerance,
        TomographyError::SingularProcessDesign,
    )?;
    PauliTransferMatrix::try_new(qubits, mat_mul(&outputs, &inverse, pauli_count))
}

fn eig(mut a: Vec<Complex64>, n: usize, tolerance: f64) -> Result<(Vec<f64>, Vec<Complex64>)> {
    check_tol(tolerance)?;
    let expected = n.checked_mul(n).ok_or(TomographyError::InvalidQubits)?;
    if n == 0 || a.len() != expected {
        return Err(TomographyError::Shape {
            expected,
            actual: a.len(),
        });
    }
    let scale = a
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(value.norm()))
        .max(f64::MIN_POSITIVE);
    let mut v = vec![0.0.into(); n * n];
    for i in 0..n {
        v[i * n + i] = 1.0.into()
    }
    let maximum_rotations = 64usize
        .checked_mul(n)
        .and_then(|count| count.checked_mul(n))
        .ok_or(TomographyError::InvalidQubits)?;
    for _ in 0..maximum_rotations {
        let mut p = 0;
        let mut q = 1.min(n - 1);
        let mut best = 0.0;
        for i in 0..n {
            for j in i + 1..n {
                if a[i * n + j].norm() > best {
                    best = a[i * n + j].norm();
                    p = i;
                    q = j
                }
            }
        }
        if best <= tolerance * scale {
            let eigenvalues: Vec<_> = (0..n).map(|i| a[i * n + i].re).collect();
            let mut order: Vec<_> = (0..n).collect();
            order.sort_by(|&left, &right| eigenvalues[left].total_cmp(&eigenvalues[right]));
            let sorted_values = order.iter().map(|&index| eigenvalues[index]).collect();
            let mut sorted_vectors = vec![Complex64::new(0.0, 0.0); expected];
            for (new_column, &old_column) in order.iter().enumerate() {
                for row in 0..n {
                    sorted_vectors[row * n + new_column] = v[row * n + old_column];
                }
            }
            return Ok((sorted_values, sorted_vectors));
        }
        let z = a[p * n + q];
        let app = a[p * n + p].re;
        let aqq = a[q * n + q].re;
        let tau = (aqq - app) / (2.0 * best);
        let tangent = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + tangent * tangent).sqrt();
        let s = tangent * c;
        let phase = z / best;
        for index in 0..n {
            if index == p || index == q {
                continue;
            }
            let aip = a[index * n + p];
            let aiq = a[index * n + q];
            let new_ip = aip * phase * c - aiq * s;
            let new_iq = aip * phase * s + aiq * c;
            a[index * n + p] = new_ip;
            a[p * n + index] = new_ip.conj();
            a[index * n + q] = new_iq;
            a[q * n + index] = new_iq.conj();
        }
        for row in 0..n {
            let vip = v[row * n + p];
            let viq = v[row * n + q];
            v[row * n + p] = vip * phase * c - viq * s;
            v[row * n + q] = vip * phase * s + viq * c;
        }
        a[p * n + p] = Complex64::new(c * c * app + s * s * aqq - 2.0 * c * s * best, 0.0);
        a[q * n + q] = Complex64::new(s * s * app + c * c * aqq + 2.0 * c * s * best, 0.0);
        a[p * n + q] = Complex64::new(0.0, 0.0);
        a[q * n + p] = Complex64::new(0.0, 0.0);
    }
    Err(TomographyError::EigensolverDidNotConverge)
}

/// Euclidean-projects a Hermitian estimate onto PSD, trace-one matrices.
///
/// Eigenvalues are projected onto the probability simplex by water filling;
/// this is the Frobenius-norm projection, not clipping followed by rescaling.
pub fn project_density(estimate: &Operator) -> Result<Operator> {
    project_density_with_tolerance(estimate, 1.0e-12)
}

/// Euclidean-projects a Hermitian estimate with an explicit relative
/// eigensolver tolerance.
pub fn project_density_with_tolerance(estimate: &Operator, tolerance: f64) -> Result<Operator> {
    let n = estimate.dimension();
    let mut h = vec![0.0.into(); n * n];
    for i in 0..n {
        for j in 0..n {
            h[i * n + j] = (estimate.get(i, j).unwrap() + estimate.get(j, i).unwrap().conj()) * 0.5;
        }
    }
    let (w, v) = eig(h, n, tolerance)?;
    let mut descending = w.clone();
    descending.sort_by(|left, right| right.total_cmp(left));
    let mut cumulative = 0.0;
    let mut active = 0usize;
    for (index, &value) in descending.iter().enumerate() {
        cumulative += value;
        let threshold = (cumulative - 1.0) / (index + 1) as f64;
        if value > threshold {
            active = index + 1;
        }
    }
    let threshold = (descending[..active].iter().sum::<f64>() - 1.0) / active as f64;
    let projected: Vec<_> = w
        .into_iter()
        .map(|value| (value - threshold).max(0.0))
        .collect();
    let mut out = vec![0.0.into(); n * n];
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] += v[i * n + k] * projected[k] * v[j * n + k].conj();
            }
        }
    }
    Ok(Operator::try_new(n, out)?)
}

/// Pauli-transfer matrix with `R_ab=Tr[P_a E(P_b)]/d`.
#[derive(Clone, Debug, PartialEq)]
pub struct PauliTransferMatrix {
    qubits: usize,
    values: Vec<f64>,
}
impl PauliTransferMatrix {
    /// Constructs a checked PTM in row-major order.
    pub fn try_new(qubits: usize, values: Vec<f64>) -> Result<Self> {
        let (_, p) = count(qubits)?;
        let expected = p.checked_mul(p).ok_or(TomographyError::InvalidQubits)?;
        if values.len() != expected {
            return Err(TomographyError::Shape {
                expected,
                actual: values.len(),
            });
        }
        if values.iter().any(|x| !x.is_finite()) {
            return Err(TomographyError::NonFinite);
        }
        Ok(Self { qubits, values })
    }
    /// Number of qubits acted on by the process.
    pub const fn qubits(&self) -> usize {
        self.qubits
    }
    /// Row-major PTM coefficients.
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }
    /// Applies the PTM to a Pauli-expectation vector.
    pub fn apply_expectations(&self, input: &[f64]) -> Result<Vec<f64>> {
        let (_, expected) = count(self.qubits)?;
        if input.len() != expected {
            return Err(TomographyError::Shape {
                expected,
                actual: input.len(),
            });
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(TomographyError::NonFinite);
        }
        Ok(mat_vec(&self.values, input, expected))
    }
    /// Tests trace preservation.
    pub fn is_trace_preserving(&self, t: f64) -> Result<bool> {
        check_tol(t)?;
        let p = 4usize.pow(self.qubits as u32);
        Ok((0..p).all(|j| (self.values[j] - if j == 0 { 1.0 } else { 0.0 }).abs() <= t))
    }
    /// Tests unitality.
    pub fn is_unital(&self, t: f64) -> Result<bool> {
        check_tol(t)?;
        let p = 4usize.pow(self.qubits as u32);
        Ok((0..p).all(|i| (self.values[i * p] - if i == 0 { 1.0 } else { 0.0 }).abs() <= t))
    }
    /// Converts to the normalized Choi state (unit trace for TP maps).
    pub fn normalized_choi(&self) -> Result<Operator> {
        let (d, p) = count(self.qubits)?;
        let ps: Vec<_> = pauli_strings(self.qubits)
            .iter()
            .map(|s| kron_pauli(s))
            .collect();
        let nd = d.checked_mul(d).ok_or(TomographyError::InvalidQubits)?;
        let choi_len = nd.checked_mul(nd).ok_or(TomographyError::InvalidQubits)?;
        let mut c = vec![0.0.into(); choi_len];
        for i in 0..d {
            for j in 0..d {
                for b in 0..p {
                    let coeff = ps[b][j * d + i] / d as f64;
                    for (a, pauli) in ps.iter().enumerate().take(p) {
                        let x = coeff * self.values[a * p + b] / d as f64;
                        for r in 0..d {
                            for s in 0..d {
                                c[(i * d + r) * nd + j * d + s] += x * pauli[r * d + s];
                            }
                        }
                    }
                }
            }
        }
        Ok(Operator::try_new(nd, c)?)
    }
    /// Tests complete positivity from the normalized Choi eigenvalues.
    pub fn is_completely_positive(&self, t: f64) -> Result<bool> {
        check_tol(t)?;
        let c = self.normalized_choi()?;
        let (w, _) = eig(c.as_slice().to_vec(), c.dimension(), t)?;
        Ok(w.into_iter().all(|x| x >= -t))
    }
}
fn check_tol(t: f64) -> Result<()> {
    if t.is_finite() && t > 0.0 {
        Ok(())
    } else {
        Err(TomographyError::InvalidTolerance)
    }
}

/// One multinomial gate-sequence experiment.
#[derive(Clone, Debug, PartialEq)]
pub struct GstRecord {
    /// Gate names, applied from first to last.
    pub sequence: Vec<String>,
    /// Counts in the model's POVM-effect order.
    pub counts: Vec<u64>,
}

/// Multinomial fit diagnostics suitable for an injected optimizer objective.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitDiagnostics {
    /// Sum of `count * ln(probability)`.
    pub log_likelihood: f64,
    /// Likelihood-ratio deviance from the saturated multinomial model.
    pub deviance: f64,
}

/// A checked PTM gate-set model for focused gate-set tomography interoperation.
#[derive(Clone, Debug, PartialEq)]
pub struct GateSetModel {
    dimension: usize,
    state: Vec<f64>,
    effects: Vec<Vec<f64>>,
    gates: Vec<(String, Vec<f64>)>,
}

impl GateSetModel {
    /// Constructs a model. Effect vectors use the dual convention so
    /// `p_k = effect_k · state`; gates act on state vectors from the left.
    pub fn try_new(
        state: Vec<f64>,
        effects: Vec<Vec<f64>>,
        gates: Vec<(String, Vec<f64>)>,
    ) -> Result<Self> {
        let dimension = state.len();
        if dimension == 0 || state.iter().any(|x| !x.is_finite()) {
            return Err(TomographyError::NonFinite);
        }
        if effects.is_empty() || effects.iter().any(|e| e.len() != dimension) {
            return Err(TomographyError::Shape {
                expected: dimension,
                actual: effects.first().map_or(0, Vec::len),
            });
        }
        let square = dimension
            .checked_mul(dimension)
            .ok_or(TomographyError::InvalidQubits)?;
        if effects
            .iter()
            .flatten()
            .chain(gates.iter().flat_map(|(_, g)| g))
            .any(|x| !x.is_finite())
        {
            return Err(TomographyError::NonFinite);
        }
        if gates.iter().any(|(_, g)| g.len() != square) {
            return Err(TomographyError::Shape {
                expected: square,
                actual: gates
                    .iter()
                    .find(|(_, g)| g.len() != square)
                    .map_or(0, |(_, g)| g.len()),
            });
        }
        Ok(Self {
            dimension,
            state,
            effects,
            gates,
        })
    }
    /// Returns all outcome probabilities for a named gate sequence.
    pub fn probabilities(&self, sequence: &[&str]) -> Result<Vec<f64>> {
        let mut r = self.state.clone();
        for name in sequence {
            let g = self
                .gates
                .iter()
                .find(|(n, _)| n == name)
                .ok_or_else(|| TomographyError::UnknownGate((*name).to_owned()))?;
            r = mat_vec(&g.1, &r, self.dimension);
        }
        Ok(self.effects.iter().map(|e| dot(e, &r)).collect())
    }
    /// Evaluates multinomial log-likelihood and saturated-model deviance.
    pub fn fit_diagnostics(&self, records: &[GstRecord]) -> Result<FitDiagnostics> {
        self.fit_diagnostics_with_tolerance(records, 1.0e-10)
    }

    /// Evaluates multinomial diagnostics after checking each outcome vector
    /// against an explicit probability-normalization tolerance.
    pub fn fit_diagnostics_with_tolerance(
        &self,
        records: &[GstRecord],
        tolerance: f64,
    ) -> Result<FitDiagnostics> {
        check_tol(tolerance)?;
        let mut ll = 0.0;
        let mut dev = 0.0;
        for record in records {
            if record.counts.len() != self.effects.len() {
                return Err(TomographyError::Shape {
                    expected: self.effects.len(),
                    actual: record.counts.len(),
                });
            }
            let names: Vec<_> = record.sequence.iter().map(String::as_str).collect();
            let probabilities = self.probabilities(&names)?;
            if probabilities.iter().any(|probability| {
                !probability.is_finite()
                    || *probability < -tolerance
                    || *probability > 1.0 + tolerance
            }) || (probabilities.iter().sum::<f64>() - 1.0).abs() > tolerance
            {
                return Err(TomographyError::InvalidProbabilities);
            }
            let total_count = record
                .counts
                .iter()
                .try_fold(0_u64, |total, &count| total.checked_add(count))
                .ok_or(TomographyError::CountOverflow)?;
            if total_count == 0 {
                return Err(TomographyError::InvalidProbabilities);
            }
            let total = total_count as f64;
            for (&count, &probability) in record.counts.iter().zip(&probabilities) {
                let probability = probability.clamp(0.0, 1.0);
                let c = count as f64;
                if c > 0.0 {
                    if probability == 0.0 {
                        return Err(TomographyError::InvalidProbabilities);
                    }
                    ll += c * probability.ln();
                    dev += 2.0 * c * (c / (total * probability)).ln();
                }
            }
        }
        Ok(FitDiagnostics {
            log_likelihood: ll,
            deviance: dev,
        })
    }
    /// Applies an invertible similarity gauge: `r'=Sr`, `E'=ES^-1`,
    /// and `G'=SGS^-1`. All sequence probabilities are invariant.
    pub fn gauge_transform(&self, transform: &[f64]) -> Result<Self> {
        let n = self.dimension;
        if transform.len() != n * n {
            return Err(TomographyError::Shape {
                expected: n * n,
                actual: transform.len(),
            });
        }
        if transform.iter().any(|value| !value.is_finite()) {
            return Err(TomographyError::NonFinite);
        }
        let inverse = invert_real(transform, n)?;
        let state = mat_vec(transform, &self.state, n);
        let effects = self
            .effects
            .iter()
            .map(|e| row_mat(e, &inverse, n))
            .collect();
        let gates = self
            .gates
            .iter()
            .map(|(name, g)| {
                let left = mat_mul(transform, g, n);
                (name.clone(), mat_mul(&left, &inverse, n))
            })
            .collect();
        Self::try_new(state, effects, gates)
    }
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn mat_vec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum())
        .collect()
}
fn row_mat(x: &[f64], a: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|j| (0..n).map(|i| x[i] * a[i * n + j]).sum())
        .collect()
}
fn mat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|z| {
            let i = z / n;
            let j = z % n;
            (0..n).map(|k| a[i * n + k] * b[k * n + j]).sum()
        })
        .collect()
}
fn invert_real(a: &[f64], n: usize) -> Result<Vec<f64>> {
    let relative_threshold = 64.0 * f64::EPSILON * n as f64;
    invert_real_with_threshold(
        a,
        n,
        relative_threshold,
        TomographyError::SingularGaugeTransform,
    )
}

fn invert_real_with_threshold(
    a: &[f64],
    n: usize,
    relative_threshold: f64,
    singular_error: TomographyError,
) -> Result<Vec<f64>> {
    let mut l = a.to_vec();
    let scale = l
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(value.abs()));
    if scale == 0.0 || !scale.is_finite() {
        return Err(singular_error);
    }
    let pivot_threshold = relative_threshold * scale;
    let mut r = vec![0.0; n * n];
    for i in 0..n {
        r[i * n + i] = 1.0
    }
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| l[i * n + k].abs().total_cmp(&l[j * n + k].abs()))
            .unwrap();
        if l[pivot * n + k].abs() <= pivot_threshold {
            return Err(singular_error);
        }
        for j in 0..n {
            l.swap(k * n + j, pivot * n + j);
            r.swap(k * n + j, pivot * n + j)
        }
        let d = l[k * n + k];
        for j in 0..n {
            l[k * n + j] /= d;
            r[k * n + j] /= d
        }
        for i in 0..n {
            if i != k {
                let f = l[i * n + k];
                for j in 0..n {
                    l[i * n + j] -= f * l[k * n + j];
                    r[i * n + j] -= f * r[k * n + j];
                }
            }
        }
    }
    Ok(r)
}
