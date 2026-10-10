use core::f64::consts::PI;

use crate::matrix::validate_tolerance;
use crate::{Complex64, DenseMatrix, GreenError, Result};

const DEFAULT_CAUSALITY_TOLERANCE: f64 = 1.0e-12;

fn component_max(value: Complex64) -> f64 {
    value.re.abs().max(value.im.abs())
}

/// One retarded frequency and its cluster Green-function matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct GreenPoint {
    /// Complex retarded frequency `z = omega + i eta`.
    pub frequency: Complex64,
    /// One-particle Green-function matrix in cluster-site order.
    pub green: DenseMatrix,
}

impl GreenPoint {
    /// Construct a frequency/matrix pair. Grid-level validation occurs in
    /// [`ClusterGreenGrid::try_new`].
    pub const fn new(frequency: Complex64, green: DenseMatrix) -> Self {
        Self { frequency, green }
    }
}

/// Detailed result of a matrix causality check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CausalityReport {
    /// Whether every spectral-matrix pivot was nonnegative within tolerance.
    pub causal: bool,
    /// Smallest real LDL† pivot encountered.
    pub minimum_pivot: f64,
}

/// A checked retarded cluster Green function on an ordered frequency grid.
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterGreenGrid {
    sites: usize,
    points: Vec<GreenPoint>,
}

/// CPT-embedded Green matrices for one momentum and an entire frequency grid.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedGreenGrid {
    sites: usize,
    points: Vec<GreenPoint>,
}

impl ClusterGreenGrid {
    /// Validate a retarded cluster Green-function grid using a `1e-12`
    /// scale-aware causality tolerance.
    pub fn try_new(sites: usize, points: Vec<GreenPoint>) -> Result<Self> {
        Self::try_new_with_tolerance(sites, points, DEFAULT_CAUSALITY_TOLERANCE)
    }

    /// Validate a retarded cluster Green-function grid with an explicit
    /// scale-aware causality tolerance.
    pub fn try_new_with_tolerance(
        sites: usize,
        points: Vec<GreenPoint>,
        tolerance: f64,
    ) -> Result<Self> {
        validate_tolerance(tolerance)?;
        if sites == 0 {
            return Err(GreenError::Empty("cluster sites"));
        }
        if points.is_empty() {
            return Err(GreenError::Empty("frequency grid"));
        }
        let mut previous = None;
        for (index, point) in points.iter().enumerate() {
            if !point.frequency.re.is_finite() || !point.frequency.im.is_finite() {
                return Err(GreenError::NonFinite("frequency grid"));
            }
            if point.frequency.im <= 0.0 {
                return Err(GreenError::NonPositiveBroadening { index });
            }
            if previous.is_some_and(|value| point.frequency.re <= value) {
                return Err(GreenError::UnorderedFrequencyGrid { index });
            }
            previous = Some(point.frequency.re);
            point
                .green
                .require_dimensions(sites, sites, "cluster Green matrix")?;
            let report = matrix_causality(&point.green, tolerance)?;
            if !report.causal {
                return Err(GreenError::NonCausal {
                    index,
                    minimum_pivot: report.minimum_pivot,
                });
            }
        }
        Ok(Self { sites, points })
    }

    /// Return the number of cluster sites.
    pub const fn sites(&self) -> usize {
        self.sites
    }

    /// Return the checked grid points.
    pub fn points(&self) -> &[GreenPoint] {
        &self.points
    }

    /// Apply the CPT/Dyson embedding for one momentum-dependent hopping matrix.
    ///
    /// Computes `G(k,z) = [G_cluster(z)^-1 - V(k)]^-1` independently at
    /// every frequency. `V(k)` must be finite, square, and Hermitian.
    pub fn cpt_at(&self, intercluster_hopping: &DenseMatrix) -> Result<EmbeddedGreenGrid> {
        intercluster_hopping.require_dimensions(self.sites, self.sites, "inter-cluster hopping")?;
        if !intercluster_hopping.is_hermitian(DEFAULT_CAUSALITY_TOLERANCE)? {
            return Err(GreenError::NonHermitian("inter-cluster hopping"));
        }
        let mut points = Vec::with_capacity(self.points.len());
        for point in &self.points {
            let inverse_cluster = point.green.inverse()?;
            let embedded_inverse = inverse_cluster.subtract(intercluster_hopping)?;
            points.push(GreenPoint::new(
                point.frequency,
                embedded_inverse.inverse()?,
            ));
        }
        Ok(EmbeddedGreenGrid {
            sites: self.sites,
            points,
        })
    }
}

impl EmbeddedGreenGrid {
    /// Return the number of cluster sites.
    pub const fn sites(&self) -> usize {
        self.sites
    }

    /// Return the embedded frequency-grid points.
    pub fn points(&self) -> &[GreenPoint] {
        &self.points
    }

    /// Return `-Im Tr G/(pi N)` at a frequency-grid index.
    pub fn spectral_function(&self, index: usize) -> Result<f64> {
        let point = self.point(index)?;
        let spectral = -point.green.trace()?.im / (PI * self.sites as f64);
        if spectral.is_finite() {
            Ok(spectral)
        } else {
            Err(GreenError::NonFinite("spectral function"))
        }
    }

    /// Check matrix-valued causality at every embedded frequency.
    pub fn causality(&self, tolerance: f64) -> Result<Vec<CausalityReport>> {
        validate_tolerance(tolerance)?;
        self.points
            .iter()
            .map(|point| matrix_causality(&point.green, tolerance))
            .collect()
    }

    /// Periodize the Green function at one frequency using explicit positions.
    ///
    /// Returns `sum_ab exp(-i k·(r_a-r_b)) G_ab / N`.
    pub fn periodized_green(
        &self,
        index: usize,
        positions: &[[f64; 3]],
        wavevector: [f64; 3],
    ) -> Result<Complex64> {
        let point = self.point(index)?;
        if positions.len() != self.sites {
            return Err(GreenError::Shape {
                name: "cluster-site positions",
                expected: self.sites,
                actual: positions.len(),
            });
        }
        if !positions
            .iter()
            .flatten()
            .chain(&wavevector)
            .all(|value| value.is_finite())
        {
            return Err(GreenError::NonFinite("positions and wavevector"));
        }
        let mut result = Complex64::new(0.0, 0.0);
        for a in 0..self.sites {
            for b in 0..self.sites {
                let displacement = [
                    positions[a][0] - positions[b][0],
                    positions[a][1] - positions[b][1],
                    positions[a][2] - positions[b][2],
                ];
                let phase = -displacement
                    .iter()
                    .zip(wavevector)
                    .map(|(position, wavevector)| position * wavevector)
                    .sum::<f64>();
                result += Complex64::new(phase.cos(), phase.sin())
                    * point.green.get(a, b).expect("checked dimensions");
            }
        }
        Ok(result / self.sites as f64)
    }

    /// Return `-Im G_periodized/pi` at one frequency and momentum.
    pub fn periodized_spectral_function(
        &self,
        index: usize,
        positions: &[[f64; 3]],
        wavevector: [f64; 3],
    ) -> Result<f64> {
        let value = -self.periodized_green(index, positions, wavevector)?.im / PI;
        if value.is_finite() {
            Ok(value)
        } else {
            Err(GreenError::NonFinite("periodized spectral function"))
        }
    }

    fn point(&self, index: usize) -> Result<&GreenPoint> {
        self.points.get(index).ok_or(GreenError::IndexOutOfBounds {
            name: "frequency grid",
            index,
            length: self.points.len(),
        })
    }
}

fn matrix_causality(green: &DenseMatrix, tolerance: f64) -> Result<CausalityReport> {
    validate_tolerance(tolerance)?;
    green.require_square("causality matrix")?;
    let size = green.rows();
    // Spectral numerator A = (G† - G)/(2i) = -Im_H G.
    let mut spectral = vec![Complex64::new(0.0, 0.0); size * size];
    for row in 0..size {
        for column in 0..size {
            let g_dagger = green.get(column, row).expect("square matrix").conj();
            let g = green.get(row, column).expect("square matrix");
            spectral[row * size + column] = (g_dagger - g) / Complex64::new(0.0, 2.0);
        }
    }
    let spectral_scale = spectral
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(component_max(*value)));
    let threshold = tolerance * spectral_scale;

    // Unpivoted LDL† is sufficient for positive semidefinite Hermitian A.
    // A zero pivot must have a zero residual column; otherwise A is not PSD.
    let mut lower = vec![Complex64::new(0.0, 0.0); size * size];
    let mut diagonal = vec![0.0; size];
    let mut minimum_pivot = f64::INFINITY;
    for column in 0..size {
        let correction: f64 = (0..column)
            .map(|k| lower[column * size + k].norm_sqr() * diagonal[k])
            .sum();
        let pivot_value = spectral[column * size + column] - Complex64::from(correction);
        if pivot_value.im.abs() > threshold {
            return Ok(CausalityReport {
                causal: false,
                minimum_pivot: pivot_value.re.min(-pivot_value.im.abs()),
            });
        }
        let pivot = pivot_value.re;
        minimum_pivot = minimum_pivot.min(pivot);
        if pivot < -threshold {
            return Ok(CausalityReport {
                causal: false,
                minimum_pivot,
            });
        }
        if pivot.abs() <= threshold {
            diagonal[column] = 0.0;
            for row in column + 1..size {
                let correction: Complex64 = (0..column)
                    .map(|k| lower[row * size + k] * lower[column * size + k].conj() * diagonal[k])
                    .sum();
                if component_max(spectral[row * size + column] - correction) > threshold {
                    return Ok(CausalityReport {
                        causal: false,
                        minimum_pivot: minimum_pivot.min(-threshold),
                    });
                }
            }
        } else {
            diagonal[column] = pivot;
            lower[column * size + column] = Complex64::new(1.0, 0.0);
            for row in column + 1..size {
                let correction: Complex64 = (0..column)
                    .map(|k| lower[row * size + k] * lower[column * size + k].conj() * diagonal[k])
                    .sum();
                lower[row * size + column] = (spectral[row * size + column] - correction) / pivot;
            }
        }
    }
    Ok(CausalityReport {
        causal: true,
        minimum_pivot,
    })
}
