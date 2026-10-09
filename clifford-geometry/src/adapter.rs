//! Explicit conversion and convention adapters for [`clifford_core`].
//!
//! Conversions use blade bitmaps, never storage indices. This is important for
//! legacy Cl(3,0) and STA layouts whose coefficient order differs from
//! `clifford-core` ShortLex order.

use crate::mvec::Multivector;
use clifford_core::CliffordAlgebra;
use std::fmt;

/// Error returned when a sparse/dense representation cannot be converted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterError {
    /// A sparse blade bitmap is not part of the target algebra.
    BladeOutsideAlgebra {
        /// Unsigned blade bitmap that could not be represented.
        bitmap: u32,
        /// Vector-space dimension of the target algebra.
        dimension: usize,
    },
    /// A sparse basis contains the same blade more than once.
    DuplicateBlade {
        /// Unsigned blade bitmap repeated by the sparse layout.
        bitmap: u32,
    },
    /// A dense input does not have one coefficient per target blade.
    DenseLength {
        /// Required number of dense coefficients.
        expected: usize,
        /// Number of coefficients supplied by the caller.
        actual: usize,
    },
}

impl fmt::Display for AdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BladeOutsideAlgebra { bitmap, dimension } => write!(
                formatter,
                "blade bitmap {bitmap} is outside a {dimension}-dimensional algebra"
            ),
            Self::DuplicateBlade { bitmap } => {
                write!(formatter, "blade bitmap {bitmap} occurs more than once")
            }
            Self::DenseLength { expected, actual } => write!(
                formatter,
                "dense multivector length mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for AdapterError {}

/// Convert a fixed-size sparse Versor value to `clifford-core` dense ShortLex
/// storage using the supplied blade bitmap layout.
pub fn sparse_to_core<const N: usize>(
    algebra: &CliffordAlgebra,
    basis: &[u32; N],
    value: &Multivector<N>,
) -> Result<Vec<f32>, AdapterError> {
    let mut dense = vec![0.0; algebra.n_blades];
    let mut seen = vec![false; algebra.n_blades];

    for (sparse_index, &bitmap) in basis.iter().enumerate() {
        let bitmap_index = bitmap as usize;
        if bitmap_index >= algebra.n_blades {
            return Err(AdapterError::BladeOutsideAlgebra {
                bitmap,
                dimension: algebra.dim,
            });
        }
        let dense_index = algebra.bitmap_to_index[bitmap_index];
        if seen[dense_index] {
            return Err(AdapterError::DuplicateBlade { bitmap });
        }
        seen[dense_index] = true;
        dense[dense_index] = value[sparse_index];
    }

    Ok(dense)
}

/// Project a `clifford-core` dense ShortLex value into a fixed-size sparse
/// Versor layout using blade bitmaps.
pub fn sparse_from_core<const N: usize>(
    algebra: &CliffordAlgebra,
    basis: &[u32; N],
    dense: &[f32],
) -> Result<Multivector<N>, AdapterError> {
    if dense.len() != algebra.n_blades {
        return Err(AdapterError::DenseLength {
            expected: algebra.n_blades,
            actual: dense.len(),
        });
    }

    let mut sparse = [0.0; N];
    let mut seen = vec![false; algebra.n_blades];
    for (sparse_index, &bitmap) in basis.iter().enumerate() {
        let bitmap_index = bitmap as usize;
        if bitmap_index >= algebra.n_blades {
            return Err(AdapterError::BladeOutsideAlgebra {
                bitmap,
                dimension: algebra.dim,
            });
        }
        let dense_index = algebra.bitmap_to_index[bitmap_index];
        if seen[dense_index] {
            return Err(AdapterError::DuplicateBlade { bitmap });
        }
        seen[dense_index] = true;
        sparse[sparse_index] = dense[dense_index];
    }

    Ok(Multivector::new(sparse))
}

/// Evaluate the raw `AB - BA` convention used by this crate through
/// `clifford-core`, whose native commutator includes a factor of one half.
pub fn core_raw_commutator(algebra: &CliffordAlgebra, a: &[f32], b: &[f32]) -> Vec<f32> {
    algebra
        .commutator(a, b)
        .into_iter()
        .map(|coefficient| coefficient * 2.0)
        .collect()
}

/// Construct the `clifford-core` rotor matching Versor's `exp(-B*angle/2)`
/// convention for a simple normalized Euclidean bivector.
///
/// This delegates to `clifford-core`'s closed form and therefore is not a
/// general exponential for non-simple bivectors or arbitrary signatures.
pub fn core_rotor_matching_versor(
    algebra: &CliffordAlgebra,
    bivector: &[f32],
    angle: f32,
) -> Vec<f32> {
    algebra.rotor_from_bivector(bivector, -angle)
}
