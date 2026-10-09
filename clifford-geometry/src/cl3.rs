//! Cl(3,0) full 8-component multivector — HarmonicRust compatible.
//!
//! This module provides the same dense 8-component multivector as
//! HarmonicRust's `clifford_cl3.rs`, but with product tables auto-generated
//! from the algebra rather than hand-coded.
//!
//! Basis ordering: `[1, e1, e2, e3, e12, e23, e13, e123]`
//! (matches HarmonicRust convention)

use crate::mvec::Multivector;
use crate::products::{self, DenseTable};
use std::sync::OnceLock;

// ============================================================================
// Metric and basis
// ============================================================================

/// Euclidean Cl(3,0) metric coefficients.
pub const METRIC: [i32; 3] = [1, 1, 1];

/// Full basis in HarmonicRust order: `[1, e1, e2, e3, e12, e23, e13, e123]`
/// Note: this is NOT sorted by grade then value — it matches HarmonicRust.
pub const BASIS: [u32; 8] = [
    0b000, // 1 (scalar)
    0b001, // e1
    0b010, // e2
    0b100, // e3
    0b011, // e12
    0b110, // e23
    0b101, // e13
    0b111, // e123
];

/// Grade of each basis element.
pub const GRADES: [u32; 8] = [0, 1, 1, 1, 2, 2, 2, 3];

/// Reverse signs: grades 0,1 → +1, grades 2,3 → −1.
pub const REVERSE_SIGNS: [f32; 8] = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];

// ============================================================================
// Type
// ============================================================================

/// Full Cl(3,0) multivector: 8 components.
/// Drop-in compatible with HarmonicRust's `pub type Multivector = [f32; 8]`.
pub type Cl3 = Multivector<8>;

// ============================================================================
// Constructors
// ============================================================================

#[inline]
/// Construct a scalar multivector.
pub fn scalar(s: f32) -> Cl3 {
    let mut mv = Cl3::zero();
    mv[0] = s;
    mv
}

#[inline]
/// Construct a grade-one vector in `(e1, e2, e3)` order.
pub fn vector(e1: f32, e2: f32, e3: f32) -> Cl3 {
    Multivector::new([0.0, e1, e2, e3, 0.0, 0.0, 0.0, 0.0])
}

#[inline]
/// Construct a grade-two bivector in the legacy `(e12, e23, e13)` order.
pub fn bivector(e12: f32, e23: f32, e13: f32) -> Cl3 {
    Multivector::new([0.0, 0.0, 0.0, 0.0, e12, e23, e13, 0.0])
}

#[inline]
/// Construct a grade-three pseudoscalar.
pub fn pseudoscalar(e123: f32) -> Cl3 {
    let mut mv = Cl3::zero();
    mv[7] = e123;
    mv
}

// ============================================================================
// Product table
// ============================================================================

/// Dense 8×8→8 geometric product table.
/// Equivalent to HarmonicRust's `build_geometric_product_table()`.
fn gp_table() -> &'static DenseTable<8, 8, 8> {
    static TABLE: OnceLock<DenseTable<8, 8, 8>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&BASIS, &BASIS, &BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

// ============================================================================
// Operations
// ============================================================================

/// Geometric product: c = a * b.
/// Drop-in replacement for HarmonicRust's `geometric_product()`.
#[inline]
pub fn geometric_product(a: &Cl3, b: &Cl3) -> Cl3 {
    Multivector::new(gp_table().execute(&a.data, &b.data))
}

/// Commutator: `[a, b]` = a*b - b*a.
/// Selects grade-2 from bivector×bivector, grade-1 from vector×scalar, etc.
/// CANNOT produce grade-3 in Cl(3,0) because e₁₂₃ is central.
#[inline]
pub fn commutator(a: &Cl3, b: &Cl3) -> Cl3 {
    let ab = geometric_product(a, b);
    let ba = geometric_product(b, a);
    ab - ba
}

/// Anti-commutator: {a, b} = a*b + b*a.
/// Selects grade-0 + grade-4 from bivector×bivector, etc.
/// Crucially: {vector, bivector} → grade-1 + GRADE-3.
/// This is the ONLY algebraic path to grade-3 in Cl(3,0),
/// because e₁₂₃ commutes with all elements (it's central).
#[inline]
pub fn anti_commutator(a: &Cl3, b: &Cl3) -> Cl3 {
    let ab = geometric_product(a, b);
    let ba = geometric_product(b, a);
    ab + ba
}

/// Grade reversal.
/// Drop-in replacement for HarmonicRust's `clifford_reverse()`.
#[inline]
pub fn reverse(x: &Cl3) -> Cl3 {
    let mut out = *x;
    for i in 0..8 {
        out[i] *= REVERSE_SIGNS[i];
    }
    out
}

/// Grade projection: extract components of a specific grade.
pub fn grade_project(x: &Cl3, target_grade: u32) -> Cl3 {
    let mut out = Cl3::zero();
    for i in 0..8 {
        if GRADES[i] == target_grade {
            out[i] = x[i];
        }
    }
    out
}

/// Scalar part.
#[inline]
pub fn scalar_part(x: &Cl3) -> f32 {
    x[0]
}

/// Vector part `[e1, e2, e3]`.
#[inline]
pub fn vector_part(x: &Cl3) -> [f32; 3] {
    [x[1], x[2], x[3]]
}

/// Bivector part `[e12, e23, e13]`.
#[inline]
pub fn bivector_part(x: &Cl3) -> [f32; 3] {
    [x[4], x[5], x[6]]
}

/// Pseudoscalar part.
#[inline]
pub fn pseudoscalar_part(x: &Cl3) -> f32 {
    x[7]
}

/// Norm squared: x * reverse(x), scalar part.
#[inline]
pub fn norm_sq(x: &Cl3) -> f32 {
    scalar_part(&geometric_product(x, &reverse(x)))
}

/// Convert to raw `[f32; 8]` array (for HarmonicRust interop).
#[inline]
pub fn to_array(x: &Cl3) -> [f32; 8] {
    x.data
}

/// Convert from raw `[f32; 8]` array (for HarmonicRust interop).
#[inline]
pub fn from_array(arr: [f32; 8]) -> Cl3 {
    Multivector::new(arr)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basis_vectors_square_to_one() {
        let e1 = vector(1.0, 0.0, 0.0);
        let e1e1 = geometric_product(&e1, &e1);
        assert!((e1e1[0] - 1.0).abs() < 1e-6, "e1*e1 scalar = {}", e1e1[0]);
        // All other components should be zero
        for i in 1..8 {
            assert!(e1e1[i].abs() < 1e-6, "e1*e1[{}] = {}", i, e1e1[i]);
        }
    }

    #[test]
    fn test_e1_e2_product() {
        let e1 = vector(1.0, 0.0, 0.0);
        let e2 = vector(0.0, 1.0, 0.0);
        let e1e2 = geometric_product(&e1, &e2);
        // e1*e2 = +e12 (index 4 in our basis order)
        assert!((e1e2[4] - 1.0).abs() < 1e-6, "e12 = {}", e1e2[4]);
    }

    #[test]
    fn test_e2_e1_product() {
        let e1 = vector(1.0, 0.0, 0.0);
        let e2 = vector(0.0, 1.0, 0.0);
        let e2e1 = geometric_product(&e2, &e1);
        // e2*e1 = -e12
        assert!((e2e1[4] + 1.0).abs() < 1e-6, "e12 = {}", e2e1[4]);
    }

    #[test]
    fn test_commutator_antisymmetric() {
        let a = bivector(1.0, 0.0, 0.0);
        let b = bivector(0.0, 1.0, 0.0);
        let ab = commutator(&a, &b);
        let ba = commutator(&b, &a);
        for i in 0..8 {
            assert!((ab[i] + ba[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_commutator_self_zero() {
        let a = from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let comm = commutator(&a, &a);
        for i in 0..8 {
            assert!(comm[i].abs() < 1e-5, "[a,a][{}] = {}", i, comm[i]);
        }
    }

    #[test]
    fn test_reverse() {
        let a = from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let rev = reverse(&a);
        // Scalar and vectors unchanged
        assert_eq!(rev[0], 1.0);
        assert_eq!(rev[1], 2.0);
        assert_eq!(rev[2], 3.0);
        assert_eq!(rev[3], 4.0);
        // Bivectors and trivector negated
        assert_eq!(rev[4], -5.0);
        assert_eq!(rev[5], -6.0);
        assert_eq!(rev[6], -7.0);
        assert_eq!(rev[7], -8.0);
    }

    #[test]
    fn test_grade_projection() {
        let a = from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let grade0 = grade_project(&a, 0);
        assert_eq!(grade0[0], 1.0);
        assert_eq!(grade0[1], 0.0);

        let grade1 = grade_project(&a, 1);
        assert_eq!(grade1[0], 0.0);
        assert_eq!(grade1[1], 2.0);
        assert_eq!(grade1[2], 3.0);
        assert_eq!(grade1[3], 4.0);
        assert_eq!(grade1[4], 0.0);

        let grade2 = grade_project(&a, 2);
        assert_eq!(grade2[4], 5.0);
        assert_eq!(grade2[5], 6.0);
        assert_eq!(grade2[6], 7.0);
    }

    #[test]
    fn test_array_roundtrip() {
        let arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mv = from_array(arr);
        assert_eq!(to_array(&mv), arr);
    }

    #[test]
    fn test_pseudoscalar_squares_to_minus_one() {
        let ps = pseudoscalar(1.0);
        let ps2 = geometric_product(&ps, &ps);
        // e123 * e123 = -1 in Cl(3,0)
        assert!((ps2[0] + 1.0).abs() < 1e-6, "e123² scalar = {}", ps2[0]);
    }
}
