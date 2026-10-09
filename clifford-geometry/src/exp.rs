//! Bivector exponentiation for Clifford algebras.
//!
//! Computes R = exp(-B/2) for bivectors using Taylor series expansion.
//! The exponential of a bivector always lives in the even subalgebra.
//!
//! Supports:
//! - Cl(3,0): 3-component bivector → 4-component rotor (with closed-form verification)
//! - Cl(6,0): 15-component bivector → 32-component even-grade rotor

use crate::basis;
use crate::mvec::Multivector;
use crate::products::{self, DenseTable};
use std::sync::OnceLock;

// ============================================================================
// Cl(3,0) — 3D Euclidean (for testing against known closed-form)
// ============================================================================

/// Cl(3,0) metric.
const CL3_METRIC: [i32; 3] = [1, 1, 1];

/// Cl(3,0) even subalgebra (rotor) basis: `[1, e12, e13, e23]`
/// Matches ega3d::ROT_BASIS ordering.
const CL3_EVEN_BASIS: [u32; 4] = [0b000, 0b011, 0b101, 0b110];

/// Cl(3,0) bivector basis: `[e12, e13, e23]`
/// Matches ega3d::BIV_BASIS ordering.
#[allow(dead_code)]
const CL3_BIV_BASIS: [u32; 3] = [0b011, 0b101, 0b110];

/// Even * Even geometric product for Cl(3,0): 4x4 → 4
fn cl3_even_even_gp() -> &'static DenseTable<4, 4, 4> {
    static TABLE: OnceLock<DenseTable<4, 4, 4>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(
            &CL3_EVEN_BASIS,
            &CL3_EVEN_BASIS,
            &CL3_EVEN_BASIS,
            &CL3_METRIC,
        );
        DenseTable::from_instructions(&insts)
    })
}

/// Embed a Cl(3,0) bivector (3 components) into the even subalgebra (4 components).
/// Bivector `[e12, e13, e23]` maps to even `[1, e12, e13, e23]` at indices 1..3.
pub fn embed_biv_in_even_cl3(b: &Multivector<3>) -> Multivector<4> {
    Multivector::new([0.0, b[0], b[1], b[2]])
}

/// Extract the bivector part from a Cl(3,0) even element.
pub fn extract_biv_from_even_cl3(e: &Multivector<4>) -> Multivector<3> {
    Multivector::new([e[1], e[2], e[3]])
}

/// Extract scalar part from a Cl(3,0) even element.
pub fn extract_scalar_from_even_cl3(e: &Multivector<4>) -> f32 {
    e[0]
}

/// Reverse of a Cl(3,0) even element: negate grade-2 (bivector) part.
#[allow(dead_code)]
fn reverse_even_cl3(e: &Multivector<4>) -> Multivector<4> {
    Multivector::new([e[0], -e[1], -e[2], -e[3]])
}

/// Geometric product of two Cl(3,0) even elements.
fn gp_even_even_cl3(a: &Multivector<4>, b: &Multivector<4>) -> Multivector<4> {
    Multivector::new(cl3_even_even_gp().execute(&a.data, &b.data))
}

/// exp(B) for a Cl(3,0) bivector (full exponential, no -1/2 factor).
/// Uses Taylor series, verifiable against closed-form: cos|B| + sin|B|/|B| * B.
pub fn exp_biv_cl3_full(b: &Multivector<3>) -> Multivector<4> {
    let x = embed_biv_in_even_cl3(b);
    exp_even_taylor_cl3(&x)
}

/// exp(-B/2) for a Cl(3,0) bivector → 4-component rotor.
/// Can be verified against: cos(|B|/2) - sin(|B|/2)/|B| * B.
pub fn exp_biv_cl3(b: &Multivector<3>) -> Multivector<4> {
    let x = embed_biv_in_even_cl3(&(*b * -0.5));
    exp_even_taylor_cl3(&x)
}

/// Taylor series exp for a Cl(3,0) even-subalgebra element.
fn exp_even_taylor_cl3(x: &Multivector<4>) -> Multivector<4> {
    let mut result = Multivector::<4>::zero();
    result[0] = 1.0; // identity

    let mut term = result; // term_0 = 1
    let eps = 1e-12_f32;
    let max_iter = 20;

    for n in 1..=max_iter {
        term = gp_even_even_cl3(&term, x);
        term = term * (1.0 / n as f32);
        result = result + term;
        if term.norm_sq() < eps * eps {
            break;
        }
    }
    result
}

// ============================================================================
// Cl(6,0) — 6D Euclidean
// ============================================================================

/// Cl(6,0) metric: all basis vectors square to +1.
const CL6_METRIC: [i32; 6] = [1, 1, 1, 1, 1, 1];

/// Generate all blades of a given grade from `dim` basis vectors.
/// Blades are sorted by bitmask value.
fn blades_of_grade(dim: u32, target_grade: u32) -> Vec<u32> {
    let total = 1u32 << dim;
    let mut result = Vec::new();
    for blade in 0..total {
        if basis::grade(blade) == target_grade {
            result.push(blade);
        }
    }
    result.sort();
    result
}

/// Build the even-subalgebra basis for Cl(6,0): grades 0, 2, 4, 6.
/// Total: C(6,0) + C(6,2) + C(6,4) + C(6,6) = 1 + 15 + 15 + 1 = 32.
fn cl6_even_basis() -> Vec<u32> {
    let mut basis = Vec::new();
    for grade in [0, 2, 4, 6] {
        basis.extend(blades_of_grade(6, grade));
    }
    basis
}

/// Build the bivector basis for Cl(6,0): grade 2.
/// Total: C(6,2) = 15.
fn cl6_biv_basis() -> Vec<u32> {
    blades_of_grade(6, 2)
}

/// Cached Cl(6,0) even basis (32 blades).
fn cl6_even_basis_cached() -> &'static Vec<u32> {
    static BASIS: OnceLock<Vec<u32>> = OnceLock::new();
    BASIS.get_or_init(cl6_even_basis)
}

/// Cached Cl(6,0) bivector basis (15 blades).
fn cl6_biv_basis_cached() -> &'static Vec<u32> {
    static BASIS: OnceLock<Vec<u32>> = OnceLock::new();
    BASIS.get_or_init(cl6_biv_basis)
}

/// Even * Even geometric product for Cl(6,0): 32x32 → 32
fn cl6_even_even_gp() -> &'static DenseTable<32, 32, 32> {
    static TABLE: OnceLock<Box<DenseTable<32, 32, 32>>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let even = cl6_even_basis_cached();
        let insts = products::gen_geometric_product(even, even, even, &CL6_METRIC);
        Box::new(DenseTable::from_instructions(&insts))
    })
}

/// Geometric product of two Cl(6,0) even elements.
pub fn gp_even_even_cl6(a: &Multivector<32>, b: &Multivector<32>) -> Multivector<32> {
    Multivector::new(cl6_even_even_gp().execute(&a.data, &b.data))
}

/// Embed a Cl(6,0) bivector (15 components) into the even subalgebra (32 components).
/// The bivector occupies indices 1..=15 in the even basis (after the scalar at index 0).
pub fn embed_biv_in_even(b: &Multivector<15>) -> Multivector<32> {
    let mut result = Multivector::<32>::zero();
    // Even basis is [scalar, 15 bivectors, 15 quadvectors, pseudoscalar].
    // Bivector components go at indices 1..=15.
    for i in 0..15 {
        result[1 + i] = b[i];
    }
    result
}

/// Extract bivector part (grade-2) from a Cl(6,0) even element.
pub fn extract_biv_from_even(e: &Multivector<32>) -> Multivector<15> {
    let mut result = Multivector::<15>::zero();
    for i in 0..15 {
        result[i] = e[1 + i];
    }
    result
}

/// Extract scalar part from a Cl(6,0) even element.
pub fn extract_scalar_from_even(e: &Multivector<32>) -> f32 {
    e[0]
}

/// Reverse of a Cl(6,0) even element.
/// Grade 0: sign +1, Grade 2: sign -1, Grade 4: sign +1, Grade 6: sign -1.
pub fn reverse_even_cl6(e: &Multivector<32>) -> Multivector<32> {
    let even_basis = cl6_even_basis_cached();
    let mut result = Multivector::<32>::zero();
    for (i, &blade) in even_basis.iter().enumerate() {
        result[i] = e[i] * basis::reverse_sign(blade) as f32;
    }
    result
}

/// Taylor series exp for a Cl(6,0) even-subalgebra element.
/// Computes: 1 + X + X^2/2! + X^3/3! + ...
pub fn exp_even_cl6(a: &Multivector<32>) -> Multivector<32> {
    let mut result = Multivector::<32>::zero();
    result[0] = 1.0; // identity (scalar = 1)

    let mut term = result; // term_0 = 1
    let eps = 1e-10_f32;
    let max_iter = 24; // For Cl(6,0), series terminates by degree 6 for pure bivectors

    for n in 1..=max_iter {
        term = gp_even_even_cl6(&term, a);
        term = term * (1.0 / n as f32);
        result = result + term;
        if term.norm_sq() < eps * eps {
            break;
        }
    }
    result
}

/// exp(B) for a Cl(6,0) bivector (full exponential, no -1/2 factor).
pub fn exp_biv_cl6_full(b: &Multivector<15>) -> Multivector<32> {
    let x = embed_biv_in_even(b);
    exp_even_cl6(&x)
}

/// exp(-B/2) for a Cl(6,0) bivector → 32-component even-grade rotor.
pub fn exp_biv_cl6(b: &Multivector<15>) -> Multivector<32> {
    let x = embed_biv_in_even(&(*b * -0.5));
    exp_even_cl6(&x)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // --- Cl(3,0) tests ---

    #[test]
    fn test_exp_identity() {
        // exp(0) = 1 for Cl(3,0)
        let zero = Multivector::<3>::zero();
        let r = exp_biv_cl3_full(&zero);
        assert!((r[0] - 1.0).abs() < 1e-6, "scalar = {}", r[0]);
        for i in 1..4 {
            assert!(r[i].abs() < 1e-6, "r[{}] = {}", i, r[i]);
        }
    }

    #[test]
    fn test_exp_cl3_matches_formula() {
        // For Cl(3,0): exp(B) = cos|B| + sin|B|/|B| * B
        let b = Multivector::new([0.5, 0.3, 0.7]); // bivector in Cl(3,0)
        let result = exp_biv_cl3_full(&b);

        let norm_b = b.norm();
        let cos_b = norm_b.cos();
        let sin_b = norm_b.sin();

        // Expected: scalar = cos|B|, bivector = sin|B|/|B| * B
        let expected_scalar = cos_b;
        let expected_biv = if norm_b > 1e-10 { sin_b / norm_b } else { 1.0 };

        assert!(
            (result[0] - expected_scalar).abs() < 1e-5,
            "scalar: got {} expected {}",
            result[0],
            expected_scalar
        );
        for i in 0..3 {
            let expected = expected_biv * b[i];
            assert!(
                (result[1 + i] - expected).abs() < 1e-5,
                "biv[{}]: got {} expected {}",
                i,
                result[1 + i],
                expected
            );
        }
    }

    #[test]
    fn test_exp_cl3_half_angle() {
        // exp(-B/2) should match cos(|B|/2) - sin(|B|/2)/|B| * B
        let b = Multivector::new([PI / 2.0, 0.0, 0.0]);
        let r = exp_biv_cl3(&b);

        let half = b.norm() / 2.0;
        let cos_h = half.cos();
        let sin_h = half.sin();
        let norm_b = b.norm();

        assert!(
            (r[0] - cos_h).abs() < 1e-5,
            "scalar: got {} expected {}",
            r[0],
            cos_h
        );
        // Bivector part: -sin(|B|/2)/|B| * B (the -0.5 scaling produces -B/2,
        // and exp(-B/2) = cos(|B|/2) + sin(|B|/2)/|B/2| * (-B/2))
        // = cos(|B|/2) - sin(|B|/2)/|B| * B
        for i in 0..3 {
            let expected = -sin_h / norm_b * b[i];
            assert!(
                (r[1 + i] - expected).abs() < 1e-5,
                "biv[{}]: got {} expected {}",
                i,
                r[1 + i],
                expected
            );
        }
    }

    // --- Cl(6,0) tests ---

    #[test]
    fn test_exp_cl6_identity() {
        // exp(0) = 1 for Cl(6,0)
        let zero = Multivector::<15>::zero();
        let r = exp_biv_cl6_full(&zero);
        assert!((r[0] - 1.0).abs() < 1e-6, "scalar = {}", r[0]);
        for i in 1..32 {
            assert!(r[i].abs() < 1e-6, "r[{}] = {}", i, r[i]);
        }
    }

    #[test]
    fn test_exp_cl6_normalization() {
        // R * ~R should equal 1 (scalar part = 1, all others ≈ 0)
        let mut b = Multivector::<15>::zero();
        b[0] = 0.5; // e12
        b[3] = 0.3; // e15
        b[10] = 0.7; // e35
        let r = exp_biv_cl6(&b);
        let r_rev = reverse_even_cl6(&r);
        let rr = gp_even_even_cl6(&r, &r_rev);

        assert!(
            (rr[0] - 1.0).abs() < 1e-4,
            "R*~R scalar = {} (expected 1.0)",
            rr[0]
        );
        for i in 1..32 {
            assert!(rr[i].abs() < 1e-4, "R*~R[{}] = {} (expected 0.0)", i, rr[i]);
        }
    }

    #[test]
    fn test_exp_cl6_small_angle() {
        // For small B, exp(-B/2) ≈ 1 - B/2
        let mut b = Multivector::<15>::zero();
        b[0] = 0.001;
        b[5] = 0.002;
        let r = exp_biv_cl6(&b);

        // Scalar should be very close to 1
        assert!(
            (r[0] - 1.0).abs() < 1e-4,
            "scalar = {} (expected ~1.0)",
            r[0]
        );
        // Bivector parts should be ≈ -b/2
        assert!(
            (r[1] - (-0.001 / 2.0)).abs() < 1e-5,
            "biv[0] = {} (expected {})",
            r[1],
            -0.001 / 2.0
        );
        assert!(
            (r[6] - (-0.002 / 2.0)).abs() < 1e-5,
            "biv[5] = {} (expected {})",
            r[6],
            -0.002 / 2.0
        );
    }

    #[test]
    fn test_exp_cl6_t8() {
        // exp(-T8/2) should be a valid rotor.
        // T8 is a diagonal generator of SU(3) ⊂ Cl(6,0).
        // Use a simple bivector in a single plane as a stand-in.
        let mut t8 = Multivector::<15>::zero();
        t8[0] = 1.0; // e12 plane
        let r = exp_biv_cl6(&t8);
        let r_rev = reverse_even_cl6(&r);
        let rr = gp_even_even_cl6(&r, &r_rev);

        assert!(
            (rr[0] - 1.0).abs() < 1e-4,
            "T8 rotor normalization failed: scalar = {}",
            rr[0]
        );
        for i in 1..32 {
            assert!(
                rr[i].abs() < 1e-4,
                "T8 rotor normalization failed: rr[{}] = {}",
                i,
                rr[i]
            );
        }
    }

    #[test]
    fn test_exp_cl6_sandwich_preserves_norm() {
        // For a simple bivector, R*B*~R should have the same norm as B.
        // We use even*even products: embed B in even, compute R*B_even*~R.
        let mut gen = Multivector::<15>::zero();
        gen[0] = 0.8;
        gen[7] = 0.4;
        let r = exp_biv_cl6(&gen);
        let r_rev = reverse_even_cl6(&r);

        // The "vector" to rotate is another bivector (lives in even subalgebra)
        let mut b = Multivector::<15>::zero();
        b[1] = 1.0;
        b[5] = 0.5;
        let b_even = embed_biv_in_even(&b);

        let rb = gp_even_even_cl6(&r, &b_even);
        let rbr = gp_even_even_cl6(&rb, &r_rev);

        // Extract the bivector part and check norm
        let b_rotated = extract_biv_from_even(&rbr);
        let orig_norm = b.norm();
        let rot_norm = b_rotated.norm();

        assert!(
            (orig_norm - rot_norm).abs() < 1e-3,
            "norm changed: {} → {}",
            orig_norm,
            rot_norm
        );
    }

    #[test]
    fn test_exp_composition() {
        // For small commuting bivectors, exp(A)*exp(B) ≈ exp(A+B)
        // Use two bivectors in orthogonal planes (they commute).
        let mut a = Multivector::<15>::zero();
        a[0] = 0.1; // e12 plane

        let mut b = Multivector::<15>::zero();
        b[9] = 0.1; // e45 plane (index depends on ordering; e45 = basis index 9)

        let exp_a = exp_biv_cl6_full(&a);
        let exp_b = exp_biv_cl6_full(&b);
        let exp_ab = gp_even_even_cl6(&exp_a, &exp_b);

        let ab = a + b;
        let exp_sum = exp_biv_cl6_full(&ab);

        // They should match because A and B are in orthogonal planes
        for i in 0..32 {
            assert!(
                (exp_ab[i] - exp_sum[i]).abs() < 1e-3,
                "composition[{}]: {} vs {}",
                i,
                exp_ab[i],
                exp_sum[i]
            );
        }
    }

    #[test]
    fn test_exp_periodicity() {
        // exp(-2π * B_hat) should equal identity for a unit simple bivector.
        // A simple bivector B_hat with |B_hat| = 1 in a single plane:
        // exp(2π * B_hat) = cos(2π) + sin(2π) * B_hat = 1
        let mut b_hat = Multivector::<15>::zero();
        b_hat[0] = 2.0 * PI; // 2π in the e12 plane
        let r = exp_biv_cl6_full(&b_hat);

        assert!(
            (r[0] - 1.0).abs() < 1e-3,
            "periodicity scalar = {} (expected 1.0)",
            r[0]
        );
        for i in 1..32 {
            assert!(
                r[i].abs() < 1e-3,
                "periodicity r[{}] = {} (expected 0.0)",
                i,
                r[i]
            );
        }
    }

    #[test]
    fn test_embed_extract_roundtrip() {
        let b = Multivector::new([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ]);
        let even = embed_biv_in_even(&b);
        let b2 = extract_biv_from_even(&even);
        for i in 0..15 {
            assert_eq!(b[i], b2[i], "roundtrip failed at index {}", i);
        }
        // Scalar should be zero
        assert_eq!(extract_scalar_from_even(&even), 0.0);
    }

    #[test]
    fn test_cl6_even_basis_sizes() {
        let even = cl6_even_basis_cached();
        assert_eq!(even.len(), 32, "even subalgebra should have 32 elements");

        let biv = cl6_biv_basis_cached();
        assert_eq!(biv.len(), 15, "bivector space should have 15 elements");

        // Verify grade distribution: 1 + 15 + 15 + 1 = 32
        let grade_counts: Vec<usize> = [0, 2, 4, 6]
            .iter()
            .map(|&g| even.iter().filter(|&&b| basis::grade(b) == g).count())
            .collect();
        assert_eq!(grade_counts, vec![1, 15, 15, 1]);
    }

    #[test]
    fn test_reverse_even_cl6_signs() {
        // Grade 0: +1, Grade 2: -1, Grade 4: +1, Grade 6: -1
        let mut e = Multivector::<32>::zero();
        e[0] = 1.0; // scalar (grade 0)
        e[1] = 1.0; // first bivector (grade 2)
        e[16] = 1.0; // first quadvector (grade 4)
        e[31] = 1.0; // pseudoscalar (grade 6)

        let rev = reverse_even_cl6(&e);
        assert_eq!(rev[0], 1.0, "grade 0 reverse");
        assert_eq!(rev[1], -1.0, "grade 2 reverse");
        assert_eq!(rev[16], 1.0, "grade 4 reverse");
        assert_eq!(rev[31], -1.0, "grade 6 reverse");
    }

    #[test]
    fn test_exp_cl6_multi_plane() {
        // Test with a bivector that has components in multiple planes
        let mut b = Multivector::<15>::zero();
        b[0] = 0.3; // e12
        b[5] = 0.4; // e24
        b[9] = 0.5; // e45
        b[14] = 0.2; // e56

        let r = exp_biv_cl6(&b);
        let r_rev = reverse_even_cl6(&r);
        let rr = gp_even_even_cl6(&r, &r_rev);

        // Must still be a valid rotor: R*~R = 1
        assert!(
            (rr[0] - 1.0).abs() < 1e-3,
            "multi-plane rotor normalization: scalar = {}",
            rr[0]
        );
        for i in 1..32 {
            assert!(
                rr[i].abs() < 1e-3,
                "multi-plane rotor normalization: rr[{}] = {}",
                i,
                rr[i]
            );
        }
    }
}
