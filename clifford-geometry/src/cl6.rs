//! Cl(6,0) — the flavor algebra for SU(3) geometric resonance.
//!
//! Metric: `[+1, +1, +1, +1, +1, +1]` (6 Euclidean dimensions).
//! 2^6 = 64 total basis blades.
//!
//! The 15 bivectors embed the 8 Gell-Mann matrices as Clifford bivectors,
//! enabling geometric evolution of flavor fields via the commutator bracket.
//!
//! SU(3) generators (as Cl(6,0) bivectors):
//!   Intra-generation: T1 = e02+e13, T2 = e12-e03, T3 = e01-e23
//!   Cross-generation: T4 = e04+e15, T5 = e14-e05, T6 = e24+e35, T7 = e34-e25
//!   Hypercharge:      T8 = e01+e23-2*e45  (||T8||² = 6, the anomalous norm)

use crate::mvec::Multivector;
use crate::products::{self, DenseTable};
use std::sync::OnceLock;

// ============================================================================
// Metric
// ============================================================================

/// Euclidean Cl(6,0) metric coefficients.
pub const METRIC: [i32; 6] = [1, 1, 1, 1, 1, 1];

// ============================================================================
// Basis blade enumeration
// ============================================================================

// Helper: all 6-bit bitmasks (0..63) sorted by grade then value.
// Grade 0: 1 blade, Grade 1: 6, Grade 2: 15, Grade 3: 20, Grade 4: 15, Grade 5: 6, Grade 6: 1

/// All 64 basis blades sorted by grade then bitmask value.
pub const FULL_BASIS: [u32; 64] = {
    let mut arr = [0u32; 64];
    let mut idx = 0;
    let mut g = 0u32;
    while g <= 6 {
        let mut b = 0u32;
        while b < 64 {
            if b.count_ones() == g {
                arr[idx] = b;
                idx += 1;
            }
            b += 1;
        }
        g += 1;
    }
    arr
};

/// The 15 grade-2 bivector blades, sorted by bitmask value.
pub const BIV_BASIS: [u32; 15] = {
    let mut arr = [0u32; 15];
    let mut idx = 0;
    let mut b = 0u32;
    while b < 64 {
        if b.count_ones() == 2 {
            arr[idx] = b;
            idx += 1;
        }
        b += 1;
    }
    arr
};

/// The 32 even-grade blades (grades 0, 2, 4, 6), sorted by grade then value.
pub const EVEN_BASIS: [u32; 32] = {
    let mut arr = [0u32; 32];
    let mut idx = 0;
    let mut g = 0u32;
    while g <= 6 {
        let mut b = 0u32;
        while b < 64 {
            if b.count_ones() == g {
                arr[idx] = b;
                idx += 1;
            }
            b += 1;
        }
        g += 2;
    }
    arr
};

/// Scalar basis.
pub const SCALAR_BASIS: [u32; 1] = [0];

/// The 15 grade-4 blades, sorted by bitmask value.
pub const QUADVEC_BASIS: [u32; 15] = {
    let mut arr = [0u32; 15];
    let mut idx = 0;
    let mut b = 0u32;
    while b < 64 {
        if b.count_ones() == 4 {
            arr[idx] = b;
            idx += 1;
        }
        b += 1;
    }
    arr
};

// ============================================================================
// Types (sparse multivectors)
// ============================================================================

/// Full Cl(6,0) multivector: 64 components.
pub type Cl6Full = Multivector<64>;

/// Bivector (grade 2): 15 components — the field variable F.
pub type Cl6Biv = Multivector<15>;

/// Even-grade rotor: 32 components (grades 0, 2, 4, 6).
pub type Cl6Rot = Multivector<32>;

/// Scalar: 1 component.
pub type Cl6Sca = Multivector<1>;

/// Quadvector (grade 4): 15 components.
pub type Cl6Quad = Multivector<15>;

// ============================================================================
// Product tables (OnceLock)
// ============================================================================

/// Biv × Biv → Full (15×15→64): full geometric product of two bivectors.
fn biv_biv_gp_table() -> &'static DenseTable<15, 15, 64> {
    static TABLE: OnceLock<DenseTable<15, 15, 64>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&BIV_BASIS, &BIV_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Biv × Biv → Biv (15×15→15): commutator table `[A,B]` = AB - BA projected to grade 2.
fn biv_biv_comm_table() -> &'static DenseTable<15, 15, 15> {
    static TABLE: OnceLock<DenseTable<15, 15, 15>> = OnceLock::new();
    TABLE.get_or_init(|| {
        // Generate the biv×biv→biv GP (grade-2 projection of the full product)
        let ab_insts = products::gen_geometric_product(&BIV_BASIS, &BIV_BASIS, &BIV_BASIS, &METRIC);
        let ab = DenseTable::<15, 15, 15>::from_instructions(&ab_insts);

        // Commutator: [A,B]_ijk = AB_ijk - AB_jik
        let mut table = [[[0.0f32; 15]; 15]; 15];
        for i in 0..15 {
            for j in 0..15 {
                for k in 0..15 {
                    table[i][j][k] = ab.table[i][j][k] - ab.table[j][i][k];
                }
            }
        }
        DenseTable { table }
    })
}

/// Biv × Biv → Quadvector (15×15→15): grade-4 projection of bivector product.
/// This gives the PHANTOM component: ⟨F²⟩₄ for the Higgs mechanism.
pub fn biv_biv_quadvec_table_public() -> &'static DenseTable<15, 15, 15> {
    biv_biv_quadvec_table()
}
fn biv_biv_quadvec_table() -> &'static DenseTable<15, 15, 15> {
    static TABLE: OnceLock<DenseTable<15, 15, 15>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts =
            products::gen_geometric_product(&BIV_BASIS, &BIV_BASIS, &QUADVEC_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Biv × Quadvec → Pseudoscalar (15×15→1): grade-6 projection.
/// Used to compute ⟨F³⟩₆ = ⟨F · ⟨F²⟩₄⟩₆ — the CP-violating phase.
pub fn biv_quadvec_pseudo_table_public() -> &'static DenseTable<15, 15, 1> {
    biv_quadvec_pseudo_table()
}
fn biv_quadvec_pseudo_table() -> &'static DenseTable<15, 15, 1> {
    static TABLE: OnceLock<DenseTable<15, 15, 1>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let pseudo_basis: [u32; 1] = [0b111111]; // e012345
        let insts =
            products::gen_geometric_product(&BIV_BASIS, &QUADVEC_BASIS, &pseudo_basis, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Biv × Biv → Scalar (15×15→1): scalar product (grade-0 of F²).
pub fn biv_biv_scalar_table_public() -> &'static DenseTable<15, 15, 1> {
    biv_biv_scalar_table()
}
fn biv_biv_scalar_table() -> &'static DenseTable<15, 15, 1> {
    static TABLE: OnceLock<DenseTable<15, 15, 1>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&BIV_BASIS, &BIV_BASIS, &SCALAR_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Rot × Rot → Rot (32×32→32): rotor composition.
pub fn rot_rot_gp_table_public() -> &'static DenseTable<32, 32, 32> {
    rot_rot_gp_table()
}
fn rot_rot_gp_table() -> &'static DenseTable<32, 32, 32> {
    static TABLE: OnceLock<Box<DenseTable<32, 32, 32>>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&EVEN_BASIS, &EVEN_BASIS, &EVEN_BASIS, &METRIC);
        Box::new(DenseTable::from_instructions(&insts))
    })
}

/// Rot × Biv → Full (32×15→64): rotor * bivector (first half of sandwich).
fn rot_biv_gp_table() -> &'static DenseTable<32, 15, 64> {
    static TABLE: OnceLock<Box<DenseTable<32, 15, 64>>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&EVEN_BASIS, &BIV_BASIS, &FULL_BASIS, &METRIC);
        Box::new(DenseTable::from_instructions(&insts))
    })
}

/// Full × Rot → Full (64×32→64): full * rotor (second half of sandwich).
fn full_rot_gp_table() -> &'static DenseTable<64, 32, 64> {
    static TABLE: OnceLock<Box<DenseTable<64, 32, 64>>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&FULL_BASIS, &EVEN_BASIS, &FULL_BASIS, &METRIC);
        Box::new(DenseTable::from_instructions(&insts))
    })
}

// ============================================================================
// SU(3) generator constructors
// ============================================================================

/// Helper: create a bivector with given (blade_bitmask, coefficient) pairs.
fn biv_from_pairs(pairs: &[(u32, f32)]) -> Cl6Biv {
    let mut mv = Cl6Biv::zero();
    for &(blade, coeff) in pairs {
        // Find blade in BIV_BASIS
        for (idx, &b) in BIV_BASIS.iter().enumerate() {
            if b == blade {
                mv[idx] = coeff;
                break;
            }
        }
    }
    mv
}

/// Bitmask for e_i ∧ e_j (0-indexed basis vectors).
#[inline]
const fn eij(i: u32, j: u32) -> u32 {
    (1 << i) | (1 << j)
}

/// T1 = e02 + e13
pub fn t1() -> Cl6Biv {
    biv_from_pairs(&[(eij(0, 2), 1.0), (eij(1, 3), 1.0)])
}

/// T2 = e12 - e03
pub fn t2() -> Cl6Biv {
    biv_from_pairs(&[(eij(1, 2), 1.0), (eij(0, 3), -1.0)])
}

/// T3 = e01 - e23  (Cartan generator of intra-generation SU(2))
pub fn t3() -> Cl6Biv {
    biv_from_pairs(&[(eij(0, 1), 1.0), (eij(2, 3), -1.0)])
}

/// T4 = e04 + e15
pub fn t4() -> Cl6Biv {
    biv_from_pairs(&[(eij(0, 4), 1.0), (eij(1, 5), 1.0)])
}

/// T5 = e14 - e05
pub fn t5() -> Cl6Biv {
    biv_from_pairs(&[(eij(1, 4), 1.0), (eij(0, 5), -1.0)])
}

/// T6 = e24 + e35
pub fn t6() -> Cl6Biv {
    biv_from_pairs(&[(eij(2, 4), 1.0), (eij(3, 5), 1.0)])
}

/// T7 = e34 - e25
pub fn t7() -> Cl6Biv {
    biv_from_pairs(&[(eij(3, 4), 1.0), (eij(2, 5), -1.0)])
}

/// T8 = e01 + e23 - 2*e45  (hypercharge — anomalous norm ||T8||² = 6)
pub fn t8() -> Cl6Biv {
    biv_from_pairs(&[(eij(0, 1), 1.0), (eij(2, 3), 1.0), (eij(4, 5), -2.0)])
}

/// All 8 SU(3) generators.
pub fn generators() -> [Cl6Biv; 8] {
    [t1(), t2(), t3(), t4(), t5(), t6(), t7(), t8()]
}

/// Intra-generation generators (T1, T2, T3) — commute with T8.
pub fn intra_generators() -> [Cl6Biv; 3] {
    [t1(), t2(), t3()]
}

/// Cross-generation generators (T4, T5, T6, T7) — don't commute with T8.
pub fn cross_generators() -> [Cl6Biv; 4] {
    [t4(), t5(), t6(), t7()]
}

// ============================================================================
// Operations
// ============================================================================

/// Geometric product of two bivectors → full multivector.
#[inline]
pub fn geometric_product(a: &Cl6Biv, b: &Cl6Biv) -> Cl6Full {
    Multivector::new(biv_biv_gp_table().execute(&a.data, &b.data))
}

/// Commutator `[A, B]` = AB - BA projected to grade 2 (bivector).
#[inline]
pub fn commutator(a: &Cl6Biv, b: &Cl6Biv) -> Cl6Biv {
    Multivector::new(biv_biv_comm_table().execute(&a.data, &b.data))
}

/// Scalar product of two bivectors: <A * ~B>_0.
/// In Euclidean signature, this equals the sum of component-wise products
/// with metric signs from e_i*e_i = +1.
#[inline]
pub fn scalar_product(a: &Cl6Biv, b: &Cl6Biv) -> f32 {
    biv_biv_scalar_table().execute(&a.data, &b.data)[0]
}

/// Reverse of a bivector: negates all components (grade 2 → sign = -1).
#[inline]
pub fn reverse_biv(b: &Cl6Biv) -> Cl6Biv {
    -*b
}

/// Reverse of a rotor: negate grades 2 and 6, keep grades 0 and 4.
/// reverse_sign(k) = (-1)^(k*(k-1)/2): grade 0→+1, 2→-1, 4→+1, 6→-1.
pub fn reverse_rot(r: &Cl6Rot) -> Cl6Rot {
    let mut out = *r;
    for (idx, &blade) in EVEN_BASIS.iter().enumerate() {
        let g = blade.count_ones() as i32;
        // (-1)^(g*(g-1)/2)
        let rev = if ((g * (g - 1)) / 2) % 2 == 1 {
            -1.0
        } else {
            1.0
        };
        out[idx] *= rev;
    }
    out
}

/// Squared norm of a bivector: scalar_product(b, b).
/// For Euclidean Cl(6,0), this is just the sum of squares of components,
/// but computed via the metric-correct scalar product.
#[inline]
pub fn norm_sq_biv(b: &Cl6Biv) -> f32 {
    // In Cl(6,0) with all-positive metric, biv*~biv scalar part = -sum(c_i^2)
    // because bivector reversal negates, and biv*biv gives negative scalar.
    // We want the *positive* squared norm, so negate the scalar product with reversed.
    // scalar_product(b, b) computes <b*b>_0. For Euclidean bivectors e_ij * e_ij = -1,
    // so <b*b>_0 = -||b||². We want ||b||² = -<b*b>_0.
    -scalar_product(b, b)
}

/// Sandwich product: R * B * ~R (spin a bivector by a rotor).
pub fn spin_rot_biv(r: &Cl6Rot, b: &Cl6Biv) -> Cl6Biv {
    // First half: R * B → Full
    let rb = Cl6Full::new(rot_biv_gp_table().execute(&r.data, &b.data));
    // Reverse of rotor
    let r_rev = reverse_rot(r);
    // Second half: (R*B) * ~R → Full
    let result = Cl6Full::new(full_rot_gp_table().execute(&rb.data, &r_rev.data));
    // Project to grade 2
    grade_project_biv(&result)
}

/// Rotor composition: R1 * R2.
#[inline]
pub fn rot_product(r1: &Cl6Rot, r2: &Cl6Rot) -> Cl6Rot {
    Multivector::new(rot_rot_gp_table().execute(&r1.data, &r2.data))
}

// ============================================================================
// Grade projection
// ============================================================================

/// Extract grade-2 (bivector) part from a full multivector.
pub fn grade_project_biv(full: &Cl6Full) -> Cl6Biv {
    let mut biv = Cl6Biv::zero();
    for (bi, &blade) in BIV_BASIS.iter().enumerate() {
        // Find this blade in FULL_BASIS
        for (fi, &fb) in FULL_BASIS.iter().enumerate() {
            if fb == blade {
                biv[bi] = full[fi];
                break;
            }
        }
    }
    biv
}

/// Extract scalar part from a full multivector.
#[inline]
pub fn grade_project_scalar(full: &Cl6Full) -> f32 {
    // Scalar (blade 0) is always at index 0 in FULL_BASIS
    full[0]
}

/// Extract grade-4 (quadvector) part from a full multivector.
pub fn grade_project_quadvec(full: &Cl6Full) -> Cl6Quad {
    let mut quad = Cl6Quad::zero();
    for (qi, &blade) in QUADVEC_BASIS.iter().enumerate() {
        for (fi, &fb) in FULL_BASIS.iter().enumerate() {
            if fb == blade {
                quad[qi] = full[fi];
                break;
            }
        }
    }
    quad
}

// ============================================================================
// Exponential: bivector → rotor
// ============================================================================

/// Rotor from bivector: R = exp(-B/2).
///
/// Uses truncated Taylor series (sufficient for simulation time-stepping).
/// exp(X) ≈ 1 + X + X²/2! + X³/3! + ... + X^N/N!
/// We compute up to order 12 for good convergence.
pub fn exp_biv(b: &Cl6Biv) -> Cl6Rot {
    // We need to compute exp(-B/2) in the even subalgebra.
    // Strategy: embed bivector into even subalgebra, then Taylor expand.
    let half_b = *b * (-0.5);

    // Embed bivector into rotor space (even subalgebra)
    let x = biv_to_rot(&half_b);

    // Taylor series: result = 1 + x + x²/2! + x³/3! + ...
    let mut result = Cl6Rot::zero();
    result[0] = 1.0; // scalar = 1 (identity rotor)

    let mut term = result; // current term = x^n / n!
    for n in 1..=12u32 {
        term = rot_product(&term, &x);
        term = term * (1.0 / n as f32);
        result = result + term;
    }

    result
}

/// Embed a bivector into the even subalgebra (rotor space).
fn biv_to_rot(b: &Cl6Biv) -> Cl6Rot {
    let mut rot = Cl6Rot::zero();
    for (bi, &blade) in BIV_BASIS.iter().enumerate() {
        for (ri, &rb) in EVEN_BASIS.iter().enumerate() {
            if rb == blade {
                rot[ri] = b[bi];
                break;
            }
        }
    }
    rot
}

// ============================================================================
// Interop
// ============================================================================

/// Convert bivector to `[f32; 15]` array.
#[inline]
pub fn biv_to_array(b: &Cl6Biv) -> [f32; 15] {
    b.data
}

/// Create bivector from `[f32; 15]` array.
#[inline]
pub fn biv_from_array(arr: [f32; 15]) -> Cl6Biv {
    Multivector::new(arr)
}

/// Create bivector from component array reference.
#[inline]
pub fn biv_from_components(coeffs: &[f32; 15]) -> Cl6Biv {
    Multivector::new(*coeffs)
}

// ============================================================================
// Component access
// ============================================================================

/// Get component of a bivector by index (0..14).
#[inline]
pub fn biv_component(b: &Cl6Biv, gen_index: usize) -> f32 {
    b[gen_index]
}

/// Magnetization: M_a = (1/N) * sum_x scalar_product(F(x), T_a) / ||T_a||²
///
/// Measures the mean projection of a field configuration onto a generator.
pub fn magnetization(fields: &[Cl6Biv], gen: &Cl6Biv) -> f32 {
    if fields.is_empty() {
        return 0.0;
    }
    let norm_sq = norm_sq_biv(gen);
    if norm_sq == 0.0 {
        return 0.0;
    }
    let sum: f32 = fields.iter().map(|f| scalar_product(f, gen)).sum();
    // scalar_product gives <f*gen>_0 which for Euclidean bivectors is -dot(f,gen).
    // We want the "inner product" sense, so negate to get the positive projection.
    -sum / (fields.len() as f32 * norm_sq)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: check that a bivector is approximately zero
    fn is_zero_biv(b: &Cl6Biv, tol: f32) -> bool {
        b.data.iter().all(|&c| c.abs() < tol)
    }

    #[test]
    fn test_basis_counts() {
        // Verify basis array sizes
        assert_eq!(FULL_BASIS.len(), 64);
        assert_eq!(BIV_BASIS.len(), 15);
        assert_eq!(EVEN_BASIS.len(), 32);
        assert_eq!(QUADVEC_BASIS.len(), 15);

        // Verify all BIV_BASIS elements are grade 2
        for &b in &BIV_BASIS {
            assert_eq!(
                b.count_ones(),
                2,
                "Blade {} has grade {}",
                b,
                b.count_ones()
            );
        }

        // Verify EVEN_BASIS elements are even grade
        for &b in &EVEN_BASIS {
            let g = b.count_ones();
            assert!(g % 2 == 0, "Blade {} has odd grade {}", b, g);
        }
    }

    #[test]
    fn test_t8_norm_anomaly() {
        // ||T8||² = 6 while ||T1||² = ... = ||T7||² = 2
        let gens = generators();
        for i in 0..7 {
            let nsq = norm_sq_biv(&gens[i]);
            assert!(
                (nsq - 2.0).abs() < 1e-5,
                "||T{}||² = {} (expected 2.0)",
                i + 1,
                nsq
            );
        }
        let nsq_t8 = norm_sq_biv(&gens[7]);
        assert!(
            (nsq_t8 - 6.0).abs() < 1e-5,
            "||T8||² = {} (expected 6.0)",
            nsq_t8
        );
    }

    #[test]
    fn test_t8_commutes_with_intra() {
        let t8_gen = t8();
        let intra = intra_generators();
        for (i, ti) in intra.iter().enumerate() {
            let comm = commutator(&t8_gen, ti);
            assert!(
                is_zero_biv(&comm, 1e-5),
                "[T8, T{}] should be zero, got {:?}",
                i + 1,
                comm.data
            );
        }
    }

    #[test]
    fn test_t8_does_not_commute_with_cross() {
        let t8_gen = t8();
        let cross = cross_generators();
        for (i, ti) in cross.iter().enumerate() {
            let comm = commutator(&t8_gen, ti);
            let nsq = norm_sq_biv(&comm);
            assert!(
                nsq > 1e-3,
                "[T8, T{}] should be non-zero, got norm² = {}",
                i + 4,
                nsq
            );
        }
    }

    #[test]
    fn test_intra_commutators() {
        // T1, T2, T3 should form an SU(2) subalgebra.
        // [T1, T2] should be proportional to T3.
        let t1v = t1();
        let t2v = t2();
        let t3v = t3();

        let c12 = commutator(&t1v, &t2v);
        // Check that c12 is proportional to T3
        // T3 = e01 - e23. Find which components T3 has.
        let t3_nsq = norm_sq_biv(&t3v);
        let proj = scalar_product(&c12, &t3v);
        // The residual after subtracting the T3 projection should be ~zero
        let coeff = -proj / t3_nsq; // negate because scalar_product gives -dot
        let residual = c12 - t3v * coeff;
        assert!(
            norm_sq_biv(&residual) < 1e-4,
            "[T1,T2] not proportional to T3. Residual norm² = {}",
            norm_sq_biv(&residual)
        );
        assert!(
            coeff.abs() > 0.1,
            "[T1,T2] proportionality coefficient too small: {}",
            coeff
        );
    }

    #[test]
    fn test_commutator_antisymmetric() {
        let a = t1();
        let b = t4();
        let ab = commutator(&a, &b);
        let ba = commutator(&b, &a);
        for i in 0..15 {
            assert!(
                (ab[i] + ba[i]).abs() < 1e-5,
                "[A,B] + [B,A] not zero at component {}: {} + {} = {}",
                i,
                ab[i],
                ba[i],
                ab[i] + ba[i]
            );
        }
    }

    #[test]
    fn test_commutator_self_zero() {
        // [A, A] = 0 for any A
        let a = biv_from_array([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ]);
        let comm = commutator(&a, &a);
        assert!(is_zero_biv(&comm, 1e-4), "[A,A] not zero: {:?}", comm.data);
    }

    #[test]
    fn test_grade_decomposition() {
        // F*∇F (product of two bivectors) decomposes into grade 0 + 2 + 4
        let f = t1();
        let g = t4();
        let prod = geometric_product(&f, &g);

        // Extract each grade
        let s = grade_project_scalar(&prod);
        let biv = grade_project_biv(&prod);
        let quad = grade_project_quadvec(&prod);

        // Reconstruct and check: only grades 0, 2, 4 should be nonzero
        // (bivector * bivector can only produce even grades: 0, 2, 4)
        // Verify odd-grade parts are zero
        for (fi, &blade) in FULL_BASIS.iter().enumerate() {
            let g = blade.count_ones();
            if g % 2 == 1 {
                assert!(
                    prod[fi].abs() < 1e-6,
                    "Odd-grade component at blade {} (grade {}) = {}",
                    blade,
                    g,
                    prod[fi]
                );
            }
        }

        // Verify we got something in at least one of the even grades
        let has_content = s.abs() > 1e-6
            || biv.data.iter().any(|&c| c.abs() > 1e-6)
            || quad.data.iter().any(|&c| c.abs() > 1e-6);
        assert!(has_content, "Product has no content in grades 0, 2, or 4");
    }

    #[test]
    fn test_rotor_from_bivector() {
        // exp(B) should produce a proper rotor: R * ~R ≈ 1
        let b = t1() * 0.3; // small bivector for convergence
        let r = exp_biv(&b);
        let r_rev = reverse_rot(&r);
        let rr = rot_product(&r, &r_rev);

        // Scalar part should be ≈ 1, everything else ≈ 0
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
    fn test_sandwich_preserves_norm() {
        // ||R*B*~R|| = ||B||
        let b = t1() * 0.3;
        let r = exp_biv(&b);

        let target = t4();
        let original_nsq = norm_sq_biv(&target);
        let rotated = spin_rot_biv(&r, &target);
        let rotated_nsq = norm_sq_biv(&rotated);

        assert!(
            (rotated_nsq - original_nsq).abs() < 1e-3,
            "Sandwich changed norm²: {} → {}",
            original_nsq,
            rotated_nsq
        );
    }

    #[test]
    fn test_generators_orthogonal() {
        // scalar_product(Ti, Tj) = 0 for i ≠ j (in the metric sense)
        let gens = generators();
        for i in 0..8 {
            for j in (i + 1)..8 {
                let sp = scalar_product(&gens[i], &gens[j]);
                assert!(
                    sp.abs() < 1e-5,
                    "scalar_product(T{}, T{}) = {} (expected 0)",
                    i + 1,
                    j + 1,
                    sp
                );
            }
        }
    }

    #[test]
    fn test_magnetization() {
        // A field equal to T1 should have magnetization 1.0 w.r.t. T1
        let t1_gen = t1();
        let fields = vec![t1_gen];
        let m = magnetization(&fields, &t1_gen);
        assert!(
            (m - 1.0).abs() < 1e-5,
            "Magnetization of T1 w.r.t. T1 = {} (expected 1.0)",
            m
        );

        // Orthogonal field should have magnetization 0
        let t4_gen = t4();
        let m_orth = magnetization(&fields, &t4_gen);
        assert!(
            m_orth.abs() < 1e-5,
            "Magnetization of T1 w.r.t. T4 = {} (expected 0.0)",
            m_orth
        );

        // Average of two fields
        let fields2 = vec![t1_gen, t1_gen * 3.0];
        let m2 = magnetization(&fields2, &t1_gen);
        assert!(
            (m2 - 2.0).abs() < 1e-5,
            "Magnetization of [T1, 3*T1] w.r.t. T1 = {} (expected 2.0)",
            m2
        );
    }
}
