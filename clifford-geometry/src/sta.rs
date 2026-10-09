//! Spacetime Algebra (STA) — Cl(1,3).
//!
//! Basis: γ₀ (timelike, +1), γ₁, γ₂, γ₃ (spacelike, -1 each).
//! This matches the convention used in HarmonicRust's bivector field module.
//!
//! Metric: `[+1, -1, -1, -1]`
//!
//! Types:
//! - StaVec: 4 components `[γ₀, γ₁, γ₂, γ₃]`
//! - StaBiv: 6 components `[γ₀₁, γ₀₂, γ₀₃, γ₂₃, γ₁₃, γ₁₂]`
//!                        = `[E₁,  E₂,  E₃,  B₁,  B₂,  B₃]`
//! - StaRot: 8 components `[s, bivector(6), pseudoscalar]`
//! - StaFull: 16 components (full multivector)
//!
//! The bivector decomposes into electric (γ₀ₖ) and magnetic (γⱼₖ) parts,
//! exactly matching HarmonicRust's `StaBivector { components: [E₁,E₂,E₃,B₁,B₂,B₃] }`.

use crate::mvec::Multivector;
use crate::products::{self, DenseTable};
use std::sync::OnceLock;

// ============================================================================
// Metric: Cl(1,3) — γ₀²=+1, γ₁²=γ₂²=γ₃²=-1
// ============================================================================

/// Spacetime Cl(1,3) metric coefficients.
pub const METRIC: [i32; 4] = [1, -1, -1, -1];

// ============================================================================
// Blade bitmasks
// ============================================================================

// Basis vectors: γ₀=bit0, γ₁=bit1, γ₂=bit2, γ₃=bit3
const G0: u32 = 0b0001;
const G1: u32 = 0b0010;
const G2: u32 = 0b0100;
const G3: u32 = 0b1000;

/// Bivector basis: `[γ₀₁, γ₀₂, γ₀₃, γ₂₃, γ₁₃, γ₁₂]`
/// This order maps to `[E₁, E₂, E₃, B₁, B₂, B₃]` matching HarmonicRust.
pub const BIV_BASIS: [u32; 6] = [
    G0 | G1, // γ₀₁ = E₁ (electric x)
    G0 | G2, // γ₀₂ = E₂ (electric y)
    G0 | G3, // γ₀₃ = E₃ (electric z)
    G2 | G3, // γ₂₃ = B₁ (magnetic x)
    G1 | G3, // γ₁₃ = B₂ (magnetic y)
    G1 | G2, // γ₁₂ = B₃ (magnetic z)
];

/// Scalar basis
pub const SCALAR_BASIS: [u32; 1] = [0];

/// Vector basis: `[γ₀, γ₁, γ₂, γ₃]`
pub const VEC_BASIS: [u32; 4] = [G0, G1, G2, G3];

/// Pseudoscalar basis: `[γ₀₁₂₃]`
pub const PSS_BASIS: [u32; 1] = [G0 | G1 | G2 | G3];

// ============================================================================
// Types
// ============================================================================

/// STA bivector: 6 components `[E₁, E₂, E₃, B₁, B₂, B₃]`.
/// Directly compatible with HarmonicRust's `StaBivector`.
pub type StaBiv = Multivector<6>;

/// STA vector: 4 components `[γ₀, γ₁, γ₂, γ₃]`.
pub type StaVec = Multivector<4>;

/// STA scalar
pub type StaSca = Multivector<1>;

// ============================================================================
// Constructors
// ============================================================================

/// Create an electromagnetic bivector from E and B fields.
#[inline]
pub fn em_bivector(ex: f32, ey: f32, ez: f32, bx: f32, by: f32, bz: f32) -> StaBiv {
    Multivector::new([ex, ey, ez, bx, by, bz])
}

/// Create from separate electric and magnetic field arrays.
/// Matches HarmonicRust's `StaBivector::new(e1, e2, e3, b1, b2, b3)`.
#[inline]
pub fn from_em(electric: [f32; 3], magnetic: [f32; 3]) -> StaBiv {
    Multivector::new([
        electric[0],
        electric[1],
        electric[2],
        magnetic[0],
        magnetic[1],
        magnetic[2],
    ])
}

/// Extract electric field `[E₁, E₂, E₃]`.
#[inline]
pub fn electric(f: &StaBiv) -> [f32; 3] {
    [f[0], f[1], f[2]]
}

/// Extract magnetic field `[B₁, B₂, B₃]`.
#[inline]
pub fn magnetic(f: &StaBiv) -> [f32; 3] {
    [f[3], f[4], f[5]]
}

// ============================================================================
// Product tables
// ============================================================================

/// Bivector commutator table: `[A, B]` = AB - BA for STA bivectors.
/// This is a dense 6×6→6 table (same structure as HarmonicRust's comm_table).
fn biv_comm_table() -> &'static DenseTable<6, 6, 6> {
    static TABLE: OnceLock<DenseTable<6, 6, 6>> = OnceLock::new();
    TABLE.get_or_init(|| {
        // Generate AB and BA tables, then subtract
        let ab_insts = products::gen_geometric_product(&BIV_BASIS, &BIV_BASIS, &BIV_BASIS, &METRIC);

        // Build AB table; BA is the same table with swapped indices
        let ab = DenseTable::<6, 6, 6>::from_instructions(&ab_insts);

        // Build commutator table: [A,B]_ijk = AB_ijk - AB_jik (swap inputs for BA)
        let mut table = [[[0.0f32; 6]; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                for k in 0..6 {
                    table[i][j][k] = ab.table[i][j][k] - ab.table[j][i][k];
                }
            }
        }
        DenseTable { table }
    })
}

/// Bivector geometric product table: AB for STA bivectors.
/// Result includes scalar and pseudoscalar parts too, but we project to bivector.
fn biv_gp_table() -> &'static DenseTable<6, 6, 6> {
    static TABLE: OnceLock<DenseTable<6, 6, 6>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&BIV_BASIS, &BIV_BASIS, &BIV_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Bivector scalar product table: (A * B)_scalar.
fn biv_scalar_product_table() -> &'static [f32; 36] {
    static TABLE: OnceLock<[f32; 36]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_scalar_product(&BIV_BASIS, &BIV_BASIS, &METRIC);
        let mut table = [0.0f32; 36];
        for inst in &insts {
            table[inst.src_a * 6 + inst.src_b] = inst.sign;
        }
        table
    })
}

// ============================================================================
// Operations
// ============================================================================

/// Bivector commutator `[A, B]` = AB - BA.
/// This is the key operation for the universal synchronization equation.
///
/// Uses the same dense table approach as HarmonicRust but generated
/// automatically from the algebra instead of hand-coded.
#[inline]
pub fn commutator(a: &StaBiv, b: &StaBiv) -> StaBiv {
    Multivector::new(biv_comm_table().execute(&a.data, &b.data))
}

/// Geometric product of two bivectors (projected to bivector part).
#[inline]
pub fn gp_biv_biv(a: &StaBiv, b: &StaBiv) -> StaBiv {
    Multivector::new(biv_gp_table().execute(&a.data, &b.data))
}

/// Scalar product of two bivectors.
#[inline]
pub fn scalar_product(a: &StaBiv, b: &StaBiv) -> f32 {
    let table = biv_scalar_product_table();
    let mut result = 0.0f32;
    for i in 0..6 {
        for j in 0..6 {
            result += a[i] * b[j] * table[i * 6 + j];
        }
    }
    result
}

/// Reverse of a bivector: negated (grade 2 → sign = -1).
#[inline]
pub fn reverse_biv(b: &StaBiv) -> StaBiv {
    -*b
}

/// Hodge dual of a bivector in STA: *F.
/// Maps electric ↔ magnetic via the pseudoscalar.
/// *E = B, *B = -E in Cl(1,3).
#[inline]
pub fn hodge_dual(f: &StaBiv) -> StaBiv {
    // In Cl(1,3) with metric [+,-,-,-]:
    // *γ₀₁ = γ₂₃, *γ₀₂ = -γ₁₃, *γ₀₃ = γ₁₂
    // *γ₂₃ = -γ₀₁, *γ₁₃ = γ₀₂, *γ₁₂ = -γ₀₃
    // This swaps E ↔ B with appropriate signs
    Multivector::new([
        -f[3], // E₁ ← -B₁
        f[4],  // E₂ ← B₂ (sign from metric)
        -f[5], // E₃ ← -B₃
        -f[0], // B₁ ← -E₁
        f[1],  // B₂ ← E₂
        -f[2], // B₃ ← -E₃
    ])
}

/// Energy density of an electromagnetic bivector: ½(E² - B²) in Cl(1,3).
#[inline]
pub fn energy_density(f: &StaBiv) -> f32 {
    let e = electric(f);
    let b = magnetic(f);
    let e_sq = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    let b_sq = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
    0.5 * (e_sq - b_sq)
}

/// Convert to/from HarmonicRust-compatible `[f32; 6]` array.
#[inline]
pub fn to_array(f: &StaBiv) -> [f32; 6] {
    f.data
}

#[inline]
/// Construct an STA bivector from its six stored coefficients.
pub fn from_array(arr: [f32; 6]) -> StaBiv {
    Multivector::new(arr)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_em_bivector_creation() {
        let f = em_bivector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        assert_eq!(electric(&f), [1.0, 2.0, 3.0]);
        assert_eq!(magnetic(&f), [4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_commutator_antisymmetric() {
        let a = em_bivector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let b = em_bivector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let ab = commutator(&a, &b);
        let ba = commutator(&b, &a);
        for i in 0..6 {
            assert!(
                (ab[i] + ba[i]).abs() < 1e-6,
                "Commutator not antisymmetric at {}: {} vs {}",
                i,
                ab[i],
                ba[i]
            );
        }
    }

    #[test]
    fn test_commutator_self_zero() {
        let a = em_bivector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let aa = commutator(&a, &a);
        for i in 0..6 {
            assert!((aa[i]).abs() < 1e-6, "[A,A] not zero at {}: {}", i, aa[i]);
        }
    }

    #[test]
    fn test_commutator_electric_electric() {
        // [E₁, E₂] should give a magnetic component
        // In Cl(1,3): γ₀₁ * γ₀₂ = γ₁₂ (with sign from metric)
        // γ₀₂ * γ₀₁ = -γ₁₂ (opposite sign)
        // So [γ₀₁, γ₀₂] = 2*γ₁₂ = 2*B₃
        let e1 = em_bivector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let e2 = em_bivector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let comm = commutator(&e1, &e2);
        // The result should be in the B₃ (γ₁₂) component
        let b3_idx = 5; // B₃ = γ₁₂ is at index 5
        assert!(
            comm[b3_idx].abs() > 0.1,
            "Expected non-zero B₃ component, got {:?}",
            comm.data
        );
    }

    #[test]
    fn test_reverse_bivector() {
        let f = em_bivector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let rev = reverse_biv(&f);
        for i in 0..6 {
            assert_eq!(rev[i], -f[i]);
        }
    }

    #[test]
    fn test_energy_density() {
        // Pure electric field: E = (1,0,0), B = 0
        let f = em_bivector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(energy_density(&f), 0.5);

        // Pure magnetic field: E = 0, B = (1,0,0)
        let g = em_bivector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        assert_eq!(energy_density(&g), -0.5);
    }

    #[test]
    fn test_array_roundtrip() {
        let arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let f = from_array(arr);
        assert_eq!(to_array(&f), arr);
    }

    #[test]
    fn test_bivector_add() {
        let a = em_bivector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let b = em_bivector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let c = a + b;
        assert_eq!(c[0], 1.0);
        assert_eq!(c[1], 1.0);
    }

    #[test]
    fn test_bivector_scale() {
        let a = em_bivector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let b = a * 2.0;
        assert_eq!(b[0], 2.0);
        assert_eq!(b[5], 12.0);
    }
}
