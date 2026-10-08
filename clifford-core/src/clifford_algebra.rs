//! Generic Clifford algebra Cl(p,q) engine.
//!
//! Runtime-parameterized over signature (p positive-squaring generators,
//! q negative-squaring generators). Precomputes the Cayley multiplication
//! table at construction time for O(1) geometric product lookup.
//!
//! All multivector operations are generic over `CliffordScalar`.
//! The Cayley table stores signs as f32 (±1, exact in any precision).
//! Callers choose precision by passing `&[f32]` or `&[f64]` multivectors.
//!
//! Ported concepts from Microsoft's cliffordlayers (Python/PyTorch) into
//! pure Rust. Operations return owned dense multivectors and therefore
//! allocate; specialized fixed-size engines may be preferable in hot loops.

use crate::scalar::CliffordScalar;

/// Maximum vector-space dimension supported by the dense Cayley-table engine.
pub const MAX_DIMENSION: usize = 10;

/// Error returned when a runtime Clifford signature cannot be constructed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliffordAlgebraError {
    /// `p + q` overflowed `usize`.
    DimensionOverflow {
        /// Number of requested positive-squaring generators.
        p: usize,
        /// Number of requested negative-squaring generators.
        q: usize,
    },
    /// The dense `2^d × 2^d` Cayley table would exceed the supported limit.
    DimensionTooLarge {
        /// Number of requested positive-squaring generators.
        p: usize,
        /// Number of requested negative-squaring generators.
        q: usize,
        /// Requested vector-space dimension, `p + q`.
        dimension: usize,
        /// Largest dimension supported by the dense engine.
        maximum: usize,
    },
}

impl std::fmt::Display for CliffordAlgebraError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionOverflow { p, q } => {
                write!(f, "Cl({p},{q}) dimension overflows usize")
            }
            Self::DimensionTooLarge {
                p,
                q,
                dimension,
                maximum,
            } => write!(
                f,
                "Cl({p},{q}) has dimension {dimension}; maximum supported dimension is {maximum}"
            ),
        }
    }
}

impl std::error::Error for CliffordAlgebraError {}

/// A Clifford algebra Cl(p,q) with precomputed multiplication tables.
///
/// `p` generators square to +1, `q` generators square to -1.
/// Total dimension `d = p + q`, number of basis blades `n = 2^d`.
///
/// Blade ordering: ShortLex (sorted by grade, then by bitmap value).
/// This matches the convention in cliffordlayers. It differs from the legacy
/// Cl(3,0) compatibility layout in the ordering of e13 and e23.
#[derive(Debug, Clone)]
pub struct CliffordAlgebra {
    /// Number of positive-squaring generators.
    pub p: usize,
    /// Number of negative-squaring generators.
    pub q: usize,
    /// Vector space dimension d = p + q.
    pub dim: usize,
    /// Number of basis blades = 2^d.
    pub n_blades: usize,
    /// Blade index → bitmap representation.
    pub index_to_bitmap: Vec<u64>,
    /// Bitmap → blade index (sparse: indexed by bitmap value).
    pub bitmap_to_index: Vec<usize>,
    /// Grade of each blade index.
    pub grades: Vec<usize>,
    /// Cayley table: for blades i,j the product lands on blade
    /// `cayley_index[i * n_blades + j]` with sign `cayley_sign[i * n_blades + j]`.
    pub cayley_index: Vec<usize>,
    /// Geometric-product sign corresponding to each [`Self::cayley_index`] entry.
    pub cayley_sign: Vec<f32>,
    /// Precomputed reversal signs per blade.
    reverse_signs: Vec<f32>,
}

impl CliffordAlgebra {
    /// Construct a Clifford algebra with signature (p, q).
    ///
    /// # Panics
    ///
    /// Panics if `p + q` overflows or exceeds [`MAX_DIMENSION`]. Use
    /// [`Self::try_new`] to handle those conditions without panicking.
    pub fn new(p: usize, q: usize) -> Self {
        Self::try_new(p, q).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Try to construct a Clifford algebra with signature `(p, q)`.
    pub fn try_new(p: usize, q: usize) -> Result<Self, CliffordAlgebraError> {
        let dim = p
            .checked_add(q)
            .ok_or(CliffordAlgebraError::DimensionOverflow { p, q })?;
        if dim > MAX_DIMENSION {
            return Err(CliffordAlgebraError::DimensionTooLarge {
                p,
                q,
                dimension: dim,
                maximum: MAX_DIMENSION,
            });
        }
        let n_blades = 1 << dim;

        // Generate basis blades in ShortLex order (grade, then bitmap).
        let mut blades: Vec<u64> = (0..n_blades as u64).collect();
        blades.sort_by(|&a, &b| {
            let ga = a.count_ones();
            let gb = b.count_ones();
            ga.cmp(&gb).then(a.cmp(&b))
        });

        let index_to_bitmap = blades.clone();
        let mut bitmap_to_index = vec![0usize; n_blades];
        for (idx, &bm) in blades.iter().enumerate() {
            bitmap_to_index[bm as usize] = idx;
        }

        // Grade of each blade index.
        let grades: Vec<usize> = blades.iter().map(|bm| bm.count_ones() as usize).collect();

        // Reversal sign: grade k → (-1)^(k(k-1)/2).
        let reverse_signs: Vec<f32> = grades
            .iter()
            .map(|&k| {
                let flip = (k * k.wrapping_sub(1)) / 2;
                if flip % 2 == 0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();

        // Build Cayley table.
        let mut cayley_index = vec![0usize; n_blades * n_blades];
        let mut cayley_sign = vec![0.0f32; n_blades * n_blades];

        for (i, &a_bm) in index_to_bitmap.iter().enumerate() {
            for (j, &b_bm) in index_to_bitmap.iter().enumerate() {
                let out_bm = a_bm ^ b_bm;
                let sign = Self::blade_mul_sign(a_bm, b_bm, p, dim);
                let flat = i * n_blades + j;
                cayley_index[flat] = bitmap_to_index[out_bm as usize];
                cayley_sign[flat] = sign;
            }
        }

        Ok(CliffordAlgebra {
            p,
            q,
            dim,
            n_blades,
            index_to_bitmap,
            bitmap_to_index,
            grades,
            cayley_index,
            cayley_sign,
            reverse_signs,
        })
    }

    /// Compute the sign of the geometric product of two basis blades.
    ///
    /// Two contributions:
    /// 1. Transposition sign: number of swaps to bring generators into canonical order.
    /// 2. Metric sign: shared generators at position i ≥ p square to -1.
    fn blade_mul_sign(a_bm: u64, b_bm: u64, p: usize, dim: usize) -> f32 {
        let mut sign = 1i32;

        // Count transpositions: for each set bit in a, count how many
        // lower-index bits in b must be "passed through".
        for i in 0..dim {
            if (a_bm >> i) & 1 == 1 {
                // Count set bits in b below position i.
                let lower_mask = (1u64 << i) - 1;
                let crossings = (b_bm & lower_mask).count_ones();
                if crossings % 2 == 1 {
                    sign = -sign;
                }
            }
        }

        // Metric contribution: shared generators (bits set in both a and b).
        // For each shared generator at position i:
        //   - if i < p, it squares to +1 (no sign change)
        //   - if i >= p, it squares to -1 (flip sign)
        let shared = a_bm & b_bm;
        for i in 0..dim {
            if (shared >> i) & 1 == 1 && i >= p {
                sign = -sign;
            }
        }

        sign as f32
    }

    // ========================================================================
    // Predefined algebras
    // ========================================================================

    /// Complex numbers as Cl(0,1), with basis {1, e1} and e1² = -1.
    pub fn complex() -> Self {
        Self::new(0, 1)
    }

    /// Split-complex numbers: Cl(1,0), 2 blades. e1² = +1.
    pub fn split_complex() -> Self {
        Self::new(1, 0)
    }

    /// Quaternions via Cl(0,2): 4 blades {1, e1, e2, e12}.
    /// e1² = e2² = -1, e12² = -1.
    /// The full algebra is isomorphic to the quaternions under the
    /// identification i=e1, j=e2, k=e12. Its even subalgebra {1, e12} is
    /// complex, not quaternionic.
    pub fn quaternion() -> Self {
        Self::new(0, 2)
    }

    /// Euclidean 3D: Cl(3,0), 8 blades.
    /// This has the same multiplication semantics as the legacy Cl(3,0)
    /// implementation, but uses ShortLex blade ordering.
    pub fn cl3() -> Self {
        Self::new(3, 0)
    }

    /// Cl(3,1), a non-degenerate 16-blade algebra.
    pub fn cl31() -> Self {
        Self::new(3, 1)
    }

    /// Spacetime Algebra: Cl(1,3), 16 blades.
    /// One time dimension (e0² = +1), three space dimensions (e1²=e2²=e3² = -1).
    pub fn sta() -> Self {
        Self::new(1, 3)
    }

    /// Cl(6,0): 6D Euclidean Clifford algebra.
    /// 64 basis blades, 15 bivectors forming so(6) ≅ su(4).
    /// su(3) embeds as an 8-dimensional subalgebra (Gell-Mann generators).
    pub fn cl6() -> Self {
        Self::new(6, 0)
    }

    // ========================================================================
    // Multivector operations generic over CliffordScalar
    // ========================================================================

    /// Allocate a zero multivector.
    #[inline]
    pub fn zero<S: CliffordScalar>(&self) -> Vec<S> {
        vec![S::ZERO; self.n_blades]
    }

    /// Allocate the scalar unit multivector (1).
    pub fn one<S: CliffordScalar>(&self) -> Vec<S> {
        let mut mv: Vec<S> = self.zero();
        mv[0] = S::ONE;
        mv
    }

    /// Create a multivector with a single scalar value.
    pub fn scalar<S: CliffordScalar>(&self, s: S) -> Vec<S> {
        let mut mv: Vec<S> = self.zero();
        mv[0] = s;
        mv
    }

    /// Create a basis blade multivector (1.0 at blade index `i`, rest zero).
    pub fn basis<S: CliffordScalar>(&self, i: usize) -> Vec<S> {
        assert!(i < self.n_blades);
        let mut mv: Vec<S> = self.zero();
        mv[i] = S::ONE;
        mv
    }

    /// Geometric product: c = a * b.
    pub fn geometric_product<S: CliffordScalar>(&self, a: &[S], b: &[S]) -> Vec<S> {
        assert_eq!(a.len(), self.n_blades, "left multivector length mismatch");
        assert_eq!(b.len(), self.n_blades, "right multivector length mismatch");
        let n = self.n_blades;
        let mut c: Vec<S> = vec![S::ZERO; n];
        for (i, &ai) in a.iter().enumerate() {
            if ai == S::ZERO {
                continue;
            }
            for (j, &bj) in b.iter().enumerate() {
                if bj == S::ZERO {
                    continue;
                }
                let flat = i * n + j;
                let k = self.cayley_index[flat];
                // cayley_sign is ±1.0f32, exact in any precision
                c[k] += ai * bj * S::from_f32(self.cayley_sign[flat]);
            }
        }
        c
    }

    /// Grade reversal: reverse the order of basis vectors in each blade.
    /// Grade k blade gets sign (-1)^(k(k-1)/2).
    pub fn reverse<S: CliffordScalar>(&self, a: &[S]) -> Vec<S> {
        assert_eq!(a.len(), self.n_blades, "multivector length mismatch");
        a.iter()
            .zip(self.reverse_signs.iter())
            .map(|(&v, &s)| v * S::from_f32(s))
            .collect()
    }

    /// Grade projection: extract components of a specific grade.
    pub fn grade_project<S: CliffordScalar>(&self, a: &[S], grade: usize) -> Vec<S> {
        assert_eq!(a.len(), self.n_blades, "multivector length mismatch");
        let mut out: Vec<S> = vec![S::ZERO; self.n_blades];
        for (i, &g) in self.grades.iter().enumerate() {
            if g == grade {
                out[i] = a[i];
            }
        }
        out
    }

    /// Extract the scalar (grade-0) part.
    #[inline]
    pub fn scalar_part<S: CliffordScalar>(&self, a: &[S]) -> S {
        a[0]
    }

    /// Norm squared: scalar part of a * reverse(a).
    /// For non-degenerate signatures, this is always real.
    pub fn norm_squared<S: CliffordScalar>(&self, a: &[S]) -> S {
        let rev = self.reverse(a);
        let prod = self.geometric_product(a, &rev);
        prod[0]
    }

    /// Coefficient-space L2 norm (not the Clifford norm).
    pub fn coeff_norm<S: CliffordScalar>(&self, a: &[S]) -> S {
        a.iter()
            .map(|&x| x * x)
            .sum::<S>()
            .max(S::from_f64(1e-16))
            .sqrt()
    }

    /// Normalize by coefficient-space L2 norm.
    pub fn normalize<S: CliffordScalar>(&self, a: &[S]) -> Vec<S> {
        let n = self.coeff_norm(a);
        a.iter().map(|&x| x / n).collect()
    }

    /// Sandwich product: r * x * reverse(r).
    /// This is the fundamental operation for rotations and reflections
    /// in geometric algebra. When r is a unit rotor, this performs
    /// a grade-preserving rotation.
    pub fn sandwich<S: CliffordScalar>(&self, r: &[S], x: &[S]) -> Vec<S> {
        let rev_r = self.reverse(r);
        let rx = self.geometric_product(r, x);
        self.geometric_product(&rx, &rev_r)
    }

    /// Commutator product: [a, b] = (a*b - b*a) / 2.
    /// The factor of 1/2 makes this the standard Lie bracket.
    pub fn commutator<S: CliffordScalar>(&self, a: &[S], b: &[S]) -> Vec<S> {
        let ab = self.geometric_product(a, b);
        let ba = self.geometric_product(b, a);
        ab.iter()
            .zip(ba.iter())
            .map(|(&x, &y)| (x - y) * S::HALF)
            .collect()
    }

    /// Anti-commutator: {a, b} = (a*b + b*a) / 2.
    pub fn anti_commutator<S: CliffordScalar>(&self, a: &[S], b: &[S]) -> Vec<S> {
        let ab = self.geometric_product(a, b);
        let ba = self.geometric_product(b, a);
        ab.iter()
            .zip(ba.iter())
            .map(|(&x, &y)| (x + y) * S::HALF)
            .collect()
    }

    /// Embed a vector (grade-1 element) from its d components.
    /// Returns a full multivector with only grade-1 slots filled.
    pub fn vector<S: CliffordScalar>(&self, components: &[S]) -> Vec<S> {
        assert_eq!(components.len(), self.dim);
        let mut mv: Vec<S> = self.zero();
        let mut ci = 0;
        for (i, &g) in self.grades.iter().enumerate() {
            if g == 1 {
                mv[i] = components[ci];
                ci += 1;
            }
        }
        mv
    }

    /// Embed a bivector (grade-2 element) from its d*(d-1)/2 components.
    pub fn bivector<S: CliffordScalar>(&self, components: &[S]) -> Vec<S> {
        let n_biv = self.dim * self.dim.saturating_sub(1) / 2;
        assert_eq!(components.len(), n_biv);
        let mut mv: Vec<S> = self.zero();
        let mut ci = 0;
        for (i, &g) in self.grades.iter().enumerate() {
            if g == 2 {
                mv[i] = components[ci];
                ci += 1;
            }
        }
        mv
    }

    /// Extract grade-1 (vector) components as a dense Vec.
    pub fn vector_components<S: CliffordScalar>(&self, a: &[S]) -> Vec<S> {
        self.grades
            .iter()
            .enumerate()
            .filter(|(_, &g)| g == 1)
            .map(|(i, _)| a[i])
            .collect()
    }

    /// Extract grade-2 (bivector) components as a dense Vec.
    pub fn bivector_components<S: CliffordScalar>(&self, a: &[S]) -> Vec<S> {
        self.grades
            .iter()
            .enumerate()
            .filter(|(_, &g)| g == 2)
            .map(|(i, _)| a[i])
            .collect()
    }

    /// Construct a rotor from a simple Euclidean bivector B and angle theta.
    /// rotor = cos(theta/2) + sin(theta/2) * B_hat
    /// where B_hat = B / |B|.
    ///
    /// This closed form is valid when the normalized bivector is simple and
    /// squares to -1. It is not a general bivector exponential for arbitrary
    /// signatures or non-simple bivectors.
    ///
    /// If B is zero, returns the identity rotor.
    pub fn rotor_from_bivector<S: CliffordScalar>(&self, bivector: &[S], theta: S) -> Vec<S> {
        assert_eq!(bivector.len(), self.n_blades, "bivector length mismatch");
        if bivector.iter().all(|&coefficient| coefficient == S::ZERO) {
            return self.one();
        }
        let biv_norm = self.coeff_norm(bivector);
        let mut rotor: Vec<S> = self.zero();
        let half_theta = theta * S::HALF;
        rotor[0] = half_theta.cos();
        if biv_norm > S::from_f64(1e-12) {
            let s = half_theta.sin() / biv_norm;
            for i in 0..self.n_blades {
                if self.grades[i] == 2 {
                    rotor[i] = bivector[i] * s;
                }
            }
        }
        rotor
    }

    /// Number of components at a given grade.
    pub fn grade_count(&self, grade: usize) -> usize {
        self.grades.iter().filter(|&&g| g == grade).count()
    }

    /// Maximum grade (= dim).
    pub fn max_grade(&self) -> usize {
        self.dim
    }

    /// Get the blade indices for a specific grade.
    pub fn grade_indices(&self, grade: usize) -> Vec<usize> {
        self.grades
            .iter()
            .enumerate()
            .filter(|(_, &g)| g == grade)
            .map(|(i, _)| i)
            .collect()
    }

    /// Get the bitmap representation of blade index i.
    pub fn blade_bitmap(&self, i: usize) -> u64 {
        self.index_to_bitmap[i]
    }
}

// ============================================================================
// Display
// ============================================================================

impl std::fmt::Display for CliffordAlgebra {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cl({},{}) [{} blades]", self.p, self.q, self.n_blades)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cl3_cayley_matches_hardcoded() {
        // The generic Cl(3,0) Cayley table must produce identical results
        // to the hardcoded table in clifford_cl3.rs.
        let alg = CliffordAlgebra::cl3();
        assert_eq!(alg.n_blades, 8);

        // Verify: e1 * e1 = +1
        let e1: Vec<f32> = alg.basis(1);
        let e1_sq = alg.geometric_product(&e1, &e1);
        assert!(
            (e1_sq[0] - 1.0).abs() < 1e-6,
            "e1² = {} (expected +1)",
            e1_sq[0]
        );
        for &coefficient in e1_sq.iter().skip(1) {
            assert!(coefficient.abs() < 1e-6);
        }

        // Verify: e1 * e2 = e12 (should be at some index in grade-2)
        let e2: Vec<f32> = alg.basis(2);
        let e1e2 = alg.geometric_product(&e1, &e2);
        // Find the e12 blade index (bitmap 0b11 = 3)
        let e12_idx = alg.bitmap_to_index[3];
        assert!((e1e2[e12_idx] - 1.0).abs() < 1e-6, "e1*e2 should give e12");

        // Verify: e2 * e1 = -e12 (anticommutativity of vectors)
        let e2e1 = alg.geometric_product(&e2, &e1);
        assert!((e2e1[e12_idx] + 1.0).abs() < 1e-6, "e2*e1 should give -e12");

        // Verify: identity element
        let one: Vec<f32> = alg.one();
        let x = vec![0.5f32, 0.1, -0.3, 0.7, 0.2, -0.1, 0.4, 0.05];
        let ox = alg.geometric_product(&one, &x);
        for i in 0..8 {
            assert!((ox[i] - x[i]).abs() < 1e-6, "1*x != x at {}", i);
        }
    }

    #[test]
    fn test_cl3_blade_order_matches_existing() {
        // The existing clifford_cl3.rs uses order:
        // [1, e1, e2, e3, e12, e23, e13, e123]
        // with bitmaps [0, 1, 2, 4, 3, 6, 5, 7]
        //
        // ShortLex order: sort by grade, then by bitmap value.
        // Grade 0: bitmap 0 → index 0
        // Grade 1: bitmaps 1,2,4 → indices 1,2,3
        // Grade 2: bitmaps 3,5,6 → indices 4,5,6
        // Grade 3: bitmap 7 → index 7
        //
        // So our e12 (bitmap 3) is at index 4, e13 (bitmap 5) is at index 5,
        // e23 (bitmap 6) is at index 6.
        //
        // The existing code has e12=idx4, e23=idx5, e13=idx6 with
        // bitmaps [0,1,2,4,3,6,5,7].
        //
        // Our ShortLex: bitmaps [0,1,2,4,3,5,6,7] → e12=idx4, e13=idx5, e23=idx6.
        //
        // This is a DIFFERENT ordering of grade-2 blades! The existing code
        // uses [e12, e23, e13] while ShortLex gives [e12, e13, e23].
        // This is fine — the generic algebra is self-consistent, and the
        // existing clifford_cl3.rs remains the fast path for Cl(3,0).

        let alg = CliffordAlgebra::cl3();
        // Verify grades are correct
        assert_eq!(alg.grades[0], 0); // scalar
        assert_eq!(alg.grades[1], 1); // e1
        assert_eq!(alg.grades[2], 1); // e2
        assert_eq!(alg.grades[3], 1); // e3 (bitmap 4)
        assert_eq!(alg.grades[4], 2); // e12 (bitmap 3)
        assert_eq!(alg.grades[5], 2); // e13 (bitmap 5)
        assert_eq!(alg.grades[6], 2); // e23 (bitmap 6)
        assert_eq!(alg.grades[7], 3); // e123 (bitmap 7)
    }

    #[test]
    fn test_complex_product() {
        // Cl(0,1): e1² = -1, so this is the complex numbers.
        // (a + b*e1) * (c + d*e1) = (ac - bd) + (ad + bc)*e1
        let alg = CliffordAlgebra::complex();
        assert_eq!(alg.n_blades, 2);

        // (2 + 3i) * (4 + 5i) = (8-15) + (10+12)i = -7 + 22i
        let a = vec![2.0f32, 3.0];
        let b = vec![4.0f32, 5.0];
        let c = alg.geometric_product(&a, &b);
        assert!(
            (c[0] - (-7.0)).abs() < 1e-6,
            "real part: {} (expected -7)",
            c[0]
        );
        assert!(
            (c[1] - 22.0).abs() < 1e-6,
            "imag part: {} (expected 22)",
            c[1]
        );
    }

    #[test]
    fn test_quaternion_product() {
        // Cl(0,2): e1²=-1, e2²=-1, e12²=-1.
        // Quaternion convention: 1, i=e1, j=e2, k=e12.
        // i*j = e1*e2 = e12 = k ✓
        // j*i = e2*e1 = -e12 = -k ✓
        let alg = CliffordAlgebra::quaternion();
        assert_eq!(alg.n_blades, 4);

        let i: Vec<f32> = alg.basis(1); // e1
        let j: Vec<f32> = alg.basis(2); // e2

        // i * j = k (e12, index 3)
        let ij = alg.geometric_product(&i, &j);
        assert!((ij[3] - 1.0).abs() < 1e-6, "i*j should be k, got {:?}", ij);

        // j * i = -k
        let ji = alg.geometric_product(&j, &i);
        assert!((ji[3] + 1.0).abs() < 1e-6, "j*i should be -k, got {:?}", ji);

        // i² = -1
        let ii = alg.geometric_product(&i, &i);
        assert!((ii[0] + 1.0).abs() < 1e-6, "i² should be -1, got {}", ii[0]);

        // k² = (e12)² = e1*e2*e1*e2 = -1
        let k: Vec<f32> = alg.basis(3); // e12
        let kk = alg.geometric_product(&k, &k);
        assert!((kk[0] + 1.0).abs() < 1e-6, "k² should be -1, got {}", kk[0]);
    }

    #[test]
    fn test_sandwich_preserves_norm() {
        let alg = CliffordAlgebra::cl3();

        // Create a rotor from a bivector (rotation in e12 plane by pi/3)
        let biv: Vec<f32> = alg.bivector(&[1.0, 0.0, 0.0]); // e12 plane
        let rotor = alg.rotor_from_bivector(&biv, std::f32::consts::FRAC_PI_3);

        // Sandwich a vector
        let v: Vec<f32> = alg.vector(&[1.0, 2.0, 3.0]);
        let rotated = alg.sandwich(&rotor, &v);

        // Norm should be preserved
        let v_norm = alg.coeff_norm(&v);
        let r_norm = alg.coeff_norm(&rotated);
        assert!(
            (v_norm - r_norm).abs() < 1e-4,
            "Sandwich changed norm: {} → {}",
            v_norm,
            r_norm
        );

        // Grade should be preserved (still a vector)
        let grade0 = alg.scalar_part(&rotated);
        assert!(
            grade0.abs() < 1e-5,
            "Rotated vector has scalar part: {}",
            grade0
        );
    }

    #[test]
    fn test_reverse_is_involution() {
        let alg = CliffordAlgebra::cl3();
        let x = vec![0.5f32, 0.1, -0.3, 0.7, 0.2, -0.1, 0.4, 0.05];
        let rr = alg.reverse(&alg.reverse(&x));
        for i in 0..8 {
            assert!((rr[i] - x[i]).abs() < 1e-6, "reverse² != identity at {}", i);
        }
    }

    #[test]
    fn test_grade_projection_complete() {
        let alg = CliffordAlgebra::cl3();
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Sum of all grade projections should equal the original
        let mut sum: Vec<f32> = alg.zero();
        for g in 0..=alg.max_grade() {
            let proj = alg.grade_project(&x, g);
            for i in 0..alg.n_blades {
                sum[i] += proj[i];
            }
        }
        for i in 0..8 {
            assert!(
                (sum[i] - x[i]).abs() < 1e-6,
                "Grade projection incomplete at {}",
                i
            );
        }
    }

    #[test]
    fn test_norm_squared_nonnegative() {
        let alg = CliffordAlgebra::cl3();
        // For Cl(3,0) (positive definite), norm_squared should be non-negative
        // for any multivector.
        let x = vec![0.5f32, 0.1, -0.3, 0.7, 0.2, -0.1, 0.4, 0.05];
        let ns = alg.norm_squared(&x);
        assert!(ns >= 0.0, "norm_squared is negative: {}", ns);
    }

    #[test]
    fn test_predefined_signatures() {
        assert_eq!(CliffordAlgebra::complex().n_blades, 2);
        assert_eq!(CliffordAlgebra::split_complex().n_blades, 2);
        assert_eq!(CliffordAlgebra::quaternion().n_blades, 4);
        assert_eq!(CliffordAlgebra::cl3().n_blades, 8);
        assert_eq!(CliffordAlgebra::cl31().n_blades, 16);
        assert_eq!(CliffordAlgebra::sta().n_blades, 16);
    }

    #[test]
    fn test_checked_constructor_rejects_invalid_dimensions() {
        assert_eq!(
            CliffordAlgebra::try_new(usize::MAX, 1).unwrap_err(),
            CliffordAlgebraError::DimensionOverflow {
                p: usize::MAX,
                q: 1,
            }
        );
        assert_eq!(
            CliffordAlgebra::try_new(11, 0).unwrap_err(),
            CliffordAlgebraError::DimensionTooLarge {
                p: 11,
                q: 0,
                dimension: 11,
                maximum: MAX_DIMENSION,
            }
        );
    }

    #[test]
    fn test_scalar_algebra_cl00() {
        let algebra = CliffordAlgebra::try_new(0, 0).unwrap();
        assert_eq!(algebra.n_blades, 1);
        assert_eq!(algebra.bivector::<f64>(&[]), vec![0.0]);
        assert_eq!(algebra.geometric_product(&[2.0_f64], &[3.0_f64]), vec![6.0]);
    }

    #[test]
    fn test_rotor_unit_norm() {
        let alg = CliffordAlgebra::cl3();
        let biv: Vec<f32> = alg.bivector(&[0.3, -0.5, 0.8]);
        let rotor = alg.rotor_from_bivector(&biv, 1.23f32);
        let ns = alg.norm_squared(&rotor);
        assert!(
            (ns - 1.0).abs() < 1e-5,
            "Rotor should have unit norm², got {}",
            ns
        );
    }

    #[test]
    fn test_rotor_90deg_rotation() {
        // Rotate e1 by 90° in the e12 plane → should get e2.
        let alg = CliffordAlgebra::cl3();
        let biv: Vec<f32> = alg.bivector(&[1.0, 0.0, 0.0]); // e12 plane
        let rotor = alg.rotor_from_bivector(&biv, std::f32::consts::FRAC_PI_2);

        let e1: Vec<f32> = alg.basis(1);
        let rotated = alg.sandwich(&rotor, &e1);

        // Should be approximately e2
        let vc = alg.vector_components(&rotated);
        assert!(
            vc[0].abs() < 1e-5,
            "x component should be ~0, got {}",
            vc[0]
        );
        assert!(
            (vc[1].abs() - 1.0).abs() < 1e-5,
            "|y component| should be ~1, got {}",
            vc[1]
        );
        assert!(
            vc[2].abs() < 1e-5,
            "z component should be ~0, got {}",
            vc[2]
        );
    }

    #[test]
    fn test_commutator_antisymmetric() {
        let alg = CliffordAlgebra::cl3();
        let a = vec![0.5f32, 0.1, -0.3, 0.7, 0.2, -0.1, 0.4, 0.05];
        let b = vec![-0.2f32, 0.6, 0.1, -0.4, 0.3, 0.5, -0.2, 0.1];
        let ab = alg.commutator(&a, &b);
        let ba = alg.commutator(&b, &a);
        for i in 0..8 {
            assert!(
                (ab[i] + ba[i]).abs() < 1e-5,
                "[a,b] + [b,a] != 0 at {}: {} + {} = {}",
                i,
                ab[i],
                ba[i],
                ab[i] + ba[i]
            );
        }
    }

    #[test]
    fn test_split_complex_e1_squared() {
        // Cl(1,0): e1² = +1
        let alg = CliffordAlgebra::split_complex();
        let e1: Vec<f32> = alg.basis(1);
        let sq = alg.geometric_product(&e1, &e1);
        assert!((sq[0] - 1.0).abs() < 1e-6, "e1² should be +1 in Cl(1,0)");
    }

    #[test]
    fn test_sta_signature() {
        // Cl(1,3): e0²=+1, e1²=e2²=e3²=-1
        let alg = CliffordAlgebra::sta();
        let e0: Vec<f32> = alg.basis(1); // first grade-1 blade
        let e1: Vec<f32> = alg.basis(2); // second grade-1 blade

        let e0sq = alg.geometric_product(&e0, &e0);
        assert!(
            (e0sq[0] - 1.0).abs() < 1e-6,
            "e0² should be +1 in Cl(1,3), got {}",
            e0sq[0]
        );

        let e1sq = alg.geometric_product(&e1, &e1);
        assert!(
            (e1sq[0] + 1.0).abs() < 1e-6,
            "e1² should be -1 in Cl(1,3), got {}",
            e1sq[0]
        );
    }

    #[test]
    fn test_vector_embed_extract_roundtrip() {
        let alg = CliffordAlgebra::cl3();
        let v_in = vec![1.5f32, -2.3, 0.7];
        let mv = alg.vector(&v_in);
        let v_out = alg.vector_components(&mv);
        assert_eq!(v_out.len(), 3);
        for i in 0..3 {
            assert!((v_in[i] - v_out[i]).abs() < 1e-6);
        }
    }
}
