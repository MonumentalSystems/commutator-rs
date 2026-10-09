//! Metric signature definitions for geometric algebras.
//!
//! Port of Versor's metric system. A metric defines the algebra's signature
//! (how many positive, negative, and zero basis vectors).

/// Metric signature for a geometric algebra.
///
/// `signs[i]` is +1, -1, or 0 for basis vector `e_{i+1}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Metric<const DIM: usize> {
    /// Per-generator square: positive, negative, or zero.
    pub signs: [i32; DIM],
}

impl<const DIM: usize> Metric<DIM> {
    /// Construct a metric from its ordered generator squares.
    pub const fn new(signs: [i32; DIM]) -> Self {
        Self { signs }
    }

    /// Number of dimensions.
    pub const fn dim(&self) -> usize {
        DIM
    }
}

// Standard metrics as const values

/// Euclidean 2D: Cl(2,0)
pub const EUCLIDEAN_2D: Metric<2> = Metric::new([1, 1]);

/// Euclidean 3D: Cl(3,0)
pub const EUCLIDEAN_3D: Metric<3> = Metric::new([1, 1, 1]);

/// Conformal 3D (CGA): Cl(4,1) — basis: e1, e2, e3, e+, e-
/// Convention: e+ squares to +1, e- squares to -1
pub const CONFORMAL_3D: Metric<5> = Metric::new([1, 1, 1, 1, -1]);

/// Projective 3D (PGA): Cl(3,0,1) — basis: e1, e2, e3, e0
/// e0 is degenerate (squares to 0)
pub const PROJECTIVE_3D: Metric<4> = Metric::new([1, 1, 1, 0]);

/// Spacetime Algebra (STA): Cl(1,3)
pub const SPACETIME: Metric<4> = Metric::new([1, -1, -1, -1]);
