//! Sparse multivector: stores only the blades it needs.
//!
//! This is the core type of the library. Each concrete GA type (Vec, Biv, Rotor,
//! Point, Motor, etc.) is a type alias for `Multivector<N>` with a specific N
//! matching the number of blades in that type.
//!
//! The blade layout (which basis blades map to which indices) is defined
//! per-algebra via the `Blade` trait and associated types.

use core::fmt;
use core::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

/// A sparse multivector storing exactly `N` scalar coefficients.
///
/// The mapping from coefficient index to basis blade is defined externally
/// by the algebra module (e.g., `ega3d`, `cga3d`).
#[derive(Clone, Copy, PartialEq)]
pub struct Multivector<const N: usize> {
    /// Coefficients in the basis layout defined by the owning algebra module.
    pub data: [f32; N],
}

impl<const N: usize> Multivector<N> {
    /// Zero multivector.
    #[inline]
    pub const fn zero() -> Self {
        Self { data: [0.0; N] }
    }

    /// Create from an array of coefficients.
    #[inline]
    pub const fn new(data: [f32; N]) -> Self {
        Self { data }
    }

    /// Number of components.
    #[inline]
    pub const fn len(&self) -> usize {
        N
    }

    /// Squared norm (sum of squares of all coefficients).
    #[inline]
    pub fn norm_sq(&self) -> f32 {
        let mut sum = 0.0f32;
        let mut i = 0;
        while i < N {
            sum += self.data[i] * self.data[i];
            i += 1;
        }
        sum
    }

    /// Euclidean norm.
    #[inline]
    pub fn norm(&self) -> f32 {
        self.norm_sq().sqrt()
    }

    /// Normalize to unit length. Returns zero if norm is zero.
    #[inline]
    pub fn normalized(&self) -> Self {
        let n = self.norm();
        if n == 0.0 {
            Self::zero()
        } else {
            *self * (1.0 / n)
        }
    }

    /// Scale by a scalar.
    #[inline]
    pub fn scale(&self, s: f32) -> Self {
        let mut result = Self::zero();
        let mut i = 0;
        while i < N {
            result.data[i] = self.data[i] * s;
            i += 1;
        }
        result
    }
}

// --- Arithmetic impls ---

impl<const N: usize> Add for Multivector<N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut result = Self::zero();
        let mut i = 0;
        while i < N {
            result.data[i] = self.data[i] + rhs.data[i];
            i += 1;
        }
        result
    }
}

impl<const N: usize> Sub for Multivector<N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let mut result = Self::zero();
        let mut i = 0;
        while i < N {
            result.data[i] = self.data[i] - rhs.data[i];
            i += 1;
        }
        result
    }
}

impl<const N: usize> Neg for Multivector<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let mut result = Self::zero();
        let mut i = 0;
        while i < N {
            result.data[i] = -self.data[i];
            i += 1;
        }
        result
    }
}

impl<const N: usize> Mul<f32> for Multivector<N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        self.scale(rhs)
    }
}

impl<const N: usize> Mul<Multivector<N>> for f32 {
    type Output = Multivector<N>;
    #[inline]
    fn mul(self, rhs: Multivector<N>) -> Multivector<N> {
        rhs.scale(self)
    }
}

impl<const N: usize> Index<usize> for Multivector<N> {
    type Output = f32;
    #[inline]
    fn index(&self, idx: usize) -> &f32 {
        &self.data[idx]
    }
}

impl<const N: usize> IndexMut<usize> for Multivector<N> {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut f32 {
        &mut self.data[idx]
    }
}

impl<const N: usize> fmt::Debug for Multivector<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mv{:?}", &self.data[..])
    }
}

impl<const N: usize> fmt::Display for Multivector<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for i in 0..N {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.4}", self.data[i])?;
        }
        write!(f, "]")
    }
}

impl<const N: usize> Default for Multivector<N> {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let v = Multivector::<3>::zero();
        assert_eq!(v.data, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_add() {
        let a = Multivector::new([1.0, 2.0, 3.0]);
        let b = Multivector::new([4.0, 5.0, 6.0]);
        let c = a + b;
        assert_eq!(c.data, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub() {
        let a = Multivector::new([4.0, 5.0, 6.0]);
        let b = Multivector::new([1.0, 2.0, 3.0]);
        let c = a - b;
        assert_eq!(c.data, [3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_scale() {
        let a = Multivector::new([1.0, 2.0, 3.0]);
        let b = a * 2.0;
        assert_eq!(b.data, [2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_norm() {
        let a = Multivector::new([3.0, 4.0]);
        assert_eq!(a.norm(), 5.0);
    }

    #[test]
    fn test_normalized() {
        let a = Multivector::new([3.0, 4.0]);
        let n = a.normalized();
        assert!((n.norm() - 1.0).abs() < 1e-6);
    }
}
