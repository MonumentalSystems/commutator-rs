//! Scalar, vector, and bivector fields on 3D grids.
//!
//! Port of `vsr_field.h`. Provides a generic `Field<T>` on a cubic lattice
//! with trilinear interpolation, useful for physics simulation (force fields,
//! velocity fields, etc.).

use crate::cga3d::*;

/// A 3D cubic lattice of grid points with associated data.
///
/// The grid spans from `(0, 0, 0)` to `((w-1)*sx, (h-1)*sy, (d-1)*sz)`
/// where `sx, sy, sz` are the spacing in each dimension.
/// Every axis contains at least two grid points so interpolation and
/// one-sided finite differences always have a neighboring sample.
#[derive(Clone, Debug)]
pub struct Field<T: Clone> {
    width: usize,
    height: usize,
    depth: usize,
    spacing_x: f32,
    spacing_y: f32,
    spacing_z: f32,
    data: Vec<T>,
    grid: Vec<Pnt>,
}

impl<T: Clone + Default> Field<T> {
    /// Create a new field with given dimensions and spacing.
    ///
    /// # Panics
    ///
    /// Panics if any dimension is smaller than two grid points, if the total
    /// number of grid points overflows `usize`, or if a spacing is non-finite
    /// or not positive.
    pub fn new(width: usize, height: usize, depth: usize) -> Self {
        Self::with_spacing(width, height, depth, 1.0, 1.0, 1.0)
    }

    /// Create a new field with given dimensions and custom spacing.
    ///
    /// # Panics
    ///
    /// Panics if any dimension is smaller than two grid points.
    pub fn with_spacing(
        width: usize,
        height: usize,
        depth: usize,
        sx: f32,
        sy: f32,
        sz: f32,
    ) -> Self {
        assert!(
            width >= 2 && height >= 2 && depth >= 2,
            "field dimensions must each contain at least two grid points"
        );
        assert!(
            sx.is_finite() && sx > 0.0 && sy.is_finite() && sy > 0.0 && sz.is_finite() && sz > 0.0,
            "field spacing must be finite and positive"
        );
        let num = width
            .checked_mul(height)
            .and_then(|area| area.checked_mul(depth))
            .expect("field dimensions overflow usize");
        let mut grid = Vec::with_capacity(num);
        for i in 0..width {
            for j in 0..height {
                for k in 0..depth {
                    grid.push(point(i as f32 * sx, j as f32 * sy, k as f32 * sz));
                }
            }
        }
        Self {
            width,
            height,
            depth,
            spacing_x: sx,
            spacing_y: sy,
            spacing_z: sz,
            data: vec![T::default(); num],
            grid,
        }
    }

    /// Total number of grid points.
    pub fn num(&self) -> usize {
        self.data.len()
    }

    /// Return the grid dimensions as `(width, height, depth)`.
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.width, self.height, self.depth)
    }

    /// Return the grid spacing as `(sx, sy, sz)`.
    pub fn spacing(&self) -> (f32, f32, f32) {
        (self.spacing_x, self.spacing_y, self.spacing_z)
    }

    /// Borrow all field samples in row-major grid order.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Mutably borrow all field samples without changing the field shape.
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Borrow all conformal grid points in row-major grid order.
    pub fn grid_points(&self) -> &[Pnt] {
        &self.grid
    }

    /// Convert (i, j, k) coordinates to a flat index.
    #[inline]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        i * self.height * self.depth + j * self.depth + k
    }

    /// Get data at grid coordinates (i, j, k).
    pub fn at(&self, i: usize, j: usize, k: usize) -> &T {
        &self.data[self.idx(i, j, k)]
    }

    /// Set data at grid coordinates (i, j, k).
    pub fn at_mut(&mut self, i: usize, j: usize, k: usize) -> &mut T {
        let idx = self.idx(i, j, k);
        &mut self.data[idx]
    }

    /// Get the grid point (CGA point) at (i, j, k).
    pub fn grid_point(&self, i: usize, j: usize, k: usize) -> &Pnt {
        &self.grid[self.idx(i, j, k)]
    }

    /// Resize the field, resetting all data to default.
    ///
    /// # Panics
    ///
    /// Panics if any dimension is smaller than two grid points.
    pub fn resize(&mut self, w: usize, h: usize, d: usize) {
        *self = Self::with_spacing(w, h, d, self.spacing_x, self.spacing_y, self.spacing_z);
    }

    /// Zero out all data.
    pub fn zero(&mut self)
    where
        T: Default,
    {
        for d in self.data.iter_mut() {
            *d = T::default();
        }
    }
}

// ============================================================================
// Scalar field specialization
// ============================================================================

/// A scalar field on a 3D grid.
pub type ScalarField = Field<f32>;

impl ScalarField {
    /// Trilinear interpolation at normalized coordinates (u, v, w) in `[0, 1]`.
    pub fn sample(&self, u: f32, v: f32, w: f32) -> f32 {
        let u = u.clamp(0.0, 1.0);
        let v = v.clamp(0.0, 1.0);
        let w = w.clamp(0.0, 1.0);

        let fi = u * (self.width - 1) as f32;
        let fj = v * (self.height - 1) as f32;
        let fk = w * (self.depth - 1) as f32;

        let i0 = (fi as usize).min(self.width - 2);
        let j0 = (fj as usize).min(self.height - 2);
        let k0 = (fk as usize).min(self.depth - 2);
        let i1 = i0 + 1;
        let j1 = j0 + 1;
        let k1 = k0 + 1;

        let fu = fi - i0 as f32;
        let fv = fj - j0 as f32;
        let fw = fk - k0 as f32;

        let c000 = *self.at(i0, j0, k0);
        let c100 = *self.at(i1, j0, k0);
        let c010 = *self.at(i0, j1, k0);
        let c110 = *self.at(i1, j1, k0);
        let c001 = *self.at(i0, j0, k1);
        let c101 = *self.at(i1, j0, k1);
        let c011 = *self.at(i0, j1, k1);
        let c111 = *self.at(i1, j1, k1);

        let c00 = c000 * (1.0 - fu) + c100 * fu;
        let c10 = c010 * (1.0 - fu) + c110 * fu;
        let c01 = c001 * (1.0 - fu) + c101 * fu;
        let c11 = c011 * (1.0 - fu) + c111 * fu;

        let c0 = c00 * (1.0 - fv) + c10 * fv;
        let c1 = c01 * (1.0 - fv) + c11 * fv;

        c0 * (1.0 - fw) + c1 * fw
    }

    /// Compute gradient at grid point (i, j, k) using central differences.
    /// Returns (dF/dx, dF/dy, dF/dz).
    pub fn gradient(&self, i: usize, j: usize, k: usize) -> (f32, f32, f32) {
        let dx = if i > 0 && i < self.width - 1 {
            (*self.at(i + 1, j, k) - *self.at(i - 1, j, k)) / (2.0 * self.spacing_x)
        } else if i == 0 {
            (*self.at(i + 1, j, k) - *self.at(i, j, k)) / self.spacing_x
        } else {
            (*self.at(i, j, k) - *self.at(i - 1, j, k)) / self.spacing_x
        };

        let dy = if j > 0 && j < self.height - 1 {
            (*self.at(i, j + 1, k) - *self.at(i, j - 1, k)) / (2.0 * self.spacing_y)
        } else if j == 0 {
            (*self.at(i, j + 1, k) - *self.at(i, j, k)) / self.spacing_y
        } else {
            (*self.at(i, j, k) - *self.at(i, j - 1, k)) / self.spacing_y
        };

        let dz = if k > 0 && k < self.depth - 1 {
            (*self.at(i, j, k + 1) - *self.at(i, j, k - 1)) / (2.0 * self.spacing_z)
        } else if k == 0 {
            (*self.at(i, j, k + 1) - *self.at(i, j, k)) / self.spacing_z
        } else {
            (*self.at(i, j, k) - *self.at(i, j, k - 1)) / self.spacing_z
        };

        (dx, dy, dz)
    }

    /// Compute Laplacian at grid point (i, j, k) using central differences.
    pub fn laplacian(&self, i: usize, j: usize, k: usize) -> f32 {
        let c = *self.at(i, j, k);
        let mut lap = 0.0;

        if i > 0 && i < self.width - 1 {
            lap += (*self.at(i + 1, j, k) - 2.0 * c + *self.at(i - 1, j, k))
                / (self.spacing_x * self.spacing_x);
        }
        if j > 0 && j < self.height - 1 {
            lap += (*self.at(i, j + 1, k) - 2.0 * c + *self.at(i, j - 1, k))
                / (self.spacing_y * self.spacing_y);
        }
        if k > 0 && k < self.depth - 1 {
            lap += (*self.at(i, j, k + 1) - 2.0 * c + *self.at(i, j, k - 1))
                / (self.spacing_z * self.spacing_z);
        }

        lap
    }
}

// ============================================================================
// Vector field specialization
// ============================================================================

/// A 3D vector field (stores `[f32; 3]` at each grid point).
pub type VectorField = Field<[f32; 3]>;

impl VectorField {
    /// Compute divergence at grid point (i, j, k) using central differences.
    pub fn divergence(&self, i: usize, j: usize, k: usize) -> f32 {
        let mut div = 0.0;

        if i > 0 && i < self.width - 1 {
            div += (self.at(i + 1, j, k)[0] - self.at(i - 1, j, k)[0]) / (2.0 * self.spacing_x);
        }
        if j > 0 && j < self.height - 1 {
            div += (self.at(i, j + 1, k)[1] - self.at(i, j - 1, k)[1]) / (2.0 * self.spacing_y);
        }
        if k > 0 && k < self.depth - 1 {
            div += (self.at(i, j, k + 1)[2] - self.at(i, j, k - 1)[2]) / (2.0 * self.spacing_z);
        }

        div
    }

    /// Compute curl at grid point (i, j, k) using central differences.
    /// Returns the curl vector (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy).
    pub fn curl(&self, i: usize, j: usize, k: usize) -> [f32; 3] {
        let dfz_dy = if j > 0 && j < self.height - 1 {
            (self.at(i, j + 1, k)[2] - self.at(i, j - 1, k)[2]) / (2.0 * self.spacing_y)
        } else {
            0.0
        };

        let dfy_dz = if k > 0 && k < self.depth - 1 {
            (self.at(i, j, k + 1)[1] - self.at(i, j, k - 1)[1]) / (2.0 * self.spacing_z)
        } else {
            0.0
        };

        let dfx_dz = if k > 0 && k < self.depth - 1 {
            (self.at(i, j, k + 1)[0] - self.at(i, j, k - 1)[0]) / (2.0 * self.spacing_z)
        } else {
            0.0
        };

        let dfz_dx = if i > 0 && i < self.width - 1 {
            (self.at(i + 1, j, k)[2] - self.at(i - 1, j, k)[2]) / (2.0 * self.spacing_x)
        } else {
            0.0
        };

        let dfy_dx = if i > 0 && i < self.width - 1 {
            (self.at(i + 1, j, k)[1] - self.at(i - 1, j, k)[1]) / (2.0 * self.spacing_x)
        } else {
            0.0
        };

        let dfx_dy = if j > 0 && j < self.height - 1 {
            (self.at(i, j + 1, k)[0] - self.at(i, j - 1, k)[0]) / (2.0 * self.spacing_y)
        } else {
            0.0
        };

        [dfz_dy - dfy_dz, dfx_dz - dfz_dx, dfy_dx - dfx_dy]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-3;
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_scalar_field_creation() {
        let mut f = ScalarField::new(3, 3, 3);
        assert_eq!(f.num(), 27);
        assert_eq!(f.dimensions(), (3, 3, 3));
        assert_eq!(f.spacing(), (1.0, 1.0, 1.0));
        assert_eq!(f.data().len(), 27);
        assert_eq!(f.grid_points().len(), 27);
        f.data_mut()[0] = 2.0;
        assert_eq!(f.data()[0], 2.0);
    }

    #[test]
    #[should_panic(expected = "field dimensions must each contain at least two grid points")]
    fn test_scalar_field_rejects_zero_extent() {
        let _ = ScalarField::new(0, 2, 2);
    }

    #[test]
    #[should_panic(expected = "field dimensions must each contain at least two grid points")]
    fn test_scalar_field_rejects_singleton_extent() {
        let _ = ScalarField::new(2, 1, 2);
    }

    #[test]
    #[should_panic(expected = "field dimensions must each contain at least two grid points")]
    fn test_vector_field_rejects_singleton_extent() {
        let _ = VectorField::new(2, 2, 1);
    }

    #[test]
    #[should_panic(expected = "field spacing must be finite and positive")]
    fn test_field_rejects_invalid_spacing() {
        let _ = ScalarField::with_spacing(2, 2, 2, 1.0, 0.0, 1.0);
    }

    #[test]
    fn test_scalar_field_idx() {
        let f = ScalarField::new(3, 4, 5);
        // idx(0,0,0) = 0
        assert_eq!(f.idx(0, 0, 0), 0);
        // idx(1,0,0) = 4*5 = 20
        assert_eq!(f.idx(1, 0, 0), 20);
    }

    #[test]
    fn test_scalar_field_set_get() {
        let mut f = ScalarField::new(3, 3, 3);
        *f.at_mut(1, 1, 1) = 42.0;
        assert!(approx(*f.at(1, 1, 1), 42.0));
    }

    #[test]
    fn test_scalar_field_sample_constant() {
        let mut f = ScalarField::new(3, 3, 3);
        // Set all values to 5.0
        for d in f.data.iter_mut() {
            *d = 5.0;
        }
        // Sampling anywhere should return 5.0
        assert!(approx(f.sample(0.5, 0.5, 0.5), 5.0));
        assert!(approx(f.sample(0.0, 0.0, 0.0), 5.0));
        assert!(approx(f.sample(1.0, 1.0, 1.0), 5.0));
    }

    #[test]
    fn test_scalar_field_sample_linear() {
        let mut f = ScalarField::new(2, 2, 2);
        // Set values: f(0,0,0)=0, f(1,0,0)=10, all others 0
        *f.at_mut(0, 0, 0) = 0.0;
        *f.at_mut(1, 0, 0) = 10.0;
        let val = f.sample(0.5, 0.0, 0.0);
        assert!(approx(val, 5.0), "val={}", val);
    }

    #[test]
    fn test_scalar_field_gradient() {
        let mut f = ScalarField::new(5, 5, 5);
        // Set a linear field: f(i,j,k) = i
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    *f.at_mut(i, j, k) = i as f32;
                }
            }
        }
        let (gx, gy, gz) = f.gradient(2, 2, 2);
        assert!(approx(gx, 1.0), "gx={}", gx);
        assert!(approx(gy, 0.0), "gy={}", gy);
        assert!(approx(gz, 0.0), "gz={}", gz);
    }

    #[test]
    fn test_scalar_field_laplacian_linear() {
        let mut f = ScalarField::new(5, 5, 5);
        // Linear field: Laplacian should be 0
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    *f.at_mut(i, j, k) = i as f32 + j as f32;
                }
            }
        }
        let lap = f.laplacian(2, 2, 2);
        assert!(approx(lap, 0.0), "laplacian={}", lap);
    }

    #[test]
    fn test_vector_field_creation() {
        let f = VectorField::new(3, 3, 3);
        assert_eq!(f.num(), 27);
    }

    #[test]
    fn test_vector_field_divergence() {
        let mut f = VectorField::new(5, 5, 5);
        // Set a constant vector field: divergence should be 0
        for d in f.data.iter_mut() {
            *d = [1.0, 2.0, 3.0];
        }
        let div = f.divergence(2, 2, 2);
        assert!(approx(div, 0.0), "divergence={}", div);
    }

    #[test]
    fn test_vector_field_curl_constant() {
        let mut f = VectorField::new(5, 5, 5);
        // Constant field: curl should be zero
        for d in f.data.iter_mut() {
            *d = [1.0, 0.0, 0.0];
        }
        let c = f.curl(2, 2, 2);
        assert!(approx(c[0], 0.0));
        assert!(approx(c[1], 0.0));
        assert!(approx(c[2], 0.0));
    }

    #[test]
    fn test_field_grid_points() {
        let f = ScalarField::with_spacing(3, 3, 3, 2.0, 2.0, 2.0);
        let p = f.grid_point(2, 2, 2);
        let (x, y, z) = Round::location(p);
        assert!(approx(x, 4.0), "x={}", x);
        assert!(approx(y, 4.0), "y={}", y);
        assert!(approx(z, 4.0), "z={}", z);
    }

    #[test]
    fn test_field_zero() {
        let mut f = ScalarField::new(3, 3, 3);
        for d in f.data.iter_mut() {
            *d = 42.0;
        }
        f.zero();
        for d in f.data.iter() {
            assert!(approx(*d, 0.0));
        }
    }
}
