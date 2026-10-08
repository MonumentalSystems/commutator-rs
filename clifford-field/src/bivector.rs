//! Precision-generic Cl(1,3) bivectors and spatial fields.

use crate::{bivector_rayon_threshold as par_threshold, FieldScalar};
use clifford_core::CliffordAlgebra;
use rayon::prelude::*;
use std::sync::OnceLock;

const E1_IDX: usize = 5;
const E2_IDX: usize = 6;
const B3_IDX: usize = 7;
const E3_IDX: usize = 8;
const B2_IDX: usize = 9;
const B1_IDX: usize = 10;

const COMP_TO_STA: [usize; 6] = [E1_IDX, E2_IDX, E3_IDX, B1_IDX, B2_IDX, B3_IDX];
const STA_TO_COMP: [(usize, usize); 6] = [
    (E1_IDX, 0),
    (E2_IDX, 1),
    (E3_IDX, 2),
    (B1_IDX, 3),
    (B2_IDX, 4),
    (B3_IDX, 5),
];

type CommDenseTable = [[[f32; 6]; 6]; 6];

fn build_comm_table() -> CommDenseTable {
    let algebra = CliffordAlgebra::sta();
    let mut table = [[[0.0f32; 6]; 6]; 6];

    for i in 0..6 {
        for j in 0..6 {
            let mut a = vec![0.0f32; 16];
            let mut b = vec![0.0f32; 16];
            a[COMP_TO_STA[i]] = 1.0;
            b[COMP_TO_STA[j]] = 1.0;

            let ab = algebra.geometric_product(&a, &b);
            let ba = algebra.geometric_product(&b, &a);
            for &(sta_idx, comp_idx) in &STA_TO_COMP {
                let coefficient = ab[sta_idx] - ba[sta_idx];
                if coefficient.abs() > 1e-10 {
                    table[i][j][comp_idx] = coefficient;
                }
            }
        }
    }

    table
}

fn comm_table() -> &'static CommDenseTable {
    static TABLE: OnceLock<CommDenseTable> = OnceLock::new();
    TABLE.get_or_init(build_comm_table)
}

// ============================================================================
// StaBivector — compact 6-component STA bivector
// ============================================================================

/// A bivector in Cl(1,3) spacetime algebra, stored as 6 components:
/// [E₁, E₂, E₃, B₁, B₂, B₃] where E_k ↔ γ₀ₖ (electric) and
/// B_k ↔ ½ε_{kjl}γ_{jl} (magnetic).
#[derive(Debug, Clone, Copy)]
pub struct StaBivector<S: FieldScalar> {
    /// Components ordered as `(E1, E2, E3, B1, B2, B3)`.
    pub components: [S; 6],
}

impl<S: FieldScalar> StaBivector<S> {
    /// Return the additive identity bivector.
    pub fn zero() -> Self {
        StaBivector {
            components: [S::ZERO; 6],
        }
    }

    /// Construct a bivector from electric-like and magnetic-like components.
    pub fn new(e1: S, e2: S, e3: S, b1: S, b2: S, b3: S) -> Self {
        StaBivector {
            components: [e1, e2, e3, b1, b2, b3],
        }
    }

    /// Electric field components [E₁, E₂, E₃].
    #[inline]
    pub fn electric(&self) -> [S; 3] {
        [self.components[0], self.components[1], self.components[2]]
    }

    /// Magnetic field components [B₁, B₂, B₃].
    #[inline]
    pub fn magnetic(&self) -> [S; 3] {
        [self.components[3], self.components[4], self.components[5]]
    }

    /// Expand to full 16-component STA multivector.
    pub fn to_sta_multivector(&self) -> Vec<S> {
        let mut mv = vec![S::ZERO; 16];
        for (comp_idx, &sta_idx) in COMP_TO_STA.iter().enumerate() {
            mv[sta_idx] = self.components[comp_idx];
        }
        mv
    }

    /// Extract from a 16-component STA multivector (takes grade-2 part).
    pub fn from_sta_multivector(mv: &[S]) -> Self {
        let mut components = [S::ZERO; 6];
        for &(sta_idx, comp_idx) in &STA_TO_COMP {
            components[comp_idx] = mv[sta_idx];
        }
        StaBivector { components }
    }

    /// Coefficient-space L2 norm squared.
    #[inline]
    pub fn norm_squared(&self) -> S {
        self.components.iter().map(|&x| x * x).sum()
    }

    /// Coefficient-space L2 norm.
    #[inline]
    pub fn norm(&self) -> S {
        self.norm_squared().max(S::from_f64(1e-16)).sqrt()
    }

    /// Bivector commutator [self, other] using dense precomputed table.
    /// This is the key operation: `[A,B] = AB - BA` for STA bivectors.
    /// Returns a bivector (grade-2 output guaranteed by algebra).
    ///
    /// Uses a dense 6×6×6 table (864 bytes, L1-resident) for SIMD-friendly
    /// inner loops — no pointer chasing, no branches.
    #[inline]
    pub fn commutator(&self, other: &StaBivector<S>) -> StaBivector<S> {
        let table = comm_table();
        let a = &self.components;
        let b = &other.components;
        let mut result = [S::ZERO; 6];
        for i in 0..6 {
            let ai = a[i];
            if ai == S::ZERO {
                continue;
            }
            for j in 0..6 {
                let aibj = ai * b[j];
                if aibj == S::ZERO {
                    continue;
                }
                // Inner loop over 6 output components — auto-vectorizable.
                let row = &table[i][j];
                for k in 0..6 {
                    result[k] += aibj * S::from_f32(row[k]);
                }
            }
        }
        StaBivector { components: result }
    }

    /// Add another bivector.
    pub fn add(&self, other: &StaBivector<S>) -> StaBivector<S> {
        let mut c = [S::ZERO; 6];
        for (i, component) in c.iter_mut().enumerate() {
            *component = self.components[i] + other.components[i];
        }
        StaBivector { components: c }
    }

    /// Subtract another bivector.
    pub fn sub(&self, other: &StaBivector<S>) -> StaBivector<S> {
        let mut c = [S::ZERO; 6];
        for (i, component) in c.iter_mut().enumerate() {
            *component = self.components[i] - other.components[i];
        }
        StaBivector { components: c }
    }

    /// Scale by a scalar.
    pub fn scale(&self, s: S) -> StaBivector<S> {
        let mut c = [S::ZERO; 6];
        for (i, component) in c.iter_mut().enumerate() {
            *component = self.components[i] * s;
        }
        StaBivector { components: c }
    }

    /// Add in-place.
    pub fn add_assign(&mut self, other: &StaBivector<S>) {
        for i in 0..6 {
            self.components[i] += other.components[i];
        }
    }

    /// Scale-add in-place: self += s * other.
    pub fn scale_add_assign(&mut self, s: S, other: &StaBivector<S>) {
        for i in 0..6 {
            self.components[i] += s * other.components[i];
        }
    }
}

// ============================================================================
// Chiral decomposition: so(3,1) ≅ su(2)_L ⊕ su(2)_R
// ============================================================================

/// Chiral decomposition of a bivector field.
/// Self-dual (left): B_L = (E + B) / 2
/// Anti-self-dual (right): B_R = (E - B) / 2
pub struct ChiralDecomposition<S: FieldScalar> {
    /// Self-dual (left-chiral) sector: 3 components per point.
    pub left: Vec<[S; 3]>,
    /// Anti-self-dual (right-chiral) sector: 3 components per point.
    pub right: Vec<[S; 3]>,
    /// Order parameter for left sector.
    pub r_left: S,
    /// Order parameter for right sector.
    pub r_right: S,
}

/// Decompose a single bivector into chiral sectors.
pub fn chiral_split<S: FieldScalar>(bv: &StaBivector<S>) -> ([S; 3], [S; 3]) {
    let e = bv.electric();
    let b = bv.magnetic();
    let left = [
        (e[0] + b[0]) * S::HALF,
        (e[1] + b[1]) * S::HALF,
        (e[2] + b[2]) * S::HALF,
    ];
    let right = [
        (e[0] - b[0]) * S::HALF,
        (e[1] - b[1]) * S::HALF,
        (e[2] - b[2]) * S::HALF,
    ];
    (left, right)
}

/// Compute order parameter for a set of 3-component vectors.
pub(crate) fn sector_order_parameter<S: FieldScalar>(sectors: &[[S; 3]]) -> S {
    if sectors.is_empty() {
        return S::ZERO;
    }
    let n = S::from_usize(sectors.len());
    let mut mean = [S::ZERO; 3];
    for s in sectors {
        mean[0] += s[0];
        mean[1] += s[1];
        mean[2] += s[2];
    }
    mean[0] /= n;
    mean[1] /= n;
    mean[2] /= n;
    (mean[0] * mean[0] + mean[1] * mean[1] + mean[2] * mean[2])
        .sqrt()
        .min(S::ONE)
}

// ============================================================================
// BivectorField — spatially discretized bivector field
// ============================================================================

/// Boundary condition type for the spatial grid.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum BoundaryCondition {
    /// Wraps around: `F[N] = F[0]`.
    Periodic,
    /// Zero at boundaries: `F[-1] = F[N] = 0`.
    Fixed,
    /// Zero gradient at boundaries: `F[-1] = F[0]`, `F[N] = F[N-1]`.
    Free,
}

/// A spatially discretized field of Cl(1,3) bivectors.
pub struct BivectorField<S: FieldScalar> {
    /// Grid dimensions: `[Nx]` for 1D, `[Nx, Ny]` for 2D, `[Nx, Ny, Nz]` for 3D.
    pub shape: Vec<usize>,
    /// Field data: one StaBivector per grid point, row-major order.
    pub data: Vec<StaBivector<S>>,
    /// Uniform grid spacing.
    pub dx: S,
    /// Propagation speed squared (v²).
    pub v_squared: S,
    /// Optional per-point propagation speed squared for spatially varying v_A(x).
    /// When set, overrides `v_squared` at each grid point.
    /// Used for solar wind simulations where v_A varies with heliocentric distance.
    pub v_squared_per_point: Option<Vec<S>>,
    /// Boundary conditions.
    pub boundary: BoundaryCondition,
}

impl<S: FieldScalar> BivectorField<S> {
    /// Create a 1D field initialized to zero.
    pub fn new_1d(nx: usize, dx: S, v: S, bc: BoundaryCondition) -> Self {
        BivectorField {
            shape: vec![nx],
            data: vec![StaBivector::zero(); nx],
            dx,
            v_squared: v * v,
            v_squared_per_point: None,
            boundary: bc,
        }
    }

    /// Create a 2D field initialized to zero.
    pub fn new_2d(nx: usize, ny: usize, dx: S, v: S, bc: BoundaryCondition) -> Self {
        BivectorField {
            shape: vec![nx, ny],
            data: vec![StaBivector::zero(); nx * ny],
            dx,
            v_squared: v * v,
            v_squared_per_point: None,
            boundary: bc,
        }
    }

    /// Create a 3D field initialized to zero.
    pub fn new_3d(nx: usize, ny: usize, nz: usize, dx: S, v: S, bc: BoundaryCondition) -> Self {
        BivectorField {
            shape: vec![nx, ny, nz],
            data: vec![StaBivector::zero(); nx * ny * nz],
            dx,
            v_squared: v * v,
            v_squared_per_point: None,
            boundary: bc,
        }
    }

    /// Get v² at a specific grid point, using per-point value if available.
    #[inline]
    pub fn v2_at(&self, point: usize) -> S {
        self.v_squared_per_point
            .as_ref()
            .map(|v| v[point])
            .unwrap_or(self.v_squared)
    }

    /// Return the number of lattice points in the field.
    pub fn n_points(&self) -> usize {
        self.data.len()
    }

    /// Return the number of spatial dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Convert N-dim coordinates to flat index.
    pub fn index(&self, coords: &[usize]) -> usize {
        let mut idx = 0;
        let mut stride = 1;
        for d in (0..self.ndim()).rev() {
            idx += coords[d] * stride;
            stride *= self.shape[d];
        }
        idx
    }

    /// Convert flat index to N-dim coordinates.
    pub fn coords(&self, flat: usize) -> Vec<usize> {
        let mut coords = vec![0usize; self.ndim()];
        let mut rem = flat;
        for d in (0..self.ndim()).rev() {
            coords[d] = rem % self.shape[d];
            rem /= self.shape[d];
        }
        coords
    }

    /// Get neighbor index along dimension `dim` with signed `offset` (-1 or +1).
    /// Returns None if out of bounds with Fixed boundary.
    fn neighbor(&self, flat: usize, dim: usize, offset: i32) -> Option<usize> {
        let mut coords = self.coords(flat);
        let n = self.shape[dim];
        let c = coords[dim] as i32 + offset;

        match self.boundary {
            BoundaryCondition::Periodic => {
                coords[dim] = ((c % n as i32 + n as i32) % n as i32) as usize;
                Some(self.index(&coords))
            }
            BoundaryCondition::Fixed => {
                if c < 0 || c >= n as i32 {
                    None // use zero bivector
                } else {
                    coords[dim] = c as usize;
                    Some(self.index(&coords))
                }
            }
            BoundaryCondition::Free => {
                if c < 0 || c >= n as i32 {
                    Some(flat) // copy boundary value
                } else {
                    coords[dim] = c as usize;
                    Some(self.index(&coords))
                }
            }
        }
    }

    /// Get field value at neighbor, or zero for Fixed boundary out-of-bounds.
    fn get_neighbor(&self, flat: usize, dim: usize, offset: i32) -> StaBivector<S> {
        match self.neighbor(flat, dim, offset) {
            Some(idx) => self.data[idx],
            None => StaBivector::zero(),
        }
    }

    // ========================================================================
    // Spatial operators
    // ========================================================================

    /// Compute spatial gradient: dF/dx_d at each point, for each dimension d.
    /// Returns Vec of length n_points * ndim, ordered [point0_dim0, point0_dim1, ..., point1_dim0, ...].
    /// Uses central differences: (F[i+1] - F[i-1]) / (2*dx).
    /// Parallelized with rayon for large grids.
    pub fn spatial_gradient(&self) -> Vec<StaBivector<S>> {
        let np = self.n_points();
        let nd = self.ndim();
        let inv_2dx = S::HALF / self.dx;

        if nd == 1 {
            // Fast path for 1D: avoid coords/index overhead.
            self.spatial_gradient_1d(inv_2dx)
        } else if np >= par_threshold() {
            let mut grad = vec![StaBivector::zero(); np * nd];
            grad.par_chunks_mut(nd).enumerate().for_each(|(p, chunk)| {
                for (d, value) in chunk.iter_mut().enumerate() {
                    let fp = self.get_neighbor(p, d, 1);
                    let fm = self.get_neighbor(p, d, -1);
                    *value = fp.sub(&fm).scale(inv_2dx);
                }
            });
            grad
        } else {
            let mut grad = vec![StaBivector::zero(); np * nd];
            for p in 0..np {
                for d in 0..nd {
                    let fp = self.get_neighbor(p, d, 1);
                    let fm = self.get_neighbor(p, d, -1);
                    grad[p * nd + d] = fp.sub(&fm).scale(inv_2dx);
                }
            }
            grad
        }
    }

    /// Fast 1D spatial gradient — no coords/index overhead.
    /// Uses rayon when grid exceeds par_threshold().
    fn spatial_gradient_1d(&self, inv_2dx: S) -> Vec<StaBivector<S>> {
        let n = self.data.len();
        let data = &self.data;
        let bc = self.boundary;

        let compute = |p: usize| -> StaBivector<S> {
            let fp = if p + 1 < n {
                data[p + 1]
            } else {
                match bc {
                    BoundaryCondition::Periodic => data[0],
                    BoundaryCondition::Fixed => StaBivector::zero(),
                    BoundaryCondition::Free => data[n - 1],
                }
            };
            let fm = if p > 0 {
                data[p - 1]
            } else {
                match bc {
                    BoundaryCondition::Periodic => data[n - 1],
                    BoundaryCondition::Fixed => StaBivector::zero(),
                    BoundaryCondition::Free => data[0],
                }
            };
            fp.sub(&fm).scale(inv_2dx)
        };

        if n >= par_threshold() {
            (0..n).into_par_iter().map(compute).collect()
        } else {
            (0..n).map(compute).collect()
        }
    }

    /// Compute spatial Laplacian ∇²F at each point.
    /// Uses the 3-point stencil `(F[i+1] - 2*F[i] + F[i-1]) / dx²`, summed over dimensions.
    /// Parallelized with rayon for large grids.
    pub fn spatial_laplacian(&self) -> Vec<StaBivector<S>> {
        let np = self.n_points();
        let nd = self.ndim();
        let inv_dx2 = S::ONE / (self.dx * self.dx);

        if nd == 1 {
            self.spatial_laplacian_1d(inv_dx2)
        } else if np >= par_threshold() {
            let mut lap = vec![StaBivector::zero(); np];
            lap.par_iter_mut().enumerate().for_each(|(p, lp)| {
                for d in 0..nd {
                    let fp = self.get_neighbor(p, d, 1);
                    let fm = self.get_neighbor(p, d, -1);
                    let fc = self.data[p];
                    let term = fp.add(&fm).sub(&fc.scale(S::TWO)).scale(inv_dx2);
                    lp.add_assign(&term);
                }
            });
            lap
        } else {
            let mut lap = vec![StaBivector::zero(); np];
            for (p, lp) in lap.iter_mut().enumerate() {
                for d in 0..nd {
                    let fp = self.get_neighbor(p, d, 1);
                    let fm = self.get_neighbor(p, d, -1);
                    let fc = self.data[p];
                    let term = fp.add(&fm).sub(&fc.scale(S::TWO)).scale(inv_dx2);
                    lp.add_assign(&term);
                }
            }
            lap
        }
    }

    /// Fast 1D spatial Laplacian — no coords/index overhead.
    /// Uses rayon when grid exceeds par_threshold().
    fn spatial_laplacian_1d(&self, inv_dx2: S) -> Vec<StaBivector<S>> {
        let n = self.data.len();
        let data = &self.data;
        let bc = self.boundary;

        let compute = |p: usize| -> StaBivector<S> {
            let fp = if p + 1 < n {
                data[p + 1]
            } else {
                match bc {
                    BoundaryCondition::Periodic => data[0],
                    BoundaryCondition::Fixed => StaBivector::zero(),
                    BoundaryCondition::Free => data[n - 1],
                }
            };
            let fm = if p > 0 {
                data[p - 1]
            } else {
                match bc {
                    BoundaryCondition::Periodic => data[n - 1],
                    BoundaryCondition::Fixed => StaBivector::zero(),
                    BoundaryCondition::Free => data[0],
                }
            };
            let fc = data[p];
            fp.add(&fm).sub(&fc.scale(S::TWO)).scale(inv_dx2)
        };

        if n >= par_threshold() {
            (0..n).into_par_iter().map(compute).collect()
        } else {
            (0..n).map(compute).collect()
        }
    }

    /// Compute the nonlinear commutator term: Σ_d [F(p), ∇_d F(p)].
    /// The coupling coefficient is exactly 1, fixed by the geometric product.
    /// Uses rayon for grids above par_threshold().
    pub fn commutator_term(&self) -> Vec<StaBivector<S>> {
        let np = self.n_points();
        let nd = self.ndim();
        let grad = self.spatial_gradient();

        if nd == 1 && np >= par_threshold() {
            self.data
                .par_iter()
                .zip(grad.par_iter())
                .map(|(f, g)| f.commutator(g))
                .collect()
        } else if nd == 1 {
            self.data
                .iter()
                .zip(grad.iter())
                .map(|(f, g)| f.commutator(g))
                .collect()
        } else if np >= par_threshold() {
            self.data
                .par_iter()
                .enumerate()
                .map(|(p, f)| {
                    let mut c = StaBivector::zero();
                    for d in 0..nd {
                        c.add_assign(&f.commutator(&grad[p * nd + d]));
                    }
                    c
                })
                .collect()
        } else {
            self.data
                .iter()
                .enumerate()
                .map(|(p, f)| {
                    let mut c = StaBivector::zero();
                    for d in 0..nd {
                        c.add_assign(&f.commutator(&grad[p * nd + d]));
                    }
                    c
                })
                .collect()
        }
    }

    /// Total field energy: Σ_p ||F(p)||².
    pub fn total_energy(&self) -> S {
        if self.data.len() >= par_threshold() {
            self.data.par_iter().map(|bv| bv.norm_squared()).sum()
        } else {
            self.data.iter().map(|bv| bv.norm_squared()).sum()
        }
    }

    /// Chiral decomposition of the entire field.
    pub fn chiral_decompose(&self) -> ChiralDecomposition<S> {
        let pairs: Vec<([S; 3], [S; 3])> = if self.data.len() >= par_threshold() {
            self.data.par_iter().map(|bv| chiral_split(bv)).collect()
        } else {
            self.data.iter().map(|bv| chiral_split(bv)).collect()
        };
        let left: Vec<[S; 3]> = pairs.iter().map(|(l, _)| *l).collect();
        let right: Vec<[S; 3]> = pairs.iter().map(|(_, r)| *r).collect();
        let r_left = sector_order_parameter(&left);
        let r_right = sector_order_parameter(&right);
        ChiralDecomposition {
            left,
            right,
            r_left,
            r_right,
        }
    }
}

// ============================================================================
// Type aliases for f32 (default) and f64 precision
// ============================================================================

/// Type aliases for f32 (default) and f64 precision.
pub type StaBivector32 = StaBivector<f32>;
/// Double-precision spacetime bivector.
pub type StaBivector64 = StaBivector<f64>;
/// Single-precision bivector field.
pub type BivectorField32 = BivectorField<f32>;
/// Double-precision bivector field.
pub type BivectorField64 = BivectorField<f64>;
