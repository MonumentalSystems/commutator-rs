//! Basic shape generation using CGA primitives.
//!
//! Port of `vsr_shapes.h`. Generates point sets for spheres, circles,
//! cylinders, and lines using CGA rotors and translators.

use crate::cga3d::*;
use std::f32::consts::PI;

/// Generate points on a unit sphere centered at the origin.
///
/// Returns `slices * stacks` points parameterized by spherical coordinates.
pub fn sphere_points(radius: f32, slices: usize, stacks: usize) -> Vec<(f32, f32, f32)> {
    let mut pts = Vec::with_capacity(slices * stacks);
    for i in 0..slices {
        let theta = 2.0 * PI * (i as f32) / (slices as f32);
        for j in 0..stacks {
            let phi = -PI / 2.0 + PI * (j as f32) / (stacks as f32);
            let x = radius * phi.cos() * theta.cos();
            let y = radius * phi.cos() * theta.sin();
            let z = radius * phi.sin();
            pts.push((x, y, z));
        }
    }
    pts
}

/// Generate CGA points on a sphere.
pub fn sphere_cga(radius: f32, slices: usize, stacks: usize) -> Vec<Pnt> {
    sphere_points(radius, slices, stacks)
        .into_iter()
        .map(|(x, y, z)| point(x, y, z))
        .collect()
}

/// Generate points on a circle in the XY plane.
///
/// Returns `n` evenly-spaced points.
pub fn circle_points(radius: f32, n: usize) -> Vec<(f32, f32, f32)> {
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let theta = 2.0 * PI * (i as f32) / (n as f32);
        pts.push((radius * theta.cos(), radius * theta.sin(), 0.0));
    }
    pts
}

/// Generate CGA points on a circle in the XY plane.
pub fn circle_cga(radius: f32, n: usize) -> Vec<Pnt> {
    circle_points(radius, n)
        .into_iter()
        .map(|(x, y, z)| point(x, y, z))
        .collect()
}

/// Generate CGA points on a cylinder along the Y axis.
pub fn cylinder_cga(radius: f32, height: f32, slices: usize, stacks: usize) -> Vec<Pnt> {
    let mut pts = Vec::with_capacity(slices * stacks);
    for i in 0..slices {
        let theta = 2.0 * PI * (i as f32) / (slices as f32);
        let x = radius * theta.cos();
        let z = radius * theta.sin();
        for j in 0..stacks {
            let y = -height / 2.0 + height * (j as f32) / (stacks as f32);
            pts.push(point(x, y, z));
        }
    }
    pts
}

/// Create a CGA circle (grade-3 trivector) from three points.
pub fn circle_from_points(a: &Pnt, b: &Pnt, c: &Pnt) -> Cir {
    let pair = op_pnt_pnt(a, b);
    op_par_pnt(&pair, c)
}

/// Create a CGA sphere (grade-4) from four points.
pub fn sphere_from_points(a: &Pnt, b: &Pnt, c: &Pnt, d: &Pnt) -> Sph {
    let cir = circle_from_points(a, b, c);
    op_cir_pnt(&cir, d)
}

/// Create a CGA dual sphere from center and radius.
pub fn dual_sphere(cx: f32, cy: f32, cz: f32, radius: f32) -> Dls {
    Round::dls(&point(cx, cy, cz), radius)
}

/// Create a CGA point pair from two points.
pub fn point_pair(a: &Pnt, b: &Pnt) -> Par {
    op_pnt_pnt(a, b)
}

/// Line segment between two points, returned as `n` CGA points.
pub fn line_points(a: (f32, f32, f32), b: (f32, f32, f32), n: usize) -> Vec<Pnt> {
    let mut pts = Vec::with_capacity(n);
    for i in 0..n {
        let t = if n > 1 {
            (i as f32) / ((n - 1) as f32)
        } else {
            0.0
        };
        let x = a.0 + t * (b.0 - a.0);
        let y = a.1 + t * (b.1 - a.1);
        let z = a.2 + t * (b.2 - a.2);
        pts.push(point(x, y, z));
    }
    pts
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    #[test]
    fn test_sphere_point_count() {
        let pts = sphere_points(1.0, 10, 10);
        assert_eq!(pts.len(), 100);
    }

    #[test]
    fn test_sphere_points_on_surface() {
        let pts = sphere_points(2.0, 20, 20);
        for (x, y, z) in &pts {
            let r = (x * x + y * y + z * z).sqrt();
            assert!((r - 2.0).abs() < EPS, "point not on sphere: r={}", r);
        }
    }

    #[test]
    fn test_circle_point_count() {
        let pts = circle_points(1.0, 32);
        assert_eq!(pts.len(), 32);
    }

    #[test]
    fn test_circle_points_on_circle() {
        let pts = circle_points(3.0, 16);
        for (x, y, z) in &pts {
            let r = (x * x + y * y).sqrt();
            assert!((r - 3.0).abs() < EPS);
            assert!(z.abs() < EPS);
        }
    }

    #[test]
    fn test_circle_from_three_points() {
        let a = point(1.0, 0.0, 0.0);
        let b = point(0.0, 1.0, 0.0);
        let c = point(-1.0, 0.0, 0.0);
        let cir = circle_from_points(&a, &b, &c);
        assert!(cir.norm() > EPS, "Circle should be non-zero");
    }

    #[test]
    fn test_sphere_from_four_points() {
        let a = point(1.0, 0.0, 0.0);
        let b = point(0.0, 1.0, 0.0);
        let c = point(0.0, 0.0, 1.0);
        let d = point(-1.0, 0.0, 0.0);
        let sph = sphere_from_points(&a, &b, &c, &d);
        assert!(sph.norm() > EPS, "Sphere should be non-zero");
    }

    #[test]
    fn test_dual_sphere() {
        let dls = dual_sphere(0.0, 0.0, 0.0, 5.0);
        let r2 = Round::radius_squared(&dls);
        assert!((r2 - 25.0).abs() < EPS, "r2={}", r2);
    }

    #[test]
    fn test_line_points() {
        let pts = line_points((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), 11);
        assert_eq!(pts.len(), 11);
        let (x0, _, _) = Round::location(&pts[0]);
        let (x10, _, _) = Round::location(&pts[10]);
        assert!((x0 - 0.0).abs() < EPS);
        assert!((x10 - 10.0).abs() < EPS);
    }

    #[test]
    fn test_cylinder_cga() {
        let pts = cylinder_cga(1.0, 2.0, 8, 4);
        assert_eq!(pts.len(), 32);
    }
}
