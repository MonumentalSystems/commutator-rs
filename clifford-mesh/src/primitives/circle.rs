use core::f32::consts::PI;

use crate::math::{add, checked_u32, frame, scale, Vec3};
use crate::{
    Circle3, CircleOptions, LineSegments, MeshError, Result, TriangleMesh, TubeOptions, Vertex,
};

/// Tessellates a circle as explicit independent edges with no implicit strip.
pub fn circle_loop(circle: &Circle3, options: CircleOptions) -> Result<LineSegments> {
    let count = options.segments() as usize;
    let (tangent_u, tangent_v) = frame(circle.normal())?;
    let positions = (0..count)
        .map(|index| {
            let angle = 2.0 * PI * index as f32 / count as f32;
            circle_position(circle, tangent_u, tangent_v, angle)
        })
        .collect::<Vec<_>>();
    let mut segments = Vec::with_capacity(count);
    for index in 0..count {
        segments.push([checked_u32(index)?, checked_u32((index + 1) % count)?]);
    }
    LineSegments::try_from_parts(positions, segments)
}

/// Tessellates a filled circle as a triangle fan.
pub fn circle_disc(circle: &Circle3, options: CircleOptions) -> Result<TriangleMesh> {
    let count = options.segments() as usize;
    let (tangent_u, tangent_v) = frame(circle.normal())?;
    let mut vertices = Vec::with_capacity(count.checked_add(1).ok_or(MeshError::SizeOverflow)?);
    vertices.push(Vertex::generated(circle.center(), circle.normal()));
    for index in 0..count {
        let angle = 2.0 * PI * index as f32 / count as f32;
        vertices.push(Vertex::generated(
            circle_position(circle, tangent_u, tangent_v, angle),
            circle.normal(),
        ));
    }
    let mut triangles = Vec::with_capacity(count);
    for index in 0..count {
        let triangle = [
            0,
            checked_u32(index + 1)?,
            checked_u32(((index + 1) % count) + 1)?,
        ];
        triangles.push(TriangleMesh::orient_triangle(
            &vertices,
            triangle,
            circle.normal(),
        )?);
    }
    TriangleMesh::generated(vertices, triangles)
}

/// Tessellates a circle as a regular closed tube with explicit major/minor resolution.
///
/// `tube_radius` must be smaller than the circle's major radius so the result
/// has neither a cusp nor a self-intersection.
pub fn circle_tube(
    circle: &Circle3,
    tube_radius: f32,
    options: TubeOptions,
) -> Result<TriangleMesh> {
    let tube_radius = crate::math::ensure_positive("tube radius", tube_radius)?;
    if tube_radius >= circle.radius() {
        return Err(MeshError::InvalidRelation(
            "tube radius must be smaller than circle radius",
        ));
    }
    let ring_count = options.ring_segments() as usize;
    let cross_count = options.cross_segments() as usize;
    let vertex_count = ring_count
        .checked_mul(cross_count)
        .ok_or(MeshError::SizeOverflow)?;
    let triangle_count = vertex_count.checked_mul(2).ok_or(MeshError::SizeOverflow)?;
    checked_u32(vertex_count)?;
    let (tangent_u, tangent_v) = frame(circle.normal())?;
    let mut vertices = Vec::with_capacity(vertex_count);
    for ring in 0..ring_count {
        let theta = 2.0 * PI * ring as f32 / ring_count as f32;
        let radial = add(scale(tangent_u, theta.cos()), scale(tangent_v, theta.sin()));
        let ring_center = add(circle.center(), scale(radial, circle.radius()));
        for cross_index in 0..cross_count {
            let phi = 2.0 * PI * cross_index as f32 / cross_count as f32;
            let normal = add(scale(radial, phi.cos()), scale(circle.normal(), phi.sin()));
            vertices.push(Vertex::generated(
                add(ring_center, scale(normal, tube_radius)),
                normal,
            ));
        }
    }
    let index = |ring: usize, cross_index: usize| -> Result<u32> {
        checked_u32(
            (ring % ring_count)
                .checked_mul(cross_count)
                .and_then(|value| value.checked_add(cross_index % cross_count))
                .ok_or(MeshError::SizeOverflow)?,
        )
    };
    let mut triangles = Vec::with_capacity(triangle_count);
    for ring in 0..ring_count {
        for cross_index in 0..cross_count {
            let a = index(ring, cross_index)?;
            let b = index(ring + 1, cross_index)?;
            let c = index(ring, cross_index + 1)?;
            let d = index(ring + 1, cross_index + 1)?;
            push_oriented(&vertices, &mut triangles, [a, b, c])?;
            push_oriented(&vertices, &mut triangles, [c, b, d])?;
        }
    }
    TriangleMesh::generated(vertices, triangles)
}

fn circle_position(circle: &Circle3, tangent_u: Vec3, tangent_v: Vec3, angle: f32) -> Vec3 {
    add(
        circle.center(),
        add(
            scale(tangent_u, circle.radius() * angle.cos()),
            scale(tangent_v, circle.radius() * angle.sin()),
        ),
    )
}

fn push_oriented(
    vertices: &[Vertex],
    triangles: &mut Vec<[u32; 3]>,
    triangle: [u32; 3],
) -> Result<()> {
    let desired = triangle.iter().try_fold([0.0; 3], |sum, index| {
        let normal = vertices
            .get(*index as usize)
            .ok_or(MeshError::IndexOutOfRange)?
            .normal();
        Ok::<_, MeshError>(add(sum, normal))
    })?;
    triangles.push(TriangleMesh::orient_triangle(vertices, triangle, desired)?);
    Ok(())
}
