use core::f32::consts::PI;

use crate::math::{add, checked_u32, ensure_positive, frame, scale};
use crate::{CylinderOptions, Line3, LineSegments, MeshError, Result, TriangleMesh, Vertex};

/// Generates a finite segment centered on a line point.
pub fn line_segment(line: &Line3, half_length: f32) -> Result<LineSegments> {
    let half_length = ensure_positive("line half-length", half_length)?;
    LineSegments::try_from_parts(
        vec![
            add(line.point(), scale(line.direction(), -half_length)),
            add(line.point(), scale(line.direction(), half_length)),
        ],
        vec![[0, 1]],
    )
}

/// Generates an arrow from the line point in its oriented direction.
///
/// `head_half_width` is the distance from the head centerline to each of its
/// four transverse tips.
pub fn arrow(
    line: &Line3,
    length: f32,
    head_length: f32,
    head_half_width: f32,
) -> Result<LineSegments> {
    let length = ensure_positive("arrow length", length)?;
    let head_length = ensure_positive("arrow head length", head_length)?;
    let head_half_width = ensure_positive("arrow head half-width", head_half_width)?;
    if head_length >= length {
        return Err(MeshError::InvalidRelation(
            "arrow head length must be smaller than arrow length",
        ));
    }
    let start = line.point();
    let end = add(start, scale(line.direction(), length));
    let head_center = add(end, scale(line.direction(), -head_length));
    let (axis_u, axis_v) = frame(line.direction())?;
    let positions = vec![
        start,
        end,
        add(head_center, scale(axis_u, head_half_width)),
        add(head_center, scale(axis_u, -head_half_width)),
        add(head_center, scale(axis_v, head_half_width)),
        add(head_center, scale(axis_v, -head_half_width)),
    ];
    LineSegments::try_from_parts(positions, vec![[0, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
}

/// Tessellates an uncapped cylinder centered on a line point.
pub fn open_cylinder(
    line: &Line3,
    half_length: f32,
    radius: f32,
    options: CylinderOptions,
) -> Result<TriangleMesh> {
    let half_length = ensure_positive("cylinder half-length", half_length)?;
    let radius = ensure_positive("cylinder radius", radius)?;
    let count = options.radial_segments() as usize;
    let vertex_count = count.checked_mul(2).ok_or(MeshError::SizeOverflow)?;
    let triangle_count = count.checked_mul(2).ok_or(MeshError::SizeOverflow)?;
    checked_u32(vertex_count)?;
    let (axis_u, axis_v) = frame(line.direction())?;
    let start_center = add(line.point(), scale(line.direction(), -half_length));
    let end_center = add(line.point(), scale(line.direction(), half_length));
    let mut vertices = Vec::with_capacity(vertex_count);
    for side in [start_center, end_center] {
        for index in 0..count {
            let angle = 2.0 * PI * index as f32 / count as f32;
            let normal = add(scale(axis_u, angle.cos()), scale(axis_v, angle.sin()));
            vertices.push(Vertex::generated(add(side, scale(normal, radius)), normal));
        }
    }
    let mut triangles = Vec::with_capacity(triangle_count);
    for index in 0..count {
        let next = (index + 1) % count;
        let a = checked_u32(index)?;
        let b = checked_u32(next)?;
        let c = checked_u32(count + index)?;
        let d = checked_u32(count + next)?;
        push_oriented(&vertices, &mut triangles, [a, c, b])?;
        push_oriented(&vertices, &mut triangles, [b, c, d])?;
    }
    TriangleMesh::generated(vertices, triangles)
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
