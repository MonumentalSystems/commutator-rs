use crate::math::{add, checked_u32, ensure_positive, frame, scale};
use crate::{LineSegments, MeshError, Plane3, PlanePatchOptions, Result, TriangleMesh, Vertex};

/// Tessellates a square plane patch centered on the descriptor point.
pub fn plane_patch(plane: &Plane3, size: f32, options: PlanePatchOptions) -> Result<TriangleMesh> {
    let size = ensure_positive("plane size", size)?;
    let u_count = options.u_segments() as usize;
    let v_count = options.v_segments() as usize;
    let u_vertices = u_count.checked_add(1).ok_or(MeshError::SizeOverflow)?;
    let v_vertices = v_count.checked_add(1).ok_or(MeshError::SizeOverflow)?;
    let vertex_count = u_vertices
        .checked_mul(v_vertices)
        .ok_or(MeshError::SizeOverflow)?;
    let triangle_count = u_count
        .checked_mul(v_count)
        .and_then(|count| count.checked_mul(2))
        .ok_or(MeshError::SizeOverflow)?;
    checked_u32(vertex_count)?;
    let (axis_u, axis_v) = frame(plane.normal())?;
    let mut vertices = Vec::with_capacity(vertex_count);
    for v_index in 0..=v_count {
        let v_offset = (v_index as f32 / v_count as f32 - 0.5) * size;
        for u_index in 0..=u_count {
            let u_offset = (u_index as f32 / u_count as f32 - 0.5) * size;
            let position = add(
                plane.point(),
                add(scale(axis_u, u_offset), scale(axis_v, v_offset)),
            );
            vertices.push(Vertex::generated(position, plane.normal()));
        }
    }
    let index = |u_index: usize, v_index: usize| -> Result<u32> {
        checked_u32(
            v_index
                .checked_mul(u_vertices)
                .and_then(|value| value.checked_add(u_index))
                .ok_or(MeshError::SizeOverflow)?,
        )
    };
    let mut triangles = Vec::with_capacity(triangle_count);
    for v_index in 0..v_count {
        for u_index in 0..u_count {
            let a = index(u_index, v_index)?;
            let b = index(u_index + 1, v_index)?;
            let c = index(u_index, v_index + 1)?;
            let d = index(u_index + 1, v_index + 1)?;
            triangles.push(TriangleMesh::orient_triangle(
                &vertices,
                [a, b, c],
                plane.normal(),
            )?);
            triangles.push(TriangleMesh::orient_triangle(
                &vertices,
                [b, d, c],
                plane.normal(),
            )?);
        }
    }
    TriangleMesh::generated(vertices, triangles)
}

/// Generates a square plane grid as independent segments without connectors.
pub fn plane_grid(plane: &Plane3, size: f32, options: PlanePatchOptions) -> Result<LineSegments> {
    let size = ensure_positive("plane size", size)?;
    let u_count = options.u_segments() as usize;
    let v_count = options.v_segments() as usize;
    let line_count = u_count
        .checked_add(1)
        .and_then(|count| count.checked_add(v_count + 1))
        .ok_or(MeshError::SizeOverflow)?;
    let vertex_count = line_count.checked_mul(2).ok_or(MeshError::SizeOverflow)?;
    checked_u32(vertex_count)?;
    let (axis_u, axis_v) = frame(plane.normal())?;
    let half = size * 0.5;
    let mut positions = Vec::with_capacity(vertex_count);
    let mut segments = Vec::with_capacity(line_count);

    for index in 0..=u_count {
        let offset = (index as f32 / u_count as f32 - 0.5) * size;
        let center = add(plane.point(), scale(axis_u, offset));
        add_segment(
            &mut positions,
            &mut segments,
            add(center, scale(axis_v, -half)),
            add(center, scale(axis_v, half)),
        )?;
    }
    for index in 0..=v_count {
        let offset = (index as f32 / v_count as f32 - 0.5) * size;
        let center = add(plane.point(), scale(axis_v, offset));
        add_segment(
            &mut positions,
            &mut segments,
            add(center, scale(axis_u, -half)),
            add(center, scale(axis_u, half)),
        )?;
    }
    LineSegments::try_from_parts(positions, segments)
}

/// Generates one segment from the plane point along its unit normal.
pub fn plane_normal_indicator(plane: &Plane3, length: f32) -> Result<LineSegments> {
    let length = ensure_positive("normal indicator length", length)?;
    LineSegments::try_from_parts(
        vec![
            plane.point(),
            add(plane.point(), scale(plane.normal(), length)),
        ],
        vec![[0, 1]],
    )
}

fn add_segment(
    positions: &mut Vec<[f32; 3]>,
    segments: &mut Vec<[u32; 2]>,
    start: [f32; 3],
    end: [f32; 3],
) -> Result<()> {
    let start_index = checked_u32(positions.len())?;
    positions.extend_from_slice(&[start, end]);
    segments.push([start_index, start_index + 1]);
    Ok(())
}
