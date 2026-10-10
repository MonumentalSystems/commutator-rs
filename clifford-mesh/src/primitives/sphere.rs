use std::collections::HashMap;

use core::f32::consts::PI;

use crate::math::{add, checked_u32, normalize, scale, Vec3};
use crate::{IcosphereOptions, MeshError, Result, Sphere3, TriangleMesh, UvSphereOptions, Vertex};

/// Tessellates a sphere with unique poles and no zero-area cap faces.
pub fn uv_sphere(sphere: &Sphere3, options: UvSphereOptions) -> Result<TriangleMesh> {
    let stacks = options.stacks() as usize;
    let slices = options.slices() as usize;
    let ring_count = stacks.checked_sub(1).ok_or(MeshError::SizeOverflow)?;
    let vertex_count = ring_count
        .checked_mul(slices)
        .and_then(|count| count.checked_add(2))
        .ok_or(MeshError::SizeOverflow)?;
    let triangle_count = slices
        .checked_mul(stacks.checked_sub(1).ok_or(MeshError::SizeOverflow)?)
        .and_then(|count| count.checked_mul(2))
        .ok_or(MeshError::SizeOverflow)?;
    checked_u32(vertex_count)?;

    let center = sphere.center();
    let radius = sphere.radius();
    let mut vertices = Vec::with_capacity(vertex_count);
    vertices.push(Vertex::generated(
        add(center, [0.0, radius, 0.0]),
        [0.0, 1.0, 0.0],
    ));
    for stack in 1..stacks {
        let phi = PI * stack as f32 / stacks as f32;
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        for slice in 0..slices {
            let theta = 2.0 * PI * slice as f32 / slices as f32;
            let normal = [sin_phi * theta.cos(), cos_phi, sin_phi * theta.sin()];
            vertices.push(Vertex::generated(
                add(center, scale(normal, radius)),
                normal,
            ));
        }
    }
    let bottom = checked_u32(vertices.len())?;
    vertices.push(Vertex::generated(
        add(center, [0.0, -radius, 0.0]),
        [0.0, -1.0, 0.0],
    ));

    let ring_index = |ring: usize, slice: usize| -> Result<u32> {
        let index = 1_usize
            .checked_add(ring.checked_mul(slices).ok_or(MeshError::SizeOverflow)?)
            .and_then(|value| value.checked_add(slice % slices))
            .ok_or(MeshError::SizeOverflow)?;
        checked_u32(index)
    };
    let mut triangles = Vec::with_capacity(triangle_count);
    for slice in 0..slices {
        push_oriented(
            &vertices,
            &mut triangles,
            [0, ring_index(0, slice)?, ring_index(0, slice + 1)?],
        )?;
    }
    for ring in 0..ring_count.saturating_sub(1) {
        for slice in 0..slices {
            let a = ring_index(ring, slice)?;
            let b = ring_index(ring + 1, slice)?;
            let c = ring_index(ring, slice + 1)?;
            let d = ring_index(ring + 1, slice + 1)?;
            push_oriented(&vertices, &mut triangles, [a, b, c])?;
            push_oriented(&vertices, &mut triangles, [c, b, d])?;
        }
    }
    let last_ring = ring_count - 1;
    for slice in 0..slices {
        push_oriented(
            &vertices,
            &mut triangles,
            [
                bottom,
                ring_index(last_ring, slice + 1)?,
                ring_index(last_ring, slice)?,
            ],
        )?;
    }
    TriangleMesh::generated(vertices, triangles)
}

/// Tessellates a sphere by recursively subdividing an icosahedron.
pub fn icosphere(sphere: &Sphere3, options: IcosphereOptions) -> Result<TriangleMesh> {
    let t = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let inverse_length = (1.0 + t * t).sqrt().recip();
    let mut positions = vec![
        [-inverse_length, t * inverse_length, 0.0],
        [inverse_length, t * inverse_length, 0.0],
        [-inverse_length, -t * inverse_length, 0.0],
        [inverse_length, -t * inverse_length, 0.0],
        [0.0, -inverse_length, t * inverse_length],
        [0.0, inverse_length, t * inverse_length],
        [0.0, -inverse_length, -t * inverse_length],
        [0.0, inverse_length, -t * inverse_length],
        [t * inverse_length, 0.0, -inverse_length],
        [t * inverse_length, 0.0, inverse_length],
        [-t * inverse_length, 0.0, -inverse_length],
        [-t * inverse_length, 0.0, inverse_length],
    ];
    let mut faces = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    for _ in 0..options.subdivisions() {
        let new_face_capacity = faces.len().checked_mul(4).ok_or(MeshError::SizeOverflow)?;
        let mut next_faces = Vec::with_capacity(new_face_capacity);
        let mut midpoint_cache = HashMap::new();
        for [a, b, c] in faces {
            let ab = midpoint(a, b, &mut positions, &mut midpoint_cache)?;
            let bc = midpoint(b, c, &mut positions, &mut midpoint_cache)?;
            let ca = midpoint(c, a, &mut positions, &mut midpoint_cache)?;
            next_faces.extend_from_slice(&[[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        faces = next_faces;
    }
    checked_u32(positions.len())?;
    let vertices = positions
        .iter()
        .map(|normal| {
            Vertex::generated(
                add(sphere.center(), scale(*normal, sphere.radius())),
                *normal,
            )
        })
        .collect::<Vec<_>>();
    let mut triangles = Vec::with_capacity(faces.len());
    for face in faces {
        push_oriented(&vertices, &mut triangles, face)?;
    }
    TriangleMesh::generated(vertices, triangles)
}

fn midpoint(
    a: u32,
    b: u32,
    positions: &mut Vec<Vec3>,
    cache: &mut HashMap<(u32, u32), u32>,
) -> Result<u32> {
    let key = (a.min(b), a.max(b));
    if let Some(index) = cache.get(&key) {
        return Ok(*index);
    }
    let position_a = *positions
        .get(a as usize)
        .ok_or(MeshError::IndexOutOfRange)?;
    let position_b = *positions
        .get(b as usize)
        .ok_or(MeshError::IndexOutOfRange)?;
    let position = normalize(
        "icosphere midpoint",
        scale(add(position_a, position_b), 0.5),
    )?;
    let index = checked_u32(positions.len())?;
    positions.push(position);
    cache.insert(key, index);
    Ok(index)
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
