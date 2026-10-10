use crate::math::{cross, dot, ensure_finite, sub, Vec3};
use crate::{MeshError, Result};

/// Vertex containing a three-dimensional position and surface normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    position: Vec3,
    normal: Vec3,
}

impl Vertex {
    /// Constructs a finite vertex.
    pub fn new(position: Vec3, normal: Vec3) -> Result<Self> {
        let normal = ensure_finite("vertex normal", normal)?;
        if normal.iter().all(|component| *component == 0.0) {
            return Err(MeshError::ZeroVector("vertex normal"));
        }
        Ok(Self {
            position: ensure_finite("vertex position", position)?,
            normal,
        })
    }

    pub(crate) const fn generated(position: Vec3, normal: Vec3) -> Self {
        Self { position, normal }
    }

    /// Vertex position.
    pub const fn position(self) -> Vec3 {
        self.position
    }

    /// Vertex normal.
    pub const fn normal(self) -> Vec3 {
        self.normal
    }

    /// Whether every position and normal component is finite.
    pub fn is_finite(self) -> bool {
        self.position
            .iter()
            .chain(self.normal.iter())
            .all(|component| component.is_finite())
    }
}

/// Indexed triangle mesh with finite vertices and in-range faces.
#[derive(Debug, Clone, PartialEq)]
pub struct TriangleMesh {
    vertices: Vec<Vertex>,
    triangles: Vec<[u32; 3]>,
}

impl TriangleMesh {
    /// Validates and constructs an indexed triangle mesh.
    pub fn try_from_parts(vertices: Vec<Vertex>, triangles: Vec<[u32; 3]>) -> Result<Self> {
        if vertices.len() > u32::MAX as usize {
            return Err(MeshError::SizeOverflow);
        }
        if !vertices.iter().all(|vertex| vertex.is_finite()) {
            return Err(MeshError::NonFinite("mesh vertex"));
        }
        let vertex_count = vertices.len();
        for triangle in &triangles {
            let [a, b, c] = triangle.map(|index| index as usize);
            if a >= vertex_count || b >= vertex_count || c >= vertex_count {
                return Err(MeshError::IndexOutOfRange);
            }
            let pa = vertices[a].position;
            let pb = vertices[b].position;
            let pc = vertices[c].position;
            let geometric = cross(sub(pb, pa), sub(pc, pa));
            if geometric.iter().any(|component| !component.is_finite()) {
                return Err(MeshError::NonFinite("triangle geometry"));
            }
            if geometric.iter().all(|component| *component == 0.0) {
                return Err(MeshError::DegenerateTriangle);
            }
        }
        Ok(Self {
            vertices,
            triangles,
        })
    }

    /// Vertices in stable index order.
    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    /// Indexed triangle faces.
    ///
    /// Crate-provided tessellators use counter-clockwise winding when viewed
    /// from their outward normal. Meshes built with [`Self::try_from_parts`]
    /// retain the caller's winding.
    pub fn triangles(&self) -> &[[u32; 3]] {
        &self.triangles
    }

    /// Number of indexed triangle faces.
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// Consumes the mesh into vertex and triangle buffers.
    pub fn into_parts(self) -> (Vec<Vertex>, Vec<[u32; 3]>) {
        (self.vertices, self.triangles)
    }

    /// Returns `[px, py, pz, nx, ny, nz, ...]` for renderer adapters.
    pub fn to_interleaved_position_normal_f32(&self) -> Vec<f32> {
        let mut output = Vec::with_capacity(self.vertices.len().saturating_mul(6));
        for vertex in &self.vertices {
            output.extend_from_slice(&vertex.position);
            output.extend_from_slice(&vertex.normal);
        }
        output
    }

    /// Returns flat triangle indices `[a, b, c, ...]`.
    pub fn to_flat_indices(&self) -> Vec<u32> {
        self.triangles.iter().flatten().copied().collect()
    }

    pub(crate) fn generated(vertices: Vec<Vertex>, triangles: Vec<[u32; 3]>) -> Result<Self> {
        Self::try_from_parts(vertices, triangles)
    }

    pub(crate) fn orient_triangle(
        vertices: &[Vertex],
        triangle: [u32; 3],
        desired_normal: Vec3,
    ) -> Result<[u32; 3]> {
        let [a, b, c] = triangle.map(|index| index as usize);
        let pa = vertices.get(a).ok_or(MeshError::IndexOutOfRange)?.position;
        let pb = vertices.get(b).ok_or(MeshError::IndexOutOfRange)?.position;
        let pc = vertices.get(c).ok_or(MeshError::IndexOutOfRange)?.position;
        let geometric =
            crate::math::normalize("triangle geometry", cross(sub(pb, pa), sub(pc, pa)))?;
        let desired_normal = crate::math::normalize("triangle desired normal", desired_normal)?;
        Ok(if dot(geometric, desired_normal) >= 0.0 {
            triangle
        } else {
            [triangle[0], triangle[2], triangle[1]]
        })
    }
}

/// Indexed independent line segments.
#[derive(Debug, Clone, PartialEq)]
pub struct LineSegments {
    positions: Vec<Vec3>,
    segments: Vec<[u32; 2]>,
}

impl LineSegments {
    /// Validates and constructs independent line segments.
    pub fn try_from_parts(positions: Vec<Vec3>, segments: Vec<[u32; 2]>) -> Result<Self> {
        if positions.len() > u32::MAX as usize {
            return Err(MeshError::SizeOverflow);
        }
        if positions
            .iter()
            .any(|position| position.iter().any(|component| !component.is_finite()))
        {
            return Err(MeshError::NonFinite("line position"));
        }
        let count = positions.len();
        for [start, end] in &segments {
            let start = *start as usize;
            let end = *end as usize;
            if start >= count || end >= count {
                return Err(MeshError::IndexOutOfRange);
            }
            let delta = sub(positions[end], positions[start]);
            if delta.iter().any(|component| !component.is_finite()) {
                return Err(MeshError::NonFinite("line segment geometry"));
            }
            if delta.iter().all(|component| *component == 0.0) {
                return Err(MeshError::DegenerateSegment);
            }
        }
        Ok(Self {
            positions,
            segments,
        })
    }

    /// Positions in stable index order.
    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    /// Independent indexed edges.
    pub fn segments(&self) -> &[[u32; 2]] {
        &self.segments
    }

    /// Number of independent edges.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Consumes the topology into position and segment buffers.
    pub fn into_parts(self) -> (Vec<Vec3>, Vec<[u32; 2]>) {
        (self.positions, self.segments)
    }

    /// Returns flat position data `[x, y, z, ...]`.
    pub fn to_position_f32(&self) -> Vec<f32> {
        self.positions.iter().flatten().copied().collect()
    }

    /// Returns flat edge indices `[a, b, ...]`.
    pub fn to_flat_indices(&self) -> Vec<u32> {
        self.segments.iter().flatten().copied().collect()
    }
}

/// Unconnected finite three-dimensional points.
#[derive(Debug, Clone, PartialEq)]
pub struct PointCloud {
    positions: Vec<Vec3>,
}

impl PointCloud {
    /// Validates and constructs a point cloud.
    pub fn try_from_positions(positions: Vec<Vec3>) -> Result<Self> {
        if positions
            .iter()
            .any(|position| position.iter().any(|component| !component.is_finite()))
        {
            return Err(MeshError::NonFinite("point-cloud position"));
        }
        Ok(Self { positions })
    }

    /// Point positions.
    pub fn positions(&self) -> &[Vec3] {
        &self.positions
    }

    /// Consumes the cloud into its position buffer.
    pub fn into_positions(self) -> Vec<Vec3> {
        self.positions
    }

    /// Returns flat position data `[x, y, z, ...]`.
    pub fn to_position_f32(&self) -> Vec<f32> {
        self.positions.iter().flatten().copied().collect()
    }
}
