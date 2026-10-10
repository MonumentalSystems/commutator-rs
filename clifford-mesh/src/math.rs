use crate::{MeshError, Result};

pub(crate) type Vec3 = [f32; 3];

pub(crate) fn ensure_finite(name: &'static str, value: Vec3) -> Result<Vec3> {
    if value.iter().all(|component| component.is_finite()) {
        Ok(value)
    } else {
        Err(MeshError::NonFinite(name))
    }
}

pub(crate) fn ensure_positive(name: &'static str, value: f32) -> Result<f32> {
    if !value.is_finite() {
        Err(MeshError::NonFinite(name))
    } else if value <= 0.0 {
        Err(MeshError::NonPositive(name))
    } else {
        Ok(value)
    }
}

pub(crate) fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub(crate) fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn scale(value: Vec3, scalar: f32) -> Vec3 {
    [value[0] * scalar, value[1] * scalar, value[2] * scalar]
}

pub(crate) fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn norm_squared(value: Vec3) -> f32 {
    dot(value, value)
}

pub(crate) fn normalize(name: &'static str, value: Vec3) -> Result<Vec3> {
    let value = ensure_finite(name, value)?;
    let maximum = value
        .iter()
        .fold(0.0_f32, |current, component| current.max(component.abs()));
    if maximum == 0.0 {
        return Err(MeshError::ZeroVector(name));
    }
    let scaled = [value[0] / maximum, value[1] / maximum, value[2] / maximum];
    let length = norm_squared(scaled).sqrt();
    let normalized = scale(scaled, length.recip());
    ensure_finite(name, normalized)
}

pub(crate) fn frame(normal: Vec3) -> Result<(Vec3, Vec3)> {
    let normal = normalize("frame normal", normal)?;
    let reference = if normal[0].abs() <= normal[1].abs() && normal[0].abs() <= normal[2].abs() {
        [1.0, 0.0, 0.0]
    } else if normal[1].abs() <= normal[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let tangent_u = normalize("derived tangent", cross(reference, normal))?;
    let tangent_v = normalize("derived tangent", cross(normal, tangent_u))?;
    Ok((tangent_u, tangent_v))
}

pub(crate) fn checked_u32(value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| MeshError::SizeOverflow)
}
