//! Portable reference steppers for the bivector synchronization equation.

use crate::{BivectorField, FieldScalar};

/// Advance one explicit Euler step of
/// `dF/dt = v² laplacian(F) + [F, gradient(F)]`.
///
/// This is the precision-generic reference path. Specialized hosts may use
/// fused kernels as long as they validate their trajectory against this path.
pub fn step_euler_reference<S: FieldScalar>(field: &mut BivectorField<S>, dt: S) {
    let v2 = v2_snapshot(field);
    let laplacian = field.spatial_laplacian();
    let commutator = field.commutator_term();

    for (((value, &v2), laplacian), commutator) in field
        .data
        .iter_mut()
        .zip(&v2)
        .zip(&laplacian)
        .zip(&commutator)
    {
        value.scale_add_assign(dt * v2, laplacian);
        value.scale_add_assign(dt, commutator);
    }
}

/// Advance one three-pass Strang step.
///
/// The nonlinear term is evaluated after the first half linear step and the
/// final Laplacian is evaluated from that updated state.
pub fn step_strang_reference<S: FieldScalar>(field: &mut BivectorField<S>, dt: S) {
    let v2 = v2_snapshot(field);
    let half_dt = S::HALF * dt;

    let laplacian = field.spatial_laplacian();
    for ((value, &v2), laplacian) in field.data.iter_mut().zip(&v2).zip(&laplacian) {
        value.scale_add_assign(half_dt * v2, laplacian);
    }

    let commutator = field.commutator_term();
    for (value, commutator) in field.data.iter_mut().zip(&commutator) {
        value.scale_add_assign(dt, commutator);
    }

    let final_laplacian = field.spatial_laplacian();
    for ((value, &v2), laplacian) in field.data.iter_mut().zip(&v2).zip(&final_laplacian) {
        value.scale_add_assign(half_dt * v2, laplacian);
    }
}

fn v2_snapshot<S: FieldScalar>(field: &BivectorField<S>) -> Vec<S> {
    match &field.v_squared_per_point {
        Some(per_point) => per_point.clone(),
        None => vec![field.v_squared; field.n_points()],
    }
}
