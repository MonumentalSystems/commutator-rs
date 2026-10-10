use crate::matrix_ops::validate_square;
use crate::{Complex64, DenseMatrix, KeldyshError, Result, TwoTimeMatrix};

/// Return the one-particle density matrix `rho(t) = -i G^<(t,t)`.
pub fn density_matrix(lesser: &TwoTimeMatrix, time_index: usize) -> Result<DenseMatrix> {
    let equal_time = lesser.get(time_index, time_index)?;
    Ok(DenseMatrix::try_new(
        lesser.orbitals(),
        lesser.orbitals(),
        equal_time
            .as_slice()
            .iter()
            .map(|value| Complex64::new(0.0, -1.0) * value)
            .collect(),
    )?)
}

/// Return `Tr rho`, requiring the result to be real within an absolute tolerance.
pub fn particle_number(lesser: &TwoTimeMatrix, time_index: usize, tolerance: f64) -> Result<f64> {
    real_checked(density_matrix(lesser, time_index)?.trace()?, tolerance)
}

/// Return the expectation `Tr[O rho]` of a one-body matrix.
///
/// The caller supplies any physical prefactor contained in `O`. The result is
/// required to be real within the requested absolute tolerance.
pub fn one_body_expectation(
    lesser: &TwoTimeMatrix,
    time_index: usize,
    observable: &DenseMatrix,
    tolerance: f64,
) -> Result<f64> {
    validate_square(observable, lesser.orbitals(), "one-body observable")?;
    let value = observable
        .multiply(&density_matrix(lesser, time_index)?)?
        .trace()?;
    real_checked(value, tolerance)
}

/// Return the directed bond contribution flowing **into** orbital `a` from `b`.
///
/// For `H = sum_ab h_ab c†_a c_b`, this uses
/// `J_(b->a) = 2 q/hbar Im[h_ab rho_ba]`. Supply `charge_over_hbar=1` for
/// particle-number flow. Reversing the orientation changes the sign when `h`
/// and `rho` are Hermitian.
pub fn bond_flow_into(
    lesser: &TwoTimeMatrix,
    time_index: usize,
    a: usize,
    b: usize,
    hopping_ab: Complex64,
    charge_over_hbar: f64,
) -> Result<f64> {
    if !hopping_ab.re.is_finite() || !hopping_ab.im.is_finite() || !charge_over_hbar.is_finite() {
        return Err(KeldyshError::NonFinite("bond-current inputs"));
    }
    if a >= lesser.orbitals() {
        return Err(KeldyshError::IndexOutOfBounds {
            name: "bond orbital a",
            index: a,
            length: lesser.orbitals(),
        });
    }
    if b >= lesser.orbitals() {
        return Err(KeldyshError::IndexOutOfBounds {
            name: "bond orbital b",
            index: b,
            length: lesser.orbitals(),
        });
    }
    let rho_ba = density_matrix(lesser, time_index)?
        .get(b, a)
        .expect("validated orbital indices");
    Ok(2.0 * charge_over_hbar * (hopping_ab * rho_ba).im)
}

fn real_checked(value: Complex64, tolerance: f64) -> Result<f64> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(KeldyshError::InvalidTolerance);
    }
    if value.im.abs() > tolerance {
        Err(KeldyshError::NonRealObservable {
            imaginary: value.im,
        })
    } else {
        Ok(value.re)
    }
}
