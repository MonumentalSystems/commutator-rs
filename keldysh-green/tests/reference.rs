use keldysh_green::{
    bond_flow_into, causal_volterra_convolution, density_matrix, langreth_product, particle_number,
    retarded_dyson, time_convolution, Complex64, DenseMatrix, KeldyshError, KeldyshGreen,
    RealTimeGrid, TwoTimeMatrix,
};

fn scalar(value: Complex64) -> DenseMatrix {
    DenseMatrix::from_scalar(value).expect("finite scalar")
}

fn scalar_function<F>(points: usize, mut function: F) -> TwoTimeMatrix
where
    F: FnMut(usize, usize) -> Complex64,
{
    TwoTimeMatrix::try_from_fn(points, 1, |i, j| Ok(scalar(function(i, j))))
        .expect("valid function")
}

#[cfg(target_pointer_width = "64")]
#[test]
fn enormous_two_time_shape_returns_error_without_panicking() {
    let result = std::panic::catch_unwind(|| {
        TwoTimeMatrix::try_from_fn(1usize << 31, 1, |_, _| Ok(scalar(0.0.into())))
    });
    assert!(
        result.is_ok(),
        "checked two-time construction must not panic"
    );
    assert_eq!(result.unwrap(), Err(KeldyshError::SizeOverflow));
}

fn equilibrium_level(grid: RealTimeGrid, energy: f64, occupation: f64) -> KeldyshGreen {
    let points = grid.len();
    let phase = |i: usize, j: usize| {
        let delta = grid.times()[i] - grid.times()[j];
        Complex64::new((-energy * delta).cos(), (-energy * delta).sin())
    };
    let retarded = scalar_function(points, |i, j| {
        if i > j {
            Complex64::new(0.0, -1.0) * phase(i, j)
        } else if i == j {
            Complex64::new(0.0, -0.5)
        } else {
            0.0.into()
        }
    });
    let advanced = scalar_function(points, |i, j| {
        if i < j {
            Complex64::new(0.0, 1.0) * phase(i, j)
        } else if i == j {
            Complex64::new(0.0, 0.5)
        } else {
            0.0.into()
        }
    });
    let lesser = scalar_function(points, |i, j| Complex64::new(0.0, occupation) * phase(i, j));
    let greater = scalar_function(points, |i, j| {
        Complex64::new(0.0, -(1.0 - occupation)) * phase(i, j)
    });
    KeldyshGreen::try_new(grid, retarded, advanced, lesser, greater).expect("compatible")
}

#[test]
fn noninteracting_equilibrium_level_obeys_all_identities() {
    let green = equilibrium_level(
        RealTimeGrid::try_new(vec![0.0, 0.2, 0.7, 1.0]).unwrap(),
        1.3,
        0.37,
    );
    let report = green.consistency_report().unwrap();
    assert!(report.is_consistent(1.0e-12).unwrap(), "{report:?}");
    assert!((particle_number(green.lesser(), 2, 1.0e-12).unwrap() - 0.37).abs() < 1.0e-12);
}

#[test]
fn grid_weights_integrate_constants_on_nonuniform_grid() {
    let grid = RealTimeGrid::try_new(vec![0.0, 0.1, 0.4, 1.0]).unwrap();
    assert!((grid.weights().iter().sum::<f64>() - 1.0).abs() < 1.0e-15);
    let ones = scalar_function(grid.len(), |_, _| 1.0.into());
    let full = time_convolution(&grid, &ones, &ones).unwrap();
    assert!((full.get(1, 2).unwrap().get(0, 0).unwrap().re - 1.0).abs() < 1.0e-15);

    let causal = causal_volterra_convolution(&grid, &ones, &ones).unwrap();
    assert!((causal.get(3, 1).unwrap().get(0, 0).unwrap().re - 0.9).abs() < 1.0e-15);
    assert_eq!(causal.get(1, 3).unwrap().get(0, 0).unwrap(), 0.0.into());
}

#[test]
fn zero_self_energy_dyson_returns_free_propagator() {
    let grid = RealTimeGrid::try_new(vec![0.0, 0.25, 0.5, 0.75]).unwrap();
    let free = equilibrium_level(grid.clone(), 0.8, 0.4).retarded().clone();
    let zeros = scalar_function(grid.len(), |_, _| 0.0.into());
    assert_eq!(retarded_dyson(&grid, &free, &zeros).unwrap(), free);
}

#[test]
fn dyson_rejects_acausal_retarded_input() {
    let grid = RealTimeGrid::try_new(vec![0.0, 0.5, 1.0]).unwrap();
    let free = scalar_function(grid.len(), |i, j| {
        if i >= j || (i == 0 && j == 1) {
            1.0.into()
        } else {
            0.0.into()
        }
    });
    let zeros = scalar_function(grid.len(), |_, _| 0.0.into());
    assert!(matches!(
        retarded_dyson(&grid, &free, &zeros),
        Err(KeldyshError::AcausalRetardedInput { .. })
    ));
}

#[test]
fn grid_rejects_nonfinite_derived_weights() {
    assert_eq!(
        RealTimeGrid::try_new(vec![-f64::MAX, f64::MAX]),
        Err(KeldyshError::NonFinite("real-time quadrature weights"))
    );
}

#[test]
fn volterra_dyson_matches_constant_kernel_analytic_limit() {
    let times: Vec<_> = (0..41).map(|index| index as f64 / 40.0).collect();
    let grid = RealTimeGrid::try_new(times).unwrap();
    let free = scalar_function(
        grid.len(),
        |i, j| {
            if i >= j {
                1.0.into()
            } else {
                0.0.into()
            }
        },
    );
    let lambda = 0.36_f64;
    let self_energy = scalar_function(
        grid.len(),
        |i, j| {
            if i >= j {
                lambda.into()
            } else {
                0.0.into()
            }
        },
    );
    let solved = retarded_dyson(&grid, &free, &self_energy).unwrap();
    let numerical = solved.get(40, 0).unwrap().get(0, 0).unwrap().re;
    let expected = lambda.sqrt().cosh();
    assert!(
        (numerical - expected).abs() < 3.0e-3,
        "{numerical} != {expected}"
    );
}

#[test]
fn langreth_lesser_rule_has_both_terms() {
    let grid = RealTimeGrid::try_new(vec![0.0, 0.5, 1.0]).unwrap();
    let points = grid.len();
    let component = |value: f64| scalar_function(points, |_, _| value.into());
    let left = KeldyshGreen::try_new(
        grid.clone(),
        component(1.0),
        component(0.0),
        component(2.0),
        component(3.0),
    )
    .unwrap();
    let right = KeldyshGreen::try_new(
        grid,
        component(4.0),
        component(5.0),
        component(6.0),
        component(7.0),
    )
    .unwrap();
    let product = langreth_product(&left, &right).unwrap();
    // Integral length one: 1*6 + 2*5.
    assert!((product.lesser().get(0, 2).unwrap().get(0, 0).unwrap().re - 16.0).abs() < 1.0e-12);
}

#[test]
fn equal_time_density_and_oriented_bond_flow_follow_documented_signs() {
    let lesser = TwoTimeMatrix::try_from_fn(2, 2, |i, j| {
        if i == j {
            DenseMatrix::try_new(
                2,
                2,
                vec![
                    Complex64::new(0.0, 0.4),
                    Complex64::new(-0.2, 0.0),
                    Complex64::new(0.2, 0.0),
                    Complex64::new(0.0, 0.6),
                ],
            )
            .map_err(Into::into)
        } else {
            DenseMatrix::try_new(2, 2, vec![0.0.into(); 4]).map_err(Into::into)
        }
    })
    .unwrap();
    let density = density_matrix(&lesser, 0).unwrap();
    assert_eq!(density.get(0, 0).unwrap(), Complex64::new(0.4, 0.0));
    assert!((bond_flow_into(&lesser, 0, 0, 1, 1.0.into(), 1.0).unwrap() + 0.4).abs() < 1.0e-12);
}

#[test]
fn invalid_grids_shapes_and_tolerances_are_rejected() {
    assert!(matches!(
        RealTimeGrid::try_new(vec![0.0, 0.0]),
        Err(KeldyshError::UnorderedTimeGrid { index: 1 })
    ));
    assert!(matches!(
        RealTimeGrid::try_new(vec![0.0, f64::NAN]),
        Err(KeldyshError::NonFinite(_))
    ));
    assert!(matches!(
        TwoTimeMatrix::try_new(2, 1, vec![scalar(0.0.into())]),
        Err(KeldyshError::Shape { .. })
    ));
    let green = equilibrium_level(RealTimeGrid::try_new(vec![0.0, 1.0]).unwrap(), 1.0, 0.5);
    assert_eq!(
        green.consistency_report().unwrap().is_consistent(0.0),
        Err(KeldyshError::InvalidTolerance)
    );
}

#[test]
fn deliberate_support_violation_is_reported() {
    let grid = RealTimeGrid::try_new(vec![0.0, 1.0]).unwrap();
    let bad = scalar_function(2, |_, _| 1.0.into());
    let zero = scalar_function(2, |_, _| 0.0.into());
    let green = KeldyshGreen::try_new(grid, bad, zero.clone(), zero.clone(), zero).unwrap();
    let report = green.consistency_report().unwrap();
    assert_eq!(report.retarded_causality, 1.0);
    assert!(!report.is_consistent(1.0e-12).unwrap());
}
