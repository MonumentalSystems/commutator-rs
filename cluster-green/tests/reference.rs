use core::f64::consts::{PI, TAU};

use cluster_green::{ClusterGreenGrid, DenseMatrix, GreenError, GreenPoint};
use num_complex::Complex64;

fn scalar_green(z: Complex64, energy: f64) -> DenseMatrix {
    DenseMatrix::from_scalar(Complex64::new(1.0, 0.0) / (z - Complex64::from(energy))).unwrap()
}

#[test]
fn one_site_cpt_matches_lorentzian() {
    let z = Complex64::new(0.2, 0.05);
    let grid =
        ClusterGreenGrid::try_new(1, vec![GreenPoint::new(z, scalar_green(z, 0.0))]).unwrap();
    let embedded = grid
        .cpt_at(&DenseMatrix::from_scalar(Complex64::from(0.3)).unwrap())
        .unwrap();
    let expected_green = Complex64::new(1.0, 0.0) / (z - Complex64::from(0.3));
    assert!((embedded.points()[0].green.get(0, 0).unwrap() - expected_green).norm() < 1.0e-13);
    let expected_spectral = z.im / (PI * ((z.re - 0.3).powi(2) + z.im.powi(2)));
    assert!((embedded.spectral_function(0).unwrap() - expected_spectral).abs() < 1.0e-12);
}

#[test]
fn two_site_embedding_matches_direct_resolvent() {
    let z = Complex64::new(-0.1, 0.08);
    let cluster_inverse =
        DenseMatrix::try_new(2, 2, vec![z, Complex64::from(0.7), Complex64::from(0.7), z]).unwrap();
    let cluster_green = cluster_inverse.inverse().unwrap();
    let grid = ClusterGreenGrid::try_new(2, vec![GreenPoint::new(z, cluster_green)]).unwrap();
    let hopping = DenseMatrix::try_new(
        2,
        2,
        vec![
            Complex64::from(0.2),
            Complex64::from(-0.15),
            Complex64::from(-0.15),
            Complex64::from(-0.1),
        ],
    )
    .unwrap();
    let embedded = grid.cpt_at(&hopping).unwrap();
    let actual = &embedded.points()[0].green;
    let expected = cluster_inverse
        .subtract(&hopping)
        .unwrap()
        .inverse()
        .unwrap();
    for (actual, expected) in actual.as_slice().iter().zip(expected.as_slice()) {
        assert!((*actual - *expected).norm() < 1.0e-12);
    }
}

#[test]
fn inversion_uses_pivoting_and_detects_singular_matrices() {
    let matrix =
        DenseMatrix::try_new(2, 2, vec![0.0.into(), 1.0.into(), 2.0.into(), 3.0.into()]).unwrap();
    let identity = matrix.multiply(&matrix.inverse().unwrap()).unwrap();
    for row in 0..2 {
        for column in 0..2 {
            let expected = if row == column { 1.0 } else { 0.0 };
            assert!(
                (identity.get(row, column).unwrap() - Complex64::from(expected)).norm() < 1.0e-13
            );
        }
    }
    let singular =
        DenseMatrix::try_new(2, 2, vec![1.0.into(), 2.0.into(), 2.0.into(), 4.0.into()]).unwrap();
    assert!(matches!(
        singular.inverse(),
        Err(GreenError::SingularMatrix { .. })
    ));
}

#[test]
fn periodization_uses_explicit_site_phases() {
    let z = Complex64::new(0.0, 1.0);
    let green = DenseMatrix::try_new(
        2,
        2,
        vec![
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, -0.5),
            Complex64::new(0.0, -0.5),
            Complex64::new(0.0, -1.0),
        ],
    )
    .unwrap();
    let grid = ClusterGreenGrid::try_new(2, vec![GreenPoint::new(z, green)]).unwrap();
    let embedded = grid
        .cpt_at(&DenseMatrix::try_new(2, 2, vec![Complex64::new(0.0, 0.0); 4]).unwrap())
        .unwrap();
    let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let bonding = embedded.periodized_green(0, &positions, [0.0; 3]).unwrap();
    let antibonding = embedded
        .periodized_green(0, &positions, [PI, 0.0, 0.0])
        .unwrap();
    assert!((bonding - Complex64::new(0.0, -1.5)).norm() < 1.0e-12);
    assert!((antibonding - Complex64::new(0.0, -0.5)).norm() < 1.0e-12);

    // Periodic phases are invariant under reciprocal shifts for integer sites.
    let shifted = embedded
        .periodized_green(0, &positions, [TAU, 0.0, 0.0])
        .unwrap();
    assert!((shifted - bonding).norm() < 1.0e-12);
}

#[test]
fn invalid_broadening_order_causality_and_hopping_are_rejected() {
    let retarded = Complex64::new(0.0, 0.1);
    assert!(matches!(
        ClusterGreenGrid::try_new(
            1,
            vec![GreenPoint::new(
                Complex64::new(0.0, -0.1),
                scalar_green(retarded, 0.0),
            )]
        ),
        Err(GreenError::NonPositiveBroadening { index: 0 })
    ));
    assert!(matches!(
        ClusterGreenGrid::try_new(
            1,
            vec![
                GreenPoint::new(Complex64::new(1.0, 0.1), scalar_green(retarded, 0.0)),
                GreenPoint::new(Complex64::new(0.0, 0.1), scalar_green(retarded, 0.0)),
            ]
        ),
        Err(GreenError::UnorderedFrequencyGrid { index: 1 })
    ));
    assert!(matches!(
        ClusterGreenGrid::try_new(
            1,
            vec![GreenPoint::new(
                retarded,
                DenseMatrix::from_scalar(Complex64::new(0.0, 1.0)).unwrap(),
            )]
        ),
        Err(GreenError::NonCausal { index: 0, .. })
    ));

    let grid = ClusterGreenGrid::try_new(
        2,
        vec![GreenPoint::new(
            retarded,
            DenseMatrix::try_new(
                2,
                2,
                vec![
                    Complex64::new(0.0, -1.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, -1.0),
                ],
            )
            .unwrap(),
        )],
    )
    .unwrap();
    let nonhermitian =
        DenseMatrix::try_new(2, 2, vec![0.0.into(), 1.0.into(), 0.0.into(), 0.0.into()]).unwrap();
    assert!(matches!(
        grid.cpt_at(&nonhermitian),
        Err(GreenError::NonHermitian(_))
    ));
    let wrong_shape = DenseMatrix::try_new(1, 4, vec![0.0.into(); 4]).unwrap();
    assert!(matches!(
        grid.cpt_at(&wrong_shape),
        Err(GreenError::Dimensions {
            expected: (2, 2),
            actual: (1, 4),
            ..
        })
    ));
    assert!(matches!(
        DenseMatrix::from_scalar(Complex64::new(f64::NAN, 0.0)),
        Err(GreenError::NonFinite(_))
    ));
}
