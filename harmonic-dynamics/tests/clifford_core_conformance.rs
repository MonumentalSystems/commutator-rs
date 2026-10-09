use clifford_core::CliffordAlgebra;
use harmonic_dynamics::quaternion::Quaternion;

#[test]
fn hamilton_product_matches_cl02_shortlex_product() {
    let algebra = CliffordAlgebra::quaternion();
    let left = Quaternion::new(0.3, -0.4, 0.5, 0.7).unwrap();
    let right = Quaternion::new(-0.2, 0.6, 0.1, -0.3).unwrap();

    // Cl(0,2) ShortLex is [1, e1, e2, e12], exactly [w, x, y, z].
    let expected = algebra.geometric_product(left.as_array(), right.as_array());
    let actual = left.hamilton_product(right).unwrap().into_array();

    assert!(actual
        .iter()
        .zip(expected)
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-6));
}
