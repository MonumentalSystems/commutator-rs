use cluster_embedding::{embed_self_energy, Complex64 as GreenComplex, DenseMatrix};
use open_quantum_systems::{CollapseOperator, DensityMatrix, LindbladModel, Operator};
use quantum_shadows::{MeasurementBasis, Pauli, PauliString, ShadowDataset, Snapshot};
use quantum_tomography::linear_inversion;
use superconducting_dynamics::{Complex64, OrderParameter, SiteParameters, TdglModel};
use superconductivity::{NormalHamiltonian, OnsiteSWaveModel};

fn close(a: GreenComplex, b: GreenComplex, tolerance: f64) {
    assert!((a - b).norm() < tolerance, "{a} != {b}");
}

#[test]
fn tomography_state_survives_open_system_amplitude_damping_roundtrip() {
    let initial = linear_inversion(1, &[1.0, 0.0, 0.0, -1.0], 1e-12).unwrap();
    let initial = DensityMatrix::try_new(initial, 1e-12).unwrap();
    let zero = Operator::try_new(2, vec![0.0.into(); 4]).unwrap();
    let lowering =
        Operator::try_new(2, vec![0.0.into(), 1.0.into(), 0.0.into(), 0.0.into()]).unwrap();
    let model = LindbladModel::try_new(
        zero,
        vec![CollapseOperator::try_new(lowering, 0.7).unwrap()],
        1e-12,
    )
    .unwrap();
    let evolved = model.evolve_rk4(&initial, 0.002, 500, 1e-10).unwrap();
    let z = evolved.get(0, 0).unwrap().re - evolved.get(1, 1).unwrap().re;
    let reconstructed = linear_inversion(1, &[1.0, 0.0, 0.0, z], 1e-10).unwrap();
    for row in 0..2 {
        for column in 0..2 {
            close(
                reconstructed.get(row, column).unwrap(),
                evolved.get(row, column).unwrap(),
                2e-8,
            );
        }
    }
}

#[test]
fn shadow_pauli_estimates_feed_tomography_without_reordering() {
    let snapshots = vec![
        Snapshot::try_new(vec![MeasurementBasis::X], vec![1]).unwrap(),
        Snapshot::try_new(vec![MeasurementBasis::X], vec![-1]).unwrap(),
        Snapshot::try_new(vec![MeasurementBasis::Y], vec![1]).unwrap(),
        Snapshot::try_new(vec![MeasurementBasis::Y], vec![-1]).unwrap(),
        Snapshot::try_new(vec![MeasurementBasis::Z], vec![1]).unwrap(),
        Snapshot::try_new(vec![MeasurementBasis::Z], vec![1]).unwrap(),
    ];
    let data = ShadowDataset::try_new(1, snapshots).unwrap();
    let estimate = |p| data.mean(&PauliString::try_new(vec![p]).unwrap()).unwrap();
    let rho = linear_inversion(
        1,
        &[
            estimate(Pauli::I),
            estimate(Pauli::X),
            estimate(Pauli::Y),
            estimate(Pauli::Z),
        ],
        1e-12,
    )
    .unwrap();
    close(rho.get(0, 0).unwrap(), 1.0.into(), 1e-12);
    close(rho.get(1, 1).unwrap(), 0.0.into(), 1e-12);
}

#[test]
fn scalar_dyson_embedding_matches_transport_and_keldysh_matrix_type() {
    let energy = 0.4;
    let eta = 0.2;
    let hopping = DenseMatrix::from_scalar(GreenComplex::new(-0.3, 0.0)).unwrap();
    let sigma = DenseMatrix::from_scalar(GreenComplex::new(-0.1, -0.15)).unwrap();
    let embedded = embed_self_energy(
        &[GreenComplex::new(energy, eta)],
        std::slice::from_ref(&sigma),
        &hopping,
        0.0,
    )
    .unwrap();
    let transported =
        quantum_transport::retarded_device_green(energy, eta, &hopping, &[&sigma], 1e-12).unwrap();
    close(
        embedded.points()[0].green.get(0, 0).unwrap(),
        transported.get(0, 0).unwrap(),
        1e-12,
    );
    let _: keldysh_green::DenseMatrix = transported;
}

#[test]
fn tdgl_order_parameter_scales_into_particle_hole_symmetric_bdg() {
    let site = SiteParameters::try_new(-1.0, 1.0, 1.0).unwrap();
    let tdgl = TdglModel::uniform_rectangular(2, 1, site, 1.0).unwrap();
    let order = OrderParameter::try_new(
        &tdgl,
        vec![Complex64::new(0.2, 0.1), Complex64::new(-0.1, 0.3)],
    )
    .unwrap();
    let gaps = tdgl.scaled_pairing_gaps(&order, 1.5).unwrap();
    let normal = NormalHamiltonian::try_from_dense(2, vec![0.0.into(); 16], 1e-12).unwrap();
    let bdg = OnsiteSWaveModel::try_new(normal, 0.0, gaps)
        .unwrap()
        .hamiltonian();
    assert!(bdg.is_hermitian(1e-12));
    assert!(bdg.particle_hole_residual() < 1e-12);
}
