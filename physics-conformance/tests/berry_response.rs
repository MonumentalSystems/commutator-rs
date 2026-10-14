use berry_response::{BerryConvention, Complex64, Dipole3D, FieldOrdering, Relaxation};
use clifford_core::CliffordAlgebra;
use experiment_core::{RunMetadata, WorkResult, WorkUnit};
use std::collections::BTreeMap;

// Independent geometric-product route: j/K = dual(E wedge (D^T E)).
// Cl(3,0), canonical e123, right dual B I^-1 with I^-1=-I.
fn geometric_current(algebra: &CliffordAlgebra, d: [[f64; 3]; 3], e: [f64; 3]) -> [f64; 3] {
    let u: [f64; 3] = std::array::from_fn(|i| (0..3).map(|b| d[b][i] * e[b]).sum());
    let product = algebra.geometric_product(&algebra.vector(&e), &algebra.vector(&u));
    let wedge = algebra.grade_project(&product, 2);
    let mut inverse_i = algebra.basis::<f64>(algebra.bitmap_to_index[7]);
    inverse_i.iter_mut().for_each(|v| *v = -*v);
    let dual = algebra.geometric_product(&wedge, &inverse_i);
    let values = algebra.vector_components(&dual);
    [values[0], values[1], values[2]]
}

#[test]
fn berry_tensor_matches_clifford_dual_in_one_frozen_frame() {
    let algebra = CliffordAlgebra::cl3();
    let relaxation = Relaxation::new(1e-13, 3e12).unwrap();
    let prefactor = relaxation.harmonic_prefactor().unwrap();
    for index in 0..64 {
        let s = index as f64 + 0.17;
        let d =
            std::array::from_fn(|i| std::array::from_fn(|j| ((3 * i + j + 1) as f64 * s).cos()));
        let e = [s.sin(), (1.7 * s).cos(), (0.37 * s).sin()];
        let reference = geometric_current(&algebra, d, e);
        for ordering in [FieldOrdering::Ordered, FieldOrdering::Symmetric] {
            let response = Dipole3D::new(d, BerryConvention::ConnectionI)
                .unwrap()
                .harmonic_response(relaxation, ordering)
                .unwrap();
            let current = response.current(e.map(|x| Complex64::new(x, 0.0))).unwrap();
            for axis in 0..3 {
                assert!((current[axis] - prefactor * reference[axis]).norm() < 1e-14);
            }
        }
    }
}

#[test]
fn experiment_transport_preserves_signed_complex_results_and_conventions() {
    // Payloads at the serialization boundary deliberately use plain SI arrays.
    // Domain types are reconstructed through checked constructors by the worker.
    let metadata = RunMetadata {
        seed: 0,
        backend: Some("portable-cpu-reference".into()),
        precision: Some("f64".into()),
        implementation_version: Some(env!("CARGO_PKG_VERSION").into()),
        schema_version: 1,
        attributes: BTreeMap::from([
            ("berry_connection".into(), "i<u|grad u>".into()),
            ("frame".into(), "right-handed; epsilon_xyz=+1".into()),
            ("dipole".into(), "dimensionless; diag(1,1,-2)".into()),
            ("drive_amplitude".into(), "complex peak V/m".into()),
            ("fourier".into(), "exp(+i omega t)".into()),
            ("tau_seconds".into(), "1e-13".into()),
            ("omega_radians_per_second".into(), "3e12".into()),
            (
                "fixture_source".into(),
                "Tsirkin2018 Eq5/A4/B4; synthetic fields".into(),
            ),
        ]),
    };
    let work = WorkUnit {
        experiment_id: "berry-response-reference-v1".into(),
        instance_id: "te-mixed-field-control".into(),
        unit_id: "one-field".into(),
        generation: 0,
        run: metadata,
        payload: [1.0, 0.0, 1.0],
    };
    let encoded = serde_json::to_string(&work).unwrap();
    let received: WorkUnit<[f64; 3]> = serde_json::from_str(&encoded).unwrap();
    let response = Dipole3D::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]],
        BerryConvention::ConnectionI,
    )
    .unwrap()
    .harmonic_response(
        Relaxation::new(1e-13, 3e12).unwrap(),
        FieldOrdering::Ordered,
    )
    .unwrap();
    let j = response
        .current(received.payload.map(|x| Complex64::new(x, 0.0)))
        .unwrap();
    assert!(j[1].re > 0.0 && j[1].im < 0.0);
    let result = WorkResult {
        experiment_id: received.experiment_id,
        instance_id: received.instance_id,
        unit_id: received.unit_id,
        generation: received.generation,
        run: received.run,
        elapsed_ms: 0,
        metrics: BTreeMap::new(),
        payload: j.map(|z| [z.re, z.im]),
    };
    let encoded = serde_json::to_string(&result).unwrap();
    let decoded: WorkResult<[[f64; 2]; 3]> = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, result);
    assert_eq!(decoded.run, work.run);
}
