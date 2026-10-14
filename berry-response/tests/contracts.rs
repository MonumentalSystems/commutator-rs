use berry_response::{
    Amplitude, BerryConvention, Complex64, Dipole2D, Dipole3D, FieldOrdering, HallBar,
    LinearTransport, Relaxation, ResponseError, ResponseKind,
};

fn c(value: f64) -> Complex64 {
    Complex64::new(value, 0.0)
}
fn near(actual: Complex64, expected: Complex64) {
    let scale = expected.norm().max(1e-30);
    assert!(
        (actual - expected).norm() <= 3e-13 * scale,
        "actual {actual:?}, expected {expected:?}"
    );
}
fn te() -> Dipole3D {
    Dipole3D::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]],
        BerryConvention::ConnectionI,
    )
    .unwrap()
}
fn relaxation() -> Relaxation {
    Relaxation::new(1e-13, 1e13).unwrap()
}

#[test]
fn te_mixed_permutations_give_three_not_two() {
    let k = relaxation().harmonic_prefactor().unwrap();
    let tensor = te()
        .harmonic_response(relaxation(), FieldOrdering::Ordered)
        .unwrap();
    let data = tensor.components();
    near(data[1][0][2], k);
    near(data[1][2][0], 2.0 * k);
    let j = tensor.current([c(2.0), c(0.0), c(3.0)]).unwrap();
    near(j[1], 18.0 * k);
    assert!((j[1] - 12.0 * k).norm() > k.norm());
    assert_eq!(tensor.kind(), ResponseKind::SecondHarmonic);
}

#[test]
fn symmetrized_tensor_retains_the_full_contraction() {
    let d = Dipole3D::new(
        [[0.2, 0.7, -0.6], [1.0, -0.8, 0.1], [0.4, -0.3, 0.9]],
        BerryConvention::ConnectionI,
    )
    .unwrap();
    let ordered = d
        .harmonic_response(relaxation(), FieldOrdering::Ordered)
        .unwrap();
    let sym = d
        .harmonic_response(relaxation(), FieldOrdering::Symmetric)
        .unwrap();
    let fields = [
        Complex64::new(0.2, 0.7),
        Complex64::new(-0.8, 0.1),
        Complex64::new(1.1, -0.6),
    ];
    let a = ordered.current(fields).unwrap();
    let b = sym.current(fields).unwrap();
    for (actual, expected) in a.into_iter().zip(b) {
        near(actual, expected);
    }
    let data = sym.components();
    for plane in data {
        for (b, row) in plane.iter().enumerate() {
            for (c, value) in row.iter().enumerate() {
                assert_eq!(*value, plane[c][b]);
            }
        }
    }
    let sym_te = te()
        .harmonic_response(relaxation(), FieldOrdering::Symmetric)
        .unwrap();
    near(
        sym_te.components()[1][0][2],
        1.5 * relaxation().harmonic_prefactor().unwrap(),
    );
    assert_eq!(sym.ordering(), FieldOrdering::Symmetric);
}

#[test]
fn arbitrary_tensors_match_independent_cross_product_and_transversality() {
    let k = relaxation().harmonic_prefactor().unwrap();
    for sample in 1..=128 {
        let s = sample as f64;
        let d =
            core::array::from_fn(|b| core::array::from_fn(|v| ((b * 3 + v + 1) as f64 * s).sin()));
        let field = [(s * 0.17).sin(), (s * 0.31).cos(), (s * 0.43).sin()];
        let v: [f64; 3] = core::array::from_fn(|i| (0..3).map(|b| d[b][i] * field[b]).sum());
        let cross = [
            field[1] * v[2] - field[2] * v[1],
            field[2] * v[0] - field[0] * v[2],
            field[0] * v[1] - field[1] * v[0],
        ];
        let tensor = Dipole3D::new(d, BerryConvention::ConnectionI)
            .unwrap()
            .harmonic_response(relaxation(), FieldOrdering::Ordered)
            .unwrap();
        let j = tensor.current(field.map(c)).unwrap();
        for (actual, coefficient) in j.into_iter().zip(cross) {
            assert!((actual - k * coefficient).norm() < 1e-15);
        }
        let dot: Complex64 = j
            .into_iter()
            .zip(field)
            .map(|(current, e)| current * e)
            .sum();
        assert!(dot.norm() < 1e-15);
    }
}

#[test]
fn isotropic_tensor_and_te_in_plane_are_zero() {
    let d = Dipole3D::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        BerryConvention::ConnectionI,
    )
    .unwrap();
    assert_eq!(
        d.harmonic_response(relaxation(), FieldOrdering::Ordered)
            .unwrap()
            .current([c(1.0), c(2.0), c(3.0)])
            .unwrap(),
        [c(0.0); 3]
    );
    assert_eq!(
        te().harmonic_response(relaxation(), FieldOrdering::Ordered)
            .unwrap()
            .current([c(1.0), c(2.0), c(0.0)])
            .unwrap(),
        [c(0.0); 3]
    );
}

#[test]
fn two_dimensional_sign_and_berry_convention_are_explicit() {
    let d = Dipole2D::new([2e-9, 3e-9], BerryConvention::ConnectionI).unwrap();
    let k = relaxation().harmonic_prefactor().unwrap();
    let tensor = d
        .harmonic_response(relaxation(), FieldOrdering::Ordered)
        .unwrap();
    let j = tensor.current([c(2.0), c(0.0)]).unwrap();
    assert_eq!(j[0], c(0.0));
    near(j[1], -8e-9 * k);
    assert!(j[1].re < 0.0);
    let other = Dipole2D::new([-2e-9, -3e-9], BerryConvention::ConnectionMinusI).unwrap();
    assert_eq!(d, other);
    assert_eq!(other.metres(), [2e-9, 3e-9]);
    let bulk = Dipole3D::new(
        te().dimensionless().map(|r| r.map(|v| -v)),
        BerryConvention::ConnectionMinusI,
    )
    .unwrap();
    assert_eq!(bulk, te());
}

#[test]
fn static_is_twice_the_zero_frequency_ac_limit_and_disallows_ac_fields() {
    let tau = 1e-13;
    let ac = te()
        .harmonic_response(Relaxation::new(tau, 0.0).unwrap(), FieldOrdering::Ordered)
        .unwrap();
    let dc = te().static_response(tau, FieldOrdering::Ordered).unwrap();
    let field = [c(2.0), c(0.0), c(1.0)];
    near(
        dc.current(field).unwrap()[1],
        2.0 * ac.current(field).unwrap()[1],
    );
    assert_eq!(dc.kind(), ResponseKind::Static);
    assert!(dc
        .current([Complex64::new(1.0, 0.1), c(0.0), c(1.0)])
        .is_err());
    let transport = LinearTransport::bulk_conductivity_siemens_per_metre([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    .unwrap();
    let bar = HallBar::bulk(0, 1.0, 1.0, [0.0, 1.0, 0.0]).unwrap();
    assert!(bar
        .second_harmonic_voltage(&transport, &dc, c(1.0), Amplitude::Peak, Amplitude::Peak)
        .is_err());
}

#[test]
fn harmonic_relaxation_phase_and_doubled_drive_phase() {
    let zero = Relaxation::new(1e-13, 0.0)
        .unwrap()
        .harmonic_prefactor()
        .unwrap();
    let k = relaxation().harmonic_prefactor().unwrap();
    near(k, zero * Complex64::new(0.5, -0.5));
    let tensor = te()
        .harmonic_response(relaxation(), FieldOrdering::Ordered)
        .unwrap();
    let phase = Complex64::from_polar(1.0, 0.37);
    let a = tensor.current([c(1.0), c(0.0), c(2.0)]).unwrap();
    let b = tensor.current([phase, c(0.0), 2.0 * phase]).unwrap();
    near(b[1], a[1] * phase * phase);
    assert!(Relaxation::new(1.0, 1e200)
        .unwrap()
        .harmonic_prefactor()
        .unwrap()
        .im
        .is_finite());
}

#[test]
fn sheet_open_circuit_sign_and_lead_reversal() {
    let transport = LinearTransport::sheet_conductance_siemens([[4.0, 0.0], [0.0, 2.0]]).unwrap();
    let bar = HallBar::sheet(0, 2.0, [0.0, 3.0]).unwrap();
    let reversed = HallBar::sheet(0, 2.0, [0.0, -3.0]).unwrap();
    let source = [c(0.0), c(-8.0)];
    let field = transport.open_circuit_field(source).unwrap();
    assert_eq!(field[0], c(0.0));
    near(field[1], c(4.0));
    near(
        bar.open_circuit_voltage(&transport, source, Amplitude::Peak)
            .unwrap(),
        c(-12.0),
    );
    near(
        reversed
            .open_circuit_voltage(&transport, source, Amplitude::Peak)
            .unwrap(),
        c(12.0),
    );
}

#[test]
fn full_anisotropic_transport_is_used_for_drive_and_detection() {
    let transport = LinearTransport::sheet_conductance_siemens([[4.0, 1.0], [1.0, 2.0]]).unwrap();
    let bar = HallBar::sheet(0, 1.0, [0.0, 3.0]).unwrap();
    let field = bar
        .drive_field(&transport, c(4.0), Amplitude::Peak)
        .unwrap();
    near(field[0], c(8.0 / 7.0));
    near(field[1], c(-4.0 / 7.0));
    near(
        bar.open_circuit_voltage(&transport, [c(7.0), c(0.0)], Amplitude::Peak)
            .unwrap(),
        c(-3.0),
    );
    assert_eq!(transport.components_si(), [[4.0, 1.0], [1.0, 2.0]]);
    assert_eq!(bar.drive_axis(), 0);
    assert_eq!(bar.current_cross_section_si(), 1.0);
    assert_eq!(bar.voltage_displacement_metres(), [0.0, 3.0]);
}

#[test]
fn rms_fundamental_is_squared_but_rms_detection_is_divided_once() {
    let transport = LinearTransport::sheet_conductance_siemens([[0.01, 0.0], [0.0, 0.02]]).unwrap();
    let bar = HallBar::sheet(0, 1e-6, [0.0, 1e-6]).unwrap();
    let response = Dipole2D::new([1e-9, 0.0], BerryConvention::ConnectionI)
        .unwrap()
        .harmonic_response(relaxation(), FieldOrdering::Ordered)
        .unwrap();
    let p = bar
        .second_harmonic_voltage(
            &transport,
            &response,
            c(1e-6),
            Amplitude::Peak,
            Amplitude::Peak,
        )
        .unwrap();
    let input_rms = bar
        .second_harmonic_voltage(
            &transport,
            &response,
            c(1e-6),
            Amplitude::Rms,
            Amplitude::Peak,
        )
        .unwrap();
    let both_rms = bar
        .second_harmonic_voltage(
            &transport,
            &response,
            c(1e-6),
            Amplitude::Rms,
            Amplitude::Rms,
        )
        .unwrap();
    near(input_rms, 2.0 * p);
    near(both_rms, core::f64::consts::SQRT_2 * p);
    near(
        Amplitude::Rms
            .from_peak(Amplitude::Rms.to_peak(c(2.0)).unwrap())
            .unwrap(),
        c(2.0),
    );
}

#[test]
fn sheet_bulk_conversions_require_both_dipole_and_conductivity_thickness() {
    let thickness = 5e-9;
    let width = 2e-6;
    let sigma3 = LinearTransport::bulk_conductivity_siemens_per_metre([
        [1e5, 0.0, 0.0],
        [0.0, 2e5, 0.0],
        [0.0, 0.0, 3e5],
    ])
    .unwrap();
    let sigma2 = LinearTransport::sheet_conductance_siemens([
        [1e5 * thickness, 0.0],
        [0.0, 2e5 * thickness],
    ])
    .unwrap();
    let d3 = Dipole3D::new(
        [[0.0, 0.0, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        BerryConvention::ConnectionI,
    )
    .unwrap();
    let d2 = Dipole2D::new([0.1 * thickness, 0.0], BerryConvention::ConnectionI).unwrap();
    let bar3 = HallBar::bulk(0, width, thickness, [0.0, width, 0.0]).unwrap();
    let bar2 = HallBar::sheet(0, width, [0.0, width]).unwrap();
    let response3 = d3
        .harmonic_response(relaxation(), FieldOrdering::Ordered)
        .unwrap();
    let response2 = d2
        .harmonic_response(relaxation(), FieldOrdering::Ordered)
        .unwrap();
    let v3 = bar3
        .second_harmonic_voltage(
            &sigma3,
            &response3,
            c(1e-6),
            Amplitude::Peak,
            Amplitude::Peak,
        )
        .unwrap();
    let v2 = bar2
        .second_harmonic_voltage(
            &sigma2,
            &response2,
            c(1e-6),
            Amplitude::Peak,
            Amplitude::Peak,
        )
        .unwrap();
    near(v3, v2);
    assert_eq!(bar3.current_cross_section_si(), width * thickness);
}

#[test]
fn bad_physical_scalars_fail_before_use() {
    for value in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(Relaxation::new(value, 0.0).is_err());
        assert!(te().static_response(value, FieldOrdering::Ordered).is_err());
        assert!(HallBar::sheet(0, value, [0.0, 1.0]).is_err());
        assert!(HallBar::bulk(0, 1.0, value, [0.0, 1.0, 0.0]).is_err());
    }
    for value in [-0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(Relaxation::new(1.0, value).is_err());
    }
    assert!(HallBar::sheet(2, 1.0, [0.0, 1.0]).is_err());
    assert!(HallBar::sheet(0, 1.0, [0.0, 0.0]).is_err());
    assert!(HallBar::bulk(0, 1e308, 1e308, [0.0, 1.0, 0.0]).is_err());
    assert!(HallBar::bulk(0, 1e-308, 1e-308, [0.0, 1.0, 0.0]).is_err());
}

#[test]
fn invalid_matrix_data_and_nonfinite_phasors_are_rejected() {
    for data in [
        [[0.0, 0.0], [0.0, 1.0]],
        [[-0.0, 0.0], [0.0, 1.0]],
        [[1.0, 2.0], [2.0, 1.0]],
        [[1.0, 0.1], [0.0, 1.0]],
        [[1.0, f64::NAN], [0.0, 1.0]],
    ] {
        assert!(LinearTransport::sheet_conductance_siemens(data).is_err());
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(Dipole2D::new([value, 0.0], BerryConvention::ConnectionI).is_err());
        assert!(Dipole3D::new([[value; 3]; 3], BerryConvention::ConnectionI).is_err());
        assert!(HallBar::sheet(0, 1.0, [value, 0.0]).is_err());
        assert!(Amplitude::Peak.to_peak(c(value)).is_err());
        assert!(te()
            .harmonic_response(relaxation(), FieldOrdering::Ordered)
            .unwrap()
            .current([c(value), c(0.0), c(0.0)])
            .is_err());
    }
}

#[test]
fn overflows_are_errors_instead_of_nonfinite_outputs() {
    assert_eq!(
        Amplitude::Rms.to_peak(c(f64::MAX)),
        Err(ResponseError::NumericalOverflow)
    );
    assert!(Relaxation::new(f64::MAX, 1.0).is_err());
    assert!(Relaxation::new(1e100, 1e300).is_err());
    let tensor = te()
        .harmonic_response(relaxation(), FieldOrdering::Ordered)
        .unwrap();
    assert_eq!(
        tensor.current([c(f64::MAX); 3]),
        Err(ResponseError::NumericalOverflow)
    );
    let transport =
        LinearTransport::sheet_conductance_siemens([[1e-300, 0.0], [0.0, 1e-300]]).unwrap();
    assert_eq!(
        transport.electric_field([c(f64::MAX), c(0.0)]),
        Err(ResponseError::NumericalOverflow)
    );
}
