//! Emit a synthetic SI calibration using the experiment result boundary.
//!
//! No electronic-structure calculation or material data is used in this example.
use berry_response::{
    Amplitude, BerryConvention, Complex64, Dipole2D, FieldOrdering, HallBar, LinearTransport,
    Relaxation,
};
use experiment_core::{RunMetadata, WorkResult};
use std::collections::BTreeMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dipole = Dipole2D::new([1e-9, 0.0], BerryConvention::ConnectionI)?;
    let relaxation = Relaxation::new(1e-13, std::f64::consts::TAU * 100.0)?;
    let response = dipole.harmonic_response(relaxation, FieldOrdering::Symmetric)?;
    let linear = LinearTransport::sheet_conductance_siemens([[1e-4, 0.0], [0.0, 1e-4]])?;
    let bar = HallBar::sheet(0, 2e-6, [0.0, 2e-6])?;
    let voltage = bar.second_harmonic_voltage(
        &linear,
        &response,
        Complex64::new(1e-6, 0.0),
        Amplitude::Peak,
        Amplitude::Peak,
    )?;
    let source_revision = std::env::var("COMMUTATOR_SOURCE_REVISION").unwrap_or_else(|_| {
        "unrecorded; supply COMMUTATOR_SOURCE_REVISION for a reproducible run".into()
    });
    let metadata = RunMetadata {
        seed: 0,
        backend: Some("berry-response-portable-cpu".into()),
        precision: Some("f64".into()),
        implementation_version: Some(env!("CARGO_PKG_VERSION").into()),
        schema_version: 1,
        attributes: BTreeMap::from([
            (
                "scope".into(),
                "synthetic calibration; not a material prediction".into(),
            ),
            ("source_revision".into(), source_revision),
            ("berry_connection".into(), "i<u|grad u>".into()),
            ("frame".into(), "right-handed xy sheet; normal +z".into()),
            ("dipole_metres".into(), "[1e-9,0]".into()),
            ("field_ordering".into(), "symmetrized; current-first".into()),
            (
                "amplitudes".into(),
                "input current peak; output voltage peak".into(),
            ),
            ("tau_seconds".into(), "1e-13".into()),
            ("drive_frequency_hz".into(), "100".into()),
            ("sheet_conductance_siemens".into(), "diag(1e-4,1e-4)".into()),
            ("bar_width_metres".into(), "2e-6".into()),
            ("voltage_lead_displacement_metres".into(), "[0,2e-6]".into()),
            ("drive_current_amperes_peak".into(), "[1e-6,0]".into()),
            (
                "result_layout".into(),
                "[voltage_real_V, voltage_imag_V]".into(),
            ),
        ]),
    };
    let result = WorkResult {
        experiment_id: "berry-response-reference-v1".into(),
        instance_id: "synthetic-one-nanometre".into(),
        unit_id: "one-field".into(),
        generation: 0,
        run: metadata,
        elapsed_ms: 0,
        metrics: BTreeMap::new(),
        payload: [voltage.re, voltage.im],
    };
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
