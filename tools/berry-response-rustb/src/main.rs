use Rustb::{BerryCurvature, Conditions, FieldSymmetry, Integration, Model, Parameters};
use berry_response::{BerryConvention, Dipole2D, FieldOrdering, Relaxation};
use ndarray::{Array2, array};
use num_complex::Complex64;

fn model(tilt: f64, lattice: f64, chirality: f64) -> Model<false, 2> {
    let mut m =
        Model::<false, 2>::tb_model(Array2::eye(2) * lattice, Array2::zeros((2, 2)), None).unwrap();
    m.set_onsite(&array![1.0, -1.0], None);
    for r in [array![1, 0], array![0, 1]] {
        m.add_hop(0.5, 0, 0, &r, None);
        m.add_hop(-0.5, 1, 1, &r, None);
    }
    for state in 0..2 {
        m.add_hop(
            Complex64::new(0.0, -tilt / 2.0),
            state,
            state,
            &array![1, 0],
            None,
        );
    }
    for (r, amp) in [
        (array![1, 0], Complex64::new(0.0, -0.5)),
        (array![-1, 0], Complex64::new(0.0, 0.5)),
        (array![0, 1], Complex64::new(-0.5 * chirality, 0.0)),
        (array![0, -1], Complex64::new(0.5 * chirality, 0.0)),
    ] {
        m.add_hop(amp, 0, 1, &r, None);
    }
    m
}

// Analytic eigensystem of H(q)=t sin(qx) I + sin(qx) sx + c sin(qy) sy
// +(1+cos(qx)+cos(qy)) sz. q is dimensionless, Cartesian k=q/a.
// Lower/upper Berry curvatures follow A=i<u|grad u>, Ω=-2 Im<dxu|dyu>.
fn analytic(qx: f64, qy: f64, tilt: f64, chirality: f64) -> [(f64, f64, f64, f64); 2] {
    let (sx, cx) = qx.sin_cos();
    let (sy, cy) = qy.sin_cos();
    let r = (sx * sx + sy * sy + (1.0 + cx + cy).powi(2)).sqrt();
    let numerator = chirality * (cx + cy + cx * cy);
    let drdx = -sx * (1.0 + cy) / r;
    let omega = numerator / (2.0 * r.powi(3));
    let domega = -chirality * sx * (1.0 + cy) / (2.0 * r.powi(3))
        - 3.0 * numerator * drdx / (2.0 * r.powi(4));
    [-1.0, 1.0].map(|s| {
        (
            tilt * sx + s * r,
            tilt * cx + s * drdx,
            -s * omega,
            -s * domega,
        )
    })
}

fn reference(n: usize, tilt: f64, lattice: f64, chirality: f64) -> (f64, f64) {
    let width = 8.617333262145e-5 * 300.0; // eV: exact SI kB/e, rounded here to 13 digits.
    let mut fs = 0.0;
    let mut sea = 0.0;
    for ix in 0..n {
        for iy in 0..n {
            let qx = std::f64::consts::TAU * ix as f64 / n as f64;
            let qy = std::f64::consts::TAU * iy as f64 / n as f64;
            for (energy, velocity, berry, derivative) in analytic(qx, qy, tilt, chirality) {
                let u = (energy - 1.5) / width;
                let f = if u >= 0.0 {
                    let z = (-u).exp();
                    z / (1.0 + z)
                } else {
                    1.0 / (1.0 + u.exp())
                };
                let z = (-u.abs()).exp();
                let window = z / ((1.0 + z).powi(2) * width);
                fs += velocity * berry * window;
                sea += f * derivative;
            }
        }
    }
    // avg_BZ / a² * (a velocity)*(a² Berry) = a*avg.
    let scale = lattice / (n * n) as f64;
    (fs * scale, sea * scale)
}

fn component(m: &Model<false, 2>, n: usize, dirs: [[f64; 2]; 3], sym: FieldSymmetry) -> f64 {
    let p = Parameters {
        conditions: Conditions::fixed(300.0, 1.5, 0.0),
        kmesh: [n, n],
        integration: Integration::Direct,
    };
    m.extrinsic_nonlinear_hall(&p, dirs, 0.0, None, sym)
        .unwrap()
        .single()
        .unwrap()
}

fn close(label: &str, actual: f64, expected: f64, atol: f64, rtol: f64) {
    assert!(
        (actual - expected).abs() <= atol + rtol * expected.abs(),
        "{label}: actual={actual:e}, expected={expected:e}"
    );
}

fn qualify() -> serde_json::Value {
    let x = [1.0, 0.0];
    let y = [0.0, 1.0];
    let m = model(0.3, 1.0, 1.0);
    // Independent analytic bands and curvature at generic (non-high-symmetry) points.
    let mut point_max_error: f64 = 0.0;
    for k in [[0.0, 0.0], [0.127, 0.231], [0.42, 0.34], [0.77, 0.91]] {
        let b = m
            .berry_curvature_at(&array![k[0], k[1]], [x, y], 0.0, None)
            .unwrap();
        let a = analytic(
            std::f64::consts::TAU * k[0],
            std::f64::consts::TAU * k[1],
            0.3,
            1.0,
        );
        for (i, expected) in a.iter().enumerate() {
            close("analytic band", b.energies[i], expected.0, 2e-13, 2e-13);
            close(
                "analytic Berry",
                b.berry_curvature[i],
                expected.2,
                2e-13,
                2e-13,
            );
            point_max_error = point_max_error.max((b.berry_curvature[i] - expected.2).abs());
        }
    }
    let mut meshes = Vec::new();
    let mut last = 0.0;
    for n in [64, 128, 256, 512] {
        let (fs, sea) = reference(n, 0.3, 1.0, 1.0);
        let ordered = component(&m, n, [y, x, x], FieldSymmetry::Ordered);
        close("Rustb minus Dx", ordered, -fs, 1e-12, 1e-9);
        if n == 256 {
            close("FS vs sea", fs, sea, 1e-8, 1e-6);
        }
        meshes.push(serde_json::json!({"mesh":[n,n],"rustb_ordered_yxx_angstrom":ordered,"analytic_Dx_FS_angstrom":fs,"analytic_Dx_sea_angstrom":sea}));
        if n == 512 {
            close("mesh convergence", ordered, last, 1e-9, 1e-6);
            close("FS vs sea final", fs, sea, 1e-10, 1e-8);
        }
        last = ordered;
    }
    let n = 128;
    let ordered_xyx = component(&m, n, [x, y, x], FieldSymmetry::Ordered);
    let ordered_xxy = component(&m, n, [x, x, y], FieldSymmetry::Ordered);
    let sym_xyx = component(&m, n, [x, y, x], FieldSymmetry::Symmetrized);
    let sym_xxy = component(&m, n, [x, x, y], FieldSymmetry::Symmetrized);
    close("ordered forbidden", ordered_xxy, 0.0, 1e-13, 0.0);
    close(
        "sym averaged",
        sym_xyx,
        0.5 * (ordered_xyx + ordered_xxy),
        1e-13,
        1e-11,
    );
    close(
        "sum preserved",
        sym_xyx + sym_xxy,
        ordered_xyx + ordered_xxy,
        1e-13,
        1e-11,
    );
    let zero = component(&model(0.0, 1.0, 1.0), n, [y, x, x], FieldSymmetry::Ordered);
    let reversal = component(&model(-0.3, 1.0, 1.0), n, [y, x, x], FieldSymmetry::Ordered);
    let chirality = component(&model(0.3, 1.0, -1.0), n, [y, x, x], FieldSymmetry::Ordered);
    let scale = component(&model(0.3, 2.3, 1.0), n, [y, x, x], FieldSymmetry::Ordered);
    close("zero tilt inversion control", zero, 0.0, 1e-12, 0.0);
    close("tilt sign reversal", reversal, ordered_xyx, 1e-12, 1e-10);
    close(
        "Berry chirality reversal",
        chirality,
        ordered_xyx,
        1e-12,
        1e-10,
    );
    close("2D length scaling", scale, -2.3 * ordered_xyx, 1e-12, 1e-10);
    // Boundary conversion: Angstrom -> metre, no additional (2pi)^2 or cell area.
    let dx_m = ordered_xyx * 1e-10;
    let dipole = Dipole2D::new([dx_m, 0.0], BerryConvention::ConnectionI).unwrap();
    let relaxation = Relaxation::new(1e-13, 1e12).unwrap();
    let core = dipole
        .harmonic_response(relaxation, FieldOrdering::Ordered)
        .unwrap();
    // Rustb orders Berry field first and velocity field second. The core orders
    // velocity field first and Berry field second, so swap those two indices.
    let prefactor = relaxation.harmonic_prefactor().unwrap();
    let core_data = core.components();
    close(
        "core ordered xxy import",
        (core_data[0][0][1] / prefactor).re,
        ordered_xyx * 1e-10,
        1e-25,
        1e-12,
    );
    close(
        "core ordered xyx import",
        (core_data[0][1][0] / prefactor).norm(),
        ordered_xxy.abs() * 1e-10,
        1e-25,
        1e-12,
    );
    let fields = [Complex64::new(100.0, 20.0), Complex64::new(30.0, -10.0)];
    let current = core.current(fields).unwrap();
    let sym_current = dipole
        .harmonic_response(relaxation, FieldOrdering::Symmetric)
        .unwrap()
        .current(fields)
        .unwrap();
    for a in 0..2 {
        close(
            "core contraction symmetrization",
            (current[a] - sym_current[a]).norm(),
            0.0,
            1e-22,
            0.0,
        );
    }
    let report = serde_json::json!({
        "schema":"berry-response-rustb-qualification-v1", "passed":true,
        "rustb_git_rev":"ba04a2a0746f9798433cb4d9826675d10177fbc7",
        "backend":"netlib-system", "temperature_K":300.0,"mu_eV":1.5,"eta_eV":0.0,
        "model":"periodic tilted two-band QWZ; mathematical fixture, not a fitted material Hamiltonian; generally breaks time reversal",
        "convention":"A=i<u|grad u>; Omega_xy=-2Im; D_x=int_BZ f d_kx Omega_xy d²k/(2pi)²; lattice Angstrom; no e,hbar,tau factors in Rustb output",
        "pointwise_Berry_max_abs_error":point_max_error,
        "mesh_checks":meshes,
        "field_permutations":{"ordered_xyx":ordered_xyx,"ordered_xxy":ordered_xxy,"symmetric_xyx":sym_xyx,"symmetric_xxy":sym_xxy},
        "controls":{"zero_tilt":zero,"tilt_reversal":reversal,"chirality_reversal":chirality,"lattice_2_3_angstrom":scale},
        "core_boundary":{"Dx_metres":dx_m,"tau_seconds":1e-13,"omega_rad_per_second":1e12,"peak_fields_V_per_m":[[100.0,20.0],[30.0,-10.0]],"peak_sheet_current_A_per_m":[[current[0].re,current[0].im],[current[1].re,current[1].im]],"mapping":"core[a][b][c] = SI_prefactor * Rustb[a][c][b] * 1e-10; both full contractions identical after symmetrization"},
        "limitations":["finite-temperature direct quadrature only; zero-temperature energy-cut unqualified","positive orthogonal lattices only","no material-specific, scattering, spin-current or full position-matrix qualification"]
    });
    report
}

fn main() {
    println!("{}", serde_json::to_string_pretty(&qualify()).unwrap());
}

#[cfg(test)]
mod tests {
    #[test]
    fn pinned_backend_matches_independent_eigensystem_and_dipole_integrals() {
        assert_eq!(super::qualify()["passed"], true);
    }
}
