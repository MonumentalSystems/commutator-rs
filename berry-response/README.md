# berry-response

Portable, checked reference contracts for the **intrinsic constant-relaxation-time
Berry-curvature-dipole response** and its conversion to a uniform-bar measurement.
This is an unpublished validation crate. It starts from an integrated dipole; it
does not calculate bands, import Wannier files, integrate a Brillouin zone, or
claim to reproduce a material's measured response.

## Conventions are inputs, not inferred metadata

- SI throughout: electric field V/m, time s, angular frequency rad/s, applied
  current A, and measured voltage V. The fundamental drive uses
  `E(t) = Re[E_peak exp(+i omega t)]`.
- Cartesian `x,y,z` is right handed, `epsilon_xyz = +1`, and elementary charge
  `e` is positive (electron charge `-e`).
- Canonical Berry connection is `A = i <u|grad_k u>`, hence
  `Omega_z = -2 Im <d_x u|d_y u>`. An imported opposite convention must be
  declared `BerryConvention::ConnectionMinusI`; the constructors negate it.
- `D[b][d] = sum_n integral d^dimension k/(2 pi)^dimension f_n d_b Omega_n,d`.
  The derivative index comes first. A **3D integrated dipole is dimensionless**;
  a **2D integrated dipole has length units**, represented here in metres.
  The 2D sheet is `xy` with normal `+z`, and stores `[D_xz,D_yz]`.
  A momentum-resolved Berry-curvature distribution or dipole integrand is not D.
- The external backend must handle its own `angstrom -> metre`, `eV -> joule`,
  k-space measure, spin degeneracy and layer/supercell normalization before
  construction. This crate does not guess whether those factors are present.

`Dipole2D` and `Dipole3D` are separate checked types. Their exported component
arrays are copies in the canonical convention, suitable for a backend-neutral
adapter or a versioned experiment record.

## Tensor contract

The current index is first. An ordered harmonic response is

```text
K(omega) = e^3 tau / [2 hbar^2 (1 + i omega tau)]
chi[a][b][c] = -K sum_d epsilon[a,d,c] D[b][d]
j_peak^(2omega)[a] = sum_bc chi[a][b][c] E_peak[b] E_peak[c]
```

The vector identity is `j = K E cross (D^T E)`. In 2D the same identity is
embedded in 3D with `D^T E = (0,0,D_xz E_x + D_yz E_y)`, then projected into
the sheet. Thus `D_xz > 0`, `E = (E_x,0)` implies `j_y = -K D_xz E_x^2`.

`FieldOrdering::Symmetric` stores `(chi[a][b][c]+chi[a][c][b])/2`.
Both representations contract **all** field permutations. Symmetrizing a tensor
does not justify doubling one unsymmetrized component. For
`D = diag(d,d,-2d)`, the ordered coefficients are `chi_yxz=Kd`, `chi_yzx=2Kd`,
and `j_y=3Kd E_x E_z`; the symmetric coefficients are each `3Kd/2`.

The harmonic factor `1/2` belongs to **peak input and peak output**. RMS
fundamental inputs are multiplied by `sqrt(2)` before the quadratic response;
RMS second-harmonic output is divided by `sqrt(2)` afterward. Consequently, using
the same numeric current input with RMS input and RMS output multiplies the
result by `sqrt(2)` relative to peak/peak, not by `1/2` or `2`.

`static_response` is a distinct true constant-field regime with coefficient
`e^3 tau / hbar^2` and no harmonic factor `1/2`. Its contraction rejects imaginary
field components. A harmonic tensor at `omega=0` is only an AC-limit tensor;
it remains half the constant-field coefficient. The complete harmonic-voltage
method rejects static tensors. AC rectification (`E E*`) and general mixed-
frequency susceptibility are not provided.

```rust
use berry_response::{
    Amplitude, BerryConvention, Complex64, Dipole2D, FieldOrdering,
    HallBar, LinearTransport, Relaxation,
};

let d = Dipole2D::new([1e-9, 0.0], BerryConvention::ConnectionI)?;
let chi = d.harmonic_response(
    Relaxation::new(1e-13, 2.0 * std::f64::consts::PI * 100.0)?,
    FieldOrdering::Ordered,
)?;
let transport = LinearTransport::sheet_conductance_siemens([
    [0.01, 0.0], [0.0, 0.02],
])?;
// x drive, 2 micrometre transverse width; +y lead is the positive lead.
let bar = HallBar::sheet(0, 2e-6, [0.0, 2e-6])?;
let voltage_rms = bar.second_harmonic_voltage(
    &transport, &chi, Complex64::new(1e-6, 0.0),
    Amplitude::Rms, Amplitude::Rms,
)?;
assert!(voltage_rms.re < 0.0);
# Ok::<(), berry_response::ResponseError>(())
```

## Geometry and open-circuit contract

`LinearTransport<2>` accepts **sheet conductance in S**, converting sheet
current A/m into electric field V/m. `LinearTransport<3>` accepts **bulk
conductivity in S/m**, converting bulk current A/m² into electric field V/m.
No implicit thickness conversion occurs. The full real symmetric
positive-definite matrix is inverted by a scaled Cholesky solve, retaining
anisotropic mixing. Exact symmetry is required; callers must explicitly decide
how to handle noisy or rounded asymmetric data.

`HallBar<2>` divides imposed current by transverse width. `HallBar<3>` divides
it by width times thickness; both dimensions span the plane perpendicular to
the selected drive axis. A signed Cartesian displacement runs from the negative
voltage lead to the positive lead. It need not be aligned with a coordinate axis.

```text
E_peak^(omega) = transport^-1 J_imposed_peak^(omega)
E_peak^(2omega) = -transport^-1 j_source_peak^(2omega)
V_positive - V_negative = -E_peak^(2omega) dot lead_displacement
```

At the second harmonic, **all** terminal current components are assumed zero.
The weak-response approximation evaluates the nonlinear source with the
fundamental field only. A single real frequency-independent transport matrix
is used at both frequencies. This is appropriate for a simple quasistatic
open-circuit reference, not a complex-conductivity or external-circuit solver.

An equivalent bulk-to-sheet reduction needs both `G_sheet = thickness * sigma`
and `D_sheet = thickness * D_bulk`, plus consistent geometry. Converting only
one silently introduces a thickness error. Neither reduction establishes that
a real multilayer system has the bulk band structure.

## Validation and limits

Constructors reject nonfinite data, nonpositive relaxation time and bar lengths,
negative angular frequency (including negative zero), invalid axes, coincident
voltage leads and non-SPD transport. Operations reject nonfinite intermediates
and outputs. Signed dipole entries, signed lead displacements and phasor phases
are allowed. Finite checks do not constitute conditioning/error estimates;
near-singular matrices or very small physical scales still need a numerical
convergence and uncertainty study.

Focused tests check the Te mixed-field factor, ordered/symmetric equivalence,
independent Cartesian cross products, transversality, isotropic and in-plane
zeros, 2D sign and Berry conversion, harmonic phase, static/AC separation,
anisotropic geometry, voltage lead reversal, sheet/bulk consistency, RMS
conversion and invalid/overflow inputs. These are floating-point reference
checks, not formal proofs, DFT replication or evidence of an experimental error.

The normalization under audit follows the current-first tensor form in
[Sodemann and Fu, PRL 115, 216806 (2015)](https://arxiv.org/abs/1508.00571).
The cross-axis Te contraction can also be checked against Appendix B of
[Tsirkin et al., PRB 97, 035158 (2018)](https://arxiv.org/abs/1710.03204).
This crate freezes one explicit convention to compare implementations; it does
not silently reconcile mutually inconsistent printed tensor/vector equations.
