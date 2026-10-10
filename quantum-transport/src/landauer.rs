use crate::{Result, TransportError, ELEMENTARY_CHARGE_COULOMBS, PLANCK_CONSTANT_JOULE_SECONDS};

/// Equilibrium reservoir parameters in the same energy units as the grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reservoir {
    /// Chemical potential.
    pub chemical_potential: f64,
    /// Thermal energy `k_B T`, not temperature in kelvin.
    pub thermal_energy: f64,
}

impl Reservoir {
    /// Construct checked reservoir parameters.
    pub fn try_new(chemical_potential: f64, thermal_energy: f64) -> Result<Self> {
        if !chemical_potential.is_finite() || !thermal_energy.is_finite() {
            return Err(TransportError::NonFinite("reservoir parameters"));
        }
        if thermal_energy < 0.0 {
            return Err(TransportError::NegativeParameter("thermal energy"));
        }
        Ok(Self {
            chemical_potential,
            thermal_energy,
        })
    }

    fn occupation(self, energy: f64) -> Result<f64> {
        fermi_function(energy, self.chemical_potential, self.thermal_energy)
    }
}

/// Evaluate the Fermi–Dirac occupation with stable tails.
///
/// `thermal_energy` is `k_B T` in the same units as `energy` and
/// `chemical_potential`. At zero temperature the value is `1`, `1/2`, or `0`
/// below, exactly at, or above the chemical potential.
pub fn fermi_function(energy: f64, chemical_potential: f64, thermal_energy: f64) -> Result<f64> {
    if !energy.is_finite() || !chemical_potential.is_finite() || !thermal_energy.is_finite() {
        return Err(TransportError::NonFinite("Fermi-function inputs"));
    }
    if thermal_energy < 0.0 {
        return Err(TransportError::NegativeParameter("thermal energy"));
    }
    if thermal_energy == 0.0 {
        return Ok(if energy < chemical_potential {
            1.0
        } else if energy > chemical_potential {
            0.0
        } else {
            0.5
        });
    }
    let scaled = (energy - chemical_potential) / thermal_energy;
    Ok(if scaled >= 0.0 {
        let tail = (-scaled).exp();
        tail / (1.0 + tail)
    } else {
        1.0 / (1.0 + scaled.exp())
    })
}

/// Integrate `T(E) [f_L(E)-f_R(E)]` with the trapezoidal rule.
///
/// Energies must be finite and strictly increasing. Transmission values must
/// be finite and nonnegative. The result has the energy units of the grid.
pub fn landauer_integral(
    energies: &[f64],
    transmissions: &[f64],
    left: Reservoir,
    right: Reservoir,
) -> Result<f64> {
    if energies.len() < 2 {
        return Err(TransportError::EnergyGridTooShort);
    }
    if transmissions.len() != energies.len() {
        return Err(TransportError::LengthMismatch {
            expected: energies.len(),
            actual: transmissions.len(),
        });
    }
    if !energies.iter().all(|value| value.is_finite())
        || !transmissions.iter().all(|value| value.is_finite())
    {
        return Err(TransportError::NonFinite("Landauer grid"));
    }
    if transmissions.iter().any(|value| *value < 0.0) {
        return Err(TransportError::NegativeTransmission {
            value: transmissions
                .iter()
                .copied()
                .find(|value| *value < 0.0)
                .expect("negative value exists"),
        });
    }
    let mut integrand = Vec::with_capacity(energies.len());
    for (&energy, &transmission) in energies.iter().zip(transmissions) {
        integrand.push(transmission * (left.occupation(energy)? - right.occupation(energy)?));
    }
    let mut integral = 0.0;
    for index in 0..energies.len() - 1 {
        let width = energies[index + 1] - energies[index];
        if width <= 0.0 {
            return Err(TransportError::UnorderedEnergyGrid { index: index + 1 });
        }
        integral += 0.5 * width * (integrand[index] + integrand[index + 1]);
    }
    if integral.is_finite() {
        Ok(integral)
    } else {
        Err(TransportError::NonFinite("Landauer integral"))
    }
}

/// Return the Landauer current in amperes for an electron-volt energy grid.
///
/// The conversion is `I = g e^2/h integral_eV`. `degeneracy` is commonly
/// `2` for spin-degenerate transport and `1` when spin is represented
/// explicitly.
pub fn landauer_current_ev(
    energies_ev: &[f64],
    transmissions: &[f64],
    left: Reservoir,
    right: Reservoir,
    degeneracy: f64,
) -> Result<f64> {
    if !degeneracy.is_finite() {
        return Err(TransportError::NonFinite("transport degeneracy"));
    }
    if degeneracy < 0.0 {
        return Err(TransportError::NegativeParameter("transport degeneracy"));
    }
    let integral_ev = landauer_integral(energies_ev, transmissions, left, right)?;
    Ok(
        degeneracy * ELEMENTARY_CHARGE_COULOMBS.powi(2) / PLANCK_CONSTANT_JOULE_SECONDS
            * integral_ev,
    )
}
