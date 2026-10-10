//! Validated eigensystems and superconducting observables.

use crate::{orbital_index, BdGMatrix, Complex64, Result, Spin, SuperconductivityError};

/// A complete, validated orthonormal eigensystem of a BdG Hamiltonian.
///
/// Eigenvectors are stored column-major: all components of eigenvector zero,
/// followed by all components of eigenvector one, and so on.
#[derive(Clone, Debug, PartialEq)]
pub struct BdGEigensystem {
    site_count: usize,
    energies: Vec<f64>,
    eigenvectors: Vec<Complex64>,
}

impl BdGEigensystem {
    /// Validates a complete set of eigenvalues and column-major eigenvectors.
    ///
    /// `tolerance` is an absolute threshold applied to vector
    /// orthonormality and to every component of `H psi - E psi`.
    pub fn try_new(
        matrix: &BdGMatrix,
        energies: Vec<f64>,
        eigenvectors: Vec<Complex64>,
        tolerance: f64,
    ) -> Result<Self> {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(SuperconductivityError::InvalidPositiveParameter {
                parameter: "eigenpair tolerance",
            });
        }
        let n = matrix.dimension();
        if energies.len() != n {
            return Err(SuperconductivityError::DimensionMismatch {
                context: "eigenvalues",
                expected: n,
                actual: energies.len(),
            });
        }
        if eigenvectors.len() != n * n {
            return Err(SuperconductivityError::DimensionMismatch {
                context: "eigenvectors",
                expected: n * n,
                actual: eigenvectors.len(),
            });
        }
        if let Some(index) = energies.iter().position(|energy| !energy.is_finite()) {
            return Err(SuperconductivityError::NonFinite {
                context: "eigenvalues",
                index,
            });
        }
        if let Some(index) = eigenvectors.iter().position(|entry| !entry.is_finite()) {
            return Err(SuperconductivityError::NonFinite {
                context: "eigenvectors",
                index,
            });
        }

        let mut orthonormality_residual: f64 = 0.0;
        for left in 0..n {
            for right in 0..n {
                let mut inner = Complex64::ZERO;
                for component in 0..n {
                    inner += eigenvectors[left * n + component].conj()
                        * eigenvectors[right * n + component];
                }
                let expected = if left == right {
                    Complex64::ONE
                } else {
                    Complex64::ZERO
                };
                orthonormality_residual = orthonormality_residual.max((inner - expected).norm());
            }
        }
        if orthonormality_residual > tolerance {
            return Err(SuperconductivityError::NonOrthonormalEigenvectors {
                residual: orthonormality_residual,
                tolerance,
            });
        }

        for eigenpair in 0..n {
            let vector = &eigenvectors[eigenpair * n..(eigenpair + 1) * n];
            let product = matrix.multiply(vector);
            let mut residual: f64 = 0.0;
            for component in 0..n {
                residual = residual
                    .max((product[component] - vector[component] * energies[eigenpair]).norm());
            }
            if residual > tolerance {
                return Err(SuperconductivityError::InvalidEigenpair {
                    eigenpair,
                    residual,
                    tolerance,
                });
            }
        }

        Ok(Self {
            site_count: matrix.site_count(),
            energies,
            eigenvectors,
        })
    }

    /// Returns all eigenvalues in the caller-supplied order.
    pub fn energies(&self) -> &[f64] {
        &self.energies
    }

    /// Returns the full Nambu dimension.
    pub fn dimension(&self) -> usize {
        self.energies.len()
    }

    /// Returns one component of an eigenvector.
    ///
    /// # Panics
    ///
    /// Panics when either index is out of bounds.
    pub fn eigenvector_component(&self, eigenpair: usize, component: usize) -> Complex64 {
        self.eigenvectors[eigenpair * self.dimension() + component]
    }

    /// Computes the spin-summed electron local density of states at one site.
    ///
    /// The delta peaks are represented by normalized Lorentzians with
    /// half-width `broadening`. Because the complete positive and negative
    /// eigenspectrum is used, only particle-sector spectral weights enter.
    pub fn local_density_of_states(
        &self,
        site: usize,
        energy: f64,
        broadening: f64,
    ) -> Result<f64> {
        self.check_site(site)?;
        if !energy.is_finite() {
            return Err(SuperconductivityError::NonFinite {
                context: "LDOS energy",
                index: 0,
            });
        }
        if !broadening.is_finite() || broadening <= 0.0 {
            return Err(SuperconductivityError::InvalidPositiveParameter {
                parameter: "LDOS broadening",
            });
        }
        let mut density = 0.0;
        for (eigenpair, &eigenvalue) in self.energies.iter().enumerate() {
            let mut weight = 0.0;
            for spin in [Spin::Up, Spin::Down] {
                weight += self
                    .eigenvector_component(eigenpair, orbital_index(site, spin))
                    .norm_sqr();
            }
            let offset = energy - eigenvalue;
            let lorentzian = broadening
                / (core::f64::consts::PI * (offset.mul_add(offset, broadening * broadening)));
            density += weight * lorentzian;
        }
        Ok(density)
    }

    /// Computes the onsite spin-singlet anomalous amplitude.
    ///
    /// For positive-energy modes this evaluates
    /// `(F_up,down - F_down,up) / 2`, where
    /// `F_a,b = sum_n u_(a,n) v*_(b,n) tanh(E_n / (2 T))`.
    /// At zero temperature, the thermal factor is one. Zero modes are omitted
    /// because their occupation is convention-dependent.
    pub fn singlet_pair_amplitude(&self, site: usize, temperature: f64) -> Result<Complex64> {
        self.check_site(site)?;
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(SuperconductivityError::InvalidTemperature);
        }
        let orbitals = self.site_count * 2;
        let up = orbital_index(site, Spin::Up);
        let down = orbital_index(site, Spin::Down);
        let mut up_down = Complex64::ZERO;
        let mut down_up = Complex64::ZERO;
        for (eigenpair, &energy) in self.energies.iter().enumerate() {
            if energy <= 0.0 {
                continue;
            }
            let thermal = if temperature == 0.0 {
                1.0
            } else {
                (energy / (2.0 * temperature)).tanh()
            };
            let u_up = self.eigenvector_component(eigenpair, up);
            let u_down = self.eigenvector_component(eigenpair, down);
            let v_up = self.eigenvector_component(eigenpair, orbitals + up).conj();
            let v_down = self
                .eigenvector_component(eigenpair, orbitals + down)
                .conj();
            up_down += u_up * v_down * thermal;
            down_up += u_down * v_up * thermal;
        }
        Ok((up_down - down_up) * 0.5)
    }

    fn check_site(&self, site: usize) -> Result<()> {
        if site >= self.site_count {
            Err(SuperconductivityError::SiteOutOfBounds {
                site,
                site_count: self.site_count,
            })
        } else {
            Ok(())
        }
    }
}
