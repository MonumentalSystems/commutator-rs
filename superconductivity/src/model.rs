//! Checked spinful tight-binding and onsite s-wave pairing models.

use crate::{BdGMatrix, Complex64, Result, SuperconductivityError};

/// Spin label within a lattice site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Spin {
    /// Spin-up orbital.
    Up,
    /// Spin-down orbital.
    Down,
}

impl Spin {
    const fn offset(self) -> usize {
        match self {
            Self::Up => 0,
            Self::Down => 1,
        }
    }
}

/// Returns the flat spin-orbital index for a site.
pub const fn orbital_index(site: usize, spin: Spin) -> usize {
    site * 2 + spin.offset()
}

/// A spin-independent hopping used by [`NormalHamiltonian::spin_independent`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hopping {
    /// First endpoint of the undirected hopping.
    pub from: usize,
    /// Second endpoint of the undirected hopping.
    pub to: usize,
    /// Matrix element `h[from, to]`; the reverse receives its conjugate.
    pub amplitude: Complex64,
}

/// A checked dense spinful normal-state Hamiltonian.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalHamiltonian {
    site_count: usize,
    entries: Vec<Complex64>,
}

impl NormalHamiltonian {
    /// Builds a normal Hamiltonian from a row-major spin-orbital matrix.
    ///
    /// The matrix dimension is `2 * site_count`. Hermiticity is validated at
    /// the supplied absolute element-wise tolerance.
    pub fn try_from_dense(
        site_count: usize,
        entries: Vec<Complex64>,
        hermiticity_tolerance: f64,
    ) -> Result<Self> {
        if site_count == 0 {
            return Err(SuperconductivityError::EmptyLattice);
        }
        if !hermiticity_tolerance.is_finite() || hermiticity_tolerance < 0.0 {
            return Err(SuperconductivityError::InvalidTolerance {
                parameter: "hermiticity_tolerance",
            });
        }
        let orbitals =
            site_count
                .checked_mul(2)
                .ok_or(SuperconductivityError::DimensionMismatch {
                    context: "normal Hamiltonian",
                    expected: usize::MAX,
                    actual: entries.len(),
                })?;
        let expected =
            orbitals
                .checked_mul(orbitals)
                .ok_or(SuperconductivityError::DimensionMismatch {
                    context: "normal Hamiltonian",
                    expected: usize::MAX,
                    actual: entries.len(),
                })?;
        if entries.len() != expected {
            return Err(SuperconductivityError::DimensionMismatch {
                context: "normal Hamiltonian",
                expected,
                actual: entries.len(),
            });
        }
        if let Some(index) = entries.iter().position(|entry| !entry.is_finite()) {
            return Err(SuperconductivityError::NonFinite {
                context: "normal Hamiltonian",
                index,
            });
        }
        let mut residual: f64 = 0.0;
        for row in 0..orbitals {
            for column in 0..orbitals {
                residual = residual.max(
                    (entries[row * orbitals + column] - entries[column * orbitals + row].conj())
                        .norm(),
                );
            }
        }
        if residual > hermiticity_tolerance {
            return Err(SuperconductivityError::NotHermitian {
                residual,
                tolerance: hermiticity_tolerance,
            });
        }
        Ok(Self {
            site_count,
            entries,
        })
    }

    /// Builds a spin-independent model from site energies and hoppings.
    ///
    /// Each site energy is copied to both spins. Every hopping is added to
    /// both spin sectors along with its Hermitian conjugate. Repeated edges
    /// accumulate, which makes independent physical hopping channels easy to
    /// compose.
    pub fn spin_independent(site_energies: &[f64], hoppings: &[Hopping]) -> Result<Self> {
        let site_count = site_energies.len();
        if site_count == 0 {
            return Err(SuperconductivityError::EmptyLattice);
        }
        let orbitals = site_count * 2;
        let mut entries = vec![Complex64::ZERO; orbitals * orbitals];
        for (site, &energy) in site_energies.iter().enumerate() {
            if !energy.is_finite() {
                return Err(SuperconductivityError::NonFinite {
                    context: "site energies",
                    index: site,
                });
            }
            for spin in [Spin::Up, Spin::Down] {
                let orbital = orbital_index(site, spin);
                entries[orbital * orbitals + orbital] = energy.into();
            }
        }
        for (index, hopping) in hoppings.iter().enumerate() {
            if hopping.from >= site_count {
                return Err(SuperconductivityError::SiteOutOfBounds {
                    site: hopping.from,
                    site_count,
                });
            }
            if hopping.to >= site_count {
                return Err(SuperconductivityError::SiteOutOfBounds {
                    site: hopping.to,
                    site_count,
                });
            }
            if hopping.from == hopping.to {
                return Err(SuperconductivityError::SelfHopping { site: hopping.from });
            }
            if !hopping.amplitude.is_finite() {
                return Err(SuperconductivityError::NonFinite {
                    context: "hoppings",
                    index,
                });
            }
            for spin in [Spin::Up, Spin::Down] {
                let from = orbital_index(hopping.from, spin);
                let to = orbital_index(hopping.to, spin);
                entries[from * orbitals + to] += hopping.amplitude;
                entries[to * orbitals + from] += hopping.amplitude.conj();
            }
        }
        Self::try_from_dense(site_count, entries, 0.0)
    }

    /// Returns the number of lattice sites.
    pub fn site_count(&self) -> usize {
        self.site_count
    }

    /// Returns the number of spin orbitals.
    pub fn orbital_count(&self) -> usize {
        self.site_count * 2
    }

    /// Returns a normal-state matrix element.
    ///
    /// # Panics
    ///
    /// Panics when either orbital index is out of bounds.
    pub fn get(&self, row: usize, column: usize) -> Complex64 {
        self.entries[row * self.orbital_count() + column]
    }

    /// Returns all row-major normal-state matrix entries.
    pub fn as_slice(&self) -> &[Complex64] {
        &self.entries
    }
}

/// A checked spinful BdG model with onsite spin-singlet pairing.
#[derive(Clone, Debug, PartialEq)]
pub struct OnsiteSWaveModel {
    normal: NormalHamiltonian,
    chemical_potential: f64,
    gaps: Vec<Complex64>,
}

impl OnsiteSWaveModel {
    /// Creates a model from a normal Hamiltonian, chemical potential, and one
    /// complex pairing gap per site.
    pub fn try_new(
        normal: NormalHamiltonian,
        chemical_potential: f64,
        gaps: Vec<Complex64>,
    ) -> Result<Self> {
        if !chemical_potential.is_finite() {
            return Err(SuperconductivityError::NonFinite {
                context: "chemical potential",
                index: 0,
            });
        }
        if gaps.len() != normal.site_count() {
            return Err(SuperconductivityError::DimensionMismatch {
                context: "onsite pairing gaps",
                expected: normal.site_count(),
                actual: gaps.len(),
            });
        }
        if let Some(index) = gaps.iter().position(|gap| !gap.is_finite()) {
            return Err(SuperconductivityError::NonFinite {
                context: "onsite pairing gaps",
                index,
            });
        }
        Ok(Self {
            normal,
            chemical_potential,
            gaps,
        })
    }

    /// Returns the normal-state Hamiltonian.
    pub fn normal(&self) -> &NormalHamiltonian {
        &self.normal
    }

    /// Returns the chemical potential subtracted from the normal-state block.
    pub fn chemical_potential(&self) -> f64 {
        self.chemical_potential
    }

    /// Returns the complex onsite pairing gaps.
    pub fn gaps(&self) -> &[Complex64] {
        &self.gaps
    }

    /// Constructs the dense Hermitian BdG Hamiltonian.
    pub fn hamiltonian(&self) -> BdGMatrix {
        let orbitals = self.normal.orbital_count();
        let n = orbitals * 2;
        let mut entries = vec![Complex64::ZERO; n * n];

        for row in 0..orbitals {
            for column in 0..orbitals {
                let mut normal = self.normal.get(row, column);
                if row == column {
                    normal -= self.chemical_potential.into();
                }
                entries[row * n + column] = normal;
                entries[(row + orbitals) * n + column + orbitals] = -self.normal.get(column, row)
                    + if row == column {
                        self.chemical_potential.into()
                    } else {
                        Complex64::ZERO
                    };
            }
        }

        for (site, &gap) in self.gaps.iter().enumerate() {
            let up = orbital_index(site, Spin::Up);
            let down = orbital_index(site, Spin::Down);
            entries[up * n + down + orbitals] = gap;
            entries[down * n + up + orbitals] = -gap;
            entries[(down + orbitals) * n + up] = gap.conj();
            entries[(up + orbitals) * n + down] = -gap.conj();
        }

        BdGMatrix {
            site_count: self.normal.site_count(),
            entries,
        }
    }
}
