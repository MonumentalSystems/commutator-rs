use crate::{Complex64, MagnetismError, Result};

/// An oriented two-site spin-1/2 interaction.
///
/// The bond contributes anisotropic exchange
/// `sum_a J_a S_i^a S_j^a` and the oriented DM term
/// `D · (S_i × S_j)`. Reversing `i` and `j` therefore requires negating
/// [`dzyaloshinskii_moriya`](Self::dzyaloshinskii_moriya).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bond {
    /// First oriented endpoint.
    pub i: usize,
    /// Second oriented endpoint.
    pub j: usize,
    /// Exchange couplings `[Jx, Jy, Jz]`.
    pub exchange: [f64; 3],
    /// Dzyaloshinskii–Moriya vector `[Dx, Dy, Dz]`.
    pub dzyaloshinskii_moriya: [f64; 3],
}

impl Bond {
    /// Construct an anisotropic exchange bond without DM coupling.
    pub const fn xyz(i: usize, j: usize, jx: f64, jy: f64, jz: f64) -> Self {
        Self {
            i,
            j,
            exchange: [jx, jy, jz],
            dzyaloshinskii_moriya: [0.0; 3],
        }
    }

    /// Construct an isotropic Heisenberg bond with `Jx = Jy = Jz = coupling`.
    pub const fn heisenberg(i: usize, j: usize, coupling: f64) -> Self {
        Self::xyz(i, j, coupling, coupling, coupling)
    }

    /// Attach an oriented Dzyaloshinskii–Moriya vector.
    pub const fn with_dm(mut self, dm: [f64; 3]) -> Self {
        self.dzyaloshinskii_moriya = dm;
        self
    }
}

/// Builder for a checked finite [`SpinModel`].
#[derive(Debug, Clone)]
pub struct SpinModelBuilder {
    spins: usize,
    bonds: Vec<Bond>,
    fields: Vec<f64>,
    invalid_field_site: Option<usize>,
}

impl SpinModelBuilder {
    pub(crate) fn new(spins: usize) -> Self {
        Self {
            spins,
            bonds: Vec::new(),
            fields: vec![0.0; spins],
            invalid_field_site: None,
        }
    }

    /// Append a bond. Bonds may repeat, in which case their contributions add.
    pub fn bond(mut self, bond: Bond) -> Self {
        self.bonds.push(bond);
        self
    }

    /// Append several bonds.
    pub fn bonds(mut self, bonds: impl IntoIterator<Item = Bond>) -> Self {
        self.bonds.extend(bonds);
        self
    }

    /// Set the longitudinal field `h_i` in the term `-h_i S_i^z`.
    ///
    /// Validation is deferred to [`build`](Self::build).
    pub fn longitudinal_field(mut self, site: usize, field: f64) -> Self {
        if let Some(value) = self.fields.get_mut(site) {
            *value = field;
        } else {
            self.invalid_field_site = Some(site);
        }
        self
    }

    /// Replace all longitudinal fields in site order.
    pub fn longitudinal_fields(mut self, fields: impl IntoIterator<Item = f64>) -> Self {
        self.fields = fields.into_iter().collect();
        self.invalid_field_site = None;
        self
    }

    /// Validate all graph data and construct the model.
    pub fn build(self) -> Result<SpinModel> {
        if let Some(site) = self.invalid_field_site {
            return Err(MagnetismError::SiteOutOfBounds {
                site,
                spins: self.spins,
            });
        }
        SpinModel::try_new(self.spins, self.bonds, self.fields)
    }
}

/// A checked finite spin-1/2 graph Hamiltonian.
///
/// Computational-basis index bit `i = 1` means spin-up at site `i`, while
/// bit `0` means spin-down. Every field `h_i` contributes `-h_i S_i^z`.
/// The exact Hilbert-space dimension is `2^spins`.
#[derive(Debug, Clone, PartialEq)]
pub struct SpinModel {
    spins: usize,
    dimension: usize,
    bonds: Vec<Bond>,
    fields: Vec<f64>,
}

impl SpinModel {
    /// Start constructing a model with `spins` spin-1/2 sites.
    pub fn builder(spins: usize) -> SpinModelBuilder {
        SpinModelBuilder::new(spins)
    }

    /// Construct and validate a model from its graph data.
    pub fn try_new(spins: usize, bonds: Vec<Bond>, fields: Vec<f64>) -> Result<Self> {
        if spins == 0 {
            return Err(MagnetismError::EmptyModel);
        }
        let dimension = 1usize
            .checked_shl(spins as u32)
            .ok_or(MagnetismError::HilbertSpaceOverflow { spins })?;
        if fields.len() != spins {
            return Err(MagnetismError::Shape {
                name: "longitudinal fields",
                expected: spins,
                actual: fields.len(),
            });
        }
        if !fields.iter().all(|value| value.is_finite()) {
            return Err(MagnetismError::NonFinite("longitudinal fields"));
        }
        for bond in &bonds {
            for site in [bond.i, bond.j] {
                if site >= spins {
                    return Err(MagnetismError::SiteOutOfBounds { site, spins });
                }
            }
            if bond.i == bond.j {
                return Err(MagnetismError::SelfBond { site: bond.i });
            }
            if !bond
                .exchange
                .iter()
                .chain(&bond.dzyaloshinskii_moriya)
                .all(|value| value.is_finite())
            {
                return Err(MagnetismError::NonFinite("bond couplings"));
            }
        }
        Ok(Self {
            spins,
            dimension,
            bonds,
            fields,
        })
    }

    /// Return the number of sites.
    pub const fn spins(&self) -> usize {
        self.spins
    }

    /// Return the exact Hilbert-space dimension `2^spins`.
    pub const fn hilbert_dimension(&self) -> usize {
        self.dimension
    }

    /// Return the validated bonds in insertion order.
    pub fn bonds(&self) -> &[Bond] {
        &self.bonds
    }

    /// Return the longitudinal fields in site order.
    pub fn longitudinal_fields(&self) -> &[f64] {
        &self.fields
    }

    pub(crate) fn validate_state(&self, name: &'static str, state: &[Complex64]) -> Result<f64> {
        if state.len() != self.dimension {
            return Err(MagnetismError::Shape {
                name,
                expected: self.dimension,
                actual: state.len(),
            });
        }
        if !state.iter().all(|value| value.is_finite()) {
            return Err(MagnetismError::NonFinite(name));
        }
        let norm: f64 = state.iter().map(|value| value.norm_sqr()).sum();
        if !norm.is_finite() {
            return Err(MagnetismError::NonFinite("state norm"));
        }
        if norm <= 0.0 {
            return Err(MagnetismError::ZeroNorm);
        }
        Ok(norm)
    }

    /// Apply the Hamiltonian into a caller-owned output buffer.
    ///
    /// This operation does not assemble or store a `2^N × 2^N` matrix. The
    /// input and output must not alias and the output is overwritten.
    pub fn apply(&self, state: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        self.validate_state("state", state)?;
        if output.len() != self.dimension {
            return Err(MagnetismError::Shape {
                name: "Hamiltonian output",
                expected: self.dimension,
                actual: output.len(),
            });
        }
        output.fill(Complex64::ZERO);

        for (basis, amplitude) in state.iter().copied().enumerate() {
            let mut diagonal = 0.0;
            for (site, field) in self.fields.iter().copied().enumerate() {
                diagonal -= field * spin_sign(basis, site) * 0.5;
            }
            for bond in &self.bonds {
                let si = spin_sign(basis, bond.i);
                let sj = spin_sign(basis, bond.j);
                let [jx, jy, jz] = bond.exchange;
                let [dx, dy, dz] = bond.dzyaloshinskii_moriya;
                diagonal += 0.25 * jz * si * sj;

                let flip_both = basis ^ (1usize << bond.i) ^ (1usize << bond.j);
                let exchange_flip = 0.25 * (jx - jy * si * sj);
                let dm_z_flip = Complex64::new(0.0, 0.25 * dz * (si - sj));
                output[flip_both] += amplitude * (Complex64::from(exchange_flip) + dm_z_flip);

                let flip_i = basis ^ (1usize << bond.i);
                let dm_i = Complex64::new(-0.25 * dy * sj, -0.25 * dx * si * sj);
                output[flip_i] += amplitude * dm_i;

                let flip_j = basis ^ (1usize << bond.j);
                let dm_j = Complex64::new(0.25 * dy * si, 0.25 * dx * si * sj);
                output[flip_j] += amplitude * dm_j;
            }
            output[basis] += amplitude * diagonal;
        }
        if output.iter().all(|value| value.is_finite()) {
            Ok(())
        } else {
            Err(MagnetismError::NumericalFailure(
                "Hamiltonian application overflowed",
            ))
        }
    }

    /// Allocate and return `H|state>`.
    pub fn applied(&self, state: &[Complex64]) -> Result<Vec<Complex64>> {
        let mut output = vec![Complex64::ZERO; self.dimension];
        self.apply(state, &mut output)?;
        Ok(output)
    }
}

pub(crate) fn spin_sign(basis: usize, site: usize) -> f64 {
    if basis & (1usize << site) == 0 {
        -1.0
    } else {
        1.0
    }
}
