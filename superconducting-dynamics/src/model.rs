//! Checked graph, order parameter, energy, gradient, current, and gauge tools.

use crate::{Complex64, DynamicsError, Result};

/// Local Ginzburg--Landau coefficients and dissipative mobility.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SiteParameters {
    /// Quadratic free-energy coefficient `alpha`.
    pub(crate) alpha: f64,
    /// Positive quartic coefficient `beta` in `beta |psi|^4 / 2`.
    pub(crate) beta: f64,
    /// Positive TDGL mobility `Gamma`, where `d psi / dt = -Gamma dF/dpsi*`.
    pub(crate) mobility: f64,
}

impl SiteParameters {
    /// Creates checked local coefficients.
    pub fn try_new(alpha: f64, beta: f64, mobility: f64) -> Result<Self> {
        if !alpha.is_finite() {
            return Err(DynamicsError::NonFinite {
                context: "alpha",
                index: 0,
            });
        }
        if !beta.is_finite() || beta <= 0.0 {
            return Err(DynamicsError::InvalidPositiveParameter { parameter: "beta" });
        }
        if !mobility.is_finite() || mobility <= 0.0 {
            return Err(DynamicsError::InvalidPositiveParameter {
                parameter: "mobility",
            });
        }
        Ok(Self {
            alpha,
            beta,
            mobility,
        })
    }

    /// Returns the quadratic free-energy coefficient.
    pub fn alpha(self) -> f64 {
        self.alpha
    }

    /// Returns the positive quartic free-energy coefficient.
    pub fn beta(self) -> f64 {
        self.beta
    }

    /// Returns the positive dissipative TDGL mobility.
    pub fn mobility(self) -> f64 {
        self.mobility
    }
}

/// An oriented, gauge-covariant graph link.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaugeLink {
    /// Tail of the stored orientation.
    pub(crate) from: usize,
    /// Head of the stored orientation.
    pub(crate) to: usize,
    /// Positive phase stiffness multiplying the squared covariant difference.
    pub(crate) stiffness: f64,
    /// Dimensionless line-integrated vector-potential phase `A_from,to`.
    pub(crate) phase: f64,
}

impl GaugeLink {
    /// Creates a link. Endpoint bounds are checked when constructing a model.
    pub fn try_new(from: usize, to: usize, stiffness: f64, phase: f64) -> Result<Self> {
        if from == to {
            return Err(DynamicsError::SelfLink { site: from });
        }
        if !stiffness.is_finite() || stiffness <= 0.0 {
            return Err(DynamicsError::InvalidPositiveParameter {
                parameter: "link stiffness",
            });
        }
        if !phase.is_finite() {
            return Err(DynamicsError::NonFinite {
                context: "link phase",
                index: 0,
            });
        }
        Ok(Self {
            from,
            to,
            stiffness,
            phase,
        })
    }

    /// Returns the tail of the stored orientation.
    pub fn from(self) -> usize {
        self.from
    }

    /// Returns the head of the stored orientation.
    pub fn to(self) -> usize {
        self.to
    }

    /// Returns the positive link stiffness.
    pub fn stiffness(self) -> f64 {
        self.stiffness
    }

    /// Returns the dimensionless oriented link phase.
    pub fn phase(self) -> f64 {
        self.phase
    }

    fn transporter(self) -> Complex64 {
        Complex64::from_polar(1.0, self.phase)
    }
}

/// Direction in which a stored graph link is traversed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LinkDirection {
    /// Traverse from the link's `from` endpoint to its `to` endpoint.
    Forward,
    /// Traverse opposite to the stored orientation.
    Reverse,
}

/// One oriented edge in a closed-loop flux query.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoopEdge {
    /// Index into [`TdglModel::links`].
    pub link: usize,
    /// Traversal direction.
    pub direction: LinkDirection,
}

impl LoopEdge {
    /// Creates a forward traversal.
    pub const fn forward(link: usize) -> Self {
        Self {
            link,
            direction: LinkDirection::Forward,
        }
    }

    /// Creates a reverse traversal.
    pub const fn reverse(link: usize) -> Self {
        Self {
            link,
            direction: LinkDirection::Reverse,
        }
    }
}

/// A checked complex superconducting order parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct OrderParameter {
    values: Vec<Complex64>,
}

impl OrderParameter {
    /// Creates a state with one finite complex value per model site.
    pub fn try_new(model: &TdglModel, values: Vec<Complex64>) -> Result<Self> {
        model.validate_values(&values)?;
        Ok(Self { values })
    }

    /// Creates a spatially uniform state.
    pub fn uniform(model: &TdglModel, value: Complex64) -> Result<Self> {
        Self::try_new(model, vec![value; model.site_count()])
    }

    /// Returns the site amplitudes and phases as complex values.
    pub fn as_slice(&self) -> &[Complex64] {
        &self.values
    }

    /// Returns the number of sites.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether the state contains no sites.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// A finite-graph Ginzburg--Landau model with compact U(1) link phases.
#[derive(Clone, Debug, PartialEq)]
pub struct TdglModel {
    sites: Vec<SiteParameters>,
    links: Vec<GaugeLink>,
}

impl TdglModel {
    /// Builds a checked graph model.
    ///
    /// Parallel links are allowed and represent independent coupling channels.
    pub fn try_new(sites: Vec<SiteParameters>, links: Vec<GaugeLink>) -> Result<Self> {
        if sites.is_empty() {
            return Err(DynamicsError::EmptyGraph);
        }
        for link in &links {
            for site in [link.from, link.to] {
                if site >= sites.len() {
                    return Err(DynamicsError::SiteOutOfBounds {
                        site,
                        site_count: sites.len(),
                    });
                }
            }
        }
        Ok(Self { sites, links })
    }

    /// Builds an open-boundary rectangular nearest-neighbor graph.
    ///
    /// Sites use row-major order. Horizontal links point right and vertical
    /// links point down; every initial link phase is zero.
    pub fn uniform_rectangular(
        width: usize,
        height: usize,
        site: SiteParameters,
        stiffness: f64,
    ) -> Result<Self> {
        let count = width
            .checked_mul(height)
            .ok_or(DynamicsError::SizeOverflow)?;
        if count == 0 {
            return Err(DynamicsError::EmptyGraph);
        }
        if !stiffness.is_finite() || stiffness <= 0.0 {
            return Err(DynamicsError::InvalidPositiveParameter {
                parameter: "link stiffness",
            });
        }
        let mut links = Vec::new();
        for row in 0..height {
            for column in 0..width {
                let from = row * width + column;
                if column + 1 < width {
                    links.push(GaugeLink::try_new(from, from + 1, stiffness, 0.0)?);
                }
                if row + 1 < height {
                    links.push(GaugeLink::try_new(from, from + width, stiffness, 0.0)?);
                }
            }
        }
        Self::try_new(vec![site; count], links)
    }

    /// Returns local coefficients in site order.
    pub fn sites(&self) -> &[SiteParameters] {
        &self.sites
    }

    /// Returns oriented gauge links.
    pub fn links(&self) -> &[GaugeLink] {
        &self.links
    }

    /// Returns the number of sites.
    pub fn site_count(&self) -> usize {
        self.sites.len()
    }

    /// Returns the number of links.
    pub fn link_count(&self) -> usize {
        self.links.len()
    }

    pub(crate) fn validate_values(&self, values: &[Complex64]) -> Result<()> {
        if values.len() != self.site_count() {
            return Err(DynamicsError::DimensionMismatch {
                context: "order parameter",
                expected: self.site_count(),
                actual: values.len(),
            });
        }
        if let Some(index) = values
            .iter()
            .position(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(DynamicsError::NonFinite {
                context: "order parameter",
                index,
            });
        }
        Ok(())
    }

    /// Evaluates the free energy
    /// `sum_i(alpha_i |psi_i|^2 + beta_i |psi_i|^4/2)
    /// + sum_ij K_ij |psi_j - exp(i A_ij) psi_i|^2`.
    pub fn free_energy(&self, state: &OrderParameter) -> Result<f64> {
        self.validate_values(state.as_slice())?;
        let mut energy = 0.0;
        for (&psi, site) in state.as_slice().iter().zip(&self.sites) {
            let norm = psi.norm_sqr();
            energy += site.alpha * norm + 0.5 * site.beta * norm * norm;
        }
        for link in &self.links {
            let difference = state.values[link.to] - link.transporter() * state.values[link.from];
            energy += link.stiffness * difference.norm_sqr();
        }
        if !energy.is_finite() {
            return Err(DynamicsError::NonFinite {
                context: "free energy",
                index: 0,
            });
        }
        Ok(energy)
    }

    /// Evaluates the Wirtinger derivative `dF/dpsi*` at every site.
    ///
    /// Consequently `dF/d Re(psi) = 2 Re(gradient)` and
    /// `dF/d Im(psi) = 2 Im(gradient)`.
    pub fn gradient(&self, state: &OrderParameter) -> Result<Vec<Complex64>> {
        self.validate_values(state.as_slice())?;
        let mut gradient = Vec::with_capacity(self.site_count());
        for (&psi, site) in state.as_slice().iter().zip(&self.sites) {
            gradient.push((site.alpha + site.beta * psi.norm_sqr()) * psi);
        }
        for link in &self.links {
            let transporter = link.transporter();
            let from = state.values[link.from];
            let to = state.values[link.to];
            gradient[link.from] += link.stiffness * (from - transporter.conj() * to);
            gradient[link.to] += link.stiffness * (to - transporter * from);
        }
        Ok(gradient)
    }

    /// Returns the dimensionless pair current along a stored link orientation.
    ///
    /// The convention is `J_ij = -dF/dA_ij =
    /// 2 K_ij Im(conj(exp(i A_ij) psi_i) psi_j)`. Multiply by the desired
    /// charge-over-action scale to obtain a current in external units.
    pub fn link_current(&self, state: &OrderParameter, link: usize) -> Result<f64> {
        self.validate_values(state.as_slice())?;
        let edge = self.links.get(link).ok_or(DynamicsError::LinkOutOfBounds {
            link,
            link_count: self.link_count(),
        })?;
        Ok(2.0
            * edge.stiffness
            * ((edge.transporter() * state.values[edge.from]).conj() * state.values[edge.to]).im)
    }

    /// Computes the signed dimensionless magnetic flux around a closed loop.
    ///
    /// The result is the unwrapped oriented phase sum. It equals
    /// `q_pair Phi / hbar` for links produced by a vector potential and is
    /// gauge invariant for a closed loop.
    pub fn loop_phase(&self, path: &[LoopEdge]) -> Result<f64> {
        if path.is_empty() {
            return Err(DynamicsError::OpenLoop { position: 0 });
        }
        let oriented = |entry: LoopEdge| -> Result<(usize, usize, f64)> {
            let link = self
                .links
                .get(entry.link)
                .ok_or(DynamicsError::LinkOutOfBounds {
                    link: entry.link,
                    link_count: self.link_count(),
                })?;
            Ok(match entry.direction {
                LinkDirection::Forward => (link.from, link.to, link.phase),
                LinkDirection::Reverse => (link.to, link.from, -link.phase),
            })
        };
        let (first, mut current, first_phase) = oriented(path[0])?;
        let mut sum = first_phase;
        for (position, &entry) in path.iter().enumerate().skip(1) {
            let (from, to, phase) = oriented(entry)?;
            if from != current {
                return Err(DynamicsError::OpenLoop { position });
            }
            current = to;
            sum += phase;
            if !sum.is_finite() {
                return Err(DynamicsError::NonFinite {
                    context: "loop phase",
                    index: position,
                });
            }
        }
        if current != first {
            return Err(DynamicsError::OpenLoop {
                position: path.len(),
            });
        }
        Ok(sum)
    }

    /// Applies a local U(1) gauge transformation to both model and state.
    ///
    /// The operation validates all phases and constructs both outputs before
    /// returning, leaving its inputs unchanged on every error path.
    pub fn gauge_transform(
        &self,
        state: &OrderParameter,
        chi: &[f64],
    ) -> Result<(Self, OrderParameter)> {
        self.validate_values(state.as_slice())?;
        if chi.len() != self.site_count() {
            return Err(DynamicsError::DimensionMismatch {
                context: "gauge phases",
                expected: self.site_count(),
                actual: chi.len(),
            });
        }
        if let Some(index) = chi.iter().position(|phase| !phase.is_finite()) {
            return Err(DynamicsError::NonFinite {
                context: "gauge phases",
                index,
            });
        }
        let values = state
            .as_slice()
            .iter()
            .zip(chi)
            .map(|(&psi, &phase)| Complex64::from_polar(1.0, phase) * psi)
            .collect::<Vec<_>>();
        let links = self
            .links
            .iter()
            .map(|link| {
                GaugeLink::try_new(
                    link.from,
                    link.to,
                    link.stiffness,
                    link.phase + chi[link.to] - chi[link.from],
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let model = Self::try_new(self.sites.clone(), links)?;
        let state = OrderParameter::try_new(&model, values)?;
        Ok((model, state))
    }

    /// Converts a snapshot to onsite s-wave BdG pairing gaps.
    ///
    /// `gap_scale` maps the TDGL order-parameter unit to the BdG energy unit.
    /// The returned vector passes directly to
    /// `superconductivity::OnsiteSWaveModel::try_new`; BdG assembly and
    /// validation remain in that crate.
    pub fn scaled_pairing_gaps(
        &self,
        state: &OrderParameter,
        gap_scale: f64,
    ) -> Result<Vec<Complex64>> {
        self.validate_values(state.as_slice())?;
        if !gap_scale.is_finite() {
            return Err(DynamicsError::NonFinite {
                context: "gap scale",
                index: 0,
            });
        }
        Ok(state
            .as_slice()
            .iter()
            .map(|&value| gap_scale * value)
            .collect())
    }
}

impl OrderParameter {
    pub(crate) fn replace(&mut self, values: Vec<Complex64>) {
        self.values = values;
    }
}
