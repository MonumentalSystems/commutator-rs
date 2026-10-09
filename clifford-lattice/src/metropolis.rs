//! Generic sequential Metropolis reference sweep.

use crate::error::finite_scalar;
use crate::lattice::PeriodicLattice2;
use crate::rng::RandomSource;
use crate::{LatticeError, Result};

/// Symmetric nearest-neighbor interaction used by the lattice action.
///
/// Implementations must satisfy `score(a, b) == score(b, a)` within their
/// numerical tolerance. The sequential local-action calculation relies on
/// that contract.
pub trait SymmetricPairInteraction<T> {
    /// Returns the pair score for two neighboring sites.
    fn score(&self, left: &T, right: &T) -> f64;
}

/// Site-local proposal used by a Metropolis sweep.
pub trait Proposal<T, R: RandomSource> {
    /// Proposes a replacement for `current` at the supplied row-major index.
    fn propose(&self, current: &T, site_index: usize, random: &mut R) -> Result<T>;
}

/// Aggregate outcome of one complete lattice sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SweepStats {
    attempted: usize,
    accepted: usize,
}

impl SweepStats {
    /// Number of attempted site updates.
    pub const fn attempted(self) -> usize {
        self.attempted
    }

    /// Number of accepted site updates.
    pub const fn accepted(self) -> usize {
        self.accepted
    }

    /// Fraction of attempted updates that were accepted.
    pub fn acceptance_rate(self) -> f64 {
        if self.attempted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.attempted as f64
        }
    }
}

/// Computes `-beta_coupling * sum(score)` over positive-direction bonds.
///
/// Each site contributes its `+x` and `+y` bond exactly once.
pub fn total_action<T, I>(
    lattice: &PeriodicLattice2<T>,
    beta_coupling: f64,
    interaction: &I,
) -> Result<f64>
where
    I: SymmetricPairInteraction<T>,
{
    finite_scalar("beta coupling", beta_coupling)?;
    let mut score_sum = 0.0;
    for site_index in 0..lattice.len() {
        let neighbors = lattice.neighbors(site_index)?;
        let site = lattice.get(site_index)?;
        for neighbor_index in [neighbors[0], neighbors[2]] {
            let score = interaction.score(site, lattice.get(neighbor_index)?);
            finite_scalar("pair interaction score", score)?;
            score_sum += score;
        }
    }
    let action = -beta_coupling * score_sum;
    finite_scalar("total lattice action", action)?;
    Ok(action)
}

/// Computes the full action change caused by replacing one site.
pub fn replacement_delta_action<T, I>(
    lattice: &PeriodicLattice2<T>,
    site_index: usize,
    replacement: &T,
    beta_coupling: f64,
    interaction: &I,
) -> Result<f64>
where
    I: SymmetricPairInteraction<T>,
{
    finite_scalar("beta coupling", beta_coupling)?;
    let current = lattice.get(site_index)?;
    let mut old_score = 0.0;
    let mut new_score = 0.0;
    for neighbor_index in lattice.neighbors(site_index)? {
        let neighbor = lattice.get(neighbor_index)?;
        let old_pair = interaction.score(current, neighbor);
        let new_pair = interaction.score(replacement, neighbor);
        finite_scalar("pair interaction score", old_pair)?;
        finite_scalar("pair interaction score", new_pair)?;
        old_score += old_pair;
        new_score += new_pair;
    }
    let delta = -beta_coupling * (new_score - old_score);
    finite_scalar("replacement action delta", delta)?;
    Ok(delta)
}

/// Performs one row-major sequential Metropolis sweep.
///
/// `beta_coupling` is the dimensionless inverse-temperature-weighted
/// coupling, often written `βJ`. Each proposal observes updates accepted at
/// earlier row-major sites in the same sweep.
///
/// The sweep is intentionally non-transactional: if a later proposal or
/// interaction returns an error, updates accepted at earlier sites remain in
/// the lattice.
pub fn sequential_metropolis_sweep<T, I, P, R>(
    lattice: &mut PeriodicLattice2<T>,
    beta_coupling: f64,
    interaction: &I,
    proposal: &P,
    random: &mut R,
) -> Result<SweepStats>
where
    I: SymmetricPairInteraction<T>,
    P: Proposal<T, R>,
    R: RandomSource,
{
    finite_scalar("beta coupling", beta_coupling)?;
    let mut accepted = 0;
    for site_index in 0..lattice.len() {
        let replacement = proposal.propose(lattice.get(site_index)?, site_index, random)?;
        let delta = replacement_delta_action(
            lattice,
            site_index,
            &replacement,
            beta_coupling,
            interaction,
        )?;
        let accept = delta <= 0.0 || random.uniform_f64() < (-delta).exp();
        if accept {
            lattice.set(site_index, replacement)?;
            accepted += 1;
        }
    }
    if accepted > lattice.len() {
        return Err(LatticeError::InvalidDomain("accepted update count"));
    }
    Ok(SweepStats {
        attempted: lattice.len(),
        accepted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    struct ProductInteraction;

    impl SymmetricPairInteraction<f64> for ProductInteraction {
        fn score(&self, left: &f64, right: &f64) -> f64 {
            left * right
        }
    }

    struct FlipProposal;

    impl<R: RandomSource> Proposal<f64, R> for FlipProposal {
        fn propose(&self, current: &f64, _: usize, _: &mut R) -> Result<f64> {
            Ok(-current)
        }
    }

    #[test]
    fn local_delta_matches_full_action_difference() {
        let lattice = PeriodicLattice2::from_sites(
            3,
            3,
            vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0],
        )
        .unwrap();
        let before = total_action(&lattice, 0.7, &ProductInteraction).unwrap();
        let replacement = -lattice.sites()[4];
        let delta =
            replacement_delta_action(&lattice, 4, &replacement, 0.7, &ProductInteraction).unwrap();
        let mut changed = lattice.clone();
        changed.set(4, replacement).unwrap();
        let after = total_action(&changed, 0.7, &ProductInteraction).unwrap();
        assert!((delta - (after - before)).abs() < 1.0e-12);
    }

    #[test]
    fn zero_beta_accepts_every_finite_proposal() {
        let mut lattice = PeriodicLattice2::filled(3, 3, 1.0).unwrap();
        let mut random = SplitMix64::new(7);
        let stats = sequential_metropolis_sweep(
            &mut lattice,
            0.0,
            &ProductInteraction,
            &FlipProposal,
            &mut random,
        )
        .unwrap();
        assert_eq!(stats.attempted(), 9);
        assert_eq!(stats.accepted(), 9);
        assert_eq!(stats.acceptance_rate(), 1.0);
    }
}
