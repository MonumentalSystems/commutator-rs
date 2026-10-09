use clifford_lattice::cl6::{
    GaussianBivectorProposal, RotorScalarInteraction, Spin6Rotor, BIVECTOR_COMPONENTS,
};
use clifford_lattice::lattice::PeriodicLattice2;
use clifford_lattice::metropolis::{
    replacement_delta_action, sequential_metropolis_sweep, total_action,
};
use clifford_lattice::rng::SplitMix64;

fn plane_rotor(component: usize, angle: f64) -> Spin6Rotor {
    let mut bivector = [0.0; BIVECTOR_COMPONENTS];
    bivector[component] = angle;
    Spin6Rotor::exp_bivector(bivector).unwrap()
}

fn close(left: f64, right: f64, tolerance: f64) {
    assert!(
        (left - right).abs() <= tolerance,
        "{left} differs from {right} by more than {tolerance}"
    );
}

#[test]
fn identity_lattice_has_exact_bond_count() {
    let lattice = PeriodicLattice2::filled(3, 4, Spin6Rotor::identity()).unwrap();
    close(
        total_action(&lattice, 0.75, &RotorScalarInteraction).unwrap(),
        -0.75 * 2.0 * 12.0,
        1.0e-12,
    );
}

#[test]
fn rotor_local_delta_matches_full_recomputation() {
    let sites = (0..12)
        .map(|index| plane_rotor(index % BIVECTOR_COMPONENTS, index as f64 * 0.07))
        .collect();
    let lattice = PeriodicLattice2::from_sites(3, 4, sites).unwrap();
    let replacement = plane_rotor(9, -0.41);
    let before = total_action(&lattice, 0.63, &RotorScalarInteraction).unwrap();
    let delta =
        replacement_delta_action(&lattice, 5, &replacement, 0.63, &RotorScalarInteraction).unwrap();
    let mut changed = lattice.clone();
    changed.set(5, replacement).unwrap();
    let after = total_action(&changed, 0.63, &RotorScalarInteraction).unwrap();
    close(delta, after - before, 2.0e-12);
}

#[test]
fn local_delta_handles_minimum_periodic_extents() {
    for (width, height) in [(2, 2), (2, 3), (3, 2)] {
        let sites = (0..width * height)
            .map(|index| plane_rotor(index % BIVECTOR_COMPONENTS, index as f64 * 0.09))
            .collect();
        let lattice = PeriodicLattice2::from_sites(width, height, sites).unwrap();
        let site_index = lattice.len() / 2;
        let replacement = plane_rotor(13, -0.27);
        let before = total_action(&lattice, 0.41, &RotorScalarInteraction).unwrap();
        let delta = replacement_delta_action(
            &lattice,
            site_index,
            &replacement,
            0.41,
            &RotorScalarInteraction,
        )
        .unwrap();
        let mut changed = lattice.clone();
        changed.set(site_index, replacement).unwrap();
        let after = total_action(&changed, 0.41, &RotorScalarInteraction).unwrap();
        close(delta, after - before, 2.0e-12);
    }
}

#[test]
fn common_left_rotation_preserves_action() {
    let sites = (0..9)
        .map(|index| plane_rotor(index % BIVECTOR_COMPONENTS, index as f64 * 0.11))
        .collect::<Vec<_>>();
    let lattice = PeriodicLattice2::from_sites(3, 3, sites.clone()).unwrap();
    let common = plane_rotor(11, 0.83);
    let rotated = PeriodicLattice2::from_sites(
        3,
        3,
        sites
            .into_iter()
            .map(|site| common.compose(site).unwrap())
            .collect(),
    )
    .unwrap();
    close(
        total_action(&lattice, 1.2, &RotorScalarInteraction).unwrap(),
        total_action(&rotated, 1.2, &RotorScalarInteraction).unwrap(),
        2.0e-12,
    );
}

#[test]
fn seeded_sweeps_are_reproducible_and_preserve_rotors() {
    let initial = PeriodicLattice2::filled(4, 4, Spin6Rotor::identity()).unwrap();
    let proposal = GaussianBivectorProposal::canonical(0.08).unwrap();
    let mut first = initial.clone();
    let mut second = initial;
    let mut first_random = SplitMix64::new(0x5eed);
    let mut second_random = SplitMix64::new(0x5eed);

    let first_stats = sequential_metropolis_sweep(
        &mut first,
        0.7,
        &RotorScalarInteraction,
        &proposal,
        &mut first_random,
    )
    .unwrap();
    let second_stats = sequential_metropolis_sweep(
        &mut second,
        0.7,
        &RotorScalarInteraction,
        &proposal,
        &mut second_random,
    )
    .unwrap();

    assert_eq!(first_stats, second_stats);
    assert_eq!(first, second);
    assert_eq!(first_stats.attempted(), 16);
    assert_eq!(first_stats.accepted(), 13);
    assert_eq!(first_random.state(), 10_050_863_829_183_678_365);
    close(
        total_action(&first, 0.7, &RotorScalarInteraction).unwrap(),
        -20.567_140_209_393_802,
        5.0e-13,
    );
    close(
        first.sites()[0].coefficients()[0],
        0.946_103_248_159_68,
        1.0e-13,
    );
    close(
        first.sites()[0].coefficients()[13],
        -0.152_141_695_653_293,
        1.0e-13,
    );
    close(
        first.sites()[7].coefficients()[10],
        0.076_945_834_766_598_94,
        1.0e-13,
    );
    for rotor in first.sites() {
        close(rotor.norm_squared().unwrap(), 1.0, 2.0e-12);
        assert!(rotor.versor_defect().unwrap() < 2.0e-12);
    }
}

#[test]
fn large_simple_plane_exponential_remains_accurate() {
    let angle = 37.0;
    let rotor = plane_rotor(4, angle);
    close(rotor.coefficients()[0], angle.cos(), 2.0e-12);
    close(rotor.coefficients()[5], angle.sin(), 2.0e-12);
    close(rotor.norm_squared().unwrap(), 1.0, 2.0e-12);
}
