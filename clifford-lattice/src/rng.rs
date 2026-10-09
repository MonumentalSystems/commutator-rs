//! Small deterministic random-source contract for reference simulations.

/// Integer random source consumed by proposal and acceptance kernels.
///
/// The trait keeps `clifford-lattice` independent of a particular RNG crate.
/// Implementations only need to provide uniformly distributed `u64` words.
pub trait RandomSource {
    /// Returns the next random word.
    fn next_u64(&mut self) -> u64;

    /// Returns a uniform sample in the half-open interval `[0, 1)`.
    fn uniform_f64(&mut self) -> f64 {
        const DENOMINATOR: f64 = (1_u64 << 53) as f64;
        (self.next_u64() >> 11) as f64 / DENOMINATOR
    }

    /// Returns a standard-normal sample using the Box–Muller transform.
    fn standard_normal_f64(&mut self) -> f64 {
        const DENOMINATOR: f64 = ((1_u64 << 53) + 1) as f64;
        let first = (self.next_u64() >> 11) as f64;
        let second = (self.next_u64() >> 11) as f64;
        let u1 = (first + 1.0) / DENOMINATOR;
        let u2 = second / (1_u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }
}

/// Deterministic SplitMix64 source suitable for reproducible reference runs.
///
/// This is not a cryptographic random number generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Creates a source from any 64-bit seed, including zero.
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns the current internal state for checkpointing.
    pub const fn state(&self) -> u64 {
        self.state
    }
}

impl RandomSource for SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_integer_sequence_is_frozen() {
        let mut source = SplitMix64::new(0);
        assert_eq!(source.next_u64(), 0xe220_a839_7b1d_cdaf);
        assert_eq!(source.next_u64(), 0x6e78_9e6a_a1b9_65f4);
        assert_eq!(source.next_u64(), 0x06c4_5d18_8009_454f);
    }

    #[test]
    fn floating_samples_have_valid_domains() {
        let mut source = SplitMix64::new(42);
        for _ in 0..128 {
            let uniform = source.uniform_f64();
            assert!((0.0..1.0).contains(&uniform));
            assert!(source.standard_normal_f64().is_finite());
        }
    }

    #[test]
    fn floating_sequences_are_frozen() {
        let mut uniform_source = SplitMix64::new(0x5eed);
        let expected_uniforms = [
            0.038_848_734_697_185_194,
            0.332_801_108_739_429_8,
            0.364_681_856_378_138_2,
        ];
        for expected in expected_uniforms {
            assert_eq!(uniform_source.uniform_f64(), expected);
        }

        let mut source = SplitMix64::new(0x5eed);
        let expected_normals = [
            -1.266_989_807_509_723_2,
            -1.322_960_699_311_736_2,
            -1.694_689_711_534_951_5,
        ];
        for expected in expected_normals {
            assert!((source.standard_normal_f64() - expected).abs() < 5.0e-15);
        }
    }
}
