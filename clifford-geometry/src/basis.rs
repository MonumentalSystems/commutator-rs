//! Bit-level blade operations for geometric algebra.
//!
//! Port of `vsr_basis.h` — compile-time blade representation using bitmasks.
//! Each basis blade is a `u32` where set bits indicate which basis vectors
//! are present: e.g. `e1 = 0b001`, `e2 = 0b010`, `e12 = 0b011`.

/// Grade of a blade = number of basis vectors (popcount).
#[inline]
pub const fn grade(blade: u32) -> u32 {
    blade.count_ones()
}

/// Geometric product of two basis blades = XOR of their bitmasks.
/// The resulting blade contains exactly the symmetric difference of basis vectors.
#[inline]
pub const fn product(a: u32, b: u32) -> u32 {
    a ^ b
}

/// Outer product is valid only when blades share no basis vectors.
#[inline]
pub const fn outer(a: u32, b: u32) -> bool {
    (a & b) == 0
}

/// Inner product is valid when grade(a) <= grade(b) and a is "contained" in b.
#[inline]
pub const fn inner(a: u32, b: u32) -> bool {
    let ga = grade(a);
    let gb = grade(b);
    if ga == 0 || gb == 0 {
        return false;
    }
    // Left contraction: grade of result = gb - ga
    grade(a ^ b) == gb - ga
}

/// Sign flip from reordering basis vectors in the geometric product.
/// Counts the number of transpositions needed to bring `a * b` into canonical order.
/// Returns true if the sign is negative.
///
/// For each basis vector in `a`, count how many basis vectors in `b` have
/// a lower index and thus need to be swapped past.
#[inline]
pub const fn sign_flip(a: u32, b: u32) -> bool {
    // We need to count: for each set bit i in a, how many set bits in b
    // are at positions *below* i. Sum all those counts; if odd, sign is negative.
    let mut swaps = 0u32;
    let mut mask_a = a;
    // Process each set bit of a from lowest to highest
    while mask_a != 0 {
        // Find lowest set bit position
        let bit_pos = mask_a.trailing_zeros();
        // Count bits in b below this position
        if bit_pos > 0 {
            let lower_mask = (1u32 << bit_pos) - 1;
            swaps += (b & lower_mask).count_ones();
        }
        // Clear this bit
        mask_a &= mask_a - 1;
    }
    swaps % 2 == 1
}

/// Sign of the geometric product of two Euclidean basis blades.
/// Returns +1 or -1.
#[inline]
pub const fn sign(a: u32, b: u32) -> i32 {
    if sign_flip(a, b) {
        -1
    } else {
        1
    }
}

/// Sign for metric contraction. For each shared basis vector at position i,
/// the metric signature applies: +1 for positive, -1 for negative, 0 for degenerate.
/// `metric_signs` is indexed by basis vector position (0..dim).
/// Returns the combined sign, or 0 if any degenerate dimension is contracted.
///
/// # Panics
///
/// Panics if the metric has more than 32 entries or either blade refers to a
/// basis vector not covered by the metric.
#[inline]
pub const fn metric_sign(a: u32, b: u32, metric_signs: &[i32]) -> i32 {
    assert!(
        metric_signs.len() <= u32::BITS as usize,
        "metric signature exceeds the u32 blade representation"
    );
    let covered = if metric_signs.len() == u32::BITS as usize {
        u32::MAX
    } else {
        (1u32 << metric_signs.len()) - 1
    };
    assert!(
        (a | b) & !covered == 0,
        "metric signature does not cover every basis vector in the blades"
    );

    let shared = a & b;
    let mut result = 1i32;
    let mut i = 0;
    while i < metric_signs.len() {
        if (shared >> i) & 1 == 1 {
            result *= metric_signs[i];
            if result == 0 {
                return 0;
            }
        }
        i += 1;
    }
    result
}

/// Reverse sign: (-1)^(k*(k-1)/2) where k = grade.
#[inline]
pub const fn reverse_sign(blade: u32) -> i32 {
    let k = grade(blade) as i32;
    if (k * (k - 1) / 2) % 2 == 0 {
        1
    } else {
        -1
    }
}

/// Involution sign: (-1)^k where k = grade.
#[inline]
pub const fn involute_sign(blade: u32) -> i32 {
    if grade(blade) % 2 == 0 {
        1
    } else {
        -1
    }
}

/// Conjugate sign: reverse composed with involution = (-1)^(k*(k+1)/2).
#[inline]
pub const fn conjugate_sign(blade: u32) -> i32 {
    let k = grade(blade);
    if (k * (k + 1) / 2) % 2 == 0 {
        1
    } else {
        -1
    }
}

/// Pseudoscalar for an N-dimensional algebra: all bits set up to N.
///
/// # Panics
///
/// Panics when `dim` exceeds the 32 dimensions representable by a `u32` blade.
#[inline]
pub const fn pseudoscalar(dim: u32) -> u32 {
    assert!(
        dim <= u32::BITS,
        "dimension exceeds the u32 blade representation"
    );
    if dim == u32::BITS {
        u32::MAX
    } else {
        (1u32 << dim) - 1
    }
}

/// Dual of a blade: complement with respect to the pseudoscalar.
///
/// # Panics
///
/// Panics when `dim` exceeds 32 or `blade` contains a basis vector outside
/// the requested dimension.
#[inline]
pub const fn dual(blade: u32, dim: u32) -> u32 {
    let ps = pseudoscalar(dim);
    assert!(
        blade & !ps == 0,
        "blade contains a basis vector outside the requested dimension"
    );
    blade ^ ps
}

/// Convert blade bitmask to a human-readable string like "e1", "e12", "e123".
/// Returns "s" for scalar (blade == 0).
pub fn blade_name(blade: u32) -> String {
    if blade == 0 {
        return "s".to_string();
    }
    let mut s = String::from("e");
    let mut i = 0;
    let mut b = blade;
    while b != 0 {
        if b & 1 == 1 {
            s.push_str(&(i + 1).to_string());
        }
        b >>= 1;
        i += 1;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grade() {
        assert_eq!(grade(0b000), 0); // scalar
        assert_eq!(grade(0b001), 1); // e1
        assert_eq!(grade(0b011), 2); // e12
        assert_eq!(grade(0b111), 3); // e123
    }

    #[test]
    fn test_product() {
        assert_eq!(product(0b001, 0b010), 0b011); // e1 * e2 = e12
        assert_eq!(product(0b011, 0b010), 0b001); // e12 * e2 = e1
        assert_eq!(product(0b001, 0b001), 0b000); // e1 * e1 = scalar
    }

    #[test]
    fn test_sign_flip() {
        // e1 * e2 = +e12 (no swaps needed)
        assert!(!sign_flip(0b001, 0b010));
        // e2 * e1 = -e12 (one swap)
        assert!(sign_flip(0b010, 0b001));
        // e1 * e1 = +1 (no swaps)
        assert!(!sign_flip(0b001, 0b001));
        // e12 * e3 = +e123
        assert!(!sign_flip(0b011, 0b100));
        // e3 * e12: e3 (bit 2) needs to pass past bits 0,1 of b=0b011 → 2 swaps (even) → positive
        assert!(!sign_flip(0b100, 0b011));
        // e21 → e12 needs 1 swap → negative
        assert!(sign_flip(0b010, 0b001));
    }

    #[test]
    fn test_sign() {
        assert_eq!(sign(0b001, 0b010), 1); // e1*e2 = +e12
        assert_eq!(sign(0b010, 0b001), -1); // e2*e1 = -e12
    }

    #[test]
    fn test_reverse_sign() {
        assert_eq!(reverse_sign(0b000), 1); // scalar: unchanged
        assert_eq!(reverse_sign(0b001), 1); // vector: unchanged
        assert_eq!(reverse_sign(0b011), -1); // bivector: flipped
        assert_eq!(reverse_sign(0b111), -1); // trivector: flipped
    }

    #[test]
    fn test_involute_sign() {
        assert_eq!(involute_sign(0b000), 1); // scalar: even grade
        assert_eq!(involute_sign(0b001), -1); // vector: odd grade
        assert_eq!(involute_sign(0b011), 1); // bivector: even grade
        assert_eq!(involute_sign(0b111), -1); // trivector: odd grade
    }

    #[test]
    fn test_outer() {
        assert!(outer(0b001, 0b010)); // e1 ^ e2 is valid
        assert!(!outer(0b001, 0b001)); // e1 ^ e1 is zero
        assert!(!outer(0b011, 0b010)); // e12 ^ e2 is zero
    }

    #[test]
    fn test_inner() {
        // e1 . e1 = scalar (valid: grade 1 <= 1, result grade 0 = 1-1)
        assert!(inner(0b001, 0b001));
        // e1 . e12 = e2 (valid: grade 1 <= 2, result grade 1 = 2-1)
        assert!(inner(0b001, 0b011));
        // scalar . anything: invalid
        assert!(!inner(0b000, 0b001));
    }

    #[test]
    fn test_pseudoscalar() {
        assert_eq!(pseudoscalar(3), 0b111); // e123
        assert_eq!(pseudoscalar(5), 0b11111); // e12345
        assert_eq!(pseudoscalar(32), u32::MAX);
    }

    #[test]
    #[should_panic(expected = "dimension exceeds the u32 blade representation")]
    fn test_pseudoscalar_rejects_dimension_above_32() {
        let _ = pseudoscalar(33);
    }

    #[test]
    fn test_dual() {
        assert_eq!(dual(0b001, 3), 0b110); // dual of e1 in 3D = e23
        assert_eq!(dual(0b000, 3), 0b111); // dual of scalar = pseudoscalar
        assert_eq!(dual(1u32 << 31, 32), u32::MAX ^ (1u32 << 31));
    }

    #[test]
    #[should_panic(expected = "outside the requested dimension")]
    fn test_dual_rejects_blade_outside_dimension() {
        let _ = dual(0b1000, 3);
    }

    #[test]
    fn test_blade_name() {
        assert_eq!(blade_name(0), "s");
        assert_eq!(blade_name(0b001), "e1");
        assert_eq!(blade_name(0b011), "e12");
        assert_eq!(blade_name(0b111), "e123");
    }

    #[test]
    fn test_metric_sign_euclidean() {
        let euclidean = [1, 1, 1]; // all positive
        assert_eq!(metric_sign(0b001, 0b001, &euclidean), 1); // e1*e1 = +1
        assert_eq!(metric_sign(0b001, 0b010, &euclidean), 1); // no shared = +1
    }

    #[test]
    fn test_metric_sign_minkowski() {
        let minkowski = [1, 1, 1, -1]; // 3+1 spacetime
        assert_eq!(metric_sign(0b1000, 0b1000, &minkowski), -1); // e4*e4 = -1
    }

    #[test]
    #[should_panic(expected = "metric signature does not cover every basis vector")]
    fn test_metric_sign_rejects_short_metric() {
        let _ = metric_sign(0b1000, 0b1000, &[1, 1, 1]);
    }
}
