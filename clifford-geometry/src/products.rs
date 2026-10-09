//! Product table generation for fixed-size sparse multivectors.
//!
//! Port of `vsr_products.h` — generates instruction tables for geometric,
//! outer, and inner products between specific multivector types. Algebra
//! modules cache the generated dense tables with `OnceLock`.
//!
//! Each product between types A and B produces a result type C. The product
//! is defined by a table of instructions: `(sign, idx_a, idx_b, idx_c)`.
//! At runtime, execution is a simple loop over instructions with no branches.

use crate::basis;

/// A single product instruction: multiply `a[src_a]` by `b[src_b]`,
/// apply `sign`, and accumulate into `result[dst]`.
#[derive(Debug, Clone, Copy)]
pub struct Instruction {
    /// Signed metric and reordering coefficient.
    pub sign: f32,
    /// Coefficient index in the left operand.
    pub src_a: usize,
    /// Coefficient index in the right operand.
    pub src_b: usize,
    /// Coefficient index in the result.
    pub dst: usize,
}

/// Execute a product given instruction table, two input arrays, and output array.
#[inline]
pub fn execute_product(instructions: &[Instruction], a: &[f32], b: &[f32], result: &mut [f32]) {
    for inst in instructions {
        result[inst.dst] += inst.sign * a[inst.src_a] * b[inst.src_b];
    }
}

/// Generate geometric product instructions between two basis sets
/// under a given metric.
///
/// `basis_a`: the blade bitmasks for type A (e.g., [0b001, 0b010, 0b100] for a vector)
/// `basis_b`: the blade bitmasks for type B
/// `basis_r`: the blade bitmasks for the result type
/// `metric_signs`: the metric signature (e.g., `[1,1,1]` for Euclidean 3D)
///
/// Returns a Vec of instructions.
pub fn gen_geometric_product(
    basis_a: &[u32],
    basis_b: &[u32],
    basis_r: &[u32],
    metric_signs: &[i32],
) -> Vec<Instruction> {
    let mut instructions = Vec::new();

    for (ia, &ba) in basis_a.iter().enumerate() {
        for (ib, &bb) in basis_b.iter().enumerate() {
            let result_blade = basis::product(ba, bb);
            let reorder_sign = basis::sign(ba, bb);
            let met_sign = basis::metric_sign(ba, bb, metric_signs);
            let total_sign = reorder_sign * met_sign;

            if total_sign == 0 {
                continue;
            }

            // Find result_blade in basis_r
            if let Some(ir) = basis_r.iter().position(|&br| br == result_blade) {
                instructions.push(Instruction {
                    sign: total_sign as f32,
                    src_a: ia,
                    src_b: ib,
                    dst: ir,
                });
            }
        }
    }
    instructions
}

/// Generate outer product instructions (only when blades don't share basis vectors).
pub fn gen_outer_product(basis_a: &[u32], basis_b: &[u32], basis_r: &[u32]) -> Vec<Instruction> {
    let mut instructions = Vec::new();

    for (ia, &ba) in basis_a.iter().enumerate() {
        for (ib, &bb) in basis_b.iter().enumerate() {
            if !basis::outer(ba, bb) {
                continue;
            }
            let result_blade = basis::product(ba, bb);
            let reorder_sign = basis::sign(ba, bb);

            if let Some(ir) = basis_r.iter().position(|&br| br == result_blade) {
                instructions.push(Instruction {
                    sign: reorder_sign as f32,
                    src_a: ia,
                    src_b: ib,
                    dst: ir,
                });
            }
        }
    }
    instructions
}

/// Generate inner product (left contraction) instructions.
pub fn gen_inner_product(
    basis_a: &[u32],
    basis_b: &[u32],
    basis_r: &[u32],
    metric_signs: &[i32],
) -> Vec<Instruction> {
    let mut instructions = Vec::new();

    for (ia, &ba) in basis_a.iter().enumerate() {
        for (ib, &bb) in basis_b.iter().enumerate() {
            if !basis::inner(ba, bb) {
                continue;
            }
            let result_blade = basis::product(ba, bb);
            let reorder_sign = basis::sign(ba, bb);
            let met_sign = basis::metric_sign(ba, bb, metric_signs);
            let total_sign = reorder_sign * met_sign;

            if total_sign == 0 {
                continue;
            }

            if let Some(ir) = basis_r.iter().position(|&br| br == result_blade) {
                instructions.push(Instruction {
                    sign: total_sign as f32,
                    src_a: ia,
                    src_b: ib,
                    dst: ir,
                });
            }
        }
    }
    instructions
}

/// Generate scalar product instructions (only scalar output).
pub fn gen_scalar_product(
    basis_a: &[u32],
    basis_b: &[u32],
    metric_signs: &[i32],
) -> Vec<Instruction> {
    let mut instructions = Vec::new();

    for (ia, &ba) in basis_a.iter().enumerate() {
        for (ib, &bb) in basis_b.iter().enumerate() {
            let result_blade = basis::product(ba, bb);
            if result_blade != 0 {
                continue; // Only keep scalar results
            }
            let reorder_sign = basis::sign(ba, bb);
            let met_sign = basis::metric_sign(ba, bb, metric_signs);
            let total_sign = reorder_sign * met_sign;

            if total_sign == 0 {
                continue;
            }

            instructions.push(Instruction {
                sign: total_sign as f32,
                src_a: ia,
                src_b: ib,
                dst: 0,
            });
        }
    }
    instructions
}

/// Compute the result basis for a geometric product of two types.
/// Returns sorted unique blade bitmasks that appear in the product.
pub fn result_basis_gp(basis_a: &[u32], basis_b: &[u32], metric_signs: &[i32]) -> Vec<u32> {
    let mut blades = Vec::new();
    for &ba in basis_a {
        for &bb in basis_b {
            let result_blade = basis::product(ba, bb);
            let met_sign = basis::metric_sign(ba, bb, metric_signs);
            if met_sign != 0 && !blades.contains(&result_blade) {
                blades.push(result_blade);
            }
        }
    }
    blades.sort_by(|a, b| {
        let ga = basis::grade(*a);
        let gb = basis::grade(*b);
        ga.cmp(&gb).then(a.cmp(b))
    });
    blades
}

/// Compute the result basis for an outer product.
pub fn result_basis_op(basis_a: &[u32], basis_b: &[u32]) -> Vec<u32> {
    let mut blades = Vec::new();
    for &ba in basis_a {
        for &bb in basis_b {
            if basis::outer(ba, bb) {
                let result_blade = basis::product(ba, bb);
                if !blades.contains(&result_blade) {
                    blades.push(result_blade);
                }
            }
        }
    }
    blades.sort_by(|a, b| {
        let ga = basis::grade(*a);
        let gb = basis::grade(*b);
        ga.cmp(&gb).then(a.cmp(b))
    });
    blades
}

/// Dense product table: `table[i][j]` contains `(result_index, sign)` pairs.
/// This is the highest-performance representation, matching HarmonicRust's
/// existing table-driven approach.
pub struct DenseTable<const NA: usize, const NB: usize, const NR: usize> {
    /// For each (i, j) pair, the coefficient to add to result`[k]`.
    /// table`[i]``[j]``[k]` = sign (0.0 if this combination doesn't contribute to k).
    pub table: [[[f32; NR]; NB]; NA],
}

impl<const NA: usize, const NB: usize, const NR: usize> DenseTable<NA, NB, NR> {
    /// Build a dense table from instructions.
    pub fn from_instructions(instructions: &[Instruction]) -> Self {
        let mut table = [[[0.0f32; NR]; NB]; NA];
        for inst in instructions {
            table[inst.src_a][inst.src_b][inst.dst] = inst.sign;
        }
        Self { table }
    }

    /// Execute the product using the dense table.
    #[inline]
    pub fn execute(&self, a: &[f32; NA], b: &[f32; NB]) -> [f32; NR] {
        let mut result = [0.0f32; NR];
        for i in 0..NA {
            let ai = a[i];
            if ai == 0.0 {
                continue;
            }
            for j in 0..NB {
                let bj = b[j];
                if bj == 0.0 {
                    continue;
                }
                let ab = ai * bj;
                for k in 0..NR {
                    result[k] += ab * self.table[i][j][k];
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_3d_vec_vec_gp() {
        // Vector basis in Cl(3,0): e1, e2, e3
        let vec_basis = [0b001, 0b010, 0b100];
        let metric = [1, 1, 1];

        // Vec * Vec in Cl(3,0) should produce: scalar + bivector
        let result_basis = result_basis_gp(&vec_basis, &vec_basis, &metric);
        // scalar(0), e12(011), e13(101), e23(110)
        assert!(result_basis.contains(&0b000)); // scalar
        assert!(result_basis.contains(&0b011)); // e12
        assert!(result_basis.contains(&0b101)); // e13
        assert!(result_basis.contains(&0b110)); // e23
    }

    #[test]
    fn test_euclidean_3d_vec_vec_outer() {
        let vec_basis = [0b001, 0b010, 0b100];
        let result_basis = result_basis_op(&vec_basis, &vec_basis);
        // Outer product of two vectors should only produce bivectors
        assert!(!result_basis.contains(&0b000)); // no scalar
        assert!(result_basis.contains(&0b011)); // e12
        assert!(result_basis.contains(&0b101)); // e13
        assert!(result_basis.contains(&0b110)); // e23
    }

    #[test]
    fn test_gp_instructions_e1_e2() {
        let vec_basis = [0b001, 0b010, 0b100];
        let result_basis = [0b000, 0b011, 0b101, 0b110]; // scalar + bivectors
        let metric = [1, 1, 1];

        let instructions = gen_geometric_product(&vec_basis, &vec_basis, &result_basis, &metric);

        // e1*e1 should give +1 scalar
        let e1e1: Vec<_> = instructions
            .iter()
            .filter(|i| i.src_a == 0 && i.src_b == 0)
            .collect();
        assert_eq!(e1e1.len(), 1);
        assert_eq!(e1e1[0].sign, 1.0);
        assert_eq!(e1e1[0].dst, 0); // scalar index

        // e1*e2 should give +1 e12
        let e1e2: Vec<_> = instructions
            .iter()
            .filter(|i| i.src_a == 0 && i.src_b == 1)
            .collect();
        assert_eq!(e1e2.len(), 1);
        assert_eq!(e1e2[0].sign, 1.0);
        assert_eq!(e1e2[0].dst, 1); // e12 index

        // e2*e1 should give -1 e12
        let e2e1: Vec<_> = instructions
            .iter()
            .filter(|i| i.src_a == 1 && i.src_b == 0)
            .collect();
        assert_eq!(e2e1.len(), 1);
        assert_eq!(e2e1[0].sign, -1.0);
        assert_eq!(e2e1[0].dst, 1); // e12 index
    }

    #[test]
    fn test_dense_table_vec_dot_vec() {
        // Test that v1 . v2 = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        let vec_basis: [u32; 3] = [0b001, 0b010, 0b100];
        let metric = [1i32, 1, 1];

        let instructions = gen_scalar_product(&vec_basis, &vec_basis, &metric);
        // Build result manually
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let mut r = [0.0f32];
        execute_product(&instructions, &a, &b, &mut r);
        assert_eq!(r[0], 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0);
    }

    #[test]
    fn product_instructions_preserve_indices_above_u8() {
        let basis: Vec<u32> = (0..=256).collect();
        let instructions = gen_outer_product(&basis, &[0], &basis);
        let boundary = instructions
            .iter()
            .find(|instruction| instruction.src_a == 256)
            .expect("the 257th coefficient should produce an instruction");

        assert_eq!(boundary.src_b, 0);
        assert_eq!(boundary.dst, 256);

        let mut a = vec![0.0; 257];
        let b = [3.0];
        let mut result = vec![0.0; 257];
        a[256] = 2.0;
        execute_product(&[*boundary], &a, &b, &mut result);
        assert_eq!(result[256], 6.0);
    }
}
