use clifford_core::CliffordAlgebra;
use majorana_fermions::MajoranaMonomial;

#[test]
fn sparse_products_match_small_clifford_algebra() {
    const DIMENSION: usize = 5;
    let algebra = CliffordAlgebra::new(DIMENSION, 0);
    for left_bitmap in 0..(1_u64 << DIMENSION) {
        for right_bitmap in 0..(1_u64 << DIMENSION) {
            let left = monomial_from_bitmap(left_bitmap, DIMENSION);
            let right = monomial_from_bitmap(right_bitmap, DIMENSION);
            let product = left.multiply(&right);

            let left_index = algebra.bitmap_to_index[left_bitmap as usize];
            let right_index = algebra.bitmap_to_index[right_bitmap as usize];
            let table_offset = left_index * algebra.n_blades + right_index;
            let expected_index = algebra.cayley_index[table_offset];
            let expected_bitmap = algebra.index_to_bitmap[expected_index];
            let actual_bitmap = product
                .monomial()
                .indices()
                .iter()
                .fold(0_u64, |bitmap, index| bitmap | (1_u64 << index));

            assert_eq!(actual_bitmap, expected_bitmap);
            assert_eq!(f32::from(product.sign()), algebra.cayley_sign[table_offset]);
        }
    }
}

fn monomial_from_bitmap(bitmap: u64, dimension: usize) -> MajoranaMonomial {
    MajoranaMonomial::new(
        (0..dimension)
            .filter(|index| bitmap & (1_u64 << index) != 0)
            .collect(),
    )
    .unwrap()
}
