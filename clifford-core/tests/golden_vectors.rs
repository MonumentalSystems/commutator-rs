use clifford_core::{CliffordAlgebra, CliffordScalar};
use serde_json::Value;

trait GoldenScalar: CliffordScalar + std::fmt::Debug {
    fn from_fixture(value: f64) -> Self;
    fn as_f64(self) -> f64;
    fn tolerance(case: &Value) -> f64;
}

impl GoldenScalar for f32 {
    fn from_fixture(value: f64) -> Self {
        value as f32
    }

    fn as_f64(self) -> f64 {
        self as f64
    }

    fn tolerance(case: &Value) -> f64 {
        case["tolerance"]["f32_abs"].as_f64().unwrap()
    }
}

impl GoldenScalar for f64 {
    fn from_fixture(value: f64) -> Self {
        value
    }

    fn as_f64(self) -> f64 {
        self
    }

    fn tolerance(case: &Value) -> f64 {
        case["tolerance"]["f64_abs"].as_f64().unwrap()
    }
}

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/clifford-golden-v1.json")).unwrap()
}

fn signature(case: &Value) -> CliffordAlgebra {
    let signature = case["signature"].as_array().unwrap();
    CliffordAlgebra::new(
        signature[0].as_u64().unwrap() as usize,
        signature[1].as_u64().unwrap() as usize,
    )
}

fn terms<S: GoldenScalar>(algebra: &CliffordAlgebra, value: &Value) -> Vec<S> {
    let mut dense = algebra.zero();
    let mut previous_bitmap = None;
    for term in value.as_array().unwrap() {
        let pair = term.as_array().unwrap();
        let bitmap = pair[0].as_u64().unwrap() as usize;
        let coefficient = pair[1].as_f64().unwrap();
        assert!(coefficient.is_finite() && coefficient != 0.0);
        if let Some(previous) = previous_bitmap {
            assert!(bitmap > previous, "fixture terms must be sorted and unique");
        }
        previous_bitmap = Some(bitmap);
        assert!(bitmap < algebra.bitmap_to_index.len());
        dense[algebra.bitmap_to_index[bitmap]] = S::from_fixture(coefficient);
    }
    dense
}

fn assert_expected<S: GoldenScalar>(case: &Value, algebra: &CliffordAlgebra, actual: &[S]) {
    let expected = terms::<S>(algebra, &case["expected"]);
    let tolerance = S::tolerance(case);
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let difference = (actual.as_f64() - expected.as_f64()).abs();
        assert!(
            difference <= tolerance,
            "{}: blade bitmap {} expected {:?}, got {:?} (difference {}, tolerance {})",
            case["id"].as_str().unwrap(),
            algebra.index_to_bitmap[index],
            expected,
            actual,
            difference,
            tolerance
        );
    }
}

fn run_operation_case<S: GoldenScalar>(case: &Value) {
    let algebra = signature(case);
    let actual = match case["operation"].as_str().unwrap() {
        "geometric_product" => {
            let a = terms::<S>(&algebra, &case["a"]);
            let b = terms::<S>(&algebra, &case["b"]);
            algebra.geometric_product(&a, &b)
        }
        "commutator" => {
            let a = terms::<S>(&algebra, &case["a"]);
            let b = terms::<S>(&algebra, &case["b"]);
            let scale = case["scale"].as_f64().unwrap();
            let multiplier = S::from_fixture(scale / 0.5);
            algebra
                .commutator(&a, &b)
                .into_iter()
                .map(|coefficient| coefficient * multiplier)
                .collect()
        }
        "reverse" => {
            let a = terms::<S>(&algebra, &case["a"]);
            algebra.reverse(&a)
        }
        "grade_project" => {
            let a = terms::<S>(&algebra, &case["a"]);
            algebra.grade_project(&a, case["grade"].as_u64().unwrap() as usize)
        }
        "rotor" => {
            let bivector = terms::<S>(&algebra, &case["bivector"]);
            let signed_theta =
                case["theta"].as_f64().unwrap() * case["exponent_sign"].as_f64().unwrap();
            algebra.rotor_from_bivector(&bivector, S::from_fixture(signed_theta))
        }
        "sandwich" => {
            let rotor = terms::<S>(&algebra, &case["r"]);
            let x = terms::<S>(&algebra, &case["x"]);
            algebra.sandwich(&rotor, &x)
        }
        operation => panic!("unsupported fixture operation {operation}"),
    };
    assert_expected::<S>(case, &algebra, &actual);
}

#[test]
fn fixture_metadata_and_basis_orders_are_stable() {
    let fixture = fixture();
    assert_eq!(fixture["schema"], "clifford-golden-v1");
    assert_eq!(
        fixture["provenance"]["versor_commit"],
        "93a5a1334ab77131e5ef6b88105cdaac57d267e6"
    );

    for case in fixture["basis_cases"].as_array().unwrap() {
        let algebra = signature(case);
        let actual: Vec<u64> = match case["grade"].as_u64() {
            Some(grade) => algebra
                .grade_indices(grade as usize)
                .into_iter()
                .map(|index| algebra.blade_bitmap(index))
                .collect(),
            None => algebra.index_to_bitmap.clone(),
        };
        let expected: Vec<u64> = case["expected_bitmaps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|bitmap| bitmap.as_u64().unwrap())
            .collect();
        assert_eq!(actual, expected, "{}", case["id"].as_str().unwrap());
    }
}

#[test]
fn golden_operations_match_f32() {
    let fixture = fixture();
    for case in fixture["operation_cases"].as_array().unwrap() {
        run_operation_case::<f32>(case);
    }
}

#[test]
fn golden_operations_match_f64() {
    let fixture = fixture();
    for case in fixture["operation_cases"].as_array().unwrap() {
        run_operation_case::<f64>(case);
    }
}
