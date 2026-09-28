//! Stories of the interval bounds' exact order (F2-008): a decimal bound
//! compares by the exact value it denotes, however large its exponent,
//! and a decimal whose exponent exceeds the bound is refused, never
//! truncated to a wrong order.

use fhy_core::expression::{BigInt, Decimal, DecimalPartsError, LiteralValue};
use fhy_core::param::{Inclusivity, ParamBuildError, check_bounds_are_ordered};
use rstest::rstest;

fn decimal(coefficient: i64, exponent: i64) -> LiteralValue {
    LiteralValue::Decimal(Decimal::from_parts(BigInt::from(coefficient), exponent).expect("parts"))
}

fn ordered(lower: &LiteralValue, upper: &LiteralValue) -> bool {
    check_bounds_are_ordered(lower, upper, Inclusivity::Inclusive, Inclusivity::Inclusive).is_ok()
}

#[test]
fn a_decimal_bound_with_a_huge_exponent_is_refused_not_misordered() {
    let limit = i64::from(Decimal::MAX_EXPONENT_MAGNITUDE);
    let huge = (1_i64 << 32) + 1;

    for exponent in [huge, -huge, limit + 1, -(limit + 1), i64::MAX, i64::MIN] {
        assert_eq!(
            Decimal::from_parts(BigInt::from(1), exponent),
            Err(DecimalPartsError::ExponentOutOfRange { exponent }),
            "{exponent}"
        );
    }
    // Truncated to its coefficient, `1 * 10^(2^32 + 1)` would have read as
    // `1`, below the float `2.0`. At the bound the order is exact.
    let at_the_bound = decimal(1, limit);
    assert!(!ordered(&at_the_bound, &LiteralValue::Float(2.0)));
    assert!(!ordered(&at_the_bound, &LiteralValue::Float(f64::MAX)));
    assert!(ordered(&at_the_bound, &LiteralValue::Float(f64::INFINITY)));
    assert!(ordered(&LiteralValue::Float(f64::MAX), &at_the_bound));
    let tiny = decimal(1, -limit);
    assert!(ordered(&LiteralValue::Float(0.0), &tiny));
    assert!(!ordered(&tiny, &LiteralValue::Float(0.0)));
    assert!(ordered(&tiny, &LiteralValue::Float(f64::from_bits(1))));
    assert_eq!(
        check_bounds_are_ordered(
            &at_the_bound,
            &LiteralValue::Int(BigInt::from(10).pow(10_000) - 1),
            Inclusivity::Inclusive,
            Inclusivity::Inclusive,
        )
        .map_err(|error| matches!(error, ParamBuildError::UnorderedBounds)),
        Err(true)
    );
}

#[rstest]
#[case::trailing_zeros_move_into_the_exponent(1500, -3, "1.5")]
#[case::zero_has_no_exponent(0, 50_000, "0")]
#[case::normalized_at_the_bound(100, 9_998, &format!("1{}", "0".repeat(10_000)))]
fn from_parts_normalizes_before_the_bound(
    #[case] coefficient: i64,
    #[case] exponent: i64,
    #[case] text: &str,
) {
    let decimal = Decimal::from_parts(BigInt::from(coefficient), exponent).expect("in bounds");

    assert_eq!(decimal.to_string(), text);
    assert_eq!(decimal, text.parse().expect("a decimal"));
}

#[test]
fn from_parts_bounds_the_normalized_exponent() {
    assert_eq!(
        Decimal::from_parts(BigInt::from(100), 9_999),
        Err(DecimalPartsError::ExponentOutOfRange { exponent: 9_999 })
    );
    assert_eq!(
        Decimal::from_parts(BigInt::from(100), -10_002)
            .expect("normalized to 10^-10000")
            .exponent(),
        -10_000
    );
}

#[test]
fn from_parts_refuses_a_negative_coefficient() {
    assert_eq!(
        Decimal::from_parts(BigInt::from(-1), 0),
        Err(DecimalPartsError::NegativeCoefficient)
    );
    assert_eq!(
        DecimalPartsError::NegativeCoefficient.to_string(),
        "a decimal's coefficient must be non-negative"
    );
    assert_eq!(
        DecimalPartsError::ExponentOutOfRange { exponent: 10_001 }.to_string(),
        "decimal exponent 10001 exceeds the bound of 10000 in magnitude"
    );
}
