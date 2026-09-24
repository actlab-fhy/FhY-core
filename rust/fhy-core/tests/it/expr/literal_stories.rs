//! Tests for literal values: their variants, the text grammar and its
//! normalization, equality and hashing, `Display`, and the sorts a literal
//! has.

use crate::support::hashing as hashing_support;

use fhy_core::expr::{BigInt, Decimal, FunctionSort, LiteralTextError, LiteralValue};
use hashing_support::hash_of;
use rstest::rstest;

fn parse_literal(text: &str) -> LiteralValue {
    LiteralValue::parse_text(text).expect("the text is a literal text")
}

/// Build a literal from a test-table spelling: `b:` Boolean, `i:` integer,
/// `f:` float, `t:` parsed text.
fn build_sample_literal(spec: &str) -> LiteralValue {
    let (kind, value) = spec.split_at(2);
    match kind {
        "b:" => LiteralValue::from(value == "true"),
        "i:" => LiteralValue::from(value.parse::<BigInt>().expect("an integer")),
        "f:" => LiteralValue::from(value.parse::<f64>().expect("a float")),
        "t:" => parse_literal(value),
        _ => panic!("unknown literal spec {spec:?}"),
    }
}

// =============================================================================
// Variants
// =============================================================================

#[rstest]
#[case::zero(0)]
#[case::five(5)]
#[case::negative(-3)]
#[case::million(1_000_000)]
#[case::minimum(i64::MIN)]
#[case::maximum(i64::MAX)]
fn literal_value_from_integer_keeps_the_integer(#[case] value: i64) {
    let literal = LiteralValue::from(value);

    assert!(matches!(&literal, LiteralValue::Int(stored) if *stored == BigInt::from(value)));
}

#[test]
fn literal_value_from_every_integer_type_builds_an_integer() {
    let literals = [
        (LiteralValue::from(-7_i32), BigInt::from(-7)),
        (LiteralValue::from(-7_i64), BigInt::from(-7)),
        (LiteralValue::from(-7_i128), BigInt::from(-7)),
        (LiteralValue::from(7_u32), BigInt::from(7)),
        (LiteralValue::from(7_u64), BigInt::from(7)),
        (LiteralValue::from(7_usize), BigInt::from(7)),
        (LiteralValue::from(i128::MAX), BigInt::from(i128::MAX)),
        (LiteralValue::from(u64::MAX), BigInt::from(u64::MAX)),
    ];

    for (literal, expected) in &literals {
        assert!(
            matches!(literal, LiteralValue::Int(stored) if stored == expected),
            "{literal:?} holds {expected}"
        );
    }
}

/// Test an integer beyond the `i64` range is held exactly.
#[test]
fn literal_value_from_big_integer_keeps_every_digit() {
    let big: BigInt = "1000000000000000000000000000000".parse().expect("digits");

    let literal = LiteralValue::from(big.clone());

    assert!(matches!(&literal, LiteralValue::Int(stored) if *stored == big));
    assert_eq!(literal.to_string(), "1000000000000000000000000000000");
}

/// Test a float literal keeps the float's exact bits.
#[rstest]
#[case::zero(0.0)]
#[case::fraction(2.75)]
#[case::negative(-2.5)]
#[case::tiny(1e-10)]
#[case::negative_zero(-0.0)]
#[case::infinity(f64::INFINITY)]
fn literal_value_from_float_keeps_the_float(#[case] value: f64) {
    let literal = LiteralValue::from(value);

    let LiteralValue::Float(stored) = literal else {
        panic!("expected a float, got {literal:?}");
    };
    assert_eq!(stored.to_bits(), value.to_bits());
}

#[test]
fn literal_value_from_nan_keeps_a_nan() {
    let literal = LiteralValue::from(f64::NAN);

    assert!(matches!(literal, LiteralValue::Float(stored) if stored.is_nan()));
}

#[rstest]
#[case::true_value(true)]
#[case::false_value(false)]
fn literal_value_from_bool_keeps_the_boolean(#[case] value: bool) {
    let literal = LiteralValue::from(value);

    assert!(matches!(literal, LiteralValue::Bool(stored) if stored == value));
}

/// Test an integer text parses to the integer it spells, leading zeros
/// dropped.
#[rstest]
#[case::zero("0", 0)]
#[case::five("5", 5)]
#[case::forty_two("42", 42)]
#[case::double_zero("00", 0)]
#[case::triple_zero("000", 0)]
#[case::leading_zero("01", 1)]
#[case::padded_five("05", 5)]
fn literal_value_parse_text_normalizes_integer_text(#[case] text: &str, #[case] expected: i64) {
    let literal = parse_literal(text);

    assert!(
        matches!(&literal, LiteralValue::Int(stored) if *stored == BigInt::from(expected)),
        "{text:?} parses to {literal:?}"
    );
}

/// Test a decimal text of every accepted shape parses to its normalized
/// decimal: no trailing zeros in the coefficient, and exponent `0` for zero.
#[rstest]
#[case::pi_ish("3.14", 314, -2)]
#[case::trailing_zero("1.50", 15, -1)]
#[case::zero("0.0", 0, 0)]
#[case::trailing_point("1.", 1, 0)]
#[case::leading_point(".5", 5, -1)]
#[case::one_tenth("0.1", 1, -1)]
#[case::hundred("100.0", 1, 2)]
#[case::fraction("100.001", 100_001, -3)]
fn literal_value_parse_text_normalizes_decimal_text(
    #[case] text: &str,
    #[case] expected_coefficient: i64,
    #[case] expected_exponent: i64,
) {
    let literal = parse_literal(text);

    let LiteralValue::Decimal(decimal) = &literal else {
        panic!("{text:?} parses to {literal:?}, not a decimal");
    };
    assert_eq!(decimal.coefficient(), &BigInt::from(expected_coefficient));
    assert_eq!(decimal.exponent(), expected_exponent);
}

/// Test parsing keeps no spelling: the spellings of one value parse to
/// equal literals of one variant with one `Display`.
#[test]
fn literal_value_parse_text_normalizes_spelling() {
    let padded_five = parse_literal("05");
    let decimals = ["1.50", "1.5", "01.500"].map(parse_literal);
    let hundred = parse_literal("100.0");

    assert!(matches!(&padded_five, LiteralValue::Int(value) if *value == BigInt::from(5)));
    assert_eq!(padded_five.to_string(), "5");
    for decimal in &decimals {
        assert!(matches!(decimal, LiteralValue::Decimal(_)), "{decimal:?}");
        assert_eq!(decimal, &decimals[0]);
        assert_eq!(decimal.to_string(), "1.5");
    }
    assert_eq!(hundred.to_string(), "100");
}

/// Test a text outside the grammar is refused with an error naming the text.
#[rstest]
#[case::word("not_a_number")]
#[case::infinity("inf")]
#[case::negative_infinity("-inf")]
#[case::infinity_word("Infinity")]
#[case::nan("NaN")]
#[case::exponent("1e10")]
#[case::hex("0x1f")]
#[case::negative("-5")]
#[case::positive_sign("+5")]
#[case::fraction_exponent("5.5e2")]
#[case::empty("")]
#[case::blank("  ")]
#[case::trailing_space("5 ")]
#[case::bare_point(".")]
#[case::two_points("1.2.3")]
#[case::trailing_newline("5\n")]
#[case::arabic_indic_digit("\u{665}")]
#[case::fullwidth_digits("\u{ff11}.\u{ff15}")]
fn literal_value_parse_text_rejects_text_outside_the_grammar(#[case] text: &str) {
    let result = LiteralValue::parse_text(text);

    let Err(error) = &result else {
        panic!("expected a refusal of {text:?}, got {result:?}");
    };
    assert_eq!(error.text(), text);
}

#[test]
fn literal_text_error_display_names_the_text() {
    let error: LiteralTextError = LiteralValue::parse_text("1e10").expect_err("an exponent");

    assert_eq!(
        error.to_string(),
        "invalid literal text \"1e10\": expected ASCII digits with at most one decimal point"
    );
}

// =============================================================================
// Equality and hashing
// =============================================================================

#[test]
fn literal_value_differs_when_values_differ() {
    let smaller = LiteralValue::from(5);
    let larger = LiteralValue::from(10);

    assert_ne!(smaller, larger);
    assert_ne!(larger, smaller);
}

/// Test a float and a decimal text of the same number are unequal.
#[rstest]
#[case::one(1.0, "1.0")]
#[case::one_and_a_half(1.5, "1.5")]
#[case::half(0.5, "0.5")]
fn literal_value_float_differs_from_decimal_text(#[case] float: f64, #[case] text: &str) {
    let binary = LiteralValue::from(float);
    let decimal = parse_literal(text);

    assert_ne!(binary, decimal);
    assert_ne!(decimal, binary);
}

/// Test Booleans, integers and floats of equal numeric value are unequal.
#[rstest]
#[case::int_zero_and_false("i:0", "b:false")]
#[case::int_one_and_true("i:1", "b:true")]
#[case::int_zero_and_float_zero("i:0", "f:0.0")]
#[case::int_one_and_float_one("i:1", "f:1.0")]
#[case::false_and_float_zero("b:false", "f:0.0")]
#[case::true_and_float_one("b:true", "f:1.0")]
fn literal_value_differs_across_bool_int_and_float(#[case] left: &str, #[case] right: &str) {
    let left = build_sample_literal(left);
    let right = build_sample_literal(right);

    assert_ne!(left, right);
    assert_ne!(right, left);
}

#[rstest]
#[case::zero(0, "0")]
#[case::one(1, "1")]
#[case::forty_two(42, "42")]
#[case::zero_padded_zero(0, "00")]
#[case::zero_padded_one(1, "01")]
fn literal_value_integer_equals_integer_text(#[case] integer: i64, #[case] text: &str) {
    let left = LiteralValue::from(integer);
    let right = parse_literal(text);

    assert_eq!(left, right);
    assert_eq!(right, left);
    assert!(matches!(right, LiteralValue::Int(_)), "{right:?}");
    assert_eq!(left.to_string(), right.to_string());
}

#[rstest]
#[case::leading_zero("5", "05")]
#[case::zero_forms("00", "0")]
#[case::larger_value("42", "042")]
fn literal_value_integer_texts_with_distinct_spelling_are_equal(
    #[case] left: &str,
    #[case] right: &str,
) {
    let left = parse_literal(left);
    let right = parse_literal(right);

    assert_eq!(left, right);
    assert_eq!(right, left);
    assert!(matches!(
        (&left, &right),
        (LiteralValue::Int(_), LiteralValue::Int(_))
    ));
    assert_eq!(left.to_string(), right.to_string());
}

#[rstest]
#[case::trailing_zero("1.5", "1.50")]
#[case::trailing_zero_after_leading_zero("0.1", "0.10")]
#[case::missing_leading_zero(".5", "0.5")]
#[case::trailing_point("1.", "1.0")]
#[case::leading_zero("1.5", "01.5")]
fn literal_value_decimal_texts_with_distinct_spelling_are_equal(
    #[case] left: &str,
    #[case] right: &str,
) {
    let left = parse_literal(left);
    let right = parse_literal(right);

    assert_eq!(left, right);
    assert_eq!(right, left);
    assert!(matches!(
        (&left, &right),
        (LiteralValue::Decimal(_), LiteralValue::Decimal(_))
    ));
    assert_eq!(left.to_string(), right.to_string());
}

/// Test literals of different variants are unequal even when their numbers
/// agree.
#[rstest]
#[case::exact_decimal_and_float_approximation("t:0.1", "f:0.1")]
#[case::integer_text_and_decimal_text("t:5", "t:5.0")]
#[case::integer_and_decimal_text("i:5", "t:5.0")]
#[case::integer_text_and_float("t:5", "f:5.0")]
fn literal_value_differs_across_buckets(#[case] left: &str, #[case] right: &str) {
    let left = build_sample_literal(left);
    let right = build_sample_literal(right);

    assert_ne!(left, right);
    assert_ne!(right, left);
}

/// Test every NaN equals every other NaN, whatever its sign or payload.
#[test]
fn literal_value_nan_equals_every_nan() {
    let quiet = LiteralValue::from(f64::NAN);
    let computed = LiteralValue::from(f64::INFINITY - f64::INFINITY);
    let negative_payload = LiteralValue::from(f64::from_bits(0xFFFF_FFFF_FFFF_FFFF));

    assert_eq!(quiet, computed);
    assert_eq!(computed, quiet);
    assert_eq!(quiet, negative_payload);
}

/// Test a NaN literal equals its own clone, so equality is reflexive.
#[test]
fn literal_value_nan_equals_its_clone() {
    let literal = LiteralValue::from(f64::NAN);

    let copy = literal.clone();

    assert_eq!(literal, copy);
}

/// Test negative zero equals zero and hashes alike.
#[test]
fn literal_value_negative_zero_equals_zero() {
    let zero = LiteralValue::from(0.0);
    let negative_zero = LiteralValue::from(-0.0);

    assert_eq!(zero, negative_zero);
    assert_eq!(hash_of(&zero), hash_of(&negative_zero));
}

/// Every variant, `-0.0`, NaNs of both signs, and decimals longer than 28
/// significant digits.
const EQUIVALENCE_SAMPLE: [&str; 22] = [
    "b:true",
    "b:false",
    "i:0",
    "i:5",
    "t:5",
    "t:05",
    "f:0.0",
    "f:-0.0",
    "f:5.0",
    "f:1.5",
    "f:inf",
    "f:-inf",
    "f:NaN",
    "f:-NaN",
    "t:0.0",
    "t:.0",
    "t:5.0",
    "t:1.5",
    "t:1.50",
    "t:1.00000000000000000000000000001",
    "t:1.00000000000000000000000000002",
    "t:1.0",
];

/// Test equality over a sample spanning every variant is an equivalence,
/// relates only literals of one variant, and agrees with hashing.
#[test]
fn literal_value_equality_over_every_variant_is_an_equivalence_agreeing_with_hash() {
    let literals: Vec<(&str, LiteralValue)> = EQUIVALENCE_SAMPLE
        .iter()
        .map(|spec| (*spec, build_sample_literal(spec)))
        .collect();
    let mut violations = Vec::new();

    for (left_spec, left) in &literals {
        if *left != left.clone() {
            violations.push(format!("{left_spec} is not equal to itself"));
        }
        for (right_spec, right) in &literals {
            let equal = left == right;
            if equal != (right == left) {
                violations.push(format!("{left_spec} vs {right_spec}: not symmetric"));
            }
            if equal && std::mem::discriminant(left) != std::mem::discriminant(right) {
                violations.push(format!("{left_spec} vs {right_spec}: variants differ"));
            }
            if equal && hash_of(left) != hash_of(right) {
                violations.push(format!("{left_spec} vs {right_spec}: hashes differ"));
            }
            for (middle_spec, middle) in &literals {
                if equal && right == middle && left != middle {
                    violations.push(format!(
                        "{left_spec} vs {right_spec} vs {middle_spec}: not transitive"
                    ));
                }
            }
        }
    }

    assert!(violations.is_empty(), "{violations:#?}");
}

#[rstest]
#[case::integer_and_text("i:5", "t:05")]
#[case::decimal_spellings("t:1.5", "t:1.50")]
#[case::zeros("f:0.0", "f:-0.0")]
#[case::nans("f:NaN", "f:-NaN")]
fn literal_value_equal_literals_hash_equally(#[case] left: &str, #[case] right: &str) {
    let left = build_sample_literal(left);
    let right = build_sample_literal(right);

    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
}

/// Test distinct literals, of one variant and across variants, are unequal
/// both ways.
#[rstest]
#[case::booleans("b:true", "b:false")]
#[case::integers("i:0", "i:5")]
#[case::integer_texts("t:5", "t:6")]
#[case::floats("f:1.5", "f:5.0")]
#[case::infinities("f:inf", "f:-inf")]
#[case::decimal_texts("t:1.5", "t:5.0")]
#[case::decimals_in_the_thirtieth_digit(
    "t:1.00000000000000000000000000001",
    "t:1.00000000000000000000000000002"
)]
#[case::integer_and_float("i:5", "f:5.0")]
fn literal_value_distinct_literals_are_unequal(#[case] left: &str, #[case] right: &str) {
    let left = build_sample_literal(left);
    let right = build_sample_literal(right);

    assert_ne!(left, right);
    assert_ne!(right, left);
}

// =============================================================================
// Display
// =============================================================================

/// Test `Display` follows Rust conventions: `true` and `false`, integer
/// digits, a float as `{}` writes an `f64`, and a decimal positionally, and
/// never the Python spellings (`True`, `1e+16`, `nan`, `1E+2`, `05`).
#[rstest]
#[case::bool_true("b:true", "true")]
#[case::bool_false("b:false", "false")]
#[case::integer("i:5", "5")]
#[case::negative_integer("i:-3", "-3")]
#[case::big_integer(
    "i:-10000000000000000000000000000000000000000",
    "-10000000000000000000000000000000000000000"
)]
#[case::float("f:1.5", "1.5")]
#[case::integral_float("f:1.0", "1")]
#[case::large_float("f:1e16", "10000000000000000")]
#[case::larger_float("f:1.2345678901234568e17", "123456789012345680")]
#[case::small_float("f:0.00001", "0.00001")]
#[case::negative_zero("f:-0.0", "-0")]
#[case::nan("f:NaN", "NaN")]
#[case::infinity("f:inf", "inf")]
#[case::negative_infinity("f:-inf", "-inf")]
#[case::integer_text("t:05", "5")]
#[case::decimal_text("t:1.50", "1.5")]
#[case::decimal_text_trailing_point("t:1.", "1")]
#[case::decimal_hundred("t:100.0", "100")]
#[case::decimal_thousandth("t:0.001", "0.001")]
#[case::decimal_ten_millionth("t:0.0000001", "0.0000001")]
#[case::decimal_zero("t:0.000", "0")]
#[case::decimal_leading_point("t:.5", "0.5")]
fn literal_value_display_follows_rust_conventions(#[case] spec: &str, #[case] expected: &str) {
    let literal = build_sample_literal(spec);

    let text = literal.to_string();

    assert_eq!(text, expected);
}

/// Test a decimal displays positionally, normalized, and never longer than
/// one character more than the text it was parsed from.
#[rstest]
#[case::integral("5", "5")]
#[case::trailing_zeros("500", "500")]
#[case::fraction("1.50", "1.5")]
#[case::leading_zeros("0001.5", "1.5")]
#[case::leading_point(".5", "0.5")]
#[case::bare_point("5.", "5")]
#[case::tiny(".000000000000000000003", "0.000000000000000000003")]
#[case::huge("1000000000000000000000000.0", "1000000000000000000000000")]
#[case::zero("00.00", "0")]
fn decimal_display_is_positional_and_normalized(#[case] text: &str, #[case] expected: &str) {
    let decimal: Decimal = text.parse().expect("the text is in the literal grammar");

    let displayed = decimal.to_string();

    assert_eq!(displayed, expected);
    assert!(
        displayed.len() <= text.len() + 1,
        "{displayed:?} from {text:?}"
    );
    assert_eq!(displayed.parse::<Decimal>(), Ok(decimal));
}

/// Test a decimal parses from any text in the literal grammar, integer
/// texts included, and refuses the rest with the literal-text error.
#[rstest]
#[case::integer_text("5", Ok((5, 0)))]
#[case::decimal_text("2.50", Ok((25, -1)))]
#[case::signed("-1.5", Err("-1.5"))]
#[case::exponent("1e3", Err("1e3"))]
fn decimal_from_str_accepts_exactly_the_literal_grammar(
    #[case] text: &str,
    #[case] expected: Result<(i64, i64), &str>,
) {
    let parsed = text.parse::<Decimal>();

    let parsed = parsed
        .as_ref()
        .map(|decimal| (decimal.coefficient().clone(), decimal.exponent()))
        .map_err(LiteralTextError::text);
    let expected = expected.map(|(coefficient, exponent)| (BigInt::from(coefficient), exponent));
    assert_eq!(parsed, expected);
}

// =============================================================================
// Sorts
// =============================================================================

#[rstest]
#[case::true_value(true)]
#[case::false_value(false)]
fn function_sort_bool_accepts_boolean_literals(#[case] value: bool) {
    let literal = LiteralValue::from(value);

    assert!(FunctionSort::Bool.accepts_literal(&literal));
}

#[rstest]
#[case::zero("i:0")]
#[case::one("i:1")]
#[case::negative_one("i:-1")]
#[case::float_zero("f:0.0")]
#[case::float("f:1.5")]
#[case::negative_float("f:-2.5")]
#[case::integer_text("t:1")]
#[case::decimal_text("t:1.5")]
fn function_sort_bool_rejects_numeric_literals(#[case] spec: &str) {
    let literal = build_sample_literal(spec);

    assert!(!FunctionSort::Bool.accepts_literal(&literal));
}

#[rstest]
#[case::zero("i:0")]
#[case::one("i:1")]
#[case::two("i:2")]
#[case::hundred("i:100")]
#[case::text("t:007")]
#[case::big("i:100000000000000000000000000000")]
fn function_sort_nat_accepts_non_negative_integers(#[case] spec: &str) {
    let literal = build_sample_literal(spec);

    assert!(FunctionSort::Nat.accepts_literal(&literal));
}

#[rstest]
#[case::minus_one(-1)]
#[case::minus_hundred(-100)]
fn function_sort_nat_rejects_negative_integers(#[case] value: i64) {
    let literal = LiteralValue::from(value);

    assert!(!FunctionSort::Nat.accepts_literal(&literal));
}

#[rstest]
#[case::true_value(true)]
#[case::false_value(false)]
fn function_sort_nat_rejects_booleans(#[case] value: bool) {
    let literal = LiteralValue::from(value);

    assert!(!FunctionSort::Nat.accepts_literal(&literal));
}

/// Test the natural sort rejects floats and decimals, integral or not.
#[rstest]
#[case::zero("f:0.0")]
#[case::fraction("f:1.5")]
#[case::negative("f:-2.5")]
#[case::decimal_text("t:5.0")]
fn function_sort_nat_rejects_floats(#[case] spec: &str) {
    let literal = build_sample_literal(spec);

    assert!(!FunctionSort::Nat.accepts_literal(&literal));
}

#[rstest]
#[case::zero("i:0")]
#[case::one("i:1")]
#[case::negative_one("i:-1")]
#[case::hundred("i:100")]
#[case::negative_hundred("i:-100")]
#[case::text("t:05")]
fn function_sort_int_accepts_integers(#[case] spec: &str) {
    let literal = build_sample_literal(spec);

    assert!(FunctionSort::Int.accepts_literal(&literal));
}

#[rstest]
#[case::true_value(true)]
#[case::false_value(false)]
fn function_sort_int_rejects_booleans(#[case] value: bool) {
    let literal = LiteralValue::from(value);

    assert!(!FunctionSort::Int.accepts_literal(&literal));
}

/// Test the integer sort rejects floats and decimals.
#[rstest]
#[case::zero("f:0.0")]
#[case::fraction("f:1.5")]
#[case::negative("f:-2.5")]
#[case::decimal_text("t:5.0")]
fn function_sort_int_rejects_floats(#[case] spec: &str) {
    let literal = build_sample_literal(spec);

    assert!(!FunctionSort::Int.accepts_literal(&literal));
}

/// Test the real sort accepts integers, floats, and decimals.
#[rstest]
#[case::zero("i:0")]
#[case::one("i:1")]
#[case::negative_one("i:-1")]
#[case::float_zero("f:0.0")]
#[case::fraction("f:1.5")]
#[case::negative_fraction("f:-2.5")]
#[case::integer_text("t:5")]
#[case::decimal_text("t:1.50")]
fn function_sort_real_accepts_numbers(#[case] spec: &str) {
    let literal = build_sample_literal(spec);

    assert!(FunctionSort::Real.accepts_literal(&literal));
}

#[rstest]
#[case::true_value(true)]
#[case::false_value(false)]
fn function_sort_real_rejects_booleans(#[case] value: bool) {
    let literal = LiteralValue::from(value);

    assert!(!FunctionSort::Real.accepts_literal(&literal));
}

/// Test the real sort accepts the infinities and NaN.
#[rstest]
#[case::infinity(f64::INFINITY)]
#[case::negative_infinity(f64::NEG_INFINITY)]
#[case::nan(f64::NAN)]
fn function_sort_real_accepts_special_floats(#[case] value: f64) {
    let literal = LiteralValue::from(value);

    assert!(FunctionSort::Real.accepts_literal(&literal));
}

// =============================================================================
// Edge cases across every variant
// =============================================================================

/// Test the variant and the verdicts of the Boolean, natural, integer and
/// real sorts (in that order) of edge-case literals of every variant.
#[rstest]
#[case::bool_true("b:true", "bool", [true, false, false, false])]
#[case::integer_zero("i:0", "int", [false, true, true, true])]
#[case::integer_above_u64("i:18446744073709551616", "int", [false, true, true, true])]
#[case::integer_forty_one_digits("i:10000000000000000000000000000000000000000", "int", [false, true, true, true])]
#[case::negative_integer_forty_one_digits("i:-10000000000000000000000000000000000000000", "int", [false, false, true, true])]
#[case::integer_text_zero("t:0", "int", [false, true, true, true])]
#[case::integer_text_fifty_digits("t:12345678901234567890123456789012345678901234567890", "int", [false, true, true, true])]
#[case::decimal_text_zero("t:0.0", "decimal", [false, false, false, true])]
#[case::decimal_text_leading_zero("t:0.5", "decimal", [false, false, false, true])]
#[case::decimal_text_bare_leading_point("t:.5", "decimal", [false, false, false, true])]
#[case::decimal_text_bare_trailing_point("t:5.", "decimal", [false, false, false, true])]
#[case::decimal_text_forty_digits("t:1.000000000000000000000000000000000000001", "decimal", [false, false, false, true])]
#[case::float_fraction("f:1.5", "float", [false, false, false, true])]
#[case::negative_float_fraction("f:-1.5", "float", [false, false, false, true])]
#[case::float_zero("f:0.0", "float", [false, false, false, true])]
#[case::float_one_tenth("f:0.1", "float", [false, false, false, true])]
#[case::float_huge("f:1e300", "float", [false, false, false, true])]
#[case::positive_infinity("f:inf", "float", [false, false, false, true])]
#[case::negative_nan("f:-NaN", "float", [false, false, false, true])]
#[case::smallest_subnormal("f:5e-324", "float", [false, false, false, true])]
#[case::three_smallest_subnormals("f:1.5e-323", "float", [false, false, false, true])]
#[case::even_tie("f:667929902981260.2", "float", [false, false, false, true])]
fn literal_value_edge_case_has_its_variant_and_sorts(
    #[case] spec: &str,
    #[case] expected_variant: &str,
    #[case] expected_sort_verdicts: [bool; 4],
) {
    let literal = build_sample_literal(spec);

    let variant = match literal {
        LiteralValue::Bool(_) => "bool",
        LiteralValue::Int(_) => "int",
        LiteralValue::Float(_) => "float",
        LiteralValue::Decimal(_) => "decimal",
    };
    let sort_verdicts = [
        FunctionSort::Bool,
        FunctionSort::Nat,
        FunctionSort::Int,
        FunctionSort::Real,
    ]
    .map(|sort| sort.accepts_literal(&literal));

    assert_eq!(variant, expected_variant);
    assert_eq!(sort_verdicts, expected_sort_verdicts);
}

/// Test the re-exported `BigInt` is num-bigint's own type, so a value built
/// with num-bigint becomes a literal without conversion.
#[test]
fn big_int_re_export_is_the_num_bigint_type() {
    let from_num_bigint: num_bigint::BigInt =
        "123456789012345678901234567890".parse().expect("digits");

    let re_exported: BigInt = from_num_bigint.clone();

    assert_eq!(
        LiteralValue::from(re_exported),
        LiteralValue::from(from_num_bigint)
    );
}
