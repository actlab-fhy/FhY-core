//! Tests for literal values: their stored forms, the text grammar, the
//! equivalence buckets and canonical keys, the integer-bucket predicate,
//! `Display`, and the sorts a literal has.
//!
//! Public API only (`fhy_core::expr`).

#[path = "common/hashing.rs"]
pub mod hashing_support;

use fhy_core::expr::{BigInt, FunctionSort, LiteralKind, LiteralTextError, LiteralValue};
use hashing_support::hash_of;
use rstest::rstest;

/// Parse `text` as a literal, failing the test if it is refused.
fn parse_literal(text: &str) -> LiteralValue {
    LiteralValue::parse_text(text).expect("the text is a literal text")
}

/// Build a literal from a test-table spelling: `b:` Boolean, `i:` integer,
/// `f:` float, `t:` text.
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
// Stored forms
// =============================================================================

/// Test an integer literal keeps the integer it was built from.
#[rstest]
#[case::zero(0)]
#[case::five(5)]
#[case::negative(-3)]
#[case::million(1_000_000)]
#[case::minimum(i64::MIN)]
#[case::maximum(i64::MAX)]
fn literal_value_from_integer_keeps_the_integer(#[case] value: i64) {
    let literal = LiteralValue::from(value);

    assert_eq!(literal.kind(), LiteralKind::Int(&BigInt::from(value)));
}

/// Test an integer beyond the `i64` range is held exactly.
#[test]
fn literal_value_from_big_integer_keeps_every_digit() {
    let big: BigInt = "1000000000000000000000000000000".parse().expect("digits");

    let literal = LiteralValue::from(big.clone());

    assert_eq!(literal.kind(), LiteralKind::Int(&big));
    assert_eq!(
        literal.canonical_key(),
        "int:1000000000000000000000000000000"
    );
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

    let LiteralKind::Float(stored) = literal.kind() else {
        panic!("expected a float, got {:?}", literal.kind());
    };
    assert_eq!(stored.to_bits(), value.to_bits());
}

/// Test a NaN float literal keeps a NaN.
#[test]
fn literal_value_from_nan_keeps_a_nan() {
    let literal = LiteralValue::from(f64::NAN);

    assert!(matches!(literal.kind(), LiteralKind::Float(stored) if stored.is_nan()));
}

/// Test a Boolean literal keeps the Boolean, through either constructor.
#[rstest]
#[case::true_value(true)]
#[case::false_value(false)]
fn literal_value_from_bool_keeps_the_boolean(#[case] value: bool) {
    let from_trait = LiteralValue::from(value);
    let from_constructor = LiteralValue::from_bool(value);

    assert_eq!(from_trait.kind(), LiteralKind::Bool(value));
    assert_eq!(from_constructor.kind(), LiteralKind::Bool(value));
}

/// Test an integer text is kept verbatim, leading zeros included.
#[rstest]
#[case::zero("0")]
#[case::five("5")]
#[case::forty_two("42")]
#[case::double_zero("00")]
#[case::leading_zero("01")]
fn literal_value_parse_text_keeps_integer_text_verbatim(#[case] text: &str) {
    let literal = parse_literal(text);

    assert_eq!(literal.kind(), LiteralKind::IntegerText(text));
}

/// Test a decimal text is kept verbatim, in every accepted shape.
#[rstest]
#[case::pi_ish("3.14")]
#[case::zero("0.0")]
#[case::trailing_point("1.")]
#[case::leading_point(".5")]
#[case::one_tenth("0.1")]
#[case::fraction("100.001")]
fn literal_value_parse_text_keeps_decimal_text_verbatim(#[case] text: &str) {
    let literal = parse_literal(text);

    assert_eq!(literal.kind(), LiteralKind::DecimalText(text));
}

/// Test a text outside the integer and decimal grammar is refused, and the
/// error names the text.
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

/// Test the grammar error writes the refused text quoted.
#[test]
fn literal_text_error_display_names_the_text() {
    let error: LiteralTextError = LiteralValue::parse_text("1e10").expect_err("an exponent");

    assert_eq!(
        error.to_string(),
        "invalid literal text \"1e10\": expected ASCII digits with at most one decimal point"
    );
}

// =============================================================================
// Equivalence buckets
// =============================================================================

/// Test two integers with different values are unequal in both directions.
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

/// Test an integer equals an integer text of the same value.
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
}

/// Test integer texts spelling one value differently are equal.
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
}

/// Test decimal texts spelling one decimal differently are equal.
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
}

/// Test literals from different buckets are unequal even when their numbers
/// agree.
#[rstest]
#[case::decimal_text_and_float("t:1.5", "f:1.5")]
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

/// Test two decimals differing only in their thirtieth digit stay unequal.
#[test]
fn literal_value_decimals_differing_in_the_thirtieth_digit_are_unequal() {
    let left = parse_literal(&format!("1.{}1", "0".repeat(28)));
    let right = parse_literal(&format!("1.{}2", "0".repeat(28)));

    assert_ne!(left, right);
    assert_ne!(right, left);
}

/// Test the keys of two decimals differing only in their thirtieth digit
/// differ.
#[test]
fn literal_value_canonical_key_differs_in_the_thirtieth_digit() {
    let left = parse_literal(&format!("1.{}1", "0".repeat(28)));
    let right = parse_literal(&format!("1.{}2", "0".repeat(28)));

    assert_ne!(left.canonical_key(), right.canonical_key());
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

/// Test negative zero equals zero and shares its key.
#[test]
fn literal_value_negative_zero_equals_zero() {
    let zero = LiteralValue::from(0.0);
    let negative_zero = LiteralValue::from(-0.0);

    assert_eq!(zero, negative_zero);
    assert_eq!(zero.canonical_key(), negative_zero.canonical_key());
}

/// Every bucket, `-0.0`, NaNs of both signs, and decimals longer than 28
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

/// Test two literals are equal exactly when their keys are equal, across a
/// sample spanning every bucket.
#[test]
fn literal_value_canonical_key_is_shared_exactly_by_equal_literals() {
    let literals: Vec<(&str, LiteralValue)> = EQUIVALENCE_SAMPLE
        .iter()
        .map(|spec| (*spec, build_sample_literal(spec)))
        .collect();
    let mut disagreements = Vec::new();

    for (left_spec, left) in &literals {
        for (right_spec, right) in &literals {
            let equal = left == right;
            let keyed_alike = left.canonical_key() == right.canonical_key();
            if equal != keyed_alike {
                disagreements.push(format!("{left_spec} vs {right_spec}: equal={equal}"));
            }
        }
    }

    assert!(disagreements.is_empty(), "{disagreements:#?}");
}

/// Test equal literals hash equally.
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

/// Test distinct literals, within a bucket and across buckets, are unequal
/// both ways and hash differently.
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
#[case::integer_text_and_decimal_text("t:5", "t:5.0")]
#[case::float_and_decimal_text("f:1.5", "t:1.5")]
fn literal_value_distinct_literals_are_unequal_and_hash_differently(
    #[case] left: &str,
    #[case] right: &str,
) {
    let left = build_sample_literal(left);
    let right = build_sample_literal(right);

    assert_ne!(left, right);
    assert_ne!(right, left);
    assert_ne!(hash_of(&left), hash_of(&right));
}

/// Test the key text of representative literals in every bucket.
#[rstest]
#[case::bool_true("b:true", "bool:True")]
#[case::bool_false("b:false", "bool:False")]
#[case::integer_text("t:05", "int:5")]
#[case::negative_integer("i:-5", "int:-5")]
#[case::zero_text("t:000", "int:0")]
#[case::negative_zero("f:-0.0", "float-binary:0.0")]
#[case::nan("f:NaN", "float-binary:nan")]
#[case::negative_infinity("f:-inf", "float-binary:-inf")]
#[case::float_notation_boundary("f:1e16", "float-binary:1e+16")]
#[case::float_below_notation_boundary("f:1e15", "float-binary:1000000000000000.0")]
#[case::small_float("f:0.00001", "float-binary:1e-05")]
#[case::decimal_trailing_zero("t:1.50", "float-decimal:1.5")]
#[case::decimal_whole_number("t:100.0", "float-decimal:1E+2")]
#[case::decimal_one_millionth("t:0.000001", "float-decimal:0.000001")]
#[case::decimal_one_ten_millionth("t:0.0000001", "float-decimal:1E-7")]
#[case::decimal_zero("t:0.000", "float-decimal:0")]
fn literal_value_canonical_key_renders_bucket_and_canonical_form(
    #[case] spec: &str,
    #[case] expected_key: &str,
) {
    let literal = build_sample_literal(spec);

    let key = literal.canonical_key();

    assert_eq!(key, expected_key);
}

// =============================================================================
// Integer-bucket predicate
// =============================================================================

/// Test the predicate holds for both spellings of an integer literal.
#[rstest]
#[case::integer_zero("i:0")]
#[case::integer("i:5")]
#[case::text_zero("t:0")]
#[case::text("t:5")]
#[case::text_leading_zero("t:05")]
#[case::negative_integer("i:-7")]
fn literal_value_is_integer_valued_for_every_integer_bucket_form(#[case] spec: &str) {
    let literal = build_sample_literal(spec);

    assert!(literal.is_integer_valued());
}

/// Test the predicate fails for the Boolean and both float buckets, even
/// without a fractional part.
#[rstest]
#[case::bool_true("b:true")]
#[case::bool_false("b:false")]
#[case::float("f:5.0")]
#[case::float_fraction("f:1.5")]
#[case::decimal("t:5.0")]
#[case::decimal_fraction("t:1.5")]
#[case::decimal_without_integer_part("t:.5")]
fn literal_value_is_not_integer_valued_for_other_buckets(#[case] spec: &str) {
    let literal = build_sample_literal(spec);

    assert!(!literal.is_integer_valued());
}

/// Test the predicate answers alike for equal literals.
#[rstest]
#[case::integer_and_text("i:5", "t:5")]
#[case::text_and_padded_text("t:5", "t:05")]
#[case::float_pair("f:1.5", "f:1.5")]
#[case::decimal_pair("t:1.5", "t:1.50")]
#[case::bool_pair("b:true", "b:true")]
fn literal_value_is_integer_valued_agrees_across_equal_literals(
    #[case] left: &str,
    #[case] right: &str,
) {
    let left = build_sample_literal(left);
    let right = build_sample_literal(right);
    assert_eq!(left, right);

    assert_eq!(left.is_integer_valued(), right.is_integer_valued());
}

// =============================================================================
// Display
// =============================================================================

/// Test `Display` writes each literal as it was given.
#[rstest]
#[case::bool_true("b:true", "True")]
#[case::bool_false("b:false", "False")]
#[case::integer("i:5", "5")]
#[case::negative_integer("i:-3", "-3")]
#[case::big_integer(
    "i:-10000000000000000000000000000000000000000",
    "-10000000000000000000000000000000000000000"
)]
#[case::float("f:1.5", "1.5")]
#[case::integral_float("f:5.0", "5.0")]
#[case::large_float("f:1e16", "1e+16")]
#[case::negative_zero("f:-0.0", "-0.0")]
#[case::nan("f:NaN", "nan")]
#[case::infinity("f:inf", "inf")]
#[case::integer_text("t:05", "05")]
#[case::decimal_text("t:1.50", "1.50")]
#[case::decimal_text_trailing_point("t:1.", "1.")]
fn literal_value_display_writes_the_value_as_given(#[case] spec: &str, #[case] expected: &str) {
    let literal = build_sample_literal(spec);

    let text = literal.to_string();

    assert_eq!(text, expected);
}

// =============================================================================
// Sorts
// =============================================================================

/// Test the Boolean sort accepts both Booleans.
#[rstest]
#[case::true_value(true)]
#[case::false_value(false)]
fn function_sort_bool_accepts_boolean_literals(#[case] value: bool) {
    let literal = LiteralValue::from(value);

    assert!(FunctionSort::Bool.accepts_literal(&literal));
}

/// Test the Boolean sort rejects every number.
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

/// Test the natural sort accepts non-negative integers and integer texts.
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

/// Test the natural sort rejects negative integers.
#[rstest]
#[case::minus_one(-1)]
#[case::minus_hundred(-100)]
fn function_sort_nat_rejects_negative_integers(#[case] value: i64) {
    let literal = LiteralValue::from(value);

    assert!(!FunctionSort::Nat.accepts_literal(&literal));
}

/// Test the natural sort rejects Booleans.
#[rstest]
#[case::true_value(true)]
#[case::false_value(false)]
fn function_sort_nat_rejects_booleans(#[case] value: bool) {
    let literal = LiteralValue::from(value);

    assert!(!FunctionSort::Nat.accepts_literal(&literal));
}

/// Test the natural sort rejects floats and decimal texts, integral or not.
#[rstest]
#[case::zero("f:0.0")]
#[case::fraction("f:1.5")]
#[case::negative("f:-2.5")]
#[case::decimal_text("t:5.0")]
fn function_sort_nat_rejects_floats(#[case] spec: &str) {
    let literal = build_sample_literal(spec);

    assert!(!FunctionSort::Nat.accepts_literal(&literal));
}

/// Test the integer sort accepts every integer and integer text.
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

/// Test the integer sort rejects Booleans.
#[rstest]
#[case::true_value(true)]
#[case::false_value(false)]
fn function_sort_int_rejects_booleans(#[case] value: bool) {
    let literal = LiteralValue::from(value);

    assert!(!FunctionSort::Int.accepts_literal(&literal));
}

/// Test the integer sort rejects floats and decimal texts.
#[rstest]
#[case::zero("f:0.0")]
#[case::fraction("f:1.5")]
#[case::negative("f:-2.5")]
#[case::decimal_text("t:5.0")]
fn function_sort_int_rejects_floats(#[case] spec: &str) {
    let literal = build_sample_literal(spec);

    assert!(!FunctionSort::Int.accepts_literal(&literal));
}

/// Test the real sort accepts integers, floats, and both kinds of text.
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

/// Test the real sort rejects Booleans.
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
// Edge cases across every bucket
// =============================================================================

/// Test the canonical key, the integer-bucket verdict, and the verdicts of
/// the Boolean, natural, integer and real sorts (in that order) of edge-case
/// literals from every bucket.
#[rstest]
#[case::bool_true("b:true", "bool:True", false, [true, false, false, false])]
#[case::integer_zero("i:0", "int:0", true, [false, true, true, true])]
#[case::integer_above_u64("i:18446744073709551616", "int:18446744073709551616", true, [false, true, true, true])]
#[case::integer_forty_one_digits(
    "i:10000000000000000000000000000000000000000",
    "int:10000000000000000000000000000000000000000",
    true,
    [false, true, true, true]
)]
#[case::negative_integer_forty_one_digits(
    "i:-10000000000000000000000000000000000000000",
    "int:-10000000000000000000000000000000000000000",
    true,
    [false, false, true, true]
)]
#[case::integer_text_zero("t:0", "int:0", true, [false, true, true, true])]
#[case::integer_text_fifty_digits(
    "t:12345678901234567890123456789012345678901234567890",
    "int:12345678901234567890123456789012345678901234567890",
    true,
    [false, true, true, true]
)]
#[case::decimal_text_zero("t:0.0", "float-decimal:0", false, [false, false, false, true])]
#[case::decimal_text_leading_zero("t:0.5", "float-decimal:0.5", false, [false, false, false, true])]
#[case::decimal_text_bare_leading_point("t:.5", "float-decimal:0.5", false, [false, false, false, true])]
#[case::decimal_text_bare_trailing_point("t:5.", "float-decimal:5", false, [false, false, false, true])]
#[case::decimal_text_forty_digits(
    "t:1.000000000000000000000000000000000000001",
    "float-decimal:1.000000000000000000000000000000000000001",
    false,
    [false, false, false, true]
)]
#[case::float_fraction("f:1.5", "float-binary:1.5", false, [false, false, false, true])]
#[case::negative_float_fraction("f:-1.5", "float-binary:-1.5", false, [false, false, false, true])]
#[case::float_zero("f:0.0", "float-binary:0.0", false, [false, false, false, true])]
#[case::float_one_tenth("f:0.1", "float-binary:0.1", false, [false, false, false, true])]
#[case::float_huge("f:1e300", "float-binary:1e+300", false, [false, false, false, true])]
#[case::positive_infinity("f:inf", "float-binary:inf", false, [false, false, false, true])]
#[case::negative_nan("f:-NaN", "float-binary:nan", false, [false, false, false, true])]
#[case::smallest_subnormal("f:5e-324", "float-binary:5e-324", false, [false, false, false, true])]
#[case::three_smallest_subnormals("f:1.5e-323", "float-binary:1.5e-323", false, [false, false, false, true])]
#[case::even_tie("f:667929902981260.2", "float-binary:667929902981260.2", false, [false, false, false, true])]
fn literal_value_edge_case_has_its_key_bucket_and_sorts(
    #[case] spec: &str,
    #[case] expected_key: &str,
    #[case] expected_integer_valued: bool,
    #[case] expected_sort_verdicts: [bool; 4],
) {
    let literal = build_sample_literal(spec);

    let key = literal.canonical_key();
    let integer_valued = literal.is_integer_valued();
    let sort_verdicts = [
        FunctionSort::Bool,
        FunctionSort::Nat,
        FunctionSort::Int,
        FunctionSort::Real,
    ]
    .map(|sort| sort.accepts_literal(&literal));

    assert_eq!(key, expected_key);
    assert_eq!(integer_valued, expected_integer_valued);
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
