//! Text renderings that reproduce Python's own formatting byte for byte.
//!
//! Literal equivalence keys and printed expressions embed three renderings
//! whose exact text is part of their contract: a binary float
//! ([`format_float_repr`]), an exact decimal literal text after trailing-zero
//! normalization ([`normalize_decimal_text`] and
//! [`format_normalized_decimal`]), and a Boolean ([`format_bool`]).

use num_bigint::BigUint;

/// Largest decimal-point position, relative to the first significant digit,
/// at which a float is still written positionally.
const LARGEST_POSITIONAL_FLOAT_POINT: i64 = 16;

/// Smallest decimal-point position, relative to the first significant digit,
/// at which a float is still written positionally.
const SMALLEST_POSITIONAL_FLOAT_POINT: i64 = -3;

/// Smallest adjusted exponent at which a decimal is still written
/// positionally.
const SMALLEST_POSITIONAL_DECIMAL_ADJUSTED_EXPONENT: i64 = -6;

/// Fewest digits a float's scientific exponent is written with.
const FLOAT_EXPONENT_MINIMUM_DIGITS: usize = 2;

/// Number of explicit fraction bits in an `f64`.
const FLOAT_FRACTION_BITS: u32 = 52;

/// Binary exponent of the least significant bit of a subnormal `f64`.
const SMALLEST_FLOAT_BIT_EXPONENT: i64 = -1074;

/// Convert a digit count to an exponent offset.
fn convert_count_to_exponent(count: usize) -> i64 {
    i64::try_from(count).expect("a digit count fits in an i64 exponent")
}

/// Convert a non-negative exponent offset to a digit count.
fn convert_exponent_to_count(offset: i64) -> usize {
    usize::try_from(offset).expect("the offset is non-negative and bounded by a digit count")
}

/// Write `digits`, a significand whose decimal point sits `point` digits
/// after its first digit, in positional notation: `0.000ddd`, `ddd000`, or
/// `dd.d`, without a fractional part when the value is integral.
fn write_positional(digits: &str, point: i64, text: &mut String) {
    let length = convert_count_to_exponent(digits.len());
    if point <= 0 {
        text.push_str("0.");
        text.extend(std::iter::repeat_n('0', convert_exponent_to_count(-point)));
        text.push_str(digits);
    } else if point >= length {
        text.push_str(digits);
        text.extend(std::iter::repeat_n(
            '0',
            convert_exponent_to_count(point - length),
        ));
    } else {
        let (integer_part, fraction_part) = digits.split_at(convert_exponent_to_count(point));
        text.push_str(integer_part);
        text.push('.');
        text.push_str(fraction_part);
    }
}

/// Write `digits` in scientific notation as `d.ddd`, or `d` for a single
/// digit, without the exponent.
fn write_scientific_significand(digits: &str, text: &mut String) {
    let (first_digit, remaining_digits) = digits.split_at(1);
    text.push_str(first_digit);
    if !remaining_digits.is_empty() {
        text.push('.');
        text.push_str(remaining_digits);
    }
}

/// Write `marker`, the sign of `exponent`, and its magnitude zero-padded to
/// at least `minimum_digits` digits.
fn write_exponent(marker: char, exponent: i64, minimum_digits: usize, text: &mut String) {
    let magnitude = exponent.unsigned_abs().to_string();
    text.push(marker);
    text.push(if exponent < 0 { '-' } else { '+' });
    text.extend(std::iter::repeat_n(
        '0',
        minimum_digits.saturating_sub(magnitude.len()),
    ));
    text.push_str(&magnitude);
}

/// Return the integer significand `m` and binary exponent `q` with
/// `value == m * 2^q`, for a finite, non-negative `value`.
fn decode_float(value: f64) -> (u64, i64) {
    let bits = value.to_bits();
    let fraction = bits & ((1_u64 << FLOAT_FRACTION_BITS) - 1);
    let biased_exponent = i64::try_from(bits >> FLOAT_FRACTION_BITS)
        .expect("the exponent field of a non-negative float fits in an i64");
    if biased_exponent == 0 {
        (fraction, SMALLEST_FLOAT_BIT_EXPONENT)
    } else {
        (
            fraction | (1_u64 << FLOAT_FRACTION_BITS),
            biased_exponent + SMALLEST_FLOAT_BIT_EXPONENT - 1,
        )
    }
}

/// Return `base^exponent` as a big unsigned integer.
fn raise_to_power(base: u32, exponent: i64) -> BigUint {
    let exponent = u32::try_from(exponent).expect("float exponents are far below u32::MAX");
    BigUint::from(base).pow(exponent)
}

/// Return whether `value == numerator * 10^decimal_exponent / 2` exactly,
/// for a finite, non-negative `value`.
fn is_exactly_half_of(value: f64, numerator: u64, decimal_exponent: i64) -> bool {
    let (significand, binary_exponent) = decode_float(value);
    let mut left = BigUint::from(significand) * 2_u32;
    let mut right = BigUint::from(numerator);
    if binary_exponent >= 0 {
        left *= raise_to_power(2, binary_exponent);
    } else {
        right *= raise_to_power(2, -binary_exponent);
    }
    if decimal_exponent >= 0 {
        right *= raise_to_power(10, decimal_exponent);
    } else {
        left *= raise_to_power(10, -decimal_exponent);
    }
    left == right
}

/// Return whether `digits`, with the decimal point `point` digits after the
/// first digit, reads back as exactly `value`.
fn is_round_trip(digits: &str, point: i64, value: f64) -> bool {
    let (first_digit, remaining_digits) = digits.split_at(1);
    format!("{first_digit}.{remaining_digits}e{}", point - 1)
        .parse::<f64>()
        .is_ok_and(|parsed| parsed.to_bits() == value.to_bits())
}

/// Return the even-ending neighbor of the shortest digits `digits` when
/// `value` lies exactly halfway between the two and the neighbor also reads
/// back as `value`, or `None` otherwise.
///
/// The standard library's shortest formatting breaks an exact tie between
/// two equally short candidates upward; this text breaks it toward the even
/// last digit.
fn find_even_tie_digits(value: f64, digits: &str, point: i64) -> Option<String> {
    let significand: u64 = digits
        .parse()
        .expect("at most seventeen shortest float digits fit in a u64");
    if significand % 2 == 0 || significand == 1 {
        return None;
    }
    let decimal_exponent = point - convert_count_to_exponent(digits.len());
    if !is_exactly_half_of(value, 2 * significand - 1, decimal_exponent) {
        return None;
    }
    let lower = (significand - 1).to_string();
    let lower = lower.trim_end_matches('0');
    is_round_trip(lower, point, value).then(|| lower.to_owned())
}

/// Return the shortest round-trip digits of a finite, non-negative `value`
/// and the position of its decimal point relative to the first digit.
///
/// Of two equally short candidates equally close to `value`, the one ending
/// in an even digit is chosen.
fn find_shortest_digits(value: f64) -> (String, i64) {
    let scientific = format!("{value:e}");
    let (significand, exponent) = scientific
        .split_once('e')
        .expect("scientific formatting of a finite float writes an exponent");
    let exponent: i64 = exponent
        .parse()
        .expect("scientific formatting writes an integer exponent");
    let digits: String = significand.chars().filter(char::is_ascii_digit).collect();
    let point = exponent + 1;
    let digits = find_even_tie_digits(value, &digits, point).unwrap_or(digits);
    (digits, point)
}

/// Return whether `text` is a non-empty run of ASCII digits.
fn is_ascii_digit_run(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit())
}

/// Split a decimal literal text into its integer and fraction digits, or
/// return `None` when it is outside the unsigned, exponent-free grammar.
fn split_decimal_text(text: &str) -> Option<(&str, &str)> {
    let (integer_part, fraction_part) = text.split_once('.').unwrap_or((text, ""));
    let is_every_byte_a_digit = integer_part
        .bytes()
        .chain(fraction_part.bytes())
        .all(|byte| byte.is_ascii_digit());
    let has_a_digit = is_ascii_digit_run(integer_part) || is_ascii_digit_run(fraction_part);
    (is_every_byte_a_digit && has_a_digit).then_some((integer_part, fraction_part))
}

/// A non-negative decimal number with its trailing coefficient zeros removed.
///
/// The value is `digits * 10^exponent`. `digits` is a non-empty string of
/// ASCII digits with neither a leading nor a trailing zero, except that zero
/// itself is the coefficient `"0"` with exponent `0`. Two texts denote the
/// same number exactly when they normalize to equal values, however many
/// digits they carry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct NormalizedDecimal {
    digits: Box<str>,
    exponent: i64,
}

/// Format `value` as the shortest text that reads back as the same `f64`.
///
/// The digits are the shortest decimal digit string that round-trips to
/// `value`. With `n` the position of the decimal point relative to the first
/// digit (`n = 1` for `1.5`, `n = -3` for `0.0001`), the text is positional
/// when `-4 < n <= 16` and always carries a fractional part (`1.0`,
/// `1000000000000000.0`, `0.0001`). Otherwise it is scientific, with one
/// digit before an optional fractional part, a lowercase `e`, an explicit
/// exponent sign, and at least two exponent digits (`1e+16`, `1e-05`,
/// `1.2345678901234568e+17`, `5e-324`). A negative value, including
/// negative zero, starts with `-`. Infinities are `inf` and `-inf`, and every
/// NaN, whatever its sign or payload, is `nan`.
#[must_use]
pub(crate) fn format_float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_owned();
    }
    let mut text = String::new();
    if value.is_sign_negative() {
        text.push('-');
    }
    if value.is_infinite() {
        text.push_str("inf");
        return text;
    }
    let (digits, point) = find_shortest_digits(value.abs());
    if (SMALLEST_POSITIONAL_FLOAT_POINT..=LARGEST_POSITIONAL_FLOAT_POINT).contains(&point) {
        write_positional(&digits, point, &mut text);
        if point >= convert_count_to_exponent(digits.len()) {
            text.push_str(".0");
        }
    } else {
        let exponent = point - 1;
        write_scientific_significand(&digits, &mut text);
        write_exponent('e', exponent, FLOAT_EXPONENT_MINIMUM_DIGITS, &mut text);
    }
    text
}

/// Normalize a decimal literal text to its coefficient digits and exponent,
/// removing leading and trailing zeros without rounding at any length.
///
/// `text` must be an unsigned, exponent-free decimal in one of the forms
/// `[0-9]+`, `[0-9]+.[0-9]*`, or `.[0-9]+`, with ASCII digits only. Every
/// digit of `text` counts, so two texts that differ in their last
/// significant digit normalize apart.
///
/// Returns `None` when `text` is outside that grammar: empty, a bare `.`,
/// signed, with an exponent, with whitespace or a separator, or with a
/// non-ASCII digit.
#[must_use]
pub(crate) fn normalize_decimal_text(text: &str) -> Option<NormalizedDecimal> {
    let (integer_part, fraction_part) = split_decimal_text(text)?;
    let coefficient = format!("{integer_part}{fraction_part}");
    let significant = coefficient.trim_start_matches('0');
    let stripped = significant.trim_end_matches('0');
    if stripped.is_empty() {
        return Some(NormalizedDecimal {
            digits: "0".into(),
            exponent: 0,
        });
    }
    let trailing_zeros = significant.len() - stripped.len();
    Some(NormalizedDecimal {
        digits: stripped.into(),
        exponent: convert_count_to_exponent(trailing_zeros)
            - convert_count_to_exponent(fraction_part.len()),
    })
}

/// Format a normalized decimal in the general decimal notation.
///
/// With `k` the number of coefficient digits and `a = exponent + k - 1` the
/// adjusted exponent, the text is positional when `exponent <= 0` and
/// `a >= -6` (`1.5`, `0.000123`, `0.000001`, `0`). Otherwise it is
/// scientific: the first digit, then `.` and the remaining digits if there
/// are any, then a capital `E`, the sign of `a`, and `a` without padding
/// (`1E+2`, `1.2E+2`, `1E-7`, `3E-121`).
#[must_use]
pub(crate) fn format_normalized_decimal(decimal: &NormalizedDecimal) -> String {
    let digits = &*decimal.digits;
    let point = decimal.exponent + convert_count_to_exponent(digits.len());
    let adjusted_exponent = point - 1;
    let mut text = String::new();
    if decimal.exponent <= 0 && adjusted_exponent >= SMALLEST_POSITIONAL_DECIMAL_ADJUSTED_EXPONENT {
        write_positional(digits, point, &mut text);
    } else {
        write_scientific_significand(digits, &mut text);
        write_exponent('E', adjusted_exponent, 1, &mut text);
    }
    text
}

/// Format a Boolean as `True` or `False`.
#[must_use]
pub(crate) fn format_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use rstest::rstest;
    use serde::Deserialize;

    use super::*;

    const GOLDEN_JSON: &str = include_str!("../tests/golden/python_text_cases.json");

    const MIN_FLOAT_CASES: usize = 1000;

    const MIN_DECIMAL_CASES: usize = 300;

    /// One golden float: its bit pattern and the text the oracle printed.
    #[derive(Deserialize)]
    struct FloatCase {
        name: String,
        bits: String,
        repr: String,
    }

    /// One golden decimal text and, when accepted, its normalized form.
    #[derive(Deserialize)]
    struct DecimalCase {
        name: String,
        text: String,
        accepted: bool,
        digits: Option<String>,
        exponent: Option<i64>,
        #[serde(rename = "str")]
        rendered: Option<String>,
    }

    /// One golden Boolean and the text the oracle printed.
    #[derive(Deserialize)]
    struct BoolCase {
        value: bool,
        #[serde(rename = "str")]
        rendered: String,
    }

    /// The golden corpus of float, decimal and Boolean renderings.
    #[derive(Deserialize)]
    struct GoldenDocument {
        #[serde(rename = "float_repr_cases")]
        floats: Vec<FloatCase>,
        #[serde(rename = "decimal_cases")]
        decimals: Vec<DecimalCase>,
        #[serde(rename = "bool_cases")]
        bools: Vec<BoolCase>,
    }

    /// Normalize `text`, failing the test if it is outside the grammar.
    fn normalize_accepted_text(text: &str) -> NormalizedDecimal {
        normalize_decimal_text(text).unwrap_or_else(|| panic!("{text:?} is in the decimal grammar"))
    }

    /// Build the normalized decimal `digits * 10^exponent`.
    fn build_normalized_decimal(digits: &str, exponent: i64) -> NormalizedDecimal {
        NormalizedDecimal {
            digits: digits.into(),
            exponent,
        }
    }

    /// Record the mismatches between one golden float case and the Rust text.
    fn check_float_case(case: &FloatCase, mismatches: &mut Vec<String>) {
        let bits = u64::from_str_radix(&case.bits, 16).expect("golden bits are hexadecimal");

        let text = format_float_repr(f64::from_bits(bits));

        if text != case.repr {
            mismatches.push(format!(
                "float {} (bits {}): expected {:?}, got {text:?}",
                case.name, case.bits, case.repr
            ));
        }
    }

    /// Record the mismatches between one golden decimal case and the Rust
    /// normalization and text.
    fn check_decimal_case(case: &DecimalCase, mismatches: &mut Vec<String>) {
        let normalized = normalize_decimal_text(&case.text);

        match (case.accepted, normalized) {
            (false, None) => {}
            (false, Some(decimal)) => mismatches.push(format!(
                "decimal {} ({:?}): expected rejection, got {decimal:?}",
                case.name, case.text
            )),
            (true, None) => mismatches.push(format!(
                "decimal {} ({:?}): expected acceptance, got rejection",
                case.name, case.text
            )),
            (true, Some(decimal)) => {
                let expected_digits = case.digits.as_deref().expect("accepted case has digits");
                let expected_exponent = case.exponent.expect("accepted case has an exponent");
                let expected_text = case
                    .rendered
                    .as_deref()
                    .expect("accepted case has str text");
                if &*decimal.digits != expected_digits || decimal.exponent != expected_exponent {
                    mismatches.push(format!(
                        "decimal {} ({:?}): expected digits {expected_digits:?} exponent \
                         {expected_exponent}, got digits {:?} exponent {}",
                        case.name, case.text, decimal.digits, decimal.exponent
                    ));
                }
                let text = format_normalized_decimal(&decimal);
                if text != expected_text {
                    mismatches.push(format!(
                        "decimal {} ({:?}): expected text {expected_text:?}, got {text:?}",
                        case.name, case.text
                    ));
                }
            }
        }
    }

    /// Replay every case of a golden document, failing with all mismatches.
    fn replay_golden_document(json: &str, corpus_path: Option<&str>) {
        let document: GoldenDocument =
            serde_json::from_str(json).expect("golden data matches the corpus shape");
        assert!(
            document.floats.len() >= MIN_FLOAT_CASES,
            "expected at least {MIN_FLOAT_CASES} float cases, found {}",
            document.floats.len()
        );
        assert!(
            document.decimals.len() >= MIN_DECIMAL_CASES,
            "expected at least {MIN_DECIMAL_CASES} decimal cases, found {}",
            document.decimals.len()
        );
        assert_eq!(document.bools.len(), 2, "expected both Booleans");

        let mut mismatches = Vec::new();
        for case in &document.floats {
            check_float_case(case, &mut mismatches);
        }
        for case in &document.decimals {
            check_decimal_case(case, &mut mismatches);
        }
        for case in &document.bools {
            let text = format_bool(case.value);
            if text != case.rendered {
                mismatches.push(format!(
                    "bool {}: expected {:?}, got {text:?}",
                    case.value, case.rendered
                ));
            }
        }

        let location = corpus_path
            .map(|path| format!(" in {path}"))
            .unwrap_or_default();
        assert!(
            mismatches.is_empty(),
            "found {} mismatch(es){location}:\n{}",
            mismatches.len(),
            mismatches.join("\n")
        );
    }

    /// Test float formatting matches the oracle's text for representative
    /// values on each side of every notation boundary.
    #[rstest]
    #[case::zero(0.0, "0.0")]
    #[case::negative_zero(-0.0, "-0.0")]
    #[case::one(1.0, "1.0")]
    #[case::negative_one_and_a_half(-1.5, "-1.5")]
    #[case::one_tenth_plus_two_tenths(0.1 + 0.2, "0.30000000000000004")]
    #[case::one_third(1.0 / 3.0, "0.3333333333333333")]
    #[case::smallest_positional_exponent(0.0001, "0.0001")]
    #[case::just_below_positional_range(9.999_999_999_999_999e-5, "9.999999999999999e-05")]
    #[case::two_digit_negative_exponent(1e-5, "1e-05")]
    #[case::one_ten_millionth(1e-7, "1e-07")]
    #[case::largest_positional_integral(1e15, "1000000000000000.0")]
    #[case::sixteen_digit_integral(9_999_999_999_999_998.0, "9999999999999998.0")]
    #[case::first_scientific_integral(1e16, "1e+16")]
    #[case::negative_scientific_integral(-1e16, "-1e+16")]
    #[case::seventeen_significant_digits(123_456_789_012_345_680.0, "1.2345678901234568e+17")]
    #[case::large_power_of_ten(1e22, "1e+22")]
    #[case::three_digit_exponent(f64::MAX, "1.7976931348623157e+308")]
    #[case::smallest_normal(f64::MIN_POSITIVE, "2.2250738585072014e-308")]
    #[case::smallest_subnormal(f64::from_bits(1), "5e-324")]
    #[case::positive_infinity(f64::INFINITY, "inf")]
    #[case::negative_infinity(f64::NEG_INFINITY, "-inf")]
    fn format_float_repr_matches_the_oracle_text(#[case] value: f64, #[case] expected: &str) {
        let text = format_float_repr(value);

        assert_eq!(text, expected);
    }

    /// Test every NaN formats as `nan`, whatever its sign bit or payload.
    #[rstest]
    #[case::quiet(0x7ff8_0000_0000_0000)]
    #[case::negative_quiet(0xfff8_0000_0000_0000)]
    #[case::signaling_payload(0x7ff0_0000_0000_0001)]
    #[case::negative_full_payload(0xffff_ffff_ffff_ffff)]
    fn format_float_repr_writes_every_nan_as_nan(#[case] bits: u64) {
        let text = format_float_repr(f64::from_bits(bits));

        assert_eq!(text, "nan");
    }

    /// Test normalization strips leading zeros, trailing zeros and the point,
    /// and the result formats as the oracle prints it.
    #[rstest]
    #[case::integer("5", "5", 0, "5")]
    #[case::integer_leading_zero("05", "5", 0, "5")]
    #[case::integer_trailing_zeros("500", "5", 2, "5E+2")]
    #[case::integer_ten("10", "1", 1, "1E+1")]
    #[case::fraction("1.5", "15", -1, "1.5")]
    #[case::fraction_trailing_zero("1.50", "15", -1, "1.5")]
    #[case::fraction_leading_zero("01.5", "15", -1, "1.5")]
    #[case::fraction_without_integer_part(".5", "5", -1, "0.5")]
    #[case::integer_part_with_bare_point("5.", "5", 0, "5")]
    #[case::integral_fraction("1.0", "1", 0, "1")]
    #[case::trailing_zero_before_point("10.", "1", 1, "1E+1")]
    #[case::hundred_point_zero("100.0", "1", 2, "1E+2")]
    #[case::hundred_twenty_point_zero("120.0", "12", 1, "1.2E+2")]
    #[case::million_bare_point("1000000.", "1", 6, "1E+6")]
    #[case::adjusted_exponent_minus_six("0.000001", "1", -6, "0.000001")]
    #[case::adjusted_exponent_minus_seven("0.0000001", "1", -7, "1E-7")]
    #[case::two_digits_at_minus_seven("0.00000012", "12", -8, "1.2E-7")]
    #[case::trailing_zeros_small("0.00012300", "123", -6, "0.000123")]
    #[case::positional_fraction("123.45", "12345", -2, "123.45")]
    fn normalize_decimal_text_strips_zeros_and_formats_as_the_oracle(
        #[case] text: &str,
        #[case] expected_digits: &str,
        #[case] expected_exponent: i64,
        #[case] expected_text: &str,
    ) {
        let decimal = normalize_accepted_text(text);

        assert_eq!(&*decimal.digits, expected_digits);
        assert_eq!(decimal.exponent, expected_exponent);
        assert_eq!(format_normalized_decimal(&decimal), expected_text);
    }

    /// Test every spelling of zero normalizes to the coefficient `0` with
    /// exponent `0` and formats as `0`.
    #[rstest]
    #[case::integer("0")]
    #[case::integer_zeros("000")]
    #[case::fraction("0.0")]
    #[case::fraction_zeros("0.000")]
    #[case::bare_point_zero(".0")]
    #[case::zero_bare_point("0.")]
    #[case::many_fraction_zeros("00.0000000000")]
    fn normalize_decimal_text_folds_every_zero_spelling(#[case] text: &str) {
        let decimal = normalize_accepted_text(text);

        assert_eq!(&*decimal.digits, "0");
        assert_eq!(decimal.exponent, 0);
        assert_eq!(format_normalized_decimal(&decimal), "0");
    }

    /// Test texts differing only in a thirtieth significant digit normalize
    /// apart, keeping every digit.
    #[test]
    fn normalize_decimal_text_keeps_thirty_significant_digits_without_rounding() {
        let last_one = format!("1.{}1", "0".repeat(28));
        let last_two = format!("1.{}2", "0".repeat(28));

        let decimal_one = normalize_accepted_text(&last_one);
        let decimal_two = normalize_accepted_text(&last_two);

        assert_ne!(decimal_one, decimal_two);
        assert_eq!(&*decimal_one.digits, format!("1{}1", "0".repeat(28)));
        assert_eq!(decimal_one.exponent, -29);
        assert_eq!(format_normalized_decimal(&decimal_one), last_one);
    }

    /// Test a two-hundred-digit text keeps all its digits through
    /// normalization and formatting.
    #[test]
    fn normalize_decimal_text_keeps_two_hundred_digits() {
        let integer_part = "1234567890".repeat(10);
        let fraction_part = "0987654321".repeat(10);
        let text = format!("{integer_part}.{fraction_part}");

        let decimal = normalize_accepted_text(&text);

        assert_eq!(&*decimal.digits, format!("{integer_part}{fraction_part}"));
        assert_eq!(decimal.exponent, -100);
        assert_eq!(format_normalized_decimal(&decimal), text);
    }

    /// Test exponents far outside any fixed-width float range normalize and
    /// format exactly.
    #[rstest]
    #[case::tiny(&format!(".{}3", "0".repeat(120)), -121, "3E-121")]
    #[case::huge(&format!("1{}.0", "0".repeat(80)), 80, "1E+80")]
    fn normalize_decimal_text_handles_exponents_beyond_the_float_range(
        #[case] text: &str,
        #[case] expected_exponent: i64,
        #[case] expected_text: &str,
    ) {
        let decimal = normalize_accepted_text(text);

        assert_eq!(decimal.exponent, expected_exponent);
        assert_eq!(format_normalized_decimal(&decimal), expected_text);
    }

    /// Test texts outside the unsigned, exponent-free ASCII grammar are
    /// refused.
    #[rstest]
    #[case::empty("")]
    #[case::bare_point(".")]
    #[case::negative("-5")]
    #[case::explicit_plus("+5")]
    #[case::exponent("1e10")]
    #[case::fraction_exponent("1.5e3")]
    #[case::scientific_output_form("1E+2")]
    #[case::infinity("inf")]
    #[case::nan("NaN")]
    #[case::hexadecimal("0x1f")]
    #[case::blank("  ")]
    #[case::leading_space(" 5")]
    #[case::trailing_space("5 ")]
    #[case::trailing_newline("5\n")]
    #[case::two_points("1.2.3")]
    #[case::double_point("1..5")]
    #[case::underscore_separator("1_000")]
    #[case::comma("1,5")]
    #[case::letter("12a")]
    #[case::arabic_indic_digit("\u{0665}")]
    #[case::fullwidth_digits("\u{ff11}.\u{ff15}")]
    fn normalize_decimal_text_rejects_text_outside_the_grammar(#[case] text: &str) {
        let decimal = normalize_decimal_text(text);

        assert_eq!(decimal, None);
    }

    /// Test formatting a coefficient and exponent directly picks positional
    /// or scientific notation at the documented thresholds.
    #[rstest]
    #[case::zero("0", 0, "0")]
    #[case::positional_integer("12", 0, "12")]
    #[case::positional_fraction("12345", -2, "123.45")]
    #[case::adjusted_minus_six("123", -8, "0.00000123")]
    #[case::adjusted_minus_seven("123", -9, "1.23E-7")]
    #[case::positive_exponent_single_digit("1", 2, "1E+2")]
    #[case::positive_exponent_many_digits("12", 3, "1.2E+4")]
    fn format_normalized_decimal_chooses_notation_by_exponent(
        #[case] digits: &str,
        #[case] exponent: i64,
        #[case] expected: &str,
    ) {
        let decimal = build_normalized_decimal(digits, exponent);

        let text = format_normalized_decimal(&decimal);

        assert_eq!(text, expected);
    }

    /// Test Booleans format with a capitalized first letter.
    #[rstest]
    #[case::true_value(true, "True")]
    #[case::false_value(false, "False")]
    fn format_bool_capitalizes(#[case] value: bool, #[case] expected: &str) {
        let text = format_bool(value);

        assert_eq!(text, expected);
    }

    /// Test float, decimal and Boolean renderings reproduce every text the
    /// oracle recorded in the committed golden corpus.
    #[test]
    fn python_text_renderings_match_the_python_oracle() {
        replay_golden_document(GOLDEN_JSON, None);
    }

    /// Test the renderings reproduce every text the oracle recorded in an
    /// expanded corpus, read from the file named by `FHY_PYTHON_TEXT_CORPUS`.
    #[test]
    #[ignore = "requires an expanded corpus generated from the Python oracle"]
    fn python_text_renderings_match_the_python_oracle_on_an_expanded_corpus() {
        let path = std::env::var("FHY_PYTHON_TEXT_CORPUS")
            .expect("FHY_PYTHON_TEXT_CORPUS names an expanded corpus file");
        let json = std::fs::read_to_string(&path).expect("expanded corpus file is readable");

        replay_golden_document(&json, Some(&path));
    }

    /// Build a strategy over the digit strings of `1..=max_length` ASCII
    /// digits.
    fn generate_digit_strings(max_length: usize) -> impl Strategy<Value = String> {
        proptest::collection::vec(proptest::char::range('0', '9'), 1..=max_length)
            .prop_map(|digits| digits.into_iter().collect())
    }

    proptest! {
        /// Test every non-NaN float's text parses back to the identical bits.
        #[test]
        fn format_float_repr_round_trips_every_non_nan_float(bits in any::<u64>()) {
            let value = f64::from_bits(bits);
            prop_assume!(!value.is_nan());

            let text = format_float_repr(value);

            let reparsed: f64 = text.parse().expect("formatted float text parses");
            prop_assert_eq!(reparsed.to_bits(), bits, "text {:?}", text);
        }

        /// Test a formatted float never reads as an integer: it always has a
        /// point, an exponent, or is a non-finite word.
        #[test]
        fn format_float_repr_never_writes_integer_shaped_text(bits in any::<u64>()) {
            let text = format_float_repr(f64::from_bits(bits));

            prop_assert!(
                text.contains('.') || text.contains('e') || text.ends_with("inf") || text == "nan",
                "text {:?}", text
            );
        }

        /// Test padding a decimal text with leading zeros and trailing
        /// fractional zeros leaves its normalized form unchanged.
        #[test]
        fn normalize_decimal_text_ignores_zero_padding(
            integer_part in generate_digit_strings(30),
            fraction_part in generate_digit_strings(30),
            leading_zeros in 0_usize..8,
            trailing_zeros in 0_usize..8,
        ) {
            let plain = format!("{integer_part}.{fraction_part}");
            let padded = format!(
                "{}{integer_part}.{fraction_part}{}",
                "0".repeat(leading_zeros),
                "0".repeat(trailing_zeros)
            );

            let plain_decimal = normalize_decimal_text(&plain);
            let padded_decimal = normalize_decimal_text(&padded);

            prop_assert!(plain_decimal.is_some(), "{:?} is in the grammar", plain);
            prop_assert_eq!(padded_decimal, plain_decimal);
        }

        /// Test moving the decimal point left by `shift` places keeps the
        /// coefficient and lowers the exponent by `shift`.
        #[test]
        fn normalize_decimal_text_tracks_the_point_position_in_the_exponent(
            digits in generate_digit_strings(40),
            point in 0_usize..40,
            shift in 0_usize..40,
        ) {
            prop_assume!(digits.bytes().any(|digit| digit != b'0'));
            let right = point.min(digits.len());
            let left = right.saturating_sub(shift);
            let right_text = format!("{}.{}", &digits[..right], &digits[right..]);
            let left_text = format!("{}.{}", &digits[..left], &digits[left..]);

            let right_decimal = normalize_accepted_text(&right_text);
            let left_decimal = normalize_accepted_text(&left_text);

            prop_assert_eq!(&left_decimal.digits, &right_decimal.digits);
            let moved = i64::try_from(right - left).expect("shift fits in i64");
            prop_assert_eq!(left_decimal.exponent, right_decimal.exponent - moved);
        }

        /// Test a normalized coefficient is `0` or has neither a leading nor
        /// a trailing zero.
        #[test]
        fn normalize_decimal_text_yields_a_canonical_coefficient(
            integer_part in generate_digit_strings(30),
            fraction_part in generate_digit_strings(30),
        ) {
            let text = format!("{integer_part}.{fraction_part}");

            let decimal = normalize_accepted_text(&text);

            let digits = &*decimal.digits;
            if digits == "0" {
                prop_assert_eq!(decimal.exponent, 0);
            } else {
                prop_assert!(!digits.starts_with('0') && !digits.ends_with('0'), "{:?}", digits);
                prop_assert!(digits.bytes().all(|digit| digit.is_ascii_digit()), "{:?}", digits);
            }
        }
    }
}
