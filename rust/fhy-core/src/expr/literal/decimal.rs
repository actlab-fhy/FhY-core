//! Exact normalization of decimal literal text.
//!
//! A decimal literal's value is its text with leading and trailing zeros
//! removed, at any length and without rounding, so literal equality and
//! hashing compare [`NormalizedDecimal`]s. [`format_normalized_decimal`]
//! writes one back in the general decimal notation.

/// Smallest adjusted exponent at which a decimal is still written
/// positionally.
const SMALLEST_POSITIONAL_DECIMAL_ADJUSTED_EXPONENT: i64 = -6;

/// Convert a digit count to an exponent offset.
pub(super) fn convert_count_to_exponent(count: usize) -> i64 {
    i64::try_from(count).expect("a digit count fits in an i64 exponent")
}

/// Convert a non-negative exponent offset to a digit count.
fn convert_exponent_to_count(offset: i64) -> usize {
    usize::try_from(offset).expect("the offset is non-negative and bounded by a digit count")
}

/// Write `digits`, a significand whose decimal point sits `point` digits
/// after its first digit, in positional notation: `0.000ddd`, `ddd000`, or
/// `dd.d`, without a fractional part when the value is integral.
pub(super) fn write_positional(digits: &str, point: i64, text: &mut String) {
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
pub(super) fn write_scientific_significand(digits: &str, text: &mut String) {
    let (first_digit, remaining_digits) = digits.split_at(1);
    text.push_str(first_digit);
    if !remaining_digits.is_empty() {
        text.push('.');
        text.push_str(remaining_digits);
    }
}

/// Write `marker`, the sign of `exponent`, and its magnitude zero-padded to
/// at least `minimum_digits` digits.
pub(super) fn write_exponent(
    marker: char,
    exponent: i64,
    minimum_digits: usize,
    text: &mut String,
) {
    let magnitude = exponent.unsigned_abs().to_string();
    text.push(marker);
    text.push(if exponent < 0 { '-' } else { '+' });
    text.extend(std::iter::repeat_n(
        '0',
        minimum_digits.saturating_sub(magnitude.len()),
    ));
    text.push_str(&magnitude);
}

/// Return whether `text` is a non-empty run of ASCII digits.
pub(super) fn is_ascii_digit_run(text: &str) -> bool {
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
pub(super) struct NormalizedDecimal {
    digits: Box<str>,
    exponent: i64,
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
pub(super) fn normalize_decimal_text(text: &str) -> Option<NormalizedDecimal> {
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
/// Matches the Python implementation: this is the text `str(Decimal)`
/// writes.
///
/// With `k` the number of coefficient digits and `a = exponent + k - 1` the
/// adjusted exponent, the text is positional when `exponent <= 0` and
/// `a >= -6` (`1.5`, `0.000123`, `0.000001`, `0`). Otherwise it is
/// scientific: the first digit, then `.` and the remaining digits if there
/// are any, then a capital `E`, the sign of `a`, and `a` without padding
/// (`1E+2`, `1.2E+2`, `1E-7`, `3E-121`).
#[must_use]
pub(super) fn format_normalized_decimal(decimal: &NormalizedDecimal) -> String {
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

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use rstest::rstest;

    use super::*;

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

    /// Test normalization strips leading zeros, trailing zeros and the point,
    /// and the result formats in the general decimal notation.
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
    #[case::three_fraction_digits("123.456", "123456", -3, "123.456")]
    #[case::fraction_trailing_zero_after_one("0.10", "1", -1, "0.1")]
    #[case::adjusted_exponent_minus_twenty_one("0.000000000000000000001", "1", -21, "1E-21")]
    #[case::forty_ones(
        "1111111111111111111111111111111111111111",
        "1111111111111111111111111111111111111111",
        0,
        "1111111111111111111111111111111111111111"
    )]
    #[case::thirty_one_significant(
        "1.000000000000000000000000000001",
        "1000000000000000000000000000001",
        -30,
        "1.000000000000000000000000000001"
    )]
    #[case::sixty_nines(
        "999999999999999999999999999999.999999999999999999999999999999",
        "999999999999999999999999999999999999999999999999999999999999",
        -30,
        "999999999999999999999999999999.999999999999999999999999999999"
    )]
    #[case::forty_digits_with_trailing_zeros(
        "1234567890123456789012345678901234567890.0000000000",
        "123456789012345678901234567890123456789",
        1,
        "1.23456789012345678901234567890123456789E+39"
    )]
    #[case::fifty_digits_after_long_leading_zeros(
        "0.0000000000000000000012345678901234567890123456789012345678901234567890",
        "1234567890123456789012345678901234567890123456789",
        -69,
        "1.234567890123456789012345678901234567890123456789E-21"
    )]
    #[case::fifty_digits_with_long_trailing_zeros(
        "12345678901234567890123456789012345678901234567890000000000",
        "1234567890123456789012345678901234567890123456789",
        10,
        "1.234567890123456789012345678901234567890123456789E+58"
    )]
    fn normalize_decimal_text_strips_zeros_and_formats_the_result(
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

    /// Build a strategy over the digit strings of `1..=max_length` ASCII
    /// digits.
    fn generate_digit_strings(max_length: usize) -> impl Strategy<Value = String> {
        generate_digit_strings_in(1..=max_length)
    }

    /// Build a strategy over the digit strings whose length is in `lengths`.
    fn generate_digit_strings_in(
        lengths: std::ops::RangeInclusive<usize>,
    ) -> impl Strategy<Value = String> {
        proptest::collection::vec(proptest::char::range('0', '9'), lengths)
            .prop_map(|digits| digits.into_iter().collect())
    }

    /// Build a strategy over the parts of a decimal text in every form of
    /// the grammar: the integer digits and, when the text has a point, the
    /// fraction digits (`[0-9]+`, `[0-9]+.[0-9]*`, or `.[0-9]+`).
    fn generate_decimal_parts() -> impl Strategy<Value = (String, Option<String>)> {
        prop_oneof![
            generate_digit_strings(30).prop_map(|integer_part| (integer_part, None)),
            (
                generate_digit_strings(30),
                generate_digit_strings_in(0..=30)
            )
                .prop_map(|(integer_part, fraction_part)| (integer_part, Some(fraction_part))),
            generate_digit_strings(30)
                .prop_map(|fraction_part| (String::new(), Some(fraction_part))),
        ]
    }

    /// Join decimal parts into their text.
    fn join_decimal_parts(integer_part: &str, fraction_part: Option<&str>) -> String {
        match fraction_part {
            Some(fraction_part) => format!("{integer_part}.{fraction_part}"),
            None => integer_part.to_owned(),
        }
    }

    /// Return the normalized decimal a text in the general decimal notation
    /// writes: an exponent-free decimal text, optionally followed by `E` and
    /// a signed exponent.
    fn read_general_decimal_text(text: &str) -> NormalizedDecimal {
        let (mantissa, exponent) =
            text.split_once('E')
                .map_or((text, 0), |(mantissa, exponent)| {
                    (
                        mantissa,
                        exponent.parse::<i64>().expect("a signed exponent"),
                    )
                });
        let mut decimal = normalize_accepted_text(mantissa);
        if &*decimal.digits != "0" {
            decimal.exponent += exponent;
        }
        decimal
    }

    proptest! {
        /// Test a normalized decimal's general-notation text reads back as
        /// the same decimal, from a decimal text or from coefficient digits
        /// and an exponent far outside the positional range.
        #[test]
        fn format_normalized_decimal_reads_back_as_the_same_decimal(
            decimal in prop_oneof![
                generate_decimal_parts().prop_map(|(integer_part, fraction_part)| {
                    normalize_accepted_text(&join_decimal_parts(&integer_part, fraction_part.as_deref()))
                }),
                ("[1-9]([0-9]{0,30}[1-9])?", -400_i64..400).prop_map(|(digits, exponent)| {
                    build_normalized_decimal(&digits, exponent)
                }),
            ],
        ) {
            let text = format_normalized_decimal(&decimal);

            prop_assert_eq!(read_general_decimal_text(&text), decimal, "text {:?}", text);
        }

        /// Test padding a decimal text with leading zeros and trailing
        /// fractional zeros leaves its normalized form unchanged.
        #[test]
        fn normalize_decimal_text_ignores_zero_padding(
            (integer_part, fraction_part) in generate_decimal_parts(),
            leading_zeros in 0_usize..8,
            trailing_zeros in 0_usize..8,
        ) {
            let plain = join_decimal_parts(&integer_part, fraction_part.as_deref());
            let padded_fraction = fraction_part.map_or_else(
                || (trailing_zeros > 0).then(|| "0".repeat(trailing_zeros)),
                |fraction_part| Some(format!("{fraction_part}{}", "0".repeat(trailing_zeros))),
            );
            let padded = join_decimal_parts(
                &format!("{}{integer_part}", "0".repeat(leading_zeros)),
                padded_fraction.as_deref(),
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
            (integer_part, fraction_part) in generate_decimal_parts(),
        ) {
            let text = join_decimal_parts(&integer_part, fraction_part.as_deref());

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
