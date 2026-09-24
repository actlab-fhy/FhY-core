//! Exact decimals, normalized from their literal text.
//!
//! A decimal's value is its text with leading and trailing zeros removed,
//! at any length and without rounding, so decimal equality and hashing
//! compare normalized coefficients and exponents. `Display` writes the
//! value back positionally.

use std::fmt;
use std::str::FromStr;

use num_bigint::BigInt;
use serde::de::{self, Deserializer, Visitor};
use serde::{Deserialize, Serialize, Serializer};

use super::LiteralTextError;

fn convert_count_to_exponent(count: usize) -> i64 {
    i64::try_from(count).expect("a digit count fits in an i64 exponent")
}

fn convert_exponent_to_count(offset: i64) -> usize {
    usize::try_from(offset).expect("the offset is non-negative and bounded by a digit count")
}

fn write_zeros(f: &mut impl fmt::Write, count: usize) -> fmt::Result {
    (0..count).try_for_each(|_| f.write_char('0'))
}

/// Write `digits`, a significand whose decimal point sits `point` digits
/// after its first digit, in the positional notation of [`Decimal`]'s
/// `Display`.
fn write_positional(digits: &str, point: i64, f: &mut impl fmt::Write) -> fmt::Result {
    let length = convert_count_to_exponent(digits.len());
    if point <= 0 {
        f.write_str("0.")?;
        write_zeros(f, convert_exponent_to_count(-point))?;
        f.write_str(digits)
    } else if point >= length {
        f.write_str(digits)?;
        write_zeros(f, convert_exponent_to_count(point - length))
    } else {
        let (integer_part, fraction_part) = digits.split_at(convert_exponent_to_count(point));
        f.write_str(integer_part)?;
        f.write_char('.')?;
        f.write_str(fraction_part)
    }
}

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

/// A non-negative exact decimal, normalized.
///
/// The value is `coefficient * 10^exponent`. The coefficient is
/// non-negative and has no trailing zero, except that zero itself is the
/// coefficient `0` with exponent `0`. Two texts denote the same number
/// exactly when they parse to equal decimals, however many digits they
/// carry, so equality and hashing compare values.
///
/// A decimal is read from its literal text with [`FromStr`]: ASCII digits
/// with at most one decimal point and at least one digit, such as `"1.50"`,
/// `"1."`, `".5"` or `"5"`. `Display` writes it positionally, with no
/// exponent and without leading or trailing zeros: `1.5`, `100` for
/// `"100.0"`, `0.001`, `0`, and `0.5` for `".5"`. The text is at most one
/// character longer than the text the decimal was parsed from, and reads
/// back as the same decimal. A decimal serializes as that text and
/// deserializes from any text [`FromStr`] reads.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::{BigInt, Decimal};
///
/// let decimal: Decimal = "001.500".parse()?;
///
/// assert_eq!(decimal.coefficient(), &BigInt::from(15));
/// assert_eq!(decimal.exponent(), -1);
/// assert_eq!(decimal.to_string(), "1.5");
/// assert_eq!(decimal, "1.5".parse()?);
/// # Ok::<(), fhy_core::expr::LiteralTextError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Decimal {
    coefficient: BigInt,
    exponent: i64,
}

impl Decimal {
    /// Return the coefficient: the decimal is `coefficient * 10^exponent`.
    #[must_use]
    pub fn coefficient(&self) -> &BigInt {
        &self.coefficient
    }

    /// Return the exponent: the decimal is `coefficient * 10^exponent`.
    #[must_use]
    pub fn exponent(&self) -> i64 {
        self.exponent
    }
}

impl FromStr for Decimal {
    type Err = LiteralTextError;

    /// Normalize a decimal literal text, removing leading and trailing
    /// zeros without rounding at any length.
    ///
    /// The text must be an unsigned, exponent-free decimal in one of the
    /// forms `[0-9]+`, `[0-9]+.[0-9]*`, or `.[0-9]+`, with ASCII digits
    /// only. Every digit counts, so two texts that differ in their last
    /// significant digit parse apart.
    ///
    /// # Errors
    ///
    /// Returns [`LiteralTextError`] when `text` is outside that grammar:
    /// empty, a bare `.`, signed, with an exponent, with whitespace or a
    /// separator, or with a non-ASCII digit.
    fn from_str(text: &str) -> Result<Self, LiteralTextError> {
        let refuse = || LiteralTextError { text: text.into() };
        let (integer_part, fraction_part) = split_decimal_text(text).ok_or_else(refuse)?;
        let digits = format!("{integer_part}{fraction_part}");
        let significant = digits.trim_start_matches('0');
        let stripped = significant.trim_end_matches('0');
        if stripped.is_empty() {
            return Ok(Self {
                coefficient: BigInt::ZERO,
                exponent: 0,
            });
        }
        let trailing_zeros = significant.len() - stripped.len();
        let coefficient = BigInt::parse_bytes(stripped.as_bytes(), 10).ok_or_else(refuse)?;
        Ok(Self {
            coefficient,
            exponent: convert_count_to_exponent(trailing_zeros)
                - convert_count_to_exponent(fraction_part.len()),
        })
    }
}

impl fmt::Display for Decimal {
    /// Write the value positionally: `0.000ddd`, `ddd000`, or `dd.d`, with
    /// no fractional part when the value is integral.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let digits = self.coefficient.to_string();
        let point = self.exponent + convert_count_to_exponent(digits.len());
        write_positional(&digits, point, f)
    }
}

/// Serializes as the [`Display`](fmt::Display) text.
impl Serialize for Decimal {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

struct DecimalTextVisitor;

impl Visitor<'_> for DecimalTextVisitor {
    type Value = Decimal;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("the text of a decimal as a string")
    }

    fn visit_str<E: de::Error>(self, text: &str) -> Result<Decimal, E> {
        text.parse().map_err(E::custom)
    }
}

/// Deserializes from a string in the literal grammar, as
/// [`FromStr`](std::str::FromStr) parses it.
impl<'de> Deserialize<'de> for Decimal {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_str(DecimalTextVisitor)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use rstest::rstest;

    use super::*;

    fn parse_accepted_text(text: &str) -> Decimal {
        text.parse()
            .unwrap_or_else(|_| panic!("{text:?} is in the decimal grammar"))
    }

    fn build_decimal(digits: &str, exponent: i64) -> Decimal {
        Decimal {
            coefficient: digits.parse().expect("coefficient digits"),
            exponent,
        }
    }

    #[rstest]
    #[case::integer("5", "5", 0, "5")]
    #[case::integer_leading_zero("05", "5", 0, "5")]
    #[case::integer_trailing_zeros("500", "5", 2, "500")]
    #[case::integer_ten("10", "1", 1, "10")]
    #[case::fraction("1.5", "15", -1, "1.5")]
    #[case::fraction_trailing_zero("1.50", "15", -1, "1.5")]
    #[case::fraction_leading_zero("01.5", "15", -1, "1.5")]
    #[case::fraction_without_integer_part(".5", "5", -1, "0.5")]
    #[case::integer_part_with_bare_point("5.", "5", 0, "5")]
    #[case::integral_fraction("1.0", "1", 0, "1")]
    #[case::trailing_zero_before_point("10.", "1", 1, "10")]
    #[case::hundred_point_zero("100.0", "1", 2, "100")]
    #[case::hundred_twenty_point_zero("120.0", "12", 1, "120")]
    #[case::million_bare_point("1000000.", "1", 6, "1000000")]
    #[case::one_millionth("0.000001", "1", -6, "0.000001")]
    #[case::one_ten_millionth("0.0000001", "1", -7, "0.0000001")]
    #[case::two_digits_small("0.00000012", "12", -8, "0.00000012")]
    #[case::trailing_zeros_small("0.00012300", "123", -6, "0.000123")]
    #[case::positional_fraction("123.45", "12345", -2, "123.45")]
    #[case::three_fraction_digits("123.456", "123456", -3, "123.456")]
    #[case::fraction_trailing_zero_after_one("0.10", "1", -1, "0.1")]
    #[case::tiny("0.000000000000000000001", "1", -21, "0.000000000000000000001")]
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
        "1234567890123456789012345678901234567890"
    )]
    #[case::fifty_digits_after_long_leading_zeros(
        "0.0000000000000000000012345678901234567890123456789012345678901234567890",
        "1234567890123456789012345678901234567890123456789",
        -69,
        "0.000000000000000000001234567890123456789012345678901234567890123456789"
    )]
    #[case::fifty_digits_with_long_trailing_zeros(
        "12345678901234567890123456789012345678901234567890000000000",
        "1234567890123456789012345678901234567890123456789",
        10,
        "12345678901234567890123456789012345678901234567890000000000"
    )]
    fn decimal_from_str_strips_zeros_and_displays_the_result(
        #[case] text: &str,
        #[case] expected_digits: &str,
        #[case] expected_exponent: i64,
        #[case] expected_text: &str,
    ) {
        let decimal = parse_accepted_text(text);

        assert_eq!(decimal.coefficient.to_string(), expected_digits);
        assert_eq!(decimal.exponent, expected_exponent);
        assert_eq!(decimal.to_string(), expected_text);
    }

    #[rstest]
    #[case::integer("0")]
    #[case::integer_zeros("000")]
    #[case::fraction("0.0")]
    #[case::fraction_zeros("0.000")]
    #[case::bare_point_zero(".0")]
    #[case::zero_bare_point("0.")]
    #[case::many_fraction_zeros("00.0000000000")]
    fn decimal_from_str_folds_every_zero_spelling(#[case] text: &str) {
        let decimal = parse_accepted_text(text);

        assert_eq!(decimal.coefficient, BigInt::ZERO);
        assert_eq!(decimal.exponent, 0);
        assert_eq!(decimal.to_string(), "0");
    }

    #[test]
    fn decimal_from_str_keeps_thirty_significant_digits_without_rounding() {
        let last_one = format!("1.{}1", "0".repeat(28));
        let last_two = format!("1.{}2", "0".repeat(28));

        let decimal_one = parse_accepted_text(&last_one);
        let decimal_two = parse_accepted_text(&last_two);

        assert_ne!(decimal_one, decimal_two);
        assert_eq!(
            decimal_one.coefficient.to_string(),
            format!("1{}1", "0".repeat(28))
        );
        assert_eq!(decimal_one.exponent, -29);
        assert_eq!(decimal_one.to_string(), last_one);
    }

    #[test]
    fn decimal_from_str_keeps_two_hundred_digits() {
        let integer_part = "1234567890".repeat(10);
        let fraction_part = "0987654321".repeat(10);
        let text = format!("{integer_part}.{fraction_part}");

        let decimal = parse_accepted_text(&text);

        assert_eq!(
            decimal.coefficient.to_string(),
            format!("{integer_part}{fraction_part}")
        );
        assert_eq!(decimal.exponent, -100);
        assert_eq!(decimal.to_string(), text);
    }

    #[rstest]
    #[case::tiny(&format!(".{}3", "0".repeat(120)), -121, &format!("0.{}3", "0".repeat(120)))]
    #[case::huge(&format!("1{}.0", "0".repeat(80)), 80, &format!("1{}", "0".repeat(80)))]
    fn decimal_from_str_handles_exponents_beyond_the_float_range(
        #[case] text: &str,
        #[case] expected_exponent: i64,
        #[case] expected_text: &str,
    ) {
        let decimal = parse_accepted_text(text);

        assert_eq!(decimal.exponent, expected_exponent);
        assert_eq!(decimal.to_string(), expected_text);
    }

    #[rstest]
    #[case::empty("")]
    #[case::bare_point(".")]
    #[case::negative("-5")]
    #[case::explicit_plus("+5")]
    #[case::exponent("1e10")]
    #[case::fraction_exponent("1.5e3")]
    #[case::scientific_form("1E+2")]
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
    fn decimal_from_str_rejects_text_outside_the_grammar(#[case] text: &str) {
        let decimal = text.parse::<Decimal>();

        assert_eq!(decimal, Err(LiteralTextError { text: text.into() }));
    }

    #[rstest]
    #[case::zero("0", 0, "0")]
    #[case::integer("12", 0, "12")]
    #[case::fraction("12345", -2, "123.45")]
    #[case::leading_zeros("123", -8, "0.00000123")]
    #[case::point_before_the_first_digit("123", -3, "0.123")]
    #[case::positive_exponent_single_digit("1", 2, "100")]
    #[case::positive_exponent_many_digits("12", 3, "12000")]
    fn decimal_display_places_the_point_by_exponent(
        #[case] digits: &str,
        #[case] exponent: i64,
        #[case] expected: &str,
    ) {
        let decimal = build_decimal(digits, exponent);

        let text = decimal.to_string();

        assert_eq!(text, expected);
    }

    fn generate_digit_strings(
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
            generate_digit_strings(1..=30).prop_map(|integer_part| (integer_part, None)),
            (
                generate_digit_strings(1..=30),
                generate_digit_strings(0..=30)
            )
                .prop_map(|(integer_part, fraction_part)| (integer_part, Some(fraction_part))),
            generate_digit_strings(1..=30)
                .prop_map(|fraction_part| (String::new(), Some(fraction_part))),
        ]
    }

    fn join_decimal_parts(integer_part: &str, fraction_part: Option<&str>) -> String {
        match fraction_part {
            Some(fraction_part) => format!("{integer_part}.{fraction_part}"),
            None => integer_part.to_owned(),
        }
    }

    proptest! {
        #[test]
        fn decimal_display_reads_back_as_the_same_decimal(
            decimal in prop_oneof![
                generate_decimal_parts().prop_map(|(integer_part, fraction_part)| {
                    parse_accepted_text(&join_decimal_parts(&integer_part, fraction_part.as_deref()))
                }),
                ("[1-9]([0-9]{0,30}[1-9])?", -400_i64..400).prop_map(|(digits, exponent)| {
                    build_decimal(&digits, exponent)
                }),
            ],
        ) {
            let text = decimal.to_string();

            prop_assert_eq!(parse_accepted_text(&text), decimal, "text {:?}", text);
        }

        #[test]
        fn decimal_from_str_ignores_zero_padding(
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

            let plain_decimal = plain.parse::<Decimal>();
            let padded_decimal = padded.parse::<Decimal>();

            prop_assert!(plain_decimal.is_ok(), "{:?} is in the grammar", plain);
            prop_assert_eq!(padded_decimal, plain_decimal);
        }

        /// Test that moving the decimal point left by `shift` places keeps
        /// the coefficient and lowers the exponent by `shift`.
        #[test]
        fn decimal_from_str_tracks_the_point_position_in_the_exponent(
            digits in generate_digit_strings(1..=40),
            point in 0_usize..40,
            shift in 0_usize..40,
        ) {
            prop_assume!(digits.bytes().any(|digit| digit != b'0'));
            let right = point.min(digits.len());
            let left = right.saturating_sub(shift);
            let right_text = format!("{}.{}", &digits[..right], &digits[right..]);
            let left_text = format!("{}.{}", &digits[..left], &digits[left..]);

            let right_decimal = parse_accepted_text(&right_text);
            let left_decimal = parse_accepted_text(&left_text);

            prop_assert_eq!(&left_decimal.coefficient, &right_decimal.coefficient);
            let moved = i64::try_from(right - left).expect("shift fits in i64");
            prop_assert_eq!(left_decimal.exponent, right_decimal.exponent - moved);
        }

        /// Test that a normalized coefficient is `0` or has neither a leading
        /// nor a trailing zero, and is never negative.
        #[test]
        fn decimal_from_str_yields_a_canonical_coefficient(
            (integer_part, fraction_part) in generate_decimal_parts(),
        ) {
            let text = join_decimal_parts(&integer_part, fraction_part.as_deref());

            let decimal = parse_accepted_text(&text);

            let digits = decimal.coefficient.to_string();
            if digits == "0" {
                prop_assert_eq!(decimal.exponent, 0);
            } else {
                prop_assert!(!digits.starts_with('0') && !digits.ends_with('0'), "{:?}", digits);
                prop_assert!(decimal.coefficient > BigInt::ZERO, "{:?}", digits);
                prop_assert!(digits.bytes().all(|digit| digit.is_ascii_digit()), "{:?}", digits);
            }
        }
    }
}
