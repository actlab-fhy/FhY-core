//! The text Python's `repr` writes for a float and a Boolean.
//!
//! Literal equivalence keys and printed literals embed this text until they
//! move to Rust's own formatting.

use num_bigint::BigUint;

use super::decimal::{
    convert_count_to_exponent, write_exponent, write_positional, write_scientific_significand,
};

/// Largest decimal-point position, relative to the first significant digit,
/// at which a float is still written positionally.
const LARGEST_POSITIONAL_FLOAT_POINT: i64 = 16;

/// Smallest decimal-point position, relative to the first significant digit,
/// at which a float is still written positionally.
const SMALLEST_POSITIONAL_FLOAT_POINT: i64 = -3;

/// Fewest digits a float's scientific exponent is written with.
const FLOAT_EXPONENT_MINIMUM_DIGITS: usize = 2;

/// Number of explicit fraction bits in an `f64`.
const FLOAT_FRACTION_BITS: u32 = 52;

/// Binary exponent of the least significant bit of a subnormal `f64`.
const SMALLEST_FLOAT_BIT_EXPONENT: i64 = -1074;

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

/// Format `value` as the shortest text that reads back as the same `f64`.
///
/// Matches the Python implementation: this is the text `repr(float)`
/// writes.
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
pub(super) fn format_float_repr(value: f64) -> String {
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

/// Format a Boolean as `True` or `False`.
///
/// Matches the Python implementation: this is the text `repr(bool)`
/// writes.
#[must_use]
pub(super) fn format_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use rstest::rstest;

    use super::*;

    /// Test float formatting writes the expected text for representative
    /// values of every magnitude and sign, on each side of every notation
    /// boundary.
    #[rstest]
    #[case::zero(0.0, "0.0")]
    #[case::negative_zero(-0.0, "-0.0")]
    #[case::one(1.0, "1.0")]
    #[case::negative_one_and_a_half(-1.5, "-1.5")]
    #[case::negative_two_and_a_half(-2.5, "-2.5")]
    #[case::one_hundred(100.0, "100.0")]
    #[case::one_tenth(0.1, "0.1")]
    #[case::two_thirds(2.0 / 3.0, "0.6666666666666666")]
    #[case::pi(std::f64::consts::PI, "3.141592653589793")]
    #[case::e(std::f64::consts::E, "2.718281828459045")]
    #[case::machine_epsilon(f64::EPSILON, "2.220446049250313e-16")]
    #[case::one_plus_epsilon(1.0 + f64::EPSILON, "1.0000000000000002")]
    #[case::small_positional_fraction(0.00123, "0.00123")]
    #[case::long_positional_fraction(123_456_789_012_345.67, "123456789012345.67")]
    #[case::two_to_the_53(9_007_199_254_740_992.0, "9007199254740992.0")]
    #[case::two_to_the_53_plus_two(9_007_199_254_740_994.0, "9007199254740994.0")]
    #[case::sixteen_digit_point_two_digits(9.5e15, "9500000000000000.0")]
    #[case::seventeen_digit_integral(12_345_678_901_234_567.0, "1.2345678901234568e+16")]
    #[case::two_to_the_63(9_223_372_036_854_775_808.0, "9.223372036854776e+18")]
    #[case::googol(1e100, "1e+100")]
    #[case::inverse_googol(1e-100, "1e-100")]
    #[case::negative_largest_finite(f64::MIN, "-1.7976931348623157e+308")]
    #[case::largest_subnormal(f64::from_bits(0x000f_ffff_ffff_ffff), "2.225073858507201e-308")]
    #[case::twice_smallest_subnormal(f64::from_bits(2), "1e-323")]
    #[case::three_smallest_subnormals(f64::from_bits(3), "1.5e-323")]
    #[case::negative_smallest_subnormal(f64::from_bits(0x8000_0000_0000_0001), "-5e-324")]
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
    fn format_float_repr_writes_the_expected_text(#[case] value: f64, #[case] expected: &str) {
        let text = format_float_repr(value);

        assert_eq!(text, expected);
    }

    /// Test a float lying exactly halfway between two equally short digit
    /// strings is written with the one ending in an even digit.
    #[rstest]
    #[case::fifteen_integer_digits(0x4302_fbd4_64d1_0462, "667929902981260.2")]
    #[case::sixteen_integer_digits(0x431e_ff49_0c10_ae59, "2181234625358742.2")]
    #[case::two_fraction_digits(0x42d8_4f05_70a5_0528, "106910691005460.62")]
    #[case::four_fraction_digits_ending_in_two(0x4270_3c54_c1ca_8280, "1115706629288.1562")]
    #[case::four_fraction_digits_ending_in_eight(0x428a_96fe_d6a2_92c0, "3654477861970.3438")]
    fn format_float_repr_breaks_an_exact_tie_toward_the_even_digit(
        #[case] bits: u64,
        #[case] expected: &str,
    ) {
        let text = format_float_repr(f64::from_bits(bits));

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

    /// Test Booleans format with a capitalized first letter.
    #[rstest]
    #[case::true_value(true, "True")]
    #[case::false_value(false, "False")]
    fn format_bool_capitalizes(#[case] value: bool, #[case] expected: &str) {
        let text = format_bool(value);

        assert_eq!(text, expected);
    }

    /// Return the value a finite, non-negative float's text writes as
    /// `(significand, exponent)`, the value `significand * 10^exponent`, with
    /// the significand's trailing zeros removed.
    fn read_float_text_digits(text: &str) -> (u64, i64) {
        let (mantissa, exponent) =
            text.split_once('e')
                .map_or((text, 0), |(mantissa, exponent)| {
                    (
                        mantissa,
                        exponent.parse::<i64>().expect("an integer exponent"),
                    )
                });
        let (integer_part, fraction_part) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        let mut digits = format!("{integer_part}{fraction_part}");
        let mut exponent = exponent - convert_count_to_exponent(fraction_part.len());
        while digits.len() > 1 && digits.ends_with('0') {
            digits.pop();
            exponent += 1;
        }
        (digits.parse().expect("at most twenty digits"), exponent)
    }

    /// The `(fraction digits, binary exponent)` pairs `(d, k)` at which a
    /// float `I + f`, with `2^k <= I < 2^(k + 1)` and `f` an odd multiple of
    /// `2^-(d + 1)`, lies exactly halfway between two strings of `d` fraction
    /// digits that both read back as it, while no string of `d - 1` fraction
    /// digits does.
    const EXACT_TIE_SHAPES: [(u32, u32); 11] = [
        (1, 49),
        (1, 50),
        (2, 46),
        (2, 47),
        (3, 43),
        (3, 44),
        (4, 39),
        (4, 40),
        (4, 41),
        (5, 36),
        (5, 37),
    ];

    /// Build a strategy over exact halfway ties, each paired with the text
    /// breaking the tie toward the even last digit: `I.l` when the lower
    /// candidate `l` of `d` fraction digits ends in an even digit, `I.u` with
    /// the upper candidate `u = l + 1` otherwise.
    fn generate_exact_ties() -> impl Strategy<Value = (f64, String)> {
        proptest::sample::select(EXACT_TIE_SHAPES.to_vec()).prop_flat_map(
            |(fraction_digits, binary_exponent)| {
                (
                    (1_u64 << binary_exponent)..(1_u64 << (binary_exponent + 1)),
                    0_u64..(1_u64 << fraction_digits),
                )
                    .prop_map(move |(integer, half_index)| {
                        let odd = 2 * half_index + 1;
                        let denominator = 1_u64 << (fraction_digits + 1);
                        #[expect(clippy::cast_precision_loss, reason = "all three are below 2^52")]
                        let value = integer as f64 + odd as f64 / denominator as f64;
                        let tie_digits = odd * 5_u64.pow(fraction_digits + 1);
                        let lower = tie_digits / 10;
                        let even = if lower % 2 == 0 { lower } else { lower + 1 };
                        let width = usize::try_from(fraction_digits).expect("a small width");
                        (value, format!("{integer}.{even:0width$}"))
                    })
            },
        )
    }

    proptest! {
        /// Test no digit string shorter than a float's text reads back as
        /// the same float: neither neighbor one digit shorter, the digits
        /// truncated or truncated and raised by one in the last place, does.
        #[test]
        fn format_float_repr_writes_the_shortest_round_trip_digits(bits in any::<u64>()) {
            let value = f64::from_bits(bits).abs();
            prop_assume!(value.is_finite() && value != 0.0);
            let text = format_float_repr(value);
            let (significand, exponent) = read_float_text_digits(&text);
            prop_assume!(significand >= 10);

            for shorter in [significand / 10, significand / 10 + 1] {
                let shorter_text = format!("{shorter}e{}", exponent + 1);
                let reparsed: f64 = shorter_text.parse().expect("the shorter text parses");
                prop_assert_ne!(
                    reparsed.to_bits(),
                    value.to_bits(),
                    "{} is shorter than {:?} and reads back as the same float",
                    shorter_text,
                    text
                );
            }
        }

        /// Test a float lying exactly halfway between two equally short
        /// digit strings that both read back as it is written with the one
        /// ending in an even digit.
        #[test]
        fn format_float_repr_breaks_every_exact_tie_toward_the_even_digit(
            (value, expected) in generate_exact_ties(),
        ) {
            let text = format_float_repr(value);

            prop_assert_eq!(text, expected);
        }

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
    }
}
