//! Exact arithmetic on the numbers literals denote: rationals, the exact
//! value of a binary64 float and of a decimal, and the exact order of two
//! numeric literals.
//!
//! This is the one place the crate turns a float or a decimal into the
//! rational it denotes, so the solver lowering, the param bounds, the
//! ordinal order and the binding's SymPy bridge agree on what a literal
//! means. Every conversion is exact and checked: nothing is rounded,
//! truncated or replaced by a fallback.

use std::cmp::Ordering;
use std::ops::Neg;

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

use super::{Decimal, LiteralValue};

/// The number of stored fraction bits of a binary64 float.
const FRACTION_BITS: u32 = 52;

/// The exponent of the least subnormal binary64 float, `2^-1074`.
const LEAST_EXPONENT: i64 = -1074;

/// The exponent of the greatest binary64 float's leading bit, `2^1023`.
const GREATEST_EXPONENT: i64 = 1023;

/// Return the greatest common divisor of `left` and `right`, non-negative.
pub(crate) fn gcd(left: &BigInt, right: &BigInt) -> BigInt {
    let (mut left, mut right) = (left.abs(), right.abs());
    while !right.is_zero() {
        let remainder = &left % &right;
        left = right;
        right = remainder;
    }
    left
}

/// Return `base^exponent`, by squaring, for an exponent of any size.
pub(crate) fn power(base: u32, exponent: u64) -> BigInt {
    let mut result = BigInt::one();
    let mut square = BigInt::from(base);
    let mut remaining = exponent;
    while remaining > 0 {
        if remaining & 1 == 1 {
            result *= &square;
        }
        remaining >>= 1;
        if remaining > 0 {
            square = &square * &square;
        }
    }
    result
}

/// Return how often `prime` divides the non-zero `value`, and `value`
/// without those factors.
fn split_off(value: &BigInt, prime: u32) -> (u64, BigInt) {
    let prime = BigInt::from(prime);
    let mut value = value.clone();
    let mut count = 0_u64;
    while !value.is_zero() && (&value % &prime).is_zero() {
        value /= &prime;
        count += 1;
    }
    (count, value)
}

/// An exact rational `numerator / denominator`, in lowest terms with a
/// positive denominator, so two rationals are equal exactly when their
/// parts are.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct Rational {
    numerator: BigInt,
    denominator: BigInt,
}

impl Rational {
    /// Return the rational `numerator / denominator`, reduced, or `None`
    /// for a zero denominator.
    pub(crate) fn new(numerator: BigInt, denominator: BigInt) -> Option<Self> {
        if denominator.is_zero() {
            return None;
        }
        let divisor = gcd(&numerator, &denominator);
        let (mut numerator, mut denominator) = (numerator / &divisor, denominator / divisor);
        if denominator.is_negative() {
            numerator = -numerator;
            denominator = -denominator;
        }
        Some(Self {
            numerator,
            denominator,
        })
    }

    /// Return the integer `value` as a rational.
    pub(crate) fn integer(value: BigInt) -> Self {
        Self {
            numerator: value,
            denominator: BigInt::one(),
        }
    }

    /// Return the numerator, of the rational's sign.
    pub(crate) fn numerator(&self) -> &BigInt {
        &self.numerator
    }

    /// Return the denominator, positive.
    pub(crate) fn denominator(&self) -> &BigInt {
        &self.denominator
    }

    /// Return the numerator and the denominator.
    pub(crate) fn into_parts(self) -> (BigInt, BigInt) {
        (self.numerator, self.denominator)
    }

    /// Return the exact value of `value`, from its IEEE-754 decomposition,
    /// or `None` for a NaN or an infinity. Both zeros are `0`.
    pub(crate) fn of_f64(value: f64) -> Option<Self> {
        if !value.is_finite() {
            return None;
        }
        let bits = value.to_bits();
        let is_negative = bits >> 63 == 1;
        let biased = i64::from(u16::try_from((bits >> FRACTION_BITS) & 0x7ff).ok()?);
        let fraction = bits & ((1_u64 << FRACTION_BITS) - 1);
        let (mantissa, exponent) = if biased == 0 {
            (fraction, LEAST_EXPONENT)
        } else {
            (fraction | (1_u64 << FRACTION_BITS), biased - 1075)
        };
        if mantissa == 0 {
            return Some(Self::integer(BigInt::zero()));
        }
        // Lowest terms: the denominator is a power of two, so only the
        // mantissa's factors of two cancel.
        let shift = mantissa.trailing_zeros();
        let magnitude = BigInt::from(mantissa >> shift);
        let exponent = exponent + i64::from(shift);
        let magnitude = if is_negative { -magnitude } else { magnitude };
        Some(if exponent >= 0 {
            Self::integer(magnitude << exponent.unsigned_abs())
        } else {
            Self {
                numerator: magnitude,
                denominator: BigInt::one() << exponent.unsigned_abs(),
            }
        })
    }

    /// Return the binary64 float equal to the rational, or `None` when no
    /// float is: the rational must be `m * 2^e` with an odd `m` below
    /// `2^53` and `m * 2^e` within the float range, subnormals included.
    pub(crate) fn to_f64_exact(&self) -> Option<f64> {
        if self.numerator.is_zero() {
            return Some(0.0);
        }
        let (twos_below, rest) = split_off(&self.denominator, 2);
        if !rest.is_one() {
            return None;
        }
        let (twos_above, odd) = split_off(&self.numerator, 2);
        let exponent = i64::try_from(twos_above)
            .ok()?
            .checked_sub(i64::try_from(twos_below).ok()?)?;
        let mantissa = u64::try_from(odd.abs()).ok()?;
        if mantissa >= 1_u64 << (FRACTION_BITS + 1) {
            return None;
        }
        let leading = exponent.checked_add(i64::from(mantissa.ilog2()))?;
        if exponent < LEAST_EXPONENT || leading > GREATEST_EXPONENT {
            return None;
        }
        // `mantissa * 2^exponent` is representable, and scaling a float by
        // a power of two whose product is representable is exact. The
        // exponent is split in two halves so neither power overflows or
        // underflows on its own.
        #[expect(
            clippy::cast_precision_loss,
            reason = "the mantissa is below 2^53, so the cast is exact"
        )]
        let mut value = mantissa as f64;
        let first = exponent / 2;
        for half in [first, exponent - first] {
            value *= 2_f64.powi(i32::try_from(half).ok()?);
        }
        Some(if self.numerator.is_negative() {
            -value
        } else {
            value
        })
    }

    /// Return the decimal equal to the rational's magnitude, or `None` when
    /// its expansion does not end (a denominator with a prime factor other
    /// than 2 and 5) or its exponent exceeds
    /// [`Decimal::MAX_EXPONENT_MAGNITUDE`].
    pub(crate) fn to_decimal(&self) -> Option<Decimal> {
        let (twos, rest) = split_off(&self.denominator, 2);
        let (fives, rest) = split_off(&rest, 5);
        if !rest.is_one() {
            return None;
        }
        let digits_after_point = twos.max(fives);
        let coefficient = self.numerator.abs()
            * power(2, digits_after_point - twos)
            * power(5, digits_after_point - fives);
        let exponent = i64::try_from(digits_after_point).ok()?;
        Decimal::from_parts(coefficient, -exponent).ok()
    }
}

impl Neg for Rational {
    type Output = Self;

    /// Return the rational with the numerator negated, which keeps the
    /// lowest terms and the positive denominator.
    fn neg(self) -> Self {
        Self {
            numerator: -self.numerator,
            denominator: self.denominator,
        }
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rational {
    /// Compare by cross-multiplication; the denominators are positive.
    fn cmp(&self, other: &Self) -> Ordering {
        (&self.numerator * &other.denominator).cmp(&(&other.numerator * &self.denominator))
    }
}

impl Decimal {
    /// Return the exact rational the decimal denotes, in lowest terms.
    pub(crate) fn to_rational(&self) -> Rational {
        let scale = power(10, self.exponent().unsigned_abs());
        if self.exponent() >= 0 {
            Rational::integer(self.coefficient() * scale)
        } else {
            Rational::new(self.coefficient().clone(), scale)
                .unwrap_or_else(|| unreachable!("a power of ten is not zero"))
        }
    }
}

/// A number on the extended number line: an exact rational or an
/// infinity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ExactNumber {
    /// Negative infinity, below every rational.
    NegativeInfinity,
    /// A finite number.
    Finite(Rational),
    /// Positive infinity, above every rational.
    PositiveInfinity,
}

impl ExactNumber {
    /// Return the exact value of `value`, or `None` for a NaN.
    pub(crate) fn of_f64(value: f64) -> Option<Self> {
        if value.is_nan() {
            None
        } else if value == f64::INFINITY {
            Some(Self::PositiveInfinity)
        } else if value == f64::NEG_INFINITY {
            Some(Self::NegativeInfinity)
        } else {
            Rational::of_f64(value).map(Self::Finite)
        }
    }

    /// Return the integer `value`.
    pub(crate) fn integer(value: BigInt) -> Self {
        Self::Finite(Rational::integer(value))
    }
}

impl LiteralValue {
    /// Return the exact number the literal denotes, a Boolean as `0` or
    /// `1`, or `None` for a NaN.
    pub(crate) fn exact_value(&self) -> Option<ExactNumber> {
        match self {
            Self::Bool(value) => Some(ExactNumber::integer(BigInt::from(u8::from(*value)))),
            Self::Int(value) => Some(ExactNumber::integer(value.clone())),
            Self::Float(value) => ExactNumber::of_f64(*value),
            Self::Decimal(value) => Some(ExactNumber::Finite(value.to_rational())),
        }
    }

    /// Return the exact order of the numbers two literals denote, a
    /// Boolean as `0` or `1`, never rounded: the decimal `0.1` lies below
    /// the float `0.1`. A NaN is ordered against nothing.
    pub(crate) fn exact_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.exact_value()?.cmp(&other.exact_value()?))
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use rstest::rstest;

    use super::*;

    fn decimal(text: &str) -> Decimal {
        text.parse().expect("a decimal")
    }

    /// Return a strategy over every finite `f64` bit pattern, normal and
    /// subnormal, of either sign.
    fn finite_f64() -> impl Strategy<Value = f64> {
        any::<u64>()
            .prop_map(f64::from_bits)
            .prop_filter("finite", |value| value.is_finite())
    }

    #[rstest]
    #[case::zero(0.0, "0", "1")]
    #[case::negative_zero(-0.0, "0", "1")]
    #[case::half(0.5, "1", "2")]
    #[case::integer(12.0, "12", "1")]
    #[case::negative(-0.75, "-3", "4")]
    #[case::least_subnormal(f64::from_bits(1), "1", &(BigInt::one() << 1074_u32).to_string())]
    #[case::greatest(f64::MAX, &((BigInt::from((1_u64 << 53) - 1)) << 971_u32).to_string(), "1")]
    fn of_f64_is_the_float_s_exact_value(
        #[case] value: f64,
        #[case] numerator: &str,
        #[case] denominator: &str,
    ) {
        let rational = Rational::of_f64(value).expect("finite");

        assert_eq!(rational.numerator().to_string(), numerator);
        assert_eq!(rational.denominator().to_string(), denominator);
    }

    #[rstest]
    #[case::positive(3, 4, -3, 4)]
    #[case::negative(-3, 4, 3, 4)]
    #[case::zero(0, 1, 0, 1)]
    #[case::integer(7, 1, -7, 1)]
    fn negating_a_rational_flips_the_numerator_and_keeps_the_denominator(
        #[case] numerator: i64,
        #[case] denominator: i64,
        #[case] negated_numerator: i64,
        #[case] negated_denominator: i64,
    ) {
        let rational =
            Rational::new(BigInt::from(numerator), BigInt::from(denominator)).expect("non-zero");
        let expected = Rational::new(
            BigInt::from(negated_numerator),
            BigInt::from(negated_denominator),
        )
        .expect("non-zero");

        assert_eq!(-rational, expected);
    }

    #[rstest]
    #[case::nan(f64::NAN)]
    #[case::infinity(f64::INFINITY)]
    #[case::negative_infinity(f64::NEG_INFINITY)]
    fn of_f64_refuses_a_non_finite_float(#[case] value: f64) {
        assert_eq!(Rational::of_f64(value), None);
    }

    #[rstest]
    #[case::third("1", "3")]
    #[case::too_many_bits(&((BigInt::one() << 53_u32) + BigInt::one()).to_string(), "1")]
    #[case::below_the_least_subnormal("1", &(BigInt::one() << 1075_u32).to_string())]
    #[case::above_the_greatest(&(BigInt::one() << 1024_u32).to_string(), "1")]
    fn to_f64_exact_refuses_what_no_float_equals(
        #[case] numerator: &str,
        #[case] denominator: &str,
    ) {
        let rational = Rational::new(
            numerator.parse().expect("an integer"),
            denominator.parse().expect("an integer"),
        )
        .expect("a non-zero denominator");

        assert_eq!(rational.to_f64_exact(), None);
    }

    #[rstest]
    #[case::integer("12", 0, "12", "1")]
    #[case::scaled("15", -1, "3", "2")]
    #[case::large_exponent("1", 3, "1000", "1")]
    #[case::tenth("1", -1, "1", "10")]
    fn a_decimal_s_rational_is_in_lowest_terms(
        #[case] coefficient: &str,
        #[case] exponent: i64,
        #[case] numerator: &str,
        #[case] denominator: &str,
    ) {
        let decimal =
            Decimal::from_parts(coefficient.parse().expect("an integer"), exponent).expect("parts");

        let rational = decimal.to_rational();

        assert_eq!(rational.numerator().to_string(), numerator);
        assert_eq!(rational.denominator().to_string(), denominator);
    }

    #[rstest]
    #[case::half("1", "2", Some("0.5"))]
    #[case::fifth("-1", "5", Some("0.2"))]
    #[case::integer("7", "1", Some("7"))]
    #[case::third("1", "3", None)]
    #[case::sixth("1", "6", None)]
    fn to_decimal_is_the_ending_expansion_of_the_magnitude(
        #[case] numerator: &str,
        #[case] denominator: &str,
        #[case] expected: Option<&str>,
    ) {
        let rational = Rational::new(
            numerator.parse().expect("an integer"),
            denominator.parse().expect("an integer"),
        )
        .expect("a non-zero denominator");

        assert_eq!(rational.to_decimal(), expected.map(decimal));
    }

    #[test]
    fn to_decimal_refuses_an_exponent_beyond_the_bound() {
        let limit = u64::from(Decimal::MAX_EXPONENT_MAGNITUDE);
        let at_bound = Rational::new(BigInt::one(), power(2, limit)).expect("non-zero");
        let beyond = Rational::new(BigInt::one(), power(2, limit + 1)).expect("non-zero");

        assert_eq!(
            at_bound.to_decimal().map(|decimal| decimal.exponent()),
            Some(-i64::from(Decimal::MAX_EXPONENT_MAGNITUDE))
        );
        assert_eq!(beyond.to_decimal(), None);
    }

    #[rstest]
    #[case::decimal_below_its_float(
        LiteralValue::Decimal(decimal("0.1")),
        LiteralValue::Float(0.1),
        Some(Ordering::Less)
    )]
    #[case::decimal_equal_to_its_float(
        LiteralValue::Decimal(decimal("0.5")),
        LiteralValue::Float(0.5),
        Some(Ordering::Equal)
    )]
    #[case::integer_above_a_float(
        LiteralValue::from(3),
        LiteralValue::Float(2.5),
        Some(Ordering::Greater)
    )]
    #[case::huge_integer_above_a_float(
        LiteralValue::Int(BigInt::one() << 2000_u32),
        LiteralValue::Float(f64::MAX),
        Some(Ordering::Greater)
    )]
    #[case::boolean_as_one(LiteralValue::Bool(true), LiteralValue::from(1), Some(Ordering::Equal))]
    #[case::infinity_above_everything(
        LiteralValue::Float(f64::INFINITY),
        LiteralValue::Int(BigInt::one() << 2000_u32),
        Some(Ordering::Greater)
    )]
    #[case::negative_infinity_below(LiteralValue::Float(f64::NEG_INFINITY), LiteralValue::from(-5), Some(Ordering::Less))]
    #[case::the_zeros(LiteralValue::Float(-0.0), LiteralValue::from(0), Some(Ordering::Equal))]
    #[case::nan(LiteralValue::Float(f64::NAN), LiteralValue::from(0), None)]
    fn exact_cmp_orders_the_values_denoted(
        #[case] left: LiteralValue,
        #[case] right: LiteralValue,
        #[case] expected: Option<Ordering>,
    ) {
        assert_eq!(left.exact_cmp(&right), expected);
        assert_eq!(right.exact_cmp(&left), expected.map(Ordering::reverse));
    }

    /// Return the literal's exact value as a fraction by a route of its
    /// own: a float through `integer_decode`, a decimal through its text,
    /// an integer as itself.
    fn oracle_fraction(literal: &LiteralValue) -> (BigInt, BigInt) {
        match literal {
            LiteralValue::Int(value) => (value.clone(), BigInt::one()),
            LiteralValue::Float(value) => {
                // `integer_decode` is the standard library's own route to
                // `mantissa * 2^exponent`.
                let (mantissa, exponent, sign) =
                    num_traits::float::FloatCore::integer_decode(*value);
                let numerator = BigInt::from(mantissa) * BigInt::from(sign);
                let exponent = i64::from(exponent);
                if exponent >= 0 {
                    (numerator << exponent.unsigned_abs(), BigInt::one())
                } else {
                    (numerator, BigInt::one() << exponent.unsigned_abs())
                }
            }
            LiteralValue::Decimal(value) => {
                let text = value.to_string();
                let (integer, fraction) = text.split_once('.').unwrap_or((&text, ""));
                let digits: BigInt = format!("{integer}{fraction}").parse().expect("digits");
                (
                    digits,
                    power(10, u64::try_from(fraction.len()).expect("a length")),
                )
            }
            LiteralValue::Bool(value) => (BigInt::from(u8::from(*value)), BigInt::one()),
        }
    }

    fn literal_strategy() -> impl Strategy<Value = LiteralValue> {
        prop_oneof![
            (-1000_i64..1000).prop_map(LiteralValue::from),
            finite_f64().prop_map(LiteralValue::Float),
            (-40_i64..40, 0_u32..8).prop_map(|(value, shift)| LiteralValue::Float(
                f64::from(i32::try_from(value).expect("small")) / f64::from(1_u32 << shift)
            )),
            ("[0-9]{1,6}", "[0-9]{0,6}").prop_map(|(integer, fraction)| LiteralValue::Decimal(
                decimal(&format!("{integer}.{fraction}"))
            )),
        ]
    }

    proptest! {
        #[test]
        fn of_f64_round_trips_every_finite_float_through_to_f64(value in finite_f64()) {
            let rational = Rational::of_f64(value).expect("finite");

            let back = rational.to_f64_exact().expect("a float's own value");

            prop_assert_eq!(back.to_bits(), (value + 0.0).to_bits());
        }

        #[test]
        fn exact_cmp_agrees_with_cross_multiplication(
            left in literal_strategy(),
            right in literal_strategy(),
        ) {
            let (left_numerator, left_denominator) = oracle_fraction(&left);
            let (right_numerator, right_denominator) = oracle_fraction(&right);
            let expected = (left_numerator * &right_denominator).cmp(&(right_numerator * &left_denominator));

            prop_assert_eq!(left.exact_cmp(&right), Some(expected));
        }

        #[test]
        fn a_decimal_round_trips_through_its_rational(
            digits in "[1-9][0-9]{0,8}",
            exponent in -30_i64..30,
        ) {
            let value = Decimal::from_parts(digits.parse().expect("digits"), exponent).expect("parts");

            prop_assert_eq!(value.to_rational().to_decimal(), Some(value));
        }
    }
}
