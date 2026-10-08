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
use std::fmt;
use std::ops::Neg;

use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};

use crate::expression::{BinaryOperation, Decimal, LiteralValue};

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
///
/// It orders numerically and displays as its numerator alone when it is an
/// integer and as `numerator/denominator` otherwise: `3`, `-1/2`.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::{BigInt, Rational};
///
/// let half = Rational::new(BigInt::from(2), BigInt::from(-4)).expect("a non-zero denominator");
/// assert_eq!(half.to_string(), "-1/2");
/// assert!(!half.is_integer());
/// assert_eq!(Rational::from(BigInt::from(3)).to_integer(), Some(&BigInt::from(3)));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Rational {
    numerator: BigInt,
    denominator: BigInt,
}

impl Rational {
    /// Return the rational `numerator / denominator`, reduced, or `None`
    /// for a zero denominator.
    #[must_use]
    pub fn new(numerator: BigInt, denominator: BigInt) -> Option<Self> {
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
    #[must_use]
    pub fn numerator(&self) -> &BigInt {
        &self.numerator
    }

    /// Return the denominator, positive.
    #[must_use]
    pub fn denominator(&self) -> &BigInt {
        &self.denominator
    }

    /// Return whether the rational is an integer: its denominator is 1.
    #[must_use]
    pub fn is_integer(&self) -> bool {
        self.denominator.is_one()
    }

    /// Return the rational as an integer, or `None` when it is not one.
    #[must_use]
    pub fn to_integer(&self) -> Option<&BigInt> {
        self.is_integer().then_some(&self.numerator)
    }

    /// Return whether the rational is zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.numerator.is_zero()
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

impl From<BigInt> for Rational {
    /// Return the integer `value` as a rational.
    fn from(value: BigInt) -> Self {
        Self::integer(value)
    }
}

impl fmt::Display for Rational {
    /// Write the numerator alone for an integer, and
    /// `numerator/denominator` otherwise.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_integer() {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
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

/// The most bits an integer may have as the result of an operation, past
/// which the strategies decline rather than compute a number of that size.
const MAX_POWER_BITS: u64 = 1 << 20;

/// The most bits a part of a number that is not an integer may have, past
/// which the strategies decline rather than reduce or expand it. Reducing a
/// fraction by its greatest common divisor, and finding out whether it is a
/// decimal, take time quadratic in its size, so this keeps one operation
/// to milliseconds. An integer is bounded by [`MAX_POWER_BITS`] instead.
///
/// A floor division or a floor modulo of integers takes time about the
/// product of the divisor's and the quotient's sizes, so it is declined
/// when both have more bits than this.
const MAX_FRACTION_BITS: u64 = 4096;

/// The greatest root index taken: the denominator of a rational exponent.
const MAX_ROOT_INDEX: u32 = 4096;

/// Return the rational `numerator / denominator`, reduced, or `None` for a
/// zero denominator or a result the strategies decline to hold (see
/// [`keep_within_limits`]). It declines before the reduction when it would
/// take too long, so the cost of one call is bounded by the sizes.
pub(crate) fn reduce(numerator: BigInt, denominator: BigInt) -> Option<Rational> {
    if numerator.bits().min(denominator.bits()) > MAX_FRACTION_BITS {
        return None;
    }
    keep_within_limits(Rational::new(numerator, denominator)?)
}

/// Return `rational`, or `None` when the strategies decline to hold it: an
/// integer of more than [`MAX_POWER_BITS`] bits, or a fraction with a part
/// of more than [`MAX_FRACTION_BITS`].
pub(crate) fn keep_within_limits(rational: Rational) -> Option<Rational> {
    let limit = if rational.denominator().is_one() {
        MAX_POWER_BITS
    } else {
        MAX_FRACTION_BITS
    };
    let bits = rational
        .numerator()
        .bits()
        .max(rational.denominator().bits());
    (bits <= limit).then_some(rational)
}

pub(crate) fn build_integer(value: BigInt) -> Rational {
    Rational::integer(value)
}

pub(crate) fn build_zero() -> Rational {
    build_integer(BigInt::zero())
}

pub(crate) fn borrow_parts(number: &Rational) -> (&BigInt, &BigInt) {
    (number.numerator(), number.denominator())
}

/// Return the least and the greatest number of bits of the product of two
/// non-zero integers.
fn bound_product_bits(left: &BigInt, right: &BigInt) -> (u64, u64) {
    let bits = left.bits() + right.bits();
    (bits - 1, bits)
}

/// Return whether the fraction `numerator / denominator`, each a product of
/// two integers, is surely past the size limits once reduced, so the
/// products need not be formed.
///
/// The reduction divides the numerator's product by a divisor of the
/// denominator's, which has at most as many bits as that product: the
/// numerator keeps at least the difference of the bits. A zero factor makes
/// the product, and so the answer, small.
fn is_beyond_limits(numerator: (&BigInt, &BigInt), denominator: (&BigInt, &BigInt)) -> bool {
    let factors = [numerator.0, numerator.1, denominator.0, denominator.1];
    if factors.iter().any(|factor| factor.is_zero()) {
        return false;
    }
    let (numerator_least, numerator_greatest) = bound_product_bits(numerator.0, numerator.1);
    let (denominator_least, denominator_greatest) =
        bound_product_bits(denominator.0, denominator.1);
    numerator_least.saturating_sub(denominator_greatest) > MAX_POWER_BITS
        || denominator_least.saturating_sub(numerator_greatest) > MAX_FRACTION_BITS
}

/// Return `left op right` for an arithmetic operation, or `None` where it
/// has no exact rational value: a division by zero, a root that is not
/// rational, a result past the size limits (an integer of more than
/// [`MAX_POWER_BITS`] bits, a fraction with a part of more than
/// [`MAX_FRACTION_BITS`], or an integer floor division or modulo whose
/// divisor and quotient both have more bits than that), or an operation
/// that is not arithmetic.
///
/// The operands are within the same limits, as `read_number` holds them
/// to, which is what lets an operation that is surely past the limits
/// decline before it multiplies.
pub(crate) fn compute_arithmetic(
    operation: BinaryOperation,
    left: &Rational,
    right: &Rational,
) -> Option<Rational> {
    let ((a, b), (c, d)) = (borrow_parts(left), borrow_parts(right));
    let are_integers = b.is_one() && d.is_one();
    match operation {
        BinaryOperation::Add | BinaryOperation::Subtract => {
            let is_sum = operation == BinaryOperation::Add;
            if are_integers {
                return keep_within_limits(build_integer(if is_sum { a + c } else { a - c }));
            }
            // A fraction is within the limits, and an integer added to one
            // has a part past them when it has more bits than a fraction's
            // part may.
            if a.bits().max(c.bits()) > MAX_FRACTION_BITS {
                return None;
            }
            let (scaled_left, scaled_right) = (a * d, c * b);
            reduce(
                if is_sum {
                    scaled_left + scaled_right
                } else {
                    scaled_left - scaled_right
                },
                b * d,
            )
        }
        BinaryOperation::Multiply => {
            if are_integers {
                // A product has at least one bit fewer than its factors do
                // together.
                if !a.is_zero()
                    && !c.is_zero()
                    && (a.bits() + c.bits()).saturating_sub(1) > MAX_POWER_BITS
                {
                    return None;
                }
                return keep_within_limits(build_integer(a * c));
            }
            if is_beyond_limits((a, c), (b, d)) {
                return None;
            }
            reduce(a * c, b * d)
        }
        BinaryOperation::Divide => {
            if is_beyond_limits((a, d), (b, c)) {
                return None;
            }
            reduce(a * d, b * c)
        }
        BinaryOperation::FloorDivide | BinaryOperation::FloorMod => {
            if c.is_zero() {
                return None;
            }
            if are_integers {
                return floor_divide_integers(operation, a, c);
            }
            let multiple = build_integer(floor(&divide(left, right)?));
            if operation == BinaryOperation::FloorDivide {
                return Some(multiple);
            }
            let product = compute_arithmetic(BinaryOperation::Multiply, right, &multiple)?;
            compute_arithmetic(BinaryOperation::Subtract, left, &product)
        }
        BinaryOperation::Power => raise_to_power(left, right),
        _ => None,
    }
}

/// Return `left // right` or, for `operation` a floor modulo, `left %
/// right`, of integers with a divisor other than zero, with the sign
/// conventions of floor division: the modulo has the divisor's sign.
fn floor_divide_integers(
    operation: BinaryOperation,
    left: &BigInt,
    right: &BigInt,
) -> Option<Rational> {
    if right.bits().min(left.bits().saturating_sub(right.bits())) > MAX_FRACTION_BITS {
        return None;
    }
    let mut quotient = left / right;
    let mut remainder = left - &quotient * right;
    if !remainder.is_zero() && remainder.is_negative() != right.is_negative() {
        quotient -= 1;
        remainder += right;
    }
    keep_within_limits(build_integer(
        if operation == BinaryOperation::FloorDivide {
            quotient
        } else {
            remainder
        },
    ))
}

fn divide(left: &Rational, right: &Rational) -> Option<Rational> {
    compute_arithmetic(BinaryOperation::Divide, left, right)
}

/// Return the greatest integer not above `number`.
pub(crate) fn floor(number: &Rational) -> BigInt {
    let (numerator, denominator) = borrow_parts(number);
    let quotient = numerator / denominator;
    if (numerator % denominator).is_negative() {
        quotient - BigInt::one()
    } else {
        quotient
    }
}
/// Return `base ** exponent` where it is an exact rational, and `None`
/// where it is not, is undefined, or is too large.
pub(crate) fn raise_to_power(base: &Rational, exponent: &Rational) -> Option<Rational> {
    let ((base_numerator, base_denominator), (exponent_numerator, exponent_denominator)) =
        (borrow_parts(base), borrow_parts(exponent));
    if exponent_denominator.is_one() {
        return raise_to_integer_power(base_numerator, base_denominator, exponent_numerator);
    }
    if base_numerator.is_zero() {
        return exponent_numerator.is_positive().then(build_zero);
    }
    let index = exponent_denominator
        .to_u32()
        .filter(|&index| index <= MAX_ROOT_INDEX)?;
    let root = take_root(base, index)?;
    raise_to_power(&root, &build_integer(exponent_numerator.clone()))
}

/// Return the `index`-th root of `base` where it is rational, and `None`
/// for a negative base. A fractional power is exact only where the root
/// is: of a non-negative base, whose numerator and denominator both have
/// one.
pub(crate) fn take_root(base: &Rational, index: u32) -> Option<Rational> {
    if base.numerator().is_negative() {
        return None;
    }
    if base.numerator().is_zero() {
        return Some(build_zero());
    }
    let numerator = take_exact_root(base.numerator(), index)?;
    let denominator = take_exact_root(base.denominator(), index)?;
    reduce(numerator, denominator)
}

/// Return `(numerator / denominator) ** exponent` for an integer exponent.
fn raise_to_integer_power(
    numerator: &BigInt,
    denominator: &BigInt,
    exponent: &BigInt,
) -> Option<Rational> {
    if exponent.is_zero() {
        return Some(build_integer(BigInt::one()));
    }
    if numerator.is_zero() {
        return exponent.is_positive().then(build_zero);
    }
    let magnitude = exponent.abs();
    let is_unit = denominator.is_one() && numerator.abs().is_one();
    let (powered_numerator, powered_denominator) = if is_unit {
        let sign = if numerator.is_negative() && magnitude.bit(0) {
            -BigInt::one()
        } else {
            BigInt::one()
        };
        (sign, BigInt::one())
    } else {
        let size = numerator.bits().max(denominator.bits());
        let count = magnitude.to_u32()?;
        let result_bits = size.checked_mul(u64::from(count))?;
        let is_fraction = !denominator.is_one() || exponent.is_negative();
        let limit = if is_fraction {
            MAX_FRACTION_BITS
        } else {
            MAX_POWER_BITS
        };
        if result_bits > limit {
            return None;
        }
        (numerator.pow(count), denominator.pow(count))
    };
    if exponent.is_negative() {
        reduce(powered_denominator, powered_numerator)
    } else {
        reduce(powered_numerator, powered_denominator)
    }
}

/// Return the `index`-th root of the positive `value` where it is an
/// integer.
fn take_exact_root(value: &BigInt, index: u32) -> Option<BigInt> {
    let root = value.nth_root(index);
    (root.pow(index) == *value).then_some(root)
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
