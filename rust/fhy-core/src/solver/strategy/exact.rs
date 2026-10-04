//! The exact numbers the default strategies read and write, in the form
//! `SymPy`'s lifting gives them, and the exact arithmetic on them.

use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};

use crate::expression::{
    BinaryOperation, Decimal, Expression, ExpressionKind, LiteralValue, Rational, UnaryOperation,
};

/// The most bits a power may have as a result, past which the strategies
/// decline rather than compute a number of that size.
const MAX_POWER_BITS: u64 = 1 << 20;

/// The most bits of an argument whose logarithm is taken.
const MAX_LOGARITHM_BITS: u64 = 4096;

/// The greatest root index taken: the denominator of a rational exponent.
const MAX_ROOT_INDEX: u32 = 4096;

/// Return the exact number `expression` denotes, if it is written as
/// one: an integer literal, a decimal literal, a decimal literal negated,
/// or the quotient of two integer literals with a denominator other than
/// zero.
///
/// These are the forms [`number_expression`] writes, so a number one
/// strategy wrote is one another reads.
pub(super) fn number(expression: &Expression) -> Option<Rational> {
    match expression.kind() {
        ExpressionKind::Literal(LiteralValue::Int(value)) => Some(Rational::integer(value.clone())),
        ExpressionKind::Literal(LiteralValue::Decimal(value)) => decimal(value),
        ExpressionKind::Unary(unary) if unary.operation() == UnaryOperation::Negate => {
            match unary.operand().kind() {
                ExpressionKind::Literal(LiteralValue::Decimal(value)) => {
                    decimal(value).map(negated)
                }
                _ => None,
            }
        }
        ExpressionKind::Binary(binary) if binary.operation() == BinaryOperation::Divide => {
            match (binary.left().kind(), binary.right().kind()) {
                (
                    ExpressionKind::Literal(LiteralValue::Int(numerator)),
                    ExpressionKind::Literal(LiteralValue::Int(denominator)),
                ) => Rational::new(numerator.clone(), denominator.clone()),
                _ => None,
            }
        }
        _ => None,
    }
}

fn decimal(value: &Decimal) -> Option<Rational> {
    let (numerator, denominator) = value.to_rational_parts();
    Rational::new(numerator, denominator)
}

/// Return the integer `expression` is, if it is an integer literal that
/// fits an `i64`: the case the strategies compute without a big integer.
pub(super) fn small_integer(expression: &Expression) -> Option<i64> {
    match expression.kind() {
        ExpressionKind::Literal(LiteralValue::Int(value)) => value.to_i64(),
        _ => None,
    }
}

/// Return `left op right` for a sum, a difference, a product, a floor
/// division or a floor modulo of two small integers, as an integer
/// literal, or `None` for another operation or a zero divisor.
pub(super) fn small_integer_arithmetic(
    operation: BinaryOperation,
    left: i64,
    right: i64,
) -> Option<Expression> {
    let (left, right) = (i128::from(left), i128::from(right));
    let value = match operation {
        BinaryOperation::Add => left + right,
        BinaryOperation::Subtract => left - right,
        BinaryOperation::Multiply => left * right,
        BinaryOperation::FloorDivide | BinaryOperation::FloorMod if right != 0 => {
            let quotient = left / right;
            let remainder = left % right;
            let quotient = if remainder != 0 && ((remainder < 0) != (right < 0)) {
                quotient - 1
            } else {
                quotient
            };
            if operation == BinaryOperation::FloorDivide {
                quotient
            } else {
                left - right * quotient
            }
        }
        _ => return None,
    };
    Some(Expression::literal(value))
}

/// Return the Boolean `expression` is, if it is a Boolean literal.
pub(super) fn boolean(expression: &Expression) -> Option<bool> {
    match expression.kind() {
        ExpressionKind::Literal(LiteralValue::Bool(value)) => Some(*value),
        _ => None,
    }
}

/// Return the expression `SymPy`'s lifting writes for `number`: an integer
/// literal for an integer, the decimal literal of the value when a binary
/// float equals it (negated by a unary minus when it is negative), and the
/// quotient of its numerator and denominator otherwise.
pub(super) fn number_expression(number: Rational) -> Expression {
    let (numerator, denominator) = number.into_parts();
    if denominator.is_one() {
        return Expression::literal(numerator);
    }
    if let Some(decimal) = Decimal::from_rational_parts(numerator.clone(), denominator.clone()) {
        if decimal.to_f64_exact().is_some() {
            let magnitude = Expression::literal(LiteralValue::Decimal(decimal));
            return if numerator.is_negative() {
                Expression::new_unary(UnaryOperation::Negate, magnitude)
            } else {
                magnitude
            };
        }
    }
    Expression::new_binary(
        BinaryOperation::Divide,
        Expression::literal(numerator),
        Expression::literal(denominator),
    )
}

/// Return whether `expression` is a decided value in the form `SymPy`
/// answers: a Boolean literal, or a number written as
/// [`number_expression`] writes it.
pub(in crate::solver) fn is_decided(expression: &Expression) -> bool {
    boolean(expression).is_some()
        || number(expression).is_some_and(|number| number_expression(number) == *expression)
}

pub(super) fn integer(value: BigInt) -> Rational {
    Rational::integer(value)
}

pub(super) fn zero() -> Rational {
    integer(BigInt::zero())
}

fn parts(number: &Rational) -> (&BigInt, &BigInt) {
    (number.numerator(), number.denominator())
}

pub(super) fn negated(number: Rational) -> Rational {
    let (numerator, denominator) = number.into_parts();
    Rational::new(-numerator, denominator).expect("a denominator is positive")
}

/// Return `left op right` for an arithmetic operation, or `None` where it
/// has no exact rational value: a division by zero, a root that is not
/// rational, a power past [`MAX_POWER_BITS`], or an operation that is not
/// arithmetic.
pub(super) fn arithmetic(
    operation: BinaryOperation,
    left: &Rational,
    right: &Rational,
) -> Option<Rational> {
    let ((a, b), (c, d)) = (parts(left), parts(right));
    match operation {
        BinaryOperation::Add => Rational::new(a * d + c * b, b * d),
        BinaryOperation::Subtract => Rational::new(a * d - c * b, b * d),
        BinaryOperation::Multiply => Rational::new(a * c, b * d),
        BinaryOperation::Divide => Rational::new(a * d, b * c),
        BinaryOperation::FloorDivide => {
            if c.is_zero() {
                return None;
            }
            Some(integer(floor(&quotient(left, right)?)))
        }
        BinaryOperation::FloorMod => {
            if c.is_zero() {
                return None;
            }
            let multiple = arithmetic(
                BinaryOperation::Multiply,
                right,
                &integer(floor(&quotient(left, right)?)),
            )?;
            arithmetic(BinaryOperation::Subtract, left, &multiple)
        }
        BinaryOperation::Power => power(left, right),
        _ => None,
    }
}

fn quotient(left: &Rational, right: &Rational) -> Option<Rational> {
    arithmetic(BinaryOperation::Divide, left, right)
}

/// Return the greatest integer not above `number`.
pub(super) fn floor(number: &Rational) -> BigInt {
    let (numerator, denominator) = parts(number);
    let quotient = numerator / denominator;
    if (numerator % denominator).is_negative() {
        quotient - BigInt::one()
    } else {
        quotient
    }
}

/// Return the least integer not below `number`.
pub(super) fn ceiling(number: &Rational) -> BigInt {
    -floor(&negated(number.clone()))
}

/// Return the absolute value of `number`.
pub(super) fn absolute(number: &Rational) -> Rational {
    if number.numerator().is_negative() {
        negated(number.clone())
    } else {
        number.clone()
    }
}

/// Return the sign of `number`: `-1`, `0` or `1`.
pub(super) fn sign(number: &Rational) -> Rational {
    integer(BigInt::from(match number.numerator().sign() {
        num_bigint::Sign::Plus => 1,
        num_bigint::Sign::Minus => -1,
        num_bigint::Sign::NoSign => 0,
    }))
}

/// Return the greater of two numbers, the second when they are equal.
pub(super) fn larger<'n>(left: &'n Rational, right: &'n Rational) -> &'n Rational {
    if left > right { left } else { right }
}

/// Return the lesser of two numbers, the second when they are equal.
pub(super) fn smaller<'n>(left: &'n Rational, right: &'n Rational) -> &'n Rational {
    if left < right { left } else { right }
}

/// Return `base ** exponent` where it is an exact rational, and `None`
/// where it is not, is undefined, or is too large.
pub(super) fn power(base: &Rational, exponent: &Rational) -> Option<Rational> {
    let ((base_numerator, base_denominator), (exponent_numerator, exponent_denominator)) =
        (parts(base), parts(exponent));
    if exponent_denominator.is_one() {
        return integer_power(base_numerator, base_denominator, exponent_numerator);
    }
    // A fractional power is exact only where the root is: of a positive
    // base, whose numerator and denominator both have one.
    if base_numerator.is_zero() {
        return exponent_numerator.is_positive().then(zero);
    }
    if base_numerator.is_negative() {
        return None;
    }
    let index = exponent_denominator
        .to_u32()
        .filter(|&index| index <= MAX_ROOT_INDEX)?;
    let numerator = exact_root(base_numerator, index)?;
    let denominator = exact_root(base_denominator, index)?;
    let root = Rational::new(numerator, denominator)?;
    power(&root, &integer(exponent_numerator.clone()))
}

/// Return `(numerator / denominator) ** exponent` for an integer exponent.
fn integer_power(numerator: &BigInt, denominator: &BigInt, exponent: &BigInt) -> Option<Rational> {
    if exponent.is_zero() {
        return Some(integer(BigInt::one()));
    }
    if numerator.is_zero() {
        return exponent.is_positive().then(zero);
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
        if size.checked_mul(u64::from(count))? > MAX_POWER_BITS {
            return None;
        }
        (numerator.pow(count), denominator.pow(count))
    };
    if exponent.is_negative() {
        Rational::new(powered_denominator, powered_numerator)
    } else {
        Rational::new(powered_numerator, powered_denominator)
    }
}

/// Return the `index`-th root of the positive `value` where it is an
/// integer.
fn exact_root(value: &BigInt, index: u32) -> Option<BigInt> {
    let root = value.nth_root(index);
    (root.pow(index) == *value).then_some(root)
}

/// Return the integer `k` with `number = base ** k`, where there is one:
/// a positive number that is a power of `base` or the reciprocal of one.
pub(super) fn integer_logarithm(number: &Rational, base: u32) -> Option<Rational> {
    let (numerator, denominator) = parts(number);
    if !numerator.is_positive() {
        return None;
    }
    if numerator.bits().max(denominator.bits()) > MAX_LOGARITHM_BITS {
        return None;
    }
    let base = BigInt::from(base);
    let (large, sign) = match (numerator.is_one(), denominator.is_one()) {
        (_, true) => (numerator, 1_i64),
        (true, false) => (denominator, -1),
        (false, false) => return None,
    };
    let mut exponent = 0_i64;
    let mut power = BigInt::one();
    while power < *large {
        power *= &base;
        exponent += 1;
    }
    (power == *large).then(|| integer(BigInt::from(sign * exponent)))
}
