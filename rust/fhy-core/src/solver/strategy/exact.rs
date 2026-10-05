//! The exact numbers the default strategies read and write, in the form
//! `SymPy`'s lifting gives them, and the exact arithmetic on them.

use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};

use crate::expression::builtins::BuiltinFunction;
use crate::expression::{
    BinaryOperation, Callee, Decimal, Expression, ExpressionKind, LiteralValue, Rational,
    UnaryOperation,
};

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

/// The most bits of an argument whose logarithm is taken.
const MAX_LOGARITHM_BITS: u64 = 4096;

/// The greatest root index taken: the denominator of a rational exponent.
const MAX_ROOT_INDEX: u32 = 4096;

/// Return the exact number `expression` denotes, if it is written as
/// one: an integer literal, a decimal literal, a decimal literal negated,
/// or the quotient of two integer literals with a denominator other than
/// zero.
///
/// These are the forms [`build_number_expression`] writes, so a number one
/// strategy wrote is one another reads.
pub(super) fn read_number(expression: &Expression) -> Option<Rational> {
    match expression.kind() {
        ExpressionKind::Literal(LiteralValue::Int(value)) => Some(Rational::integer(value.clone())),
        ExpressionKind::Literal(LiteralValue::Decimal(value)) => read_decimal(value),
        ExpressionKind::Unary(unary) if unary.operation() == UnaryOperation::Negate => {
            match unary.operand().kind() {
                ExpressionKind::Literal(LiteralValue::Decimal(value)) => {
                    read_decimal(value).map(|magnitude| -magnitude)
                }
                _ => None,
            }
        }
        ExpressionKind::Binary(binary) if binary.operation() == BinaryOperation::Divide => {
            match (binary.left().kind(), binary.right().kind()) {
                (
                    ExpressionKind::Literal(LiteralValue::Int(numerator)),
                    ExpressionKind::Literal(LiteralValue::Int(denominator)),
                ) => reduce(numerator.clone(), denominator.clone()),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Return the exact numbers of `arguments`, or `None` when one of them is
/// not written as [`read_number`] reads.
pub(super) fn read_numbers(arguments: &[Expression]) -> Option<Vec<Rational>> {
    arguments.iter().map(read_number).collect()
}

/// Return the built-in function `node` calls and its arguments, or `None`
/// when `node` is not a call of a built-in function.
pub(super) fn read_builtin_call(node: &Expression) -> Option<(BuiltinFunction, &[Expression])> {
    let ExpressionKind::Call(call) = node.kind() else {
        return None;
    };
    let Callee::Builtin(function) = call.callee() else {
        return None;
    };
    Some((*function, call.arguments()))
}

/// Return the exact value of a decimal, or `None` when it is past the size
/// limits.
fn read_decimal(value: &Decimal) -> Option<Rational> {
    keep_within_limits(value.to_rational())
}

/// Return the rational `numerator / denominator`, reduced, or `None` for a
/// zero denominator or a result the strategies decline to hold (see
/// [`keep_within_limits`]). It declines before the reduction when it would
/// take too long, so the cost of one call is bounded by the sizes.
fn reduce(numerator: BigInt, denominator: BigInt) -> Option<Rational> {
    if numerator.bits().min(denominator.bits()) > MAX_FRACTION_BITS {
        return None;
    }
    keep_within_limits(Rational::new(numerator, denominator)?)
}

/// Return `rational`, or `None` when the strategies decline to hold it: an
/// integer of more than [`MAX_POWER_BITS`] bits, or a fraction with a part
/// of more than [`MAX_FRACTION_BITS`].
fn keep_within_limits(rational: Rational) -> Option<Rational> {
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

/// Return the integer `expression` is, if it is an integer literal that
/// fits an `i64`: the case the strategies compute without a big integer.
fn read_small_integer(expression: &Expression) -> Option<i64> {
    match expression.kind() {
        ExpressionKind::Literal(LiteralValue::Int(value)) => value.to_i64(),
        _ => None,
    }
}

/// Return `left op right` for a sum, a difference, a product, a floor
/// division or a floor modulo of two integer literals that fit an `i64`,
/// as an integer literal, or `None` for another operation, another
/// operand or a zero divisor.
pub(super) fn fold_small_integers(
    operation: BinaryOperation,
    left: &Expression,
    right: &Expression,
) -> Option<Expression> {
    let (left, right) = (
        i128::from(read_small_integer(left)?),
        i128::from(read_small_integer(right)?),
    );
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
pub(super) fn read_boolean(expression: &Expression) -> Option<bool> {
    match expression.kind() {
        ExpressionKind::Literal(LiteralValue::Bool(value)) => Some(*value),
        _ => None,
    }
}

/// Return the expression `SymPy`'s lifting writes for `number`: an integer
/// literal for an integer, the decimal literal of the value when a binary
/// float equals it (negated by a unary minus when it is negative), and the
/// quotient of its numerator and denominator otherwise.
pub(super) fn build_number_expression(number: Rational) -> Expression {
    if number.denominator().is_one() {
        return Expression::literal(number.into_parts().0);
    }
    if number.to_f64_exact().is_some() {
        if let Some(magnitude) = number.to_decimal() {
            let magnitude = Expression::literal(LiteralValue::Decimal(magnitude));
            return if number.numerator().is_negative() {
                Expression::new_unary(UnaryOperation::Negate, magnitude)
            } else {
                magnitude
            };
        }
    }
    let (numerator, denominator) = number.into_parts();
    Expression::new_binary(
        BinaryOperation::Divide,
        Expression::literal(numerator),
        Expression::literal(denominator),
    )
}

/// Return whether `expression` is a decided value in the form `SymPy`
/// answers: a Boolean literal, or a number written as
/// [`build_number_expression`] writes it.
pub(in crate::solver) fn is_decided(expression: &Expression) -> bool {
    match expression.kind() {
        ExpressionKind::Literal(LiteralValue::Bool(_) | LiteralValue::Int(_)) => true,
        _ => read_number(expression)
            .is_some_and(|number| build_number_expression(number) == *expression),
    }
}

pub(super) fn build_integer(value: BigInt) -> Rational {
    Rational::integer(value)
}

pub(super) fn build_zero() -> Rational {
    build_integer(BigInt::zero())
}

/// Return whether `number` is `1`.
pub(super) fn is_one(number: &Rational) -> bool {
    number.numerator().is_one() && number.denominator().is_one()
}

fn borrow_parts(number: &Rational) -> (&BigInt, &BigInt) {
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
/// The operands are within the same limits, as [`read_number`] holds them
/// to, which is what lets an operation that is surely past the limits
/// decline before it multiplies.
pub(super) fn compute_arithmetic(
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
pub(super) fn floor(number: &Rational) -> BigInt {
    let (numerator, denominator) = borrow_parts(number);
    let quotient = numerator / denominator;
    if (numerator % denominator).is_negative() {
        quotient - BigInt::one()
    } else {
        quotient
    }
}

/// Return the least integer not below `number`.
pub(super) fn ceil(number: &Rational) -> BigInt {
    -floor(&-number.clone())
}

/// Return the absolute value of `number`.
pub(super) fn take_absolute(number: &Rational) -> Rational {
    if number.numerator().is_negative() {
        -number.clone()
    } else {
        number.clone()
    }
}

/// Return the sign of `number`: `-1`, `0` or `1`.
pub(super) fn take_sign(number: &Rational) -> Rational {
    build_integer(BigInt::from(match number.numerator().sign() {
        num_bigint::Sign::Plus => 1,
        num_bigint::Sign::Minus => -1,
        num_bigint::Sign::NoSign => 0,
    }))
}

/// Return the greater of two numbers, the second when they are equal.
pub(super) fn pick_larger<'n>(left: &'n Rational, right: &'n Rational) -> &'n Rational {
    if left > right { left } else { right }
}

/// Return the lesser of two numbers, the second when they are equal.
pub(super) fn pick_smaller<'n>(left: &'n Rational, right: &'n Rational) -> &'n Rational {
    if left < right { left } else { right }
}

/// Return `base ** exponent` where it is an exact rational, and `None`
/// where it is not, is undefined, or is too large.
pub(super) fn raise_to_power(base: &Rational, exponent: &Rational) -> Option<Rational> {
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
pub(super) fn take_root(base: &Rational, index: u32) -> Option<Rational> {
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

/// Return the integer `k` with `number = base ** k`, where there is one:
/// a positive number that is a power of `base` or the reciprocal of one.
pub(super) fn take_integer_logarithm(number: &Rational, base: u32) -> Option<Rational> {
    let (numerator, denominator) = borrow_parts(number);
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
    (power == *large).then(|| build_integer(BigInt::from(sign * exponent)))
}
