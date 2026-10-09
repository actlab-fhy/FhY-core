//! The exact numbers the default strategies read and write, in the form
//! `SymPy`'s lifting gives them, and the exact arithmetic on them.

use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive};

use crate::expression::builtins::BuiltinFunction;
use crate::expression::{
    BinaryOperation, Callee, Decimal, Expression, ExpressionKind, LiteralValue, Rational,
    UnaryOperation, borrow_parts, build_integer, floor, keep_within_limits, reduce,
};

/// The most bits of an argument whose logarithm is taken.
const MAX_LOGARITHM_BITS: u64 = 4096;

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

/// Return whether `number` is `1`.
pub(super) fn is_one(number: &Rational) -> bool {
    number.numerator().is_one() && number.denominator().is_one()
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
