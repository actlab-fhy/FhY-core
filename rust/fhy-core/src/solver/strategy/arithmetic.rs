//! Exact arithmetic on integers and rationals.

use std::borrow::Cow;

use crate::expression::{BinaryOperation, Expression, ExpressionKind, UnaryOperation};
use crate::solver::SimplifyContext;

use super::SimplificationStrategy;
use super::exact::{build_number_expression, compute_arithmetic, fold_small_integers, read_number};

/// Rewrites `+`, `-`, `*`, `/`, `//` (floor division), `%` (the floor
/// modulo) and `**` of exact numbers, and the unary `-` and `+`, to the
/// exact result, in the form `SymPy` lifts it to.
///
/// The numbers are integers, decimals and quotients of integers, computed
/// over [`BigInt`](crate::expression::BigInt)s. It declines a float, a
/// division or a modulo by zero, a zero raised to a negative power, a power
/// whose result is irrational or complex, an integer result of more than a
/// million bits, a fraction with a numerator or denominator of more than
/// 4096 bits, and a floor division or modulo of integers whose divisor and
/// quotient both have more than 4096 bits.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::Expression;
/// use fhy_core::solver::strategy::{ExactArithmetic, SimplificationStrategy};
/// use fhy_core::solver::SimplifyContext;
///
/// let context = SimplifyContext::default();
///
/// let floor = Expression::from(-7).floor_divide(2);
/// assert_eq!(ExactArithmetic::new().rewrite(&floor, &context), Some(Expression::from(-4)));
///
/// let half = Expression::from(1) / Expression::from(2);
/// assert_eq!(
///     ExactArithmetic::new().rewrite(&half, &context),
///     Some(Expression::literal(fhy_core::expression::LiteralValue::Decimal("0.5".parse()?))),
/// );
/// # Ok::<(), fhy_core::expression::LiteralTextError>(())
/// ```
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct ExactArithmetic;

impl ExactArithmetic {
    /// The strategy's name, which [`name`](SimplificationStrategy::name)
    /// returns.
    pub const NAME: &str = "exact_arithmetic";

    /// Return the strategy.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl SimplificationStrategy for ExactArithmetic {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(Self::NAME)
    }

    fn rewrite(&self, node: &Expression, _context: &SimplifyContext<'_>) -> Option<Expression> {
        let value = match node.kind() {
            ExpressionKind::Unary(unary) => match unary.operation() {
                UnaryOperation::Negate => -read_number(unary.operand())?,
                UnaryOperation::Positive => read_number(unary.operand())?,
                UnaryOperation::LogicalNot => return None,
            },
            ExpressionKind::Binary(binary) => {
                if !is_arithmetic(binary.operation()) {
                    return None;
                }
                if let Some(folded) =
                    fold_small_integers(binary.operation(), binary.left(), binary.right())
                {
                    return Some(folded);
                }
                compute_arithmetic(
                    binary.operation(),
                    &read_number(binary.left())?,
                    &read_number(binary.right())?,
                )?
            }
            _ => return None,
        };
        Some(build_number_expression(value)).filter(|rewritten| rewritten != node)
    }
}

/// Return whether `operation` is one of the arithmetic operations.
const fn is_arithmetic(operation: BinaryOperation) -> bool {
    matches!(
        operation,
        BinaryOperation::Add
            | BinaryOperation::Subtract
            | BinaryOperation::Multiply
            | BinaryOperation::Divide
            | BinaryOperation::FloorDivide
            | BinaryOperation::FloorMod
            | BinaryOperation::Power
    )
}
