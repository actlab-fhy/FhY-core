//! Deciding comparisons.

use std::borrow::Cow;
use std::cmp::Ordering;

use crate::expression::{BinaryOperation, Expression, ExpressionKind, LiteralValue};
use crate::solver::SimplifyContext;

use super::SimplificationStrategy;
use super::exact::{boolean, number};

/// Rewrites a comparison of exact numbers to the Boolean literal it
/// decides, and an `==` or `!=` of two Booleans likewise.
///
/// It declines a comparison of a Boolean with a number, an ordering of
/// Booleans, and a comparison with a float.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::Expression;
/// use fhy_core::solver::strategy::{Comparisons, SimplificationStrategy};
/// use fhy_core::solver::SimplifyContext;
///
/// let holds = Expression::from(3).greater_equal(0);
///
/// assert_eq!(
///     Comparisons::new().rewrite(&holds, &SimplifyContext::default()),
///     Some(Expression::literal(true)),
/// );
/// ```
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct Comparisons;

impl Comparisons {
    /// Return the strategy.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl SimplificationStrategy for Comparisons {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("comparisons")
    }

    fn rewrite(&self, node: &Expression, _context: &SimplifyContext<'_>) -> Option<Expression> {
        let ExpressionKind::Binary(binary) = node.kind() else {
            return None;
        };
        let operation = binary.operation();
        if let (Some(left), Some(right)) = (boolean(binary.left()), boolean(binary.right())) {
            let is_equal = match operation {
                BinaryOperation::Equal => left == right,
                BinaryOperation::NotEqual => left != right,
                _ => return None,
            };
            return Some(Expression::literal(is_equal));
        }
        let order = match (binary.left().kind(), binary.right().kind()) {
            // Two integer literals compare without being copied.
            (
                ExpressionKind::Literal(LiteralValue::Int(left)),
                ExpressionKind::Literal(LiteralValue::Int(right)),
            ) => left.cmp(right),
            _ => number(binary.left())?.cmp(&number(binary.right())?),
        };
        let holds = match operation {
            BinaryOperation::Equal => order == Ordering::Equal,
            BinaryOperation::NotEqual => order != Ordering::Equal,
            BinaryOperation::Less => order == Ordering::Less,
            BinaryOperation::LessEqual => order != Ordering::Greater,
            BinaryOperation::Greater => order == Ordering::Greater,
            BinaryOperation::GreaterEqual => order != Ordering::Less,
            _ => return None,
        };
        Some(Expression::literal(holds))
    }
}
