//! Deciding logical operators.

use std::borrow::Cow;

use crate::expression::{Expression, ExpressionKind, LogicalOperation, UnaryOperation};
use crate::solver::SimplifyContext;

use super::SimplificationStrategy;
use super::exact::boolean;

/// Rewrites `!`, `&&` and `||` of Boolean literals to the Boolean literal
/// they decide.
///
/// The Boolean built-ins (`xor`, `nand`, `nor`, `implies`, `iff`), which the
/// `SymPy` backend refuses until they are inlined, are
/// [`ComposedBuiltins`](super::ComposedBuiltins)', a strategy that is not a
/// default.
///
/// Every operand must be a Boolean literal: `false && x` is declined, as
/// the `SymPy` backend refuses an operand that is not a Boolean, and a
/// partial answer would differ from its form.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::Expression;
/// use fhy_core::solver::strategy::{LogicalOperators, SimplificationStrategy};
/// use fhy_core::solver::SimplifyContext;
///
/// let both = Expression::literal(true).and(Expression::literal(false));
///
/// assert_eq!(
///     LogicalOperators::new().rewrite(&both, &SimplifyContext::default()),
///     Some(Expression::literal(false)),
/// );
/// ```
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct LogicalOperators;

impl LogicalOperators {
    /// Return the strategy.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl SimplificationStrategy for LogicalOperators {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("logical_operators")
    }

    fn rewrite(&self, node: &Expression, _context: &SimplifyContext<'_>) -> Option<Expression> {
        let value = match node.kind() {
            ExpressionKind::Unary(unary) if unary.operation() == UnaryOperation::LogicalNot => {
                !boolean(unary.operand())?
            }
            ExpressionKind::Logical(logical) => {
                let mut operands = logical.operands().iter().map(boolean);
                match logical.operation() {
                    LogicalOperation::And => {
                        let values: Option<Vec<bool>> = operands.by_ref().collect();
                        values?.into_iter().all(|value| value)
                    }
                    LogicalOperation::Or => {
                        let values: Option<Vec<bool>> = operands.by_ref().collect();
                        values?.into_iter().any(|value| value)
                    }
                }
            }
            _ => return None,
        };
        Some(Expression::literal(value))
    }
}
