//! Deciding logical operators.

use std::borrow::Cow;

use crate::expression::{Expression, ExpressionKind, LogicalOperation, UnaryOperation};
use crate::solver::SimplifyContext;

use super::SimplificationStrategy;
use super::exact::read_boolean;

/// Rewrites `!`, `&&` and `||` of Boolean literals to the Boolean literal
/// they decide.
///
/// The Boolean built-ins (`xor`, `nand`, `nor`, `implies`, `iff`), which the
/// `SymPy` backend refuses until they are inlined, are folded by
/// [`ComposedBuiltins`](super::ComposedBuiltins), an opt-in strategy.
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
    /// The strategy's name, which [`name`](SimplificationStrategy::name)
    /// returns.
    pub const NAME: &str = "logical_operators";

    /// Return the strategy.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl SimplificationStrategy for LogicalOperators {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(Self::NAME)
    }

    fn rewrite(&self, node: &Expression, _context: &SimplifyContext<'_>) -> Option<Expression> {
        let value = match node.kind() {
            ExpressionKind::Unary(unary) if unary.operation() == UnaryOperation::LogicalNot => {
                !read_boolean(unary.operand())?
            }
            ExpressionKind::Logical(logical) => {
                let (mut all, mut any) = (true, false);
                for operand in logical.operands() {
                    let value = read_boolean(operand)?;
                    all &= value;
                    any |= value;
                }
                match logical.operation() {
                    LogicalOperation::And => all,
                    LogicalOperation::Or => any,
                }
            }
            _ => return None,
        };
        Some(Expression::literal(value))
    }
}
