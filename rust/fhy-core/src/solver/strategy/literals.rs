//! Writing a literal as `SymPy` does.

use std::borrow::Cow;

use crate::expression::{Expression, ExpressionKind, LiteralValue};
use crate::solver::SimplifyContext;

use super::SimplificationStrategy;
use super::exact::{number, number_expression};

/// Rewrites a decimal literal to the form `SymPy` lifts its value to: an
/// integer literal for an integer, and the quotient of two integers for a
/// value no binary float equals, so `0.1` becomes `1 / 10` and `2.0`
/// becomes `2`. A decimal a binary float equals is already in that form.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::{Expression, LiteralValue};
/// use fhy_core::solver::strategy::{NormalizeLiterals, SimplificationStrategy};
/// use fhy_core::solver::SimplifyContext;
///
/// let tenth = Expression::literal(LiteralValue::Decimal("0.1".parse()?));
///
/// let rewritten = NormalizeLiterals::new().rewrite(&tenth, &SimplifyContext::default());
///
/// assert_eq!(rewritten, Some(Expression::from(1) / Expression::from(10)));
/// # Ok::<(), fhy_core::expression::LiteralTextError>(())
/// ```
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct NormalizeLiterals;

impl NormalizeLiterals {
    /// Return the strategy.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl SimplificationStrategy for NormalizeLiterals {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("normalize_literals")
    }

    fn rewrite(&self, node: &Expression, _context: &SimplifyContext<'_>) -> Option<Expression> {
        let ExpressionKind::Literal(LiteralValue::Decimal(_)) = node.kind() else {
            return None;
        };
        Some(number_expression(number(node)?)).filter(|normalized| normalized != node)
    }
}
