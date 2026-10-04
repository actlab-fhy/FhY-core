//! Deciding a piecewise.

use std::borrow::Cow;

use crate::expression::{Expression, ExpressionKind};
use crate::solver::SimplifyContext;

use super::SimplificationStrategy;
use super::exact::{boolean, is_decided};

/// Rewrites a piecewise whose every condition is a Boolean literal and
/// whose every value is decided, to the value of its first true case, or
/// to its otherwise value when none is.
///
/// It declines a piecewise with a condition it cannot decide, and one with
/// a branch it has not decided, one it does not take included: `SymPy`
/// lowers every branch, and fails on one that, say, takes a modulo by zero.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::Expression;
/// use fhy_core::solver::strategy::{PiecewiseDecision, SimplificationStrategy};
/// use fhy_core::solver::SimplifyContext;
///
/// let choice = Expression::piecewise(
///     [(Expression::literal(false), 10), (Expression::literal(true), 20)],
///     30,
/// )?;
///
/// assert_eq!(
///     PiecewiseDecision::new().rewrite(&choice, &SimplifyContext::default()),
///     Some(Expression::from(20)),
/// );
/// # Ok::<(), fhy_core::expression::PiecewiseError>(())
/// ```
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct PiecewiseDecision;

impl PiecewiseDecision {
    /// Return the strategy.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl SimplificationStrategy for PiecewiseDecision {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("piecewise")
    }

    fn rewrite(&self, node: &Expression, _context: &SimplifyContext<'_>) -> Option<Expression> {
        let ExpressionKind::Piecewise(piecewise) = node.kind() else {
            return None;
        };
        if !is_decided(piecewise.otherwise()) {
            return None;
        }
        let mut chosen = None;
        for (condition, value) in piecewise.cases() {
            let holds = boolean(condition)?;
            if !is_decided(value) {
                return None;
            }
            if holds && chosen.is_none() {
                chosen = Some(value);
            }
        }
        Some(chosen.unwrap_or_else(|| piecewise.otherwise()).clone())
    }
}
