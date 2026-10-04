//! The values of the registry's native constants.

use std::borrow::Cow;

use crate::expression::builtins::BuiltinConstant;
use crate::expression::{Expression, ExpressionKind};
use crate::solver::SimplifyContext;

use super::SimplificationStrategy;

/// Rewrites a reference to a native constant of the context's registry to
/// its value, as the `SymPy` backend lowers it. A built-in constant such as
/// `pi` is declined, and so is any reference when the context has no
/// registry.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::registry::{FunctionRegistry, NativeConstant};
/// use fhy_core::expression::{Expression, FunctionName, FunctionSort};
/// use fhy_core::solver::strategy::{RegisteredConstants, SimplificationStrategy};
/// use fhy_core::solver::SimplifyContext;
///
/// let mut registry = FunctionRegistry::new();
/// let answer = registry.register_constant(NativeConstant::new(
///     FunctionName::new("answer")?,
///     FunctionSort::Int,
///     42,
/// )?)?;
///
/// let rewritten = RegisteredConstants::new()
///     .rewrite(&Expression::from(answer), &SimplifyContext::from_registry(&registry));
///
/// assert_eq!(rewritten, Some(Expression::from(42)));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct RegisteredConstants;

impl RegisteredConstants {
    /// Return the strategy.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl SimplificationStrategy for RegisteredConstants {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("registered_constants")
    }

    fn rewrite(&self, node: &Expression, context: &SimplifyContext<'_>) -> Option<Expression> {
        let ExpressionKind::Identifier(identifier) = node.kind() else {
            return None;
        };
        if BuiltinConstant::of_identifier(identifier).is_some() {
            return None;
        }
        let constant = context.registry()?.constant(identifier)?;
        Some(Expression::literal(constant.value().clone()))
    }
}
