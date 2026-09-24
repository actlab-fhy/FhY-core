//! Registration of the expression passes in a pass registry.

use crate::pass::{PassRegistrationError, PassRegistry};

use super::node::Expression;
use super::pattern::RewriteRuleApplier;

/// Register the expression passes in `registry`.
///
/// Registers [`RewriteRuleApplier`] under its name,
/// `fhy_core.symbolic.expression.apply_rewrite_rules`, and its
/// description; [`PassRegistry::create`] then builds an applier with no
/// rules. The name is a stable registry key, not a Rust path. Registering
/// again in the same registry changes nothing.
///
/// # Errors
///
/// Returns an error if another pass is registered under one of the names.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::{Expression, register_expression_passes};
/// use fhy_core::pass::PassRegistry;
///
/// let mut registry = PassRegistry::new();
/// register_expression_passes(&mut registry)?;
/// let applier = registry
///     .create::<Expression, Expression>("fhy_core.symbolic.expression.apply_rewrite_rules")?;
///
/// assert_eq!(
///     applier.description(),
///     "Apply a sequence of rewrite rules bottom-up over an expression tree."
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn register_expression_passes(
    registry: &mut PassRegistry,
) -> Result<(), PassRegistrationError> {
    registry.register::<RewriteRuleApplier, Expression, Expression>(|| RewriteRuleApplier::new([]))
}
