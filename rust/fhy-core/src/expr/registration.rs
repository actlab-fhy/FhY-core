//! Registration of the expression passes in the process-wide pass registry.

use crate::pass::{CompilerPass, PassRegistrationError, register_pass};

use super::node::Expression;
use super::pattern::RewriteRuleApplier;

/// Register the expression passes in the process-wide pass registry.
///
/// Registers [`RewriteRuleApplier`] under its name,
/// `fhy_core.symbolic.expression.apply_rewrite_rules`, and its
/// description; [`create_pass`](crate::pass::create_pass)
/// then builds an applier with no rules. Registering again changes
/// nothing.
///
/// # Errors
///
/// Returns an error if another pass type is registered under one of the
/// names.
///
/// # Examples
///
/// ```
/// use fhy_core::pass::create_pass;
/// use fhy_core::expr::{Expression, register_expression_passes};
///
/// register_expression_passes()?;
/// let applier =
///     create_pass::<Expression, Expression>("fhy_core.symbolic.expression.apply_rewrite_rules")?;
///
/// assert_eq!(
///     applier.description(),
///     "Apply a sequence of rewrite rules bottom-up over an expression tree."
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn register_expression_passes() -> Result<(), PassRegistrationError> {
    let applier = RewriteRuleApplier::new([]);
    register_pass::<RewriteRuleApplier, Expression, Expression>(
        &applier.name(),
        &applier.description(),
        || RewriteRuleApplier::new([]),
    )
}
