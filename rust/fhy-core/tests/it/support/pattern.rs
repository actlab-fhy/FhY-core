//! Shared builders for the pattern and rewrite tests.

use std::error::Error;
use std::fmt;

use fhy_core::expr::pattern::{
    CallbackError, Capture, FiredRule, MatchBindings, Pattern, RewriteOutcome, RewriteRule,
    apply_rewrite_rules,
};
use fhy_core::expr::{BinaryOperation, Expression, LiteralValue};

/// An error a test callback fails with, recognizable after propagation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProbeError(pub(crate) &'static str);

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl Error for ProbeError {}

/// Return the [`ProbeError`] `error` holds.
///
/// # Panics
///
/// Panics if `error` wraps another error type.
#[must_use]
pub(crate) fn expect_probe_error(error: &CallbackError) -> &ProbeError {
    error
        .downcast_ref::<ProbeError>()
        .unwrap_or_else(|| panic!("expected a ProbeError, got {error:?}"))
}

/// Rewrite `expression` with `rules`, failing the test if the walk fails.
///
/// # Panics
///
/// Panics if a rule's callback or rebuild fails.
pub(crate) fn rewrite(expression: &Expression, rules: &[RewriteRule]) -> RewriteOutcome {
    apply_rewrite_rules(expression, rules).expect("no callback or rebuild fails")
}

/// Return `x + 0` for the reference `x`.
#[must_use]
pub(crate) fn build_plus_zero(x: &Expression) -> Expression {
    Expression::new_binary(BinaryOperation::Add, x, 0)
}

/// Return the `(rule index, name)` of every firing in `fired`.
#[must_use]
pub(crate) fn describe_fired(fired: &[FiredRule]) -> Vec<(usize, Option<&str>)> {
    fired
        .iter()
        .map(|firing| (firing.rule_index(), firing.name()))
        .collect()
}

/// Return a rewrite returning the expression bound to `capture`, failing
/// when `capture` is unbound.
pub(crate) fn rewrite_to_capture(
    capture: &Capture,
) -> impl Fn(&MatchBindings) -> Result<Expression, CallbackError> + Send + Sync + 'static {
    let capture = capture.clone();
    move |bindings| {
        bindings
            .get(&capture)
            .cloned()
            .ok_or_else(|| CallbackError::from(ProbeError("unbound capture")))
    }
}

/// Return a rewrite returning a literal holding `value`.
pub(crate) fn rewrite_to_literal(
    value: i64,
) -> impl Fn(&MatchBindings) -> Result<Expression, CallbackError> + Send + Sync + 'static {
    move |_| Ok(Expression::from(LiteralValue::from(value)))
}

/// Return the rule `x -> x`, named so, rewriting any node to itself through
/// a capture.
#[must_use]
pub(crate) fn build_identity_rule() -> RewriteRule {
    let x = Capture::new("x");
    RewriteRule::new(Pattern::capture(&x), rewrite_to_capture(&x)).with_name("x -> x")
}

/// Return the rule `x + 0 -> x`, named so.
#[must_use]
pub(crate) fn build_x_plus_zero_rule() -> RewriteRule {
    let x = Capture::new("x");
    RewriteRule::new(
        Pattern::binary(
            BinaryOperation::Add,
            Pattern::capture(&x),
            Pattern::literal(0),
        ),
        rewrite_to_capture(&x),
    )
    .with_name("x + 0 -> x")
}

/// Return the rule `x * 1 -> x`, named so.
#[must_use]
pub(crate) fn build_x_times_one_rule() -> RewriteRule {
    let x = Capture::new("x");
    RewriteRule::new(
        Pattern::binary(
            BinaryOperation::Multiply,
            Pattern::capture(&x),
            Pattern::literal(1),
        ),
        rewrite_to_capture(&x),
    )
    .with_name("x * 1 -> x")
}

/// Return the rule `x - x -> 0`, named so: one capture used twice.
#[must_use]
pub(crate) fn build_x_minus_x_rule() -> RewriteRule {
    let x = Capture::new("x");
    RewriteRule::new(
        Pattern::binary(
            BinaryOperation::Subtract,
            Pattern::capture(&x),
            Pattern::capture(&x),
        ),
        rewrite_to_literal(0),
    )
    .with_name("x - x -> 0")
}
