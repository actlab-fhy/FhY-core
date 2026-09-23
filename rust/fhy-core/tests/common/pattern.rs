//! Shared builders for the pattern and rewrite tests.
//!
//! Included by the `pattern_*` test targets with
//! `#[path = "common/pattern.rs"] pub mod pattern_support;`, so an item one
//! target does not use is not reported as dead code there.

use std::error::Error;
use std::fmt;

use fhy_core::symbolic::expression::pattern::{
    CallbackError, MatchBindings, Pattern, RewriteOutcome, RewriteRule, apply_rewrite_rules,
    match_pattern,
};
use fhy_core::symbolic::expression::{BinaryOperation, Expression, LiteralValue, UnaryOperation};

/// An error a test callback fails with, recognizable after propagation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeError(pub &'static str);

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl Error for ProbeError {}

/// Return the [`ProbeError`] `error` wraps.
///
/// # Panics
///
/// Panics if `error` wraps another error type.
#[must_use]
pub fn expect_probe_error(error: &CallbackError) -> &ProbeError {
    error
        .inner()
        .downcast_ref::<ProbeError>()
        .unwrap_or_else(|| panic!("expected a ProbeError, got {error:?}"))
}

/// Return the pattern capturing any expression under `name`.
///
/// # Panics
///
/// Panics if `name` is empty.
#[must_use]
pub fn build_capture(name: &str) -> Pattern {
    Pattern::capture(name, Pattern::wildcard()).expect("a non-empty capture name")
}

/// Return the pattern capturing what `sub_pattern` matches under `name`.
///
/// # Panics
///
/// Panics if `name` is empty.
#[must_use]
pub fn build_capture_of(name: &str, sub_pattern: Pattern) -> Pattern {
    Pattern::capture(name, sub_pattern).expect("a non-empty capture name")
}

/// Return the pattern matching literals stored exactly as `value`.
#[must_use]
pub fn build_literal_pattern(value: impl Into<LiteralValue>) -> Pattern {
    Pattern::literal(Some(value.into()))
}

/// Return the pattern trying `alternatives` in order.
///
/// # Panics
///
/// Panics if `alternatives` is empty.
#[must_use]
pub fn build_alternatives(alternatives: Vec<Pattern>) -> Pattern {
    Pattern::alternatives(alternatives).expect("at least one alternative")
}

/// Return the pattern of a piecewise with the given case patterns.
///
/// # Panics
///
/// Panics if `cases` is `Some` of an empty list.
#[must_use]
pub fn build_piecewise_pattern(
    cases: Option<Vec<(Pattern, Pattern)>>,
    otherwise: Pattern,
) -> Pattern {
    Pattern::piecewise(cases, otherwise).expect("no empty case list")
}

/// Match `pattern` against `expression` and return the result, failing the
/// test if a predicate fails.
///
/// # Panics
///
/// Panics if a predicate in `pattern` fails.
#[must_use]
pub fn match_infallibly(pattern: &Pattern, expression: &Expression) -> Option<MatchBindings> {
    match_pattern(pattern, expression).expect("no predicate fails")
}

/// Match `pattern` against `expression` and return the bindings, failing the
/// test if it does not match.
///
/// # Panics
///
/// Panics if a predicate fails or `pattern` does not match.
#[must_use]
pub fn expect_match(pattern: &Pattern, expression: &Expression) -> MatchBindings {
    match_infallibly(pattern, expression)
        .unwrap_or_else(|| panic!("{pattern:?} does not match {expression:?}"))
}

/// Return the expression `bindings` binds to `name`.
///
/// # Panics
///
/// Panics if `name` is unbound.
#[must_use]
pub fn expect_bound<'a>(bindings: &'a MatchBindings, name: &str) -> &'a Expression {
    bindings
        .get(name)
        .unwrap_or_else(|| panic!("{name} is unbound in {bindings:?}"))
}

/// Rewrite `expression` with `rules`, failing the test if the walk fails.
///
/// # Panics
///
/// Panics if a rule's callback or rebuild fails.
#[must_use]
pub fn rewrite(expression: &Expression, rules: &[RewriteRule]) -> RewriteOutcome {
    apply_rewrite_rules(expression, rules).expect("no callback or rebuild fails")
}

/// Return a rewrite returning the expression bound to `name`, failing when
/// `name` is unbound.
pub fn rewrite_to_capture(
    name: &'static str,
) -> impl Fn(&MatchBindings) -> Result<Expression, CallbackError> + Send + Sync + 'static {
    move |bindings| {
        bindings
            .get(name)
            .cloned()
            .ok_or_else(|| CallbackError::new(ProbeError("unbound capture")))
    }
}

/// Return a rewrite returning a literal holding `value`.
pub fn rewrite_to_literal(
    value: i64,
) -> impl Fn(&MatchBindings) -> Result<Expression, CallbackError> + Send + Sync + 'static {
    move |_| Ok(Expression::from(LiteralValue::from(value)))
}

/// Return the rule `x + 0 -> x`, named so.
#[must_use]
pub fn build_x_plus_zero_rule() -> RewriteRule {
    RewriteRule::new(
        Pattern::binary(
            Some(BinaryOperation::Add),
            build_capture("x"),
            build_literal_pattern(0),
        ),
        rewrite_to_capture("x"),
    )
    .with_name("x + 0 -> x")
}

/// Return the rule `0 + x -> x`, named so.
#[must_use]
pub fn build_zero_plus_x_rule() -> RewriteRule {
    RewriteRule::new(
        Pattern::binary(
            Some(BinaryOperation::Add),
            build_literal_pattern(0),
            build_capture("x"),
        ),
        rewrite_to_capture("x"),
    )
    .with_name("0 + x -> x")
}

/// Return the rule `x * 1 -> x`, named so.
#[must_use]
pub fn build_x_times_one_rule() -> RewriteRule {
    RewriteRule::new(
        Pattern::binary(
            Some(BinaryOperation::Multiply),
            build_capture("x"),
            build_literal_pattern(1),
        ),
        rewrite_to_capture("x"),
    )
    .with_name("x * 1 -> x")
}

/// Return the rule `x - x -> 0`, named so.
#[must_use]
pub fn build_x_minus_x_rule() -> RewriteRule {
    RewriteRule::new(
        Pattern::binary(
            Some(BinaryOperation::Subtract),
            build_capture("x"),
            build_capture("x"),
        ),
        rewrite_to_literal(0),
    )
    .with_name("x - x -> 0")
}

/// Return the rule collapsing `operation(operation(x))` to `x`.
#[must_use]
pub fn build_double_application_rule(operation: UnaryOperation) -> RewriteRule {
    RewriteRule::new(
        Pattern::unary(
            Some(operation),
            Pattern::unary(Some(operation), build_capture("x")),
        ),
        rewrite_to_capture("x"),
    )
    .with_name("op(op(x)) -> x")
}
