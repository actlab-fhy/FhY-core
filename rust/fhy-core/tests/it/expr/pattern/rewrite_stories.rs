//! Tests for rewrite rules and the rewrite walk: firing one rule at the
//! root, the bottom-up single pass, rule priority, guards, which handles the
//! walk keeps, shared subtrees, what counts as a change, the record of fired
//! rules, failing callbacks and rebuilds, and trees thousands of levels
//! deep.

use crate::support::expression as expression_support;
use crate::support::pattern as pattern_support;
use crate::support::stack as stack_support;

use std::collections::HashMap;
use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use expression_support::{
    build_callee, build_deep_sum, build_doubling_dag, build_identifier, build_literal,
    expect_binary, expect_piecewise, expect_unary,
};
use fhy_core::expr::pattern::{
    CallbackError, Capture, FiredRule, MatchBindings, Pattern, RewriteError, RewriteOutcome,
    RewriteRule, Rule, apply_rewrite_rules,
};
use fhy_core::expr::{
    BinaryOperation, Expression, ExpressionKind, PiecewiseError, RebuildError, UnaryOperation,
};
use fhy_core::identifier::Identifier;
use pattern_support::{
    ProbeError, build_identity_rule, build_plus_zero, build_x_minus_x_rule, build_x_plus_zero_rule,
    build_x_times_one_rule, describe_fired, expect_probe_error, rewrite, rewrite_to_capture,
    rewrite_to_literal,
};
use rstest::rstest;
use stack_support::{SMALL_STACK_DEPTH, run_on_small_stack};

/// Rewrite `expression` with `rules` of any rule type, failing the test if
/// the walk fails.
fn rewrite_with<R: Rule>(expression: &Expression, rules: &[R]) -> RewriteOutcome {
    apply_rewrite_rules(expression, rules).expect("no callback or rebuild fails")
}

/// Rewrite `expression` with `rules` until a walk reports no change, and
/// return the result with the number of walks.
fn rewrite_to_fixpoint(expression: &Expression, rules: &[RewriteRule]) -> (Expression, usize) {
    let mut current = expression.clone();
    for walks in 1..10 {
        let outcome = rewrite(&current, rules);
        if !outcome.is_changed() {
            return (current, walks);
        }
        current = outcome.into_output();
    }
    panic!("the fixpoint loop does not stop");
}

/// Return `leaf` with `depth` additions of zero, each around the last.
fn build_plus_zero_chain(leaf: &Expression, depth: usize) -> Expression {
    (0..depth).fold(leaf.clone(), |expression, _| build_plus_zero(&expression))
}

/// Apply `rule` at the root of `expression`, failing the test if a callback
/// fails.
fn rewrite_root(rule: &RewriteRule, expression: &Expression) -> Option<Expression> {
    rule.apply(expression).expect("no callback fails")
}

/// Return an unnamed rule rewriting every expression to the literal
/// `value`.
fn build_constant_rule(value: i64) -> RewriteRule {
    RewriteRule::new(Pattern::wildcard(), rewrite_to_literal(value))
}

fn fail_guard(_: &MatchBindings) -> Result<bool, CallbackError> {
    Err(CallbackError::from(ProbeError("guard failed")))
}

fn fail_rewrite(_: &MatchBindings) -> Result<Expression, CallbackError> {
    Err(CallbackError::from(ProbeError("rewrite failed")))
}

/// Return the build error inside `error`, with the responsible rule's index
/// and name.
fn expect_rebuild_error(error: &RewriteError) -> (usize, Option<&str>, &RebuildError) {
    let RewriteError::Rebuild { source, .. } = error else {
        panic!("expected a rebuild failure, got {error:?}");
    };
    (error.rule_index(), error.rule_name(), source)
}

/// Return the callback error inside `error`, with the failing rule's index
/// and name.
fn expect_callback_error(error: &RewriteError) -> (usize, Option<&str>, &CallbackError) {
    let RewriteError::Callback { source, .. } = error else {
        panic!("expected a callback failure, got {error:?}");
    };
    (error.rule_index(), error.rule_name(), source)
}

/// Return `rule_index` rules that never fire followed by `last`, named
/// `rule_name` if given.
fn build_rules_ending_with(
    rule_index: usize,
    rule_name: Option<&str>,
    last: RewriteRule,
) -> Vec<RewriteRule> {
    let last = match rule_name {
        Some(name) => last.with_name(name),
        None => last,
    };
    (0..rule_index)
        .map(|_| RewriteRule::new(Pattern::literal(9), rewrite_to_literal(0)))
        .chain([last])
        .collect()
}

/// Return the error of a walk over `5` in which the rule at `rule_index`,
/// named `rule_name` if given, fails its rewrite, after rules that never
/// fire.
fn build_callback_failure(rule_index: usize, rule_name: Option<&str>) -> RewriteError {
    let failing = RewriteRule::new(Pattern::wildcard(), fail_rewrite);
    let rules = build_rules_ending_with(rule_index, rule_name, failing);
    apply_rewrite_rules(&build_literal(5), &rules).expect_err("the last rule fails")
}

/// Return the error of a walk over `{5 if true; 6 otherwise}` in which the
/// rule at `rule_index`, named `rule_name` if given, rewrites the condition
/// to `1`, after rules that never fire.
fn build_rebuild_failure(rule_index: usize, rule_name: Option<&str>) -> RewriteError {
    let true_to_one = RewriteRule::new(Pattern::literal(true), rewrite_to_literal(1));
    let rules = build_rules_ending_with(rule_index, rule_name, true_to_one);
    let expression =
        Expression::piecewise([(build_literal(true), build_literal(5))], build_literal(6))
            .expect("a valid piecewise");
    apply_rewrite_rules(&expression, &rules).expect_err("the rebuild fails")
}

// =============================================================================
// RewriteRule
// =============================================================================

/// Test a new rule is unnamed and fires wherever its pattern matches.
#[test]
fn rewrite_rule_new_is_unnamed() {
    let rule = build_constant_rule(0);

    let rewritten = rewrite_root(&rule, &build_literal(5));

    assert_eq!(rule.name(), None);
    assert_eq!(rewritten, Some(build_literal(0)));
}

#[rstest]
#[case::matching(build_literal(0), Some(build_literal(1)))]
#[case::not_matching(build_literal(2), None)]
fn rewrite_rule_new_matches_with_the_given_pattern(
    #[case] expression: Expression,
    #[case] expected: Option<Expression>,
) {
    let rule = RewriteRule::new(Pattern::literal(0), rewrite_to_literal(1));

    let rewritten = rewrite_root(&rule, &expression);

    assert_eq!(rewritten, expected);
}

#[test]
fn rewrite_rule_apply_returning_its_input_declines() {
    let expression = build_literal(5);

    let rewritten = rewrite_root(&build_identity_rule(), &expression);

    assert_eq!(rewritten, None);
}

#[test]
fn rewrite_rule_with_name_replaces_an_earlier_name() {
    let rule = build_constant_rule(0)
        .with_name("first")
        .with_name("second");

    let name = rule.name();

    assert_eq!(name, Some("second"));
}

/// Test a rule fires only when every guard allows it, however the guards
/// are ordered.
#[rstest]
#[case::refusing_then_allowing(false, true)]
#[case::allowing_then_refusing(true, false)]
#[case::both_refusing(false, false)]
fn rewrite_rule_with_guard_requires_every_guard(#[case] first: bool, #[case] second: bool) {
    let rule = build_constant_rule(0)
        .with_guard(move |_| Ok(first))
        .with_guard(move |_| Ok(second));

    let rewritten = rewrite_root(&rule, &build_literal(5));

    assert_eq!(rewritten, None);
}

/// Test a rule's guards run in the order they were added, and a refusing
/// guard stops the ones after it.
#[test]
fn rewrite_rule_with_guard_runs_guards_in_the_order_added() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let record = |label: &'static str, verdict: bool| {
        let calls = Arc::clone(&calls);
        move |_: &MatchBindings| {
            calls.lock().expect("an unpoisoned lock").push(label);
            Ok(verdict)
        }
    };
    let allowing = build_constant_rule(0)
        .with_guard(record("first", true))
        .with_guard(record("second", true));
    let refusing = build_constant_rule(0)
        .with_guard(record("third", false))
        .with_guard(record("fourth", true));

    let allowed = rewrite_root(&allowing, &build_literal(5));
    let refused = rewrite_root(&refusing, &build_literal(5));

    assert_eq!(allowed, Some(build_literal(0)));
    assert_eq!(refused, None);
    assert_eq!(
        *calls.lock().expect("an unpoisoned lock"),
        ["first", "second", "third"]
    );
}

#[test]
fn rewrite_rule_new_partial_fires_with_the_returned_rewrite() {
    let rule = RewriteRule::new_partial(Pattern::wildcard(), |_| Ok(Some(build_literal(7))));

    let rewritten = rewrite_root(&rule, &build_literal(5));

    assert_eq!(rewritten, Some(build_literal(7)));
}

/// Test a partial rule whose rewrite declines does not fire.
#[test]
fn rewrite_rule_apply_returns_none_when_the_rewrite_declines() {
    let rule = RewriteRule::new_partial(Pattern::wildcard(), |_| Ok(None));

    let rewritten = rewrite_root(&rule, &build_literal(5));

    assert_eq!(rewritten, None);
}

/// Test a clone of a rule shares its callbacks and keeps its name.
#[test]
fn rewrite_rule_clone_shares_the_callbacks() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let rule = RewriteRule::new(Pattern::wildcard(), move |_| {
        counter.fetch_add(1, Ordering::SeqCst);
        Ok(build_literal(0))
    })
    .with_name("count");
    let clone = rule.clone();

    let from_original = rewrite_root(&rule, &build_literal(1));
    let from_clone = rewrite_root(&clone, &build_literal(1));

    assert_eq!(from_original, Some(build_literal(0)));
    assert_eq!(from_clone, Some(build_literal(0)));
    assert_eq!(clone.name(), Some("count"));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

// =============================================================================
// RewriteRule::apply
// =============================================================================

/// Test a matching rule returns its rewrite: here a handle to the captured
/// node.
#[test]
fn rewrite_rule_apply_returns_the_rewrite_on_a_match() {
    let (_, x) = build_identifier("x");

    let rewritten = rewrite_root(&build_x_plus_zero_rule(), &build_plus_zero(&x));

    let rewritten = rewritten.expect("`x + 0` matches");
    assert!(Expression::ptr_eq(&rewritten, &x));
}

/// Test a rule whose pattern does not match returns `None` without calling
/// its guard or rewrite.
#[test]
fn rewrite_rule_apply_returns_none_when_the_pattern_does_not_match() {
    let (_, x) = build_identifier("x");
    let calls = Arc::new(AtomicUsize::new(0));
    let (guard_calls, rewrite_calls) = (Arc::clone(&calls), Arc::clone(&calls));
    let capture = Capture::new("x");
    let rule = RewriteRule::new(
        Pattern::binary(
            BinaryOperation::Add,
            Pattern::capture(&capture),
            Pattern::literal(0),
        ),
        move |_| {
            rewrite_calls.fetch_add(1, Ordering::SeqCst);
            Ok(build_literal(0))
        },
    )
    .with_guard(move |_| {
        guard_calls.fetch_add(1, Ordering::SeqCst);
        Ok(true)
    });

    let rewritten = rewrite_root(
        &rule,
        &Expression::new_binary(BinaryOperation::Add, &x, build_literal(1)),
    );

    assert_eq!(rewritten, None);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

/// Test a guard answering false stops the rule before its rewrite.
#[test]
fn rewrite_rule_apply_returns_none_when_the_guard_refuses() {
    let (_, x) = build_identifier("x");
    let rewrite_calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&rewrite_calls);
    let capture = Capture::new("x");
    let rewrite_to_x = rewrite_to_capture(&capture);
    let rule = RewriteRule::new(
        Pattern::binary(
            BinaryOperation::Add,
            Pattern::capture(&capture),
            Pattern::any_literal(),
        ),
        move |bindings| {
            counter.fetch_add(1, Ordering::SeqCst);
            rewrite_to_x(bindings)
        },
    )
    .with_guard(|_| Ok(false));

    let rewritten = rewrite_root(&rule, &build_plus_zero(&x));

    assert_eq!(rewritten, None);
    assert_eq!(rewrite_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn rewrite_rule_apply_fires_when_the_guard_allows() {
    let (_, x) = build_identifier("x");
    let capture = Capture::new("x");
    let rule = RewriteRule::new(
        Pattern::binary(
            BinaryOperation::Add,
            Pattern::capture(&capture),
            Pattern::any_literal(),
        ),
        rewrite_to_capture(&capture),
    )
    .with_guard(|_| Ok(true));

    let rewritten = rewrite_root(&rule, &build_plus_zero(&x));

    assert!(rewritten.is_some_and(|result| Expression::ptr_eq(&result, &x)));
}

#[rstest]
#[case::literal_operand(build_literal(3), true)]
#[case::identifier_operand(build_identifier("y").1, false)]
fn rewrite_rule_apply_guard_sees_the_bindings(
    #[case] operand: Expression,
    #[case] expected_fire: bool,
) {
    let x = Capture::new("x");
    let rule = RewriteRule::new(
        Pattern::unary(UnaryOperation::Negate, Pattern::capture(&x)),
        rewrite_to_literal(0),
    )
    .with_guard(move |bindings| {
        let bound = bindings
            .get(&x)
            .ok_or_else(|| CallbackError::from(ProbeError("unbound")))?;
        Ok(matches!(bound.kind(), ExpressionKind::Literal(_)))
    });

    let rewritten = rewrite_root(&rule, &-&operand);

    assert_eq!(rewritten.is_some(), expected_fire, "got {rewritten:?}");
}

#[test]
fn rewrite_rule_apply_returns_the_guard_error() {
    let rule = build_constant_rule(0).with_guard(fail_guard);

    let result = rule.apply(&build_literal(5));

    let error = result.expect_err("the guard fails");
    assert_eq!(expect_probe_error(&error), &ProbeError("guard failed"));
}

#[test]
fn rewrite_rule_apply_returns_the_rewrite_error() {
    let rule = RewriteRule::new(Pattern::wildcard(), fail_rewrite);

    let result = rule.apply(&build_literal(5));

    let error = result.expect_err("the rewrite fails");
    assert_eq!(expect_probe_error(&error), &ProbeError("rewrite failed"));
}

#[test]
fn rewrite_rule_apply_returns_the_predicate_error() {
    let rule = RewriteRule::new(
        Pattern::try_predicate(|_| Err(CallbackError::from(ProbeError("predicate failed")))),
        rewrite_to_literal(0),
    );

    let result = rule.apply(&build_literal(5));

    let error = result.expect_err("the predicate fails");
    assert_eq!(expect_probe_error(&error), &ProbeError("predicate failed"));
}

#[test]
fn rewrite_rule_apply_operates_at_the_root_only() {
    let (_, x) = build_identifier("x");
    let outer = Expression::new_binary(
        BinaryOperation::Multiply,
        build_plus_zero(&x),
        build_literal(2),
    );

    let rewritten = rewrite_root(&build_x_plus_zero_rule(), &outer);

    assert_eq!(rewritten, None);
}

// =============================================================================
// apply_rewrite_rules: identity and change
// =============================================================================

/// Test an empty rule list returns the input itself, unchanged, with no
/// firing.
#[test]
fn apply_rewrite_rules_with_no_rules_returns_the_input_itself() {
    let expression = build_literal(5);

    let outcome = rewrite(&expression, &[]);

    assert!(Expression::ptr_eq(outcome.output(), &expression));
    assert!(!outcome.is_changed());
    assert!(outcome.fired().is_empty());
}

#[test]
fn apply_rewrite_rules_preserves_identity_when_no_rule_fires() {
    let (_, x) = build_identifier("x");
    let expression = Expression::new_binary(BinaryOperation::Add, &x, build_literal(1));

    let outcome = rewrite(&expression, &[build_x_plus_zero_rule()]);

    assert!(Expression::ptr_eq(outcome.output(), &expression));
    assert!(!outcome.is_changed());
    assert!(outcome.fired().is_empty());
}

#[test]
fn apply_rewrite_rules_reports_a_change_when_a_rule_fires() {
    let (_, x) = build_identifier("x");

    let outcome = rewrite(&build_plus_zero(&x), &[build_x_plus_zero_rule()]);

    assert!(outcome.is_changed());
    assert_eq!(
        describe_fired(outcome.fired()),
        vec![(0, Some("x + 0 -> x"))]
    );
}

/// Test a rule returning the root itself does not fire, and leaves the
/// tree unchanged.
#[test]
fn apply_rewrite_rules_with_an_identity_rewrite_at_the_root_is_unchanged() {
    let expression = build_literal(5);

    let outcome = rewrite(&expression, &[build_identity_rule()]);

    assert!(Expression::ptr_eq(outcome.output(), &expression));
    assert!(!outcome.is_changed());
    assert!(outcome.fired().is_empty(), "fired {:?}", outcome.fired());
}

/// Test a rule returning the node it matched below the root does not fire
/// either: the root is not rebuilt, the output is the input itself, and the
/// operand keeps its handle.
#[test]
fn apply_rewrite_rules_with_an_identity_rewrite_below_the_root_is_unchanged() {
    let (_, x) = build_identifier("x");
    let expression = -&x;
    let captured = Capture::new("x");
    let rule = RewriteRule::new(
        Pattern::any_identifier().captured_as(&captured),
        rewrite_to_capture(&captured),
    );

    let outcome = rewrite(&expression, &[rule]);

    assert!(!outcome.is_changed());
    assert!(Expression::ptr_eq(outcome.output(), &expression));
    assert!(outcome.fired().is_empty(), "fired {:?}", outcome.fired());
    let node = expect_unary(outcome.output());
    assert!(Expression::ptr_eq(node.operand(), &x));
}

#[test]
fn apply_rewrite_rules_tries_the_next_rule_after_an_identity_rewrite() {
    let (_, a) = build_identifier("a");

    let outcome = rewrite(
        &build_plus_zero(&a),
        &[build_identity_rule(), build_x_plus_zero_rule()],
    );

    assert!(Expression::ptr_eq(outcome.output(), &a));
    assert_eq!(describe_fired(outcome.fired()), [(1, Some("x + 0 -> x"))]);
}

/// Test an identity rule over a doubling DAG 64 levels deep leaves it
/// unchanged, without a firing.
#[test]
fn apply_rewrite_rules_with_an_identity_rule_on_a_doubling_dag_is_unchanged() {
    let (_, x) = build_identifier("x");
    let dag = build_doubling_dag(&build_plus_zero(&x), 64);

    let outcome = rewrite(&dag, &[build_identity_rule()]);

    assert!(Expression::ptr_eq(outcome.output(), &dag));
    assert!(!outcome.is_changed());
    assert!(outcome.fired().is_empty());
}

#[test]
fn rewrite_outcome_into_output_returns_the_rewritten_tree() {
    let (_, x) = build_identifier("x");
    let outcome = rewrite(&build_plus_zero(&x), &[build_x_plus_zero_rule()]);

    let output = outcome.into_output();

    assert!(Expression::ptr_eq(&output, &x));
}

// =============================================================================
// apply_rewrite_rules: where rules fire
// =============================================================================

#[test]
fn apply_rewrite_rules_rewrites_at_the_root() {
    let (_, x) = build_identifier("x");

    let outcome = rewrite(&build_plus_zero(&x), &[build_x_plus_zero_rule()]);

    assert!(Expression::ptr_eq(outcome.output(), &x));
}

/// Test a matching subtree is rewritten and its parent rebuilt around it,
/// keeping the untouched sibling's handle.
#[test]
fn apply_rewrite_rules_rewrites_a_subtree_and_rebuilds_its_parent() {
    let (_, x) = build_identifier("x");
    let sibling = build_literal(2);
    let expression =
        Expression::new_binary(BinaryOperation::Multiply, build_plus_zero(&x), &sibling);

    let outcome = rewrite(&expression, &[build_x_plus_zero_rule()]);

    assert_eq!(
        outcome.output(),
        &Expression::new_binary(BinaryOperation::Multiply, &x, build_literal(2))
    );
    let node = expect_binary(outcome.output());
    assert!(Expression::ptr_eq(node.left(), &x));
    assert!(Expression::ptr_eq(node.right(), &sibling));
}

/// Test the first rule that fires wins, and later rules are not tried.
#[test]
fn apply_rewrite_rules_uses_the_first_rule_that_fires() {
    let later_calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&later_calls);
    let later = RewriteRule::new(Pattern::wildcard(), move |_| {
        counter.fetch_add(1, Ordering::SeqCst);
        Ok(build_literal(202))
    });

    let outcome = rewrite(&build_literal(0), &[build_constant_rule(101), later]);

    assert_eq!(outcome.output(), &build_literal(101));
    assert_eq!(describe_fired(outcome.fired()), vec![(0, None)]);
    assert_eq!(later_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn apply_rewrite_rules_tries_the_next_rule_after_a_refusing_guard() {
    let refused = build_constant_rule(101).with_guard(|_| Ok(false));

    let outcome = rewrite(&build_literal(0), &[refused, build_constant_rule(202)]);

    assert_eq!(outcome.output(), &build_literal(202));
    assert_eq!(describe_fired(outcome.fired()), vec![(1, None)]);
}

/// Test nested simplifications collapse in one bottom-up walk, the child's
/// rule firing before the parent's.
#[test]
fn apply_rewrite_rules_walks_bottom_up_in_one_pass() {
    let (_, x) = build_identifier("x");
    let expression = Expression::new_binary(
        BinaryOperation::Multiply,
        build_plus_zero(&x),
        build_literal(1),
    );

    let outcome = rewrite(
        &expression,
        &[build_x_plus_zero_rule(), build_x_times_one_rule()],
    );

    assert!(Expression::ptr_eq(outcome.output(), &x));
    assert_eq!(
        describe_fired(outcome.fired()),
        vec![(0, Some("x + 0 -> x")), (1, Some("x * 1 -> x"))]
    );
}

/// Test a replacement is not rewritten again in the same walk.
#[test]
fn apply_rewrite_rules_does_not_iterate_to_a_fixpoint() {
    let rule = RewriteRule::new(Pattern::literal(0), |_| {
        Ok(Expression::new_binary(
            BinaryOperation::Add,
            build_literal(0),
            build_literal(0),
        ))
    });

    let outcome = rewrite(&build_literal(0), &[rule]);

    assert_eq!(
        outcome.output(),
        &Expression::new_binary(BinaryOperation::Add, build_literal(0), build_literal(0))
    );
    assert_eq!(outcome.fired().len(), 1);
}

#[test]
fn apply_rewrite_rules_matches_parents_against_rewritten_children() {
    let (_, x) = build_identifier("x");
    let expression = build_plus_zero(&build_plus_zero(&x));

    let outcome = rewrite(&expression, &[build_x_plus_zero_rule()]);

    assert!(Expression::ptr_eq(outcome.output(), &x));
    assert_eq!(outcome.fired().len(), 2);
}

/// Test a rewrite inside one piecewise branch keeps the other branches'
/// handles.
#[test]
fn apply_rewrite_rules_rewrites_inside_a_piecewise_branch() {
    let (_, condition) = build_identifier("c");
    let (_, x) = build_identifier("x");
    let (_, otherwise) = build_identifier("y");
    let expression = Expression::piecewise([(&condition, build_plus_zero(&x))], &otherwise)
        .expect("a piecewise");

    let outcome = rewrite(&expression, &[build_x_plus_zero_rule()]);

    let node = expect_piecewise(outcome.output());
    assert!(Expression::ptr_eq(&node.cases()[0].0, &condition));
    assert!(Expression::ptr_eq(&node.cases()[0].1, &x));
    assert!(Expression::ptr_eq(node.otherwise(), &otherwise));
}

#[test]
fn apply_rewrite_rules_rewrites_inside_call_arguments() {
    let (_, x) = build_identifier("x");
    let expression = Expression::call(build_callee("f"), [build_plus_zero(&x), build_literal(3)]);

    let outcome = rewrite(&expression, &[build_x_plus_zero_rule()]);

    let expected = Expression::call(build_callee("f"), [x, build_literal(3)]);
    assert_eq!(outcome.output(), &expected);
}

#[test]
fn apply_rewrite_rules_rewrites_inside_a_unary_operand() {
    let (_, x) = build_identifier("x");
    let expression = -build_plus_zero(&x);

    let outcome = rewrite(&expression, &[build_x_plus_zero_rule()]);

    assert_eq!(outcome.output(), &-&x);
}

/// Test `x - x -> 0` fires on equal operands.
#[test]
fn apply_rewrite_rules_applies_a_repeated_capture_rule() {
    let (_, x) = build_identifier("x");

    let outcome = rewrite(
        &Expression::new_binary(BinaryOperation::Subtract, &x, &x),
        &[build_x_minus_x_rule()],
    );

    assert_eq!(outcome.output(), &build_literal(0));
}

/// Test `x - x -> 0` leaves different operands alone.
#[test]
fn apply_rewrite_rules_repeated_capture_rule_skips_different_operands() {
    let (_, x) = build_identifier("x");
    let (_, y) = build_identifier("y");
    let expression = Expression::new_binary(BinaryOperation::Subtract, &x, &y);

    let outcome = rewrite(&expression, &[build_x_minus_x_rule()]);

    assert!(Expression::ptr_eq(outcome.output(), &expression));
    assert!(!outcome.is_changed());
}

/// Test a chain of eight `+ 0` collapses in one walk, one firing per level.
#[test]
fn apply_rewrite_rules_collapses_a_long_chain_in_one_walk() {
    let (_, x) = build_identifier("x");
    let expression = build_plus_zero_chain(&x, 8);

    let outcome = rewrite(&expression, &[build_x_plus_zero_rule()]);

    assert!(Expression::ptr_eq(outcome.output(), &x));
    assert_eq!(outcome.fired().len(), 8);
}

/// Test a subtree occurring twice is rewritten once, its rewrite reused at
/// both occurrences and its firing recorded once.
#[test]
fn apply_rewrite_rules_rewrites_a_shared_subtree_once() {
    let (_, x) = build_identifier("x");
    let shared = build_plus_zero(&x);
    let expression = Expression::new_binary(BinaryOperation::Multiply, &shared, &shared);

    let outcome = rewrite(&expression, &[build_x_plus_zero_rule()]);

    assert_eq!(
        outcome.output(),
        &Expression::new_binary(BinaryOperation::Multiply, &x, &x)
    );
    assert_eq!(
        describe_fired(outcome.fired()),
        vec![(0, Some("x + 0 -> x"))]
    );
    let node = expect_binary(outcome.output());
    assert!(Expression::ptr_eq(node.left(), &x));
    assert!(Expression::ptr_eq(node.right(), &x));
}

/// Test a leaf occurring twice is rewritten once and both occurrences
/// become one replacement node.
#[test]
fn apply_rewrite_rules_rewrites_a_shared_leaf_once() {
    let zero = build_literal(0);
    let expression = Expression::new_binary(BinaryOperation::Add, &zero, &zero);
    let zero_to_five = RewriteRule::new(Pattern::literal(0), rewrite_to_literal(5));

    let outcome = rewrite(&expression, &[zero_to_five]);

    assert_eq!(
        outcome.output(),
        &Expression::new_binary(BinaryOperation::Add, build_literal(5), build_literal(5))
    );
    assert_eq!(describe_fired(outcome.fired()), vec![(0, None)]);
    let node = expect_binary(outcome.output());
    assert!(Expression::ptr_eq(node.left(), node.right()));
}

/// Test the DAG `x_0 = x + 0`, `x_{k+1} = x_k * x_k` with 64 levels, which
/// has 2^64 occurrences of `x + 0`, is rewritten with one firing into a DAG
/// sharing its nodes the same way.
#[test]
fn apply_rewrite_rules_rewrites_a_doubling_dag_once_per_distinct_node() {
    let levels = 64;
    let (_, x) = build_identifier("x");
    let mut dag = build_plus_zero(&x);
    for _ in 0..levels {
        dag = Expression::new_binary(BinaryOperation::Multiply, &dag, &dag);
    }

    let outcome = rewrite(&dag, &[build_x_plus_zero_rule()]);

    assert_eq!(
        describe_fired(outcome.fired()),
        vec![(0, Some("x + 0 -> x"))]
    );
    let mut node = outcome.output();
    for level in 0..levels {
        let ExpressionKind::Binary(product) = node.kind() else {
            panic!("level {level} is not a product");
        };
        assert!(Expression::ptr_eq(product.left(), product.right()));
        node = product.left();
    }
    assert!(Expression::ptr_eq(node, &x));
}

/// Test rules are tried on nodes in walk order: children before their
/// parent, a piecewise's children interleaved as condition, value, and
/// otherwise last.
#[test]
fn apply_rewrite_rules_visits_nodes_in_walk_order() {
    let visited = Arc::new(Mutex::new(Vec::<Expression>::new()));
    let recorder = Arc::clone(&visited);
    let rule = RewriteRule::new(
        Pattern::predicate(move |expression| {
            recorder
                .lock()
                .expect("an unpoisoned lock")
                .push(expression.clone());
            false
        }),
        rewrite_to_literal(0),
    );
    let (c0, v0) = (build_literal(true), build_literal(10));
    let (c1, v1) = (build_literal(false), build_literal(11));
    let otherwise = build_literal(12);
    let piecewise =
        Expression::piecewise([(&c0, &v0), (&c1, &v1)], &otherwise).expect("a valid piecewise");
    let argument = build_literal(13);
    let expression = Expression::call(build_callee("f"), [&piecewise, &argument]);

    let outcome = rewrite(&expression, &[rule]);

    assert!(Expression::ptr_eq(outcome.output(), &expression));
    let visited = visited.lock().expect("an unpoisoned lock");
    let expected = [
        &c0,
        &v0,
        &c1,
        &v1,
        &otherwise,
        &piecewise,
        &argument,
        &expression,
    ];
    assert_eq!(visited.len(), expected.len(), "visited {visited:?}");
    for (index, (node, expected_node)) in visited.iter().zip(expected).enumerate() {
        assert!(
            Expression::ptr_eq(node, expected_node),
            "node {index} is {node:?}, expected {expected_node:?}"
        );
    }
}

// =============================================================================
// Rule
// =============================================================================

/// A native rule replacing each reference to an identifier its borrowed
/// environment holds with the integer literal held for it.
struct SubstituteValues<'e> {
    values: &'e HashMap<Identifier, i64>,
}

impl Rule for SubstituteValues<'_> {
    fn apply(&self, expression: &Expression) -> Result<Option<Expression>, CallbackError> {
        let ExpressionKind::Identifier(identifier) = expression.kind() else {
            return Ok(None);
        };
        Ok(self
            .values
            .get(identifier)
            .map(|value| build_literal(*value)))
    }

    fn name(&self) -> Option<&str> {
        Some("substitute")
    }
}

/// A native, unnamed rule failing on every expression.
struct FailingRule;

impl Rule for FailingRule {
    fn apply(&self, _: &Expression) -> Result<Option<Expression>, CallbackError> {
        Err(CallbackError::from(ProbeError("native rule failed")))
    }
}

/// Test a native rule borrowing its context rewrites like any rule, and
/// its firings carry its name.
#[test]
fn apply_rewrite_rules_accepts_a_native_rule_borrowing_its_context() {
    let (a_identifier, a) = build_identifier("a");
    let (_, b) = build_identifier("b");
    let values = HashMap::from([(a_identifier, 3)]);
    let rule = SubstituteValues { values: &values };

    let outcome = apply_rewrite_rules(&(&a + &b), &[rule]).expect("no rule fails");

    assert_eq!(
        outcome.output(),
        &Expression::new_binary(BinaryOperation::Add, build_literal(3), &b)
    );
    assert_eq!(describe_fired(outcome.fired()), [(0, Some("substitute"))]);
}

/// Test a list of boxed rules mixes native and pattern rules, tried in
/// order at every node.
#[test]
fn apply_rewrite_rules_accepts_a_mixed_list_of_boxed_rules() {
    let (a_identifier, a) = build_identifier("a");
    let values = HashMap::from([(a_identifier, 5)]);
    let rules: Vec<Box<dyn Rule + '_>> = vec![
        Box::new(SubstituteValues { values: &values }),
        Box::new(build_x_plus_zero_rule()),
    ];

    let outcome = apply_rewrite_rules(&(&a + 0), &rules).expect("no rule fails");

    assert_eq!(outcome.output(), &build_literal(5));
    assert_eq!(
        describe_fired(outcome.fired()),
        [(0, Some("substitute")), (1, Some("x + 0 -> x"))]
    );
}

#[test]
fn apply_rewrite_rules_accepts_rules_by_reference_and_by_arc() {
    let (_, x) = build_identifier("x");
    let rule = build_x_plus_zero_rule();

    let by_reference = rewrite_with(&build_plus_zero(&x), &[&rule]);
    let by_arc = rewrite_with(&build_plus_zero(&x), &[Arc::new(rule)]);

    assert!(Expression::ptr_eq(by_reference.output(), &x));
    assert!(Expression::ptr_eq(by_arc.output(), &x));
    assert_eq!(describe_fired(by_arc.fired()), [(0, Some("x + 0 -> x"))]);
}

/// Test a failing native rule ends the walk with an error naming its
/// position.
#[test]
fn apply_rewrite_rules_reports_a_failing_native_rule() {
    let rules: [Box<dyn Rule>; 2] = [Box::new(build_x_plus_zero_rule()), Box::new(FailingRule)];

    let result = apply_rewrite_rules(&build_literal(5), &rules);

    let error = result.expect_err("the native rule fails");
    let (rule_index, rule_name, source) = expect_callback_error(&error);
    assert_eq!((rule_index, rule_name), (1, None));
    assert_eq!(
        expect_probe_error(source),
        &ProbeError("native rule failed")
    );
}

#[test]
fn apply_rewrite_rules_tries_the_next_rule_after_a_declining_rewrite() {
    let declining = RewriteRule::new_partial(Pattern::wildcard(), |_| Ok(None));

    let outcome = rewrite(&build_literal(0), &[declining, build_constant_rule(202)]);

    assert_eq!(outcome.output(), &build_literal(202));
    assert_eq!(describe_fired(outcome.fired()), [(1, None)]);
}

// =============================================================================
// apply_rewrite_rules: caller-driven fixpoint
// =============================================================================

/// Test repeating the walk until it reports no change reaches the fixpoint
/// and stops.
#[test]
fn apply_rewrite_rules_supports_a_caller_driven_fixpoint() {
    let (_, x) = build_identifier("x");
    let expression = build_plus_zero(&build_plus_zero(&x));

    let (fixpoint, walks) = rewrite_to_fixpoint(&expression, &[build_x_plus_zero_rule()]);

    assert!(Expression::ptr_eq(&fixpoint, &x));
    assert_eq!(walks, 2);
}

/// Test a caller-driven fixpoint stops when a rule returns its input
/// wherever it matches: the identity rule never fires, so the walk reports
/// a change only while `x + 0 -> x` fires.
#[test]
fn apply_rewrite_rules_caller_driven_fixpoint_terminates_with_an_identity_rule() {
    let (_, a) = build_identifier("a");
    let expression = build_plus_zero(&build_plus_zero(&a));

    let (fixpoint, walks) = rewrite_to_fixpoint(
        &expression,
        &[build_identity_rule(), build_x_plus_zero_rule()],
    );

    assert!(Expression::ptr_eq(&fixpoint, &a));
    assert_eq!(walks, 2);
}

// =============================================================================
// apply_rewrite_rules: failures
// =============================================================================

#[test]
fn apply_rewrite_rules_reports_a_failing_guard_with_its_rule() {
    let failing = build_constant_rule(0)
        .with_guard(fail_guard)
        .with_name("guarded");

    let result = apply_rewrite_rules(&build_literal(5), &[build_x_plus_zero_rule(), failing]);

    let error = result.expect_err("the guard fails");
    let (rule_index, rule_name, source) = expect_callback_error(&error);
    assert_eq!(rule_index, 1);
    assert_eq!(rule_name, Some("guarded"));
    assert_eq!(expect_probe_error(source), &ProbeError("guard failed"));
}

#[test]
fn apply_rewrite_rules_reports_a_failing_rewrite_with_its_rule() {
    let error = build_callback_failure(0, None);

    let (rule_index, rule_name, source) = expect_callback_error(&error);
    assert_eq!(rule_index, 0);
    assert_eq!(rule_name, None);
    assert_eq!(expect_probe_error(source), &ProbeError("rewrite failed"));
}

#[test]
fn apply_rewrite_rules_reports_a_failing_predicate_with_its_rule() {
    let failing = RewriteRule::new(
        Pattern::try_predicate(|_| Err(CallbackError::from(ProbeError("predicate failed")))),
        rewrite_to_literal(0),
    )
    .with_name("probing");

    let result = apply_rewrite_rules(&-build_literal(5), &[failing]);

    let error = result.expect_err("the predicate fails");
    let (rule_index, rule_name, source) = expect_callback_error(&error);
    assert_eq!(rule_index, 0);
    assert_eq!(rule_name, Some("probing"));
    assert_eq!(expect_probe_error(source), &ProbeError("predicate failed"));
}

/// Test the walk stops at the first failing callback: no node after it is
/// tried.
#[test]
fn apply_rewrite_rules_stops_at_the_first_failure() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);
    let counting = RewriteRule::new(
        Pattern::predicate(move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
            false
        }),
        rewrite_to_literal(0),
    );
    let failing_on_one = RewriteRule::new(Pattern::literal(1), fail_rewrite);
    let expression =
        Expression::new_binary(BinaryOperation::Add, build_literal(1), build_literal(2));

    let result = apply_rewrite_rules(&expression, &[counting, failing_on_one]);

    let error = result.expect_err("the rewrite fails at the first leaf");
    assert_eq!(expect_callback_error(&error).0, 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// Test a rewrite that makes a piecewise case condition a non-Boolean
/// literal fails the rebuild of the piecewise, naming the rule that
/// rewrote the condition.
#[test]
fn apply_rewrite_rules_reports_a_failing_rebuild_with_its_rule() {
    let error = build_rebuild_failure(1, Some("true -> 1"));

    assert_eq!(
        expect_rebuild_error(&error),
        (
            1,
            Some("true -> 1"),
            &RebuildError::Piecewise(PiecewiseError::NonBooleanConditionLiteral { case_index: 0 })
        )
    );
}

/// Test a failing rebuild names the rule that rewrote the refused
/// condition, not another rule that rewrote a sibling after it.
#[test]
fn apply_rewrite_rules_blames_the_rule_that_rewrote_the_refused_condition() {
    let false_to_one =
        RewriteRule::new(Pattern::literal(false), rewrite_to_literal(1)).with_name("false -> 1");
    let six_to_seven =
        RewriteRule::new(Pattern::literal(6), rewrite_to_literal(7)).with_name("6 -> 7");
    let expression = Expression::piecewise(
        [
            (build_literal(true), build_literal(5)),
            (build_literal(false), build_literal(6)),
        ],
        build_literal(8),
    )
    .expect("a valid piecewise");

    let result = apply_rewrite_rules(&expression, &[false_to_one, six_to_seven]);

    let error = result.expect_err("the rebuild fails");
    assert_eq!(
        expect_rebuild_error(&error),
        (
            0,
            Some("false -> 1"),
            &RebuildError::Piecewise(PiecewiseError::NonBooleanConditionLiteral { case_index: 1 })
        )
    );
}

/// Test a failing rebuild names the rule that rewrote the refused
/// condition when the condition is a shared node rewritten at an earlier
/// occurrence, not the rule that fired last.
#[test]
fn apply_rewrite_rules_blames_the_rule_that_rewrote_a_shared_refused_condition() {
    let six_to_seven =
        RewriteRule::new(Pattern::literal(6), rewrite_to_literal(7)).with_name("6 -> 7");
    let true_to_one =
        RewriteRule::new(Pattern::literal(true), rewrite_to_literal(1)).with_name("true -> 1");
    let (_, x) = build_identifier("x");
    let shared_true = build_literal(true);
    let expression = Expression::piecewise(
        [(&x, &shared_true), (&shared_true, &build_literal(6))],
        build_literal(8),
    )
    .expect("a valid piecewise");

    let result = apply_rewrite_rules(&expression, &[six_to_seven, true_to_one]);

    let error = result.expect_err("the rebuild fails");
    assert_eq!(
        expect_rebuild_error(&error),
        (
            1,
            Some("true -> 1"),
            &RebuildError::Piecewise(PiecewiseError::NonBooleanConditionLiteral { case_index: 1 })
        )
    );
}

/// Test a failing rebuild still blames the rule that rewrote the refused
/// condition after an earlier replacement was thrown away: `5 -> 6` fires
/// inside the first value, `-x -> 0` then discards its replacement, and
/// `c -> 1` rewrites the second condition, which the piecewise refuses.
#[test]
fn apply_rewrite_rules_blames_the_right_rule_after_a_discarded_replacement() {
    let (c_identifier, c) = build_identifier("c");
    let five_to_six =
        RewriteRule::new(Pattern::literal(5), rewrite_to_literal(6)).with_name("5 -> 6");
    let negation_to_zero = RewriteRule::new(
        Pattern::unary(UnaryOperation::Negate, Pattern::wildcard()),
        rewrite_to_literal(0),
    )
    .with_name("-x -> 0");
    let c_to_one = RewriteRule::new(Pattern::identifier(c_identifier), rewrite_to_literal(1))
        .with_name("c -> 1");
    let expression = Expression::piecewise(
        [
            (build_literal(true), -build_literal(5)),
            (c, build_literal(7)),
        ],
        build_literal(8),
    )
    .expect("a valid piecewise");

    let result = apply_rewrite_rules(&expression, &[five_to_six, negation_to_zero, c_to_one]);

    let error = result.expect_err("the rebuild fails");
    assert_eq!(
        expect_rebuild_error(&error),
        (
            2,
            Some("c -> 1"),
            &RebuildError::Piecewise(PiecewiseError::NonBooleanConditionLiteral { case_index: 1 })
        )
    );
}

#[rstest]
#[case::unnamed_callback(build_callback_failure(2, None), "rewrite rule 2 failed")]
#[case::named_callback(
    build_callback_failure(0, Some("x + 0 -> x")),
    "rewrite rule 0 (x + 0 -> x) failed"
)]
#[case::unnamed_rebuild(
    build_rebuild_failure(1, None),
    "rebuilding a node after rewrite rule 1 failed"
)]
#[case::named_rebuild(
    build_rebuild_failure(0, Some("true -> 1")),
    "rebuilding a node after rewrite rule 0 (true -> 1) failed"
)]
fn rewrite_error_display_describes_the_failure(
    #[case] error: RewriteError,
    #[case] expected: &str,
) {
    let message = error.to_string();

    assert_eq!(message, expected);
}

#[rstest]
#[case::unnamed_callback(build_callback_failure(2, None), 2, None)]
#[case::named_callback(build_callback_failure(1, Some("fails")), 1, Some("fails"))]
#[case::unnamed_rebuild(build_rebuild_failure(1, None), 1, None)]
#[case::named_rebuild(build_rebuild_failure(0, Some("true -> 1")), 0, Some("true -> 1"))]
fn rewrite_error_accessors_report_rule_index_and_name(
    #[case] error: RewriteError,
    #[case] rule_index: usize,
    #[case] rule_name: Option<&str>,
) {
    assert_eq!(error.rule_index(), rule_index);
    assert_eq!(error.rule_name(), rule_name);
}

#[test]
fn rewrite_error_callback_source_is_the_callers_error() {
    let error = build_callback_failure(0, None);

    let source = error.source().expect("a callback failure has a source");

    assert_eq!(
        source.downcast_ref::<ProbeError>(),
        Some(&ProbeError("rewrite failed"))
    );
}

#[test]
fn rewrite_error_rebuild_source_is_the_build_error() {
    let error = build_rebuild_failure(0, None);

    let source = error.source().expect("a rebuild failure has a source");

    assert_eq!(
        source.downcast_ref::<RebuildError>(),
        Some(&RebuildError::Piecewise(
            PiecewiseError::NonBooleanConditionLiteral { case_index: 0 }
        ))
    );
}

// =============================================================================
// FiredRule
// =============================================================================

/// Test fired rules are recorded with their index and name, children before
/// parents.
#[test]
fn fired_rule_records_index_and_name_in_walk_order() {
    let (_, x) = build_identifier("x");
    let capture = Capture::new("x");
    let unnamed_times_one = RewriteRule::new(
        Pattern::binary(
            BinaryOperation::Multiply,
            Pattern::capture(&capture),
            Pattern::literal(1),
        ),
        rewrite_to_capture(&capture),
    );
    let expression = build_plus_zero(&Expression::new_binary(
        BinaryOperation::Multiply,
        &x,
        build_literal(1),
    ));

    let outcome = rewrite(&expression, &[build_x_plus_zero_rule(), unnamed_times_one]);

    let fired: &[FiredRule] = outcome.fired();
    assert_eq!(fired.len(), 2);
    assert_eq!((fired[0].rule_index(), fired[0].name()), (1, None));
    assert_eq!(
        (fired[1].rule_index(), fired[1].name()),
        (0, Some("x + 0 -> x"))
    );
}

// =============================================================================
// Deep trees
// =============================================================================

/// Test rewriting a tree [`SMALL_STACK_DEPTH`] levels deep collapses it
/// on a small thread stack, one firing per level.
#[test]
fn apply_rewrite_rules_collapses_a_deep_tree_on_a_small_stack() {
    run_on_small_stack(|| {
        let (_, x) = build_identifier("x");
        let expression = build_plus_zero_chain(&x, SMALL_STACK_DEPTH);

        let outcome = rewrite(&expression, &[build_x_plus_zero_rule()]);

        assert!(Expression::ptr_eq(outcome.output(), &x));
        assert_eq!(outcome.fired().len(), SMALL_STACK_DEPTH);
    });
}

/// Test rewriting a tree [`SMALL_STACK_DEPTH`] levels deep in which no
/// rule fires returns the input itself on a small thread stack.
#[test]
fn apply_rewrite_rules_keeps_a_deep_tree_no_rule_touches_on_a_small_stack() {
    run_on_small_stack(|| {
        let (_, x) = build_identifier("x");
        let expression = build_deep_sum(&x, SMALL_STACK_DEPTH);

        let outcome = rewrite(&expression, &[build_x_plus_zero_rule()]);

        assert!(Expression::ptr_eq(outcome.output(), &expression));
        assert!(!outcome.is_changed());
    });
}

/// Test rewriting the bottom of a tree [`SMALL_STACK_DEPTH`] levels deep
/// rebuilds every level above it on a small thread stack.
#[test]
fn apply_rewrite_rules_rebuilds_every_level_above_a_deep_rewrite_on_a_small_stack() {
    run_on_small_stack(|| {
        let (_, x) = build_identifier("x");
        let (_, y) = build_identifier("y");
        let expression = build_deep_sum(&build_plus_zero(&x), SMALL_STACK_DEPTH);

        let outcome = rewrite(&expression, &[build_x_plus_zero_rule()]);

        assert!(outcome.is_changed());
        assert!(
            outcome.output() == &build_deep_sum(&x, SMALL_STACK_DEPTH),
            "the rewritten tree differs from the deep sum over x"
        );
        assert!(
            outcome.output() != &build_deep_sum(&y, SMALL_STACK_DEPTH),
            "the rewritten tree equals the deep sum over y"
        );
        assert_eq!(outcome.fired().len(), 1);
    });
}
