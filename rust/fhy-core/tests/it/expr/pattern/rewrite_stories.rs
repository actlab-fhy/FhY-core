//! Tests for rewrite rules and the rewrite walk: firing one rule at the
//! root, the bottom-up single pass, rule priority, guards, which handles the
//! walk keeps, shared subtrees, what counts as a change, the record of fired
//! rules, failing callbacks and rebuilds, and trees thousands of levels
//! deep.
//!
//! Public API only (`fhy_core::expr::pattern`).

use crate::support::expression as expression_support;
use crate::support::pattern as pattern_support;
use crate::support::stack as stack_support;

use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use expression_support::{build_deep_sum, build_identifier, build_literal};
use fhy_core::expr::pattern::{
    CallbackError, FiredRule, MatchBindings, Pattern, RewriteError, RewriteOutcome, RewriteRule,
    apply_rewrite_rule, apply_rewrite_rules,
};
use fhy_core::expr::{
    BinaryOperation, Expression, ExpressionBuildError, ExpressionKind, UnaryOperation, build_call,
    build_piecewise,
};
use pattern_support::{
    ProbeError, build_capture, build_literal_pattern, build_x_minus_x_rule, build_x_plus_zero_rule,
    build_x_times_one_rule, expect_probe_error, rewrite, rewrite_to_capture, rewrite_to_literal,
};
use rstest::rstest;
use stack_support::{SMALL_STACK_DEPTH, run_on_small_stack};

/// Return `x + 0` for the reference `x`.
fn build_plus_zero(x: &Expression) -> Expression {
    Expression::new_binary(BinaryOperation::Add, x, build_literal(0))
}

/// Apply `rule` at the root of `expression`, failing the test if a callback
/// fails.
fn rewrite_root(rule: &RewriteRule, expression: &Expression) -> Option<Expression> {
    apply_rewrite_rule(rule, expression).expect("no callback fails")
}

/// Return the `(rule index, name)` of every firing.
fn describe_fired(outcome: &RewriteOutcome) -> Vec<(usize, Option<&str>)> {
    outcome
        .fired()
        .iter()
        .map(|fired| (fired.rule_index(), fired.name()))
        .collect()
}

/// Return an unnamed rule rewriting every expression to the literal
/// `value`.
fn build_constant_rule(value: i64) -> RewriteRule {
    RewriteRule::new(Pattern::wildcard(), rewrite_to_literal(value))
}

/// Return the rule rewriting any node to itself through a capture.
fn build_identity_rule() -> RewriteRule {
    RewriteRule::new(build_capture("x"), rewrite_to_capture("x")).with_name("x -> x")
}

/// Return a guard failing with a [`ProbeError`].
fn fail_guard(_: &MatchBindings) -> Result<bool, CallbackError> {
    Err(CallbackError::new(ProbeError("guard failed")))
}

/// Return a rewrite failing with a [`ProbeError`].
fn fail_rewrite(_: &MatchBindings) -> Result<Expression, CallbackError> {
    Err(CallbackError::new(ProbeError("rewrite failed")))
}

/// Return the build error inside `error`, with the responsible rule's index
/// and name.
fn expect_rebuild_error(error: &RewriteError) -> (usize, Option<&str>, &ExpressionBuildError) {
    let RewriteError::Rebuild {
        rule_index,
        rule_name,
        source,
    } = error
    else {
        panic!("expected a rebuild failure, got {error:?}");
    };
    (*rule_index, rule_name.as_deref(), source)
}

/// Return the callback error inside `error`, with the failing rule's index
/// and name.
fn expect_callback_error(error: &RewriteError) -> (usize, Option<&str>, &CallbackError) {
    let RewriteError::Callback {
        rule_index,
        rule_name,
        source,
    } = error
    else {
        panic!("expected a callback failure, got {error:?}");
    };
    (*rule_index, rule_name.as_deref(), source)
}

// =============================================================================
// RewriteRule
// =============================================================================

/// Test a new rule is unnamed and unguarded: it fires wherever its pattern
/// matches.
#[test]
fn rewrite_rule_new_is_unnamed_and_unguarded() {
    let rule = build_constant_rule(0);

    let rewritten = rewrite_root(&rule, &build_literal(5));

    assert_eq!(rule.name(), None);
    assert_eq!(rewritten, Some(build_literal(0)));
}

/// Test a rule matches with the pattern it was built from.
#[rstest]
#[case::matching(build_literal(0), Some(build_literal(1)))]
#[case::not_matching(build_literal(2), None)]
fn rewrite_rule_new_matches_with_the_given_pattern(
    #[case] expression: Expression,
    #[case] expected: Option<Expression>,
) {
    let rule = RewriteRule::new(build_literal_pattern(0), rewrite_to_literal(1));

    let rewritten = rewrite_root(&rule, &expression);

    assert_eq!(rewritten, expected);
}

/// Test a rule's name is the last one given.
#[test]
fn rewrite_rule_with_name_replaces_an_earlier_name() {
    let rule = build_constant_rule(0)
        .with_name("first")
        .with_name("second");

    let name = rule.name();

    assert_eq!(name, Some("second"));
}

/// Test a rule's guard is the last one given.
#[test]
fn rewrite_rule_with_guard_replaces_an_earlier_guard() {
    let rule = build_constant_rule(0)
        .with_guard(|_| Ok(false))
        .with_guard(|_| Ok(true));

    let rewritten = rewrite_root(&rule, &build_literal(5));

    assert_eq!(rewritten, Some(build_literal(0)));
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

/// Test a rule's debug form names the rule.
#[test]
fn rewrite_rule_debug_names_the_rule() {
    let rule = build_x_plus_zero_rule();

    let debug = format!("{rule:?}");

    assert!(debug.contains("x + 0 -> x"), "{debug}");
}

// =============================================================================
// apply_rewrite_rule
// =============================================================================

/// Test a matching rule returns its rewrite: here a handle to the captured
/// node.
#[test]
fn apply_rewrite_rule_returns_the_rewrite_on_a_match() {
    let (_, x) = build_identifier("x");

    let rewritten = rewrite_root(&build_x_plus_zero_rule(), &build_plus_zero(&x));

    let rewritten = rewritten.expect("`x + 0` matches");
    assert!(Expression::ptr_eq(&rewritten, &x));
}

/// Test a rule whose pattern does not match returns `None` without calling
/// its guard or rewrite.
#[test]
fn apply_rewrite_rule_returns_none_when_the_pattern_does_not_match() {
    let (_, x) = build_identifier("x");
    let calls = Arc::new(AtomicUsize::new(0));
    let (guard_calls, rewrite_calls) = (Arc::clone(&calls), Arc::clone(&calls));
    let rule = RewriteRule::new(
        Pattern::binary(
            Some(BinaryOperation::Add),
            build_capture("x"),
            build_literal_pattern(0),
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
fn apply_rewrite_rule_returns_none_when_the_guard_refuses() {
    let (_, x) = build_identifier("x");
    let rewrite_calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&rewrite_calls);
    let rule = RewriteRule::new(
        Pattern::binary(
            Some(BinaryOperation::Add),
            build_capture("x"),
            Pattern::literal(None),
        ),
        move |bindings| {
            counter.fetch_add(1, Ordering::SeqCst);
            rewrite_to_capture("x")(bindings)
        },
    )
    .with_guard(|_| Ok(false));

    let rewritten = rewrite_root(&rule, &build_plus_zero(&x));

    assert_eq!(rewritten, None);
    assert_eq!(rewrite_calls.load(Ordering::SeqCst), 0);
}

/// Test a guard answering true lets the rule fire.
#[test]
fn apply_rewrite_rule_fires_when_the_guard_allows() {
    let (_, x) = build_identifier("x");
    let rule = RewriteRule::new(
        Pattern::binary(
            Some(BinaryOperation::Add),
            build_capture("x"),
            Pattern::literal(None),
        ),
        rewrite_to_capture("x"),
    )
    .with_guard(|_| Ok(true));

    let rewritten = rewrite_root(&rule, &build_plus_zero(&x));

    assert!(rewritten.is_some_and(|result| Expression::ptr_eq(&result, &x)));
}

/// Test the guard receives the match's bindings.
#[rstest]
#[case::literal_operand(build_literal(3), true)]
#[case::identifier_operand(build_identifier("y").1, false)]
fn apply_rewrite_rule_guard_sees_the_bindings(
    #[case] operand: Expression,
    #[case] expected_fire: bool,
) {
    let rule = RewriteRule::new(
        Pattern::unary(Some(UnaryOperation::Negate), build_capture("x")),
        rewrite_to_literal(0),
    )
    .with_guard(|bindings| {
        let bound = bindings
            .get("x")
            .ok_or_else(|| CallbackError::new(ProbeError("unbound")))?;
        Ok(matches!(bound.kind(), ExpressionKind::Literal(_)))
    });

    let rewritten = rewrite_root(&rule, &-&operand);

    assert_eq!(rewritten.is_some(), expected_fire, "got {rewritten:?}");
}

/// Test a failing guard's error is returned unchanged.
#[test]
fn apply_rewrite_rule_returns_the_guard_error() {
    let rule = build_constant_rule(0).with_guard(fail_guard);

    let result = apply_rewrite_rule(&rule, &build_literal(5));

    let error = result.expect_err("the guard fails");
    assert_eq!(expect_probe_error(&error), &ProbeError("guard failed"));
}

/// Test a failing rewrite's error is returned unchanged.
#[test]
fn apply_rewrite_rule_returns_the_rewrite_error() {
    let rule = RewriteRule::new(Pattern::wildcard(), fail_rewrite);

    let result = apply_rewrite_rule(&rule, &build_literal(5));

    let error = result.expect_err("the rewrite fails");
    assert_eq!(expect_probe_error(&error), &ProbeError("rewrite failed"));
}

/// Test a failing predicate in the rule's pattern is returned unchanged.
#[test]
fn apply_rewrite_rule_returns_the_predicate_error() {
    let rule = RewriteRule::new(
        Pattern::predicate(|_| Err(CallbackError::new(ProbeError("predicate failed")))),
        rewrite_to_literal(0),
    );

    let result = apply_rewrite_rule(&rule, &build_literal(5));

    let error = result.expect_err("the predicate fails");
    assert_eq!(expect_probe_error(&error), &ProbeError("predicate failed"));
}

/// Test a rule is tried at the root only.
#[test]
fn apply_rewrite_rule_operates_at_the_root_only() {
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

/// Test a walk in which no rule fires returns the input itself, unchanged.
#[test]
fn apply_rewrite_rules_preserves_identity_when_no_rule_fires() {
    let (_, x) = build_identifier("x");
    let expression = Expression::new_binary(BinaryOperation::Add, &x, build_literal(1));

    let outcome = rewrite(&expression, &[build_x_plus_zero_rule()]);

    assert!(Expression::ptr_eq(outcome.output(), &expression));
    assert!(!outcome.is_changed());
    assert!(outcome.fired().is_empty());
}

/// Test a walk in which a rule fires reports a change.
#[test]
fn apply_rewrite_rules_reports_a_change_when_a_rule_fires() {
    let (_, x) = build_identifier("x");

    let outcome = rewrite(&build_plus_zero(&x), &[build_x_plus_zero_rule()]);

    assert!(outcome.is_changed());
    assert_eq!(describe_fired(&outcome), vec![(0, Some("x + 0 -> x"))]);
}

/// Test a rule firing at the root and returning the root itself leaves the
/// tree unchanged, although it fired.
#[test]
fn apply_rewrite_rules_with_an_identity_rewrite_at_the_root_is_unchanged() {
    let expression = build_literal(5);

    let outcome = rewrite(&expression, &[build_identity_rule()]);

    assert!(Expression::ptr_eq(outcome.output(), &expression));
    assert!(!outcome.is_changed());
    assert_eq!(describe_fired(&outcome), vec![(0, Some("x -> x"))]);
}

/// Test a rule firing below the root and returning the node it matched
/// changes nothing either: the root is not rebuilt, and the output is the
/// input itself.
#[test]
fn apply_rewrite_rules_with_an_identity_rewrite_below_the_root_is_unchanged() {
    let (_, x) = build_identifier("x");
    let expression = -&x;
    let rule = RewriteRule::new(build_capture_of_identifier("x"), rewrite_to_capture("x"));

    let outcome = rewrite(&expression, &[rule]);

    assert!(!outcome.is_changed());
    assert!(Expression::ptr_eq(outcome.output(), &expression));
    assert_eq!(describe_fired(&outcome), vec![(0, None)]);
}

/// Return the pattern capturing any identifier reference under `name`.
fn build_capture_of_identifier(name: &str) -> Pattern {
    Pattern::capture(name, Pattern::identifier(None)).expect("a non-empty capture name")
}

/// Test `into_output` gives the rewritten tree.
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

/// Test a matching root is rewritten.
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
    let ExpressionKind::Binary(node) = outcome.output().kind() else {
        panic!("a product at the root, got {:?}", outcome.output());
    };
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
    assert_eq!(describe_fired(&outcome), vec![(0, None)]);
    assert_eq!(later_calls.load(Ordering::SeqCst), 0);
}

/// Test a rule whose guard refuses gives way to the next rule.
#[test]
fn apply_rewrite_rules_tries_the_next_rule_after_a_refusing_guard() {
    let refused = build_constant_rule(101).with_guard(|_| Ok(false));

    let outcome = rewrite(&build_literal(0), &[refused, build_constant_rule(202)]);

    assert_eq!(outcome.output(), &build_literal(202));
    assert_eq!(describe_fired(&outcome), vec![(1, None)]);
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
        describe_fired(&outcome),
        vec![(0, Some("x + 0 -> x")), (1, Some("x * 1 -> x"))]
    );
}

/// Test a replacement is not rewritten again in the same walk.
#[test]
fn apply_rewrite_rules_does_not_iterate_to_a_fixpoint() {
    let rule = RewriteRule::new(build_literal_pattern(0), |_| {
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

/// Test a parent is matched with its rewritten children.
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
    let expression =
        build_piecewise([(&condition, build_plus_zero(&x))], &otherwise).expect("a piecewise");

    let outcome = rewrite(&expression, &[build_x_plus_zero_rule()]);

    let ExpressionKind::Piecewise(node) = outcome.output().kind() else {
        panic!("a piecewise at the root, got {:?}", outcome.output());
    };
    assert!(Expression::ptr_eq(&node.cases()[0].0, &condition));
    assert!(Expression::ptr_eq(&node.cases()[0].1, &x));
    assert!(Expression::ptr_eq(node.otherwise(), &otherwise));
}

/// Test rules fire inside call arguments.
#[test]
fn apply_rewrite_rules_rewrites_inside_call_arguments() {
    let (_, x) = build_identifier("x");
    let expression =
        build_call("f", [build_plus_zero(&x), build_literal(3)]).expect("a named call");

    let outcome = rewrite(&expression, &[build_x_plus_zero_rule()]);

    let expected = build_call("f", [x, build_literal(3)]).expect("a named call");
    assert_eq!(outcome.output(), &expected);
}

/// Test rules fire inside a unary operand.
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
    let mut expression = x.clone();
    for _ in 0..8 {
        expression = build_plus_zero(&expression);
    }

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
    assert_eq!(describe_fired(&outcome), vec![(0, Some("x + 0 -> x"))]);
    let ExpressionKind::Binary(node) = outcome.output().kind() else {
        panic!("expected a product, got {:?}", outcome.output());
    };
    assert!(Expression::ptr_eq(node.left(), &x));
    assert!(Expression::ptr_eq(node.right(), &x));
}

/// Test a leaf occurring twice is rewritten once and both occurrences
/// become one replacement node.
#[test]
fn apply_rewrite_rules_rewrites_a_shared_leaf_once() {
    let zero = build_literal(0);
    let expression = Expression::new_binary(BinaryOperation::Add, &zero, &zero);
    let zero_to_five = RewriteRule::new(build_literal_pattern(0), rewrite_to_literal(5));

    let outcome = rewrite(&expression, &[zero_to_five]);

    assert_eq!(
        outcome.output(),
        &Expression::new_binary(BinaryOperation::Add, build_literal(5), build_literal(5))
    );
    assert_eq!(describe_fired(&outcome), vec![(0, None)]);
    let ExpressionKind::Binary(node) = outcome.output().kind() else {
        panic!("expected a sum, got {:?}", outcome.output());
    };
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

    assert_eq!(describe_fired(&outcome), vec![(0, Some("x + 0 -> x"))]);
    let mut node = outcome.output();
    for _ in 0..levels {
        let ExpressionKind::Binary(product) = node.kind() else {
            panic!("expected a product, got {node:?}");
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
            Ok(false)
        }),
        rewrite_to_literal(0),
    );
    let (c0, v0) = (build_literal(true), build_literal(10));
    let (c1, v1) = (build_literal(false), build_literal(11));
    let otherwise = build_literal(12);
    let piecewise =
        build_piecewise([(&c0, &v0), (&c1, &v1)], &otherwise).expect("a valid piecewise");
    let argument = build_literal(13);
    let expression = build_call("f", [&piecewise, &argument]).expect("a named call");

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
// apply_rewrite_rules: caller-driven fixpoint
// =============================================================================

/// Test repeating the walk until it reports no change reaches the fixpoint
/// and stops.
#[test]
fn apply_rewrite_rules_supports_a_caller_driven_fixpoint() {
    let (_, x) = build_identifier("x");
    let rules = [build_x_plus_zero_rule()];
    let mut current = build_plus_zero(&build_plus_zero(&x));
    let mut walks = 0;

    loop {
        let outcome = rewrite(&current, &rules);
        walks += 1;
        assert!(walks < 10, "the fixpoint loop does not stop");
        if !outcome.is_changed() {
            break;
        }
        current = outcome.into_output();
    }

    assert!(Expression::ptr_eq(&current, &x));
    assert_eq!(walks, 2);
}

// =============================================================================
// apply_rewrite_rules: failures
// =============================================================================

/// Test a failing guard ends the walk with an error naming its rule.
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

/// Test a failing rewrite ends the walk with an error naming its rule.
#[test]
fn apply_rewrite_rules_reports_a_failing_rewrite_with_its_rule() {
    let failing = RewriteRule::new(Pattern::wildcard(), fail_rewrite);

    let result = apply_rewrite_rules(&build_literal(5), &[failing]);

    let error = result.expect_err("the rewrite fails");
    let (rule_index, rule_name, source) = expect_callback_error(&error);
    assert_eq!(rule_index, 0);
    assert_eq!(rule_name, None);
    assert_eq!(expect_probe_error(source), &ProbeError("rewrite failed"));
}

/// Test a failing predicate in a rule's pattern ends the walk with an error
/// naming its rule.
#[test]
fn apply_rewrite_rules_reports_a_failing_predicate_with_its_rule() {
    let failing = RewriteRule::new(
        Pattern::predicate(|_| Err(CallbackError::new(ProbeError("predicate failed")))),
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
            Ok(false)
        }),
        rewrite_to_literal(0),
    );
    let failing_on_one = RewriteRule::new(build_literal_pattern(1), fail_rewrite);
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
    let never_firing = RewriteRule::new(build_literal_pattern(9), rewrite_to_literal(0));
    let true_to_one =
        RewriteRule::new(build_literal_pattern(true), rewrite_to_literal(1)).with_name("true -> 1");
    let expression = build_piecewise([(build_literal(true), build_literal(5))], build_literal(6))
        .expect("a valid piecewise");

    let result = apply_rewrite_rules(&expression, &[never_firing, true_to_one]);

    let error = result.expect_err("the rebuild fails");
    assert_eq!(
        expect_rebuild_error(&error),
        (
            1,
            Some("true -> 1"),
            &ExpressionBuildError::NonBooleanConditionLiteral { case_index: 0 }
        )
    );
}

/// Test a failing rebuild names the rule that rewrote the refused
/// condition, not another rule that rewrote a sibling after it.
#[test]
fn apply_rewrite_rules_blames_the_rule_that_rewrote_the_refused_condition() {
    let false_to_one = RewriteRule::new(build_literal_pattern(false), rewrite_to_literal(1))
        .with_name("false -> 1");
    let six_to_seven =
        RewriteRule::new(build_literal_pattern(6), rewrite_to_literal(7)).with_name("6 -> 7");
    let expression = build_piecewise(
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
            &ExpressionBuildError::NonBooleanConditionLiteral { case_index: 1 }
        )
    );
}

/// Test a failing rebuild names the rule that rewrote the refused
/// condition when the condition is a shared node rewritten at an earlier
/// occurrence, not the rule that fired last.
#[test]
fn apply_rewrite_rules_blames_the_rule_that_rewrote_a_shared_refused_condition() {
    let six_to_seven =
        RewriteRule::new(build_literal_pattern(6), rewrite_to_literal(7)).with_name("6 -> 7");
    let true_to_one =
        RewriteRule::new(build_literal_pattern(true), rewrite_to_literal(1)).with_name("true -> 1");
    let (_, x) = build_identifier("x");
    let shared_true = build_literal(true);
    let expression = build_piecewise(
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
            &ExpressionBuildError::NonBooleanConditionLiteral { case_index: 1 }
        )
    );
}

/// Test each rewrite error displays its documented message.
#[rstest]
#[case::unnamed_callback(
    RewriteError::Callback {
        rule_index: 2,
        rule_name: None,
        source: CallbackError::new("inner"),
    },
    "rewrite rule 2 failed"
)]
#[case::named_callback(
    RewriteError::Callback {
        rule_index: 0,
        rule_name: Some(String::from("x + 0 -> x")),
        source: CallbackError::new("inner"),
    },
    "rewrite rule 0 (x + 0 -> x) failed"
)]
#[case::unnamed_rebuild(
    RewriteError::Rebuild {
        rule_index: 1,
        rule_name: None,
        source: ExpressionBuildError::NonBooleanConditionLiteral { case_index: 0 },
    },
    "rebuilding a node after rewrite rule 1 failed"
)]
#[case::named_rebuild(
    RewriteError::Rebuild {
        rule_index: 0,
        rule_name: Some(String::from("true -> 1")),
        source: ExpressionBuildError::NonBooleanConditionLiteral { case_index: 0 },
    },
    "rebuilding a node after rewrite rule 0 (true -> 1) failed"
)]
fn rewrite_error_display_describes_the_failure(
    #[case] error: RewriteError,
    #[case] expected: &str,
) {
    let message = error.to_string();

    assert_eq!(message, expected);
}

/// Test a callback failure's source is the callback's error.
#[test]
fn rewrite_error_callback_source_is_the_callback_error() {
    let error = RewriteError::Callback {
        rule_index: 0,
        rule_name: None,
        source: CallbackError::new(ProbeError("inner")),
    };

    let source = error.source().expect("a callback failure has a source");

    let callback_error = source
        .downcast_ref::<CallbackError>()
        .expect("the source is the callback error");
    assert_eq!(expect_probe_error(callback_error), &ProbeError("inner"));
}

/// Test a rebuild failure's source is the build error.
#[test]
fn rewrite_error_rebuild_source_is_the_build_error() {
    let build_error = ExpressionBuildError::NonBooleanConditionLiteral { case_index: 3 };
    let error = RewriteError::Rebuild {
        rule_index: 0,
        rule_name: None,
        source: build_error.clone(),
    };

    let source = error.source().expect("a rebuild failure has a source");

    assert_eq!(
        source.downcast_ref::<ExpressionBuildError>(),
        Some(&build_error)
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
    let unnamed_times_one = RewriteRule::new(
        Pattern::binary(
            Some(BinaryOperation::Multiply),
            build_capture("x"),
            build_literal_pattern(1),
        ),
        rewrite_to_capture("x"),
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
        let mut expression = x.clone();
        for _ in 0..SMALL_STACK_DEPTH {
            expression = build_plus_zero(&expression);
        }

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
