//! End-to-end scenarios composing the pattern API the way a caller would:
//! build a rule set, rewrite a tree with it (or walk a tree matching a
//! pattern by hand), and check the resulting tree.
//!
//! Public API only (`fhy_core::expr::pattern`).

use crate::support::expression as expression_support;
use crate::support::pattern as pattern_support;

use expression_support::{build_identifier, build_literal};
use fhy_core::expr::pattern::{MatchBindings, Pattern, RewriteRule, match_pattern};
use fhy_core::expr::{BinaryOperation, Expression, UnaryOperation};
use pattern_support::{
    build_capture, build_capture_of, build_double_application_rule, build_literal_pattern,
    build_x_minus_x_rule, build_x_plus_zero_rule, build_x_times_one_rule, build_zero_plus_x_rule,
    expect_bound, rewrite,
};

/// Return the four algebraic simplifications `x + 0 -> x`, `0 + x -> x`,
/// `x * 1 -> x`, and `x - x -> 0`.
fn build_algebraic_rule_set() -> Vec<RewriteRule> {
    vec![
        build_x_plus_zero_rule(),
        build_zero_plus_x_rule(),
        build_x_times_one_rule(),
        build_x_minus_x_rule(),
    ]
}

/// Collect every subexpression of `expression` that `pattern` matches, with
/// its bindings, in pre-order.
fn collect_matching_subexpressions(
    pattern: &Pattern,
    expression: &Expression,
) -> Vec<(Expression, MatchBindings)> {
    let mut collected = Vec::new();
    let mut pending = vec![expression.clone()];
    while let Some(node) = pending.pop() {
        if let Some(bindings) = match_pattern(pattern, &node).expect("no predicate fails") {
            collected.push((node.clone(), bindings));
        }
        let children: Vec<Expression> = node.children().cloned().collect();
        pending.extend(children.into_iter().rev());
    }
    collected
}

// =============================================================================
// A local algebraic simplifier
// =============================================================================

/// Test the four-rule simplifier reduces `((a + 0) * 1) - ((a + 0) * 1)`,
/// built from one shared subtree, to `0` in one walk, simplifying the shared
/// subtree once.
#[test]
fn algebraic_simplifier_collapses_nested_neutral_operations() {
    let (_, a) = build_identifier("a");
    let subtree = Expression::new_binary(BinaryOperation::Multiply, &a + 0, 1);
    let expression = Expression::new_binary(BinaryOperation::Subtract, &subtree, &subtree);

    let outcome = rewrite(&expression, &build_algebraic_rule_set());

    assert_eq!(outcome.output(), &build_literal(0));
    let fired_names: Vec<Option<&str>> = outcome.fired().iter().map(|fired| fired.name()).collect();
    assert_eq!(
        fired_names,
        vec![Some("x + 0 -> x"), Some("x * 1 -> x"), Some("x - x -> 0")]
    );
}

/// Test the simplifier returns an input no rule matches as itself.
#[test]
fn algebraic_simplifier_leaves_non_matching_input_unchanged() {
    let (_, a) = build_identifier("a");
    let expression = &a * 2;

    let outcome = rewrite(&expression, &build_algebraic_rule_set());

    assert!(Expression::ptr_eq(outcome.output(), &expression));
    assert!(!outcome.is_changed());
}

/// Test the `0 + x` rule fires alongside the `x + 0` rule.
#[test]
fn algebraic_simplifier_handles_a_left_zero_addend() {
    let (_, a) = build_identifier("a");
    let expression = Expression::new_binary(BinaryOperation::Add, 0, &a);

    let outcome = rewrite(&expression, &build_algebraic_rule_set());

    assert!(Expression::ptr_eq(outcome.output(), &a));
}

// =============================================================================
// A subtraction canonicalizer
// =============================================================================

/// Test `a - b -> a + (-b)` rewrites every subtraction of a nested tree in
/// one walk.
#[test]
fn subtraction_canonicalizer_rewrites_at_every_depth() {
    let (_, a) = build_identifier("a");
    let (_, b) = build_identifier("b");
    let (_, c) = build_identifier("c");
    let rule = RewriteRule::new(
        Pattern::binary(
            Some(BinaryOperation::Subtract),
            build_capture("left"),
            build_capture("right"),
        ),
        |bindings| {
            let left = expect_bound(bindings, "left");
            let right = expect_bound(bindings, "right");
            Ok(Expression::new_binary(BinaryOperation::Add, left, -right))
        },
    )
    .with_name("a - b -> a + (-b)");
    let expression = &a - (&b - &c);

    let outcome = rewrite(&expression, &[rule]);

    let expected = Expression::new_binary(
        BinaryOperation::Add,
        &a,
        -Expression::new_binary(BinaryOperation::Add, &b, -&c),
    );
    assert_eq!(outcome.output(), &expected);
    assert_eq!(outcome.fired().len(), 2);
}

// =============================================================================
// A double-negation peephole
// =============================================================================

/// Test the peephole simplifies `!(!a)` to `a`.
#[test]
fn double_negation_peephole_collapses_an_inner_double_negation() {
    let (_, a) = build_identifier("a");
    let expression = a.logical_not().logical_not();

    let outcome = rewrite(
        &expression,
        &[build_double_application_rule(UnaryOperation::LogicalNot)],
    );

    assert!(Expression::ptr_eq(outcome.output(), &a));
}

/// Test four nested negations collapse in one walk: the inner pair
/// collapses first, and the outer pair then matches over the result.
#[test]
fn double_negation_peephole_collapses_four_negations_in_one_walk() {
    let (_, a) = build_identifier("a");
    let mut expression = a.clone();
    for _ in 0..4 {
        expression = expression.logical_not();
    }
    let rules = [build_double_application_rule(UnaryOperation::LogicalNot)];

    let once = rewrite(&expression, &rules);
    let twice = rewrite(once.output(), &rules);

    assert!(Expression::ptr_eq(once.output(), &a));
    assert_eq!(once.fired().len(), 2);
    assert!(Expression::ptr_eq(twice.output(), &a));
    assert!(!twice.is_changed());
}

/// Test a replacement that would match the rule again is left for the next
/// walk: `-(0)` rewritten from `0` by `0 -> -(0)` is not rewritten again.
#[test]
fn negation_expander_needs_a_walk_per_expansion() {
    let rules = [RewriteRule::new(
        build_capture_of("x", build_literal_pattern(0)),
        |bindings| Ok(-expect_bound(bindings, "x")),
    )];

    let once = rewrite(&build_literal(0), &rules);
    let twice = rewrite(once.output(), &rules);

    assert_eq!(once.output(), &-build_literal(0));
    assert_eq!(once.fired().len(), 1);
    assert_eq!(twice.output(), &-(-build_literal(0)));
    assert_eq!(twice.fired().len(), 1);
}

// =============================================================================
// A subtree finder built on match_pattern
// =============================================================================

/// Test `match_pattern` drives a read-only walk collecting every `x + 0`
/// subtree with its capture.
#[test]
fn match_pattern_powers_a_manual_subtree_finder() {
    let (_, a) = build_identifier("a");
    let expression = Expression::new_binary(BinaryOperation::Multiply, &a + 0, &a + 0);
    let pattern = Pattern::binary(
        Some(BinaryOperation::Add),
        build_capture("x"),
        build_literal_pattern(0),
    );

    let matches = collect_matching_subexpressions(&pattern, &expression);

    assert_eq!(matches.len(), 2);
    for (matched, bindings) in &matches {
        assert_eq!(matched, &(&a + 0));
        assert!(Expression::ptr_eq(expect_bound(bindings, "x"), &a));
    }
}
