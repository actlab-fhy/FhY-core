//! End-to-end scenarios composing the pattern API the way a caller would:
//! build a rule set, rewrite a tree with it (or walk a tree matching a
//! pattern by hand), and check the resulting tree.

use crate::support::expression as expression_support;
use crate::support::pattern as pattern_support;

use expression_support::{build_identifier, build_literal};
use fhy_core::expr::pattern::{Capture, MatchBindings, Pattern, RewriteRule};
use fhy_core::expr::{BinaryOperation, Expression, ExpressionKind, UnaryOperation};
use pattern_support::{
    build_x_minus_x_rule, build_x_plus_zero_rule, build_x_times_one_rule, rewrite,
    rewrite_to_capture,
};

/// Return the rule `0 + x -> x`, named so.
#[must_use]
fn build_zero_plus_x_rule() -> RewriteRule {
    let x = Capture::new("x");
    RewriteRule::new(
        Pattern::binary(
            BinaryOperation::Add,
            Pattern::literal(0),
            Pattern::capture(&x),
        ),
        rewrite_to_capture(&x),
    )
    .with_name("0 + x -> x")
}

/// Return the rule collapsing `operation(operation(x))` to `x`.
#[must_use]
fn build_double_application_rule(operation: UnaryOperation) -> RewriteRule {
    let x = Capture::new("x");
    RewriteRule::new(
        Pattern::unary(operation, Pattern::unary(operation, Pattern::capture(&x))),
        rewrite_to_capture(&x),
    )
    .with_name("op(op(x)) -> x")
}

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

/// Return the normalizer `c + x -> x + c` for a literal `c` and an `x` that
/// is not one, which returns a sum already in that form as it is.
fn build_literal_last_normalizer() -> RewriteRule {
    let (left, right, sum) = (
        Capture::new("left"),
        Capture::new("right"),
        Capture::new("sum"),
    );
    let pattern = Pattern::binary(
        BinaryOperation::Add,
        Pattern::capture(&left),
        Pattern::capture(&right),
    )
    .captured_as(&sum);
    RewriteRule::new(pattern, move |bindings| {
        let is_literal =
            |capture: &Capture| matches!(bindings[capture].kind(), ExpressionKind::Literal(_));
        if is_literal(&left) && !is_literal(&right) {
            return Ok(Expression::new_binary(
                BinaryOperation::Add,
                &bindings[&right],
                &bindings[&left],
            ));
        }
        Ok(bindings[&sum].clone())
    })
    .with_name("c + x -> x + c")
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
        pending.extend(node.children().rev().cloned());
        if let Some(bindings) = pattern.matches(&node).expect("no predicate fails") {
            collected.push((node, bindings));
        }
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

#[test]
fn algebraic_simplifier_leaves_non_matching_input_unchanged() {
    let (_, a) = build_identifier("a");
    let expression = &a * 2;

    let outcome = rewrite(&expression, &build_algebraic_rule_set());

    assert!(Expression::ptr_eq(outcome.output(), &expression));
    assert!(!outcome.is_changed());
}

#[test]
fn algebraic_simplifier_handles_a_left_zero_addend() {
    let (_, a) = build_identifier("a");
    let expression = Expression::new_binary(BinaryOperation::Add, 0, &a);

    let outcome = rewrite(&expression, &build_algebraic_rule_set());

    assert!(Expression::ptr_eq(outcome.output(), &a));
}

/// Test a caller-driven loop reaches a fixpoint: the normalizer fires only
/// where it changes a sum, so the loop stops once nothing does.
#[test]
fn algebraic_simplifier_with_a_normalizing_rule_reaches_a_fixpoint() {
    let (_, a) = build_identifier("a");
    let (_, b) = build_identifier("b");
    let expression = Expression::new_binary(
        BinaryOperation::Add,
        Expression::new_binary(
            BinaryOperation::Multiply,
            Expression::new_binary(BinaryOperation::Add, 0, &a),
            1,
        ),
        Expression::new_binary(BinaryOperation::Add, 2, &b),
    );
    let mut rules = vec![build_literal_last_normalizer()];
    rules.extend(build_algebraic_rule_set());
    let mut current = expression;
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

    assert_eq!(
        current,
        Expression::new_binary(BinaryOperation::Add, &a, &b + 2)
    );
    assert_eq!(walks, 3);
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
    let (left, right) = (Capture::new("left"), Capture::new("right"));
    let pattern = Pattern::binary(
        BinaryOperation::Subtract,
        Pattern::capture(&left),
        Pattern::capture(&right),
    );
    let rule = RewriteRule::new(pattern, move |bindings| {
        Ok(Expression::new_binary(
            BinaryOperation::Add,
            &bindings[&left],
            -&bindings[&right],
        ))
    })
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

#[test]
fn double_negation_peephole_collapses_an_inner_double_negation() {
    let (_, a) = build_identifier("a");
    let expression = !!&a;

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
    let expression = !!!!&a;
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
    let x = Capture::new("x");
    let rules = [RewriteRule::new(
        Pattern::literal(0).captured_as(&x),
        move |bindings| Ok(-&bindings[&x]),
    )];

    let once = rewrite(&build_literal(0), &rules);
    let twice = rewrite(once.output(), &rules);

    assert_eq!(once.output(), &-build_literal(0));
    assert_eq!(once.fired().len(), 1);
    assert_eq!(twice.output(), &-(-build_literal(0)));
    assert_eq!(twice.fired().len(), 1);
}

// =============================================================================
// A subtree finder built on Pattern::matches
// =============================================================================

/// Test a read-only walk on `Pattern::matches` collects every `x + 0`
/// subtree with its capture.
#[test]
fn pattern_matches_powers_a_manual_subtree_finder() {
    let (_, a) = build_identifier("a");
    let expression = Expression::new_binary(BinaryOperation::Multiply, &a + 0, &a + 0);
    let x = Capture::new("x");
    let pattern = Pattern::binary(
        BinaryOperation::Add,
        Pattern::capture(&x),
        Pattern::literal(0),
    );

    let matches = collect_matching_subexpressions(&pattern, &expression);

    assert_eq!(matches.len(), 2);
    for (matched, bindings) in &matches {
        assert_eq!(matched, &(&a + 0));
        assert!(Expression::ptr_eq(&bindings[&x], &a));
    }
}
