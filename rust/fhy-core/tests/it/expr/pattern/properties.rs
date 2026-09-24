//! Property tests for pattern matching and the rewrite walk.
//!
//! Covers a pattern mirroring a tree's exact shape (it matches the tree and
//! binds each leaf capture to that leaf, in leaf order), the wildcard,
//! agreement of `does_pattern_match` with `match_pattern`, a mirror whose
//! root operation is swapped, a capture repeated across both operands, the
//! empty rule list as the identity, a semantics-preserving rule set checked
//! against a reference evaluator and for firing once per wrap, the link
//! between firings and handle identity, and what counts as a change.
//!
//! Public API only (`fhy_core::expr::pattern`).

use crate::support::expression as expression_support;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use expression_support::{
    ALL_BINARY_OPERATIONS as BINARY_OPERATIONS, IDENTIFIER_POOL as POOL, build_expression_strategy,
    copy_deeply,
};
use fhy_core::expr::pattern::{
    CallbackError, MatchBindings, Pattern, RewriteRule, apply_rewrite_rules, does_pattern_match,
    match_pattern,
};
use fhy_core::expr::{
    BinaryOperation, Expression, ExpressionKind, LiteralKind, LiteralValue, UnaryOperation,
    build_call,
};
use fhy_core::identifier::Identifier;
use proptest::prelude::*;
use proptest::sample::select;

/// The unary operations of the numeric fragment the reference evaluator
/// knows.
const NUMERIC_UNARY_OPERATIONS: [UnaryOperation; 2] =
    [UnaryOperation::Negate, UnaryOperation::Positive];

/// The ring operations of the numeric fragment the reference evaluator
/// knows.
const NUMERIC_BINARY_OPERATIONS: [BinaryOperation; 3] = [
    BinaryOperation::Add,
    BinaryOperation::Subtract,
    BinaryOperation::Multiply,
];

/// The division operations of the numeric fragment, whose divisor is always
/// a non-zero literal.
const NUMERIC_DIVISION_OPERATIONS: [BinaryOperation; 2] =
    [BinaryOperation::FloorDivide, BinaryOperation::Modulo];

/// The divisors of the division operations.
const NONZERO_DIVISORS: [i64; 16] = [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8];

/// The built-in functions of the numeric fragment, each the identity on an
/// integer.
const INTEGER_RESULT_FUNCTIONS: [&str; 3] = ["floor", "ceil", "round"];

/// How a node of a numeric tree is wrapped in a value-preserving no-op.
#[derive(Debug, Clone, Copy)]
enum Wrap {
    None,
    AddZero,
    MultiplyOne,
    DoubleNegate,
}

/// Return `expression` wrapped by `wrap`.
fn apply_wrap(expression: Expression, wrap: Wrap) -> Expression {
    match wrap {
        Wrap::None => expression,
        Wrap::AddZero => expression + 0,
        Wrap::MultiplyOne => expression * 1,
        Wrap::DoubleNegate => -(-expression),
    }
}

/// Return a strategy for wraps, `Wrap::None` half the time.
fn build_wrap_strategy() -> impl Strategy<Value = Wrap> {
    prop_oneof![
        3 => Just(Wrap::None),
        1 => Just(Wrap::AddZero),
        1 => Just(Wrap::MultiplyOne),
        1 => Just(Wrap::DoubleNegate),
    ]
}

/// Return a strategy for binary trees at the root, with a binary operation
/// other than the root's.
fn build_binary_root_strategy() -> impl Strategy<Value = (Expression, BinaryOperation)> {
    (
        select(BINARY_OPERATIONS.to_vec()),
        build_expression_strategy(true),
        build_expression_strategy(true),
        1..BINARY_OPERATIONS.len(),
    )
        .prop_map(|(operation, left, right, offset)| {
            let position = BINARY_OPERATIONS
                .iter()
                .position(|candidate| *candidate == operation)
                .expect("a listed operation");
            let alternate = BINARY_OPERATIONS[(position + offset) % BINARY_OPERATIONS.len()];
            (Expression::new_binary(operation, left, right), alternate)
        })
}

/// Return a strategy for a numeric tree over [`POOL`] paired with a copy in
/// which some nodes, possibly the root, are wrapped in `+ 0`, `* 1`, or
/// `-(-...)`.
///
/// The tree holds identifiers, integer literals of every size, negation and
/// unary plus, addition, subtraction and multiplication, floor division and
/// modulo by a non-zero literal, and one-argument calls to `floor`, `ceil`
/// and `round`.
fn build_wrapped_numeric_tree_strategy() -> BoxedStrategy<(Expression, Expression)> {
    let leaf = (
        prop_oneof![
            (0..POOL.len()).prop_map(|index| Expression::from(POOL[index].clone())),
            (-64_i64..=64).prop_map(|value| Expression::from(LiteralValue::from(value))),
            any::<i64>().prop_map(|value| Expression::from(LiteralValue::from(value))),
        ],
        build_wrap_strategy(),
    )
        .prop_map(|(expression, wrap)| (expression.clone(), apply_wrap(expression, wrap)));
    leaf.prop_recursive(4, 24, 2, |inner| {
        prop_oneof![
            (
                select(NUMERIC_UNARY_OPERATIONS.to_vec()),
                inner.clone(),
                build_wrap_strategy()
            )
                .prop_map(|(operation, (plain, wrapped), wrap)| {
                    (
                        Expression::new_unary(operation, plain),
                        apply_wrap(Expression::new_unary(operation, wrapped), wrap),
                    )
                }),
            (
                select(NUMERIC_BINARY_OPERATIONS.to_vec()),
                inner.clone(),
                inner.clone(),
                build_wrap_strategy(),
            )
                .prop_map(
                    |(
                        operation,
                        (left_plain, left_wrapped),
                        (right_plain, right_wrapped),
                        wrap,
                    )| {
                        (
                            Expression::new_binary(operation, left_plain, right_plain),
                            apply_wrap(
                                Expression::new_binary(operation, left_wrapped, right_wrapped),
                                wrap,
                            ),
                        )
                    }
                ),
            (
                select(NUMERIC_DIVISION_OPERATIONS.to_vec()),
                inner.clone(),
                select(NONZERO_DIVISORS.to_vec()),
                build_wrap_strategy(),
                build_wrap_strategy(),
            )
                .prop_map(
                    |(operation, (plain, wrapped), divisor, divisor_wrap, wrap)| {
                        let divisor = Expression::from(LiteralValue::from(divisor));
                        (
                            Expression::new_binary(operation, plain, divisor.clone()),
                            apply_wrap(
                                Expression::new_binary(
                                    operation,
                                    wrapped,
                                    apply_wrap(divisor, divisor_wrap),
                                ),
                                wrap,
                            ),
                        )
                    }
                ),
            (
                select(INTEGER_RESULT_FUNCTIONS.to_vec()),
                inner,
                build_wrap_strategy()
            )
                .prop_map(|(function_name, (plain, wrapped), wrap)| {
                    (
                        build_call(function_name, [plain]).expect("a named call"),
                        apply_wrap(
                            build_call(function_name, [wrapped]).expect("a named call"),
                            wrap,
                        ),
                    )
                }),
        ]
    })
    .boxed()
}

/// Return a strategy for values of the identifiers in [`POOL`].
fn build_environment_strategy() -> impl Strategy<Value = HashMap<Identifier, i64>> {
    prop::collection::vec(any::<i64>(), POOL.len()).prop_map(|values| {
        POOL.iter()
            .cloned()
            .zip(values)
            .collect::<HashMap<Identifier, i64>>()
    })
}

/// Evaluate a numeric tree with wrapping 64-bit integer arithmetic, floor
/// division and modulo rounding toward negative infinity as Python's do.
///
/// Wrapping arithmetic is a ring, so `x + 0`, `x * 1` and `-(-x)` equal `x`
/// for every `x`.
fn evaluate_numeric(expression: &Expression, environment: &HashMap<Identifier, i64>) -> i64 {
    match expression.kind() {
        ExpressionKind::Literal(literal) => match literal.kind() {
            LiteralKind::Int(value) => i64::try_from(value).expect("a 64-bit integer literal"),
            other => panic!("a numeric tree holds only integers, got {other:?}"),
        },
        ExpressionKind::Identifier(identifier) => environment[identifier],
        ExpressionKind::Unary(node) => {
            let operand = evaluate_numeric(node.operand(), environment);
            match node.operation() {
                UnaryOperation::Negate => operand.wrapping_neg(),
                UnaryOperation::Positive => operand,
                UnaryOperation::LogicalNot => panic!("a numeric tree has no logical not"),
            }
        }
        ExpressionKind::Binary(node) => {
            let left = evaluate_numeric(node.left(), environment);
            let right = evaluate_numeric(node.right(), environment);
            match node.operation() {
                BinaryOperation::Add => left.wrapping_add(right),
                BinaryOperation::Subtract => left.wrapping_sub(right),
                BinaryOperation::Multiply => left.wrapping_mul(right),
                BinaryOperation::FloorDivide => {
                    let quotient = left.wrapping_div(right);
                    let rounds_toward_zero =
                        left.wrapping_rem(right) != 0 && (left < 0) != (right < 0);
                    quotient - i64::from(rounds_toward_zero)
                }
                BinaryOperation::Modulo => {
                    let remainder = left.wrapping_rem(right);
                    if remainder != 0 && (remainder < 0) != (right < 0) {
                        remainder + right
                    } else {
                        remainder
                    }
                }
                other => panic!("a numeric tree has no {other:?}"),
            }
        }
        ExpressionKind::Call(node) if INTEGER_RESULT_FUNCTIONS.contains(&node.function_name()) => {
            let [argument] = node.arguments() else {
                panic!("a numeric call takes one argument, got {node:?}");
            };
            evaluate_numeric(argument, environment)
        }
        _ => panic!("a numeric tree has no {expression:?}"),
    }
}

/// Return a rewrite returning the expression bound to `x`.
fn rewrite_to_x(bindings: &MatchBindings) -> Result<Expression, CallbackError> {
    bindings
        .get("x")
        .cloned()
        .ok_or_else(|| CallbackError::new("`x` is unbound"))
}

/// Return the capture of any expression under `x`.
fn build_capture_x() -> Pattern {
    Pattern::capture("x", Pattern::wildcard()).expect("a non-empty capture name")
}

/// Return the rules `x + 0 -> x`, `x * 1 -> x` and `-(-x) -> x`, each
/// counting its firings in `fire_count`.
fn build_neutral_rules(fire_count: &Arc<AtomicUsize>) -> Vec<RewriteRule> {
    let counting_rewrite = |fire_count: &Arc<AtomicUsize>| {
        let fire_count = Arc::clone(fire_count);
        move |bindings: &MatchBindings| {
            fire_count.fetch_add(1, Ordering::SeqCst);
            rewrite_to_x(bindings)
        }
    };
    vec![
        RewriteRule::new(
            Pattern::binary(
                Some(BinaryOperation::Add),
                build_capture_x(),
                Pattern::literal(Some(LiteralValue::from(0))),
            ),
            counting_rewrite(fire_count),
        )
        .with_name("x + 0 -> x"),
        RewriteRule::new(
            Pattern::binary(
                Some(BinaryOperation::Multiply),
                build_capture_x(),
                Pattern::literal(Some(LiteralValue::from(1))),
            ),
            counting_rewrite(fire_count),
        )
        .with_name("x * 1 -> x"),
        RewriteRule::new(
            Pattern::unary(
                Some(UnaryOperation::Negate),
                Pattern::unary(Some(UnaryOperation::Negate), build_capture_x()),
            ),
            counting_rewrite(fire_count),
        )
        .with_name("-(-x) -> x"),
    ]
}

/// Return a pattern mirroring `expression`'s exact shape, capturing each
/// leaf under a fresh name, and the leaves in capture-name order.
fn build_mirroring_pattern(expression: &Expression) -> (Pattern, Vec<(String, Expression)>) {
    let mut captures = Vec::new();
    let pattern = mirror_node(expression, &mut captures);
    (pattern, captures)
}

/// Return the mirror of `node`, recording its leaves in `captures`.
fn mirror_node(node: &Expression, captures: &mut Vec<(String, Expression)>) -> Pattern {
    match node.kind() {
        ExpressionKind::Unary(unary) => Pattern::unary(
            Some(unary.operation()),
            mirror_node(unary.operand(), captures),
        ),
        ExpressionKind::Binary(binary) => {
            let left = mirror_node(binary.left(), captures);
            let right = mirror_node(binary.right(), captures);
            Pattern::binary(Some(binary.operation()), left, right)
        }
        ExpressionKind::Call(call) => Pattern::call(
            Some(call.function_name()),
            Some(
                call.arguments()
                    .iter()
                    .map(|argument| mirror_node(argument, captures))
                    .collect(),
            ),
        ),
        ExpressionKind::Piecewise(piecewise) => {
            let cases = piecewise
                .cases()
                .iter()
                .map(|(condition, value)| {
                    let condition = mirror_node(condition, captures);
                    (condition, mirror_node(value, captures))
                })
                .collect();
            let otherwise = mirror_node(piecewise.otherwise(), captures);
            Pattern::piecewise(Some(cases), otherwise).expect("a piecewise has cases")
        }
        ExpressionKind::Identifier(_) | ExpressionKind::Literal(_) => {
            let name = format!("leaf_{}", captures.len());
            captures.push((name.clone(), node.clone()));
            Pattern::capture(&name, Pattern::wildcard()).expect("a non-empty capture name")
        }
    }
}

/// Return the mirror of the children of the binary `root` under
/// `operation`.
fn mirror_binary_root_with(root: &Expression, operation: BinaryOperation) -> Pattern {
    let ExpressionKind::Binary(binary) = root.kind() else {
        panic!("a binary root, got {root:?}");
    };
    let mut captures = Vec::new();
    let left = mirror_node(binary.left(), &mut captures);
    let right = mirror_node(binary.right(), &mut captures);
    Pattern::binary(Some(operation), left, right)
}

/// Return the number of nodes in `expression`, counting each occurrence.
fn count_nodes(expression: &Expression) -> usize {
    1 + expression.children().map(count_nodes).sum::<usize>()
}

/// Return the number of wraps in `wrapped`, the copy of `plain` the wrapped
/// numeric tree strategy made: each wrap adds exactly two nodes.
fn count_wraps(plain: &Expression, wrapped: &Expression) -> usize {
    (count_nodes(wrapped) - count_nodes(plain)) / 2
}

/// Return a strategy for pairs of trees, half of them a tree and a copy of
/// it sharing no node with it, the rest two independent trees.
fn build_operand_pair_strategy() -> impl Strategy<Value = (Expression, Expression)> {
    prop_oneof![
        build_expression_strategy(true).prop_map(|tree| {
            let copy = copy_deeply(&tree);
            (tree, copy)
        }),
        (
            build_expression_strategy(true),
            build_expression_strategy(true)
        ),
    ]
}

/// Return the number of literal leaves in `expression`, counting each
/// occurrence.
fn count_literal_leaves(expression: &Expression) -> usize {
    match expression.kind() {
        ExpressionKind::Literal(_) => 1,
        _ => expression.children().map(count_literal_leaves).sum(),
    }
}

proptest! {
    /// Test a pattern mirroring a tree matches it and binds each leaf
    /// capture to a handle to that leaf, in leaf order.
    #[test]
    fn mirroring_pattern_matches_and_binds_every_leaf(expression in build_expression_strategy(true)) {
        let (mirror, captures) = build_mirroring_pattern(&expression);

        let bindings = match_pattern(&mirror, &expression).expect("no predicate");

        let bindings = bindings.expect("the mirror matches");
        let names: Vec<&str> = bindings.names().collect();
        let expected_names: Vec<&str> = captures.iter().map(|(name, _)| name.as_str()).collect();
        prop_assert_eq!(names, expected_names);
        for (name, leaf) in &captures {
            let bound = bindings.get(name).expect("every leaf is bound");
            prop_assert!(Expression::ptr_eq(bound, leaf), "{} is bound to {:?}", name, bound);
        }
    }

    /// Test the wildcard matches every tree and binds nothing.
    #[test]
    fn wildcard_pattern_matches_every_expression(expression in build_expression_strategy(true)) {
        let bindings = match_pattern(&Pattern::wildcard(), &expression).expect("no predicate");

        prop_assert!(bindings.is_some_and(|bindings| bindings.is_empty()));
    }

    /// Test `does_pattern_match` agrees with `match_pattern` for a mirror,
    /// the wildcard, and a literal outside the generated alphabet.
    #[test]
    fn does_pattern_match_agrees_with_match_pattern(expression in build_expression_strategy(true)) {
        let (mirror, _) = build_mirroring_pattern(&expression);
        let outside_alphabet = Pattern::literal(Some(
            LiteralValue::parse_text("99999.99999").expect("a decimal text"),
        ));

        for pattern in [mirror, Pattern::wildcard(), outside_alphabet] {
            let answer = does_pattern_match(&pattern, &expression).expect("no predicate");
            let bindings = match_pattern(&pattern, &expression).expect("no predicate");

            prop_assert_eq!(answer, bindings.is_some(), "for {:?}", pattern);
        }
    }

    /// Test a mirror whose root operation is swapped for another does not
    /// match.
    #[test]
    fn mirror_pattern_with_another_root_operation_does_not_match(
        (root, alternate) in build_binary_root_strategy()
    ) {
        let altered = mirror_binary_root_with(&root, alternate);

        let bindings = match_pattern(&altered, &root).expect("no predicate");

        prop_assert!(bindings.is_none(), "matched with {:?}", bindings);
    }

    /// Test a binary pattern capturing both operands under one name matches
    /// exactly when the operands are structurally equal, and binds the name
    /// to a handle to the left operand.
    #[test]
    fn repeated_capture_matches_exactly_equal_operands(
        (left, right) in build_operand_pair_strategy(),
        operation in select(BINARY_OPERATIONS.to_vec()),
    ) {
        let pattern = Pattern::binary(Some(operation), build_capture_x(), build_capture_x());
        let expression = Expression::new_binary(operation, &left, &right);

        let bindings = match_pattern(&pattern, &expression).expect("no predicate");

        prop_assert_eq!(bindings.is_some(), left == right);
        if let Some(bindings) = bindings {
            let bound = bindings.get("x").expect("x is bound");
            prop_assert!(Expression::ptr_eq(bound, &left), "x is bound to {:?}", bound);
            prop_assert_eq!(bindings.names().count(), 1);
        }
    }

    /// Test an empty rule list returns the input itself, unchanged.
    #[test]
    fn apply_rewrite_rules_with_no_rules_is_the_identity(expression in build_expression_strategy(true)) {
        let outcome = apply_rewrite_rules(&expression, &[]).expect("no rule to fail");

        prop_assert!(Expression::ptr_eq(outcome.output(), &expression));
        prop_assert!(!outcome.is_changed());
        prop_assert!(outcome.fired().is_empty());
    }

    /// Test the no-op rules `x + 0 -> x`, `x * 1 -> x` and `-(-x) -> x`
    /// fire at least once per wrap and keep the value of a tree wrapped in
    /// those no-ops.
    #[test]
    fn neutral_rules_preserve_evaluation(
        (plain, wrapped) in build_wrapped_numeric_tree_strategy(),
        environment in build_environment_strategy(),
    ) {
        let rules = build_neutral_rules(&Arc::new(AtomicUsize::new(0)));
        let wraps = count_wraps(&plain, &wrapped);

        let outcome = apply_rewrite_rules(&wrapped, &rules).expect("no callback fails");

        prop_assert!(
            outcome.fired().len() >= wraps,
            "{} firings for {} wraps",
            outcome.fired().len(),
            wraps
        );
        prop_assert_eq!(
            evaluate_numeric(outcome.output(), &environment),
            evaluate_numeric(&plain, &environment)
        );
    }

    /// Test, for rules that never return the node they matched, the rules
    /// fire at least once per wrap, the output is the input itself exactly
    /// when no rule fired, and the fired list counts every firing.
    #[test]
    fn neutral_rules_keep_identity_exactly_when_no_rule_fired(
        (plain, wrapped) in build_wrapped_numeric_tree_strategy()
    ) {
        let fire_count = Arc::new(AtomicUsize::new(0));
        let rules = build_neutral_rules(&fire_count);
        let wraps = count_wraps(&plain, &wrapped);

        let outcome = apply_rewrite_rules(&wrapped, &rules).expect("no callback fails");

        let fired = fire_count.load(Ordering::SeqCst);
        prop_assert!(fired >= wraps, "{} firings for {} wraps", fired, wraps);
        prop_assert_eq!(outcome.fired().len(), fired);
        prop_assert_eq!(Expression::ptr_eq(outcome.output(), &wrapped), fired == 0);
        prop_assert_eq!(outcome.is_changed(), fired != 0);
    }

    /// Test the no-op rules rewrite a tree sharing a subtree like its
    /// unshared copy, firing on the shared subtree once where the copy fires
    /// on each of its three occurrences.
    #[test]
    fn neutral_rules_rewrite_a_shared_subtree_like_its_unshared_copy(
        (_, wrapped) in build_wrapped_numeric_tree_strategy()
    ) {
        let shared = build_call("f", [wrapped.clone(), &wrapped + &wrapped]).expect("a named call");
        let unshared = copy_deeply(&shared);
        let rules = build_neutral_rules(&Arc::new(AtomicUsize::new(0)));

        let alone = apply_rewrite_rules(&wrapped, &rules).expect("no callback fails");
        let shared_outcome = apply_rewrite_rules(&shared, &rules).expect("no callback fails");
        let unshared_outcome = apply_rewrite_rules(&unshared, &rules).expect("no callback fails");

        prop_assert_eq!(shared_outcome.output(), unshared_outcome.output());
        prop_assert_eq!(
            unshared_outcome.fired().len() - shared_outcome.fired().len(),
            2 * alone.fired().len()
        );
    }

    /// Test a rule rewriting every literal to itself fires once per literal
    /// occurrence and never changes the tree: the output is the input
    /// itself.
    #[test]
    fn identity_rewrite_never_changes_the_tree(
        expression in build_expression_strategy(true)
    ) {
        let rule = RewriteRule::new(
            Pattern::capture("x", Pattern::literal(None)).expect("a non-empty capture name"),
            rewrite_to_x,
        );
        let literal_count = count_literal_leaves(&expression);

        let outcome = apply_rewrite_rules(&expression, &[rule]).expect("no callback fails");

        prop_assert_eq!(outcome.fired().len(), literal_count);
        prop_assert!(!outcome.is_changed());
        prop_assert!(Expression::ptr_eq(outcome.output(), &expression));
    }
}
