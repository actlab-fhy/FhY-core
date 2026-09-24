//! Property tests for pattern matching and the rewrite walk: patterns
//! mirroring generated trees, repeated captures and literal patterns, the
//! bindings a failed match leaves, and value-preserving rules checked
//! against a reference evaluator of numeric trees.

use crate::support::expression as expression_support;
use crate::support::pattern as pattern_support;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use expression_support::{
    ALL_BINARY_OPERATIONS as BINARY_OPERATIONS, IDENTIFIER_POOL as POOL, build_callee,
    build_expression_strategy, build_literal_strategy, copy_deeply,
};
use fhy_core::expr::builtins::BuiltinFunction;
use fhy_core::expr::pattern::{Capture, MatchBindings, Pattern, RewriteRule, apply_rewrite_rules};
use fhy_core::expr::{
    BinaryOperation, Callee, Expression, ExpressionKind, LiteralValue, UnaryOperation,
};
use fhy_core::identifier::Identifier;
use pattern_support::rewrite_to_capture;
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
    [BinaryOperation::FloorDivide, BinaryOperation::FloorMod];

const NONZERO_DIVISORS: [i64; 16] = [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8];

/// The built-in functions of the numeric fragment, each the identity on an
/// integer.
const INTEGER_RESULT_FUNCTIONS: [BuiltinFunction; 3] = [
    BuiltinFunction::Floor,
    BuiltinFunction::Ceil,
    BuiltinFunction::Round,
];

/// How a node of a numeric tree is wrapped in a value-preserving no-op.
#[derive(Debug, Clone, Copy)]
enum Wrap {
    None,
    AddZero,
    MultiplyOne,
    DoubleNegate,
}

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
                .prop_map(|(function, (plain, wrapped), wrap)| {
                    (
                        Expression::call(function, [plain]),
                        apply_wrap(Expression::call(function, [wrapped]), wrap),
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
/// division and floor modulo rounding toward negative infinity.
///
/// Wrapping arithmetic is a ring, so `x + 0`, `x * 1` and `-(-x)` equal `x`
/// for every `x`.
fn evaluate_numeric(expression: &Expression, environment: &HashMap<Identifier, i64>) -> i64 {
    match expression.kind() {
        ExpressionKind::Literal(literal) => match literal {
            LiteralValue::Int(value) => i64::try_from(value).expect("a 64-bit integer literal"),
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
                BinaryOperation::FloorMod => {
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
        ExpressionKind::Call(node)
            if INTEGER_RESULT_FUNCTIONS
                .iter()
                .any(|function| node.callee() == &Callee::Builtin(*function)) =>
        {
            let [argument] = node.arguments() else {
                panic!("a numeric call takes one argument, got {node:?}");
            };
            evaluate_numeric(argument, environment)
        }
        _ => panic!("a numeric tree has no {expression:?}"),
    }
}

/// Return the rules `x + 0 -> x`, `x * 1 -> x` and `-(-x) -> x`, each
/// counting its firings in `fire_count`.
fn build_neutral_rules(fire_count: &Arc<AtomicUsize>) -> Vec<RewriteRule> {
    let x = Capture::new("x");
    let counting_rewrite = || {
        let fire_count = Arc::clone(fire_count);
        let rewrite = rewrite_to_capture(&x);
        move |bindings: &MatchBindings| {
            fire_count.fetch_add(1, Ordering::SeqCst);
            rewrite(bindings)
        }
    };
    vec![
        RewriteRule::new(
            Pattern::binary(
                BinaryOperation::Add,
                Pattern::capture(&x),
                Pattern::literal(0),
            ),
            counting_rewrite(),
        )
        .with_name("x + 0 -> x"),
        RewriteRule::new(
            Pattern::binary(
                BinaryOperation::Multiply,
                Pattern::capture(&x),
                Pattern::literal(1),
            ),
            counting_rewrite(),
        )
        .with_name("x * 1 -> x"),
        RewriteRule::new(
            Pattern::unary(
                UnaryOperation::Negate,
                Pattern::unary(UnaryOperation::Negate, Pattern::capture(&x)),
            ),
            counting_rewrite(),
        )
        .with_name("-(-x) -> x"),
    ]
}

/// Return a pattern mirroring `expression`'s exact shape, capturing each
/// leaf with a fresh capture, and the leaves with their captures in leaf
/// order.
fn build_mirroring_pattern(expression: &Expression) -> (Pattern, Vec<(Capture, Expression)>) {
    let mut captures = Vec::new();
    let pattern = mirror_node(expression, &mut captures);
    (pattern, captures)
}

/// Return the mirror of `node`, recording its leaves in `captures`.
fn mirror_node(node: &Expression, captures: &mut Vec<(Capture, Expression)>) -> Pattern {
    match node.kind() {
        ExpressionKind::Unary(unary) => {
            Pattern::unary(unary.operation(), mirror_node(unary.operand(), captures))
        }
        ExpressionKind::Binary(binary) => {
            let left = mirror_node(binary.left(), captures);
            let right = mirror_node(binary.right(), captures);
            Pattern::binary(binary.operation(), left, right)
        }
        ExpressionKind::Logical(logical) => {
            let operands: Vec<Pattern> = logical
                .operands()
                .iter()
                .map(|operand| mirror_node(operand, captures))
                .collect();
            Pattern::logical(logical.operation(), operands)
        }
        ExpressionKind::Call(call) => {
            let arguments: Vec<Pattern> = call
                .arguments()
                .iter()
                .map(|argument| mirror_node(argument, captures))
                .collect();
            Pattern::call(call.callee().clone(), arguments)
        }
        ExpressionKind::Piecewise(piecewise) => {
            let cases: Vec<(Pattern, Pattern)> = piecewise
                .cases()
                .iter()
                .map(|(condition, value)| {
                    let condition = mirror_node(condition, captures);
                    (condition, mirror_node(value, captures))
                })
                .collect();
            let otherwise = mirror_node(piecewise.otherwise(), captures);
            Pattern::piecewise(cases, otherwise)
        }
        ExpressionKind::Identifier(_) | ExpressionKind::Literal(_) => {
            let capture = Capture::new(&format!("leaf_{}", captures.len()));
            captures.push((capture.clone(), node.clone()));
            Pattern::capture(&capture)
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
    Pattern::binary(operation, left, right)
}

/// The shape of a generated pattern, over a pool of captures picked by
/// index.
#[derive(Debug, Clone)]
enum PatternShape {
    Wildcard,
    AnyLiteral,
    AnyIdentifier,
    Capture(usize),
    CapturedAs(Box<PatternShape>, usize),
    Unary(Box<PatternShape>),
    Binary(Box<PatternShape>, Box<PatternShape>),
    Alternatives(Vec<PatternShape>),
}

const CAPTURE_POOL_SIZE: usize = 4;

/// Return a strategy for pattern shapes of alternatives and captures over
/// unary and binary nodes and leaves, up to four levels deep.
fn build_pattern_shape_strategy() -> impl Strategy<Value = PatternShape> {
    let leaf = prop_oneof![
        Just(PatternShape::Wildcard),
        Just(PatternShape::AnyLiteral),
        Just(PatternShape::AnyIdentifier),
        (0..CAPTURE_POOL_SIZE).prop_map(PatternShape::Capture),
    ];
    leaf.prop_recursive(4, 24, 3, |inner| {
        prop_oneof![
            (inner.clone(), 0..CAPTURE_POOL_SIZE)
                .prop_map(|(shape, index)| PatternShape::CapturedAs(Box::new(shape), index)),
            inner
                .clone()
                .prop_map(|operand| PatternShape::Unary(Box::new(operand))),
            (inner.clone(), inner.clone()).prop_map(|(left, right)| {
                PatternShape::Binary(Box::new(left), Box::new(right))
            }),
            prop::collection::vec(inner, 0..4).prop_map(PatternShape::Alternatives),
        ]
    })
}

/// Build the pattern `shape` describes over `pool`, recording in `used` the
/// captures it holds.
fn build_shaped_pattern(
    shape: &PatternShape,
    pool: &[Capture],
    used: &mut Vec<Capture>,
) -> Pattern {
    let mut record = |index: usize| {
        let capture = pool[index].clone();
        if !used.contains(&capture) {
            used.push(capture.clone());
        }
        capture
    };
    match shape {
        PatternShape::Wildcard => Pattern::wildcard(),
        PatternShape::AnyLiteral => Pattern::any_literal(),
        PatternShape::AnyIdentifier => Pattern::any_identifier(),
        PatternShape::Capture(index) => Pattern::capture(&record(*index)),
        PatternShape::CapturedAs(inner, index) => {
            let capture = record(*index);
            build_shaped_pattern(inner, pool, used).captured_as(&capture)
        }
        PatternShape::Unary(operand) => {
            Pattern::unary_any_operation(build_shaped_pattern(operand, pool, used))
        }
        PatternShape::Binary(left, right) => {
            let left = build_shaped_pattern(left, pool, used);
            Pattern::binary_any_operation(left, build_shaped_pattern(right, pool, used))
        }
        PatternShape::Alternatives(alternatives) => Pattern::alternatives(
            alternatives
                .iter()
                .map(|alternative| build_shaped_pattern(alternative, pool, used))
                .collect::<Vec<_>>(),
        ),
    }
}

/// Return whether `node` is a handle to `expression` or to one of its
/// subexpressions.
fn is_subexpression_of(node: &Expression, expression: &Expression) -> bool {
    Expression::ptr_eq(node, expression)
        || expression
            .children()
            .any(|child| is_subexpression_of(node, child))
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

proptest! {
    /// Test a pattern mirroring a tree matches it and binds each leaf
    /// capture to a handle to that leaf, in leaf order.
    #[test]
    fn mirroring_pattern_matches_and_binds_every_leaf(expression in build_expression_strategy(true)) {
        let (mirror, captures) = build_mirroring_pattern(&expression);

        let bindings = mirror.matches(&expression).expect("no predicate");

        let bindings = bindings.expect("the mirror matches");
        let bound: Vec<&Capture> = bindings.iter().map(|(capture, _)| capture).collect();
        let expected: Vec<&Capture> = captures.iter().map(|(capture, _)| capture).collect();
        prop_assert_eq!(bound, expected);
        for (capture, leaf) in &captures {
            let bound = &bindings[capture];
            prop_assert!(Expression::ptr_eq(bound, leaf), "{} is bound to {:?}", capture, bound);
        }
    }

    /// Test the wildcard matches every tree and binds nothing.
    #[test]
    fn wildcard_pattern_matches_every_expression(expression in build_expression_strategy(true)) {
        let bindings = Pattern::wildcard().matches(&expression).expect("no predicate");

        prop_assert_eq!(bindings, Some(MatchBindings::new()));
    }

    /// Test `is_match` agrees with `matches` for a mirror, the wildcard, and
    /// a literal outside the generated alphabet.
    #[test]
    fn is_match_agrees_with_matches(expression in build_expression_strategy(true)) {
        let (mirror, _) = build_mirroring_pattern(&expression);
        let outside_alphabet = Pattern::literal(
            LiteralValue::parse_text("99999.99999").expect("a decimal text"),
        );

        for pattern in [mirror, Pattern::wildcard(), outside_alphabet] {
            let answer = pattern.is_match(&expression).expect("no predicate");
            let bindings = pattern.matches(&expression).expect("no predicate");

            prop_assert_eq!(answer, bindings.is_some(), "for {:?}", pattern);
        }
    }

    #[test]
    fn mirror_pattern_with_another_root_operation_does_not_match(
        (root, alternate) in build_binary_root_strategy()
    ) {
        let altered = mirror_binary_root_with(&root, alternate);

        let bindings = altered.matches(&root).expect("no predicate");

        prop_assert!(bindings.is_none(), "matched with {:?}", bindings);
    }

    /// Test a binary pattern capturing both operands with one capture
    /// matches exactly when the operands are structurally equal, and binds
    /// the capture to a handle to the left operand.
    #[test]
    fn repeated_capture_matches_exactly_equal_operands(
        (left, right) in build_operand_pair_strategy(),
        operation in select(BINARY_OPERATIONS.to_vec()),
    ) {
        let x = Capture::new("x");
        let pattern = Pattern::binary(operation, Pattern::capture(&x), Pattern::capture(&x));
        let expression = Expression::new_binary(operation, &left, &right);

        let bindings = pattern.matches(&expression).expect("no predicate");

        prop_assert_eq!(bindings.is_some(), left == right);
        if let Some(bindings) = bindings {
            let bound = &bindings[&x];
            prop_assert!(Expression::ptr_eq(bound, &left), "x is bound to {:?}", bound);
            prop_assert_eq!(bindings.len(), 1);
        }
    }

    /// Test a literal pattern matches a literal exactly when a capture
    /// repeated over the two literals matches, and both follow
    /// `LiteralValue` equality.
    #[test]
    fn literal_pattern_agrees_with_capture_unification(
        value in build_literal_strategy(true),
        other in build_literal_strategy(true),
    ) {
        let x = Capture::new("x");
        let repeated = Pattern::binary_any_operation(Pattern::capture(&x), Pattern::capture(&x));
        let difference = Expression::new_binary(
            BinaryOperation::Subtract,
            Expression::from(value.clone()),
            Expression::from(other.clone()),
        );

        let by_literal = Pattern::literal(value.clone())
            .is_match(&Expression::from(other.clone()))
            .expect("no predicate");
        let by_capture = repeated.is_match(&difference).expect("no predicate");

        prop_assert_eq!(by_literal, value == other);
        prop_assert_eq!(by_capture, value == other);
    }

    /// Test a match binds only captures its pattern holds, each once and to
    /// a subexpression of the tree, and a match failing after its first
    /// operand bound captures leaves none of them behind.
    #[test]
    fn failed_matches_leave_no_bindings(
        shape in build_pattern_shape_strategy(),
        expression in build_expression_strategy(true),
    ) {
        let pool: Vec<Capture> = (0..CAPTURE_POOL_SIZE)
            .map(|index| Capture::new(&format!("c{index}")))
            .collect();
        let mut used = Vec::new();
        let pattern = build_shaped_pattern(&shape, &pool, &mut used);
        let fallback = Capture::new("fallback");
        let failing = Pattern::binary_any_operation(pattern.clone(), Pattern::nothing());
        let outer = Pattern::alternatives([failing, Pattern::capture(&fallback)]);
        let wrapped = Expression::new_binary(BinaryOperation::Add, &expression, 0);

        let bindings = pattern.matches(&expression).expect("no predicate");
        let outer_bindings = outer.matches(&wrapped).expect("no predicate");

        if let Some(bindings) = bindings {
            let bound: Vec<&Capture> = bindings.iter().map(|(capture, _)| capture).collect();
            for (index, (capture, node)) in bindings.iter().enumerate() {
                prop_assert!(used.contains(capture), "{} is not in the pattern", capture);
                prop_assert!(!bound[..index].contains(&capture), "{} is bound twice", capture);
                prop_assert!(is_subexpression_of(node, &expression), "{} is bound outside the tree", capture);
            }
        }
        let outer_bindings = outer_bindings.expect("the fallback matches");
        let outer_bound: Vec<&Capture> = outer_bindings.iter().map(|(capture, _)| capture).collect();
        prop_assert_eq!(outer_bound, vec![&fallback]);
    }

    #[test]
    fn apply_rewrite_rules_with_no_rules_is_the_identity(expression in build_expression_strategy(true)) {
        let outcome =
            apply_rewrite_rules(&expression, &[] as &[RewriteRule]).expect("no rule to fail");

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
        let shared = Expression::call(build_callee("f"), [wrapped.clone(), &wrapped + &wrapped]);
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

    /// Test a rule rewriting every literal to itself never fires and returns
    /// the input itself.
    #[test]
    fn identity_rewrite_never_fires_and_keeps_the_input(
        expression in build_expression_strategy(true)
    ) {
        let x = Capture::new("x");
        let rule = RewriteRule::new(Pattern::any_literal().captured_as(&x), rewrite_to_capture(&x));

        let outcome = apply_rewrite_rules(&expression, &[rule]).expect("no callback fails");

        prop_assert!(outcome.fired().is_empty(), "fired {:?}", outcome.fired());
        prop_assert!(!outcome.is_changed());
        prop_assert!(Expression::ptr_eq(outcome.output(), &expression));
    }
}
