//! Tests for the expression handle and its nodes: construction and getters,
//! children and rebuilding, free identifiers, substitution, structural
//! equality and hashing, handle identity, renaming equivalence, DAGs whose
//! subtrees are shared, and trees thousands of levels deep.
//!
//! Public API only (`fhy_core::symbolic::expression`).

#[path = "common/expression.rs"]
pub mod expression_support;
#[path = "common/hashing.rs"]
pub mod hashing_support;
#[path = "common/stack.rs"]
pub mod stack_support;

use std::collections::{HashMap, HashSet};

use expression_support::{
    build_call_node_or_panic, build_deep_sum, build_doubling_dag, build_identifier, build_literal,
    build_piecewise_node_or_panic, build_text_literal, is_doubling_dag_over,
};
use fhy_core::identifier::Identifier;
use fhy_core::symbolic::expression::{
    AlphaRenaming, BinaryExpression, BinaryOperation, CallExpression, Expression,
    ExpressionBuildError, ExpressionKind, PiecewiseExpression, UnaryExpression, UnaryOperation,
};
use hashing_support::hash_of;
use rstest::rstest;
use stack_support::{SMALL_STACK_DEPTH, run_on_small_stack};

/// Return the unary node `expression` refers to.
fn expect_unary(expression: &Expression) -> &UnaryExpression {
    let ExpressionKind::Unary(node) = expression.kind() else {
        panic!("expected a unary node, got {expression:?}");
    };
    node
}

/// Return the binary node `expression` refers to.
fn expect_binary(expression: &Expression) -> &BinaryExpression {
    let ExpressionKind::Binary(node) = expression.kind() else {
        panic!("expected a binary node, got {expression:?}");
    };
    node
}

/// Return the piecewise node `expression` refers to.
fn expect_piecewise(expression: &Expression) -> &PiecewiseExpression {
    let ExpressionKind::Piecewise(node) = expression.kind() else {
        panic!("expected a piecewise node, got {expression:?}");
    };
    node
}

/// Assert `actual` holds handles to exactly the nodes of `expected`, in order.
fn assert_same_nodes(actual: &[&Expression], expected: &[&Expression]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (index, (actual_node, expected_node)) in actual.iter().zip(expected).enumerate() {
        assert!(
            Expression::ptr_eq(actual_node, expected_node),
            "node {index}: {actual_node:?} is not the node {expected_node:?}"
        );
    }
}

/// Return the set of `identifiers`.
fn collect_identifiers<const N: usize>(identifiers: [&Identifier; N]) -> HashSet<Identifier> {
    identifiers.into_iter().cloned().collect()
}

// =============================================================================
// Construction and getters
// =============================================================================

/// Test a unary node exposes the operation and the operand node it was built
/// with.
#[test]
fn unary_expression_exposes_operation_and_operand() {
    let operand = build_literal(5);

    let expression = Expression::new_unary(UnaryOperation::Negate, &operand);

    let node = expect_unary(&expression);
    assert_eq!(node.operation(), UnaryOperation::Negate);
    assert!(Expression::ptr_eq(node.operand(), &operand));
}

/// Test a binary node exposes the operation and both operand nodes.
#[test]
fn binary_expression_exposes_operation_left_and_right() {
    let left = build_literal(5);
    let right = build_literal(10);

    let expression = Expression::new_binary(BinaryOperation::Add, &left, &right);

    let node = expect_binary(&expression);
    assert_eq!(node.operation(), BinaryOperation::Add);
    assert!(Expression::ptr_eq(node.left(), &left));
    assert!(Expression::ptr_eq(node.right(), &right));
}

/// Test an identifier reference exposes the identifier it was built with.
#[test]
fn expression_from_identifier_refers_to_the_identifier() {
    let identifier = Identifier::new("x");

    let expression = Expression::from(identifier.clone());

    let ExpressionKind::Identifier(held) = expression.kind() else {
        panic!("expected an identifier reference, got {expression:?}");
    };
    assert_eq!(held, &identifier);
    assert_eq!(held.name_hint(), "x");
}

/// Test a piecewise exposes its single case and its otherwise branch.
#[test]
fn piecewise_expression_exposes_cases_and_otherwise() {
    let condition = build_literal(true);
    let value = build_literal(1);
    let otherwise = build_literal(2);

    let node =
        PiecewiseExpression::try_new(vec![(condition.clone(), value.clone())], otherwise.clone())
            .expect("a valid piecewise");

    assert_eq!(node.cases().len(), 1);
    assert!(Expression::ptr_eq(&node.cases()[0].0, &condition));
    assert!(Expression::ptr_eq(&node.cases()[0].1, &value));
    assert!(Expression::ptr_eq(node.otherwise(), &otherwise));
}

/// Test a piecewise keeps its cases in the order given.
#[test]
fn piecewise_expression_keeps_cases_in_declared_order() {
    let (first_condition, second_condition) = (build_literal(true), build_literal(false));
    let (first_value, second_value) = (build_literal(1), build_literal(2));

    let node = PiecewiseExpression::try_new(
        vec![
            (first_condition.clone(), first_value.clone()),
            (second_condition.clone(), second_value.clone()),
        ],
        build_literal(0),
    )
    .expect("a valid piecewise");

    let flattened: Vec<&Expression> = node
        .cases()
        .iter()
        .flat_map(|(condition, value)| [condition, value])
        .collect();
    assert_same_nodes(
        &flattened,
        &[
            &first_condition,
            &first_value,
            &second_condition,
            &second_value,
        ],
    );
}

/// Test a piecewise with no cases is refused.
#[test]
fn piecewise_expression_try_new_rejects_zero_cases() {
    let result = PiecewiseExpression::try_new(Vec::new(), build_literal(0));

    assert!(
        matches!(result, Err(ExpressionBuildError::EmptyPiecewise)),
        "got {result:?}"
    );
}

/// Test a Boolean literal is accepted as a case condition.
#[rstest]
#[case::true_value(true)]
#[case::false_value(false)]
fn piecewise_expression_try_new_accepts_boolean_literal_condition(#[case] value: bool) {
    let condition = build_literal(value);

    let node = PiecewiseExpression::try_new(
        vec![(condition.clone(), build_literal(1))],
        build_literal(0),
    )
    .expect("a Boolean literal condition is valid");

    assert!(Expression::ptr_eq(&node.cases()[0].0, &condition));
}

/// Test a literal condition other than a Boolean is refused, naming its case.
#[rstest]
#[case::int_one(build_literal(1))]
#[case::int_zero(build_literal(0))]
#[case::float(build_literal(5.0))]
#[case::integer_text(build_text_literal("5"))]
#[case::decimal_text(build_text_literal("1.5"))]
fn piecewise_expression_try_new_rejects_non_boolean_literal_condition(
    #[case] condition: Expression,
) {
    let result =
        PiecewiseExpression::try_new(vec![(condition, build_literal(1))], build_literal(0));

    assert!(
        matches!(
            result,
            Err(ExpressionBuildError::NonBooleanConditionLiteral { case_index: 0 })
        ),
        "got {result:?}"
    );
}

/// Test the refusal names the first offending case, not the first case.
#[test]
fn piecewise_expression_try_new_names_the_first_non_boolean_condition() {
    let cases = vec![
        (build_literal(true), build_literal(1)),
        (build_literal(false), build_literal(2)),
        (build_literal(3), build_literal(3)),
        (build_literal(4), build_literal(4)),
    ];

    let result = PiecewiseExpression::try_new(cases, build_literal(0));

    assert!(
        matches!(
            result,
            Err(ExpressionBuildError::NonBooleanConditionLiteral { case_index: 2 })
        ),
        "got {result:?}"
    );
}

/// Test a condition that is not a literal is never refused, whatever it is.
#[test]
fn piecewise_expression_try_new_accepts_non_literal_condition() {
    let (_, flag) = build_identifier("flag");
    let arithmetic = &flag + 1;

    let node = PiecewiseExpression::try_new(
        vec![
            (flag.clone(), build_literal(1)),
            (arithmetic, build_literal(2)),
        ],
        build_literal(0),
    )
    .expect("non-literal conditions are valid");

    assert!(Expression::ptr_eq(&node.cases()[0].0, &flag));
}

/// Test a call exposes its function name and argument nodes in order.
#[test]
fn call_expression_exposes_name_and_arguments() {
    let (first, second) = (build_literal(1), build_literal(2));

    let node =
        CallExpression::try_new("max", vec![first.clone(), second.clone()]).expect("a valid call");

    assert_eq!(node.function_name(), "max");
    assert_same_nodes(
        &node.arguments().iter().collect::<Vec<_>>(),
        &[&first, &second],
    );
}

/// Test a call may take no arguments.
#[test]
fn call_expression_accepts_zero_arguments() {
    let node = CallExpression::try_new("nullary", Vec::new()).expect("a valid call");

    assert_eq!(node.function_name(), "nullary");
    assert!(node.arguments().is_empty());
}

/// Test a call with an empty function name is refused.
#[test]
fn call_expression_try_new_rejects_empty_function_name() {
    let result = CallExpression::try_new("", vec![build_literal(1)]);

    assert!(
        matches!(result, Err(ExpressionBuildError::EmptyFunctionName)),
        "got {result:?}"
    );
}

/// Test wrapping a copy of each node struct back into an expression yields
/// an equal tree.
#[test]
fn expression_from_node_struct_rewraps_the_node() {
    let (_, x) = build_identifier("x");
    let unary = -&x;
    let binary = &x % 3;
    let piecewise =
        build_piecewise_node_or_panic(vec![(x.less(0), build_literal(1))], build_literal(2));
    let call = build_call_node_or_panic("f", vec![x.clone()]);

    let rewrapped_unary = Expression::from(expect_unary(&unary).clone());
    let rewrapped_binary = Expression::from(expect_binary(&binary).clone());
    let rewrapped_piecewise = Expression::from(expect_piecewise(&piecewise).clone());
    let ExpressionKind::Call(call_node) = call.kind() else {
        panic!("expected a call node, got {call:?}");
    };
    let rewrapped_call = Expression::from(call_node.clone());

    assert_eq!(rewrapped_unary, unary);
    assert_eq!(rewrapped_binary, binary);
    assert_eq!(rewrapped_piecewise, piecewise);
    assert_eq!(rewrapped_call, call);
}

// =============================================================================
// Children and rebuilding
// =============================================================================

/// Test a unary node's children are its operand.
#[test]
fn expression_children_of_unary_is_the_operand() {
    let operand = build_literal(1);
    let expression = Expression::new_unary(UnaryOperation::Negate, &operand);

    let children: Vec<&Expression> = expression.children().collect();

    assert_same_nodes(&children, &[&operand]);
}

/// Test a binary node's children are left then right.
#[test]
fn expression_children_of_binary_are_left_then_right() {
    let (left, right) = (build_literal(1), build_literal(2));
    let expression = Expression::new_binary(BinaryOperation::Add, &left, &right);

    let children: Vec<&Expression> = expression.children().collect();

    assert_same_nodes(&children, &[&left, &right]);
}

/// Test a piecewise's children interleave conditions and values, then the
/// otherwise branch.
#[test]
fn expression_children_of_piecewise_interleave_cases_then_otherwise() {
    let (first_condition, second_condition) = (build_literal(true), build_literal(false));
    let (first_value, second_value) = (build_literal(1), build_literal(2));
    let otherwise = build_literal(0);
    let expression = build_piecewise_node_or_panic(
        vec![
            (first_condition.clone(), first_value.clone()),
            (second_condition.clone(), second_value.clone()),
        ],
        otherwise.clone(),
    );

    let children: Vec<&Expression> = expression.children().collect();

    assert_same_nodes(
        &children,
        &[
            &first_condition,
            &first_value,
            &second_condition,
            &second_value,
            &otherwise,
        ],
    );
}

/// Test a one-case piecewise's children are its condition, value, and
/// otherwise branch.
#[test]
fn expression_children_of_single_case_piecewise_are_condition_value_otherwise() {
    let (condition, value, otherwise) = (build_literal(true), build_literal(1), build_literal(0));
    let expression =
        build_piecewise_node_or_panic(vec![(condition.clone(), value.clone())], otherwise.clone());

    let children: Vec<&Expression> = expression.children().collect();

    assert_same_nodes(&children, &[&condition, &value, &otherwise]);
}

/// Test a call's children are its arguments in order.
#[test]
fn expression_children_of_call_are_the_arguments() {
    let arguments = [build_literal(1), build_literal(2), build_literal(3)];
    let expression = build_call_node_or_panic("select3", arguments.to_vec());

    let children: Vec<&Expression> = expression.children().collect();

    let expected: Vec<&Expression> = arguments.iter().collect();
    assert_same_nodes(&children, &expected);
}

/// Test an identifier reference and a literal have no children.
#[rstest]
#[case::identifier(build_identifier("x").1)]
#[case::literal(build_literal(7))]
fn expression_children_of_leaf_are_empty(#[case] leaf: Expression) {
    let children: Vec<&Expression> = leaf.children().collect();

    assert!(children.is_empty(), "{children:?}");
}

/// Test rebuilding a one-case piecewise from its own children reproduces it.
#[test]
fn expression_rebuild_with_children_round_trips_single_case_piecewise() {
    let expression = build_piecewise_node_or_panic(
        vec![(build_literal(true), build_literal(1))],
        build_literal(0),
    );
    let children: Vec<Expression> = expression.children().cloned().collect();

    let rebuilt = expression
        .rebuild_with_children(children)
        .expect("the node's own children rebuild it");

    assert_eq!(rebuilt, expression);
}

/// Test rebuilding a three-case piecewise from its own children reproduces
/// it with the same child nodes.
#[test]
fn expression_rebuild_with_children_round_trips_multiple_case_piecewise() {
    let conditions = [
        build_literal(true),
        build_literal(false),
        build_literal(true),
    ];
    let values = [build_literal(1), build_literal(2), build_literal(3)];
    let otherwise = build_literal(0);
    let expression = build_piecewise_node_or_panic(
        conditions
            .iter()
            .cloned()
            .zip(values.iter().cloned())
            .collect(),
        otherwise.clone(),
    );
    let children: Vec<Expression> = expression.children().cloned().collect();

    let rebuilt = expression
        .rebuild_with_children(children)
        .expect("the node's own children rebuild it");

    assert_eq!(rebuilt, expression);
    let node = expect_piecewise(&rebuilt);
    let parts: Vec<&Expression> = node
        .cases()
        .iter()
        .flat_map(|(condition, value)| [condition, value])
        .chain([node.otherwise()])
        .collect();
    assert_same_nodes(
        &parts,
        &[
            &conditions[0],
            &values[0],
            &conditions[1],
            &values[1],
            &conditions[2],
            &values[2],
            &otherwise,
        ],
    );
}

/// Test rebuilding a one-case piecewise from any other number of children is
/// refused.
#[rstest]
#[case::zero(0)]
#[case::one(1)]
#[case::two(2)]
#[case::four(4)]
#[case::five(5)]
#[case::six(6)]
fn expression_rebuild_with_children_rejects_piecewise_child_count(#[case] child_count: usize) {
    let expression = build_piecewise_node_or_panic(
        vec![(build_literal(true), build_literal(1))],
        build_literal(0),
    );
    let children: Vec<Expression> = (0..child_count).map(|_| build_literal(true)).collect();

    let result = expression.rebuild_with_children(children);

    assert_eq!(
        result,
        Err(ExpressionBuildError::ChildCountMismatch {
            expected: 3,
            actual: child_count
        })
    );
}

/// Test rebuilding a piecewise with a non-Boolean literal condition is
/// refused.
#[test]
fn expression_rebuild_with_children_rejects_numeric_piecewise_condition() {
    let expression = build_piecewise_node_or_panic(
        vec![(build_literal(true), build_literal(1))],
        build_literal(0),
    );

    let result = expression.rebuild_with_children(vec![
        build_literal(7),
        build_literal(1),
        build_literal(0),
    ]);

    assert_eq!(
        result,
        Err(ExpressionBuildError::NonBooleanConditionLiteral { case_index: 0 })
    );
}

/// Test rebuilding each non-leaf node from new children keeps its kind and
/// operation and takes the new children.
#[test]
fn expression_rebuild_with_children_keeps_kind_and_operation() {
    let (_, x) = build_identifier("x");
    let (_, y) = build_identifier("y");
    let unary = Expression::new_unary(UnaryOperation::LogicalNot, &x);
    let binary = Expression::new_binary(BinaryOperation::FloorDivide, &x, 2);
    let call = build_call_node_or_panic("f", vec![x.clone(), build_literal(1)]);

    let rebuilt_unary = unary
        .rebuild_with_children(vec![y.clone()])
        .expect("one child");
    let rebuilt_binary = binary
        .rebuild_with_children(vec![y.clone(), build_literal(3)])
        .expect("two children");
    let rebuilt_call = call
        .rebuild_with_children(vec![y.clone(), build_literal(4)])
        .expect("two children");

    assert_eq!(
        rebuilt_unary,
        Expression::new_unary(UnaryOperation::LogicalNot, &y)
    );
    assert_eq!(
        rebuilt_binary,
        Expression::new_binary(BinaryOperation::FloorDivide, &y, 3)
    );
    assert_eq!(
        rebuilt_call,
        build_call_node_or_panic("f", vec![y.clone(), build_literal(4)])
    );
}

/// Test rebuilding a unary, binary, or call node from a different number of
/// children is refused.
#[rstest]
#[case::unary_none(Expression::new_unary(UnaryOperation::Negate, 1), 0, 1)]
#[case::unary_two(Expression::new_unary(UnaryOperation::Negate, 1), 2, 1)]
#[case::binary_one(Expression::new_binary(BinaryOperation::Add, 1, 2), 1, 2)]
#[case::binary_three(Expression::new_binary(BinaryOperation::Add, 1, 2), 3, 2)]
#[case::call_fewer(build_call_node_or_panic("f", vec![build_literal(1), build_literal(2)]), 1, 2)]
#[case::call_more(build_call_node_or_panic("f", vec![build_literal(1), build_literal(2)]), 3, 2)]
#[case::call_none_to_one(build_call_node_or_panic("f", Vec::new()), 1, 0)]
fn expression_rebuild_with_children_rejects_a_different_child_count(
    #[case] expression: Expression,
    #[case] child_count: usize,
    #[case] expected: usize,
) {
    let children: Vec<Expression> = (0..child_count).map(|_| build_literal(9)).collect();

    let result = expression.rebuild_with_children(children);

    assert_eq!(
        result,
        Err(ExpressionBuildError::ChildCountMismatch {
            expected,
            actual: child_count
        })
    );
}

/// Test rebuilding a leaf from no children returns a handle to the leaf.
#[rstest]
#[case::identifier(build_identifier("x").1)]
#[case::literal(build_literal(7))]
fn expression_rebuild_with_children_of_leaf_returns_the_leaf(#[case] leaf: Expression) {
    let rebuilt = leaf.rebuild_with_children(Vec::new()).expect("no children");

    assert!(Expression::ptr_eq(&rebuilt, &leaf));
}

/// Test rebuilding a leaf from any children is refused.
#[rstest]
#[case::identifier(build_identifier("x").1)]
#[case::literal(build_literal(7))]
fn expression_rebuild_with_children_of_leaf_rejects_children(#[case] leaf: Expression) {
    let result = leaf.rebuild_with_children(vec![build_literal(1)]);

    assert_eq!(
        result,
        Err(ExpressionBuildError::ChildCountMismatch {
            expected: 0,
            actual: 1
        })
    );
}

// =============================================================================
// Free identifiers
// =============================================================================

/// Test an identifier reference reports its identifier as free.
#[test]
fn expression_free_identifiers_of_identifier_is_the_identifier() {
    let (x, reference) = build_identifier("x");

    let free = reference.free_identifiers();

    assert_eq!(free, collect_identifiers([&x]));
}

/// Test a literal has no free identifiers.
#[test]
fn expression_free_identifiers_of_literal_is_empty() {
    let free = build_literal(7).free_identifiers();

    assert!(free.is_empty(), "{free:?}");
}

/// Test a composite's free identifiers are the union over its children.
#[test]
fn expression_free_identifiers_of_composite_is_the_union_over_children() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let expression = &x_reference + &(5 / &y_reference);

    let free = expression.free_identifiers();

    assert_eq!(free, collect_identifiers([&x, &y]));
}

/// Test free identifiers are found under a unary operation.
#[test]
fn expression_free_identifiers_walk_into_a_unary_operand() {
    let (x, reference) = build_identifier("x");

    let free = (-reference).free_identifiers();

    assert_eq!(free, collect_identifiers([&x]));
}

/// Test a call's free identifiers are the union over its arguments.
#[test]
fn expression_free_identifiers_of_call_is_the_union_over_arguments() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let expression = build_call_node_or_panic("f", vec![x_reference, y_reference]);

    let free = expression.free_identifiers();

    assert_eq!(free, collect_identifiers([&x, &y]));
}

/// Test a piecewise's free identifiers cover conditions, values, and the
/// otherwise branch, and an identifier seen twice is reported once.
#[test]
fn expression_free_identifiers_of_piecewise_cover_every_branch() {
    let (c, c_reference) = build_identifier("c");
    let (v, v_reference) = build_identifier("v");
    let (o, o_reference) = build_identifier("o");
    let expression = build_piecewise_node_or_panic(
        vec![
            (c_reference.clone(), v_reference),
            (c_reference.logical_not(), build_literal(1)),
        ],
        o_reference,
    );

    let free = expression.free_identifiers();

    assert_eq!(free, collect_identifiers([&c, &v, &o]));
}

// =============================================================================
// Substitution
// =============================================================================

/// Test substituting a mapped identifier yields the replacement node itself.
#[test]
fn expression_substitute_replaces_identifier_with_the_replacement() {
    let (x, reference) = build_identifier("x");
    let replacement = build_literal(5);

    let result = reference
        .substitute(&HashMap::from([(x, replacement.clone())]))
        .expect("no piecewise to refuse");

    assert!(Expression::ptr_eq(&result, &replacement));
}

/// Test an unmapped identifier is returned as a handle to itself.
#[test]
fn expression_substitute_leaves_unmapped_identifier_untouched() {
    let (x, _) = build_identifier("x");
    let (_, y_reference) = build_identifier("y");

    let result = y_reference
        .substitute(&HashMap::from([(x, build_literal(5))]))
        .expect("no piecewise to refuse");

    assert!(Expression::ptr_eq(&result, &y_reference));
}

/// Test substituting into a literal returns a handle to the literal.
#[test]
fn expression_substitute_returns_a_literal_unchanged() {
    let literal = build_literal(3);

    let result = literal
        .substitute(&HashMap::new())
        .expect("nothing to refuse");

    assert!(Expression::ptr_eq(&result, &literal));
}

/// Test substitution rewrites mapped identifiers throughout a composite.
#[test]
fn expression_substitute_recurses_into_a_composite() {
    let (x, x_reference) = build_identifier("x");
    let (_, y_reference) = build_identifier("y");
    let expression = &x_reference + &y_reference;

    let result = expression
        .substitute(&HashMap::from([(x, build_literal(1))]))
        .expect("no piecewise to refuse");

    assert_eq!(
        result,
        Expression::new_binary(BinaryOperation::Add, 1, &y_reference)
    );
}

/// Test substitution rewrites a unary operand.
#[test]
fn expression_substitute_recurses_into_a_unary_operand() {
    let (x, reference) = build_identifier("x");

    let result = (-reference)
        .substitute(&HashMap::from([(x, build_literal(7))]))
        .expect("no piecewise to refuse");

    assert_eq!(result, Expression::new_unary(UnaryOperation::Negate, 7));
}

/// Test a leaf nothing replaces keeps its node through substitution.
#[test]
fn expression_substitute_preserves_unchanged_nested_leaves() {
    let literal = build_literal(42);
    let tree = &literal + &literal;

    let result = tree.substitute(&HashMap::new()).expect("nothing to refuse");

    let node = expect_binary(&result);
    assert!(Expression::ptr_eq(node.left(), &literal));
    assert!(Expression::ptr_eq(node.right(), &literal));
}

/// Test substituting an identifier reference for an identifier renames it.
#[test]
fn expression_substitute_renames_an_identifier() {
    let (x, x_reference) = build_identifier("x");
    let (_, y_reference) = build_identifier("y");
    let expression = &x_reference + 5;

    let result = expression
        .substitute(&HashMap::from([(x, y_reference.clone())]))
        .expect("no piecewise to refuse");

    assert_eq!(result, &y_reference + 5);
}

/// Test every occurrence of a mapped identifier becomes the same
/// replacement node.
#[test]
fn expression_substitute_shares_the_replacement_at_every_occurrence() {
    let (x, x_reference) = build_identifier("x");
    let expression = &x_reference * &x_reference;
    let replacement = build_literal(3) + 4;

    let result = expression
        .substitute(&HashMap::from([(x, replacement.clone())]))
        .expect("no piecewise to refuse");

    let node = expect_binary(&result);
    assert!(Expression::ptr_eq(node.left(), &replacement));
    assert!(Expression::ptr_eq(node.right(), &replacement));
}

/// Test substitution is simultaneous: a replacement is not substituted into
/// in turn.
#[test]
fn expression_substitute_does_not_chain_replacements() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let expression = &x_reference + &y_reference;

    let result = expression
        .substitute(&HashMap::from([
            (x, y_reference.clone()),
            (y, build_literal(1)),
        ]))
        .expect("no piecewise to refuse");

    assert_eq!(result, &y_reference + 1);
}

/// Test substitution refuses to put a numeric literal in a piecewise case
/// condition.
#[test]
fn expression_substitute_refuses_a_number_in_a_piecewise_condition() {
    let (c, condition) = build_identifier("c");
    let expression =
        build_piecewise_node_or_panic(vec![(condition, build_literal(1))], build_literal(0));

    let result = expression.substitute(&HashMap::from([(c, build_literal(1))]));

    assert_eq!(
        result,
        Err(ExpressionBuildError::NonBooleanConditionLiteral { case_index: 0 })
    );
}

/// Test substituting with an empty map yields an equal tree.
#[test]
fn expression_substitute_with_empty_map_yields_an_equal_tree() {
    let (_, x) = build_identifier("x");
    let expression = build_piecewise_node_or_panic(
        vec![(x.less(3), build_call_node_or_panic("f", vec![-&x]))],
        x.power(2),
    );

    let result = expression
        .substitute(&HashMap::new())
        .expect("nothing to refuse");

    assert_eq!(result, expression);
}

// =============================================================================
// Structural equality, hashing, and handle identity
// =============================================================================

/// Test separately built trees of the same shape are equal but are not the
/// same node.
#[rstest]
#[case::literal(|| build_literal(42))]
#[case::unary(|| Expression::new_unary(UnaryOperation::Negate, 1))]
#[case::binary(|| Expression::new_binary(BinaryOperation::Add, 1, 2))]
#[case::piecewise(|| build_piecewise_node_or_panic(vec![(build_literal(true), build_literal(1))], build_literal(0)))]
#[case::call(|| build_call_node_or_panic("max", vec![build_literal(1), build_literal(2)]))]
fn expression_separately_built_equal_trees_are_equal_but_distinct(
    #[case] build: fn() -> Expression,
) {
    let first = build();
    let second = build();

    assert_eq!(first, second);
    assert_eq!(second, first);
    assert!(!Expression::ptr_eq(&first, &second));
}

/// Test two references to one identifier are equal but distinct nodes.
#[test]
fn expression_references_to_one_identifier_are_equal_but_distinct() {
    let identifier = Identifier::new("shared");

    let first = Expression::from(identifier.clone());
    let second = Expression::from(identifier);

    assert_eq!(first, second);
    assert!(!Expression::ptr_eq(&first, &second));
}

/// Test a clone is the same node.
#[test]
fn expression_clone_is_the_same_node() {
    let expression = build_literal(1) + 2;

    let copy = expression.clone();

    assert!(Expression::ptr_eq(&copy, &expression));
}

/// Test references to distinct identifiers with the same name are unequal.
#[test]
fn expression_references_to_distinct_identifiers_are_unequal() {
    let (_, first) = build_identifier("x");
    let (_, second) = build_identifier("x");

    assert_ne!(first, second);
}

/// Test a set of equal trees built separately keeps one member.
#[rstest]
#[case::literal(|| build_literal(42))]
#[case::unary(|| Expression::new_unary(UnaryOperation::Negate, 1))]
#[case::binary(|| Expression::new_binary(BinaryOperation::Add, 1, 2))]
#[case::piecewise(|| build_piecewise_node_or_panic(vec![(build_literal(true), build_literal(1))], build_literal(0)))]
#[case::call(|| build_call_node_or_panic("max", vec![build_literal(1), build_literal(2)]))]
fn expression_set_of_equal_trees_keeps_one_member(#[case] build: fn() -> Expression) {
    let first = build();
    let second = build();

    let set: HashSet<Expression> = [first, second].into_iter().collect();

    assert_eq!(set.len(), 1);
}

/// Test equal trees hash equally, literals compared by equivalence bucket.
#[test]
fn expression_equal_trees_hash_equally() {
    let (_, x) = build_identifier("x");
    let first = &x + 5;
    let second = &x + build_text_literal("05");
    assert_eq!(first, second);

    assert_eq!(hash_of(&first), hash_of(&second));
}

/// Test literal expressions compare by literal equivalence.
#[rstest]
#[case::integer_and_text(build_literal(5), build_text_literal("05"), true)]
#[case::nan_payloads(
    build_literal(f64::NAN),
    build_literal(f64::from_bits(0xFFF8_0000_0000_0001)),
    true
)]
#[case::zeros(build_literal(0.0), build_literal(-0.0), true)]
#[case::integer_and_float(build_literal(1), build_literal(1.0), false)]
#[case::integer_and_bool(build_literal(1), build_literal(true), false)]
#[case::decimal_and_float(build_text_literal("1.5"), build_literal(1.5), false)]
fn expression_literal_equality_follows_literal_equivalence(
    #[case] left: Expression,
    #[case] right: Expression,
    #[case] expected: bool,
) {
    assert_eq!(left == right, expected);
    assert_eq!(right == left, expected);
}

/// Test reordering piecewise cases breaks equality.
#[test]
fn expression_reordered_piecewise_cases_are_unequal() {
    let first_case = (build_literal(true), build_literal(1));
    let second_case = (build_literal(false), build_literal(2));
    let forward = build_piecewise_node_or_panic(
        vec![first_case.clone(), second_case.clone()],
        build_literal(0),
    );
    let reversed = build_piecewise_node_or_panic(vec![second_case, first_case], build_literal(0));

    assert_ne!(forward, reversed);
    assert_ne!(reversed, forward);
}

/// Test trees differing in kind, operation, child, name, arity, identifier,
/// or literal value are unequal both ways and hash differently.
#[rstest]
#[case::literal_and_identifier(build_literal(1), build_identifier("x").1)]
#[case::operation(
    Expression::new_binary(BinaryOperation::Add, 1, 2),
    Expression::new_binary(BinaryOperation::Subtract, 1, 2)
)]
#[case::unary_operation(
    Expression::new_unary(UnaryOperation::Negate, 1),
    Expression::new_unary(UnaryOperation::Positive, 1)
)]
#[case::operand_order(
    Expression::new_binary(BinaryOperation::Add, 1, 2),
    Expression::new_binary(BinaryOperation::Add, 2, 1)
)]
#[case::function_name(build_call_node_or_panic("f", vec![build_literal(1)]), build_call_node_or_panic("g", vec![build_literal(1)]))]
#[case::arity(build_call_node_or_panic("f", vec![build_literal(1)]), build_call_node_or_panic("f", vec![build_literal(1), build_literal(1)]))]
#[case::otherwise(build_piecewise_node_or_panic(vec![(build_literal(true), build_literal(1))], build_literal(0)), build_piecewise_node_or_panic(vec![(build_literal(true), build_literal(1))], build_literal(2)))]
#[case::unary_and_binary(
    Expression::new_unary(UnaryOperation::Negate, 1),
    Expression::new_binary(BinaryOperation::Subtract, 0, 1)
)]
#[case::identifier(build_identifier("x").1, build_identifier("x").1)]
#[case::integer_literal(build_literal(1), build_literal(2))]
#[case::float_literal(build_literal(1.5), build_literal(2.5))]
#[case::bool_literal(build_literal(true), build_literal(false))]
#[case::decimal_text_literal(build_text_literal("1.5"), build_text_literal("2.5"))]
fn expression_trees_differing_anywhere_are_unequal_and_hash_differently(
    #[case] left: Expression,
    #[case] right: Expression,
) {
    assert_ne!(left, right);
    assert_ne!(right, left);
    assert_ne!(hash_of(&left), hash_of(&right));
}

/// Test piecewise nodes with different case counts are unequal both ways and
/// hash differently when the shorter one's children are a prefix of the
/// longer one's: `{y if p; q otherwise}` against `{y if p; z if q; w
/// otherwise}`.
#[test]
fn expression_piecewise_nodes_with_different_case_counts_are_unequal() {
    let [
        first_condition,
        second_condition,
        first_value,
        second_value,
        otherwise,
    ] = ["p", "q", "y", "z", "w"].map(|name| build_identifier(name).1);
    let one_case = build_piecewise_node_or_panic(
        vec![(first_condition.clone(), first_value.clone())],
        second_condition.clone(),
    );
    let two_cases = build_piecewise_node_or_panic(
        vec![
            (first_condition, first_value),
            (second_condition, second_value),
        ],
        otherwise,
    );

    assert_ne!(one_case, two_cases);
    assert_ne!(two_cases, one_case);
    assert_ne!(hash_of(&one_case), hash_of(&two_cases));
}

// =============================================================================
// Equivalence under a free-identifier renaming
// =============================================================================

/// Return the renaming `pairs` describe, failing the test if it is refused.
fn build_renaming<const N: usize>(pairs: [(Identifier, Identifier); N]) -> AlphaRenaming {
    AlphaRenaming::try_new(HashMap::from(pairs)).expect("the renaming is injective")
}

/// Test a tree is equivalent to its renamed copy under the renaming.
#[test]
fn expression_is_alpha_equivalent_under_the_renaming_it_was_renamed_by() {
    let (x, x_reference) = build_identifier("x");
    let (w, w_reference) = build_identifier("w");
    let expression = &x_reference * 2 + &x_reference;
    let renamed = &w_reference * 2 + &w_reference;

    let equivalent = expression.is_alpha_equivalent_under(&renamed, &build_renaming([(x, w)]));

    assert!(equivalent);
}

/// Test a renaming that swaps two identifiers relates a tree to its swapped
/// copy.
#[test]
fn expression_is_alpha_equivalent_under_a_swap() {
    let (a, a_reference) = build_identifier("a");
    let (b, b_reference) = build_identifier("b");
    let renaming = build_renaming([(a.clone(), b.clone()), (b, a)]);

    let equivalent = (&a_reference - &b_reference)
        .is_alpha_equivalent_under(&(&b_reference - &a_reference), &renaming);

    assert!(equivalent);
}

/// Test the empty renaming makes the check structural equality.
#[test]
fn expression_is_alpha_equivalent_under_empty_renaming_is_structural_equality() {
    let (_, x) = build_identifier("x");
    let (_, w) = build_identifier("w");
    let renaming = AlphaRenaming::default();

    assert!((&x + 1).is_alpha_equivalent_under(&(&x + 1), &renaming));
    assert!(!(&x + 1).is_alpha_equivalent_under(&(&w + 1), &renaming));
}

/// Test a mapped identifier must be replaced by its image, not by itself or
/// another identifier.
#[test]
fn expression_is_alpha_equivalent_under_requires_the_image() {
    let (x, x_reference) = build_identifier("x");
    let (w, _) = build_identifier("w");
    let (_, v_reference) = build_identifier("v");
    let renaming = build_renaming([(x, w)]);

    assert!(!x_reference.is_alpha_equivalent_under(&x_reference, &renaming));
    assert!(!x_reference.is_alpha_equivalent_under(&v_reference, &renaming));
}

/// Test an unmapped identifier may not stand for an image of the renaming.
#[test]
fn expression_is_alpha_equivalent_under_rejects_an_unmapped_identifier_matching_an_image() {
    let (x, _) = build_identifier("x");
    let (w, w_reference) = build_identifier("w");
    let renaming = build_renaming([(x, w)]);

    let equivalent = w_reference.is_alpha_equivalent_under(&w_reference, &renaming);

    assert!(!equivalent);
}

/// Test the renaming check still compares structure.
#[test]
fn expression_is_alpha_equivalent_under_rejects_a_different_structure() {
    let (x, x_reference) = build_identifier("x");
    let (w, w_reference) = build_identifier("w");

    let equivalent = (&x_reference + 1)
        .is_alpha_equivalent_under(&(&w_reference - 1), &build_renaming([(x, w)]));

    assert!(!equivalent);
}

/// Test a renaming sending two identifiers to one image is refused, which
/// keeps `a + b` from being equivalent to `c + c`, whichever colliding pair
/// comes first.
#[rstest]
#[case::a_first(false)]
#[case::b_first(true)]
fn alpha_renaming_try_new_refuses_a_non_injective_map(#[case] is_b_first: bool) {
    let (a, _) = build_identifier("a");
    let (b, _) = build_identifier("b");
    let (c, _) = build_identifier("c");
    let mut pairs = vec![(a, c.clone()), (b, c.clone())];
    if is_b_first {
        pairs.reverse();
    }

    let error = AlphaRenaming::try_new(pairs.into_iter().collect::<HashMap<_, _>>())
        .expect_err("two identifiers share the image c");

    assert_eq!(error.image(), &c);
    assert_eq!(
        error.to_string(),
        format!(
            "a free-identifier renaming must be injective, but more than one identifier maps \
             to c::{}",
            c.id()
        )
    );
}

/// Test the refusal comes from the renaming, whichever side of the
/// comparison the trees sit on: neither `a + b` against `c + c` nor
/// `c + c` against `a + b` can be asked under the colliding map.
#[test]
fn alpha_renaming_non_injective_refusal_is_symmetric() {
    let (a, a_reference) = build_identifier("a");
    let (b, b_reference) = build_identifier("b");
    let (c, c_reference) = build_identifier("c");
    let colliding = HashMap::from([(a.clone(), c.clone()), (b.clone(), c.clone())]);
    let sum = &a_reference + &b_reference;
    let doubled = &c_reference + &c_reference;

    let refused = AlphaRenaming::try_new(colliding);

    assert_eq!(refused.expect_err("c is a shared image").image(), &c);
    let forward = build_renaming([(a.clone(), c.clone())]);
    let backward = build_renaming([(c, a)]);
    assert!(!sum.is_alpha_equivalent_under(&doubled, &forward));
    assert!(!doubled.is_alpha_equivalent_under(&sum, &backward));
}

/// Test the identifier check follows the renaming, then identity, and
/// refuses an unmapped identifier standing for an image.
#[rstest]
#[case::mapped_to_its_image("x", "w", true)]
#[case::mapped_to_itself("x", "x", false)]
#[case::mapped_to_another("x", "v", false)]
#[case::unmapped_to_itself("v", "v", true)]
#[case::unmapped_to_another("v", "u", false)]
#[case::unmapped_to_an_image("w", "w", false)]
fn alpha_renaming_are_identifiers_alpha_equivalent_follows_the_renaming(
    #[case] left: &str,
    #[case] right: &str,
    #[case] expected: bool,
) {
    let identifiers: HashMap<&str, Identifier> = ["x", "w", "v", "u"]
        .into_iter()
        .map(|name| (name, Identifier::new(name)))
        .collect();
    let renaming = build_renaming([(identifiers["x"].clone(), identifiers["w"].clone())]);

    let equivalent =
        renaming.are_identifiers_alpha_equivalent(&identifiers[left], &identifiers[right]);

    assert_eq!(equivalent, expected);
}

// =============================================================================
// Shared subtrees
// =============================================================================

/// The number of additions in the doubling DAGs: they have `2^65 - 1`
/// occurrences, which no walk visiting every occurrence finishes.
const DOUBLING_LEVELS: usize = 64;

/// Return `(x < 3 ? f(-x) : x ** 2)` over the reference `x`.
fn build_mixed_tree(x: &Expression) -> Expression {
    build_piecewise_node_or_panic(
        vec![(x.less(3), build_call_node_or_panic("f", vec![-x]))],
        x.power(2),
    )
}

/// Test substituting a map that replaces nothing in the tree returns the
/// input itself.
#[rstest]
#[case::empty_map(HashMap::new())]
#[case::absent_identifier(HashMap::from([(Identifier::new("z"), build_literal(5))]))]
fn expression_substitute_replacing_nothing_returns_the_input_itself(
    #[case] replacements: HashMap<Identifier, Expression>,
) {
    let (_, x) = build_identifier("x");
    let expression = build_mixed_tree(&x);

    let result = expression
        .substitute(&replacements)
        .expect("nothing to refuse");

    assert!(Expression::ptr_eq(&result, &expression));
}

/// Test substitution rebuilds only the nodes above a replaced identifier:
/// a subtree with nothing replaced, and a leaf beside the replacement, keep
/// their nodes.
#[test]
fn expression_substitute_keeps_the_subtrees_it_does_not_change() {
    let (x, x_reference) = build_identifier("x");
    let (_, y_reference) = build_identifier("y");
    let one = build_literal(1);
    let changed = Expression::new_binary(BinaryOperation::Add, &x_reference, &one);
    let untouched = Expression::new_binary(BinaryOperation::Subtract, &y_reference, 2);
    let expression = Expression::new_binary(BinaryOperation::Multiply, &changed, &untouched);

    let result = expression
        .substitute(&HashMap::from([(x, build_literal(5))]))
        .expect("no piecewise to refuse");

    let node = expect_binary(&result);
    assert!(Expression::ptr_eq(node.right(), &untouched));
    assert!(!Expression::ptr_eq(node.left(), &changed));
    assert!(Expression::ptr_eq(expect_binary(node.left()).right(), &one));
    assert_eq!(
        result,
        Expression::new_binary(
            BinaryOperation::Multiply,
            Expression::new_binary(BinaryOperation::Add, 5, 1),
            &untouched
        )
    );
}

/// Test a shared subtree with nothing replaced keeps its node at every
/// occurrence, beside a replaced identifier: `(a + s) - s` becomes
/// `(b + s) - s` over the same `s`.
#[test]
fn expression_substitute_keeps_an_untouched_shared_subtree_at_every_occurrence() {
    let (a, a_reference) = build_identifier("a");
    let (_, b_reference) = build_identifier("b");
    let (_, c_reference) = build_identifier("c");
    let shared = Expression::new_binary(BinaryOperation::Multiply, &c_reference, 2);
    let expression = Expression::new_binary(
        BinaryOperation::Subtract,
        Expression::new_binary(BinaryOperation::Add, &a_reference, &shared),
        &shared,
    );

    let result = expression
        .substitute(&HashMap::from([(a, b_reference.clone())]))
        .expect("no piecewise to refuse");

    let node = expect_binary(&result);
    let sum = expect_binary(node.left());
    assert!(Expression::ptr_eq(sum.left(), &b_reference));
    assert!(Expression::ptr_eq(sum.right(), &shared));
    assert!(Expression::ptr_eq(node.right(), &shared));
}

/// Test substituting into a doubling DAG substitutes each shared node once
/// and keeps the sharing: the output is the doubling DAG over the
/// replacement.
#[test]
fn expression_substitute_of_a_doubling_dag_keeps_its_sharing() {
    let (a, a_reference) = build_identifier("a");
    let (_, b_reference) = build_identifier("b");
    let dag = build_doubling_dag(&a_reference, DOUBLING_LEVELS);

    let result = dag
        .substitute(&HashMap::from([(a, b_reference.clone())]))
        .expect("no piecewise to refuse");

    assert!(is_doubling_dag_over(&result, &b_reference, DOUBLING_LEVELS));
}

/// Test a doubling DAG nothing in which is replaced comes back as itself,
/// beside a doubling DAG that is substituted into.
#[test]
fn expression_substitute_keeps_an_untouched_doubling_dag_itself() {
    let (a, a_reference) = build_identifier("a");
    let (_, b_reference) = build_identifier("b");
    let (_, c_reference) = build_identifier("c");
    let changed = build_doubling_dag(&a_reference, DOUBLING_LEVELS);
    let untouched = build_doubling_dag(&c_reference, DOUBLING_LEVELS);
    let expression = Expression::new_binary(BinaryOperation::Multiply, &changed, &untouched);

    let result = expression
        .substitute(&HashMap::from([(a, b_reference.clone())]))
        .expect("no piecewise to refuse");

    let node = expect_binary(&result);
    assert!(Expression::ptr_eq(node.right(), &untouched));
    assert!(is_doubling_dag_over(
        node.left(),
        &b_reference,
        DOUBLING_LEVELS
    ));
}

// =============================================================================
// Deep trees
// =============================================================================

/// Test dropping a tree [`SMALL_STACK_DEPTH`] levels deep completes on
/// a small thread stack.
#[test]
fn expression_drop_of_a_deep_tree_completes_on_a_small_stack() {
    run_on_small_stack(|| {
        let (_, x) = build_identifier("x");
        let tree = build_deep_sum(&x, SMALL_STACK_DEPTH);

        drop(tree);
    });
}

/// Test a deep tree's free identifiers are found at the bottom, on a small
/// thread stack.
#[test]
fn expression_free_identifiers_of_a_deep_tree_reach_the_bottom_on_a_small_stack() {
    run_on_small_stack(|| {
        let (x, reference) = build_identifier("x");
        let tree = build_deep_sum(&reference, SMALL_STACK_DEPTH);

        let free = tree.free_identifiers();

        assert_eq!(free, collect_identifiers([&x]));
    });
}

/// Test two deep trees built separately are equal and hash equally, and
/// differ and hash differently when only their bottom leaf differs, on a
/// small thread stack.
#[test]
fn expression_equality_and_hash_of_deep_trees_reach_the_bottom_on_a_small_stack() {
    run_on_small_stack(|| {
        let (_, x) = build_identifier("x");
        let (_, y) = build_identifier("y");
        let first = build_deep_sum(&x, SMALL_STACK_DEPTH);
        let second = build_deep_sum(&x, SMALL_STACK_DEPTH);
        let other = build_deep_sum(&y, SMALL_STACK_DEPTH);

        assert!(first == second, "equal deep trees compare unequal");
        assert!(first != other, "deep trees over x and y compare equal");
        assert_eq!(hash_of(&first), hash_of(&second));
        assert_ne!(hash_of(&first), hash_of(&other));
    });
}

/// Test renaming equivalence reaches the bottom of two deep trees under a
/// non-empty renaming, on a small thread stack: the tree over `x` is
/// equivalent to the tree over `w` under `x -> w`, and not to the tree over
/// `v`.
#[test]
fn expression_is_alpha_equivalent_under_a_renaming_reaches_the_bottom_on_a_small_stack() {
    run_on_small_stack(|| {
        let (x, x_reference) = build_identifier("x");
        let (w, w_reference) = build_identifier("w");
        let (_, v_reference) = build_identifier("v");
        let tree = build_deep_sum(&x_reference, SMALL_STACK_DEPTH);
        let renamed = build_deep_sum(&w_reference, SMALL_STACK_DEPTH);
        let other = build_deep_sum(&v_reference, SMALL_STACK_DEPTH);
        let renaming = build_renaming([(x, w)]);

        assert!(tree.is_alpha_equivalent_under(&renamed, &renaming));
        assert!(!tree.is_alpha_equivalent_under(&other, &renaming));
    });
}

/// Test substitution reaches the bottom of a deep tree on a small thread
/// stack.
#[test]
fn expression_substitute_reaches_the_bottom_of_a_deep_tree_on_a_small_stack() {
    run_on_small_stack(|| {
        let (x, reference) = build_identifier("x");
        let tree = build_deep_sum(&reference, SMALL_STACK_DEPTH);

        let result = tree
            .substitute(&HashMap::from([(x, build_literal(0))]))
            .expect("no piecewise to refuse");

        assert!(
            result == build_deep_sum(&build_literal(0), SMALL_STACK_DEPTH),
            "the substituted tree differs from the tree over 0"
        );
    });
}

/// Test substituting into a doubling DAG [`SMALL_STACK_DEPTH`] levels deep
/// keeps its sharing, on a small thread stack.
#[test]
fn expression_substitute_of_a_deep_doubling_dag_keeps_its_sharing_on_a_small_stack() {
    run_on_small_stack(|| {
        let (a, a_reference) = build_identifier("a");
        let (_, b_reference) = build_identifier("b");
        let dag = build_doubling_dag(&a_reference, SMALL_STACK_DEPTH);

        let result = dag
            .substitute(&HashMap::from([(a, b_reference.clone())]))
            .expect("no piecewise to refuse");

        assert!(is_doubling_dag_over(
            &result,
            &b_reference,
            SMALL_STACK_DEPTH
        ));
    });
}
