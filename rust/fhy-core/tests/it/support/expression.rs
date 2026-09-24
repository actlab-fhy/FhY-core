//! Shared builders for the expression tests.

use std::sync::LazyLock;

use fhy_core::expr::builtins::BuiltinFunction;
use fhy_core::expr::{
    BigInt, BinaryExpression, BinaryOperation, CallExpression, Callee, Decimal, Expression,
    ExpressionKind, LiteralValue, LogicalExpression, LogicalOperation, PiecewiseExpression,
    UnaryExpression, UnaryOperation,
};
use fhy_core::identifier::Identifier;
use proptest::num::f64 as f64_class;
use proptest::prelude::*;
use proptest::sample::select;

/// Mint an identifier named `name` and return it with a reference to it.
#[must_use]
pub(crate) fn build_identifier(name: &str) -> (Identifier, Expression) {
    let identifier = Identifier::new(name);
    let reference = Expression::from(identifier.clone());
    (identifier, reference)
}

/// Return a literal expression holding `value`.
#[must_use]
pub(crate) fn build_literal(value: impl Into<LiteralValue>) -> Expression {
    Expression::from(value.into())
}

/// Return the literal `expression` holds, failing the test otherwise.
///
/// # Panics
///
/// Panics if `expression` is not a literal.
#[must_use]
pub(crate) fn expect_literal(expression: &Expression) -> &LiteralValue {
    let ExpressionKind::Literal(literal) = expression.kind() else {
        panic!("expected a literal, got {expression:?}");
    };
    literal
}

/// Return the unary node `expression` holds, failing the test otherwise.
#[must_use]
pub(crate) fn expect_unary(expression: &Expression) -> &UnaryExpression {
    let ExpressionKind::Unary(node) = expression.kind() else {
        panic!("expected an unary node, got {expression:?}");
    };
    node
}

/// Return the binary node `expression` holds, failing the test otherwise.
#[must_use]
pub(crate) fn expect_binary(expression: &Expression) -> &BinaryExpression {
    let ExpressionKind::Binary(node) = expression.kind() else {
        panic!("expected a binary node, got {expression:?}");
    };
    node
}

/// Return the logical node `expression` holds, failing the test otherwise.
#[must_use]
pub(crate) fn expect_logical(expression: &Expression) -> &LogicalExpression {
    let ExpressionKind::Logical(node) = expression.kind() else {
        panic!("expected a logical node, got {expression:?}");
    };
    node
}

/// Return the piecewise node `expression` holds, failing the test otherwise.
#[must_use]
pub(crate) fn expect_piecewise(expression: &Expression) -> &PiecewiseExpression {
    let ExpressionKind::Piecewise(node) = expression.kind() else {
        panic!("expected a piecewise node, got {expression:?}");
    };
    node
}

/// Return the call node `expression` holds, failing the test otherwise.
#[must_use]
pub(crate) fn expect_call(expression: &Expression) -> &CallExpression {
    let ExpressionKind::Call(node) = expression.kind() else {
        panic!("expected a call node, got {expression:?}");
    };
    node
}

/// Return the literal expression `LiteralValue::parse_text` reads from
/// `text`.
///
/// # Panics
///
/// Panics if `text` is outside the literal grammar.
#[must_use]
pub(crate) fn build_parsed_literal(text: &str) -> Expression {
    build_literal(LiteralValue::parse_text(text).expect("the text is a literal text"))
}

/// Return a decimal literal expression holding the value of the numeric
/// text `text`, an integer text included.
///
/// # Panics
///
/// Panics if `text` is outside the literal grammar.
#[must_use]
pub(crate) fn build_decimal_literal(text: &str) -> Expression {
    Expression::from(LiteralValue::Decimal(
        text.parse::<Decimal>().expect("the text is a literal text"),
    ))
}

/// Return the piecewise expression `Expression::piecewise` builds from
/// `cases` and `otherwise`, failing the test if it is refused.
///
/// # Panics
///
/// Panics if the builder refuses the parts.
#[must_use]
pub(crate) fn build_piecewise_or_panic<C, V, O>(
    cases: impl IntoIterator<Item = (C, V)>,
    otherwise: O,
) -> Expression
where
    C: Into<Expression>,
    V: Into<Expression>,
    O: Into<Expression>,
{
    Expression::piecewise(cases, otherwise).expect("a valid piecewise")
}

/// Return the callee named `function_name`: the built-in function of that
/// name, or else the user function of that name.
///
/// # Panics
///
/// Panics if `function_name` is empty.
#[must_use]
pub(crate) fn build_callee(function_name: &str) -> Callee {
    function_name.parse().expect("a non-empty function name")
}

/// Return the call `Expression::call` builds of the function named
/// `function_name` (see [`build_callee`]) with `arguments`.
///
/// # Panics
///
/// Panics if `function_name` is empty.
#[must_use]
pub(crate) fn build_call_or_panic<I>(function_name: &str, arguments: I) -> Expression
where
    I: IntoIterator,
    I::Item: Into<Expression>,
{
    Expression::call(build_callee(function_name), arguments)
}

/// Return `((leaf + 1) + 1) + ...`, `depth` additions deep.
#[must_use]
pub(crate) fn build_deep_sum(leaf: &Expression, depth: usize) -> Expression {
    let mut tree = leaf.clone();
    for _ in 0..depth {
        tree = tree + 1;
    }
    tree
}

/// Return the doubling DAG over `leaf`, `levels` additions deep: `x0 = leaf`
/// and `x(k+1) = xk + xk`, both operands of each addition one shared node.
///
/// The DAG has `levels + 1` distinct nodes but `2^(levels + 1) - 1`
/// occurrences, so an operation that visits every occurrence never finishes
/// on it for 64 levels.
#[must_use]
pub(crate) fn build_doubling_dag(leaf: &Expression, levels: usize) -> Expression {
    let mut dag = leaf.clone();
    for _ in 0..levels {
        dag = Expression::new_binary(BinaryOperation::Add, &dag, &dag);
    }
    dag
}

/// Return `true && (true && (... && leaf))`, `depth` two-operand
/// conjunctions deep, with `leaf` at the bottom of the right spine.
#[must_use]
pub(crate) fn build_deep_conjunction(leaf: &Expression, depth: usize) -> Expression {
    let mut tree = leaf.clone();
    for _ in 0..depth {
        tree = Expression::all([build_literal(true), tree]);
    }
    tree
}

/// Identifiers the generated trees refer to, with distinct name hints.
pub(crate) static IDENTIFIER_POOL: LazyLock<[Identifier; 3]> = LazyLock::new(|| {
    [
        Identifier::new("v0"),
        Identifier::new("v1"),
        Identifier::new("v2"),
    ]
});

/// Every unary operation, in declaration order.
pub(crate) const ALL_UNARY_OPERATIONS: [UnaryOperation; 3] = [
    UnaryOperation::Negate,
    UnaryOperation::Positive,
    UnaryOperation::LogicalNot,
];

/// Return the position of `operation` in [`ALL_UNARY_OPERATIONS`]; the
/// exhaustive `match` fails to compile when a variant is added, so the list
/// stays complete.
const fn index_unary_operation(operation: UnaryOperation) -> usize {
    match operation {
        UnaryOperation::Negate => 0,
        UnaryOperation::Positive => 1,
        UnaryOperation::LogicalNot => 2,
    }
}

/// Every binary operation, in declaration order.
pub(crate) const ALL_BINARY_OPERATIONS: [BinaryOperation; 13] = [
    BinaryOperation::Add,
    BinaryOperation::Subtract,
    BinaryOperation::Multiply,
    BinaryOperation::Divide,
    BinaryOperation::FloorDivide,
    BinaryOperation::FloorMod,
    BinaryOperation::Power,
    BinaryOperation::Equal,
    BinaryOperation::NotEqual,
    BinaryOperation::Less,
    BinaryOperation::LessEqual,
    BinaryOperation::Greater,
    BinaryOperation::GreaterEqual,
];

/// Return the position of `operation` in [`ALL_BINARY_OPERATIONS`]; the
/// exhaustive `match` fails to compile when a variant is added, so the list
/// stays complete.
const fn index_binary_operation(operation: BinaryOperation) -> usize {
    match operation {
        BinaryOperation::Add => 0,
        BinaryOperation::Subtract => 1,
        BinaryOperation::Multiply => 2,
        BinaryOperation::Divide => 3,
        BinaryOperation::FloorDivide => 4,
        BinaryOperation::FloorMod => 5,
        BinaryOperation::Power => 6,
        BinaryOperation::Equal => 7,
        BinaryOperation::NotEqual => 8,
        BinaryOperation::Less => 9,
        BinaryOperation::LessEqual => 10,
        BinaryOperation::Greater => 11,
        BinaryOperation::GreaterEqual => 12,
    }
}

/// Every logical operation, in declaration order.
pub(crate) const ALL_LOGICAL_OPERATIONS: [LogicalOperation; 2] =
    [LogicalOperation::And, LogicalOperation::Or];

/// Return the position of `operation` in [`ALL_LOGICAL_OPERATIONS`]; the
/// exhaustive `match` fails to compile when a variant is added, so the list
/// stays complete.
const fn index_logical_operation(operation: LogicalOperation) -> usize {
    match operation {
        LogicalOperation::And => 0,
        LogicalOperation::Or => 1,
    }
}

const _: () = {
    let mut index = 0;
    while index < ALL_UNARY_OPERATIONS.len() {
        assert!(index_unary_operation(ALL_UNARY_OPERATIONS[index]) == index);
        index += 1;
    }
    let mut index = 0;
    while index < ALL_BINARY_OPERATIONS.len() {
        assert!(index_binary_operation(ALL_BINARY_OPERATIONS[index]) == index);
        index += 1;
    }
    let mut index = 0;
    while index < ALL_LOGICAL_OPERATIONS.len() {
        assert!(index_logical_operation(ALL_LOGICAL_OPERATIONS[index]) == index);
        index += 1;
    }
};

/// Callees the generated calls use: every built-in function and two named
/// functions no catalogue knows.
pub(crate) static CALLEES: LazyLock<Vec<Callee>> = LazyLock::new(|| {
    BuiltinFunction::iter()
        .map(Callee::from)
        .chain(["f", "g"].map(build_callee))
        .collect()
});

/// Return `expression` unless it is a literal other than a Boolean, which
/// becomes a Boolean literal, so it can stand as a case condition.
#[must_use]
pub(crate) fn coerce_to_condition(expression: Expression) -> Expression {
    match expression.kind() {
        ExpressionKind::Literal(literal) if !matches!(literal, LiteralValue::Bool(_)) => {
            Expression::from(LiteralValue::from(true))
        }
        _ => expression,
    }
}

/// Return a copy of `expression` sharing no node with it.
///
/// # Panics
///
/// Panics if a node does not rebuild from copies of its own children,
/// which a valid tree rules out.
#[must_use]
pub(crate) fn copy_deeply(expression: &Expression) -> Expression {
    match expression.kind() {
        ExpressionKind::Identifier(identifier) => Expression::from(identifier.clone()),
        ExpressionKind::Literal(literal) => Expression::from(literal.clone()),
        _ => expression
            .rebuild_with_children(expression.children().map(copy_deeply).collect())
            .expect("a node rebuilds from copies of its own children"),
    }
}

/// Return a strategy for finite floats of every magnitude and sign,
/// subnormals and both zeros included, with a few small values that recur.
fn build_finite_float_strategy() -> impl Strategy<Value = f64> {
    prop_oneof![
        3 => f64_class::POSITIVE
            | f64_class::NEGATIVE
            | f64_class::NORMAL
            | f64_class::SUBNORMAL
            | f64_class::ZERO,
        1 => select(vec![0.0, -0.0, 1.0, 1.5, 5.0]),
    ]
}

/// Return a strategy for literals of every kind: Booleans, small and
/// big integers, floats, and integers and decimals parsed from texts, short
/// and long.
///
/// Floats are finite unless `with_non_finite_floats` is set, in which case
/// NaNs of both signs and both infinities are drawn too.
///
/// # Panics
///
/// The strategy panics while generating if a generated text is outside the
/// literal grammar, which its patterns rule out.
pub(crate) fn build_literal_strategy(with_non_finite_floats: bool) -> BoxedStrategy<LiteralValue> {
    let float = if with_non_finite_floats {
        prop_oneof![
            4 => build_finite_float_strategy(),
            1 => select(vec![f64::NAN, -f64::NAN, f64::INFINITY, f64::NEG_INFINITY]),
        ]
        .boxed()
    } else {
        build_finite_float_strategy().boxed()
    };
    prop_oneof![
        any::<bool>().prop_map(LiteralValue::from),
        (-1000_i64..1000).prop_map(LiteralValue::from),
        any::<i64>().prop_map(LiteralValue::from),
        "-?[1-9][0-9]{18,40}".prop_map(|digits| {
            LiteralValue::from(digits.parse::<BigInt>().expect("generated digits"))
        }),
        float.prop_map(LiteralValue::from),
        "[0-9]{1,6}".prop_map(|text| LiteralValue::parse_text(&text).expect("an integer text")),
        "[0-9]{0,4}\\.[0-9]{1,4}|[0-9]{1,4}\\.[0-9]{0,4}"
            .prop_map(|text| LiteralValue::parse_text(&text).expect("a decimal text")),
        "[1-9][0-9]{19,30}\\.[0-9]{0,30}"
            .prop_map(|text| LiteralValue::parse_text(&text).expect("a long decimal text")),
    ]
    .boxed()
}

/// Return a strategy for trees over [`IDENTIFIER_POOL`] of every node kind,
/// up to five levels deep.
///
/// The trees hold unary and binary nodes of every operation, logical nodes
/// of two to four operands, piecewise nodes of one to three cases, and calls
/// of zero to three arguments to built-in and unknown functions. Float
/// literals are finite unless `with_non_finite_floats` is set.
///
/// # Panics
///
/// The strategy panics while generating if a node is refused, which the
/// coerced case conditions rule out.
pub(crate) fn build_expression_strategy(with_non_finite_floats: bool) -> BoxedStrategy<Expression> {
    let leaf = prop_oneof![
        (0..IDENTIFIER_POOL.len())
            .prop_map(|index| Expression::from(IDENTIFIER_POOL[index].clone())),
        build_literal_strategy(with_non_finite_floats).prop_map(Expression::from),
    ];
    leaf.prop_recursive(5, 32, 4, |inner| {
        prop_oneof![
            (select(ALL_UNARY_OPERATIONS.to_vec()), inner.clone())
                .prop_map(|(operation, operand)| Expression::new_unary(operation, operand)),
            (
                select(ALL_BINARY_OPERATIONS.to_vec()),
                inner.clone(),
                inner.clone()
            )
                .prop_map(|(operation, left, right)| Expression::new_binary(
                    operation, left, right
                )),
            (
                select(ALL_LOGICAL_OPERATIONS.to_vec()),
                prop::collection::vec(inner.clone(), 2..5)
            )
                .prop_map(|(operation, operands)| Expression::new_logical(operation, operands)),
            (
                prop::collection::vec(
                    (inner.clone().prop_map(coerce_to_condition), inner.clone()),
                    1..4
                ),
                inner.clone(),
            )
                .prop_map(|(cases, otherwise)| {
                    Expression::piecewise(cases, otherwise).expect("conditions are coerced")
                }),
            (select(CALLEES.clone()), prop::collection::vec(inner, 0..4))
                .prop_map(|(callee, arguments)| Expression::call(callee, arguments)),
        ]
    })
    .boxed()
}
