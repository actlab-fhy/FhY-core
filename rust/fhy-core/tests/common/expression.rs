//! Shared builders for the expression tests.
//!
//! Included by the `expression_*` test targets with
//! `#[path = "common/expression.rs"] pub mod expression_support;`, so an
//! item one target does not use is not reported as dead code there.

use std::sync::LazyLock;

use fhy_core::expr::builtins::{
    ComposedFunction, NativeFunctionSignature, list_composed_functions, list_native_functions,
};
use fhy_core::expr::{
    BigInt, BinaryOperation, CallExpression, Expression, ExpressionKind, IntoOperand, LiteralKind,
    LiteralValue, PiecewiseExpression, UnaryOperation, build_call, build_logical_and,
    build_piecewise,
};
use fhy_core::identifier::Identifier;
use proptest::num::f64 as f64_class;
use proptest::prelude::*;
use proptest::sample::select;

/// Depth of the deep trees and patterns the operations documented as
/// recursive are run over, on a stack sized for that recursion.
pub const DEEP_TREE_DEPTH: usize = 4000;

/// Stack size for matching a pattern [`DEEP_TREE_DEPTH`] levels deep, which
/// recurses once per pattern level.
pub const PATTERN_MATCH_STACK_BYTES: usize = 16 << 20;

/// Stack size for a serialization round trip of a [`DEEP_TREE_DEPTH`]-level
/// tree through `serde_json` values: twice the 32 MiB `Expression`
/// documents for an unoptimized build, as headroom.
pub const SERIALIZATION_STACK_BYTES: usize = 64 << 20;

/// Mint an identifier named `name` and return it with a reference to it.
#[must_use]
pub fn build_identifier(name: &str) -> (Identifier, Expression) {
    let identifier = Identifier::new(name);
    let reference = Expression::from(identifier.clone());
    (identifier, reference)
}

/// Return a literal expression holding `value`.
#[must_use]
pub fn build_literal(value: impl Into<LiteralValue>) -> Expression {
    Expression::from(value.into())
}

/// Return a literal expression holding the numeric text `text`.
///
/// # Panics
///
/// Panics if `text` is outside the literal grammar.
#[must_use]
pub fn build_text_literal(text: &str) -> Expression {
    Expression::from(LiteralValue::parse_text(text).expect("the text is a literal text"))
}

/// Return the piecewise expression `build_piecewise` builds from `cases`
/// and `otherwise`, failing the test if it is refused.
///
/// # Panics
///
/// Panics if the builder refuses the parts.
#[must_use]
pub fn build_piecewise_or_panic<C: IntoOperand, V: IntoOperand, O: IntoOperand>(
    cases: impl IntoIterator<Item = (C, V)>,
    otherwise: O,
) -> Expression {
    build_piecewise(cases, otherwise).expect("a valid piecewise")
}

/// Return the call `build_call` builds of `function_name` with `arguments`,
/// failing the test if it is refused.
///
/// # Panics
///
/// Panics if the builder refuses the call.
#[must_use]
pub fn build_call_or_panic<I>(function_name: &str, arguments: I) -> Expression
where
    I: IntoIterator,
    I::Item: IntoOperand,
{
    build_call(function_name, arguments).expect("a named call")
}

/// Return the piecewise expression built by the node constructor
/// `PiecewiseExpression::try_new`, failing the test if it is refused.
///
/// For tests whose expected trees must not depend on the builders.
///
/// # Panics
///
/// Panics if the constructor refuses the parts.
#[must_use]
pub fn build_piecewise_node_or_panic(
    cases: Vec<(Expression, Expression)>,
    otherwise: Expression,
) -> Expression {
    Expression::from(PiecewiseExpression::try_new(cases, otherwise).expect("a valid piecewise"))
}

/// Return the call built by the node constructor `CallExpression::try_new`,
/// failing the test if it is refused.
///
/// For tests whose expected trees must not depend on the builders.
///
/// # Panics
///
/// Panics if the constructor refuses the call.
#[must_use]
pub fn build_call_node_or_panic(function_name: &str, arguments: Vec<Expression>) -> Expression {
    Expression::from(CallExpression::try_new(function_name, arguments).expect("a valid call"))
}

/// Return `((leaf + 1) + 1) + ...`, `depth` additions deep.
#[must_use]
pub fn build_deep_sum(leaf: &Expression, depth: usize) -> Expression {
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
pub fn build_doubling_dag(leaf: &Expression, levels: usize) -> Expression {
    let mut dag = leaf.clone();
    for _ in 0..levels {
        dag = Expression::new_binary(BinaryOperation::Add, &dag, &dag);
    }
    dag
}

/// Return whether `dag` is a doubling DAG `levels` additions deep over the
/// node `leaf`, both operands of each addition one shared node.
#[must_use]
pub fn is_doubling_dag_over(dag: &Expression, leaf: &Expression, levels: usize) -> bool {
    let mut node = dag;
    for _ in 0..levels {
        let ExpressionKind::Binary(binary) = node.kind() else {
            return false;
        };
        if binary.operation() != BinaryOperation::Add
            || !Expression::ptr_eq(binary.left(), binary.right())
        {
            return false;
        }
        node = binary.left();
    }
    Expression::ptr_eq(node, leaf)
}

/// Return `leaf && (true && (true && ...))`, `depth` conjunctions deep, with
/// `leaf` at the bottom of the right spine.
///
/// # Panics
///
/// Panics if a conjunction of two operands is refused.
#[must_use]
pub fn build_deep_conjunction(leaf: &Expression, depth: usize) -> Expression {
    let mut tree = leaf.clone();
    for _ in 0..depth {
        tree = build_logical_and([build_literal(true), tree])
            .expect("two operands make a conjunction");
    }
    tree
}

/// Identifiers the generated trees refer to, with distinct name hints.
pub static IDENTIFIER_POOL: LazyLock<[Identifier; 3]> = LazyLock::new(|| {
    [
        Identifier::new("v0"),
        Identifier::new("v1"),
        Identifier::new("v2"),
    ]
});

/// Every unary operation, in declaration order.
pub const ALL_UNARY_OPERATIONS: [UnaryOperation; 3] = [
    UnaryOperation::Negate,
    UnaryOperation::Positive,
    UnaryOperation::LogicalNot,
];

/// Every binary operation, in declaration order.
pub const ALL_BINARY_OPERATIONS: [BinaryOperation; 15] = [
    BinaryOperation::Add,
    BinaryOperation::Subtract,
    BinaryOperation::Multiply,
    BinaryOperation::Divide,
    BinaryOperation::FloorDivide,
    BinaryOperation::Modulo,
    BinaryOperation::Power,
    BinaryOperation::LogicalAnd,
    BinaryOperation::LogicalOr,
    BinaryOperation::Equal,
    BinaryOperation::NotEqual,
    BinaryOperation::Less,
    BinaryOperation::LessEqual,
    BinaryOperation::Greater,
    BinaryOperation::GreaterEqual,
];

/// Function names the generated calls use: every built-in function and two
/// names no catalogue knows.
static CALL_NAMES: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    list_composed_functions()
        .iter()
        .map(ComposedFunction::name)
        .chain(
            list_native_functions()
                .iter()
                .map(NativeFunctionSignature::name),
        )
        .chain(["f", "g"])
        .collect()
});

/// Return `expression` unless it is a literal other than a Boolean, which
/// becomes a Boolean literal, so it can stand as a case condition.
#[must_use]
pub fn coerce_to_condition(expression: Expression) -> Expression {
    match expression.kind() {
        ExpressionKind::Literal(literal) if !matches!(literal.kind(), LiteralKind::Bool(_)) => {
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
pub fn copy_deeply(expression: &Expression) -> Expression {
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
/// big integers, floats, and integer and decimal texts, short and long.
/// Floats are finite unless `with_non_finite_floats` is set, in which case
/// NaNs of both signs and both infinities are drawn too.
///
/// # Panics
///
/// The strategy panics while generating if a generated text is outside the
/// literal grammar, which its patterns rule out.
pub fn build_literal_strategy(with_non_finite_floats: bool) -> BoxedStrategy<LiteralValue> {
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

/// The most distinct nodes a generated expression DAG has.
pub const MAX_DAG_NODES: usize = 12;

/// How one node of a generated expression DAG is built from the nodes
/// before it, each child picked by an index into them.
#[derive(Debug, Clone)]
enum DagNodeSpecification {
    Identifier(usize),
    Literal(LiteralValue),
    Unary(UnaryOperation, prop::sample::Index),
    Binary(BinaryOperation, prop::sample::Index, prop::sample::Index),
    Piecewise(
        Vec<(prop::sample::Index, prop::sample::Index)>,
        prop::sample::Index,
    ),
    Call(&'static str, Vec<prop::sample::Index>),
}

/// Return a strategy for the specification of one DAG node of any kind.
fn build_dag_node_specification_strategy() -> BoxedStrategy<DagNodeSpecification> {
    let index = any::<prop::sample::Index>;
    prop_oneof![
        (0..IDENTIFIER_POOL.len()).prop_map(DagNodeSpecification::Identifier),
        build_literal_strategy(false).prop_map(DagNodeSpecification::Literal),
        (select(ALL_UNARY_OPERATIONS.to_vec()), index())
            .prop_map(|(operation, operand)| DagNodeSpecification::Unary(operation, operand)),
        (select(ALL_BINARY_OPERATIONS.to_vec()), index(), index()).prop_map(
            |(operation, left, right)| DagNodeSpecification::Binary(operation, left, right)
        ),
        (prop::collection::vec((index(), index()), 1..4), index())
            .prop_map(|(cases, otherwise)| DagNodeSpecification::Piecewise(cases, otherwise)),
        (
            select(CALL_NAMES.clone()),
            prop::collection::vec(index(), 0..4)
        )
            .prop_map(|(name, arguments)| DagNodeSpecification::Call(name, arguments)),
    ]
    .boxed()
}

/// Build the node `specification` describes over the earlier `nodes`, of
/// which there is at least one.
fn build_dag_node(specification: DagNodeSpecification, nodes: &[Expression]) -> Expression {
    let pick = |index: prop::sample::Index| nodes[index.index(nodes.len())].clone();
    match specification {
        DagNodeSpecification::Identifier(index) => Expression::from(IDENTIFIER_POOL[index].clone()),
        DagNodeSpecification::Literal(value) => Expression::from(value),
        DagNodeSpecification::Unary(operation, operand) => {
            Expression::new_unary(operation, pick(operand))
        }
        DagNodeSpecification::Binary(operation, left, right) => {
            Expression::new_binary(operation, pick(left), pick(right))
        }
        DagNodeSpecification::Piecewise(cases, otherwise) => {
            let cases = cases
                .into_iter()
                .map(|(condition, value)| (coerce_to_condition(pick(condition)), pick(value)))
                .collect();
            build_piecewise_node_or_panic(cases, pick(otherwise))
        }
        DagNodeSpecification::Call(name, arguments) => {
            build_call_node_or_panic(name, arguments.into_iter().map(pick).collect())
        }
    }
}

/// Return a strategy for expression DAGs over [`IDENTIFIER_POOL`] of up to
/// [`MAX_DAG_NODES`] distinct nodes of every kind, built with the node
/// constructors. The first node is an identifier reference, and every child
/// of a later node is any earlier node, so a node may occur many times.
///
/// # Panics
///
/// The strategy panics while generating if a node is refused, which the
/// coerced case conditions and the non-empty function names rule out.
pub fn build_expression_dag_strategy() -> BoxedStrategy<Expression> {
    (
        0..IDENTIFIER_POOL.len(),
        prop::collection::vec(build_dag_node_specification_strategy(), 0..MAX_DAG_NODES),
    )
        .prop_map(|(first, specifications)| {
            let mut nodes = vec![Expression::from(IDENTIFIER_POOL[first].clone())];
            for specification in specifications {
                let node = build_dag_node(specification, &nodes);
                nodes.push(node);
            }
            nodes.pop().expect("at least the first node")
        })
        .boxed()
}

/// Return a strategy for trees over [`IDENTIFIER_POOL`] of every node kind:
/// unary and binary nodes of every operation, piecewise nodes of one to
/// three cases, and calls of zero to three arguments to built-in and
/// unknown functions, up to five levels deep. Float literals are finite
/// unless `with_non_finite_floats` is set.
///
/// # Panics
///
/// The strategy panics while generating if a node is refused, which the
/// coerced case conditions and the non-empty function names rule out.
pub fn build_expression_strategy(with_non_finite_floats: bool) -> BoxedStrategy<Expression> {
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
                prop::collection::vec(
                    (inner.clone().prop_map(coerce_to_condition), inner.clone()),
                    1..4
                ),
                inner.clone(),
            )
                .prop_map(|(cases, otherwise)| {
                    build_piecewise(cases, otherwise).expect("conditions are coerced")
                }),
            (
                select(CALL_NAMES.clone()),
                prop::collection::vec(inner, 0..4)
            )
                .prop_map(|(function_name, arguments)| {
                    build_call(function_name, arguments).expect("a named call")
                }),
        ]
    })
    .boxed()
}
