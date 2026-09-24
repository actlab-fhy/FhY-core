//! Shared builders for the expression tests.

use std::sync::LazyLock;

use fhy_core::expr::builtins::BuiltinFunction;
use fhy_core::expr::{
    BigInt, BinaryOperation, Callee, Decimal, Expression, ExpressionKind, LiteralValue,
    LogicalOperation, UnaryOperation,
};
use fhy_core::identifier::Identifier;
use proptest::num::f64 as f64_class;
use proptest::prelude::*;
use proptest::sample::select;

/// Depth of the deep trees and patterns the operations documented as
/// recursive are run over, on a stack sized for that recursion.
pub(crate) const DEEP_TREE_DEPTH: usize = 4000;

/// Stack size for matching a pattern [`DEEP_TREE_DEPTH`] levels deep, which
/// recurses once per pattern level.
pub(crate) const PATTERN_MATCH_STACK_BYTES: usize = 16 << 20;

/// Stack size for a serialization round trip of a [`DEEP_TREE_DEPTH`]-level
/// tree through `serde_json` values: twice the 32 MiB `Expression`
/// documents for an unoptimized build, as headroom.
pub(crate) const SERIALIZATION_STACK_BYTES: usize = 64 << 20;

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

/// Return the piecewise expression `Expression::piecewise` builds from
/// `cases`, given as a list, failing the test if it is refused.
///
/// # Panics
///
/// Panics if the constructor refuses the parts.
#[must_use]
pub(crate) fn build_piecewise_node_or_panic(
    cases: Vec<(Expression, Expression)>,
    otherwise: Expression,
) -> Expression {
    Expression::piecewise(cases, otherwise).expect("a valid piecewise")
}

/// Return the call `Expression::call` builds of the function named
/// `function_name` (see [`build_callee`]) with `arguments`, given as a list.
///
/// # Panics
///
/// Panics if `function_name` is empty.
#[must_use]
pub(crate) fn build_call_node_or_panic(
    function_name: &str,
    arguments: Vec<Expression>,
) -> Expression {
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

/// Return whether `dag` is a doubling DAG `levels` additions deep over the
/// node `leaf`, both operands of each addition one shared node.
#[must_use]
pub(crate) fn is_doubling_dag_over(dag: &Expression, leaf: &Expression, levels: usize) -> bool {
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
static CALLEES: LazyLock<Vec<Callee>> = LazyLock::new(|| {
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

/// The most distinct nodes a generated expression DAG has.
pub(crate) const MAX_DAG_NODES: usize = 12;

/// How one node of a generated expression DAG is built from the nodes
/// before it, each child picked by an index into them.
#[derive(Debug, Clone)]
enum DagNodeSpecification {
    Identifier(usize),
    Literal(LiteralValue),
    Unary(UnaryOperation, prop::sample::Index),
    Binary(BinaryOperation, prop::sample::Index, prop::sample::Index),
    Logical(LogicalOperation, Vec<prop::sample::Index>),
    Piecewise(
        Vec<(prop::sample::Index, prop::sample::Index)>,
        prop::sample::Index,
    ),
    Call(Callee, Vec<prop::sample::Index>),
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
        (
            select(ALL_LOGICAL_OPERATIONS.to_vec()),
            prop::collection::vec(index(), 2..5)
        )
            .prop_map(|(operation, operands)| DagNodeSpecification::Logical(operation, operands)),
        (prop::collection::vec((index(), index()), 1..4), index())
            .prop_map(|(cases, otherwise)| DagNodeSpecification::Piecewise(cases, otherwise)),
        (
            select(CALLEES.clone()),
            prop::collection::vec(index(), 0..4)
        )
            .prop_map(|(callee, arguments)| DagNodeSpecification::Call(callee, arguments)),
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
        DagNodeSpecification::Logical(operation, operands) => {
            Expression::new_logical(operation, operands.into_iter().map(pick))
        }
        DagNodeSpecification::Piecewise(cases, otherwise) => {
            let cases = cases
                .into_iter()
                .map(|(condition, value)| (coerce_to_condition(pick(condition)), pick(value)))
                .collect();
            build_piecewise_node_or_panic(cases, pick(otherwise))
        }
        DagNodeSpecification::Call(callee, arguments) => {
            Expression::call(callee, arguments.into_iter().map(pick))
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
/// coerced case conditions rule out.
pub(crate) fn build_expression_dag_strategy() -> BoxedStrategy<Expression> {
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
/// unary and binary nodes of every operation, logical nodes of two to four
/// operands, piecewise nodes of one to
/// three cases, and calls of zero to three arguments to built-in and
/// unknown functions, up to five levels deep. Float literals are finite
/// unless `with_non_finite_floats` is set.
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
