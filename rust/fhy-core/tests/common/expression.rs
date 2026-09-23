//! Shared builders for the expression tests.
//!
//! Included by the `expression_*` test targets with
//! `#[path = "common/expression.rs"] pub mod expression_support;`, so an
//! item one target does not use is not reported as dead code there.

use std::sync::LazyLock;
use std::thread;

use fhy_core::identifier::Identifier;
use fhy_core::symbolic::expression::builtins::{
    ComposedFunction, NativeFunctionSignature, list_composed_functions, list_native_functions,
};
use fhy_core::symbolic::expression::{
    BigInt, BinaryOperation, Expression, ExpressionKind, LiteralKind, LiteralValue, UnaryOperation,
    build_call, build_logical_and, build_piecewise,
};
use proptest::num::f64 as f64_class;
use proptest::prelude::*;
use proptest::sample::select;

/// Depth of the deep trees the depth tests build.
pub const DEEP_TREE_DEPTH: usize = 4000;

/// Stack size for the substitution and screen walks over a deep tree.
pub const WALK_STACK_BYTES: usize = 16 << 20;

/// Stack size for a serialization round trip of a deep tree through
/// `serde_json` values, whose own recursion needs the most room.
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

/// Return `((leaf + 1) + 1) + ...`, `depth` additions deep.
#[must_use]
pub fn build_deep_sum(leaf: &Expression, depth: usize) -> Expression {
    let mut tree = leaf.clone();
    for _ in 0..depth {
        tree = tree + 1;
    }
    tree
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

/// Run `body` on a new thread with a stack of `stack_bytes` bytes and return
/// its result, re-raising its panic if it panics.
///
/// # Panics
///
/// Panics if the thread cannot be spawned, and with `body`'s panic if
/// `body` panics.
pub fn run_on_large_stack<T, F>(stack_bytes: usize, body: F) -> T
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let handle = thread::Builder::new()
        .stack_size(stack_bytes)
        .spawn(body)
        .expect("the test thread spawns");
    match handle.join() {
        Ok(result) => result,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

/// Identifiers the generated trees refer to, with distinct name hints.
pub static IDENTIFIER_POOL: LazyLock<[Identifier; 3]> = LazyLock::new(|| {
    [
        Identifier::new("v0"),
        Identifier::new("v1"),
        Identifier::new("v2"),
    ]
});

/// Every unary operation.
pub const ALL_UNARY_OPERATIONS: [UnaryOperation; 3] = [
    UnaryOperation::Negate,
    UnaryOperation::Positive,
    UnaryOperation::LogicalNot,
];

/// Every binary operation.
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
