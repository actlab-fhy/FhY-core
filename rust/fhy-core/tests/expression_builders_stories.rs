//! Tests for building expressions: the operand types, the operator
//! overloads, the operation-named builder methods, the variadic builders,
//! and the build errors.
//!
//! Public API only (`fhy_core::symbolic::expression`).

#[path = "common/expression.rs"]
pub mod expression_support;

use expression_support::{build_identifier, build_literal, build_text_literal};
use fhy_core::identifier::Identifier;
use fhy_core::symbolic::expression::{
    BinaryOperation, Expression, ExpressionBuildError, ExpressionKind, IntoOperand, LiteralValue,
    UnaryOperation, build_call, build_logical_and, build_logical_or, build_piecewise,
};
use num_bigint::BigInt;
use rstest::rstest;

/// The binary operations reachable from an expression on the left, each
/// through its operator or its builder method.
#[derive(Debug, Clone, Copy)]
enum BinaryBuilder {
    Add,
    Subtract,
    Multiply,
    Divide,
    FloorDivide,
    Modulo,
    Power,
    Equals,
    NotEquals,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

impl BinaryBuilder {
    /// Return the operation the builder applies.
    fn operation(self) -> BinaryOperation {
        match self {
            Self::Add => BinaryOperation::Add,
            Self::Subtract => BinaryOperation::Subtract,
            Self::Multiply => BinaryOperation::Multiply,
            Self::Divide => BinaryOperation::Divide,
            Self::FloorDivide => BinaryOperation::FloorDivide,
            Self::Modulo => BinaryOperation::Modulo,
            Self::Power => BinaryOperation::Power,
            Self::Equals => BinaryOperation::Equal,
            Self::NotEquals => BinaryOperation::NotEqual,
            Self::Less => BinaryOperation::Less,
            Self::LessEqual => BinaryOperation::LessEqual,
            Self::Greater => BinaryOperation::Greater,
            Self::GreaterEqual => BinaryOperation::GreaterEqual,
        }
    }

    /// Apply the builder to `left` and `right`.
    fn apply(self, left: &Expression, right: impl IntoOperand) -> Expression {
        match self {
            Self::Add => left + right,
            Self::Subtract => left - right,
            Self::Multiply => left * right,
            Self::Divide => left / right,
            Self::FloorDivide => left.floor_divide(right),
            Self::Modulo => left % right,
            Self::Power => left.power(right),
            Self::Equals => left.equals(right),
            Self::NotEquals => left.not_equals(right),
            Self::Less => left.less(right),
            Self::LessEqual => left.less_equal(right),
            Self::Greater => left.greater(right),
            Self::GreaterEqual => left.greater_equal(right),
        }
    }
}

/// A non-expression operand, turned into the expression it stands for.
#[derive(Debug, Clone)]
enum PlainOperand {
    Integer(i64),
    Float(f64),
    Identifier(Identifier),
}

impl PlainOperand {
    /// Return the expression the operand stands for.
    fn to_expression(&self) -> Expression {
        match self {
            Self::Integer(value) => build_literal(*value),
            Self::Float(value) => build_literal(*value),
            Self::Identifier(identifier) => Expression::from(identifier.clone()),
        }
    }
}

/// Return the three plain operand kinds: an integer, a float, and an
/// identifier.
fn build_plain_operands() -> [PlainOperand; 3] {
    [
        PlainOperand::Integer(10),
        PlainOperand::Float(10.5),
        PlainOperand::Identifier(Identifier::new("y")),
    ]
}

// =============================================================================
// Unary builders
// =============================================================================

/// Test negation and the positive builder build unary nodes with their
/// operation.
#[rstest]
#[case::negate(UnaryOperation::Negate)]
#[case::positive(UnaryOperation::Positive)]
#[case::logical_not(UnaryOperation::LogicalNot)]
fn expression_unary_builders_produce_the_matching_unary_node(#[case] operation: UnaryOperation) {
    let operand = build_literal(5);

    let built = match operation {
        UnaryOperation::Negate => -&operand,
        UnaryOperation::Positive => operand.positive(),
        UnaryOperation::LogicalNot => operand.logical_not(),
    };

    let ExpressionKind::Unary(node) = built.kind() else {
        panic!("expected a unary node, got {built:?}");
    };
    assert_eq!(node.operation(), operation);
    assert!(Expression::ptr_eq(node.operand(), &operand));
}

/// Test negating an owned expression builds the same node as negating a
/// borrowed one.
#[test]
fn expression_negation_of_owned_and_borrowed_expressions_agree() {
    let (_, x) = build_identifier("x");

    let owned = -x.clone();
    let borrowed = -&x;

    assert_eq!(owned, borrowed);
    assert_eq!(owned, Expression::new_unary(UnaryOperation::Negate, &x));
}

// =============================================================================
// Binary builders
// =============================================================================

/// Test each binary builder, over two expressions, builds the binary node
/// with its operation.
#[rstest]
fn expression_binary_builders_produce_the_matching_binary_node(
    #[values(
        BinaryBuilder::Add,
        BinaryBuilder::Subtract,
        BinaryBuilder::Multiply,
        BinaryBuilder::Divide,
        BinaryBuilder::FloorDivide,
        BinaryBuilder::Modulo,
        BinaryBuilder::Power,
        BinaryBuilder::Equals,
        BinaryBuilder::NotEquals,
        BinaryBuilder::Less,
        BinaryBuilder::LessEqual,
        BinaryBuilder::Greater,
        BinaryBuilder::GreaterEqual
    )]
    builder: BinaryBuilder,
) {
    let left = build_literal(5);
    let right = build_literal(10);

    let built = builder.apply(&left, &right);

    let ExpressionKind::Binary(node) = built.kind() else {
        panic!("expected a binary node, got {built:?}");
    };
    assert_eq!(node.operation(), builder.operation());
    assert!(Expression::ptr_eq(node.left(), &left));
    assert!(Expression::ptr_eq(node.right(), &right));
}

/// Test each binary builder wraps a plain right operand in the expression it
/// stands for.
#[rstest]
fn expression_binary_builders_promote_a_plain_right_operand(
    #[values(
        BinaryBuilder::Add,
        BinaryBuilder::Subtract,
        BinaryBuilder::Multiply,
        BinaryBuilder::Divide,
        BinaryBuilder::FloorDivide,
        BinaryBuilder::Modulo,
        BinaryBuilder::Power,
        BinaryBuilder::Equals,
        BinaryBuilder::NotEquals,
        BinaryBuilder::Less,
        BinaryBuilder::LessEqual,
        BinaryBuilder::Greater,
        BinaryBuilder::GreaterEqual
    )]
    builder: BinaryBuilder,
    #[values(0, 1, 2)] operand_index: usize,
) {
    let left = build_literal(5);
    let right = build_plain_operands()[operand_index].clone();

    let built = match &right {
        PlainOperand::Integer(value) => builder.apply(&left, *value),
        PlainOperand::Float(value) => builder.apply(&left, *value),
        PlainOperand::Identifier(identifier) => builder.apply(&left, identifier.clone()),
    };

    let expected = Expression::new_binary(builder.operation(), &left, right.to_expression());
    assert_eq!(built, expected);
}

/// Test each arithmetic operator wraps a plain left operand in the
/// expression it stands for; floor division and exponentiation take one
/// through `new_binary`.
#[rstest]
fn expression_arithmetic_builders_promote_a_plain_left_operand(
    #[values(
        BinaryOperation::Add,
        BinaryOperation::Subtract,
        BinaryOperation::Multiply,
        BinaryOperation::Divide,
        BinaryOperation::FloorDivide,
        BinaryOperation::Modulo,
        BinaryOperation::Power
    )]
    operation: BinaryOperation,
    #[values(0, 1, 2)] operand_index: usize,
) {
    let right = build_literal(5);
    let left = build_plain_operands()[operand_index].clone();

    let built = match (operation, left.clone()) {
        (BinaryOperation::Add, PlainOperand::Integer(value)) => value + &right,
        (BinaryOperation::Add, PlainOperand::Float(value)) => value + &right,
        (BinaryOperation::Add, PlainOperand::Identifier(value)) => value + &right,
        (BinaryOperation::Subtract, PlainOperand::Integer(value)) => value - &right,
        (BinaryOperation::Subtract, PlainOperand::Float(value)) => value - &right,
        (BinaryOperation::Subtract, PlainOperand::Identifier(value)) => value - &right,
        (BinaryOperation::Multiply, PlainOperand::Integer(value)) => value * &right,
        (BinaryOperation::Multiply, PlainOperand::Float(value)) => value * &right,
        (BinaryOperation::Multiply, PlainOperand::Identifier(value)) => value * &right,
        (BinaryOperation::Divide, PlainOperand::Integer(value)) => value / &right,
        (BinaryOperation::Divide, PlainOperand::Float(value)) => value / &right,
        (BinaryOperation::Divide, PlainOperand::Identifier(value)) => value / &right,
        (BinaryOperation::Modulo, PlainOperand::Integer(value)) => value % &right,
        (BinaryOperation::Modulo, PlainOperand::Float(value)) => value % &right,
        (BinaryOperation::Modulo, PlainOperand::Identifier(value)) => value % &right,
        (operation, PlainOperand::Integer(value)) => {
            Expression::new_binary(operation, value, &right)
        }
        (operation, PlainOperand::Float(value)) => Expression::new_binary(operation, value, &right),
        (operation, PlainOperand::Identifier(value)) => {
            Expression::new_binary(operation, value, &right)
        }
    };

    let expected = Expression::new_binary(operation, left.to_expression(), &right);
    assert_eq!(built, expected);
}

/// Build an expression with a reflected operator from a reference, paired
/// with the same expression built through `new_binary`.
type BuildReflected = fn(&Expression) -> (Expression, Expression);

/// Test the reflected operators take a big integer, a literal value, and an
/// owned expression on either side.
#[rstest]
#[case::big_integer(|x: &Expression| {
    let big: BigInt = "100000000000000000000".parse().expect("digits");
    (big.clone() * x.clone(), Expression::new_binary(BinaryOperation::Multiply, big, x))
})]
#[case::literal_value(|x: &Expression| {
    let text = LiteralValue::parse_text("1.50").expect("a decimal text");
    (text.clone() - x, Expression::new_binary(BinaryOperation::Subtract, text, x))
})]
#[case::owned_expression(|x: &Expression| {
    (x.clone() + x.clone(), Expression::new_binary(BinaryOperation::Add, x, x))
})]
fn expression_reflected_operators_accept_every_left_operand_type(#[case] build: BuildReflected) {
    let (_, x) = build_identifier("x");

    let (built, expected) = build(&x);

    assert_eq!(built, expected);
}

/// Build `left op right` with each arithmetic operator `+ - * / %`, for an
/// owned and a borrowed expression `right`, each paired with the node
/// `new_binary` builds from the same operands. `left` is evaluated once per
/// use.
macro_rules! build_with_every_operator {
    ($left:expr, $right:expr) => {{
        let right: &Expression = $right;
        vec![
            (
                $left + right,
                Expression::new_binary(BinaryOperation::Add, $left, right),
            ),
            (
                $left + right.clone(),
                Expression::new_binary(BinaryOperation::Add, $left, right),
            ),
            (
                $left - right,
                Expression::new_binary(BinaryOperation::Subtract, $left, right),
            ),
            (
                $left - right.clone(),
                Expression::new_binary(BinaryOperation::Subtract, $left, right),
            ),
            (
                $left * right,
                Expression::new_binary(BinaryOperation::Multiply, $left, right),
            ),
            (
                $left * right.clone(),
                Expression::new_binary(BinaryOperation::Multiply, $left, right),
            ),
            (
                $left / right,
                Expression::new_binary(BinaryOperation::Divide, $left, right),
            ),
            (
                $left / right.clone(),
                Expression::new_binary(BinaryOperation::Divide, $left, right),
            ),
            (
                $left % right,
                Expression::new_binary(BinaryOperation::Modulo, $left, right),
            ),
            (
                $left % right.clone(),
                Expression::new_binary(BinaryOperation::Modulo, $left, right),
            ),
        ]
    }};
}

/// Build every arithmetic operator's node from one left operand type and a
/// reference, each paired with the node `new_binary` builds.
type BuildWithEveryOperator = fn(&Expression) -> Vec<(Expression, Expression)>;

/// Test every non-expression operand type is accepted on the left of every
/// arithmetic operator, the same types [`IntoOperand`] accepts on the
/// right, with an owned or a borrowed expression on the right.
#[rstest]
#[case::i64(|x: &Expression| build_with_every_operator!(3_i64, x))]
#[case::i32(|x: &Expression| build_with_every_operator!(3_i32, x))]
#[case::u32(|x: &Expression| build_with_every_operator!(3_u32, x))]
#[case::big_integer(|x: &Expression| build_with_every_operator!(build_big_operand(), x))]
#[case::f64(|x: &Expression| build_with_every_operator!(2.5_f64, x))]
#[case::owned_identifier(|x: &Expression| {
    let y = Identifier::new("y");
    build_with_every_operator!(y.clone(), x)
})]
#[case::borrowed_identifier(|x: &Expression| {
    let y = Identifier::new("y");
    build_with_every_operator!(&y, x)
})]
#[case::literal_value(|x: &Expression| {
    build_with_every_operator!(LiteralValue::parse_text("1.50").expect("a decimal text"), x)
})]
fn expression_arithmetic_operators_accept_every_operand_type_on_the_left(
    #[case] build: BuildWithEveryOperator,
) {
    let (_, x) = build_identifier("x");

    let pairs = build(&x);

    assert_eq!(pairs.len(), 10);
    for (built, expected) in pairs {
        assert_eq!(built, expected);
    }
}

/// Build a negation of one operand type from an identifier and a reference
/// to it, paired with the expression that operand stands for.
type LiftOperand = fn(&Identifier, &Expression) -> (Expression, Expression);

/// Return the big integer the big-integer operand case lifts.
fn build_big_operand() -> BigInt {
    "-100000000000000000000".parse().expect("digits")
}

/// Test every operand type lifts to the expression it stands for.
#[rstest]
#[case::owned_expression(|_: &Identifier, reference: &Expression| (
    Expression::new_unary(UnaryOperation::Negate, reference.clone()),
    reference.clone(),
))]
#[case::borrowed_expression(|_: &Identifier, reference: &Expression| (
    Expression::new_unary(UnaryOperation::Negate, reference),
    reference.clone(),
))]
#[case::owned_identifier(|identifier: &Identifier, reference: &Expression| (
    Expression::new_unary(UnaryOperation::Negate, identifier.clone()),
    reference.clone(),
))]
#[case::borrowed_identifier(|identifier: &Identifier, reference: &Expression| (
    Expression::new_unary(UnaryOperation::Negate, identifier),
    reference.clone(),
))]
#[case::literal_value(|_: &Identifier, _: &Expression| (
    Expression::new_unary(UnaryOperation::Negate, LiteralValue::from(true)),
    build_literal(true),
))]
#[case::i64(|_: &Identifier, _: &Expression| (Expression::new_unary(UnaryOperation::Negate, 7_i64), build_literal(7)))]
#[case::i32(|_: &Identifier, _: &Expression| (Expression::new_unary(UnaryOperation::Negate, 7_i32), build_literal(7)))]
#[case::u32(|_: &Identifier, _: &Expression| (Expression::new_unary(UnaryOperation::Negate, 7_u32), build_literal(7)))]
#[case::big_integer(|_: &Identifier, _: &Expression| (
    Expression::new_unary(UnaryOperation::Negate, build_big_operand()),
    build_literal(build_big_operand()),
))]
#[case::f64(|_: &Identifier, _: &Expression| (Expression::new_unary(UnaryOperation::Negate, 7.5_f64), build_literal(7.5)))]
fn expression_new_binary_lifts_every_operand_type(#[case] lift: LiftOperand) {
    let (identifier, reference) = build_identifier("x");

    let (built, operand) = lift(&identifier, &reference);

    assert_eq!(
        built,
        Expression::new_unary(UnaryOperation::Negate, operand)
    );
}

/// Test a borrowed expression operand is shared, not copied.
#[test]
fn expression_new_binary_shares_a_borrowed_operand() {
    let (_, x) = build_identifier("x");

    let built = Expression::new_binary(BinaryOperation::Less, &x, &x);

    let ExpressionKind::Binary(node) = built.kind() else {
        panic!("expected a binary node, got {built:?}");
    };
    assert!(Expression::ptr_eq(node.left(), &x));
    assert!(Expression::ptr_eq(node.right(), &x));
}

/// Test `new_binary` wraps an integer and a float operand in literals.
#[test]
fn expression_new_binary_constructs_with_literal_coercion() {
    let built = Expression::new_binary(BinaryOperation::Add, 1, 2.5);

    let expected =
        Expression::new_binary(BinaryOperation::Add, build_literal(1), build_literal(2.5));
    assert_eq!(built, expected);
}

/// Test `new_unary` wraps a number in a literal.
#[test]
fn expression_new_unary_constructs_with_literal_coercion() {
    let built = Expression::new_unary(UnaryOperation::Negate, 5);

    assert_eq!(
        built,
        Expression::new_unary(UnaryOperation::Negate, build_literal(5))
    );
}

/// Test a numeric text becomes an operand through an explicit literal value.
#[rstest]
#[case::integer_text("5")]
#[case::decimal_text("1.5")]
fn expression_binary_builder_takes_a_parsed_text_operand(#[case] text: &str) {
    let parsed = LiteralValue::parse_text(text).expect("a literal text");

    let built = build_literal(1) + parsed;

    let ExpressionKind::Binary(node) = built.kind() else {
        panic!("expected a binary node, got {built:?}");
    };
    let ExpressionKind::Literal(right) = node.right().kind() else {
        panic!("expected a literal right operand, got {:?}", node.right());
    };
    assert_eq!(right.to_string(), text);
}

// =============================================================================
// Logical builders
// =============================================================================

/// Test the variadic logical builders fold three operands to the right,
/// wrapping an identifier operand.
#[rstest]
#[case::and(BinaryOperation::LogicalAnd)]
#[case::or(BinaryOperation::LogicalOr)]
fn build_logical_folds_three_operands_to_the_right(#[case] operation: BinaryOperation) {
    let first = build_literal(true);
    let second = build_literal(false);
    let (third, third_reference) = build_identifier("c");
    let operands = [first.clone(), second.clone(), Expression::from(third)];

    let built = match operation {
        BinaryOperation::LogicalAnd => build_logical_and(operands),
        _ => build_logical_or(operands),
    }
    .expect("three operands");

    let expected = Expression::new_binary(
        operation,
        &first,
        Expression::new_binary(operation, &second, &third_reference),
    );
    assert_eq!(built, expected);
}

/// Test the variadic logical builders accept exactly two operands.
#[rstest]
#[case::and(BinaryOperation::LogicalAnd)]
#[case::or(BinaryOperation::LogicalOr)]
fn build_logical_accepts_two_operands(#[case] operation: BinaryOperation) {
    let first = build_literal(true);
    let second = build_literal(false);

    let built = match operation {
        BinaryOperation::LogicalAnd => build_logical_and([&first, &second]),
        _ => build_logical_or([&first, &second]),
    }
    .expect("two operands");

    assert_eq!(built, Expression::new_binary(operation, &first, &second));
}

/// Test the variadic logical builders refuse fewer than two operands,
/// reporting the operation and the count.
#[rstest]
fn build_logical_rejects_fewer_than_two_operands(
    #[values(BinaryOperation::LogicalAnd, BinaryOperation::LogicalOr)] operation: BinaryOperation,
    #[values(0, 1)] count: usize,
) {
    let operands: Vec<Expression> = (0..count).map(|_| build_literal(true)).collect();

    let result = match operation {
        BinaryOperation::LogicalAnd => build_logical_and(operands),
        _ => build_logical_or(operands),
    };

    assert_eq!(
        result,
        Err(ExpressionBuildError::TooFewLogicalOperands { operation, count })
    );
}

/// Test a conjunction of four operands folds to the right.
#[test]
fn build_logical_and_folds_four_operands_to_the_right() {
    let operands = [
        build_literal(true),
        build_literal(false),
        build_literal(true),
        build_literal(false),
    ];

    let built = build_logical_and(operands.clone()).expect("four operands");

    let and = BinaryOperation::LogicalAnd;
    let expected = Expression::new_binary(
        and,
        &operands[0],
        Expression::new_binary(
            and,
            &operands[1],
            Expression::new_binary(and, &operands[2], &operands[3]),
        ),
    );
    assert_eq!(built, expected);
}

/// Test a conjunction whose first operand is an existing expression includes
/// it, as a method-style call `a.logical_and(b, c)` would.
#[rstest]
#[case::and(BinaryOperation::LogicalAnd)]
#[case::or(BinaryOperation::LogicalOr)]
fn build_logical_includes_a_leading_expression(#[case] operation: BinaryOperation) {
    let leading = build_literal(true);
    let (second, third) = (build_literal(false), build_literal(true));

    let built = match operation {
        BinaryOperation::LogicalAnd => build_logical_and([&leading, &second, &third]),
        _ => build_logical_or([&leading, &second, &third]),
    }
    .expect("three operands");

    let expected = Expression::new_binary(
        operation,
        &leading,
        Expression::new_binary(operation, &second, &third),
    );
    assert_eq!(built, expected);
}

/// Test logical negation builds a negation node, and wraps a bare
/// identifier through `new_unary`.
#[test]
fn expression_logical_not_wraps_the_operand() {
    let operand = build_literal(true);
    let (identifier, reference) = build_identifier("a");

    let negated = operand.logical_not();
    let negated_identifier = Expression::new_unary(UnaryOperation::LogicalNot, identifier);

    assert_eq!(
        negated,
        Expression::new_unary(UnaryOperation::LogicalNot, &operand)
    );
    assert_eq!(negated_identifier, reference.logical_not());
}

/// Test a conjunction keeps both bounds of `0 <= c <= 5`.
#[test]
fn build_logical_and_keeps_both_bounds_of_a_range() {
    let (_, c) = build_identifier("c");
    let lower = Expression::new_binary(BinaryOperation::LessEqual, 0, &c);
    let upper = c.less_equal(5);

    let conjunction = build_logical_and([&lower, &upper]).expect("two operands");

    let ExpressionKind::Binary(node) = conjunction.kind() else {
        panic!("expected a binary node, got {conjunction:?}");
    };
    assert_eq!(node.operation(), BinaryOperation::LogicalAnd);
    assert_eq!(node.left(), &lower);
    assert_eq!(node.right(), &upper);
}

// =============================================================================
// Piecewise and call builders
// =============================================================================

/// Test the piecewise builder keeps expression operands as they are.
#[test]
fn build_piecewise_wraps_expression_operands_directly() {
    let (condition, value, otherwise) = (build_literal(true), build_literal(1), build_literal(2));

    let built = build_piecewise([(&condition, &value)], &otherwise).expect("one case");

    let ExpressionKind::Piecewise(node) = built.kind() else {
        panic!("expected a piecewise node, got {built:?}");
    };
    assert!(Expression::ptr_eq(&node.cases()[0].0, &condition));
    assert!(Expression::ptr_eq(&node.cases()[0].1, &value));
    assert!(Expression::ptr_eq(node.otherwise(), &otherwise));
}

/// Test the piecewise builder keeps several cases in order.
#[test]
fn build_piecewise_keeps_multiple_cases_in_declared_order() {
    let (first_condition, second_condition) = (build_literal(true), build_literal(false));
    let (first_value, second_value) = (build_literal(1), build_literal(2));

    let built = build_piecewise(
        [
            (&first_condition, &first_value),
            (&second_condition, &second_value),
        ],
        0,
    )
    .expect("two cases");

    let ExpressionKind::Piecewise(node) = built.kind() else {
        panic!("expected a piecewise node, got {built:?}");
    };
    assert_eq!(node.cases().len(), 2);
    assert!(Expression::ptr_eq(&node.cases()[0].0, &first_condition));
    assert!(Expression::ptr_eq(&node.cases()[1].0, &second_condition));
    assert!(Expression::ptr_eq(&node.cases()[0].1, &first_value));
    assert!(Expression::ptr_eq(&node.cases()[1].1, &second_value));
}

/// Test the piecewise builder wraps identifiers in each position and numbers
/// in the value and otherwise positions.
#[test]
fn build_piecewise_coerces_identifiers_and_numbers() {
    let (flag, flag_reference) = build_identifier("flag");
    let (x, x_reference) = build_identifier("x");
    let (fallback, fallback_reference) = build_identifier("fallback");

    let with_identifiers = build_piecewise([(flag, x)], fallback).expect("one case");
    let with_numbers = build_piecewise([(build_literal(true), 5)], 10).expect("one case");

    let expected_identifiers =
        build_piecewise([(&flag_reference, &x_reference)], &fallback_reference).expect("one case");
    let expected_numbers =
        build_piecewise([(build_literal(true), build_literal(5))], build_literal(10))
            .expect("one case");
    assert_eq!(with_identifiers, expected_identifiers);
    assert_eq!(with_numbers, expected_numbers);
}

/// Test the piecewise builder refuses a number as a case condition.
#[test]
fn build_piecewise_rejects_a_numeric_condition() {
    let result = build_piecewise([(1, 5)], 10);

    assert_eq!(
        result,
        Err(ExpressionBuildError::NonBooleanConditionLiteral { case_index: 0 })
    );
}

/// Test the piecewise builder refuses no cases.
#[test]
fn build_piecewise_rejects_zero_cases() {
    let result = build_piecewise(Vec::<(Expression, Expression)>::new(), 0);

    assert_eq!(result, Err(ExpressionBuildError::EmptyPiecewise));
}

/// Test Boolean literal value and otherwise operands pass through as given.
#[test]
fn build_piecewise_keeps_explicit_boolean_literal_operands() {
    let value = build_literal(true);
    let otherwise = build_literal(false);

    let built = build_piecewise([(build_literal(true), &value)], &otherwise).expect("one case");

    let ExpressionKind::Piecewise(node) = built.kind() else {
        panic!("expected a piecewise node, got {built:?}");
    };
    assert!(Expression::ptr_eq(&node.cases()[0].1, &value));
    assert!(Expression::ptr_eq(node.otherwise(), &otherwise));
}

/// Test an equality built with `equals` is a valid case condition.
#[test]
fn build_piecewise_accepts_an_equals_condition() {
    let (_, x) = build_identifier("x");

    let built = build_piecewise([(x.equals(0), 7), (x.less_equal(0), 8)], 9).expect("two cases");

    let ExpressionKind::Piecewise(node) = built.kind() else {
        panic!("expected a piecewise node, got {built:?}");
    };
    assert_eq!(node.cases()[0].0, x.equals(0));
}

/// Test the call builder carries the name and the argument nodes.
#[test]
fn build_call_returns_a_call_with_name_and_arguments() {
    let (first, second) = (build_literal(1), build_literal(2));

    let built = build_call("max", [&first, &second]).expect("a named call");

    let ExpressionKind::Call(node) = built.kind() else {
        panic!("expected a call node, got {built:?}");
    };
    assert_eq!(node.function_name(), "max");
    assert_eq!(node.arguments().len(), 2);
    assert!(Expression::ptr_eq(&node.arguments()[0], &first));
    assert!(Expression::ptr_eq(&node.arguments()[1], &second));
}

/// Test the call builder takes no arguments.
#[test]
fn build_call_supports_zero_arguments() {
    let built = build_call("nullary", Vec::<Expression>::new()).expect("a named call");

    let ExpressionKind::Call(node) = built.kind() else {
        panic!("expected a call node, got {built:?}");
    };
    assert!(node.arguments().is_empty());
}

/// Test the call builder wraps identifier and number arguments.
#[test]
fn build_call_coerces_identifiers_and_numbers() {
    let (x, x_reference) = build_identifier("x");

    let with_identifier = build_call("f", [x]).expect("a named call");
    let with_numbers = build_call("max", [1, 2]).expect("a named call");

    assert_eq!(
        with_identifier,
        build_call("f", [x_reference]).expect("a named call")
    );
    assert_eq!(
        with_numbers,
        build_call("max", [build_literal(1), build_literal(2)]).expect("a named call")
    );
}

/// Test the call builder refuses an empty function name.
#[test]
fn build_call_rejects_an_empty_function_name() {
    let result = build_call("", [1]);

    assert_eq!(result, Err(ExpressionBuildError::EmptyFunctionName));
}

// =============================================================================
// Build errors
// =============================================================================

/// Test each build error's message.
#[rstest]
#[case::empty_piecewise(
    ExpressionBuildError::EmptyPiecewise,
    "a piecewise expression needs at least one case"
)]
#[case::condition_literal(
    ExpressionBuildError::NonBooleanConditionLiteral { case_index: 2 },
    "piecewise case 2 condition literal must be a boolean"
)]
#[case::empty_function_name(
    ExpressionBuildError::EmptyFunctionName,
    "a call expression needs a non-empty function name"
)]
#[case::too_few_operands(
    ExpressionBuildError::TooFewLogicalOperands { operation: BinaryOperation::LogicalOr, count: 1 },
    "logical_or requires at least two operands, but got 1"
)]
#[case::child_count(
    ExpressionBuildError::ChildCountMismatch { expected: 3, actual: 4 },
    "rebuilding the node needs 3 children, but got 4"
)]
fn expression_build_error_display_describes_the_failure(
    #[case] error: ExpressionBuildError,
    #[case] expected: &str,
) {
    let message = error.to_string();

    assert_eq!(message, expected);
}

/// Test a decimal text operand keeps its exact spelling in the tree.
#[test]
fn expression_text_operand_keeps_its_spelling() {
    let (_, x) = build_identifier("x");

    let built = &x * build_text_literal("0.10");

    let ExpressionKind::Binary(node) = built.kind() else {
        panic!("expected a binary node, got {built:?}");
    };
    assert_eq!(node.right(), &build_text_literal("0.1"));
    let ExpressionKind::Literal(literal) = node.right().kind() else {
        panic!("expected a literal, got {:?}", node.right());
    };
    assert_eq!(literal.to_string(), "0.10");
}
