//! Tests for building expressions: the operand types, the operator
//! overloads, the operation-named builder methods, the variadic builders,
//! and the build errors.
//!
//! Public API only (`fhy_core::expr`).

use crate::support::expression as expression_support;

use expression_support::{build_identifier, build_literal};
use fhy_core::expr::builtins::BuiltinFunction;
use fhy_core::expr::{
    BigInt, BinaryOperation, Callee, Expression, ExpressionKind, FunctionName, FunctionNameError,
    LiteralValue, LogicalExpression, LogicalOperation, PiecewiseError, RebuildError,
    UnaryOperation, UnknownNameError,
};
use fhy_core::identifier::Identifier;
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
    FloorMod,
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
            Self::FloorMod => BinaryOperation::FloorMod,
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
    fn apply(self, left: &Expression, right: impl Into<Expression>) -> Expression {
        match self {
            Self::Add => left + right,
            Self::Subtract => left - right,
            Self::Multiply => left * right,
            Self::Divide => left / right,
            Self::FloorDivide => left.floor_divide(right),
            Self::FloorMod => left.floor_mod(right),
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

/// Test negation, the positive builder and logical negation build unary
/// nodes with their operation.
#[rstest]
#[case::negate(UnaryOperation::Negate)]
#[case::positive(UnaryOperation::Positive)]
#[case::logical_not(UnaryOperation::LogicalNot)]
fn expression_unary_builders_produce_the_matching_unary_node(#[case] operation: UnaryOperation) {
    let operand = build_literal(5);

    let built = match operation {
        UnaryOperation::Negate => -&operand,
        UnaryOperation::Positive => operand.positive(),
        UnaryOperation::LogicalNot => !&operand,
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
        BinaryBuilder::FloorMod,
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
        BinaryBuilder::FloorMod,
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
/// expression it stands for; floor division, floor modulo and
/// exponentiation take one through `new_binary`.
#[rstest]
fn expression_arithmetic_builders_promote_a_plain_left_operand(
    #[values(
        BinaryOperation::Add,
        BinaryOperation::Subtract,
        BinaryOperation::Multiply,
        BinaryOperation::Divide,
        BinaryOperation::FloorDivide,
        BinaryOperation::FloorMod,
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

/// Build `left op right` with each arithmetic operator `+ - * /`, for an
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
        ]
    }};
}

/// Build every arithmetic operator's node from one left operand type and a
/// reference, each paired with the node `new_binary` builds.
type BuildWithEveryOperator = fn(&Expression) -> Vec<(Expression, Expression)>;

/// Test every non-expression operand type is accepted on the left of every
/// arithmetic operator, the same types that convert `Into<Expression>` on
/// the right, with an owned or a borrowed expression on the right.
#[rstest]
#[case::i64(|x: &Expression| build_with_every_operator!(3_i64, x))]
#[case::i32(|x: &Expression| build_with_every_operator!(3_i32, x))]
#[case::i128(|x: &Expression| build_with_every_operator!(3_i128, x))]
#[case::u32(|x: &Expression| build_with_every_operator!(3_u32, x))]
#[case::u64(|x: &Expression| build_with_every_operator!(3_u64, x))]
#[case::usize(|x: &Expression| build_with_every_operator!(3_usize, x))]
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

    assert_eq!(pairs.len(), 8);
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

/// Return the literal variant of `expression`, or `None` if it is not a
/// literal.
fn find_literal_variant(expression: &Expression) -> Option<std::mem::Discriminant<LiteralValue>> {
    match expression.kind() {
        ExpressionKind::Literal(literal) => Some(std::mem::discriminant(literal)),
        _ => None,
    }
}

/// Test every operand type lifts to the expression it stands for, a literal
/// operand keeping its variant.
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
#[case::integer_text_literal_value(|_: &Identifier, _: &Expression| (
    Expression::new_unary(
        UnaryOperation::Negate,
        LiteralValue::parse_text("5").expect("an integer text"),
    ),
    build_literal(LiteralValue::parse_text("5").expect("an integer text")),
))]
#[case::i64(|_: &Identifier, _: &Expression| (Expression::new_unary(UnaryOperation::Negate, 7_i64), build_literal(7)))]
#[case::i32(|_: &Identifier, _: &Expression| (Expression::new_unary(UnaryOperation::Negate, 7_i32), build_literal(7)))]
#[case::i128(|_: &Identifier, _: &Expression| (Expression::new_unary(UnaryOperation::Negate, 7_i128), build_literal(7)))]
#[case::u32(|_: &Identifier, _: &Expression| (Expression::new_unary(UnaryOperation::Negate, 7_u32), build_literal(7)))]
#[case::u64(|_: &Identifier, _: &Expression| (Expression::new_unary(UnaryOperation::Negate, 7_u64), build_literal(7)))]
#[case::usize(|_: &Identifier, _: &Expression| (Expression::new_unary(UnaryOperation::Negate, 7_usize), build_literal(7)))]
#[case::big_integer(|_: &Identifier, _: &Expression| (
    Expression::new_unary(UnaryOperation::Negate, build_big_operand()),
    build_literal(build_big_operand()),
))]
#[case::f64(|_: &Identifier, _: &Expression| (Expression::new_unary(UnaryOperation::Negate, 7.5_f64), build_literal(7.5)))]
fn expression_new_binary_lifts_every_operand_type(#[case] lift: LiftOperand) {
    let (identifier, reference) = build_identifier("x");

    let (built, operand) = lift(&identifier, &reference);

    let ExpressionKind::Unary(node) = built.kind() else {
        panic!("expected a unary node, got {built:?}");
    };
    assert_eq!(
        find_literal_variant(node.operand()),
        find_literal_variant(&operand)
    );
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

/// Test a numeric text becomes an operand through an explicit literal value,
/// normalized: an integer text to its integer, a decimal text to its
/// decimal, and the `Display` of either without the spelling.
#[rstest]
#[case::integer_text("05", "int", "5")]
#[case::decimal_text("1.50", "decimal", "1.5")]
fn expression_binary_builder_takes_a_parsed_text_operand(
    #[case] text: &str,
    #[case] expected_variant: &str,
    #[case] expected_display: &str,
) {
    let parsed = LiteralValue::parse_text(text).expect("a literal text");

    let built = build_literal(1) + parsed;

    let ExpressionKind::Binary(node) = built.kind() else {
        panic!("expected a binary node, got {built:?}");
    };
    let ExpressionKind::Literal(right) = node.right().kind() else {
        panic!("expected a literal right operand, got {:?}", node.right());
    };
    let variant = match right {
        LiteralValue::Int(_) => "int",
        LiteralValue::Decimal(_) => "decimal",
        LiteralValue::Bool(_) | LiteralValue::Float(_) => "other",
    };
    assert_eq!(variant, expected_variant);
    assert_eq!(right.to_string(), expected_display);
}

/// Test `floor_mod` builds the floor-modulo node over the receiver and its
/// operand, promoting a plain operand.
#[test]
fn expression_floor_mod_builds_a_floor_mod_node() {
    let (_, x) = build_identifier("x");

    let built = x.floor_mod(3);

    let ExpressionKind::Binary(node) = built.kind() else {
        panic!("expected a binary node, got {built:?}");
    };
    assert_eq!(node.operation(), BinaryOperation::FloorMod);
    assert!(Expression::ptr_eq(node.left(), &x));
    assert_eq!(node.right(), &build_literal(3));
}

/// Test `/` builds true division whatever the operand types, integers
/// included, never floor division.
#[rstest]
#[case::integers(|| build_literal(7) / 4)]
#[case::integer_on_the_left(|| 7 / build_literal(4))]
#[case::identifier_by_integer(|| build_identifier("x").1 / 4)]
#[case::floats(|| build_literal(7.0) / 4.0)]
fn expression_div_operator_builds_true_division(#[case] build: fn() -> Expression) {
    let built = build();

    let ExpressionKind::Binary(node) = built.kind() else {
        panic!("expected a binary node, got {built:?}");
    };
    assert_eq!(node.operation(), BinaryOperation::Divide);
}

// =============================================================================
// Logical builders
// =============================================================================

/// Build the logical node of `operation` over `operands` through the
/// builder named after it, `Expression::all` or `Expression::any`.
fn build_through_named_builder(
    operation: LogicalOperation,
    operands: impl IntoIterator<Item = Expression>,
) -> Expression {
    match operation {
        LogicalOperation::And => Expression::all(operands),
        LogicalOperation::Or => Expression::any(operands),
    }
}

/// Return the logical node `expression` refers to.
fn expect_logical(expression: &Expression) -> &LogicalExpression {
    let ExpressionKind::Logical(node) = expression.kind() else {
        panic!("expected a logical node, got {expression:?}");
    };
    node
}

/// Test `all`, `any` and `new_logical` build one logical node over every
/// operand in order, sharing each operand, whatever their number, and
/// wrapping a bare identifier operand in a reference to it.
#[rstest]
fn expression_all_and_any_build_one_node_over_every_operand(
    #[values(LogicalOperation::And, LogicalOperation::Or)] operation: LogicalOperation,
    #[values(2, 3, 4, 7)] count: usize,
) {
    let operands: Vec<Expression> = (0..count)
        .map(|index| build_identifier(&format!("p{index}")).1)
        .collect();

    let named = build_through_named_builder(operation, operands.iter().cloned());
    let general = Expression::new_logical(operation, &operands);

    let node = expect_logical(&named);
    assert_eq!(node.operation(), operation);
    assert_eq!(node.operands().len(), count);
    for (operand, expected) in node.operands().iter().zip(&operands) {
        assert!(Expression::ptr_eq(operand, expected));
    }
    assert_eq!(general, named);
}

/// Test a bare identifier operand becomes a reference to it.
#[test]
fn expression_all_wraps_identifier_operands_in_references() {
    let (first, first_reference) = build_identifier("a");
    let (second, second_reference) = build_identifier("b");

    let built = Expression::all([first, second]);

    assert_eq!(
        expect_logical(&built).operands(),
        [first_reference, second_reference]
    );
}

/// Test `and` and `or` build the two-operand logical node over the receiver
/// and their operand.
#[rstest]
#[case::and(LogicalOperation::And)]
#[case::or(LogicalOperation::Or)]
fn expression_and_and_or_build_a_two_operand_node(#[case] operation: LogicalOperation) {
    let (_, p) = build_identifier("p");
    let (_, q) = build_identifier("q");

    let built = match operation {
        LogicalOperation::And => p.and(&q),
        LogicalOperation::Or => p.or(&q),
    };

    let node = expect_logical(&built);
    assert_eq!(node.operation(), operation);
    assert!(Expression::ptr_eq(&node.operands()[0], &p));
    assert!(Expression::ptr_eq(&node.operands()[1], &q));
}

/// Test `all` of no operand is the literal `true`, `any` of none the literal
/// `false`, and either of one operand that operand's own handle.
#[rstest]
fn expression_all_and_any_of_zero_or_one_operand(
    #[values(LogicalOperation::And, LogicalOperation::Or)] operation: LogicalOperation,
) {
    let (_, p) = build_identifier("p");

    let of_nothing = build_through_named_builder(operation, []);
    let of_one = build_through_named_builder(operation, [p.clone()]);
    let general_of_one = Expression::new_logical(operation, [&p]);

    let identity = operation == LogicalOperation::And;
    assert_eq!(of_nothing, build_literal(identity));
    assert!(Expression::ptr_eq(&of_one, &p));
    assert!(Expression::ptr_eq(&general_of_one, &p));
}

/// Test `all` of nothing is `true` and `any` of nothing is `false`.
#[test]
fn expression_all_and_any_of_nothing_are_true_and_false() {
    let no_operands: [Expression; 0] = [];

    let all = Expression::all(no_operands.clone());
    let any = Expression::any(no_operands);

    assert!(matches!(
        all.kind(),
        ExpressionKind::Literal(LiteralValue::Bool(true))
    ));
    assert!(matches!(
        any.kind(),
        ExpressionKind::Literal(LiteralValue::Bool(false))
    ));
}

/// Test `all` of one operand returns that operand's handle, building no node.
#[test]
fn expression_all_of_one_operand_is_that_operand() {
    let (_, p) = build_identifier("p");
    let comparison = p.less(3);

    let built = Expression::all([&comparison]);

    assert!(Expression::ptr_eq(&built, &comparison));
}

/// Test `and` never splices a nested conjunction's operands into the new
/// node: `x.and(y.and(z))` keeps the inner node as its second operand.
#[test]
fn expression_and_does_not_flatten_a_nested_conjunction() {
    let (_, x) = build_identifier("x");
    let (_, y) = build_identifier("y");
    let (_, z) = build_identifier("z");
    let inner = y.and(&z);

    let built = x.and(&inner);

    let node = expect_logical(&built);
    assert_eq!(node.operands().len(), 2);
    assert!(Expression::ptr_eq(&node.operands()[1], &inner));
    assert_ne!(built, Expression::all([&x, &y, &z]));
}

/// Test `all` over ten thousand comparisons is one logical node with every
/// comparison as an operand, not a chain ten thousand levels deep.
#[test]
fn expression_all_of_ten_thousand_comparisons_is_one_logical_node() {
    let (_, x) = build_identifier("x");
    let comparisons: Vec<Expression> = (0..10_000).map(|bound| x.less(bound)).collect();

    let built = Expression::all(&comparisons);

    let node = expect_logical(&built);
    assert_eq!(node.operation(), LogicalOperation::And);
    assert_eq!(node.operands().len(), 10_000);
    assert_eq!(node.operands(), comparisons.as_slice());
}

/// Test logical negation builds a negation node, and wraps a bare
/// identifier through `new_unary`.
#[test]
fn expression_logical_not_wraps_the_operand() {
    let operand = build_literal(true);
    let (identifier, reference) = build_identifier("a");

    let negated = !&operand;
    let negated_identifier = Expression::new_unary(UnaryOperation::LogicalNot, identifier);

    assert_eq!(
        negated,
        Expression::new_unary(UnaryOperation::LogicalNot, &operand)
    );
    assert_eq!(negated_identifier, !&reference);
}

/// Test `!` builds a logical negation over an owned or a borrowed
/// expression, the borrowed operand shared rather than copied.
#[test]
fn expression_not_operator_builds_logical_not() {
    let (_, p) = build_identifier("p");

    let borrowed = !&p;
    let owned = !p.clone();

    let ExpressionKind::Unary(node) = borrowed.kind() else {
        panic!("expected a unary node, got {borrowed:?}");
    };
    assert_eq!(node.operation(), UnaryOperation::LogicalNot);
    assert!(Expression::ptr_eq(node.operand(), &p));
    assert_eq!(owned, borrowed);
}

/// Test a conjunction keeps both bounds of `0 <= c <= 5`.
#[test]
fn expression_all_keeps_both_bounds_of_a_range() {
    let (_, c) = build_identifier("c");
    let lower = Expression::new_binary(BinaryOperation::LessEqual, 0, &c);
    let upper = c.less_equal(5);

    let conjunction = Expression::all([&lower, &upper]);

    let node = expect_logical(&conjunction);
    assert_eq!(node.operation(), LogicalOperation::And);
    assert_eq!(node.operands(), [lower, upper]);
}

// =============================================================================
// Piecewise and call builders
// =============================================================================

/// Test the piecewise builder keeps expression operands as they are.
#[test]
fn expression_piecewise_wraps_expression_operands_directly() {
    let (condition, value, otherwise) = (build_literal(true), build_literal(1), build_literal(2));

    let built = Expression::piecewise([(&condition, &value)], &otherwise).expect("one case");

    let ExpressionKind::Piecewise(node) = built.kind() else {
        panic!("expected a piecewise node, got {built:?}");
    };
    assert!(Expression::ptr_eq(&node.cases()[0].0, &condition));
    assert!(Expression::ptr_eq(&node.cases()[0].1, &value));
    assert!(Expression::ptr_eq(node.otherwise(), &otherwise));
}

/// Test the piecewise builder keeps several cases in order.
#[test]
fn expression_piecewise_keeps_multiple_cases_in_declared_order() {
    let (first_condition, second_condition) = (build_literal(true), build_literal(false));
    let (first_value, second_value) = (build_literal(1), build_literal(2));

    let built = Expression::piecewise(
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
fn expression_piecewise_coerces_identifiers_and_numbers() {
    let (flag, flag_reference) = build_identifier("flag");
    let (x, x_reference) = build_identifier("x");
    let (fallback, fallback_reference) = build_identifier("fallback");

    let with_identifiers = Expression::piecewise([(flag, x)], fallback).expect("one case");
    let with_numbers = Expression::piecewise([(build_literal(true), 5)], 10).expect("one case");

    let expected_identifiers =
        Expression::piecewise([(&flag_reference, &x_reference)], &fallback_reference)
            .expect("one case");
    let expected_numbers =
        Expression::piecewise([(build_literal(true), build_literal(5))], build_literal(10))
            .expect("one case");
    assert_eq!(with_identifiers, expected_identifiers);
    assert_eq!(with_numbers, expected_numbers);
}

/// Test the piecewise builder refuses a number as a case condition.
#[test]
fn expression_piecewise_rejects_a_numeric_condition() {
    let result = Expression::piecewise([(1, 5)], 10);

    assert_eq!(
        result,
        Err(PiecewiseError::NonBooleanConditionLiteral { case_index: 0 })
    );
}

/// Test the piecewise builder refuses no cases.
#[test]
fn expression_piecewise_rejects_zero_cases() {
    let result = Expression::piecewise(Vec::<(Expression, Expression)>::new(), 0);

    assert_eq!(result, Err(PiecewiseError::NoCases));
}

/// Test Boolean literal value and otherwise operands pass through as given.
#[test]
fn expression_piecewise_keeps_explicit_boolean_literal_operands() {
    let value = build_literal(true);
    let otherwise = build_literal(false);

    let built =
        Expression::piecewise([(build_literal(true), &value)], &otherwise).expect("one case");

    let ExpressionKind::Piecewise(node) = built.kind() else {
        panic!("expected a piecewise node, got {built:?}");
    };
    assert!(Expression::ptr_eq(&node.cases()[0].1, &value));
    assert!(Expression::ptr_eq(node.otherwise(), &otherwise));
}

/// Test an equality built with `equals` is a valid case condition.
#[test]
fn expression_piecewise_accepts_an_equals_condition() {
    let (_, x) = build_identifier("x");

    let built =
        Expression::piecewise([(x.equals(0), 7), (x.less_equal(0), 8)], 9).expect("two cases");

    let ExpressionKind::Piecewise(node) = built.kind() else {
        panic!("expected a piecewise node, got {built:?}");
    };
    assert_eq!(node.cases()[0].0, x.equals(0));
}

/// Test the call builder carries the callee and the argument nodes.
#[test]
fn expression_call_returns_a_call_with_callee_and_arguments() {
    let (first, second) = (build_literal(1), build_literal(2));

    let built = Expression::call(BuiltinFunction::Max, [&first, &second]);

    let ExpressionKind::Call(node) = built.kind() else {
        panic!("expected a call node, got {built:?}");
    };
    assert_eq!(node.callee(), &Callee::Builtin(BuiltinFunction::Max));
    assert_eq!(node.callee().name(), "max");
    assert_eq!(node.arguments().len(), 2);
    assert!(Expression::ptr_eq(&node.arguments()[0], &first));
    assert!(Expression::ptr_eq(&node.arguments()[1], &second));
}

/// Test the call builder accepts zero arguments and a named callee.
#[test]
fn expression_call_supports_zero_arguments() {
    let nullary = FunctionName::try_new("nullary").expect("a user function name");

    let built = Expression::call(nullary.clone(), Vec::<Expression>::new());

    let ExpressionKind::Call(node) = built.kind() else {
        panic!("expected a call node, got {built:?}");
    };
    assert_eq!(node.callee(), &Callee::Named(nullary));
    assert!(node.arguments().is_empty());
}

/// Test the call builder wraps identifier and number arguments.
#[test]
fn expression_call_coerces_identifiers_and_numbers() {
    let (x, x_reference) = build_identifier("x");
    let f = FunctionName::try_new("f").expect("a user function name");

    let with_identifier = Expression::call(f.clone(), [x]);
    let with_numbers = Expression::call(BuiltinFunction::Max, [1, 2]);

    assert_eq!(with_identifier, Expression::call(f, [x_reference]));
    assert_eq!(
        with_numbers,
        Expression::call(BuiltinFunction::Max, [build_literal(1), build_literal(2)])
    );
}

/// Test parsing a callee resolves a built-in function's name to the
/// built-in and any other name to a named function.
#[rstest]
#[case::composed("max", Callee::Builtin(BuiltinFunction::Max))]
#[case::native("log10", Callee::Builtin(BuiltinFunction::Log10))]
#[case::snake_case("clamp_symmetric", Callee::Builtin(BuiltinFunction::ClampSymmetric))]
#[case::user("softplus", Callee::Named(FunctionName::try_new("softplus").unwrap()))]
#[case::other_case("Max", Callee::Named(FunctionName::try_new("Max").unwrap()))]
#[case::constant("pi", Callee::Named(FunctionName::try_new("pi").unwrap()))]
fn callee_from_str_resolves_builtin_names(#[case] name: &str, #[case] expected: Callee) {
    let callee: Callee = name.parse().expect("a non-empty name");

    assert_eq!(callee, expected);
    assert_eq!(callee.name(), name);
    assert_eq!(callee.to_string(), name);
}

/// Test parsing an empty callee name is refused.
#[test]
fn callee_from_str_refuses_an_empty_name() {
    let result = "".parse::<Callee>();

    assert_eq!(result, Err(FunctionNameError::Empty));
}

/// Test a function name refuses every built-in function's name, naming the
/// built-in.
#[test]
fn function_name_refuses_a_builtin_name() {
    for function in BuiltinFunction::iter() {
        let result = FunctionName::try_new(function.name());

        assert_eq!(result, Err(FunctionNameError::Builtin(function)));
    }
}

/// Test a function name refuses the empty name and every built-in name, and
/// accepts any other name as given.
#[rstest]
#[case::empty("", Err(FunctionNameError::Empty))]
#[case::builtin("sqrt", Err(FunctionNameError::Builtin(BuiltinFunction::Sqrt)))]
#[case::user("f", Ok("f"))]
#[case::spaced(" max", Ok(" max"))]
fn function_name_try_new_rejects_empty_and_builtin_names(
    #[case] name: &str,
    #[case] expected: Result<&str, FunctionNameError>,
) {
    let result = FunctionName::try_new(name);

    assert_eq!(
        result.as_ref().map(FunctionName::as_str),
        expected.as_deref()
    );
    if let Ok(accepted) = result {
        assert_eq!(accepted.to_string(), name);
    }
}

// =============================================================================
// Conversions
// =============================================================================

/// Test each number type converts into the integer or float literal of its
/// value, and an identifier or expression, owned or borrowed, into the node
/// it stands for.
#[rstest]
#[case::i32(Expression::from(-3_i32), build_literal(LiteralValue::Int(BigInt::from(-3))))]
#[case::i64(Expression::from(-3_i64), build_literal(LiteralValue::Int(BigInt::from(-3))))]
#[case::i128(
    Expression::from(i128::MIN),
    build_literal(LiteralValue::Int(BigInt::from(i128::MIN)))
)]
#[case::u32(
    Expression::from(u32::MAX),
    build_literal(LiteralValue::Int(BigInt::from(u32::MAX)))
)]
#[case::u64(
    Expression::from(u64::MAX),
    build_literal(LiteralValue::Int(BigInt::from(u64::MAX)))
)]
#[case::usize(
    Expression::from(7_usize),
    build_literal(LiteralValue::Int(BigInt::from(7)))
)]
#[case::big_integer(
    Expression::from(build_big_operand()),
    build_literal(LiteralValue::Int(build_big_operand()))
)]
#[case::f64(
    Expression::from(f64::NAN),
    build_literal(LiteralValue::Float(f64::NAN))
)]
#[case::literal_value(Expression::literal(true), Expression::from(LiteralValue::Bool(true)))]
fn expression_from_every_primitive_builds_its_literal(
    #[case] built: Expression,
    #[case] expected: Expression,
) {
    assert!(matches!(built.kind(), ExpressionKind::Literal(_)));
    assert_eq!(built, expected);
}

/// Test a borrowed identifier converts into a reference to it, and a
/// borrowed expression into a handle sharing its node.
#[test]
fn expression_from_a_borrowed_identifier_or_expression_builds_its_node() {
    let (x, reference) = build_identifier("x");

    let from_identifier = Expression::from(&x);
    let from_expression = Expression::from(&reference);

    assert_eq!(from_identifier, reference);
    assert!(Expression::ptr_eq(&from_expression, &reference));
}

// =============================================================================
// Build errors
// =============================================================================

/// Test each piecewise error's message.
#[rstest]
#[case::no_cases(PiecewiseError::NoCases, "piecewise has no cases")]
#[case::condition_literal(
    PiecewiseError::NonBooleanConditionLiteral { case_index: 2 },
    "condition of piecewise case 2 is a non-boolean literal"
)]
fn piecewise_error_display_describes_the_failure(
    #[case] error: PiecewiseError,
    #[case] expected: &str,
) {
    let message = error.to_string();

    assert_eq!(message, expected);
}

/// Test each rebuild error's message, which names the piecewise failure
/// only through its source.
#[rstest]
#[case::child_count(
    RebuildError::ChildCount { expected: 3, actual: 4 },
    "expected 3 children, got 4"
)]
#[case::piecewise(RebuildError::Piecewise(PiecewiseError::NoCases), "invalid piecewise")]
fn rebuild_error_display_describes_the_failure(
    #[case] error: RebuildError,
    #[case] expected: &str,
) {
    let message = error.to_string();

    assert_eq!(message, expected);
}

/// Test each function-name error's message.
#[rstest]
#[case::empty(FunctionNameError::Empty, "function name is empty")]
#[case::builtin(
    FunctionNameError::Builtin(BuiltinFunction::ClampSymmetric),
    "function name `clamp_symmetric` is a built-in function"
)]
fn function_name_error_display_describes_the_failure(
    #[case] error: FunctionNameError,
    #[case] expected: &str,
) {
    let message = error.to_string();

    assert_eq!(message, expected);
}

/// Test the unknown-name error's message names the enum and the name.
#[test]
fn unknown_name_error_display_describes_the_failure() {
    let error: UnknownNameError = "plus"
        .parse::<BinaryOperation>()
        .expect_err("no binary operation is named plus");

    assert_eq!(error.to_string(), "unknown binary operation `plus`");
}

/// Test each refusal of the piecewise builder is a piecewise error of the
/// variant naming its cause.
#[test]
fn expression_piecewise_errors_are_piecewise_errors() {
    let no_cases = Expression::piecewise(Vec::<(Expression, Expression)>::new(), 0);
    let numeric_condition =
        Expression::piecewise([(build_literal(true), 1), (build_literal(2), 3)], 0);

    let causes = [no_cases, numeric_condition].map(|result| match result {
        Err(PiecewiseError::NoCases) => "no cases".to_owned(),
        Err(PiecewiseError::NonBooleanConditionLiteral { case_index }) => {
            format!("condition {case_index}")
        }
        Err(other) => format!("another piecewise error: {other}"),
        Ok(expression) => format!("built {expression}"),
    });

    assert_eq!(causes, ["no cases", "condition 1"].map(str::to_owned));
}
