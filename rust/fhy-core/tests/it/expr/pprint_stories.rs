//! Tests for printing expressions: the default options, both notations for
//! every node kind and operation, literal and identifier text, identifier
//! ids, piecewise and call layout, shared subtrees, and trees thousands of
//! levels deep printed on a small thread stack.

use crate::support::expression as expression_support;
use crate::support::stack as stack_support;

use expression_support::{
    build_callee, build_decimal_literal, build_deep_conjunction, build_deep_sum, build_identifier,
    build_literal,
};
use fhy_core::expr::{
    BigInt, BinaryOperation, Expression, FormatOptions, IdentifierStyle, LiteralValue,
    LogicalOperation, Notation, UnaryOperation,
};
use fhy_core::identifier::Identifier;
use rstest::rstest;
use stack_support::{SMALL_STACK_DEPTH, run_on_small_stack};

fn build_id_options(notation: Notation) -> FormatOptions {
    FormatOptions::default()
        .with_notation(notation)
        .with_identifier_style(IdentifierStyle::NameHintWithId)
}

/// Return `expression` printed symbolically with name hints only.
fn format_symbolic(expression: &Expression) -> String {
    expression
        .display(FormatOptions::default().with_notation(Notation::Symbolic))
        .to_string()
}

/// Return `expression` printed functionally with name hints only.
fn format_functional(expression: &Expression) -> String {
    expression
        .display(FormatOptions::default().with_notation(Notation::Functional))
        .to_string()
}

/// Return `expression` printed in `notation` with identifier ids.
fn format_with_ids(expression: &Expression, notation: Notation) -> String {
    expression.display(build_id_options(notation)).to_string()
}

/// Return the `name::id` text expected for `identifier` when ids are shown.
fn write_name_hint_with_id(identifier: &Identifier) -> String {
    format!("{}::{}", identifier.name_hint(), identifier.id())
}

/// Return the tree `build` makes from references to fresh identifiers named
/// `x` and `y`.
fn build_over_x_and_y(build: impl FnOnce(&Expression, &Expression) -> Expression) -> Expression {
    let (_, x) = build_identifier("x");
    let (_, y) = build_identifier("y");
    build(&x, &y)
}

/// Return `-(-(-leaf))`, `depth` negations deep.
fn build_deep_negation(leaf: &Expression, depth: usize) -> Expression {
    (0..depth).fold(leaf.clone(), |tree, _| -tree)
}

/// Return `{1 if condition; {1 if condition; ... leaf otherwise} otherwise}`,
/// `depth` piecewise nodes deep along the otherwise branches.
fn build_deep_piecewise_in_otherwise(
    condition: &Expression,
    leaf: &Expression,
    depth: usize,
) -> Expression {
    (0..depth).fold(leaf.clone(), |tree, _| {
        Expression::piecewise([(condition, build_literal(1))], tree)
            .expect("an identifier condition is accepted")
    })
}

/// Return `{1 if {1 if ... leaf ...; 0 otherwise}; 0 otherwise}`, `depth`
/// piecewise nodes deep along the case conditions.
fn build_deep_piecewise_in_condition(leaf: &Expression, depth: usize) -> Expression {
    (0..depth).fold(leaf.clone(), |tree, _| {
        Expression::piecewise([(tree, build_literal(1))], build_literal(0))
            .expect("a piecewise condition is accepted")
    })
}

/// Return `f(1, f(1, ... leaf))`, `depth` calls deep along the last argument.
fn build_deep_call_in_last_argument(leaf: &Expression, depth: usize) -> Expression {
    (0..depth).fold(leaf.clone(), |tree, _| {
        Expression::call(build_callee("f"), [build_literal(1), tree])
    })
}

/// Return `f(f(... leaf, 2), 2)`, `depth` calls deep along the first argument.
fn build_deep_call_in_first_argument(leaf: &Expression, depth: usize) -> Expression {
    (0..depth).fold(leaf.clone(), |tree, _| {
        Expression::call(build_callee("f"), [tree, build_literal(2)])
    })
}

// =============================================================================
// Options
// =============================================================================

#[test]
fn format_options_default_is_symbolic_with_name_hints() {
    let default = FormatOptions::default();

    assert_eq!(default.notation(), Notation::Symbolic);
    assert_eq!(default.identifier_style(), IdentifierStyle::NameHint);
}

#[rstest]
#[case::symbolic(Notation::Symbolic)]
#[case::functional(Notation::Functional)]
fn format_options_with_notation_sets_only_the_notation(#[case] notation: Notation) {
    let base = FormatOptions::default().with_identifier_style(IdentifierStyle::NameHintWithId);

    let options = base.with_notation(notation);

    assert_eq!(options.notation(), notation);
    assert_eq!(options.identifier_style(), IdentifierStyle::NameHintWithId);
}

#[rstest]
#[case::name_hint(IdentifierStyle::NameHint)]
#[case::name_hint_with_id(IdentifierStyle::NameHintWithId)]
fn format_options_with_identifier_style_sets_only_the_style(#[case] style: IdentifierStyle) {
    let base = FormatOptions::default().with_notation(Notation::Functional);

    let options = base.with_identifier_style(style);

    assert_eq!(options.identifier_style(), style);
    assert_eq!(options.notation(), Notation::Functional);
}

#[test]
fn format_expression_default_options_write_name_hint_only() {
    let (_, name_only) = build_identifier("name_only");

    let text = name_only.display(FormatOptions::default()).to_string();

    assert_eq!(text, "name_only");
}

#[test]
fn format_expression_default_options_write_symbolic_notation() {
    let sum = build_literal(1) + 2;

    let text = sum.display(FormatOptions::default()).to_string();

    assert_eq!(text, "(1 + 2)");
}

#[rstest]
#[case::identifier(|| build_identifier("x").1)]
#[case::sum(|| build_identifier("x").1 + 1)]
#[case::logical_not(|| !build_literal(true))]
#[case::piecewise(|| {
    Expression::piecewise([(build_identifier("p").1, 1)], build_literal(2.5))
        .expect("an identifier condition is accepted")
})]
#[case::call(|| Expression::call(build_callee("f"), [build_identifier("x").1, build_literal(1)]))]
fn expression_display_equals_display_with_default_options(#[case] build: fn() -> Expression) {
    let expression = build();

    let own = expression.to_string();
    let with_default_options = expression.display(FormatOptions::default()).to_string();

    assert_eq!(own, with_default_options);
}

#[test]
fn expression_display_writes_into_the_surrounding_format() {
    let (x_identifier, x) = build_identifier("x");
    let sum = &x + 1;

    let text = format!(
        "[{}|{sum}]",
        sum.display(build_id_options(Notation::Functional))
    );

    assert_eq!(text, format!("[(add x::{} 1)|(x + 1)]", x_identifier.id()));
}

// =============================================================================
// Notations
// =============================================================================

#[rstest]
#[case::float(build_literal(4.5), "4.5")]
#[case::decimal(build_decimal_literal("0.1"), "0.1")]
#[case::identifier(build_identifier("baz").1, "baz")]
#[case::logical_not(!build_literal(true), "(!true)")]
#[case::product(build_decimal_literal("3.14") * 10.5, "(3.14 * 10.5)")]
fn format_expression_writes_symbolic_notation(
    #[case] expression: Expression,
    #[case] expected: &str,
) {
    let text = format_symbolic(&expression);

    assert_eq!(text, expected);
}

#[rstest]
#[case::integer(build_literal(5), "5")]
#[case::identifier(build_identifier("test_identifier").1, "test_identifier")]
#[case::negation(-build_literal(5), "(negate 5)")]
#[case::sum(build_literal(5) + 10, "(add 5 10)")]
#[case::quotient_of_negation(-build_literal(5) / 10, "(divide (negate 5) 10)")]
fn format_expression_writes_functional_notation(
    #[case] expression: Expression,
    #[case] expected: &str,
) {
    let text = format_functional(&expression);

    assert_eq!(text, expected);
}

#[rstest]
#[case::negate(UnaryOperation::Negate, "(-x)", "(negate x)")]
#[case::positive(UnaryOperation::Positive, "(+x)", "(positive x)")]
#[case::logical_not(UnaryOperation::LogicalNot, "(!x)", "(logical_not x)")]
fn format_expression_writes_each_unary_operation(
    #[case] operation: UnaryOperation,
    #[case] symbolic: &str,
    #[case] functional: &str,
) {
    let (_, x) = build_identifier("x");
    let node = Expression::new_unary(operation, &x);

    let texts = (format_symbolic(&node), format_functional(&node));

    assert_eq!(texts, (symbolic.to_owned(), functional.to_owned()));
}

#[rstest]
#[case::add(BinaryOperation::Add, "(x + 2)", "(add x 2)")]
#[case::subtract(BinaryOperation::Subtract, "(x - 2)", "(subtract x 2)")]
#[case::multiply(BinaryOperation::Multiply, "(x * 2)", "(multiply x 2)")]
#[case::divide(BinaryOperation::Divide, "(x / 2)", "(divide x 2)")]
#[case::floor_divide(BinaryOperation::FloorDivide, "(x // 2)", "(floor_divide x 2)")]
#[case::floor_mod(BinaryOperation::FloorMod, "(x % 2)", "(floor_mod x 2)")]
#[case::power(BinaryOperation::Power, "(x ** 2)", "(power x 2)")]
#[case::equal(BinaryOperation::Equal, "(x == 2)", "(equal x 2)")]
#[case::not_equal(BinaryOperation::NotEqual, "(x != 2)", "(not_equal x 2)")]
#[case::less(BinaryOperation::Less, "(x < 2)", "(less x 2)")]
#[case::less_equal(BinaryOperation::LessEqual, "(x <= 2)", "(less_equal x 2)")]
#[case::greater(BinaryOperation::Greater, "(x > 2)", "(greater x 2)")]
#[case::greater_equal(BinaryOperation::GreaterEqual, "(x >= 2)", "(greater_equal x 2)")]
fn format_expression_writes_each_binary_operation(
    #[case] operation: BinaryOperation,
    #[case] symbolic: &str,
    #[case] functional: &str,
) {
    let (_, x) = build_identifier("x");
    let node = Expression::new_binary(operation, &x, 2);

    let texts = (format_symbolic(&node), format_functional(&node));

    assert_eq!(texts, (symbolic.to_owned(), functional.to_owned()));
}

/// Test every unary and binary node is parenthesized, so the text shows the
/// nesting without precedence or associativity rules.
#[rstest]
#[case::sum_of_product(build_over_x_and_y(|x, y| x + (y * 2)), "(x + (y * 2))")]
#[case::product_of_sum(build_over_x_and_y(|x, y| (x + y) * 2), "((x + y) * 2)")]
#[case::right_nested_difference(build_over_x_and_y(|x, y| x - (y - 1)), "(x - (y - 1))")]
#[case::left_nested_difference(build_over_x_and_y(|x, y| (x - y) - 1), "((x - y) - 1)")]
#[case::right_nested_power(build_over_x_and_y(|x, y| x.power(y.power(2))), "(x ** (y ** 2))")]
#[case::negated_sum(build_over_x_and_y(|x, _| -(x + 1)), "(-(x + 1))")]
fn format_expression_parenthesizes_every_operation(
    #[case] expression: Expression,
    #[case] expected: &str,
) {
    let text = format_symbolic(&expression);

    assert_eq!(text, expected);
}

#[test]
fn format_expression_writes_a_logical_node_with_every_operand() {
    let (_, x) = build_identifier("x");
    let (_, y) = build_identifier("y");
    let (_, z) = build_identifier("z");
    let conjunction = Expression::all([x, y, z]);

    let texts = (
        format_symbolic(&conjunction),
        format_functional(&conjunction),
    );

    assert_eq!(
        texts,
        ("(x && y && z)".to_owned(), "(and x y z)".to_owned())
    );
}

/// Test a logical node prints in both notations, and in its own parentheses
/// when nested.
#[rstest]
#[case::and_of_two(LogicalOperation::And, 2, "(p0 && p1)", "(and p0 p1)")]
#[case::or_of_two(LogicalOperation::Or, 2, "(p0 || p1)", "(or p0 p1)")]
#[case::or_of_four(LogicalOperation::Or, 4, "(p0 || p1 || p2 || p3)", "(or p0 p1 p2 p3)")]
fn expression_display_writes_a_logical_node_in_both_notations(
    #[case] operation: LogicalOperation,
    #[case] count: usize,
    #[case] symbolic: &str,
    #[case] functional: &str,
) {
    let operands: Vec<Expression> = (0..count)
        .map(|index| build_identifier(&format!("p{index}")).1)
        .collect();
    let node = Expression::new_logical(operation, operands);
    let nested = Expression::any([node.clone(), build_literal(false)]);

    let texts = (format_symbolic(&node), format_functional(&node));
    let nested_text = format_symbolic(&nested);

    assert_eq!(texts, (symbolic.to_owned(), functional.to_owned()));
    assert_eq!(nested_text, format!("({symbolic} || false)"));
}

// =============================================================================
// Literals
// =============================================================================

#[rstest]
#[case::true_value(LiteralValue::from(true), "true")]
#[case::false_value(LiteralValue::from(false), "false")]
#[case::zero(LiteralValue::from(0), "0")]
#[case::negative_integer(LiteralValue::from(-1), "-1")]
#[case::big_integer(
    LiteralValue::from(BigInt::from(10).pow(30)),
    "1000000000000000000000000000000"
)]
#[case::integral_float(LiteralValue::from(1.0), "1")]
#[case::tenth(LiteralValue::from(0.1), "0.1")]
#[case::third(LiteralValue::from(1.0 / 3.0), "0.3333333333333333")]
#[case::large_float(LiteralValue::from(1e15), "1000000000000000")]
#[case::larger_float(LiteralValue::from(1e16), "10000000000000000")]
#[case::long_mantissa(LiteralValue::from(1.234_567_890_123_456_8e17), "123456789012345680")]
#[case::float_1e22(LiteralValue::from(1e22), "10000000000000000000000")]
#[case::small_float(LiteralValue::from(0.0001), "0.0001")]
#[case::smaller_float(LiteralValue::from(1e-7), "0.0000001")]
#[case::negative_zero(LiteralValue::from(-0.0), "-0")]
#[case::nan(LiteralValue::from(f64::NAN), "NaN")]
#[case::infinity(LiteralValue::from(f64::INFINITY), "inf")]
#[case::negative_infinity(LiteralValue::from(f64::NEG_INFINITY), "-inf")]
#[case::padded_integer_text(LiteralValue::parse_text("05").expect("an integer text"), "5")]
#[case::decimal_trailing_zero(LiteralValue::parse_text("1.50").expect("a decimal text"), "1.5")]
#[case::decimal_no_whole_part(LiteralValue::parse_text(".5").expect("a decimal text"), "0.5")]
#[case::decimal_no_fraction(LiteralValue::parse_text("1.").expect("a decimal text"), "1")]
#[case::decimal_hundred(LiteralValue::parse_text("100.0").expect("a decimal text"), "100")]
fn format_expression_writes_a_literal_as_its_display_text(
    #[case] value: LiteralValue,
    #[case] expected: &str,
) {
    let literal = Expression::from(value);

    let texts = (format_symbolic(&literal), format_functional(&literal));

    assert_eq!(texts, (expected.to_owned(), expected.to_owned()));
}

#[rstest]
#[case::negate(UnaryOperation::Negate, "(--1)", "(negate -1)")]
#[case::positive(UnaryOperation::Positive, "(+-1)", "(positive -1)")]
#[case::logical_not(UnaryOperation::LogicalNot, "(!-1)", "(logical_not -1)")]
fn format_expression_writes_unary_of_negative_literal_with_two_signs(
    #[case] operation: UnaryOperation,
    #[case] symbolic: &str,
    #[case] functional: &str,
) {
    let expression = Expression::new_unary(operation, build_literal(-1));

    let texts = (format_symbolic(&expression), format_functional(&expression));

    assert_eq!(texts, (symbolic.to_owned(), functional.to_owned()));
}

/// Test a negative literal operand of a binary node is written bare, with
/// its sign, so a negative literal base reads `(-1 ** 2)` while a negation
/// of a positive base reads `((-1) ** 2)`.
#[rstest]
#[case::subtract_negative_integer(
    || Expression::new_binary(BinaryOperation::Subtract, build_identifier("x").1, -1),
    "(x - -1)",
    "(subtract x -1)"
)]
#[case::add_negative_float(
    || Expression::new_binary(BinaryOperation::Add, build_identifier("x").1, -1.5),
    "(x + -1.5)",
    "(add x -1.5)"
)]
#[case::multiply_negative_zero(
    || Expression::new_binary(BinaryOperation::Multiply, build_identifier("x").1, -0.0),
    "(x * -0)",
    "(multiply x -0)"
)]
#[case::negative_literal_base(
    || Expression::new_binary(BinaryOperation::Power, -1, 2),
    "(-1 ** 2)",
    "(power -1 2)"
)]
#[case::negated_base(
    || Expression::new_binary(
        BinaryOperation::Power,
        Expression::new_unary(UnaryOperation::Negate, 1),
        2,
    ),
    "((-1) ** 2)",
    "(power (negate 1) 2)"
)]
#[case::negated_power(
    || Expression::new_unary(
        UnaryOperation::Negate,
        Expression::new_binary(BinaryOperation::Power, 1, 2),
    ),
    "(-(1 ** 2))",
    "(negate (power 1 2))"
)]
fn format_expression_writes_a_negative_literal_operand_bare(
    #[case] build: fn() -> Expression,
    #[case] symbolic: &str,
    #[case] functional: &str,
) {
    let expression = build();

    let texts = (format_symbolic(&expression), format_functional(&expression));

    assert_eq!(texts, (symbolic.to_owned(), functional.to_owned()));
}

#[test]
fn format_expression_writes_int_float_and_decimal_one_alike() {
    let integer = build_literal(1);
    let float = build_literal(1.0);
    let decimal = build_decimal_literal("1.0");

    let texts = [&integer, &float, &decimal].map(format_symbolic);

    assert_ne!(integer, float);
    assert_ne!(integer, decimal);
    assert_eq!(texts, ["1", "1", "1"].map(str::to_owned));
}

// =============================================================================
// Identifiers
// =============================================================================

#[test]
fn format_expression_with_ids_writes_name_hint_and_id() {
    let (foo, reference) = build_identifier("foo");

    let text = format_with_ids(&reference, Notation::Symbolic);

    assert_eq!(text, format!("foo::{}", foo.id()));
}

#[test]
fn format_expression_with_ids_applies_in_functional_notation() {
    let (x, reference) = build_identifier("x");
    let sum = &reference + 1;

    let text = format_with_ids(&sum, Notation::Functional);

    assert_eq!(text, format!("(add {} 1)", write_name_hint_with_id(&x)));
}

#[test]
fn format_expression_with_ids_reaches_piecewise_conditions() {
    let (condition, reference) = build_identifier("cond");
    let piecewise =
        Expression::piecewise([(reference, 1)], 0).expect("an identifier condition is accepted");

    let text = format_with_ids(&piecewise, Notation::Symbolic);

    assert_eq!(
        text,
        format!(
            "{{1 if {}; 0 otherwise}}",
            write_name_hint_with_id(&condition)
        )
    );
}

#[test]
fn format_expression_with_ids_reaches_call_arguments() {
    let (argument, reference) = build_identifier("arg");
    let call = Expression::call(build_callee("nested_show_id"), [reference]);

    let text = format_with_ids(&call, Notation::Symbolic);

    assert_eq!(
        text,
        format!("nested_show_id({})", write_name_hint_with_id(&argument))
    );
}

#[test]
fn format_expression_with_ids_reaches_every_node_kind() {
    let (p, p_reference) = build_identifier("p");
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let tree = Expression::piecewise(
        [(!&p_reference, -&x_reference)],
        Expression::call(build_callee("f"), [&y_reference + &x_reference]),
    )
    .expect("a negation condition is accepted");
    let (p_text, x_text, y_text) = (
        write_name_hint_with_id(&p),
        write_name_hint_with_id(&x),
        write_name_hint_with_id(&y),
    );

    let texts = (
        format_with_ids(&tree, Notation::Symbolic),
        format_with_ids(&tree, Notation::Functional),
    );

    assert_eq!(
        texts,
        (
            format!("{{(-{x_text}) if (!{p_text}); f(({y_text} + {x_text})) otherwise}}"),
            format!(
                "(piecewise (logical_not {p_text}) (negate {x_text}) (f (add {y_text} {x_text})))"
            ),
        )
    );
}

#[test]
fn format_expression_tells_same_named_identifiers_apart_only_with_ids() {
    let (first, first_reference) = build_identifier("x");
    let (second, second_reference) = build_identifier("x");
    let difference = &first_reference - &second_reference;

    let texts = (
        format_symbolic(&difference),
        format_with_ids(&difference, Notation::Symbolic),
    );

    assert_eq!(
        texts,
        (
            "(x - x)".to_owned(),
            format!(
                "({} - {})",
                write_name_hint_with_id(&first),
                write_name_hint_with_id(&second)
            ),
        )
    );
}

/// Test a name hint is written without quoting or escaping, with and without
/// its id.
#[rstest]
#[case::space("a b")]
#[case::colons("m::n")]
#[case::non_ascii("\u{e9}t\u{e9}")]
#[case::digits("42")]
#[case::boolean_word("True")]
#[case::parentheses("(p)")]
fn format_expression_writes_name_hints_raw(#[case] name_hint: &str) {
    let (identifier, reference) = build_identifier(name_hint);

    let texts = (
        format_functional(&reference),
        format_with_ids(&reference, Notation::Functional),
    );

    assert_eq!(
        texts,
        (
            name_hint.to_owned(),
            format!("{name_hint}::{}", identifier.id())
        )
    );
}

// =============================================================================
// Piecewise
// =============================================================================

#[test]
fn format_expression_writes_single_case_piecewise_in_braces() {
    let piecewise = Expression::piecewise([(LiteralValue::from(true), 1)], 2)
        .expect("a Boolean condition is accepted");

    let text = format_symbolic(&piecewise);

    assert_eq!(text, "{1 if true; 2 otherwise}");
}

#[test]
fn format_expression_writes_every_piecewise_case_then_otherwise() {
    let piecewise = Expression::piecewise(
        [
            (LiteralValue::from(true), 1),
            (LiteralValue::from(false), 2),
        ],
        3,
    )
    .expect("Boolean conditions are accepted");

    let text = format_symbolic(&piecewise);

    assert_eq!(text, "{1 if true; 2 if false; 3 otherwise}");
}

#[test]
fn format_expression_writes_piecewise_functionally() {
    let piecewise = Expression::piecewise([(LiteralValue::from(true), 1)], 2)
        .expect("a Boolean condition is accepted");

    let text = format_functional(&piecewise);

    assert_eq!(text, "(piecewise true 1 2)");
}

#[test]
fn format_expression_writes_multi_case_piecewise_functionally() {
    let piecewise = Expression::piecewise(
        [
            (LiteralValue::from(true), 1),
            (LiteralValue::from(false), 2),
        ],
        3,
    )
    .expect("Boolean conditions are accepted");

    let text = format_functional(&piecewise);

    assert_eq!(text, "(piecewise true 1 false 2 3)");
}

#[test]
fn format_expression_orders_piecewise_case_parts_per_notation() {
    let (_, p) = build_identifier("p");
    let (_, x) = build_identifier("x");
    let (_, y) = build_identifier("y");
    let piecewise =
        Expression::piecewise([(p, x)], y).expect("an identifier condition is accepted");

    let texts = (format_symbolic(&piecewise), format_functional(&piecewise));

    assert_eq!(
        texts,
        (
            "{x if p; y otherwise}".to_owned(),
            "(piecewise p x y)".to_owned()
        )
    );
}

#[test]
fn format_expression_writes_a_wide_piecewise_in_order() {
    const CASE_COUNT: i64 = 120;
    let (_, x) = build_identifier("x");
    let piecewise = Expression::piecewise(
        (0..CASE_COUNT).map(|index| (x.equals(index), build_literal(index))),
        -1,
    )
    .expect("comparison conditions are accepted");
    let expected_symbolic = format!(
        "{{{}; -1 otherwise}}",
        (0..CASE_COUNT)
            .map(|index| format!("{index} if (x == {index})"))
            .collect::<Vec<_>>()
            .join("; ")
    );
    let expected_functional = format!(
        "(piecewise {} -1)",
        (0..CASE_COUNT)
            .map(|index| format!("(equal x {index}) {index}"))
            .collect::<Vec<_>>()
            .join(" ")
    );

    let texts = (format_symbolic(&piecewise), format_functional(&piecewise));

    assert_eq!(texts, (expected_symbolic, expected_functional));
}

#[test]
fn format_expression_writes_nested_piecewise_inside_a_case_value() {
    let inner = Expression::piecewise([(LiteralValue::from(false), 1)], 2)
        .expect("a Boolean condition is accepted");
    let outer = Expression::piecewise([(LiteralValue::from(true), inner)], 3)
        .expect("a Boolean condition is accepted");

    let text = format_symbolic(&outer);

    assert_eq!(text, "{{1 if false; 2 otherwise} if true; 3 otherwise}");
}

// =============================================================================
// Calls
// =============================================================================

#[test]
fn format_expression_writes_call_with_arguments() {
    let call = Expression::call(build_callee("max"), [1, 2]);

    let text = format_symbolic(&call);

    assert_eq!(text, "max(1, 2)");
}

#[test]
fn format_expression_writes_zero_argument_call_with_empty_parentheses() {
    let call = Expression::call(build_callee("noargs"), Vec::<Expression>::new());

    let text = format_symbolic(&call);

    assert_eq!(text, "noargs()");
}

#[test]
fn format_expression_writes_call_functionally() {
    let call = Expression::call(build_callee("max"), [1, 2]);

    let text = format_functional(&call);

    assert_eq!(text, "(max 1 2)");
}

#[test]
fn format_expression_writes_zero_argument_call_functionally() {
    let call = Expression::call(build_callee("f"), Vec::<Expression>::new());

    let text = format_functional(&call);

    assert_eq!(text, "(f)");
}

#[test]
fn format_expression_writes_nested_call_in_argument_position() {
    let inner = Expression::call(build_callee("min"), [1, 2]);
    let outer = Expression::call(build_callee("max"), [inner, build_literal(3)]);

    let text = format_symbolic(&outer);

    assert_eq!(text, "max(min(1, 2), 3)");
}

// =============================================================================
// Shared subtrees and composed bodies
// =============================================================================

#[test]
fn format_expression_writes_a_shared_subtree_at_every_occurrence() {
    let (_, x) = build_identifier("x");
    let shared = &x + 1;
    let square = &shared * &shared;

    let text = format_symbolic(&square);

    assert_eq!(text, "((x + 1) * (x + 1))");
}

/// Test the body of the Gaussian error linear unit prints in both notations.
#[test]
fn format_expression_writes_the_gelu_body() {
    let (_, x) = build_identifier("x");
    let root_two = Expression::call(build_callee("sqrt"), [2.0]);
    let error_function = Expression::call(build_callee("erf"), [&x / root_two]);
    let gelu = (0.5 * &x) * (1.0 + error_function);

    let texts = (format_symbolic(&gelu), format_functional(&gelu));

    assert_eq!(
        texts,
        (
            "((0.5 * x) * (1 + erf((x / sqrt(2)))))".to_owned(),
            "(multiply (multiply 0.5 x) (add 1 (erf (divide x (sqrt 2)))))".to_owned(),
        )
    );
}

/// Test the body of the sign function prints its float comparisons and
/// integer results as given.
#[test]
fn format_expression_writes_the_sign_body() {
    let (_, x) = build_identifier("x");
    let sign = Expression::piecewise(
        [
            (x.greater(0.0), build_literal(1)),
            (x.less(0.0), build_literal(-1)),
        ],
        0,
    )
    .expect("comparison conditions are accepted");

    let texts = (format_symbolic(&sign), format_functional(&sign));

    assert_eq!(
        texts,
        (
            "{1 if (x > 0); -1 if (x < 0); 0 otherwise}".to_owned(),
            "(piecewise (greater x 0) 1 (less x 0) -1 0)".to_owned(),
        )
    );
}

// =============================================================================
// Deep trees
// =============================================================================

/// Return `opening` [`SMALL_STACK_DEPTH`] times, the leaf `x`, then `closing`
/// as many times.
fn repeat_around_leaf((opening, closing): (&str, &str)) -> String {
    format!(
        "{}x{}",
        opening.repeat(SMALL_STACK_DEPTH),
        closing.repeat(SMALL_STACK_DEPTH)
    )
}

/// The spines along which the deep-tree tests nest a node kind.
#[derive(Debug, Clone, Copy)]
enum DeepShape {
    LeftSum,
    RightConjunction,
    Negation,
    PiecewiseInOtherwise,
    PiecewiseInCondition,
    CallInLastArgument,
    CallInFirstArgument,
}

impl DeepShape {
    /// Build the shape [`SMALL_STACK_DEPTH`] levels deep over the leaf `x`,
    /// with `p` as the condition where one is needed.
    fn build(self) -> Expression {
        let (_, x) = build_identifier("x");
        let (_, p) = build_identifier("p");
        match self {
            Self::LeftSum => build_deep_sum(&x, SMALL_STACK_DEPTH),
            Self::RightConjunction => build_deep_conjunction(&x, SMALL_STACK_DEPTH),
            Self::Negation => build_deep_negation(&x, SMALL_STACK_DEPTH),
            Self::PiecewiseInOtherwise => {
                build_deep_piecewise_in_otherwise(&p, &x, SMALL_STACK_DEPTH)
            }
            Self::PiecewiseInCondition => build_deep_piecewise_in_condition(&x, SMALL_STACK_DEPTH),
            Self::CallInLastArgument => build_deep_call_in_last_argument(&x, SMALL_STACK_DEPTH),
            Self::CallInFirstArgument => build_deep_call_in_first_argument(&x, SMALL_STACK_DEPTH),
        }
    }

    /// Return the text expected for the shape: `(opening, closing)` pieces
    /// written [`SMALL_STACK_DEPTH`] times around the leaf `x`, in symbolic
    /// then functional notation.
    fn expected_pieces(self) -> [(&'static str, &'static str); 2] {
        match self {
            Self::LeftSum => [("(", " + 1)"), ("(add ", " 1)")],
            Self::RightConjunction => [("(true && ", ")"), ("(and true ", ")")],
            Self::Negation => [("(-", ")"), ("(negate ", ")")],
            Self::PiecewiseInOtherwise => [("{1 if p; ", " otherwise}"), ("(piecewise p 1 ", ")")],
            Self::PiecewiseInCondition => [("{1 if ", "; 0 otherwise}"), ("(piecewise ", " 1 0)")],
            Self::CallInLastArgument => [("f(1, ", ")"), ("(f 1 ", ")")],
            Self::CallInFirstArgument => [("f(", ", 2)"), ("(f ", " 2)")],
        }
    }
}

/// Test a tree [`SMALL_STACK_DEPTH`] levels deep prints in both notations on a
/// stack far too small for one frame per level.
#[rstest]
#[case::left_sum(DeepShape::LeftSum)]
#[case::right_conjunction(DeepShape::RightConjunction)]
#[case::negation(DeepShape::Negation)]
#[case::piecewise_in_otherwise(DeepShape::PiecewiseInOtherwise)]
#[case::piecewise_in_condition(DeepShape::PiecewiseInCondition)]
#[case::call_in_last_argument(DeepShape::CallInLastArgument)]
#[case::call_in_first_argument(DeepShape::CallInFirstArgument)]
fn format_expression_prints_a_deep_tree_on_a_small_stack(#[case] shape: DeepShape) {
    run_on_small_stack(move || {
        let tree = shape.build();
        let expected = shape.expected_pieces().map(repeat_around_leaf);

        let symbolic = format_symbolic(&tree);
        let functional = format_functional(&tree);

        assert!(
            symbolic == expected[0],
            "symbolic text differs for {shape:?}"
        );
        assert!(
            functional == expected[1],
            "functional text differs for {shape:?}"
        );
    });
}

/// Test an expression's own `Display` of a tree [`SMALL_STACK_DEPTH`] levels
/// deep completes on a stack far too small for one frame per level.
#[rstest]
#[case::left_sum(DeepShape::LeftSum)]
#[case::right_conjunction(DeepShape::RightConjunction)]
#[case::piecewise_in_condition(DeepShape::PiecewiseInCondition)]
#[case::call_in_first_argument(DeepShape::CallInFirstArgument)]
fn expression_display_of_a_deep_tree_completes_on_a_small_stack(#[case] shape: DeepShape) {
    run_on_small_stack(move || {
        let tree = shape.build();
        let expected = repeat_around_leaf(shape.expected_pieces()[0]);

        let text = tree.to_string();

        assert!(text == expected, "text differs for {shape:?}");
    });
}
