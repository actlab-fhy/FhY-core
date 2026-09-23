//! Tests for printing expressions: the default options, both notations for
//! every node kind and operation, literal and identifier text, identifier
//! ids, piecewise and call layout, shared subtrees, and trees thousands of
//! levels deep printed on a small thread stack.
//!
//! Public API only (`fhy_core::symbolic::expression`).

#[path = "common/expression.rs"]
pub mod expression_support;

use std::thread;

use expression_support::{
    DEEP_TREE_DEPTH, build_deep_conjunction, build_deep_sum, build_identifier, build_literal,
    build_text_literal,
};
use fhy_core::identifier::Identifier;
use fhy_core::symbolic::expression::{
    BinaryOperation, Expression, FormatOptions, IdentifierStyle, LiteralValue, Notation,
    UnaryOperation, build_call, build_logical_and, build_piecewise, format_expression,
};
use num_bigint::BigInt;
use rstest::rstest;

/// Stack size of the thread the deep trees are printed on: far below what a
/// printer recursing once per level would need for a tree
/// [`DEEP_TREE_DEPTH`] levels deep.
const SMALL_STACK_BYTES: usize = 128 << 10;

/// Return the options writing `notation` with name hints only.
fn build_name_hint_options(notation: Notation) -> FormatOptions {
    FormatOptions::new(notation, IdentifierStyle::NameHint)
}

/// Return the options writing `notation` with identifier ids.
fn build_id_options(notation: Notation) -> FormatOptions {
    FormatOptions::new(notation, IdentifierStyle::NameHintWithId)
}

/// Return `expression` printed symbolically with name hints only.
fn format_symbolic(expression: &Expression) -> String {
    format_expression(expression, build_name_hint_options(Notation::Symbolic))
}

/// Return `expression` printed functionally with name hints only.
fn format_functional(expression: &Expression) -> String {
    format_expression(expression, build_name_hint_options(Notation::Functional))
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

/// Print `expression` under `options` on a thread with a
/// [`SMALL_STACK_BYTES`] stack, and return the text with the tree, so the
/// tree is dropped on the caller's thread.
fn format_on_small_stack(expression: Expression, options: FormatOptions) -> (String, Expression) {
    let handle = thread::Builder::new()
        .stack_size(SMALL_STACK_BYTES)
        .spawn(move || (format_expression(&expression, options), expression))
        .expect("the printing thread spawns");
    match handle.join() {
        Ok(result) => result,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

/// Return `-(-(-leaf))`, `depth` negations deep.
fn build_deep_negation(leaf: &Expression, depth: usize) -> Expression {
    let mut tree = leaf.clone();
    for _ in 0..depth {
        tree = -tree;
    }
    tree
}

/// Return `{1 if condition; {1 if condition; ... leaf otherwise} otherwise}`,
/// `depth` piecewise nodes deep along the otherwise branches.
fn build_deep_piecewise_in_otherwise(
    condition: &Expression,
    leaf: &Expression,
    depth: usize,
) -> Expression {
    let mut tree = leaf.clone();
    for _ in 0..depth {
        tree = build_piecewise([(condition, build_literal(1))], tree)
            .expect("an identifier condition is accepted");
    }
    tree
}

/// Return `{1 if {1 if ... leaf ...; 0 otherwise}; 0 otherwise}`, `depth`
/// piecewise nodes deep along the case conditions.
fn build_deep_piecewise_in_condition(leaf: &Expression, depth: usize) -> Expression {
    let mut tree = leaf.clone();
    for _ in 0..depth {
        tree = build_piecewise([(tree, build_literal(1))], build_literal(0))
            .expect("a piecewise condition is accepted");
    }
    tree
}

/// Return `f(1, f(1, ... leaf))`, `depth` calls deep along the last argument.
fn build_deep_call_in_last_argument(leaf: &Expression, depth: usize) -> Expression {
    let mut tree = leaf.clone();
    for _ in 0..depth {
        tree = build_call("f", [build_literal(1), tree]).expect("a named call");
    }
    tree
}

/// Return `f(f(... leaf, 2), 2)`, `depth` calls deep along the first argument.
fn build_deep_call_in_first_argument(leaf: &Expression, depth: usize) -> Expression {
    let mut tree = leaf.clone();
    for _ in 0..depth {
        tree = build_call("f", [tree, build_literal(2)]).expect("a named call");
    }
    tree
}

// =============================================================================
// Options
// =============================================================================

/// Test the default options are symbolic notation with name hints only.
#[test]
fn format_options_default_is_symbolic_with_name_hints() {
    let default = FormatOptions::default();

    assert_eq!(
        default,
        FormatOptions::new(Notation::Symbolic, IdentifierStyle::NameHint)
    );
}

/// Test the default options write an identifier as its name hint alone.
#[test]
fn format_expression_default_options_write_name_hint_only() {
    let (_, name_only) = build_identifier("name_only");

    let text = format_expression(&name_only, FormatOptions::default());

    assert_eq!(text, "name_only");
}

/// Test the default options write a binary node with its operator symbol.
#[test]
fn format_expression_default_options_write_symbolic_notation() {
    let sum = build_literal(1) + 2;

    let text = format_expression(&sum, FormatOptions::default());

    assert_eq!(text, "(1 + 2)");
}

// =============================================================================
// Notations
// =============================================================================

/// Test symbolic notation writes literals, identifiers, and operator nodes.
#[rstest]
#[case::float(build_literal(4.5), "4.5")]
#[case::decimal_text(build_text_literal("0.1"), "0.1")]
#[case::identifier(build_identifier("baz").1, "baz")]
#[case::logical_not(build_literal(true).logical_not(), "(!True)")]
#[case::product(build_text_literal("3.14") * 10.5, "(3.14 * 10.5)")]
fn format_expression_writes_symbolic_notation(
    #[case] expression: Expression,
    #[case] expected: &str,
) {
    let text = format_symbolic(&expression);

    assert_eq!(text, expected);
}

/// Test functional notation writes literals, identifiers, and prefix
/// operation names.
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

/// Test every unary operation is written with its symbol, and with its
/// name in functional notation.
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

/// Test every binary operation is written infix with its symbol, and prefix
/// with its name in functional notation.
#[rstest]
#[case::add(BinaryOperation::Add, "(x + 2)", "(add x 2)")]
#[case::subtract(BinaryOperation::Subtract, "(x - 2)", "(subtract x 2)")]
#[case::multiply(BinaryOperation::Multiply, "(x * 2)", "(multiply x 2)")]
#[case::divide(BinaryOperation::Divide, "(x / 2)", "(divide x 2)")]
#[case::floor_divide(BinaryOperation::FloorDivide, "(x // 2)", "(floor_divide x 2)")]
#[case::modulo(BinaryOperation::Modulo, "(x % 2)", "(modulo x 2)")]
#[case::power(BinaryOperation::Power, "(x ** 2)", "(power x 2)")]
#[case::logical_and(BinaryOperation::LogicalAnd, "(x && 2)", "(logical_and x 2)")]
#[case::logical_or(BinaryOperation::LogicalOr, "(x || 2)", "(logical_or x 2)")]
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

/// Test every unary and binary node is parenthesized, so the text shows how
/// operations nest without any precedence or associativity rule.
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

/// Test a conjunction of three operands prints its right fold.
#[test]
fn format_expression_writes_a_folded_conjunction_as_nested_pairs() {
    let (_, x) = build_identifier("x");
    let (_, y) = build_identifier("y");
    let (_, z) = build_identifier("z");
    let conjunction = build_logical_and([x, y, z]).expect("three operands make a conjunction");

    let texts = (
        format_symbolic(&conjunction),
        format_functional(&conjunction),
    );

    assert_eq!(
        texts,
        (
            "(x && (y && z))".to_owned(),
            "(logical_and x (logical_and y z))".to_owned()
        )
    );
}

// =============================================================================
// Literals
// =============================================================================

/// Test a literal is written as the value it was given, in both notations.
#[rstest]
#[case::true_value(LiteralValue::from(true), "True")]
#[case::false_value(LiteralValue::from(false), "False")]
#[case::zero(LiteralValue::from(0), "0")]
#[case::negative_integer(LiteralValue::from(-1), "-1")]
#[case::big_integer(
    LiteralValue::from(BigInt::from(10).pow(30)),
    "1000000000000000000000000000000"
)]
#[case::integral_float(LiteralValue::from(1.0), "1.0")]
#[case::tenth(LiteralValue::from(0.1), "0.1")]
#[case::third(LiteralValue::from(1.0 / 3.0), "0.3333333333333333")]
#[case::largest_positional_float(LiteralValue::from(1e15), "1000000000000000.0")]
#[case::scientific_large_float(LiteralValue::from(1e16), "1e+16")]
#[case::scientific_long_mantissa(
    LiteralValue::from(1.234_567_890_123_456_8e17),
    "1.2345678901234568e+17"
)]
#[case::scientific_1e22(LiteralValue::from(1e22), "1e+22")]
#[case::smallest_positional_float(LiteralValue::from(0.0001), "0.0001")]
#[case::scientific_small_float(LiteralValue::from(1e-7), "1e-07")]
#[case::negative_zero(LiteralValue::from(-0.0), "-0.0")]
#[case::nan(LiteralValue::from(f64::NAN), "nan")]
#[case::infinity(LiteralValue::from(f64::INFINITY), "inf")]
#[case::negative_infinity(LiteralValue::from(f64::NEG_INFINITY), "-inf")]
#[case::padded_integer_text(LiteralValue::parse_text("05").expect("an integer text"), "05")]
#[case::decimal_text_trailing_zero(LiteralValue::parse_text("1.50").expect("a decimal text"), "1.50")]
#[case::decimal_text_no_whole_part(LiteralValue::parse_text(".5").expect("a decimal text"), ".5")]
#[case::decimal_text_no_fraction(LiteralValue::parse_text("1.").expect("a decimal text"), "1.")]
fn format_expression_writes_a_literal_as_given(
    #[case] value: LiteralValue,
    #[case] expected: &str,
) {
    let literal = Expression::from(value);

    let texts = (format_symbolic(&literal), format_functional(&literal));

    assert_eq!(texts, (expected.to_owned(), expected.to_owned()));
}

/// Test a unary operation over a negative literal writes the literal's sign
/// after the operator's.
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

/// Test an integer and the integer text of the same value print alike.
#[test]
fn format_expression_writes_integer_and_integer_text_alike() {
    let integer = build_literal(5);
    let text = build_text_literal("5");

    let texts = (format_symbolic(&integer), format_symbolic(&text));

    assert_eq!(texts, ("5".to_owned(), "5".to_owned()));
}

// =============================================================================
// Identifiers
// =============================================================================

/// Test showing ids writes an identifier as its name hint, two colons, and
/// its id.
#[test]
fn format_expression_with_ids_writes_name_hint_and_id() {
    let (foo, reference) = build_identifier("foo");

    let text = format_expression(&reference, build_id_options(Notation::Symbolic));

    assert_eq!(text, format!("foo::{}", foo.id()));
}

/// Test showing ids applies in functional notation too.
#[test]
fn format_expression_with_ids_applies_in_functional_notation() {
    let (x, reference) = build_identifier("x");
    let sum = &reference + 1;

    let text = format_expression(&sum, build_id_options(Notation::Functional));

    assert_eq!(text, format!("(add {} 1)", write_name_hint_with_id(&x)));
}

/// Test showing ids reaches an identifier nested in a piecewise condition.
#[test]
fn format_expression_with_ids_reaches_piecewise_conditions() {
    let (condition, reference) = build_identifier("cond");
    let piecewise =
        build_piecewise([(reference, 1)], 0).expect("an identifier condition is accepted");

    let text = format_expression(&piecewise, build_id_options(Notation::Symbolic));

    assert_eq!(
        text,
        format!(
            "{{1 if {}; 0 otherwise}}",
            write_name_hint_with_id(&condition)
        )
    );
}

/// Test showing ids reaches an identifier nested in call arguments.
#[test]
fn format_expression_with_ids_reaches_call_arguments() {
    let (argument, reference) = build_identifier("arg");
    let call = build_call("nested_show_id", [reference]).expect("a named call");

    let text = format_expression(&call, build_id_options(Notation::Symbolic));

    assert_eq!(
        text,
        format!("nested_show_id({})", write_name_hint_with_id(&argument))
    );
}

/// Test showing ids reaches every identifier of a tree holding every node
/// kind, in both notations.
#[test]
fn format_expression_with_ids_reaches_every_node_kind() {
    let (p, p_reference) = build_identifier("p");
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let tree = build_piecewise(
        [(p_reference.logical_not(), -&x_reference)],
        build_call("f", [&y_reference + &x_reference]).expect("a named call"),
    )
    .expect("a negation condition is accepted");
    let (p_text, x_text, y_text) = (
        write_name_hint_with_id(&p),
        write_name_hint_with_id(&x),
        write_name_hint_with_id(&y),
    );

    let texts = (
        format_expression(&tree, build_id_options(Notation::Symbolic)),
        format_expression(&tree, build_id_options(Notation::Functional)),
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

/// Test two identifiers sharing a name hint print alike under name hints and
/// apart when ids are shown.
#[test]
fn format_expression_tells_same_named_identifiers_apart_only_with_ids() {
    let (first, first_reference) = build_identifier("x");
    let (second, second_reference) = build_identifier("x");
    let difference = &first_reference - &second_reference;

    let texts = (
        format_symbolic(&difference),
        format_expression(&difference, build_id_options(Notation::Symbolic)),
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

/// Test a name hint is written raw, without quoting or escaping, with and
/// without its id.
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
        format_expression(&reference, build_id_options(Notation::Functional)),
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

/// Test a one-case piecewise is written as its case then its otherwise
/// branch, in braces.
#[test]
fn format_expression_writes_single_case_piecewise_in_braces() {
    let piecewise = build_piecewise([(LiteralValue::from(true), 1)], 2)
        .expect("a Boolean condition is accepted");

    let text = format_symbolic(&piecewise);

    assert_eq!(text, "{1 if True; 2 otherwise}");
}

/// Test a multi-case piecewise writes every case in order, then the
/// otherwise branch.
#[test]
fn format_expression_writes_every_piecewise_case_then_otherwise() {
    let piecewise = build_piecewise(
        [
            (LiteralValue::from(true), 1),
            (LiteralValue::from(false), 2),
        ],
        3,
    )
    .expect("Boolean conditions are accepted");

    let text = format_symbolic(&piecewise);

    assert_eq!(text, "{1 if True; 2 if False; 3 otherwise}");
}

/// Test functional notation writes a piecewise as `(piecewise c v o)`.
#[test]
fn format_expression_writes_piecewise_functionally() {
    let piecewise = build_piecewise([(LiteralValue::from(true), 1)], 2)
        .expect("a Boolean condition is accepted");

    let text = format_functional(&piecewise);

    assert_eq!(text, "(piecewise True 1 2)");
}

/// Test functional notation lists every condition and value pair, then the
/// otherwise branch.
#[test]
fn format_expression_writes_multi_case_piecewise_functionally() {
    let piecewise = build_piecewise(
        [
            (LiteralValue::from(true), 1),
            (LiteralValue::from(false), 2),
        ],
        3,
    )
    .expect("Boolean conditions are accepted");

    let text = format_functional(&piecewise);

    assert_eq!(text, "(piecewise True 1 False 2 3)");
}

/// Test symbolic notation writes a case's value before its condition and
/// functional notation its condition first.
#[test]
fn format_expression_orders_piecewise_case_parts_per_notation() {
    let (_, p) = build_identifier("p");
    let (_, x) = build_identifier("x");
    let (_, y) = build_identifier("y");
    let piecewise = build_piecewise([(p, x)], y).expect("an identifier condition is accepted");

    let texts = (format_symbolic(&piecewise), format_functional(&piecewise));

    assert_eq!(
        texts,
        (
            "{x if p; y otherwise}".to_owned(),
            "(piecewise p x y)".to_owned()
        )
    );
}

/// Test a piecewise of 120 cases writes every case in order in both
/// notations.
#[test]
fn format_expression_writes_a_wide_piecewise_in_order() {
    const CASE_COUNT: i64 = 120;
    let (_, x) = build_identifier("x");
    let piecewise = build_piecewise(
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

/// Test a piecewise nested in a case value is written in place.
#[test]
fn format_expression_writes_nested_piecewise_inside_a_case_value() {
    let inner = build_piecewise([(LiteralValue::from(false), 1)], 2)
        .expect("a Boolean condition is accepted");
    let outer = build_piecewise([(LiteralValue::from(true), inner)], 3)
        .expect("a Boolean condition is accepted");

    let text = format_symbolic(&outer);

    assert_eq!(text, "{{1 if False; 2 otherwise} if True; 3 otherwise}");
}

// =============================================================================
// Calls
// =============================================================================

/// Test a call is written as its name and its arguments in parentheses,
/// separated by a comma and a space.
#[test]
fn format_expression_writes_call_with_arguments() {
    let call = build_call("max", [1, 2]).expect("a named call");

    let text = format_symbolic(&call);

    assert_eq!(text, "max(1, 2)");
}

/// Test a call without arguments is written with empty parentheses.
#[test]
fn format_expression_writes_zero_argument_call_with_empty_parentheses() {
    let call = build_call("noargs", Vec::<Expression>::new()).expect("a named call");

    let text = format_symbolic(&call);

    assert_eq!(text, "noargs()");
}

/// Test functional notation writes a call as its name and arguments in one
/// pair of parentheses.
#[test]
fn format_expression_writes_call_functionally() {
    let call = build_call("max", [1, 2]).expect("a named call");

    let text = format_functional(&call);

    assert_eq!(text, "(max 1 2)");
}

/// Test functional notation writes a call without arguments as its name in
/// parentheses.
#[test]
fn format_expression_writes_zero_argument_call_functionally() {
    let call = build_call("f", Vec::<Expression>::new()).expect("a named call");

    let text = format_functional(&call);

    assert_eq!(text, "(f)");
}

/// Test a call nested in an argument is written in place.
#[test]
fn format_expression_writes_nested_call_in_argument_position() {
    let inner = build_call("min", [1, 2]).expect("a named call");
    let outer = build_call("max", [inner, build_literal(3)]).expect("a named call");

    let text = format_symbolic(&outer);

    assert_eq!(text, "max(min(1, 2), 3)");
}

// =============================================================================
// Shared subtrees and composed bodies
// =============================================================================

/// Test a subtree shared by several positions is written at each of them.
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
    let root_two = build_call("sqrt", [2.0]).expect("a named call");
    let error_function = build_call("erf", [&x / root_two]).expect("a named call");
    let gelu = (0.5 * &x) * (1.0 + error_function);

    let texts = (format_symbolic(&gelu), format_functional(&gelu));

    assert_eq!(
        texts,
        (
            "((0.5 * x) * (1.0 + erf((x / sqrt(2.0)))))".to_owned(),
            "(multiply (multiply 0.5 x) (add 1.0 (erf (divide x (sqrt 2.0)))))".to_owned(),
        )
    );
}

/// Test the body of the sign function prints its float comparisons and
/// integer results as given.
#[test]
fn format_expression_writes_the_sign_body() {
    let (_, x) = build_identifier("x");
    let sign = build_piecewise(
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
            "{1 if (x > 0.0); -1 if (x < 0.0); 0 otherwise}".to_owned(),
            "(piecewise (greater x 0.0) 1 (less x 0.0) -1 0)".to_owned(),
        )
    );
}

// =============================================================================
// Deep trees
// =============================================================================

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
    /// Build the shape [`DEEP_TREE_DEPTH`] levels deep over the leaf `x`,
    /// with `p` as the condition where one is needed.
    fn build(self) -> Expression {
        let (_, x) = build_identifier("x");
        let (_, p) = build_identifier("p");
        match self {
            Self::LeftSum => build_deep_sum(&x, DEEP_TREE_DEPTH),
            Self::RightConjunction => build_deep_conjunction(&x, DEEP_TREE_DEPTH),
            Self::Negation => build_deep_negation(&x, DEEP_TREE_DEPTH),
            Self::PiecewiseInOtherwise => {
                build_deep_piecewise_in_otherwise(&p, &x, DEEP_TREE_DEPTH)
            }
            Self::PiecewiseInCondition => build_deep_piecewise_in_condition(&x, DEEP_TREE_DEPTH),
            Self::CallInLastArgument => build_deep_call_in_last_argument(&x, DEEP_TREE_DEPTH),
            Self::CallInFirstArgument => build_deep_call_in_first_argument(&x, DEEP_TREE_DEPTH),
        }
    }

    /// Return the text expected for the shape: `(opening, closing)` pieces
    /// written [`DEEP_TREE_DEPTH`] times around the leaf `x`, in symbolic
    /// then functional notation.
    fn expected_pieces(self) -> [(&'static str, &'static str); 2] {
        match self {
            Self::LeftSum => [("(", " + 1)"), ("(add ", " 1)")],
            Self::RightConjunction => [("(True && ", ")"), ("(logical_and True ", ")")],
            Self::Negation => [("(-", ")"), ("(negate ", ")")],
            Self::PiecewiseInOtherwise => [("{1 if p; ", " otherwise}"), ("(piecewise p 1 ", ")")],
            Self::PiecewiseInCondition => [("{1 if ", "; 0 otherwise}"), ("(piecewise ", " 1 0)")],
            Self::CallInLastArgument => [("f(1, ", ")"), ("(f 1 ", ")")],
            Self::CallInFirstArgument => [("f(", ", 2)"), ("(f ", " 2)")],
        }
    }
}

/// Test a tree thousands of levels deep prints in both notations on a thread
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
    let tree = shape.build();
    let expected = shape.expected_pieces().map(|(opening, closing)| {
        format!(
            "{}x{}",
            opening.repeat(DEEP_TREE_DEPTH),
            closing.repeat(DEEP_TREE_DEPTH)
        )
    });

    let (symbolic, tree) = format_on_small_stack(tree, build_name_hint_options(Notation::Symbolic));
    let (functional, _tree) =
        format_on_small_stack(tree, build_name_hint_options(Notation::Functional));

    assert!(
        symbolic == expected[0],
        "symbolic text differs for {shape:?}"
    );
    assert!(
        functional == expected[1],
        "functional text differs for {shape:?}"
    );
}
