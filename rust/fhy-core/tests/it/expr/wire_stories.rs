//! Tests for the wire form of expressions: the node table of each node
//! kind, literal values of every kind as strings, round trips through JSON
//! values, JSON text and postcard, and the tables a decode refuses. The
//! round trips of deep trees, large logical nodes and DAGs through both
//! formats are in `serde_format_stories`.
//!
//! Public API only (`fhy_core::expr`, `fhy_core::identifier`).

use crate::support::expression as expression_support;
use crate::support::stack as stack_support;

use std::fmt::Write as _;

use expression_support::{
    build_callee, build_decimal_literal, build_identifier, build_literal, expect_literal,
};
use fhy_core::expr::builtins::BuiltinFunction;
use fhy_core::expr::{
    BigInt, BinaryOperation, Expression, ExpressionKind, LiteralValue, UnaryOperation,
};
use fhy_core::identifier::Identifier;
use rstest::rstest;
use serde_json::{Value, json};
use stack_support::run_on_stack;

/// Stack size of the threads the deep round trips and decodes run on.
const WIRE_STACK_BYTES: usize = 256 << 10;

/// Serialize `expression` to a JSON value.
fn encode(expression: &Expression) -> Value {
    serde_json::to_value(expression).expect("the expression serializes")
}

/// Deserialize an expression from a JSON value.
fn decode(value: Value) -> Result<Expression, serde_json::Error> {
    serde_json::from_value(value)
}

/// Return the table holding `nodes`, in order.
fn build_table(nodes: impl IntoIterator<Item = Value>) -> Value {
    json!({"nodes": nodes.into_iter().collect::<Vec<_>>()})
}

/// Return the table node of a literal whose wire form is `value`.
fn build_literal_node(value: &Value) -> Value {
    json!({"literal": value})
}

/// Return the table node of a reference to `identifier`.
fn build_identifier_node(identifier: &Identifier) -> Value {
    json!({"identifier": {"id": identifier.id(), "name_hint": identifier.name_hint()}})
}

/// Assert decoding `table` is refused with a data error whose message
/// contains `expected`.
fn assert_refused(table: &Value, expected: &str) {
    let error = decode(table.clone()).expect_err("the table is refused");

    assert!(error.is_data(), "a data error for {table}: {error}");
    assert!(
        error.to_string().contains(expected),
        "{table} is refused with {error}, not {expected:?}"
    );
}

// =============================================================================
// Shape of each node kind
// =============================================================================

/// Test a Boolean literal serializes to its pinned table and decodes back.
#[test]
fn expression_literal_round_trips_through_its_wire_form() {
    let expression = build_literal(true);
    let expected = build_table([build_literal_node(&json!({"bool": true}))]);

    let wire = encode(&expression);
    let restored = decode(expected.clone()).expect("a valid table");

    assert_eq!(wire, expected);
    assert_eq!(restored, expression);
}

/// Test `x + 1` serializes to exactly its documented JSON text.
#[test]
fn expression_serializes_a_sum_as_its_documented_text() {
    let (x, reference) = build_identifier("x");
    let sum = &reference + 1;

    let text = serde_json::to_string(&sum).expect("the expression serializes");

    let expected = format!(
        r#"{{"nodes":[{{"identifier":{{"id":{},"name_hint":"x"}}}},{{"literal":{{"int":"1"}}}},{{"binary":{{"operation":"add","left":0,"right":1}}}}]}}"#,
        x.id()
    );
    assert_eq!(text, expected);
}

/// Test each node kind serializes to its table node, children first, the
/// root last, a shared node once.
#[test]
fn expression_serializes_every_node_kind_in_its_wire_shape() {
    let (x, x_reference) = build_identifier("x");
    let expression = Expression::piecewise(
        [(x_reference.less(3), -&x_reference)],
        Expression::call(
            BuiltinFunction::Max,
            [
                Expression::any([x_reference.greater(0), build_literal(false)]),
                build_literal(1.5),
                build_decimal_literal("2.50"),
                Expression::call(build_callee("f"), Vec::<Expression>::new()),
            ],
        ),
    )
    .expect("a valid piecewise");

    let wire = encode(&expression);

    let expected = build_table([
        build_identifier_node(&x),
        build_literal_node(&json!({"int": "3"})),
        json!({"binary": {"operation": "less", "left": 0, "right": 1}}),
        json!({"unary": {"operation": "negate", "operand": 0}}),
        build_literal_node(&json!({"int": "0"})),
        json!({"binary": {"operation": "greater", "left": 0, "right": 4}}),
        build_literal_node(&json!({"bool": false})),
        json!({"logical": {"operation": "or", "operands": [5, 6]}}),
        build_literal_node(&json!({"float": "1.5"})),
        build_literal_node(&json!({"decimal": "2.5"})),
        json!({"call": {"callee": {"named": "f"}, "arguments": []}}),
        json!({"call": {"callee": {"builtin": "max"}, "arguments": [7, 8, 9, 10]}}),
        json!({"piecewise": {"cases": [[2, 3]], "otherwise": 11}}),
    ]);
    assert_eq!(wire, expected);
    assert_eq!(decode(wire).expect("its own table decodes"), expression);
}

/// Test the fields of each node are written in declaration order.
#[test]
fn expression_serializes_fields_in_declaration_order() {
    let (_, x) = build_identifier("x");
    let expression = Expression::piecewise(
        [(x.greater(0), -(&x + 1))],
        Expression::call(build_callee("f"), [Expression::all([&x, &x])]),
    )
    .expect("a valid piecewise");

    let text = serde_json::to_string(&expression).expect("the expression serializes");

    for fragment in [
        r#"{"binary":{"operation":"greater","left":0,"right":1}}"#,
        r#"{"unary":{"operation":"negate","operand":4}}"#,
        r#"{"logical":{"operation":"and","operands":[0,0]}}"#,
        r#"{"call":{"callee":{"named":"f"},"arguments":[6]}}"#,
        r#"{"piecewise":{"cases":[[2,5]],"otherwise":7}}"#,
    ] {
        assert!(text.contains(fragment), "{fragment} in {text}");
    }
}

/// Test the operation of each unary, binary and logical node is written as
/// its wire name.
#[test]
fn expression_serializes_operations_by_wire_name() {
    let (_, x) = build_identifier("x");
    let unary = !&x;
    let binary = x.floor_divide(2);
    let logical = x.or(&x);

    let unary_wire = encode(&unary);
    let binary_wire = encode(&binary);
    let logical_wire = encode(&logical);

    assert_eq!(
        unary_wire["nodes"][1]["unary"]["operation"],
        json!("logical_not")
    );
    assert_eq!(
        binary_wire["nodes"][2]["binary"]["operation"],
        json!("floor_divide")
    );
    assert_eq!(
        logical_wire["nodes"][1]["logical"]["operation"],
        json!("or")
    );
}

/// Test each literal kind serializes as its tagged string form, and decodes
/// back to the same variant.
#[rstest]
#[case::boolean(LiteralValue::from(false), json!({"bool": false}))]
#[case::integer(LiteralValue::from(5), json!({"int": "5"}))]
#[case::negative_integer(LiteralValue::from(-12), json!({"int": "-12"}))]
#[case::zero(LiteralValue::from(0), json!({"int": "0"}))]
#[case::float(LiteralValue::from(1.5), json!({"float": "1.5"}))]
#[case::integral_float(LiteralValue::from(1.0), json!({"float": "1"}))]
#[case::large_float(LiteralValue::from(1e16), json!({"float": "10000000000000000"}))]
#[case::negative_zero(LiteralValue::from(-0.0), json!({"float": "-0"}))]
#[case::nan(LiteralValue::from(f64::NAN), json!({"float": "NaN"}))]
#[case::infinity(LiteralValue::from(f64::INFINITY), json!({"float": "inf"}))]
#[case::negative_infinity(LiteralValue::from(f64::NEG_INFINITY), json!({"float": "-inf"}))]
#[case::decimal(LiteralValue::parse_text("01.50").unwrap(), json!({"decimal": "1.5"}))]
#[case::whole_decimal(LiteralValue::parse_text("100.0").unwrap(), json!({"decimal": "100"}))]
fn expression_literal_serializes_as_its_tagged_string_form(
    #[case] literal: LiteralValue,
    #[case] expected: Value,
) {
    let wire = serde_json::to_value(&literal).expect("every literal serializes");
    let restored: LiteralValue = serde_json::from_value(expected.clone()).expect("it decodes");

    assert_eq!(wire, expected);
    assert_eq!(restored, literal);
    assert_eq!(
        std::mem::discriminant(&restored),
        std::mem::discriminant(&literal)
    );
}

/// Test an integer of any size is written as a decimal string, digit for
/// digit, and reads back equal.
#[rstest]
#[case::ten_to_the_thirty("1000000000000000000000000000000")]
#[case::above_u64("18446744073709551616")]
#[case::below_i64("-9223372036854775809")]
#[case::far_below_i64("-123456789012345678901234567890123456789")]
fn literal_big_int_serializes_as_a_decimal_string(#[case] digits: &str) {
    let value: BigInt = digits.parse().expect("digits");
    let expression = build_literal(value);

    let wire = encode(&expression);
    let restored = decode(wire.clone()).expect("a valid table");

    assert_eq!(
        wire,
        build_table([build_literal_node(&json!({"int": digits}))])
    );
    assert_eq!(restored, expression);
}

/// Test every float, the non-finite ones included, round-trips through JSON
/// text and postcard to the same bits, NaN to a NaN.
#[rstest]
#[case::tenth(0.1)]
#[case::huge(1e300)]
#[case::largest(f64::MAX)]
#[case::smallest_normal(f64::MIN_POSITIVE)]
#[case::smallest_subnormal(5e-324)]
#[case::negative_zero(-0.0)]
#[case::infinity(f64::INFINITY)]
#[case::negative_infinity(f64::NEG_INFINITY)]
#[case::nan(f64::NAN)]
fn expression_float_literal_round_trips_exactly(#[case] value: f64) {
    let expression = build_literal(value);

    let text = serde_json::to_string(&expression).expect("every float serializes");
    let from_text: Expression = serde_json::from_str(&text).expect("the text decodes");
    let bytes = postcard::to_allocvec(&expression).expect("every float serializes");
    let from_bytes: Expression = postcard::from_bytes(&bytes).expect("the bytes decode");

    for restored in [from_text, from_bytes] {
        let LiteralValue::Float(restored) = expect_literal(&restored) else {
            panic!("a float literal decodes as a float");
        };
        if value.is_nan() {
            assert!(restored.is_nan(), "{text}");
        } else {
            assert_eq!(restored.to_bits(), value.to_bits(), "{text}");
        }
    }
}

/// Test a decimal string reads back as the normalized decimal it spells.
#[rstest]
#[case::trailing_zeros("1.50", "1.5")]
#[case::leading_zeros("007", "7")]
#[case::leading_point(".5", "0.5")]
#[case::trailing_point("1.", "1")]
fn expression_literal_reads_a_text_as_a_normalized_decimal(
    #[case] text: &str,
    #[case] expected: &str,
) {
    let table = build_table([build_literal_node(&json!({"decimal": text}))]);

    let restored = decode(table).expect("a decimal text decodes");

    let LiteralValue::Decimal(decimal) = expect_literal(&restored) else {
        panic!("a decimal text decodes as a decimal");
    };
    assert_eq!(decimal.to_string(), expected);
}

// =============================================================================
// Round trips
// =============================================================================

/// Test a multi-case piecewise round-trips through a JSON value.
#[test]
fn expression_piecewise_round_trips_through_a_json_value() {
    let expression = Expression::piecewise(
        [
            (build_literal(true), build_literal(1)),
            (build_literal(false), build_literal(2)),
        ],
        build_literal(0),
    )
    .expect("a valid piecewise");

    let restored = decode(encode(&expression)).expect("a valid table");

    assert_eq!(restored, expression);
}

/// Test a piecewise round-trips through JSON text.
#[test]
fn expression_piecewise_round_trips_through_json_text() {
    let expression = Expression::piecewise([(build_literal(true), build_literal(1))], 0)
        .expect("a valid piecewise");

    let text = serde_json::to_string(&expression).expect("the expression serializes");
    let restored: Expression = serde_json::from_str(&text).expect("the text deserializes");

    assert_eq!(restored, expression);
}

/// Test a piecewise nested in another round-trips.
#[test]
fn expression_nested_piecewise_round_trips() {
    let inner =
        Expression::piecewise([(build_literal(true), build_literal(1))], 0).expect("a piecewise");
    let outer = Expression::piecewise([(build_literal(false), inner)], build_literal(9))
        .expect("a piecewise");

    let restored = decode(encode(&outer)).expect("a valid table");

    assert_eq!(restored, outer);
}

/// Test a logical node serializes to its operation's wire name and its
/// operands in order, and decodes back.
#[test]
fn expression_logical_node_round_trips_through_its_wire_form() {
    let (x, x_reference) = build_identifier("x");
    let expression = Expression::any([x_reference.less(3), build_literal(true), x_reference]);

    let wire = encode(&expression);
    let restored = decode(wire.clone()).expect("a valid table");

    let expected = build_table([
        build_identifier_node(&x),
        build_literal_node(&json!({"int": "3"})),
        json!({"binary": {"operation": "less", "left": 0, "right": 1}}),
        build_literal_node(&json!({"bool": true})),
        json!({"logical": {"operation": "or", "operands": [2, 3, 0]}}),
    ]);
    assert_eq!(wire, expected);
    assert_eq!(restored, expression);
}

/// Test a decoded identifier is the identifier that was encoded.
#[test]
fn expression_round_trip_restores_the_same_identifier() {
    let (x, reference) = build_identifier("x");
    let expression = &reference * 2;

    let restored = decode(encode(&expression)).expect("a valid table");

    assert_eq!(restored, expression);
    assert_eq!(
        restored.free_identifiers().into_iter().collect::<Vec<_>>(),
        vec![x]
    );
}

// =============================================================================
// Refused tables
// =============================================================================

/// Test a table breaking a structural check is refused with the crate's own
/// message.
#[rstest]
#[case::no_nodes(build_table([]), "expression payload has no nodes")]
#[case::forward_reference(
    build_table([
        build_literal_node(&json!({"int": "1"})),
        build_literal_node(&json!({"int": "2"})),
        json!({"binary": {"operation": "add", "left": 5, "right": 1}}),
    ]),
    "node 2 refers to node 5, which does not precede it"
)]
#[case::self_reference(
    build_table([
        build_literal_node(&json!({"int": "1"})),
        json!({"unary": {"operation": "negate", "operand": 1}}),
    ]),
    "node 1 refers to node 1, which does not precede it"
)]
#[case::index_beyond_usize(
    build_table([
        build_literal_node(&json!({"int": "1"})),
        json!({"unary": {"operation": "negate", "operand": u64::MAX}}),
    ]),
    "node 1 refers to node 18446744073709551615, which does not precede it"
)]
#[case::logical_with_one_operand(
    build_table([
        build_literal_node(&json!({"bool": true})),
        json!({"logical": {"operation": "and", "operands": [0]}}),
    ]),
    "logical node 1 has 1 operands, expected at least 2"
)]
#[case::logical_with_no_operand(
    build_table([json!({"logical": {"operation": "or", "operands": []}})]),
    "logical node 0 has 0 operands, expected at least 2"
)]
#[case::piecewise_without_cases(
    build_table([
        build_literal_node(&json!({"int": "0"})),
        json!({"piecewise": {"cases": [], "otherwise": 0}}),
    ]),
    "piecewise node 1 has no cases"
)]
#[case::numeric_condition(
    build_table([
        build_literal_node(&json!({"bool": true})),
        build_literal_node(&json!({"int": "1"})),
        json!({"piecewise": {"cases": [[0, 1], [1, 1]], "otherwise": 1}}),
    ]),
    "condition of case 1 of piecewise node 2 is a non-boolean literal"
)]
#[case::decimal_condition(
    build_table([
        build_literal_node(&json!({"decimal": "1.5"})),
        json!({"piecewise": {"cases": [[0, 0]], "otherwise": 0}}),
    ]),
    "condition of case 0 of piecewise node 1 is a non-boolean literal"
)]
#[case::unreferenced_node(
    build_table([
        build_literal_node(&json!({"int": "1"})),
        build_literal_node(&json!({"int": "2"})),
        json!({"unary": {"operation": "negate", "operand": 1}}),
    ]),
    "node 0 is not referenced"
)]
fn expression_deserialize_rejects_a_malformed_table(
    #[case] table: Value,
    #[case] expected_message: &str,
) {
    assert_refused(&table, expected_message);
}

/// Test a literal outside its wire grammar is refused naming its text.
#[rstest]
#[case::leading_zero(json!({"int": "007"}), "invalid integer literal \"007\"")]
#[case::negative_zero(json!({"int": "-0"}), "invalid integer literal \"-0\"")]
#[case::plus_sign(json!({"int": "+1"}), "invalid integer literal \"+1\"")]
#[case::exponent(json!({"int": "1e3"}), "invalid integer literal \"1e3\"")]
#[case::empty_integer(json!({"int": ""}), "invalid integer literal \"\"")]
#[case::lone_sign(json!({"int": "-"}), "invalid integer literal \"-\"")]
#[case::float_text(json!({"float": "abc"}), "invalid float literal \"abc\"")]
#[case::signed_decimal(
    json!({"decimal": "-1.5"}),
    "invalid literal text \"-1.5\": expected ASCII digits with at most one decimal point"
)]
#[case::non_ascii_decimal(
    json!({"decimal": "\u{665}"}),
    "invalid literal text \"\u{665}\": expected ASCII digits with at most one decimal point"
)]
fn expression_deserialize_rejects_a_malformed_literal(
    #[case] literal: Value,
    #[case] expected_message: &str,
) {
    assert_refused(
        &build_table([build_literal_node(&literal)]),
        expected_message,
    );
}

/// Test a call naming a built-in function or no function as a named callee
/// is refused with the function-name error.
#[rstest]
#[case::builtin_name("max", "function name `max` is a built-in function")]
#[case::empty_name("", "function name is empty")]
fn expression_deserialize_rejects_an_invalid_named_callee(
    #[case] name: &str,
    #[case] expected_message: &str,
) {
    let table = build_table([json!({"call": {"callee": {"named": name}, "arguments": []}})]);

    assert_refused(&table, expected_message);
}

/// Test tables of the wrong shape, an unknown node kind, field or name, or
/// a value of the wrong type, are refused as data errors; their text is
/// serde's, so it is not pinned.
#[rstest]
#[case::unknown_node_kind(build_table([json!({"ternary": {"condition": 0}})]))]
#[case::unknown_field(build_table([
    build_literal_node(&json!({"int": "1"})),
    json!({"unary": {"operation": "negate", "operand": 0, "extra": 1}}),
]))]
#[case::unknown_table_field(json!({"nodes": [build_literal_node(&json!({"int": "1"}))], "extra": 1}))]
#[case::missing_nodes(json!({}))]
#[case::unknown_operation(build_table([
    build_literal_node(&json!({"int": "1"})),
    json!({"unary": {"operation": "-", "operand": 0}}),
]))]
#[case::unknown_literal_kind(build_table([build_literal_node(&json!({"text": "1"}))]))]
#[case::numeric_boolean(build_table([build_literal_node(&json!({"bool": 1}))]))]
#[case::integer_as_a_number(build_table([build_literal_node(&json!({"int": 1}))]))]
#[case::float_as_a_number(build_table([build_literal_node(&json!({"float": 1.5}))]))]
#[case::unknown_builtin(build_table([
    json!({"call": {"callee": {"builtin": "softplus"}, "arguments": []}}),
]))]
#[case::negative_index(build_table([
    build_literal_node(&json!({"int": "1"})),
    json!({"unary": {"operation": "negate", "operand": -1}}),
]))]
#[case::identifier_without_name_hint(build_table([json!({"identifier": {"id": 1}})]))]
#[case::node_not_a_map(build_table([json!(5)]))]
fn expression_deserialize_rejects_an_unknown_node_kind(#[case] table: Value) {
    let error = decode(table.clone()).expect_err("the table is refused");

    assert!(error.is_data(), "a data error for {table}: {error}");
}

/// Test a linear chain of 1,000,000 nodes decodes from JSON text on a
/// 256 KiB stack.
#[test]
fn expression_million_node_chain_decodes_on_a_small_stack() {
    const NODES: usize = 1_000_000;
    let mut text = String::from(r#"{"nodes":[{"literal":{"int":"0"}}"#);
    for operand in 0..NODES - 1 {
        write!(
            text,
            r#",{{"unary":{{"operation":"negate","operand":{operand}}}}}"#
        )
        .expect("writing to a string succeeds");
    }
    text.push_str("]}");

    let depth = run_on_stack(WIRE_STACK_BYTES, move || {
        let chain: Expression = serde_json::from_str(&text).expect("the chain decodes");
        let mut depth = 0;
        let mut node = &chain;
        while let ExpressionKind::Unary(unary) = node.kind() {
            depth += 1;
            node = unary.operand();
        }
        depth
    });

    assert_eq!(depth, NODES - 1);
}

/// Test every strict prefix of an encoded expression is a postcard error,
/// not a panic.
#[test]
fn expression_truncated_postcard_bytes_are_an_error_not_a_panic() {
    let (_, x) = build_identifier("x");
    let expression = Expression::piecewise(
        [(x.less(1.5), Expression::call(BuiltinFunction::Floor, [&x]))],
        Expression::all([&x, &-&x]),
    )
    .expect("a valid piecewise");
    let bytes = postcard::to_allocvec(&expression).expect("the expression serializes");

    for length in 0..bytes.len() {
        let result = postcard::from_bytes::<Expression>(&bytes[..length]);

        assert!(result.is_err(), "a {length}-byte prefix decoded");
    }
}

/// Test a postcard table whose child index is out of range is refused.
#[test]
fn expression_postcard_index_out_of_range_is_refused() {
    let negation = -build_literal(1);
    let mut bytes = postcard::to_allocvec(&negation).expect("the expression serializes");
    let last = bytes.last_mut().expect("the table is not empty");
    assert_eq!(*last, 0, "the negation's operand index is the last byte");
    *last = 5;

    let result = postcard::from_bytes::<Expression>(&bytes);

    assert!(result.is_err(), "an index out of range decoded");
}

/// Test decoding a table of unary and binary nodes rebuilds the operations
/// given.
#[test]
fn expression_decode_rebuilds_the_operations_given() {
    let table = build_table([
        build_literal_node(&json!({"int": "2"})),
        json!({"unary": {"operation": "negate", "operand": 0}}),
        json!({"binary": {"operation": "floor_mod", "left": 1, "right": 0}}),
    ]);

    let restored = decode(table).expect("a valid table");

    let two = build_literal(2);
    assert_eq!(
        restored,
        Expression::new_binary(
            BinaryOperation::FloorMod,
            Expression::new_unary(UnaryOperation::Negate, &two),
            &two
        )
    );
}
