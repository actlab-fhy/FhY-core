//! Tests for the wire form of expressions: the exact shape of each node
//! kind, literal values of every kind, round trips through JSON values and
//! JSON text, refused payloads, and the identifier counter left untouched by
//! a refused payload.
//!
//! Public API only (`fhy_core::symbolic::expression`,
//! `fhy_core::identifier`).

#[path = "common/expression.rs"]
pub mod expression_support;

use expression_support::{
    DEEP_TREE_DEPTH, SERIALIZATION_STACK_BYTES, build_deep_sum, build_identifier, build_literal,
    build_text_literal, run_on_large_stack,
};
use fhy_core::identifier::Identifier;
use fhy_core::symbolic::expression::{
    BinaryOperation, Expression, ExpressionKind, LiteralKind, UnaryOperation, build_call,
    build_piecewise,
};
use num_bigint::BigInt;
use rstest::rstest;
use serde_json::{Value, json};

/// Serialize `expression` to a JSON value.
fn encode(expression: &Expression) -> Value {
    serde_json::to_value(expression).expect("the expression serializes")
}

/// Deserialize an expression from a JSON value.
fn decode(value: Value) -> Result<Expression, serde_json::Error> {
    serde_json::from_value(value)
}

/// Return the wire form of a literal holding `value`.
fn build_literal_wire(value: &Value) -> Value {
    json!({"__type__": "literal_expression", "__data__": {"value": value}})
}

/// Return the wire form of a reference to `identifier`.
fn build_identifier_wire(identifier: &Identifier) -> Value {
    json!({
        "__type__": "identifier_expression",
        "__data__": {"identifier": {"id": identifier.id(), "name_hint": identifier.name_hint()}}
    })
}

/// Return the literal an expression refers to, failing the test otherwise.
fn expect_literal(expression: &Expression) -> LiteralKind<'_> {
    let ExpressionKind::Literal(literal) = expression.kind() else {
        panic!("expected a literal, got {expression:?}");
    };
    literal.kind()
}

// =============================================================================
// Shape of each node kind
// =============================================================================

/// Test a Boolean literal serializes to its pinned dict and decodes back.
#[test]
fn expression_literal_round_trips_through_its_wire_form() {
    let expression = build_literal(true);
    let expected = build_literal_wire(&json!(true));

    let wire = encode(&expression);
    let restored = decode(expected.clone()).expect("a valid payload");

    assert_eq!(wire, expected);
    assert_eq!(restored, expression);
}

/// Test each node kind serializes to its type id and fields, in order.
#[test]
fn expression_serializes_every_node_kind_in_its_wire_shape() {
    let (x, x_reference) = build_identifier("x");
    let x_wire = build_identifier_wire(&x);
    let expression = build_piecewise(
        [(x_reference.less(3), -&x_reference)],
        build_call(
            "max",
            [
                x_reference.clone(),
                build_literal(1.5),
                build_text_literal("2.50"),
            ],
        )
        .expect("a named call"),
    )
    .expect("a valid piecewise");

    let wire = encode(&expression);

    let expected = json!({
        "__type__": "piecewise_expression",
        "__data__": {
            "conditions": [{
                "__type__": "binary_expression",
                "__data__": {"operation": "less", "left": x_wire, "right": build_literal_wire(&json!(3))}
            }],
            "values": [{
                "__type__": "unary_expression",
                "__data__": {"operation": "negate", "operand": x_wire}
            }],
            "otherwise": {
                "__type__": "call_expression",
                "__data__": {
                    "function_name": "max",
                    "arguments": [x_wire, build_literal_wire(&json!(1.5)), build_literal_wire(&json!("2.50"))]
                }
            }
        }
    });
    assert_eq!(wire, expected);
}

/// Test the fields of each node are written in declaration order.
#[test]
fn expression_serializes_fields_in_declaration_order() {
    let (_, x) = build_identifier("x");
    let expression = build_piecewise(
        [(x.greater(0), -(&x + 1))],
        build_call("f", [&x]).expect("a call"),
    )
    .expect("a valid piecewise");

    let text = serde_json::to_string(&expression).expect("the expression serializes");

    let order = [
        "\"__type__\":\"piecewise_expression\"",
        "\"conditions\"",
        "\"operation\":\"greater\"",
        "\"left\"",
        "\"right\"",
        "\"values\"",
        "\"operation\":\"negate\"",
        "\"operand\"",
        "\"otherwise\"",
        "\"function_name\":\"f\"",
        "\"arguments\"",
    ];
    let positions: Vec<usize> = order
        .iter()
        .map(|needle| {
            text.find(needle)
                .unwrap_or_else(|| panic!("{needle} missing from {text}"))
        })
        .collect();
    assert!(positions.windows(2).all(|pair| pair[0] < pair[1]), "{text}");
    assert!(text.starts_with("{\"__type__\""), "{text}");
}

/// Test each literal kind serializes to the JSON value of its kind.
#[rstest]
#[case::bool(build_literal(false), json!(false))]
#[case::integer(build_literal(-5), json!(-5))]
#[case::float(build_literal(2.0), json!(2.0))]
#[case::negative_zero(build_literal(-0.0), json!(-0.0))]
#[case::integer_text(build_text_literal("05"), json!("05"))]
#[case::decimal_text(build_text_literal("1.50"), json!("1.50"))]
fn expression_literal_serializes_as_the_json_value_of_its_kind(
    #[case] expression: Expression,
    #[case] value: Value,
) {
    let wire = encode(&expression);

    assert_eq!(wire, build_literal_wire(&value));
}

/// Test an integer beyond the `i64` range is written as a JSON integer and
/// read back exactly.
#[rstest]
#[case::ten_to_the_thirty("1000000000000000000000000000000")]
#[case::below_i64_min("-9223372036854775809")]
#[case::u64_max("18446744073709551615")]
fn expression_literal_writes_a_big_integer_as_a_json_integer(#[case] digits: &str) {
    let value: BigInt = digits.parse().expect("digits");
    let expression = build_literal(value.clone());

    let text = serde_json::to_string(&expression).expect("the expression serializes");
    let restored: Expression = serde_json::from_str(&text).expect("the text deserializes");

    assert_eq!(
        text,
        format!("{{\"__type__\":\"literal_expression\",\"__data__\":{{\"value\":{digits}}}}}")
    );
    assert_eq!(expect_literal(&restored), LiteralKind::Int(&value));
}

/// Test an integer token reads as an integer literal and a float token as a
/// float literal, never the one as the other.
#[rstest]
#[case::integer("5", "int")]
#[case::float_with_point("5.0", "float")]
#[case::float_with_exponent("5e0", "float")]
#[case::negative_integer("-5", "int")]
fn expression_literal_keeps_the_json_number_kind(#[case] token: &str, #[case] kind: &str) {
    let text =
        format!("{{\"__type__\":\"literal_expression\",\"__data__\":{{\"value\":{token}}}}}");

    let restored: Expression = serde_json::from_str(&text).expect("the text deserializes");

    match (expect_literal(&restored), kind) {
        (LiteralKind::Int(value), "int") => assert_eq!(value.to_string(), token),
        (LiteralKind::Float(value), "float") => assert_eq!(value.to_bits(), 5.0_f64.to_bits()),
        (actual, _) => panic!("expected a {kind} literal from {token}, got {actual:?}"),
    }
}

/// Test a text literal reads back with its spelling and bucket.
#[rstest]
#[case::integer_text("05", LiteralKind::IntegerText("05"))]
#[case::decimal_text("1.50", LiteralKind::DecimalText("1.50"))]
#[case::bare_point(".5", LiteralKind::DecimalText(".5"))]
fn expression_literal_reads_a_text_with_its_spelling(
    #[case] text: &str,
    #[case] expected: LiteralKind<'static>,
) {
    let restored = decode(build_literal_wire(&json!(text))).expect("a valid payload");

    assert_eq!(expect_literal(&restored), expected);
}

/// Test a NaN or infinite float literal fails to serialize.
#[rstest]
#[case::nan(f64::NAN)]
#[case::infinity(f64::INFINITY)]
#[case::negative_infinity(f64::NEG_INFINITY)]
fn expression_literal_refuses_to_serialize_a_non_finite_float(#[case] value: f64) {
    let expression = build_literal(value) + 1;

    let result = serde_json::to_value(&expression);

    result.expect_err("a non-finite float has no JSON form");
}

// =============================================================================
// Round trips
// =============================================================================

/// Test a multi-case piecewise round-trips through a JSON value.
#[test]
fn expression_piecewise_round_trips_through_a_json_value() {
    let expression = build_piecewise(
        [
            (build_literal(true), build_literal(1)),
            (build_literal(false), build_literal(2)),
        ],
        build_literal(0),
    )
    .expect("a valid piecewise");

    let restored = decode(encode(&expression)).expect("a valid payload");

    assert_eq!(restored, expression);
}

/// Test a piecewise round-trips through JSON text.
#[test]
fn expression_piecewise_round_trips_through_json_text() {
    let expression =
        build_piecewise([(build_literal(true), build_literal(1))], 0).expect("a valid piecewise");

    let text = serde_json::to_string(&expression).expect("the expression serializes");
    let restored: Expression = serde_json::from_str(&text).expect("the text deserializes");

    assert_eq!(restored, expression);
}

/// Test a piecewise nested in another round-trips.
#[test]
fn expression_nested_piecewise_round_trips() {
    let inner = build_piecewise([(build_literal(true), build_literal(1))], 0).expect("a piecewise");
    let outer =
        build_piecewise([(build_literal(false), inner)], build_literal(9)).expect("a piecewise");

    let restored = decode(encode(&outer)).expect("a valid payload");

    assert_eq!(restored, outer);
}

/// Test a decoded identifier is the identifier that was encoded.
#[test]
fn expression_round_trip_restores_the_same_identifier() {
    let (x, reference) = build_identifier("x");
    let expression = &reference * 2;

    let restored = decode(encode(&expression)).expect("a valid payload");

    assert_eq!(restored, expression);
    assert_eq!(
        restored.free_identifiers().into_iter().collect::<Vec<_>>(),
        vec![x]
    );
}

/// Test a tree thousands of levels deep round-trips.
#[test]
fn expression_deep_tree_round_trips_through_a_json_value() {
    run_on_large_stack(SERIALIZATION_STACK_BYTES, || {
        let (_, x) = build_identifier("x");
        let tree = build_deep_sum(&x, DEEP_TREE_DEPTH);

        let restored = decode(encode(&tree)).expect("a valid payload");

        assert_eq!(restored, tree);
    });
}

// =============================================================================
// Refused payloads
// =============================================================================

/// Test malformed literal payloads are refused.
#[rstest]
#[case::missing_value(json!({"__type__": "literal_expression", "__data__": {}}))]
#[case::list_value(build_literal_wire(&json!([1, 2, 3])))]
#[case::null_value(build_literal_wire(&Value::Null))]
#[case::object_value(build_literal_wire(&json!({"value": 1})))]
#[case::text_outside_the_grammar(build_literal_wire(&json!("abc")))]
#[case::signed_text(build_literal_wire(&json!("-5")))]
#[case::non_ascii_digit_text(build_literal_wire(&json!("\u{665}")))]
#[case::extra_key(json!({"__type__": "literal_expression", "__data__": {"value": 1, "extra": 1}}))]
fn expression_deserialize_rejects_a_malformed_literal(#[case] payload: Value) {
    let result = decode(payload);

    result.expect_err("the payload is refused");
}

/// Test a payload with an unknown type id is refused, naming the id.
#[test]
fn expression_deserialize_rejects_an_unknown_type_id() {
    let payload = json!({
        "__type__": "ternary_expression",
        "__data__": {
            "condition": build_literal_wire(&json!(true)),
            "true_value": build_literal_wire(&json!(1)),
            "false_value": build_literal_wire(&json!(2))
        }
    });

    let error = decode(payload).expect_err("the type id is unknown");

    assert!(error.to_string().contains("ternary_expression"), "{error}");
}

/// Test payloads breaking a node's structure or invariants are refused.
#[rstest]
#[case::length_mismatch(json!({
    "__type__": "piecewise_expression",
    "__data__": {
        "conditions": [build_literal_wire(&json!(true)), build_literal_wire(&json!(false))],
        "values": [build_literal_wire(&json!(1))],
        "otherwise": build_literal_wire(&json!(0))
    }
}))]
#[case::no_cases(json!({
    "__type__": "piecewise_expression",
    "__data__": {"conditions": [], "values": [], "otherwise": build_literal_wire(&json!(0))}
}))]
#[case::numeric_condition(json!({
    "__type__": "piecewise_expression",
    "__data__": {
        "conditions": [build_literal_wire(&json!(1))],
        "values": [build_literal_wire(&json!(1))],
        "otherwise": build_literal_wire(&json!(0))
    }
}))]
#[case::empty_function_name(json!({
    "__type__": "call_expression",
    "__data__": {"function_name": "", "arguments": []}
}))]
#[case::unknown_operation(json!({
    "__type__": "unary_expression",
    "__data__": {"operation": "-", "operand": build_literal_wire(&json!(1))}
}))]
#[case::missing_operand(json!({
    "__type__": "binary_expression",
    "__data__": {"operation": "add", "left": build_literal_wire(&json!(1))}
}))]
#[case::extra_field(json!({
    "__type__": "call_expression",
    "__data__": {"function_name": "f", "arguments": [], "extra": 1}
}))]
#[case::extra_top_level_key(json!({
    "__type__": "literal_expression",
    "__data__": {"value": 1},
    "extra": 1
}))]
#[case::missing_data(json!({"__type__": "literal_expression"}))]
#[case::identifier_without_name_hint(json!({
    "__type__": "identifier_expression",
    "__data__": {"identifier": {"id": 1}}
}))]
#[case::child_not_a_map(json!({
    "__type__": "unary_expression",
    "__data__": {"operation": "negate", "operand": 5}
}))]
fn expression_deserialize_rejects_a_malformed_node(#[case] payload: Value) {
    let result = decode(payload);

    result.expect_err("the payload is refused");
}

/// Test a payload refused after an identifier in it was read leaves the id
/// counter untouched, and the same payload made valid restores the id.
#[test]
fn expression_deserialize_restores_no_identifier_from_a_refused_payload() {
    let ahead_id = Identifier::new("probe").id() + (1 << 40);
    let identifier_wire = json!({
        "__type__": "identifier_expression",
        "__data__": {"identifier": {"id": ahead_id, "name_hint": "ahead"}}
    });
    let refused = json!({
        "__type__": "binary_expression",
        "__data__": {
            "operation": "add",
            "left": identifier_wire,
            "right": {
                "__type__": "piecewise_expression",
                "__data__": {
                    "conditions": [build_literal_wire(&json!(7))],
                    "values": [build_literal_wire(&json!(1))],
                    "otherwise": build_literal_wire(&json!(0))
                }
            }
        }
    });
    let accepted = json!({
        "__type__": "binary_expression",
        "__data__": {"operation": "add", "left": identifier_wire, "right": build_literal_wire(&json!(1))}
    });

    let refusal = decode(refused);
    let next_after_refusal = Identifier::new("after_refusal").id();
    let restored = decode(accepted).expect("a valid payload");
    let next_after_restore = Identifier::new("after_restore").id();

    refusal.expect_err("the numeric case condition is refused");
    assert!(
        next_after_refusal < ahead_id,
        "the refused payload advanced the counter"
    );
    assert!(
        next_after_restore > ahead_id,
        "the accepted payload did not restore its id"
    );
    let ExpressionKind::Binary(node) = restored.kind() else {
        panic!("expected a binary node, got {restored:?}");
    };
    let ExpressionKind::Identifier(identifier) = node.left().kind() else {
        panic!("expected an identifier reference, got {:?}", node.left());
    };
    assert_eq!(identifier.id(), ahead_id);
    assert_eq!(identifier.name_hint(), "ahead");
}

/// Test the operation of each unary and binary node is written as its wire
/// name.
#[test]
fn expression_serializes_operations_by_wire_name() {
    let (_, x) = build_identifier("x");
    let unary = Expression::new_unary(UnaryOperation::LogicalNot, &x);
    let binary = Expression::new_binary(BinaryOperation::FloorDivide, &x, 2);

    let unary_wire = encode(&unary);
    let binary_wire = encode(&binary);

    assert_eq!(unary_wire["__data__"]["operation"], json!("logical_not"));
    assert_eq!(binary_wire["__data__"]["operation"], json!("floor_divide"));
}
