//! Tests for the wire form of expressions: the exact shape of each node
//! kind, literal values of every kind, round trips through JSON values and
//! JSON text, refused payloads, and the identifier counter left untouched by
//! a refused payload.
//!
//! Public API only (`fhy_core::expr`, `fhy_core::identifier`).

use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::support::expression as expression_support;
use crate::support::stack as stack_support;

use expression_support::{
    DEEP_TREE_DEPTH, SERIALIZATION_STACK_BYTES, build_decimal_literal, build_deep_sum,
    build_identifier, build_literal,
};
use fhy_core::expr::{
    BigInt, BinaryOperation, Expression, ExpressionBuildError, ExpressionKind, LiteralValue,
    UnaryOperation, build_call, build_piecewise,
};
use fhy_core::identifier::Identifier;
use rstest::rstest;
use serde_json::{Value, json};
use stack_support::run_on_stack;

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
fn expect_literal(expression: &Expression) -> &LiteralValue {
    let ExpressionKind::Literal(literal) = expression.kind() else {
        panic!("expected a literal, got {expression:?}");
    };
    literal
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
                build_decimal_literal("2.50"),
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
                    "arguments": [x_wire, build_literal_wire(&json!(1.5)), build_literal_wire(&json!("2.5"))]
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
#[case::integer_text(
    build_literal(LiteralValue::parse_text("05").expect("an integer text")),
    json!(5)
)]
#[case::decimal(build_decimal_literal("1.50"), json!("1.5"))]
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
    assert!(matches!(expect_literal(&restored), LiteralValue::Int(restored) if *restored == value));
}

/// Test an integer token reads as an integer literal and a float token as a
/// float literal, never the one as the other.
#[rstest]
#[case::integer("5", "int")]
#[case::float_with_point("5.0", "float")]
#[case::float_with_exponent("5e0", "float")]
#[case::negative_integer("-5", "int")]
fn expression_literal_keeps_the_json_number_kind(#[case] token: &str, #[case] kind: &str) {
    let restored: Expression =
        serde_json::from_str(&build_literal_wire_text(token)).expect("the text deserializes");

    match (expect_literal(&restored), kind) {
        (LiteralValue::Int(value), "int") => assert_eq!(value.to_string(), token),
        (LiteralValue::Float(value), "float") => assert_eq!(value.to_bits(), 5.0_f64.to_bits()),
        (actual, _) => panic!("expected a {kind} literal from {token}, got {actual:?}"),
    }
}

/// Return the wire text of a literal whose value is the JSON number `token`.
fn build_literal_wire_text(token: &str) -> String {
    format!("{{\"__type__\":\"literal_expression\",\"__data__\":{{\"value\":{token}}}}}")
}

/// Test a float token beyond the range of an f64 is refused with a message
/// naming the token as `serde_json` spells it.
#[rstest]
#[case::positive("1e400", "in `value`: the float 1e+400 does not fit an f64")]
#[case::negative("-1e400", "in `value`: the float -1e+400 does not fit an f64")]
fn expression_literal_refuses_a_float_token_beyond_the_f64_range(
    #[case] token: &str,
    #[case] expected: &str,
) {
    let result = serde_json::from_str::<Expression>(&build_literal_wire_text(token));

    let error = result.expect_err("the float does not fit an f64");
    assert_eq!(error.to_string(), expected);
}

/// Test a float token below the smallest subnormal reads as a float zero of
/// the token's sign.
#[rstest]
#[case::positive("1e-400", 0.0)]
#[case::negative("-1e-400", -0.0)]
fn expression_literal_reads_an_underflowing_float_token_as_a_signed_zero(
    #[case] token: &str,
    #[case] expected: f64,
) {
    let restored: Expression =
        serde_json::from_str(&build_literal_wire_text(token)).expect("the text deserializes");

    let LiteralValue::Float(value) = expect_literal(&restored) else {
        panic!("expected a float literal from {token}");
    };
    assert_eq!(value.to_bits(), expected.to_bits());
}

/// Test a text literal reads back as the normalized decimal it spells,
/// keeping no spelling.
#[rstest]
#[case::integer_text("05", 5, 0)]
#[case::decimal_text("1.50", 15, -1)]
#[case::bare_point(".5", 5, -1)]
fn expression_literal_reads_a_text_as_a_normalized_decimal(
    #[case] text: &str,
    #[case] expected_coefficient: i64,
    #[case] expected_exponent: i64,
) {
    let restored = decode(build_literal_wire(&json!(text))).expect("a valid payload");

    let LiteralValue::Decimal(decimal) = expect_literal(&restored) else {
        panic!("expected a decimal from {text:?}, got {restored:?}");
    };
    assert_eq!(decimal.coefficient(), &BigInt::from(expected_coefficient));
    assert_eq!(decimal.exponent(), expected_exponent);
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
    run_on_stack(SERIALIZATION_STACK_BYTES, || {
        let (_, x) = build_identifier("x");
        let tree = build_deep_sum(&x, DEEP_TREE_DEPTH);

        let restored = decode(encode(&tree)).expect("a valid payload");

        assert_eq!(restored, tree);
    });
}

/// Test JSON text decodes an expression nested as deep as `serde_json`'s
/// nesting limit allows, two JSON levels per tree level, and refuses one
/// level more with an error rather than a crash; a JSON value parsed from
/// that text meets the same limit.
#[test]
fn expression_json_text_decodes_up_to_the_serde_json_nesting_limit() {
    let deepest = build_deep_sum(&build_literal(1), 62);
    let too_deep = build_deep_sum(&build_literal(1), 63);
    let deepest_text = serde_json::to_string(&deepest).expect("the expression serializes");
    let too_deep_text = serde_json::to_string(&too_deep).expect("the expression serializes");

    let decoded: Expression =
        serde_json::from_str(&deepest_text).expect("62 levels over a leaf decode");
    let refusal = serde_json::from_str::<Expression>(&too_deep_text);
    let value_refusal = serde_json::from_str::<Value>(&too_deep_text);

    assert_eq!(decoded, deepest);
    let error = refusal.expect_err("63 levels over a leaf exceed the nesting limit");
    assert!(
        error.to_string().starts_with("recursion limit exceeded"),
        "{error}"
    );
    let error = value_refusal.expect_err("63 levels over a leaf exceed the nesting limit");
    assert!(
        error.to_string().starts_with("recursion limit exceeded"),
        "{error}"
    );
}

// =============================================================================
// Refused payloads
// =============================================================================

/// Serializes the tests that restore an id far ahead of the counter or check
/// the counter has not passed one: a far-ahead id one test restores passes
/// the ids every other such test reserved.
static ID_COUNTER_LOCK: Mutex<()> = Mutex::new(());

/// Hold the id counter against every other counter-observing test.
fn hold_id_counter() -> MutexGuard<'static, ()> {
    ID_COUNTER_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Return an id far ahead of the id counter, for a test that holds
/// [`hold_id_counter`].
fn reserve_ahead_id() -> u64 {
    Identifier::new("probe").id() + (1 << 40)
}

/// Return whether the id counter has moved past `id`, drawing one id to find
/// out.
fn has_counter_passed(id: u64) -> bool {
    Identifier::new("counter-probe").id() > id
}

/// Return the wire form of a reference to an identifier with the id `id`.
fn build_identifier_id_wire(id: u64) -> Value {
    json!({
        "__type__": "identifier_expression",
        "__data__": {"identifier": {"id": id, "name_hint": "ahead"}}
    })
}

/// Return the wire form of `left + right`.
fn build_sum_wire(left: &Value, right: &Value) -> Value {
    json!({
        "__type__": "binary_expression",
        "__data__": {"operation": "add", "left": left, "right": right}
    })
}

/// Assert the payload `build_refused` returns, given a reference to an
/// identifier far ahead of the counter and decoded as the right operand of a
/// sum whose left operand refers to another such identifier, is refused with
/// ``in `right`: `` followed by `expected_message`, and leaves the counter
/// short of both ids.
///
/// Building the sum would restore the left operand before the right one, so
/// a refusal only raised while building the right operand fails the counter
/// check.
fn assert_refused_before_any_restore(
    build_refused: impl FnOnce(Value) -> Value,
    expected_message: &str,
) {
    let _guard = hold_id_counter();
    let left_id = reserve_ahead_id();
    let inner_id = reserve_ahead_id();
    let refused = build_refused(build_identifier_id_wire(inner_id));

    let result = decode(build_sum_wire(&build_identifier_id_wire(left_id), &refused));

    let error = result.expect_err("the payload is refused");
    assert_eq!(error.to_string(), format!("in `right`: {expected_message}"));
    assert!(
        !has_counter_passed(left_id.min(inner_id)),
        "the refused payload advanced the counter"
    );
}

/// Test malformed literal payloads are refused, naming the defect, before any
/// identifier is restored.
#[rstest]
#[case::missing_value(
    json!({"__type__": "literal_expression", "__data__": {}}),
    "missing field `value` in a literal"
)]
#[case::list_value(
    build_literal_wire(&json!([1, 2, 3])),
    "in `value`: expected a Boolean, a number, or a numeric text as a literal value, got a list"
)]
#[case::null_value(
    build_literal_wire(&Value::Null),
    "in `value`: expected a Boolean, a number, or a numeric text as a literal value, got null"
)]
#[case::object_value(
    build_literal_wire(&json!({"value": 1})),
    "in `value`: expected a Boolean, a number, or a numeric text as a literal value, got a map"
)]
#[case::text_outside_the_grammar(
    build_literal_wire(&json!("abc")),
    "in `value`: invalid literal text \"abc\": expected ASCII digits with at most one decimal \
     point"
)]
#[case::signed_text(
    build_literal_wire(&json!("-5")),
    "in `value`: invalid literal text \"-5\": expected ASCII digits with at most one decimal \
     point"
)]
#[case::non_ascii_digit_text(
    build_literal_wire(&json!("\u{665}")),
    "in `value`: invalid literal text \"\u{665}\": expected ASCII digits with at most one \
     decimal point"
)]
#[case::extra_key(
    json!({"__type__": "literal_expression", "__data__": {"value": 1, "extra": 1}}),
    "unknown field `extra` in a literal"
)]
fn expression_deserialize_rejects_a_malformed_literal(
    #[case] refused: Value,
    #[case] expected_message: &str,
) {
    assert_refused_before_any_restore(|_| refused, expected_message);
}

/// Test a payload with an unknown type id is refused with a message naming
/// the id.
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

    assert_eq!(
        error.to_string(),
        "unknown expression type id `ternary_expression`"
    );
}

/// Test payloads breaking a node's structure are refused, naming the defect,
/// before any identifier is restored.
#[rstest]
#[case::length_mismatch(json!({
    "__type__": "piecewise_expression",
    "__data__": {
        "conditions": [build_literal_wire(&json!(true)), build_literal_wire(&json!(false))],
        "values": [build_literal_wire(&json!(1))],
        "otherwise": build_literal_wire(&json!(0))
    }
}), "a piecewise has 2 conditions but 1 values")]
#[case::unknown_operation(json!({
    "__type__": "unary_expression",
    "__data__": {"operation": "-", "operand": build_literal_wire(&json!(1))}
}), "in `operation`: invalid value: string \"-\", expected a unary operation name such as \
     negate or logical_not")]
#[case::unknown_unary_operation_name(json!({
    "__type__": "unary_expression",
    "__data__": {"operation": "not", "operand": build_literal_wire(&json!(1))}
}), "in `operation`: invalid value: string \"not\", expected a unary operation name such as \
     negate or logical_not")]
#[case::unknown_binary_operation_name(json!({
    "__type__": "binary_expression",
    "__data__": {
        "operation": "plus",
        "left": build_literal_wire(&json!(1)),
        "right": build_literal_wire(&json!(2))
    }
}), "in `operation`: invalid value: string \"plus\", expected a binary operation name such as \
     add or floor_divide")]
#[case::arguments_not_a_list(json!({
    "__type__": "call_expression",
    "__data__": {"function_name": "f", "arguments": build_literal_wire(&json!(1))}
}), "in `arguments`: expected a list, got a map")]
#[case::missing_operand(json!({
    "__type__": "binary_expression",
    "__data__": {"operation": "add", "left": build_literal_wire(&json!(1))}
}), "missing field `right` in a binary node")]
#[case::extra_field(json!({
    "__type__": "call_expression",
    "__data__": {"function_name": "f", "arguments": [], "extra": 1}
}), "unknown field `extra` in a call")]
#[case::extra_top_level_key(json!({
    "__type__": "literal_expression",
    "__data__": {"value": 1},
    "extra": 1
}), "unknown field `extra` in an expression")]
#[case::missing_data(
    json!({"__type__": "literal_expression"}),
    "missing field `__data__` in an expression"
)]
#[case::identifier_without_name_hint(json!({
    "__type__": "identifier_expression",
    "__data__": {"identifier": {"id": 1}}
}), "in `identifier`: missing field `name_hint`")]
#[case::child_not_a_map(json!({
    "__type__": "unary_expression",
    "__data__": {"operation": "negate", "operand": 5}
}), "in `operand`: expected the fields of an expression as a map, got a number")]
fn expression_deserialize_rejects_a_malformed_node(
    #[case] refused: Value,
    #[case] expected_message: &str,
) {
    assert_refused_before_any_restore(|_| refused, expected_message);
}

/// Test payloads breaking a node invariant are refused with the error the
/// node's constructor reports, before any identifier is restored, even one
/// inside the refused node.
#[rstest]
#[case::no_cases(
    |ahead: Value| json!({
        "__type__": "piecewise_expression",
        "__data__": {"conditions": [], "values": [], "otherwise": ahead}
    }),
    ExpressionBuildError::EmptyPiecewise
)]
#[case::numeric_condition(
    |ahead: Value| json!({
        "__type__": "piecewise_expression",
        "__data__": {
            "conditions": [build_literal_wire(&json!(true)), build_literal_wire(&json!(1))],
            "values": [ahead, build_literal_wire(&json!(1))],
            "otherwise": build_literal_wire(&json!(0))
        }
    }),
    ExpressionBuildError::NonBooleanConditionLiteral { case_index: 1 }
)]
#[case::text_condition(
    |ahead: Value| json!({
        "__type__": "piecewise_expression",
        "__data__": {
            "conditions": [build_literal_wire(&json!("1.5"))],
            "values": [build_literal_wire(&json!(1))],
            "otherwise": ahead
        }
    }),
    ExpressionBuildError::NonBooleanConditionLiteral { case_index: 0 }
)]
#[case::empty_function_name(
    |ahead: Value| json!({
        "__type__": "call_expression",
        "__data__": {"function_name": "", "arguments": [ahead]}
    }),
    ExpressionBuildError::EmptyFunctionName
)]
fn expression_deserialize_rejects_a_node_its_constructor_refuses(
    #[case] build_refused: fn(Value) -> Value,
    #[case] expected: ExpressionBuildError,
) {
    assert_refused_before_any_restore(build_refused, &expected.to_string());
}

/// Test a piecewise nested below a sum is refused for a numeric condition,
/// naming the path to it, before any identifier is restored.
#[test]
fn expression_deserialize_rejects_a_nested_numeric_condition() {
    let nested = build_sum_wire(
        &build_literal_wire(&json!(1)),
        &json!({
            "__type__": "piecewise_expression",
            "__data__": {
                "conditions": [build_literal_wire(&json!(1))],
                "values": [build_literal_wire(&json!(1))],
                "otherwise": build_literal_wire(&json!(0))
            }
        }),
    );
    let expected = format!(
        "in `right`: {}",
        ExpressionBuildError::NonBooleanConditionLiteral { case_index: 0 }
    );

    assert_refused_before_any_restore(|_| nested, &expected);
}

/// Test a payload refused after an identifier in it was read leaves the id
/// counter untouched, and the same payload made valid restores the id.
#[test]
fn expression_deserialize_restores_no_identifier_from_a_refused_payload() {
    let _guard = hold_id_counter();
    let ahead_id = reserve_ahead_id();
    let identifier_wire = build_identifier_id_wire(ahead_id);
    let refused = build_sum_wire(
        &identifier_wire,
        &json!({
            "__type__": "piecewise_expression",
            "__data__": {
                "conditions": [build_literal_wire(&json!(7))],
                "values": [build_literal_wire(&json!(1))],
                "otherwise": build_literal_wire(&json!(0))
            }
        }),
    );
    let accepted = build_sum_wire(&identifier_wire, &build_literal_wire(&json!(1)));

    let refusal = decode(refused);
    let passed_after_refusal = has_counter_passed(ahead_id);
    let restored = decode(accepted).expect("a valid payload");
    let passed_after_restore = has_counter_passed(ahead_id);

    refusal.expect_err("the numeric case condition is refused");
    assert!(
        !passed_after_refusal,
        "the refused payload advanced the counter"
    );
    assert!(
        passed_after_restore,
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
