//! Tests for the serde form of values, constraints and systems
//! (`fhy_core::constraint::wire`).

use crate::support::constraint::{TestOpaque, member};
use crate::support::foreign::{TestResolver, WireCustom, WireToken};

use fhy_core::constraint::wire::{ConstraintData, ConstraintSystemData, ValueData};
use fhy_core::constraint::{
    Constraint, ConstraintSystem, EquationConstraint, Member, MemberSet, Polarity, SetConstraint,
    Value,
};
use fhy_core::expression::{BigInt, Decimal, Expression};
use fhy_core::foreign::BuildError;
use fhy_core::identifier::Identifier;
use rstest::rstest;

fn restored(id: u64, name: &str) -> Identifier {
    Identifier::try_restore(id, name).expect("the id is below the cap")
}

fn decimal(text: &str) -> Value {
    Value::Decimal(text.parse::<Decimal>().expect("a decimal"))
}

fn text_of(value: &impl serde::Serialize) -> String {
    serde_json::to_string(value).expect("encodes")
}

#[rstest]
#[case::bool(Value::Bool(true), r#"{"bool":true}"#)]
#[case::big_int(
    Value::Int(BigInt::from(2).pow(100)),
    r#"{"int":"1267650600228229401496703205376"}"#
)]
#[case::negative_int(Value::Int(BigInt::from(-12)), r#"{"int":"-12"}"#)]
#[case::float(Value::Float(2.5), r#"{"float":"2.5"}"#)]
#[case::integral_float(Value::Float(2.0), r#"{"float":"2"}"#)]
#[case::nan(Value::Float(f64::NAN), r#"{"float":"NaN"}"#)]
#[case::infinity(Value::Float(f64::NEG_INFINITY), r#"{"float":"-inf"}"#)]
#[case::decimal(decimal("100.0"), r#"{"decimal":"100"}"#)]
#[case::str(Value::Str("é\"".to_owned()), r#"{"str":"é\""}"#)]
#[case::tuple(
    Value::Tuple(vec![Value::Int(1.into()), Value::Str("a".to_owned())]),
    r#"{"tuple":[{"int":"1"},{"str":"a"}]}"#
)]
#[case::frozen_set(
    Value::FrozenSet(vec![Value::Int(3.into())]),
    r#"{"frozen_set":[{"int":"3"}]}"#
)]
fn a_value_serializes_in_its_kind_form(#[case] value: Value, #[case] text: &str) {
    assert_eq!(text_of(&value), text);
    let decoded: Value = serde_json::from_str(text).expect("decodes");
    assert_eq!(text_of(&decoded), text);
    let bytes = postcard::to_allocvec(&value).expect("encodes");
    let from_postcard: Value = postcard::from_bytes(&bytes).expect("decodes");
    assert_eq!(text_of(&from_postcard), text);
}

#[test]
fn a_member_writes_its_sets_in_canonical_order_and_reads_any_order() {
    let set = member(Value::FrozenSet(vec![
        Value::Str("b".to_owned()),
        Value::Int(2.into()),
        Value::Float(0.5),
    ]));

    assert_eq!(
        text_of(&set),
        r#"{"frozen_set":[{"float":"0.5"},{"int":"2"},{"str":"b"}]}"#
    );
    let shuffled: Member =
        serde_json::from_str(r#"{"frozen_set":[{"str":"b"},{"float":"0.5"},{"int":"2"}]}"#)
            .expect("decodes");
    assert_eq!(shuffled, set);
}

#[test]
fn a_member_refuses_a_value_that_is_no_member() {
    serde_json::from_str::<Member>(r#"{"decimal":"1.5"}"#).unwrap_err();
    serde_json::from_str::<Member>(r#"{"float":"NaN"}"#).unwrap_err();
    let data: ValueData = serde_json::from_str(r#"{"decimal":"1.5"}"#).expect("reads");
    assert!(matches!(
        data.build_member(&TestResolver),
        Err(BuildError::Invalid(_))
    ));
}

#[test]
fn a_value_refuses_non_canonical_numbers() {
    serde_json::from_str::<Value>(r#"{"int":"007"}"#).unwrap_err();
    serde_json::from_str::<Value>(r#"{"int":12}"#).unwrap_err();
    serde_json::from_str::<Value>(r#"{"float":1.5}"#).unwrap_err();
}

#[test]
fn an_opaque_value_serializes_as_its_foreign_part_and_resolves() {
    let value = WireToken::value(7);
    let text = text_of(&value);

    assert_eq!(text, r#"{"opaque":{"type_id":"test.token","data":"7"}}"#);
    serde_json::from_str::<Value>(&text).unwrap_err();
    let member = serde_json::from_str::<ValueData>(&text)
        .expect("reads")
        .build_member(&TestResolver)
        .expect("the resolver knows the part");
    assert_eq!(member, crate::support::constraint::member(value));
}

#[test]
fn an_opaque_value_without_a_wire_form_fails_to_serialize() {
    let message = serde_json::to_string(&TestOpaque::token(1).into_value())
        .expect_err("it has no wire form")
        .to_string();

    assert_eq!(message, "`Token` has no wire form");
}

fn build_set(polarity: Polarity, values: Vec<Value>) -> Constraint {
    let members = values.into_iter().map(member);
    Constraint::Set(SetConstraint::new(
        restored(61_500, "x"),
        MemberSet::new(members),
        polarity,
    ))
}

#[rstest]
#[case::in_set(
    build_set(Polarity::In, vec![Value::Str("a".to_owned()), Value::Bool(true), Value::Int(1.into())]),
    r#"{"in_set":{"variable":{"id":61500,"name_hint":"x"},"values":[{"bool":true},{"int":"1"},{"str":"a"}]}}"#
)]
#[case::not_in_set(
    build_set(Polarity::NotIn, vec![Value::Int(3.into())]),
    r#"{"not_in_set":{"variable":{"id":61500,"name_hint":"x"},"values":[{"int":"3"}]}}"#
)]
#[case::equation(
    Constraint::Equation(EquationConstraint::new(
        Expression::from(restored(61_501, "y")).greater(Expression::from(0))
    )),
    concat!(
        r#"{"equation":{"expression":{"nodes":[{"identifier":{"id":61501,"name_hint":"y"}},"#,
        r#"{"literal":{"int":"0"}},{"binary":{"operation":"greater","left":0,"right":1}}]}}}"#,
    )
)]
fn a_constraint_serializes_in_its_kind_form_and_round_trips(
    #[case] constraint: Constraint,
    #[case] text: &str,
) {
    assert_eq!(text_of(&constraint), text);
    let from_json: Constraint = serde_json::from_str(text).expect("decodes");
    assert!(from_json.is_structurally_equivalent(&constraint));
    let bytes = postcard::to_allocvec(&constraint).expect("encodes");
    let from_postcard: Constraint = postcard::from_bytes(&bytes).expect("decodes");
    assert!(from_postcard.is_structurally_equivalent(&constraint));
}

#[test]
fn a_set_constraint_reads_its_members_in_any_order() {
    let text = r#"{"in_set":{"variable":{"id":61500,"name_hint":"x"},"values":[{"str":"a"},{"int":"1"},{"int":"1"}]}}"#;

    let decoded: Constraint = serde_json::from_str(text).expect("decodes");

    assert_eq!(
        text_of(&decoded),
        r#"{"in_set":{"variable":{"id":61500,"name_hint":"x"},"values":[{"int":"1"},{"str":"a"}]}}"#
    );
}

#[test]
fn a_custom_constraint_serializes_as_its_foreign_part_and_resolves() {
    let custom = WireCustom::build("even");
    let text = text_of(&custom);

    assert_eq!(
        text,
        r#"{"custom":{"type_id":"test.custom","data":"even"}}"#
    );
    serde_json::from_str::<Constraint>(&text).unwrap_err();
    let data: ConstraintData = serde_json::from_str(&text).expect("reads");
    assert!(data.foreign().is_some());
    let rebuilt = data.build(&TestResolver).expect("the resolver knows it");
    assert!(rebuilt.is_structurally_equivalent(&custom));
}

#[test]
fn a_system_round_trips_in_canonical_order_with_its_foreign_parts() {
    let system = ConstraintSystem::new([
        build_set(
            Polarity::In,
            vec![WireToken::value(2), Value::Int(1.into())],
        ),
        WireCustom::build("even"),
        Constraint::Equation(EquationConstraint::new(Expression::literal(true))),
    ])
    .expect("every member has a key");
    let text = text_of(&system);

    let data: ConstraintSystemData = serde_json::from_str(&text).expect("reads");
    let rebuilt = data
        .build(&TestResolver)
        .expect("the resolver knows the parts");

    assert_eq!(text_of(&rebuilt), text);
    assert!(rebuilt.is_structurally_equivalent(&system));
    let bytes = postcard::to_allocvec(&system).expect("encodes");
    let from_postcard: ConstraintSystemData = postcard::from_bytes(&bytes).expect("reads");
    assert_eq!(
        text_of(&from_postcard.build(&TestResolver).expect("builds")),
        text
    );
}

#[test]
fn a_system_without_foreign_parts_round_trips_through_plain_serde() {
    let system = ConstraintSystem::new([
        build_set(Polarity::NotIn, vec![Value::Int(3.into())]),
        Constraint::Equation(EquationConstraint::new(Expression::literal(false))),
    ])
    .expect("every member has a key");

    let decoded: ConstraintSystem = serde_json::from_str(&text_of(&system)).expect("decodes");

    assert!(decoded.is_structurally_equivalent(&system));
    assert!(text_of(&system).starts_with(r#"{"constraints":["#));
}

/// Return a strategy of values up to three levels deep, with every number
/// kind, NaN and the infinities included.
fn value_strategy() -> impl proptest::strategy::Strategy<Value = Value> {
    use proptest::prelude::*;
    let leaf = prop_oneof![
        any::<bool>().prop_map(Value::Bool),
        any::<i128>().prop_map(|value| Value::Int(value.into())),
        any::<f64>().prop_map(Value::Float),
        "[a-zé\"\\\\ ]{0,6}".prop_map(Value::Str),
        (0_u32..1_000, 0_u32..4).prop_map(|(digits, places)| {
            let text = format!("{digits}.{}", "5".repeat(places as usize));
            decimal(&text)
        }),
    ];
    leaf.prop_recursive(3, 16, 4, |inner| {
        prop_oneof![
            proptest::collection::vec(inner.clone(), 0..4).prop_map(Value::Tuple),
            proptest::collection::vec(inner, 0..4).prop_map(Value::FrozenSet),
        ]
    })
}

proptest::proptest! {
    #[test]
    fn any_value_round_trips_to_the_same_text(value in value_strategy()) {
        let text = text_of(&value);
        let from_json: Value = serde_json::from_str(&text).expect("decodes");
        proptest::prop_assert_eq!(text_of(&from_json), text.clone());
        let bytes = postcard::to_allocvec(&value).expect("encodes");
        let from_postcard: Value = postcard::from_bytes(&bytes).expect("decodes");
        proptest::prop_assert_eq!(text_of(&from_postcard), text);
    }
}

/// Return the postcard bytes of `depth` nested one-element tuples around
/// `true`, written by hand so no deep value is built or dropped.
fn nested_tuple_bytes(depth: usize) -> Vec<u8> {
    // Postcard writes a variant as its index and a sequence as its length,
    // both varints: `Tuple` is variant 5 and `Bool` variant 0.
    let mut bytes = Vec::with_capacity(2 * depth + 2);
    for _ in 0..depth {
        bytes.extend([5, 1]);
    }
    bytes.extend([0, 1]);
    bytes
}

/// Return `depth` nested one-element tuples around `true`.
fn nested_tuple(depth: usize) -> Value {
    (0..depth).fold(Value::Bool(true), |value, _| Value::Tuple(vec![value]))
}

#[test]
fn a_postcard_value_nested_200000_deep_is_refused() {
    let bytes = nested_tuple_bytes(200_000);

    let error = postcard::from_bytes::<Value>(&bytes).expect_err("too deep");

    assert!(
        matches!(error, postcard::Error::SerdeDeCustom),
        "unexpected error {error:?}"
    );
    let error = postcard::from_bytes::<Member>(&bytes).expect_err("too deep");
    assert!(matches!(error, postcard::Error::SerdeDeCustom));
    let error = postcard::from_bytes::<ValueData>(&bytes).expect_err("too deep");
    assert!(matches!(error, postcard::Error::SerdeDeCustom));
}

#[test]
fn a_value_nested_128_deep_round_trips() {
    let value = nested_tuple(128);

    let bytes = postcard::to_allocvec(&value).expect("encodes");
    assert_eq!(bytes, nested_tuple_bytes(128));
    let decoded: Value = postcard::from_bytes(&bytes).expect("128 levels decode");

    assert_eq!(text_of(&decoded), text_of(&value));
    postcard::from_bytes::<Value>(&nested_tuple_bytes(129)).expect_err("129 levels are refused");
}

#[test]
fn a_value_nested_too_deep_is_refused_with_the_depth_message() {
    // A JSON text this deep exceeds serde_json's own recursion limit first,
    // so the tree is decoded from a `serde_json::Value`, which has none.
    let tree = (0..129).fold(
        serde_json::json!({"bool": true}),
        |tree, _| serde_json::json!({"tuple": [tree]}),
    );

    let error = serde_json::from_value::<Value>(tree.clone()).expect_err("too deep");
    assert_eq!(error.to_string(), "value nesting exceeds 128 levels");
    let error = serde_json::from_value::<ValueData>(tree).expect_err("too deep");
    assert_eq!(error.to_string(), "value nesting exceeds 128 levels");
}
