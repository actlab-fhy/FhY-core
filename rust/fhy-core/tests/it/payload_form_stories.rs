//! Tests that an expression payload decodes from its map form only.
//!
//! Public API only. An expression payload whose fields arrive as a sequence
//! (the same values, in field order, without their keys) is refused, while
//! the map form of the same values decodes. Each case pairs the two forms,
//! so a refusal can only come from the sequence shape. Both the
//! `{"__type__": .., "__data__": ..}` wrapper and its `__data__` are refused
//! as sequences. Every other type decodes through plain serde derives,
//! which accept the sequence form.

use std::fmt::Debug;

use fhy_core::expr::Expression;
use fhy_core::identifier::Identifier;
use rstest::rstest;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

/// The types whose payload form is under test.
#[derive(Debug, Clone, Copy)]
enum PayloadType {
    Expression,
    ExpressionData,
}

/// Return the wire form of a fresh identifier named `name_hint`.
fn build_identifier_payload(name_hint: &str) -> Value {
    let identifier = Identifier::new(name_hint);
    json!({"id": identifier.id(), "name_hint": identifier.name_hint()})
}

impl PayloadType {
    /// Return a well-formed map payload for this type and the same field
    /// values as a sequence, in field order.
    ///
    /// Every call names a fresh identifier, so no case depends on what
    /// another case registered.
    fn build_map_and_sequence(self) -> (Value, Value) {
        match self {
            Self::Expression => {
                let data = json!({"identifier": build_identifier_payload("sequence-expression")});
                (
                    json!({"__type__": "identifier_expression", "__data__": data}),
                    json!(["identifier_expression", data]),
                )
            }
            Self::ExpressionData => {
                let identifier = build_identifier_payload("sequence-expression");
                (
                    json!({
                        "__type__": "identifier_expression",
                        "__data__": {"identifier": identifier},
                    }),
                    json!({"__type__": "identifier_expression", "__data__": [identifier]}),
                )
            }
        }
    }

    /// Return the text the refusal of this type's sequence form contains.
    fn describe_sequence_refusal(self) -> &'static str {
        match self {
            Self::ExpressionData => {
                "expected the fields of an identifier reference as a map, got a list"
            }
            Self::Expression => "invalid type: sequence",
        }
    }

    /// Decode `payload` from its JSON text as this type, returning the
    /// error text on refusal.
    fn decode(self, payload: &Value) -> Result<(), String> {
        let text = payload.to_string();
        match self {
            Self::Expression | Self::ExpressionData => decode_text::<Expression>(&text),
        }
    }
}

/// Decode `text` as `T`, returning the error text on refusal.
fn decode_text<T: DeserializeOwned + Debug>(text: &str) -> Result<(), String> {
    serde_json::from_str::<T>(text)
        .map(drop)
        .map_err(|error| error.to_string())
}

/// Test a payload given as a sequence of its field values is refused, while
/// the map form of the same values decodes.
#[rstest]
#[case::expression(PayloadType::Expression)]
#[case::expression_data(PayloadType::ExpressionData)]
fn a_payload_given_as_a_sequence_is_refused(#[case] payload_type: PayloadType) {
    let (map, sequence) = payload_type.build_map_and_sequence();

    let map_result = payload_type.decode(&map);
    let sequence_result = payload_type.decode(&sequence);

    assert_eq!(map_result, Ok(()), "{payload_type:?} map form {map}");
    let error = sequence_result.expect_err(&format!(
        "{payload_type:?} sequence form {sequence} is refused"
    ));
    assert!(
        error.contains(payload_type.describe_sequence_refusal()),
        "{payload_type:?} refuses {sequence} as a sequence, not for another reason: {error}"
    );
}
