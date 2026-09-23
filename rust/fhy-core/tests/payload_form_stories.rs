//! Tests that every decoded payload accepts only its map form.
//!
//! Public API only. A payload whose fields arrive as a sequence (the same
//! values, in field order, without their keys) is refused for every type
//! that decodes through the crate's payload machinery, while the map form
//! of the same values decodes. Each case pairs the two forms, so a refusal
//! can only come from the sequence shape. For an expression or a provenance,
//! wrapped as `{"__type__": .., "__data__": ..}`, both the wrapper and its
//! `__data__` are refused as sequences.

use std::fmt::Debug;

use fhy_core::diagnostic::{Note, NoteKind};
use fhy_core::identifier::Identifier;
use fhy_core::interned::Canonical;
use fhy_core::op_attribute::OpAttribute;
use fhy_core::provenance::{Position, Provenance, Span};
use fhy_core::symbolic::expression::Expression;
use fhy_core::value_domain::ValueDomain;
use rstest::rstest;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

/// The types whose payload form is under test.
#[derive(Debug, Clone, Copy)]
enum PayloadType {
    Identifier,
    OpAttribute,
    CanonicalOpAttribute,
    ValueDomain,
    CanonicalValueDomain,
    NoteKind,
    CanonicalNoteKind,
    Note,
    Expression,
    ExpressionData,
    Position,
    Span,
    Provenance,
    ProvenanceData,
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
            Self::Identifier => {
                let identifier = Identifier::new("sequence-identifier");
                (
                    json!({"id": identifier.id(), "name_hint": "sequence-identifier"}),
                    json!([identifier.id(), "sequence-identifier"]),
                )
            }
            Self::OpAttribute | Self::CanonicalOpAttribute => {
                let name = build_identifier_payload("sequence-attribute");
                (
                    json!({"name": name, "description": "d"}),
                    json!([name, "d"]),
                )
            }
            Self::ValueDomain | Self::CanonicalValueDomain => {
                let name = build_identifier_payload("sequence-domain");
                (
                    json!({"name": name, "description": "d", "parent": null}),
                    json!([name, "d", null]),
                )
            }
            Self::NoteKind | Self::CanonicalNoteKind => {
                let name = build_identifier_payload("sequence-note-kind");
                (
                    json!({"name": name, "description": "d"}),
                    json!([name, "d"]),
                )
            }
            Self::Note => {
                let kind = json!({
                    "name": build_identifier_payload("sequence-note-kind"),
                    "description": "d",
                });
                (json!({"message": "m", "kind": kind}), json!(["m", kind]))
            }
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
            Self::Position => (json!({"line": 1, "column": 2}), json!([1, 2])),
            Self::Span => (
                json!({
                    "start_offset": 0,
                    "end_offset": 1,
                    "start_position": null,
                    "end_position": null,
                }),
                json!([0, 1, null, null]),
            ),
            Self::Provenance => (
                json!({"__type__": "provenance.unknown", "__data__": {}}),
                json!(["provenance.unknown", {}]),
            ),
            Self::ProvenanceData => (
                json!({
                    "__type__": "provenance.file",
                    "__data__": {"file_path": "a.fhy", "span": null},
                }),
                json!({"__type__": "provenance.file", "__data__": ["a.fhy", null]}),
            ),
        }
    }

    /// Return the text the refusal of this type's sequence form contains.
    fn describe_sequence_refusal(self) -> &'static str {
        match self {
            Self::ExpressionData => {
                "expected the fields of an identifier reference as a map, got a list"
            }
            _ => "invalid type: sequence",
        }
    }

    /// Decode `payload` from its JSON text as this type, returning the
    /// error text on refusal.
    fn decode(self, payload: &Value) -> Result<(), String> {
        let text = payload.to_string();
        match self {
            Self::Identifier => decode_text::<Identifier>(&text),
            Self::OpAttribute => decode_text::<OpAttribute>(&text),
            Self::CanonicalOpAttribute => decode_text::<Canonical<OpAttribute>>(&text),
            Self::ValueDomain => decode_text::<ValueDomain>(&text),
            Self::CanonicalValueDomain => decode_text::<Canonical<ValueDomain>>(&text),
            Self::NoteKind => decode_text::<NoteKind>(&text),
            Self::CanonicalNoteKind => decode_text::<Canonical<NoteKind>>(&text),
            Self::Note => decode_text::<Note>(&text),
            Self::Expression | Self::ExpressionData => decode_text::<Expression>(&text),
            Self::Position => decode_text::<Position>(&text),
            Self::Span => decode_text::<Span>(&text),
            Self::Provenance | Self::ProvenanceData => decode_text::<Provenance>(&text),
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
#[case::identifier(PayloadType::Identifier)]
#[case::op_attribute(PayloadType::OpAttribute)]
#[case::canonical_op_attribute(PayloadType::CanonicalOpAttribute)]
#[case::value_domain(PayloadType::ValueDomain)]
#[case::canonical_value_domain(PayloadType::CanonicalValueDomain)]
#[case::note_kind(PayloadType::NoteKind)]
#[case::canonical_note_kind(PayloadType::CanonicalNoteKind)]
#[case::note(PayloadType::Note)]
#[case::expression(PayloadType::Expression)]
#[case::expression_data(PayloadType::ExpressionData)]
#[case::position(PayloadType::Position)]
#[case::span(PayloadType::Span)]
#[case::provenance(PayloadType::Provenance)]
#[case::provenance_data(PayloadType::ProvenanceData)]
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

/// Test a refused sequence payload leaves no trace: the identifier it names
/// is not restored, so a later map payload naming the same id registers a
/// fresh canonical attribute rather than finding one.
#[test]
fn a_refused_sequence_payload_registers_nothing() {
    let name = build_identifier_payload("sequence-never-registered");
    let sequence = json!([name, "from the sequence"]);
    let map = json!({"name": name, "description": "from the map"});

    let sequence_result = decode_text::<Canonical<OpAttribute>>(&sequence.to_string());
    let restored: Canonical<OpAttribute> =
        serde_json::from_value(map).expect("the map form decodes");

    assert!(sequence_result.is_err(), "the sequence form is refused");
    assert_eq!(restored.description(), "from the map");
}
