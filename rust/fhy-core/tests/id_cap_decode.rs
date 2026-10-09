//! Decoding the largest advancing payload id leaves identifiers readable.
//!
//! This binary holds a single test because the test moves the process-global
//! id counter to `ADVANCE_CAP`, past every id the `it` binary's tests
//! decode, which would change what they observe. Do not add a second test.

use fhy_core::diagnostic::NoteKind;
use fhy_core::expression::builtins::BuiltinFunction;
use fhy_core::identifier::{ADVANCE_CAP, ID_CAP, Identifier};
use fhy_core::op_attribute::OpAttribute;
use fhy_core::value_domain::ValueDomain;
use serde_json::json;

fn decode(id: u64) -> Result<Identifier, serde_json::Error> {
    serde_json::from_value(json!({"id": id, "name_hint": "payload"}))
}

/// Test a payload can raise the counter at most to `ADVANCE_CAP`, after
/// which a fresh identifier still round-trips through JSON and postcard,
/// the shipped tags keep their reserved ids and the built-in parameters are
/// creatable, while an id at or above `ADVANCE_CAP` that this process did
/// not issue is refused.
#[test]
fn deserializing_the_largest_advancing_payload_id_leaves_identifiers_readable() {
    let largest = decode(ADVANCE_CAP - 1).expect("the largest advancing payload id decodes");
    let foreign = decode(ADVANCE_CAP);

    let fresh = Identifier::new("fresh");
    let from_json: Identifier =
        serde_json::from_str(&serde_json::to_string(&fresh).expect("the identifier encodes"))
            .expect("an identifier issued here decodes");
    let bytes = postcard::to_allocvec(&fresh).expect("the identifier encodes");
    let from_postcard: Identifier =
        postcard::from_bytes(&bytes).expect("an identifier issued here decodes");
    let beyond = decode(ID_CAP - 1);

    assert_eq!(largest.id(), ADVANCE_CAP - 1);
    assert!(foreign.is_err(), "a foreign id decoded: {foreign:?}");
    assert_eq!(fresh.id(), ADVANCE_CAP);
    assert_eq!(
        (from_json.id(), from_json.name_hint()),
        (ADVANCE_CAP, "fresh")
    );
    assert_eq!(from_postcard, fresh);
    assert!(beyond.is_err(), "an id not issued here decoded: {beyond:?}");
    assert_eq!(NoteKind::other().name().id(), 3);
    assert_eq!(OpAttribute::commutative().name().id(), 16);
    assert_eq!(ValueDomain::data().name().id(), 32);
    let gelu = BuiltinFunction::Gelu
        .composed()
        .expect("gelu is a composed built-in");
    assert!(
        gelu.parameters()
            .iter()
            .all(|parameter| parameter.id() > ADVANCE_CAP)
    );
}
