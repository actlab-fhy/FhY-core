//! Decoding the largest payload id leaves construction working.
//!
//! This binary holds a single test because the test moves the process-global
//! id counter to `ID_CAP`, after which no fresh identifier can round-trip
//! through serde: a fresh id at or above the cap is rejected as a payload
//! id. Sharing a process with the `it` binary's round-trip tests would fail
//! them. Do not add a second test.

use fhy_core::diagnostic::NoteKind;
use fhy_core::expr::builtins::BuiltinFunction;
use fhy_core::identifier::{ID_CAP, Identifier};
use fhy_core::op_attribute::OpAttribute;
use fhy_core::value_domain::ValueDomain;
use serde_json::json;

/// Test a payload can raise the counter at most to `ID_CAP`, which leaves
/// fresh ids to construct, the shipped tags on their reserved ids, and the
/// built-in parameters creatable, while an id at the cap is rejected.
#[test]
fn deserializing_the_largest_payload_id_leaves_construction_working() {
    let largest: Identifier =
        serde_json::from_value(json!({"id": ID_CAP - 1, "name_hint": "largest"}))
            .expect("the largest payload id decodes");
    let rejected = serde_json::from_value::<Identifier>(json!({"id": ID_CAP, "name_hint": "cap"}));

    let first = Identifier::new("first");
    let second = Identifier::new("second");

    assert_eq!(largest.id(), ID_CAP - 1);
    assert!(rejected.is_err(), "the cap decoded: {rejected:?}");
    assert_eq!((first.id(), second.id()), (ID_CAP, ID_CAP + 1));
    assert_eq!(NoteKind::other().name().id(), 3);
    assert_eq!(OpAttribute::commutative().name().id(), 16);
    assert_eq!(ValueDomain::data().name().id(), 32);
    let gelu = BuiltinFunction::Gelu
        .composed()
        .expect("gelu is a composed built-in");
    assert!(
        gelu.parameters()
            .iter()
            .all(|parameter| parameter.id() > ID_CAP)
    );

    let bytes = postcard::to_allocvec(&largest).expect("the identifier encodes");
    let decoded: Identifier = postcard::from_bytes(&bytes).expect("the identifier decodes");
    assert_eq!((decoded.id(), decoded.name_hint()), (ID_CAP - 1, "largest"));
}
