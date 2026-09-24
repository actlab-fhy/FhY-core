//! User-story tests for `fhy_core::identifier`: fresh ids, and the fixed ids
//! of the identifiers this crate ships.
//!
//! Public API only. These tests share the process-global id counter with
//! every other test in the binary, so they only compare ids they allocated
//! themselves or the fixed reserved ids, and decode only ids already issued.

use fhy_core::diagnostic::{
    get_other_note_kind, get_rationale_note_kind, get_remark_note_kind, get_suggestion_note_kind,
};
use fhy_core::identifier::{Identifier, RESERVED_ID_COUNT};
use fhy_core::interned::Canonical;
use fhy_core::op_attribute::{
    OpAttribute, get_associative, get_commutative, get_elementwise, get_pure,
};
use fhy_core::value_domain::{get_address_domain, get_data_domain};
use serde_json::json;

/// Test the reserved block holds the ids `0..65_536`.
#[test]
fn the_reserved_block_holds_65536_ids() {
    assert_eq!(RESERVED_ID_COUNT, 65_536);
}

/// Test fresh identifiers never take an id from the reserved block.
#[test]
fn fresh_identifiers_are_never_reserved() {
    let ids: Vec<u64> = (0..1_000).map(|_| Identifier::new("fresh").id()).collect();

    assert!(
        ids.iter().all(|&id| id >= RESERVED_ID_COUNT),
        "a fresh id is reserved: {ids:?}"
    );
}

/// Test every identifier this crate ships holds its fixed reserved id and
/// name hint, whatever order the shipped values are first used in.
#[test]
fn shipped_identifiers_hold_their_reserved_ids() {
    let shipped = [
        (get_rationale_note_kind().name(), 0, "rationale"),
        (get_suggestion_note_kind().name(), 1, "suggestion"),
        (get_remark_note_kind().name(), 2, "remark"),
        (get_other_note_kind().name(), 3, "other"),
        (get_commutative().name(), 16, "commutative"),
        (get_associative().name(), 17, "associative"),
        (get_pure().name(), 18, "pure"),
        (get_elementwise().name(), 19, "elementwise"),
        (get_data_domain().name(), 32, "data"),
        (get_address_domain().name(), 33, "address"),
    ];

    for (name, id, name_hint) in shipped {
        assert_eq!((name.id(), name.name_hint()), (id, name_hint), "{name:?}");
    }
}

/// Test a payload naming a shipped tag by its reserved id decodes to the
/// shipped tag, keeping its canonical description and name hint.
#[test]
fn a_shipped_tag_decoded_by_its_reserved_id_is_the_shipped_tag() {
    let payload = json!({
        "name": {"id": 16, "name_hint": "whatever"},
        "description": "other",
    });

    let decoded: Canonical<OpAttribute> =
        serde_json::from_value(payload).expect("the payload decodes");

    assert_eq!(&decoded, get_commutative());
    assert_eq!(decoded.name().name_hint(), "commutative");
    assert_eq!(decoded.description(), get_commutative().description());
}
