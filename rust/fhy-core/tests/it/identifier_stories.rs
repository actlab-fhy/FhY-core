//! User-story tests for `fhy_core::identifier`: fresh ids, and the fixed ids
//! of the identifiers this crate ships.
//!
//! These tests share the process-global id counter with every other test in
//! the binary, so they only compare ids they allocated themselves or the fixed
//! reserved ids, and decode only ids already issued.

use std::collections::HashSet;

use fhy_core::diagnostic::NoteKind;
use fhy_core::expr::builtins::{BuiltinFunction, ComposedFunction};
use fhy_core::identifier::{Identifier, RESERVED_ID_COUNT};
use fhy_core::interned::Canonical;
use fhy_core::op_attribute::OpAttribute;
use fhy_core::value_domain::ValueDomain;
use serde_json::json;

#[test]
fn the_reserved_block_holds_65536_ids() {
    assert_eq!(RESERVED_ID_COUNT, 65_536);
}

#[test]
fn identifiers_created_with_one_name_hint_are_distinct() {
    let first = Identifier::new("accumulator");
    let second = Identifier::new("accumulator");

    assert_ne!(first, second);
    assert_ne!(first.id(), second.id());
}

/// Test the built-in composed functions' parameters are pairwise distinct,
/// and that none equals an identifier created afterward with the same name
/// hint as a parameter.
#[test]
fn composed_function_parameters_are_pairwise_distinct_and_unaliased() {
    let parameters: Vec<Identifier> = BuiltinFunction::iter()
        .filter_map(BuiltinFunction::composed)
        .flat_map(ComposedFunction::parameters)
        .cloned()
        .collect();

    let lookalikes: Vec<Identifier> = ["a", "b", "x", "lo", "hi", "bound", "slope"]
        .into_iter()
        .map(Identifier::new)
        .collect();

    let distinct_ids: HashSet<u64> = parameters.iter().map(Identifier::id).collect();
    assert_eq!(
        distinct_ids.len(),
        parameters.len(),
        "a parameter id repeats"
    );
    for lookalike in &lookalikes {
        assert!(
            !parameters.contains(lookalike),
            "{lookalike:?} aliases a built-in parameter"
        );
    }
}

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
        (NoteKind::rationale().name(), 0, "rationale"),
        (NoteKind::suggestion().name(), 1, "suggestion"),
        (NoteKind::remark().name(), 2, "remark"),
        (NoteKind::other().name(), 3, "other"),
        (OpAttribute::commutative().name(), 16, "commutative"),
        (OpAttribute::associative().name(), 17, "associative"),
        (OpAttribute::pure().name(), 18, "pure"),
        (OpAttribute::elementwise().name(), 19, "elementwise"),
        (ValueDomain::data().name(), 32, "data"),
        (ValueDomain::address().name(), 33, "address"),
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

    assert!(Canonical::ptr_eq(&decoded, OpAttribute::commutative()));
    assert_eq!(decoded.name().name_hint(), "commutative");
    assert_eq!(
        decoded.description(),
        OpAttribute::commutative().description()
    );
}
