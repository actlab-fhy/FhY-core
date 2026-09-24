//! Tests that every public `Serialize` type round-trips through JSON text
//! and through postcard, a format that is not self-describing.
//!
//! Public API only. A decode that fails partway may leave the identifiers
//! and canonical values it already read behind; the stories at the end pin
//! that documented partial effect.

use std::fmt::Debug;

use fhy_core::diagnostic::{Note, NoteKind};
use fhy_core::identifier::Identifier;
use fhy_core::interned::{Canonical, Interned};
use fhy_core::op_attribute::OpAttribute;
use fhy_core::provenance::{
    CallSiteProvenance, FileProvenance, FusedProvenance, NamedProvenance, Position, Provenance,
    Span,
};
use fhy_core::value_domain::ValueDomain;
use rstest::rstest;
use serde::Serialize;
use serde::de::DeserializeOwned;

// =============================================================================
// Helpers
// =============================================================================

/// Assert `value` round-trips through JSON text and through postcard, each
/// decode equal to `value`.
fn assert_round_trips<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
    let json = serde_json::to_string(value).expect("the value encodes as JSON");
    let from_json: T = serde_json::from_str(&json).expect("the JSON text decodes");
    assert_eq!(&from_json, value, "the JSON round trip of {json}");

    let bytes = postcard::to_allocvec(value).expect("the value encodes as postcard");
    let from_postcard: T = postcard::from_bytes(&bytes).expect("the postcard bytes decode");
    assert_eq!(
        &from_postcard, value,
        "the postcard round trip of {value:?}"
    );
}

/// Build the position at `line` and `column`, which must both be non-zero.
fn build_position(line: u64, column: u64) -> Position {
    Position::try_new(line, column).expect("line and column are non-zero")
}

/// Build the file provenance for `path` over `span`.
fn build_file(path: &str, span: Option<Span>) -> Provenance {
    Provenance::File(FileProvenance::new(path, span))
}

/// Build the named provenance `name` over `child`.
fn build_named(name: &str, child: Provenance) -> Provenance {
    Provenance::Named(NamedProvenance::try_new(name, child).expect("name is non-empty"))
}

/// Build a provenance using every variant: a labelled fusion of a named
/// call-site chain, an unlabelled fusion and an unknown provenance.
fn build_nested_provenance() -> Provenance {
    let span = Span::from_offsets(0..3).expect("ordered offsets are valid");
    let call_site = Provenance::CallSite(CallSiteProvenance::new(
        build_named("inlined", build_file("callee.fhy", Some(span))),
        build_file("caller.fhy", None),
    ));
    Provenance::Fused(FusedProvenance::labelled(
        vec![
            build_named("chain", call_site),
            Provenance::Fused(FusedProvenance::new(vec![build_file("a.fhy", None)])),
            Provenance::Unknown,
        ],
        "cse",
    ))
}

// =============================================================================
// Diagnostic and provenance types
// =============================================================================

/// A function returning one of the shipped note kinds.
type ShippedKind = fn() -> &'static Canonical<NoteKind>;

/// Test a note of each shipped kind round-trips through both formats and
/// decodes to the shipped handle.
#[rstest]
#[case::rationale(NoteKind::rationale)]
#[case::suggestion(NoteKind::suggestion)]
#[case::remark(NoteKind::remark)]
#[case::other(NoteKind::other)]
fn note_round_trips_through_postcard(#[case] get_kind: ShippedKind) {
    let note = Note::new("a message", get_kind().clone());

    assert_round_trips(&note);

    let bytes = postcard::to_allocvec(&note).expect("the note encodes");
    let restored: Note = postcard::from_bytes(&bytes).expect("the note decodes");
    assert!(Canonical::ptr_eq(restored.kind(), get_kind()));
}

/// Test a note tagged with a caller's kind round-trips through both formats.
#[test]
fn note_with_a_custom_kind_round_trips_through_postcard() {
    let kind = NoteKind::register(Identifier::new("postcard-kind"), "A custom kind.");

    assert_round_trips(&Note::new("custom", kind));
}

/// Test a position round-trips through both formats, at the smallest and
/// largest lines and columns.
#[rstest]
#[case::smallest(1, 1)]
#[case::typical(7, 3)]
#[case::largest(u64::MAX, u64::MAX)]
fn position_round_trips_through_postcard(#[case] line: u64, #[case] column: u64) {
    assert_round_trips(&build_position(line, column));
}

/// Test spans with every mix of bounds round-trip through both formats.
#[rstest]
#[case::unknown(Span::unknown())]
#[case::offsets_only(Span::from_offsets(0..3).unwrap())]
#[case::positions_only(Span::from_positions(build_position(1, 1)..build_position(2, 4)).unwrap())]
#[case::start_offset_only(Span::unknown().with_start_offset(5).unwrap())]
#[case::end_offset_only(Span::unknown().with_end_offset(5).unwrap())]
#[case::start_position_only(Span::unknown().with_start_position(build_position(3, 1)).unwrap())]
#[case::end_position_only(Span::unknown().with_end_position(build_position(3, 1)).unwrap())]
#[case::both_pairs(
    Span::from_offsets(0..3)
        .and_then(|span| span.with_positions(build_position(1, 1)..build_position(1, 4)))
        .unwrap()
)]
fn span_round_trips_through_postcard(#[case] span: Span) {
    assert_round_trips(&span);
}

/// Test each provenance variant, and a tree nesting all of them,
/// round-trips through both formats.
#[rstest]
#[case::unknown(Provenance::Unknown)]
#[case::file_without_span(build_file("a.fhy", None))]
#[case::file_with_span(build_file("./src//a.fhy", Some(Span::from_offsets(2..9).unwrap())))]
#[case::named(build_named("fhy.add", Provenance::Unknown))]
#[case::call_site(Provenance::CallSite(CallSiteProvenance::new(
    build_file("callee.fhy", None),
    build_file("caller.fhy", None),
)))]
#[case::unlabelled_fusion(Provenance::Fused(FusedProvenance::new(vec![
    build_file("a.fhy", None),
    build_file("b.fhy", None),
])))]
#[case::labelled_fusion(Provenance::Fused(FusedProvenance::labelled(
    vec![build_file("a.fhy", None)],
    "loop-fusion",
)))]
#[case::empty_labelled_fusion(Provenance::Fused(FusedProvenance::labelled(vec![], "")))]
#[case::nested(build_nested_provenance())]
fn provenance_round_trips_through_postcard(#[case] provenance: Provenance) {
    assert_round_trips(&provenance);
}

// =============================================================================
// Identity and tag types
// =============================================================================

/// Test an identifier round-trips through both formats with its id.
#[test]
fn identifier_round_trips_through_postcard() {
    assert_round_trips(&Identifier::new("postcard-identifier"));
}

/// Test shipped and registered op attributes round-trip through both
/// formats.
#[test]
fn op_attribute_round_trips_through_postcard() {
    let registered = OpAttribute::register(Identifier::new("postcard-attribute"), "Custom.");

    assert_round_trips(OpAttribute::commutative());
    assert_round_trips(&registered);
}

/// Test shipped note kinds and a registered one round-trip through both
/// formats.
#[test]
fn note_kind_round_trips_through_postcard() {
    let registered = NoteKind::register(Identifier::new("postcard-note-kind"), "Custom.");

    assert_round_trips(NoteKind::remark());
    assert_round_trips(&registered);
}

/// Test a value domain with a parent chain round-trips through both
/// formats.
#[test]
fn value_domain_with_a_parent_chain_round_trips_through_postcard() {
    let tile = ValueDomain::register_child(
        Identifier::new("postcard-tile"),
        "A tile of data.",
        ValueDomain::data(),
    )
    .expect("the name is fresh");
    let subtile = ValueDomain::register_child(Identifier::new("postcard-subtile"), "Part.", &tile)
        .expect("the name is fresh");

    assert_round_trips(ValueDomain::address());
    assert_round_trips(&subtile);
}

// =============================================================================
// Adversarial input
// =============================================================================

/// Test every strict prefix of an encoded nested provenance and of an
/// encoded note decodes to an error rather than a panic.
#[test]
fn truncated_postcard_bytes_are_an_error_not_a_panic() {
    let provenance = postcard::to_allocvec(&build_nested_provenance()).expect("encodes");
    let note = postcard::to_allocvec(&Note::with_other_kind("truncated")).expect("encodes");

    for length in 0..provenance.len() {
        let result = postcard::from_bytes::<Provenance>(&provenance[..length]);
        assert!(result.is_err(), "a {length}-byte prefix decoded");
    }
    for length in 0..note.len() {
        let result = postcard::from_bytes::<Note>(&note[..length]);
        assert!(result.is_err(), "a {length}-byte prefix decoded");
    }
}

/// Test a note decode that fails after reading its kind leaves that kind
/// registered, and a retry with a valid message returns the same canonical
/// kind.
#[test]
fn a_note_decode_that_fails_after_its_kind_leaves_the_kind_registered() {
    let name = Identifier::new("partial-note-kind");
    let kind = format!(
        r#"{{"name": {{"id": {}, "name_hint": "partial-note-kind"}}, "description": "d"}}"#,
        name.id()
    );

    let failure = serde_json::from_str::<Note>(&format!(r#"{{"kind": {kind}, "message": 5}}"#));
    let registered = NoteKind::intern_registry().get(&name);
    let retry: Note = serde_json::from_str(&format!(r#"{{"kind": {kind}, "message": "m"}}"#))
        .expect("the retry decodes");

    assert!(failure.is_err(), "a numeric message is refused");
    let registered = registered.expect("the failed decode registered the kind");
    assert!(Canonical::ptr_eq(retry.kind(), &registered));
}
