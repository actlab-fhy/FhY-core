//! Stories of custom provenances: the part's display, equality and hash
//! through its hooks, its wire form and the refusal of plain decoding,
//! `ProvenanceData` with a resolver, and `Provenance::fuse` over a custom
//! source.

use fhy_core::foreign::{BuildError, ForeignError, NoForeign, Part};
use fhy_core::provenance::wire::ProvenanceData;
use fhy_core::provenance::{CallSiteProvenance, FusedProvenance, NamedProvenanceError, Provenance};
use rstest::rstest;

use crate::support::hashing::hash_of;
use crate::support::provenance::{
    EDGE_PROPAGATION, EdgePropagation, EdgeResolver, UnwiredProvenance, build_file, build_named,
};

/// The JSON text of the custom provenance of the edge `e3`.
const EDGE_E3_JSON: &str = r#"{"custom":{"type_id":"test.edge_propagation","data":"e3"}}"#;

/// Return the call site of `callee` at `site`.
fn build_call_site(callee: Provenance, site: Provenance) -> Provenance {
    Provenance::CallSite(CallSiteProvenance::new(callee, site))
}

/// Return the fusion of `sources`, unlabelled, kept as given.
fn build_fused(sources: Vec<Provenance>) -> Provenance {
    Provenance::Fused(FusedProvenance::new(sources))
}

/// Where a story places a custom provenance: as it is, or inside a named,
/// a call-site or a fused provenance.
#[derive(Debug, Clone, Copy)]
enum Nesting {
    /// The custom provenance itself.
    Bare,
    /// The child of a named provenance.
    Named,
    /// The callee of a call site.
    Callee,
    /// The caller of a call site.
    Caller,
    /// A source of an unlabelled fusion.
    Fused,
    /// A source of a labelled fusion.
    LabelledFused,
    /// The child of a named provenance in a fusion in a named provenance.
    NamedFusedNamed,
}

/// Return `custom` placed as `nesting` says.
fn nest(custom: &Provenance, nesting: Nesting) -> Provenance {
    match nesting {
        Nesting::Bare => custom.clone(),
        Nesting::Named => build_named("n", custom.clone()),
        Nesting::Callee => build_call_site(custom.clone(), build_file("caller.fhy", None)),
        Nesting::Caller => build_call_site(build_file("callee.fhy", None), custom.clone()),
        Nesting::Fused => build_fused(vec![build_file("a.fhy", None), custom.clone()]),
        Nesting::LabelledFused => Provenance::Fused(FusedProvenance::labelled(
            vec![custom.clone(), build_file("b.fhy", None)],
            "label",
        )),
        Nesting::NamedFusedNamed => build_named(
            "outer",
            build_fused(vec![build_named("inner", custom.clone())]),
        ),
    }
}

/// Return the JSON text of `provenance`.
fn write_json(provenance: &Provenance) -> String {
    serde_json::to_string(provenance).expect("a provenance with wire forms encodes")
}

// ---------------------------------------------------------------------------
// Display, equality and hash through the hooks
// ---------------------------------------------------------------------------

/// Test a custom provenance displays as its part's own display.
#[test]
fn custom_provenance_displays_as_its_part() {
    let custom = EdgePropagation::build("e3");

    assert_eq!(custom.to_string(), "edge<e3>");
}

/// Test a custom provenance displays inside the variants that hold it.
#[test]
fn custom_provenance_displays_inside_other_variants() {
    let custom = EdgePropagation::build("e3");

    assert_eq!(build_named("n", custom.clone()).to_string(), "n (edge<e3>)");
    assert_eq!(
        build_call_site(custom.clone(), build_file("f.fhy", None)).to_string(),
        "edge<e3> at f.fhy"
    );
    assert_eq!(
        build_fused(vec![build_file("a.fhy", None), custom]).to_string(),
        "fused[a.fhy, edge<e3>]"
    );
}

/// Test two separately built parts of equal data are equal provenances
/// with equal hashes, through the part's hooks.
#[test]
fn custom_provenance_equality_and_hash_go_through_the_hooks() {
    let first = EdgePropagation::build("e3");
    let second = EdgePropagation::build("e3");

    assert_eq!(first, second);
    assert_eq!(hash_of(&first), hash_of(&second));
}

/// Test parts of different data are different provenances, and a custom
/// provenance is not any other variant.
#[test]
fn custom_provenance_of_different_data_is_unequal() {
    let first = EdgePropagation::build("e3");

    assert_ne!(first, EdgePropagation::build("e4"));
    assert_ne!(first, Provenance::Unknown);
    assert_ne!(first, build_file("e3", None));
    assert_ne!(hash_of(&first), hash_of(&EdgePropagation::build("e4")));
}

/// Test a provenance holding a custom part is equal to one holding an equal
/// part, with an equal hash, and unequal to one holding another part, at
/// any depth.
#[rstest]
fn custom_provenance_equality_holds_at_depth(
    #[values(
        Nesting::Named,
        Nesting::Callee,
        Nesting::Caller,
        Nesting::Fused,
        Nesting::LabelledFused,
        Nesting::NamedFusedNamed
    )]
    nesting: Nesting,
) {
    let left = nest(&EdgePropagation::build("e3"), nesting);
    let right = nest(&EdgePropagation::build("e3"), nesting);
    let other = nest(&EdgePropagation::build("e4"), nesting);

    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
    assert_ne!(left, other);
}

/// Test a part that keeps the default `eq_part` is equal to itself and its
/// clones only, never to a separate part of equal data.
#[test]
fn custom_provenance_default_equality_is_identity() {
    let part = Part::new(UnwiredProvenance("p".to_owned()));
    let provenance = Provenance::Custom(part.clone());

    let separate = UnwiredProvenance::build("p");

    assert_eq!(provenance, Provenance::Custom(part));
    assert_eq!(provenance, provenance.clone());
    assert_ne!(provenance, separate);
    assert_ne!(separate, UnwiredProvenance::build("p"));
}

/// Test the default `hash_part` feeds nothing, so a clone hashes as its
/// original.
#[test]
fn custom_provenance_clone_hashes_as_its_original() {
    let provenance = UnwiredProvenance::build("p");

    assert_eq!(hash_of(&provenance), hash_of(&provenance.clone()));
}

// ---------------------------------------------------------------------------
// Serialization and plain decoding
// ---------------------------------------------------------------------------

/// Test a custom provenance serializes as the tag `custom` over its foreign
/// part.
#[test]
fn custom_provenance_serializes_as_a_foreign_part() {
    let custom = EdgePropagation::build("e3");

    let text = write_json(&custom);

    assert_eq!(text, EDGE_E3_JSON);
}

/// Test a custom provenance nested in a named one serializes in place.
#[test]
fn custom_provenance_serializes_inside_a_named_provenance() {
    let named = build_named("n", EdgePropagation::build("e3"));

    let text = write_json(&named);

    assert_eq!(
        text,
        format!(r#"{{"named":{{"name":"n","child":{EDGE_E3_JSON}}}}}"#)
    );
}

/// Test a part with no wire form fails serialization, naming its type, in
/// any variant that holds it.
#[rstest]
fn custom_provenance_with_no_wire_form_fails_to_serialize(
    #[values(
        Nesting::Bare,
        Nesting::Named,
        Nesting::Callee,
        Nesting::Caller,
        Nesting::Fused,
        Nesting::LabelledFused,
        Nesting::NamedFusedNamed
    )]
    nesting: Nesting,
) {
    let provenance = nest(&UnwiredProvenance::build("p"), nesting);

    let error = serde_json::to_string(&provenance).expect_err("no wire form");

    assert!(
        error
            .to_string()
            .contains("`UnwiredProvenance` has no wire form"),
        "{error}"
    );
}

/// Test postcard refuses a part with no wire form too.
#[test]
fn custom_provenance_with_no_wire_form_fails_to_serialize_as_postcard() {
    let unwired = UnwiredProvenance::build("p");

    let error = postcard::to_allocvec(&unwired).expect_err("no wire form");

    assert!(
        matches!(error, postcard::Error::SerdeSerCustom),
        "{error:?}"
    );
}

/// Test `Provenance`'s own decoding refuses a custom provenance, naming its
/// type id, wherever it occurs.
#[rstest]
fn provenance_deserialize_refuses_a_custom_provenance_at_any_depth(
    #[values(
        Nesting::Bare,
        Nesting::Named,
        Nesting::Callee,
        Nesting::Caller,
        Nesting::Fused,
        Nesting::LabelledFused,
        Nesting::NamedFusedNamed
    )]
    nesting: Nesting,
) {
    let text = write_json(&nest(&EdgePropagation::build("e3"), nesting));

    let error = serde_json::from_str::<Provenance>(&text).expect_err("no resolver");

    assert!(
        error
            .to_string()
            .contains("no implementation for the foreign part `test.edge_propagation`"),
        "{text}: {error}"
    );
}

// ---------------------------------------------------------------------------
// ProvenanceData
// ---------------------------------------------------------------------------

/// Test the wire form of `ProvenanceData::of` is the provenance's own JSON.
#[rstest]
fn provenance_data_of_serializes_as_the_provenance_does(
    #[values(
        Nesting::Bare,
        Nesting::Named,
        Nesting::Callee,
        Nesting::Caller,
        Nesting::Fused,
        Nesting::LabelledFused,
        Nesting::NamedFusedNamed
    )]
    nesting: Nesting,
) {
    let provenance = nest(&EdgePropagation::build("e3"), nesting);
    let data = ProvenanceData::of(&provenance).expect("a wire form");

    let text = serde_json::to_string(&data).expect("encodes");

    assert_eq!(text, write_json(&provenance));
}

/// Test the data of a custom provenance, written as JSON and read back,
/// builds an equal provenance with a resolver.
#[test]
fn provenance_data_round_trips_a_custom_provenance_through_json() {
    let custom = EdgePropagation::build("e3");

    let data = ProvenanceData::of(&custom).expect("a wire form");
    let text = serde_json::to_string(&data).expect("encodes");
    let decoded: ProvenanceData = serde_json::from_str(&text).expect("decodes");
    let built = decoded.build(&EdgeResolver).expect("the resolver reads it");

    assert_eq!(text, EDGE_E3_JSON);
    assert_eq!(built, custom);
}

/// Test the round trip holds inside a named, a call-site and a fused
/// provenance, and nested deeper.
#[rstest]
fn provenance_data_round_trips_nested_custom_provenances_through_json(
    #[values(
        Nesting::Named,
        Nesting::Callee,
        Nesting::Caller,
        Nesting::Fused,
        Nesting::LabelledFused,
        Nesting::NamedFusedNamed
    )]
    nesting: Nesting,
) {
    let provenance = nest(&EdgePropagation::build("e3"), nesting);
    let data = ProvenanceData::of(&provenance).expect("a wire form");
    let text = serde_json::to_string(&data).expect("encodes");

    let decoded: ProvenanceData = serde_json::from_str(&text).expect("decodes");
    let built = decoded.build(&EdgeResolver).expect("the resolver reads it");

    assert_eq!(built, provenance, "{text}");
}

/// Test the round trip through postcard builds an equal provenance, nested
/// or not.
#[rstest]
fn provenance_data_round_trips_custom_provenances_through_postcard(
    #[values(
        Nesting::Bare,
        Nesting::Named,
        Nesting::Callee,
        Nesting::Caller,
        Nesting::Fused,
        Nesting::LabelledFused,
        Nesting::NamedFusedNamed
    )]
    nesting: Nesting,
) {
    let provenance = nest(&EdgePropagation::build("e3"), nesting);
    let data = ProvenanceData::of(&provenance).expect("a wire form");
    let bytes = postcard::to_allocvec(&data).expect("encodes");

    let decoded: ProvenanceData = postcard::from_bytes(&bytes).expect("decodes");
    let built = decoded.build(&EdgeResolver).expect("the resolver reads it");

    assert_eq!(built, provenance);
}

/// Test the data reads the bytes `Provenance` itself writes in postcard.
#[test]
fn provenance_data_reads_the_postcard_form_of_a_provenance() {
    let provenance = build_named("n", EdgePropagation::build("e3"));
    let bytes = postcard::to_allocvec(&provenance).expect("encodes");

    let decoded: ProvenanceData = postcard::from_bytes(&bytes).expect("decodes");
    let built = decoded.build(&EdgeResolver).expect("the resolver reads it");

    assert_eq!(built, provenance);
}

/// Test a provenance holding no custom part builds with `NoForeign`.
#[test]
fn provenance_data_of_a_provenance_without_custom_parts_builds_without_a_resolver() {
    let provenance = build_call_site(
        build_named("n", build_file("a.fhy", None)),
        build_fused(vec![build_file("b.fhy", None), Provenance::Unknown]),
    );

    let data = ProvenanceData::of(&provenance).expect("a wire form");
    let built = data.build(&NoForeign).expect("no part to resolve");

    assert_eq!(built, provenance);
}

/// Test a resolver that does not know the part's type id refuses it as
/// unresolved, at any depth.
#[rstest]
fn provenance_data_build_reports_an_unresolved_custom_provenance(
    #[values(
        Nesting::Bare,
        Nesting::Named,
        Nesting::Callee,
        Nesting::Caller,
        Nesting::Fused,
        Nesting::LabelledFused,
        Nesting::NamedFusedNamed
    )]
    nesting: Nesting,
) {
    let data =
        ProvenanceData::of(&nest(&EdgePropagation::build("e3"), nesting)).expect("a wire form");

    let result = data.build(&NoForeign);

    assert!(
        matches!(
            &result,
            Err(BuildError::Foreign(ForeignError::Unresolved { type_id }))
                if type_id == EDGE_PROPAGATION
        ),
        "{result:?}"
    );
}

/// Test a named provenance with an empty name decodes as data and fails to
/// build as invalid.
#[rstest]
#[case::bare(r#"{"named":{"name":"","child":{"unknown":{}}}}"#)]
#[case::in_a_call_site(
    r#"{"call_site":{"callee":{"unknown":{}},"caller":{"named":{"name":"","child":{"unknown":{}}}}}}"#
)]
#[case::in_a_fusion(
    r#"{"fused":{"sources":[{"unknown":{}},{"named":{"name":"","child":{"unknown":{}}}}],"label":null}}"#
)]
#[case::above_a_custom_part(
    r#"{"named":{"name":"","child":{"custom":{"type_id":"test.edge_propagation","data":"e3"}}}}"#
)]
fn provenance_data_build_reports_an_empty_name_as_invalid(#[case] text: &str) {
    let data: ProvenanceData = serde_json::from_str(text).expect("the shape is valid");

    let result = data.build(&EdgeResolver);

    let Err(BuildError::Invalid(source)) = &result else {
        panic!("expected Invalid, got {result:?}");
    };
    assert_eq!(
        source.downcast_ref::<NamedProvenanceError>(),
        Some(&NamedProvenanceError::EmptyName)
    );
}

/// Test a part with no wire form has no data, at any depth.
#[rstest]
fn provenance_data_of_a_part_with_no_wire_form_fails(
    #[values(
        Nesting::Bare,
        Nesting::Named,
        Nesting::Callee,
        Nesting::Caller,
        Nesting::Fused,
        Nesting::LabelledFused,
        Nesting::NamedFusedNamed
    )]
    nesting: Nesting,
) {
    let provenance = nest(&UnwiredProvenance::build("p"), nesting);

    let result = ProvenanceData::of(&provenance);

    assert!(
        matches!(
            &result,
            Err(ForeignError::NoWireForm { type_name }) if type_name == "UnwiredProvenance"
        ),
        "{result:?}"
    );
}

// ---------------------------------------------------------------------------
// fuse
// ---------------------------------------------------------------------------

/// Test fusing keeps a custom source whole and in order.
#[test]
fn fuse_keeps_a_custom_source_whole_and_in_order() {
    let a = build_file("a.fhy", None);
    let b = build_file("b.fhy", None);
    let custom = EdgePropagation::build("e3");

    let fused = Provenance::fuse([a.clone(), Provenance::Unknown, custom.clone(), b.clone()]);

    let Provenance::Fused(fused) = &fused else {
        panic!("expected a fusion, got {fused:?}");
    };
    assert_eq!(fused.sources(), [a, custom, b]);
    assert_eq!(fused.label(), None);
}

/// Test fusing a lone custom provenance is that provenance.
#[test]
fn fuse_of_a_lone_custom_provenance_is_that_provenance() {
    let custom = EdgePropagation::build("e3");

    let fused = Provenance::fuse([Provenance::Unknown, custom.clone(), Provenance::Unknown]);

    assert_eq!(fused, custom);
    let Provenance::Custom(part) = &fused else {
        panic!("expected a custom provenance, got {fused:?}");
    };
    let Provenance::Custom(original) = &custom else {
        panic!("expected a custom provenance");
    };
    assert!(Part::ptr_eq(part, original));
}

/// Test a labelled fusion holds a custom source whole.
#[test]
fn fuse_labelled_keeps_a_custom_source() {
    let custom = EdgePropagation::build("e3");

    let fused = Provenance::fuse_labelled([custom.clone()], "loop-fusion");

    let Provenance::Fused(fused) = &fused else {
        panic!("expected a fusion, got {fused:?}");
    };
    assert_eq!(fused.sources(), [custom]);
    assert_eq!(fused.label(), Some("loop-fusion"));
}
