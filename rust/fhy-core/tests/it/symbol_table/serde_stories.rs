//! Tests for the serde form of the frames and the table
//! (`fhy_core::symbol_table::wire`).

use crate::support::foreign::{NamedType, TestResolver};
use crate::support::types::scalar;

use fhy_core::foreign::{BuildError, Foreign, ForeignError};
use fhy_core::identifier::Identifier;
use fhy_core::symbol_table::wire::{SymbolFrameData, SymbolTableData};
use fhy_core::symbol_table::{
    FunctionFrame, FunctionKeyword, ImportFrame, SymbolFrame, SymbolTable, VariableFrame,
};
use fhy_core::types::{CoreDataType, TypeQualifier};
use rstest::rstest;

fn restored(id: u64, name: &str) -> Identifier {
    Identifier::try_restore(id, name).expect("the id is below the cap")
}

fn round_trip_json(table: &SymbolTable<SymbolFrame>) -> SymbolTable<SymbolFrame> {
    serde_json::from_str(&serde_json::to_string(table).expect("encodes")).expect("decodes")
}

fn round_trip_postcard(table: &SymbolTable<SymbolFrame>) -> SymbolTable<SymbolFrame> {
    postcard::from_bytes(&postcard::to_allocvec(table).expect("encodes")).expect("decodes")
}

#[rstest]
#[case::import(
    SymbolFrame::Import(ImportFrame::new(restored(61_400, "f"))),
    r#"{"import":{"name":{"id":61400,"name_hint":"f"}}}"#
)]
#[case::variable(
    SymbolFrame::Variable(VariableFrame::new(
        restored(61_401, "x"),
        scalar(CoreDataType::Int32),
        TypeQualifier::State,
    )),
    concat!(
        r#"{"variable":{"name":{"id":61401,"name_hint":"x"},"#,
        r#""type":{"numerical":{"data_type":{"primitive":"int32"},"shape":[]}},"#,
        r#""type_qualifier":"state"}}"#,
    )
)]
#[case::function(
    SymbolFrame::Function(FunctionFrame::new(
        restored(61_402, "g"),
        FunctionKeyword::Operation,
        [(TypeQualifier::Input, scalar(CoreDataType::Bool))],
    )),
    concat!(
        r#"{"function":{"name":{"id":61402,"name_hint":"g"},"keyword":"op","signature":["#,
        r#"{"type_qualifier":"input","type":{"numerical":{"data_type":{"primitive":"bool"},"shape":[]}}}]}}"#,
    )
)]
fn a_frame_serializes_in_its_variant_form_and_round_trips(
    #[case] frame: SymbolFrame,
    #[case] text: &str,
) {
    assert_eq!(serde_json::to_string(&frame).expect("encodes"), text);
    assert_eq!(
        serde_json::from_str::<SymbolFrame>(text).expect("decodes"),
        frame
    );
    let bytes = postcard::to_allocvec(&frame).expect("encodes");
    assert_eq!(
        postcard::from_bytes::<SymbolFrame>(&bytes).expect("decodes"),
        frame
    );
}

fn build_table() -> SymbolTable<SymbolFrame> {
    let (root, child) = (restored(61_410, "root"), restored(61_411, "child"));
    let (x, f, y) = (
        restored(61_412, "x"),
        restored(61_413, "f"),
        restored(61_414, "y"),
    );
    let mut table = SymbolTable::new();
    table.add_namespace(root.clone(), None).expect("new");
    table
        .add_namespace(child.clone(), Some(root.clone()))
        .expect("new");
    table
        .add_symbol(
            &root,
            x.clone(),
            SymbolFrame::Variable(VariableFrame::new(
                x,
                scalar(CoreDataType::Float64),
                TypeQualifier::Input,
            )),
        )
        .expect("new");
    table
        .add_symbol(&root, f.clone(), SymbolFrame::Import(ImportFrame::new(f)))
        .expect("new");
    table
        .add_symbol(&child, y.clone(), SymbolFrame::Import(ImportFrame::new(y)))
        .expect("new");
    table
}

#[test]
fn a_table_round_trips_in_order_through_json_and_postcard() {
    let table = build_table();

    let from_json = round_trip_json(&table);
    let from_postcard = round_trip_postcard(&table);

    assert_eq!(from_json, table);
    assert_eq!(from_postcard, table);
    let names = |table: &SymbolTable<SymbolFrame>| {
        table
            .namespaces()
            .map(|namespace| namespace.name().clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(names(&from_json), names(&table));
}

#[test]
fn a_table_serializes_parents_and_symbols_in_order() {
    let text = serde_json::to_string(&build_table()).expect("encodes");

    assert!(text.starts_with(
        r#"{"namespaces":[{"namespace_name":{"id":61410,"name_hint":"root"},"parent_namespace_name":null,"symbols":[{"symbol_name":{"id":61412,"name_hint":"x"},"#
    ));
    assert!(text.contains(
        r#"{"namespace_name":{"id":61411,"name_hint":"child"},"parent_namespace_name":{"id":61410,"name_hint":"root"}"#
    ));
}

#[test]
fn a_table_refuses_a_namespace_defined_twice() {
    let text = concat!(
        r#"{"namespaces":["#,
        r#"{"namespace_name":{"id":61420,"name_hint":"a"},"parent_namespace_name":null,"symbols":[]},"#,
        r#"{"namespace_name":{"id":61420,"name_hint":"a"},"parent_namespace_name":null,"symbols":[]}]}"#,
    );

    let message = serde_json::from_str::<SymbolTable<SymbolFrame>>(text)
        .expect_err("the namespace is defined twice")
        .to_string();

    assert!(message.contains("already defined"), "{message}");
}

#[test]
fn a_table_refuses_a_symbol_its_parent_defines() {
    let x = r#"{"id":61423,"name_hint":"x"}"#;
    let text = format!(
        concat!(
            r#"{{"namespaces":["#,
            r#"{{"namespace_name":{{"id":61421,"name_hint":"p"}},"parent_namespace_name":null,"symbols":[{{"symbol_name":{x},"frame":{{"import":{{"name":{x}}}}}}}]}},"#,
            r#"{{"namespace_name":{{"id":61422,"name_hint":"c"}},"parent_namespace_name":{{"id":61421,"name_hint":"p"}},"symbols":[{{"symbol_name":{x},"frame":{{"import":{{"name":{x}}}}}}}]}}]}}"#,
        ),
        x = x
    );

    serde_json::from_str::<SymbolTable<SymbolFrame>>(&text).unwrap_err();
}

#[test]
fn a_custom_frame_is_read_as_its_foreign_part_and_refused_by_a_symbol_frame() {
    let data = SymbolFrameData::custom(Foreign::new("pkg.frame", "{}"));
    let text = serde_json::to_string(&data).expect("encodes");

    assert_eq!(text, r#"{"custom":{"type_id":"pkg.frame","data":"{}"}}"#);
    let read: SymbolFrameData = serde_json::from_str(&text).expect("reads");
    assert_eq!(read.foreign().map(Foreign::type_id), Some("pkg.frame"));
    assert!(matches!(
        read.build(&TestResolver),
        Err(BuildError::Foreign(ForeignError::Unresolved { .. }))
    ));
    serde_json::from_str::<SymbolFrame>(&text).unwrap_err();
}

#[test]
fn a_table_of_wire_frames_builds_through_a_frame_builder() {
    let (ns, x) = (restored(61_430, "ns"), restored(61_431, "x"));
    let mut table: SymbolTable<SymbolFrame> = SymbolTable::new();
    table.add_namespace(ns.clone(), None).expect("new");
    table
        .add_symbol(
            &ns,
            x.clone(),
            SymbolFrame::Variable(VariableFrame::new(
                x,
                NamedType::build("q"),
                TypeQualifier::Param,
            )),
        )
        .expect("new");
    let text = serde_json::to_string(&table).expect("encodes");

    let data: SymbolTableData<SymbolFrameData> = serde_json::from_str(&text).expect("reads");
    let rebuilt = data
        .build(|frame| frame.build(&TestResolver))
        .expect("the resolver knows the type");

    assert_eq!(rebuilt, table);
    serde_json::from_str::<SymbolTable<SymbolFrame>>(&text).unwrap_err();
}

#[test]
fn a_table_wire_form_is_written_from_any_frame_type() {
    let (ns, x) = (restored(61_440, "ns"), restored(61_441, "x"));
    let mut table: SymbolTable<&str> = SymbolTable::new();
    table.add_namespace(ns.clone(), None).expect("new");
    table.add_symbol(&ns, x, "a frame").expect("new");

    let data = SymbolTableData::of(&table, |frame| Ok::<_, ForeignError>(frame.len()))
        .expect("every frame maps");

    assert!(
        serde_json::to_string(&data)
            .expect("encodes")
            .ends_with(r#""frame":7}]}]}"#)
    );
}

/// A table built through the checked API with a child added before its
/// parent: the child's symbol could not be added while its parent was
/// missing on decode (F2-020).
#[test]
fn a_table_whose_child_was_added_before_its_parent_round_trips() {
    let (parent, child, x) = (
        restored(61_430, "parent"),
        restored(61_431, "child"),
        restored(61_432, "x"),
    );
    let mut table = SymbolTable::new();
    table
        .add_namespace(child.clone(), Some(parent.clone()))
        .expect("new");
    table.add_namespace(parent.clone(), None).expect("new");
    table
        .add_symbol(
            &child,
            x.clone(),
            SymbolFrame::Import(ImportFrame::new(x.clone())),
        )
        .expect("new");
    assert!(table.violations().is_empty());

    for decoded in [round_trip_json(&table), round_trip_postcard(&table)] {
        assert_eq!(decoded, table);
        let names: Vec<&Identifier> = decoded
            .namespaces()
            .map(|namespace| namespace.name())
            .collect();
        assert_eq!(names, [&child, &parent]);
        assert_eq!(
            serde_json::to_string(&decoded).expect("encodes"),
            serde_json::to_string(&table).expect("encodes")
        );
    }
}

/// The TYP probe's second table: `y` added to a child and then to its
/// parent. `add_symbol` now refuses the second, so no such table is built
/// through the checked API, and one built unchecked fails to decode.
#[test]
fn a_symbol_added_to_a_child_then_its_parent_is_refused() {
    let (parent, child, y) = (
        restored(61_433, "parent"),
        restored(61_434, "child"),
        restored(61_435, "y"),
    );
    let frame = || SymbolFrame::Import(ImportFrame::new(y.clone()));
    let mut table = SymbolTable::new();
    table.add_namespace(parent.clone(), None).expect("new");
    table
        .add_namespace(child.clone(), Some(parent.clone()))
        .expect("new");
    table.add_symbol(&child, y.clone(), frame()).expect("new");

    let refused = table.add_symbol(&parent, y.clone(), frame());

    assert!(
        matches!(
            refused,
            Err(fhy_core::symbol_table::SymbolTableError::SymbolDefinedInDescendant { .. })
        ),
        "{refused:?}"
    );
    let mut unchecked = SymbolTable::new();
    unchecked.insert_namespace(parent.clone(), None, [(y.clone(), frame())]);
    unchecked.insert_namespace(child.clone(), Some(parent.clone()), [(y.clone(), frame())]);
    let text = serde_json::to_string(&unchecked).expect("encodes");
    let error = serde_json::from_str::<SymbolTable<SymbolFrame>>(&text).expect_err("shadowing");
    assert!(error.to_string().contains("already defined"), "{error}");
}
