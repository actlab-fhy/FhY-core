//! Tests for the built-in frames and `FunctionKeyword`.

use crate::support::hashing::hash_of;
use crate::support::types::Equivalent;
use crate::support::types::{array, identifier_dimension, scalar};

use fhy_core::identifier::Identifier;
use fhy_core::symbol_table::{
    Frame, FunctionFrame, FunctionKeyword, ImportFrame, SymbolFrame, VariableFrame,
};
use fhy_core::types::{CoreDataType, TypeQualifier};

#[test]
fn an_import_frame_names_its_symbol() {
    let name = Identifier::new("module");
    let frame = ImportFrame::new(name.clone());

    assert_eq!(frame.name(), &name);
    assert_eq!(SymbolFrame::from(frame).name(), &name);
}

#[test]
fn a_variable_frame_keeps_its_type_and_qualifier() {
    let name = Identifier::new("x");
    let frame = VariableFrame::new(
        name.clone(),
        scalar(CoreDataType::Int32),
        TypeQualifier::Input,
    );

    assert_eq!(frame.name(), &name);
    assert_eq!(frame.ty(), &scalar(CoreDataType::Int32));
    assert_eq!(frame.qualifier(), TypeQualifier::Input);
    assert_eq!(SymbolFrame::from(frame).name(), &name);
}

#[test]
fn a_function_frame_keeps_its_keyword_and_signature() {
    let name = Identifier::new("f");
    let signature = vec![
        (TypeQualifier::Input, scalar(CoreDataType::Int32)),
        (TypeQualifier::Output, scalar(CoreDataType::Float32)),
    ];
    let frame = FunctionFrame::new(name.clone(), FunctionKeyword::Procedure, signature.clone());

    assert_eq!(frame.name(), &name);
    assert_eq!(frame.keyword(), FunctionKeyword::Procedure);
    assert_eq!(frame.signature(), signature.as_slice());
    assert_eq!(SymbolFrame::from(frame).name(), &name);
}

#[test]
fn a_function_frame_takes_its_signature_from_any_iterable() {
    let name = Identifier::new("f");
    let from_array = FunctionFrame::new(
        name.clone(),
        FunctionKeyword::Native,
        [(TypeQualifier::Param, scalar(CoreDataType::Bool))],
    );
    let from_iterator = FunctionFrame::new(
        name.clone(),
        FunctionKeyword::Native,
        std::iter::once((TypeQualifier::Param, scalar(CoreDataType::Bool))),
    );
    let empty = FunctionFrame::new(name, FunctionKeyword::Operation, []);

    assert_eq!(from_array, from_iterator);
    assert!(empty.signature().is_empty());
}

#[test]
fn frames_are_equal_and_hash_alike_over_every_field() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let n = Identifier::new("n");
    let variable = |name: &Identifier, qualifier| {
        SymbolFrame::from(VariableFrame::new(
            name.clone(),
            array(CoreDataType::Int32, [identifier_dimension(&n)]),
            qualifier,
        ))
    };

    let left = variable(&x, TypeQualifier::State);
    let right = variable(&x, TypeQualifier::State);
    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
    assert_ne!(left, variable(&y, TypeQualifier::State));
    assert_ne!(left, variable(&x, TypeQualifier::Temp));
    assert_ne!(
        left,
        SymbolFrame::from(VariableFrame::new(
            x.clone(),
            scalar(CoreDataType::Int32),
            TypeQualifier::State
        ))
    );
    assert_ne!(left, SymbolFrame::from(ImportFrame::new(x.clone())));

    let function = |keyword| {
        SymbolFrame::from(FunctionFrame::new(
            x.clone(),
            keyword,
            [(TypeQualifier::Input, scalar(CoreDataType::Int8))],
        ))
    };
    assert_eq!(
        hash_of(&function(FunctionKeyword::Operation)),
        hash_of(&function(FunctionKeyword::Operation))
    );
    assert_ne!(
        function(FunctionKeyword::Operation),
        function(FunctionKeyword::Native)
    );
}

#[test]
fn structural_equivalence_compares_kinds_names_and_types() {
    let x = Identifier::new("x");
    let import = SymbolFrame::from(ImportFrame::new(x.clone()));
    let variable = |ty| SymbolFrame::from(VariableFrame::new(x.clone(), ty, TypeQualifier::State));
    let function = |signature: Vec<(TypeQualifier, _)>| {
        SymbolFrame::from(FunctionFrame::new(
            x.clone(),
            FunctionKeyword::Procedure,
            signature,
        ))
    };

    assert!(import.is_equivalent(&SymbolFrame::from(ImportFrame::new(x.clone()))));
    assert!(!import.is_equivalent(&SymbolFrame::from(ImportFrame::new(Identifier::new("x")))));
    assert!(!import.is_equivalent(&variable(scalar(CoreDataType::Int32))));
    assert!(
        variable(scalar(CoreDataType::Int32)).is_equivalent(&variable(scalar(CoreDataType::Int32)))
    );
    assert!(
        !variable(scalar(CoreDataType::Int32))
            .is_equivalent(&variable(scalar(CoreDataType::Int64)))
    );
    let signature = vec![(TypeQualifier::Input, scalar(CoreDataType::Int32))];
    assert!(function(signature.clone()).is_equivalent(&function(signature.clone())));
    assert!(!function(signature).is_equivalent(&function(vec![])));
    assert!(
        !function(vec![(TypeQualifier::Input, scalar(CoreDataType::Int32))]).is_equivalent(
            &function(vec![(TypeQualifier::Output, scalar(CoreDataType::Int32))])
        )
    );
}

#[test]
fn the_variants_answer_structural_equivalence_alone() {
    let x = Identifier::new("x");
    let variable = VariableFrame::new(x.clone(), scalar(CoreDataType::Int32), TypeQualifier::State);
    let function = FunctionFrame::new(x, FunctionKeyword::Procedure, []);

    assert!(variable.is_equivalent(&variable.clone()));
    assert!(function.is_equivalent(&function.clone()));
}

#[test]
fn function_keywords_display_and_parse_their_short_names() {
    let cases = [
        (FunctionKeyword::Procedure, "proc"),
        (FunctionKeyword::Operation, "op"),
        (FunctionKeyword::Native, "native"),
    ];
    for (keyword, text) in cases {
        assert_eq!(keyword.as_str(), text);
        assert_eq!(keyword.to_string(), text);
        assert_eq!(text.parse::<FunctionKeyword>().expect("a keyword"), keyword);
    }
    let error = "procedure"
        .parse::<FunctionKeyword>()
        .expect_err("not a keyword");
    assert_eq!(error.name(), "procedure");
}

#[test]
fn function_keywords_round_trip_through_json_and_postcard() {
    for keyword in [
        FunctionKeyword::Procedure,
        FunctionKeyword::Operation,
        FunctionKeyword::Native,
    ] {
        let json = serde_json::to_string(&keyword).expect("serializes");
        assert_eq!(json, format!("\"{}\"", keyword.as_str()));
        assert_eq!(
            serde_json::from_str::<FunctionKeyword>(&json).expect("decodes"),
            keyword
        );
        let bytes = postcard::to_allocvec(&keyword).expect("serializes");
        assert_eq!(
            postcard::from_bytes::<FunctionKeyword>(&bytes).expect("decodes"),
            keyword
        );
    }
    serde_json::from_str::<FunctionKeyword>("\"procedure\"").expect_err("not a keyword");
}
