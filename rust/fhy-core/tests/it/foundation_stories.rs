//! Tests for the layer-1 foundation modules: `fhy_core::error`, whose
//! [`UnknownNameError`] every enum with a stable name text refuses an
//! unknown name with.

use std::fmt::Debug;
use std::str::FromStr;

use fhy_core::diagnostic::DiagnosticLevel;
use fhy_core::error::UnknownNameError;
use fhy_core::param::DomainKind;
use fhy_core::pass::PassHook;
use fhy_core::solver::{Logic, QueryKind};
use fhy_core::types::{CoreDataType, TypeQualifier};
use rstest::rstest;

/// Assert each of `variants` parses back from its `text`, and that an
/// unknown name is refused with the enum's `expected` description.
fn assert_parses_its_own_text<T>(variants: &[T], text: fn(&T) -> &'static str, expected: &str)
where
    T: FromStr<Err = UnknownNameError> + PartialEq + Debug,
{
    for variant in variants {
        assert_eq!(text(variant).parse::<T>().as_ref(), Ok(variant));
    }
    let error = "no such name".parse::<T>().expect_err("an unknown name");
    assert_eq!(error.name(), "no such name");
    assert_eq!(
        error.to_string(),
        format!("unknown {expected} `no such name`")
    );
    let first = text(&variants[0]);
    let recased = if first.chars().any(char::is_lowercase) {
        first.to_uppercase()
    } else {
        first.to_lowercase()
    };
    let error = recased
        .parse::<T>()
        .expect_err("the text is case-sensitive");
    assert!(
        error
            .to_string()
            .starts_with(&format!("unknown {expected} "))
    );
}

#[rstest]
#[case::diagnostic_level(|| assert_parses_its_own_text(
    &[DiagnosticLevel::Error, DiagnosticLevel::Warning, DiagnosticLevel::Info],
    |level| level.as_str(),
    "diagnostic level",
))]
#[case::pass_hook(|| assert_parses_its_own_text(
    &[
        PassHook::ValidateInput,
        PassHook::Skip,
        PassHook::Run,
        PassHook::ValidateOutput,
        PassHook::DidChange,
        PassHook::PreservedAnalyses,
    ],
    |hook| hook.as_str(),
    "pass hook",
))]
#[case::query_kind(|| assert_parses_its_own_text(
    &[
        QueryKind::Simplification,
        QueryKind::Satisfiability,
        QueryKind::Implication,
        QueryKind::UniversalValidity,
    ],
    |kind| kind.as_str(),
    "query kind",
))]
#[case::logic(|| assert_parses_its_own_text(
    &[
        Logic::QfLia,
        Logic::QfLra,
        Logic::QfNia,
        Logic::QfNra,
        Logic::Lia,
        Logic::Lra,
        Logic::Nia,
        Logic::Nra,
        Logic::All,
    ],
    |logic| logic.as_str(),
    "logic",
))]
#[case::domain_kind(|| assert_parses_its_own_text(
    &[
        DomainKind::Integer,
        DomainKind::IntervalInteger,
        DomainKind::Real,
        DomainKind::Ordinal,
        DomainKind::Categorical,
        DomainKind::Permutation,
        DomainKind::Custom,
    ],
    |kind| kind.name(),
    "domain kind",
))]
#[case::type_qualifier(|| assert_parses_its_own_text(
    &[
        TypeQualifier::Input,
        TypeQualifier::Output,
        TypeQualifier::State,
        TypeQualifier::Param,
        TypeQualifier::Temp,
    ],
    |qualifier| qualifier.as_str(),
    "type qualifier",
))]
#[case::core_data_type(|| assert_parses_its_own_text(
    &[CoreDataType::Int32, CoreDataType::Float64],
    |data_type| data_type.as_str(),
    "core data type",
))]
fn every_name_enum_parses_its_own_text(#[case] check: fn()) {
    check();
}

#[test]
fn a_query_kind_parses_its_name_not_its_words() {
    assert_eq!(
        QueryKind::UniversalValidity.to_string(),
        "universal validity"
    );
    "universal validity"
        .parse::<QueryKind>()
        .expect_err("the words are its display text, not its name");
}
