//! Tests for the serde form of domains, params and assignments
//! (`fhy_core::param::wire`).

use crate::support::foreign::{TestResolver, WireDomain, WireToken};
use crate::support::param::{at_least, in_set, ints};

use fhy_core::constraint::Value;
use fhy_core::foreign::BuildError;
use fhy_core::identifier::Identifier;
use fhy_core::param::wire::{ParamAssignmentData, ParamData, ParamDomainData};
use fhy_core::param::{
    CategoricalDomain, IntegerDomain, IntervalIntegerDomain, OrdinalDomain, Param, ParamAssignment,
    ParamContext, ParamDomain, PermutationDomain, RealDomain,
};
use fhy_core::solver::Solver;
use rstest::rstest;

fn restored(id: u64, name: &str) -> Identifier {
    Identifier::try_restore(id, name).expect("the id is below the cap")
}

fn text_of(value: &impl serde::Serialize) -> String {
    serde_json::to_string(value).expect("encodes")
}

fn strs(values: &[&str]) -> Vec<Value> {
    values
        .iter()
        .map(|value| Value::Str((*value).to_owned()))
        .collect()
}

#[rstest]
#[case::integer(
    ParamDomain::from(IntegerDomain::new(false, true)),
    r#"{"integer":{"non_negative":false,"zero_included":true}}"#
)]
#[case::natural(
    ParamDomain::from(IntegerDomain::new(true, false)),
    r#"{"integer":{"non_negative":true,"zero_included":false}}"#
)]
#[case::interval(
    ParamDomain::from(IntervalIntegerDomain::new(true, false, true)),
    r#"{"interval_integer":{"prefer_inclusive":true,"non_negative":false,"zero_included":true}}"#
)]
#[case::real(ParamDomain::from(RealDomain), r#""real""#)]
#[case::ordinal(
    ParamDomain::from(OrdinalDomain::new(vec![Value::Int(3.into()), Value::Float(1.5), Value::Int(1.into())]).expect("valid")),
    r#"{"ordinal":{"sorted_values":[{"int":"1"},{"float":"1.5"},{"int":"3"}]}}"#
)]
#[case::categorical(
    ParamDomain::from(CategoricalDomain::new(strs(&["b", "a"])).expect("valid")),
    r#"{"categorical":{"categories":[{"str":"a"},{"str":"b"}]}}"#
)]
#[case::permutation(
    ParamDomain::from(PermutationDomain::new(strs(&["n", "c", "h"])).expect("valid")),
    r#"{"permutation":{"ordered_members":[{"str":"n"},{"str":"c"},{"str":"h"}]}}"#
)]
fn a_domain_serializes_in_its_kind_form_and_round_trips(
    #[case] domain: ParamDomain,
    #[case] text: &str,
) {
    assert_eq!(text_of(&domain), text);
    let from_json: ParamDomain = serde_json::from_str(text).expect("decodes");
    assert!(from_json.is_structurally_equivalent(&domain));
    let bytes = postcard::to_allocvec(&domain).expect("encodes");
    let from_postcard: ParamDomain = postcard::from_bytes(&bytes).expect("decodes");
    assert!(from_postcard.is_structurally_equivalent(&domain));
}

#[rstest]
#[case::empty(r#"{"ordinal":{"sorted_values":[]}}"#)]
#[case::duplicate(r#"{"categorical":{"categories":[{"str":"a"},{"str":"a"}]}}"#)]
#[case::nested(r#"{"permutation":{"ordered_members":[{"tuple":[]}]}}"#)]
#[case::incomparable(r#"{"ordinal":{"sorted_values":[{"int":"1"},{"str":"a"}]}}"#)]
fn a_finite_domain_refuses_what_its_constructor_refuses(#[case] text: &str) {
    let data: ParamDomainData = serde_json::from_str(text).expect("the shape reads");

    assert!(matches!(
        data.build(&TestResolver),
        Err(BuildError::Invalid(_))
    ));
    serde_json::from_str::<ParamDomain>(text).unwrap_err();
}

#[test]
fn an_ordinal_domain_reads_its_values_in_any_order() {
    let decoded: ParamDomain = serde_json::from_str(
        r#"{"ordinal":{"sorted_values":[{"int":"3"},{"int":"1"},{"int":"2"}]}}"#,
    )
    .expect("decodes");

    assert_eq!(
        text_of(&decoded),
        r#"{"ordinal":{"sorted_values":[{"int":"1"},{"int":"2"},{"int":"3"}]}}"#
    );
}

#[test]
fn a_custom_domain_and_opaque_members_resolve() {
    let custom = WireDomain::build("even");
    let categorical = ParamDomain::from(
        CategoricalDomain::new(vec![WireToken::value(2), WireToken::value(1)]).expect("valid"),
    );

    for domain in [custom, categorical] {
        let text = text_of(&domain);
        assert!(
            serde_json::from_str::<ParamDomain>(&text).is_err(),
            "{text}"
        );
        let rebuilt = serde_json::from_str::<ParamDomainData>(&text)
            .expect("reads")
            .build(&TestResolver)
            .expect("the resolver knows the parts");
        assert!(rebuilt.is_structurally_equivalent(&domain));
        assert_eq!(text_of(&rebuilt), text);
    }
    assert_eq!(
        text_of(&WireDomain::build("even")),
        r#"{"custom":{"type_id":"test.domain","data":"even"}}"#
    );
}

fn build_param(variable: &Identifier) -> Param {
    let solver = Solver::new();
    Param::new(
        ParamDomain::from(IntegerDomain::new(true, true)),
        variable.clone(),
        [in_set(variable, ints([1, 2, 3])), at_least(variable, 1)],
        &ParamContext::new(&solver),
    )
    .expect("the constraints are in scope")
}

#[test]
fn a_param_serializes_its_domain_variable_and_system() {
    let variable = restored(61_600, "p");
    let param = build_param(&variable);

    let text = text_of(&param);

    assert!(
        text.starts_with(
            r#"{"domain":{"integer":{"non_negative":true,"zero_included":true}},"variable":{"id":61600,"name_hint":"p"},"constraint_system":{"constraints":["#
        ),
        "{text}"
    );
    let decoded: Param = serde_json::from_str(&text).expect("decodes");
    assert!(decoded.is_structurally_equivalent(&param));
    assert_eq!(text_of(&decoded), text);
    let from_postcard: Param =
        postcard::from_bytes(&postcard::to_allocvec(&param).expect("encodes")).expect("decodes");
    assert!(from_postcard.is_structurally_equivalent(&param));
}

#[test]
fn a_param_refuses_a_constraint_outside_its_variable_s_scope() {
    let (variable, other) = (restored(61_601, "p"), restored(61_602, "q"));
    let text = text_of(&build_param(&variable)).replace(
        r#""variable":{"id":61601,"name_hint":"p"},"constraint_system""#,
        r#""variable":{"id":61602,"name_hint":"q"},"constraint_system""#,
    );
    let _ = other;

    let data: ParamData = serde_json::from_str(&text).expect("reads");
    let solver = Solver::new();

    assert!(matches!(
        data.build(&TestResolver, &ParamContext::new(&solver)),
        Err(BuildError::Invalid(_))
    ));
}

#[test]
fn an_assignment_round_trips_without_checking_its_value() {
    let variable = restored(61_603, "p");
    let assignment = ParamAssignment::new_unchecked(build_param(&variable), Value::Int(2.into()));

    let text = text_of(&assignment);

    assert!(text.ends_with(r#""value":{"int":"2"}}"#), "{text}");
    let decoded: ParamAssignment = serde_json::from_str(&text).expect("decodes");
    assert!(decoded.is_structurally_equivalent(&assignment));
    let unchecked = text.replace(r#""value":{"int":"2"}"#, r#""value":{"int":"9"}"#);
    let data: ParamAssignmentData = serde_json::from_str(&unchecked).expect("reads");
    let solver = Solver::new();
    let (_, value) = data.clone().into_parts();
    assert_eq!(text_of(&value), r#"{"int":"9"}"#);
    data.build(&TestResolver, &ParamContext::new(&solver))
        .unwrap();
}
