//! Tests for the serde form of domains, params and assignments
//! (`fhy_core::param::wire`).

use crate::support::foreign::{TestResolver, WireDomain, WireToken};
use crate::support::param::{at_least, in_set, ints};
use fhy_core::param::{Inclusivity, Sign, ZeroInclusion};

use fhy_core::constraint::Value;
use fhy_core::foreign::BuildError;
use fhy_core::identifier::Identifier;
use fhy_core::param::wire::{ParamAssignmentData, ParamData, ParamDomainData};
use fhy_core::param::{
    CategoricalDomain, IntegerDomain, IntervalIntegerDomain, OrdinalDomain, Param, ParamAssignment,
    ParamContext, ParamDomain, PermutationDomain, RealDomain,
};
use fhy_core::solver::Solver;
use proptest::prelude::*;
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
    ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included)),
    r#"{"integer":{"non_negative":false,"zero_included":true}}"#
)]
#[case::natural(
    ParamDomain::from(IntegerDomain::new(Sign::NonNegative, ZeroInclusion::Excluded)),
    r#"{"integer":{"non_negative":true,"zero_included":false}}"#
)]
#[case::interval(
    ParamDomain::from(IntervalIntegerDomain::new(
        Inclusivity::Inclusive,
        Sign::Any,
        ZeroInclusion::Included
    )),
    r#"{"interval_integer":{"prefer_inclusive":true,"non_negative":false,"zero_included":true}}"#
)]
#[case::real(ParamDomain::from(RealDomain), r#"{"real":{}}"#)]
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
        ParamDomain::from(IntegerDomain::new(
            Sign::NonNegative,
            ZeroInclusion::Included,
        )),
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
fn an_assignment_round_trips_and_its_value_is_checked_on_decode() {
    let variable = restored(61_603, "p");
    let assignment = ParamAssignment::new_unvalidated(build_param(&variable), Value::Int(2.into()));

    let text = text_of(&assignment);

    assert!(text.ends_with(r#""value":{"int":"2"}}"#), "{text}");
    let decoded: ParamAssignment = serde_json::from_str(&text).expect("decodes");
    assert!(decoded.is_structurally_equivalent(&assignment));
    let from_postcard: ParamAssignment =
        postcard::from_bytes(&postcard::to_allocvec(&assignment).expect("encodes"))
            .expect("decodes");
    assert!(from_postcard.is_structurally_equivalent(&assignment));
    let violating = text.replace(r#""value":{"int":"2"}"#, r#""value":{"int":"9"}"#);
    let data: ParamAssignmentData = serde_json::from_str(&violating).expect("reads");
    let (_, value) = data.into_parts();
    assert_eq!(text_of(&value), r#"{"int":"9"}"#);
    let error =
        serde_json::from_str::<ParamAssignment>(&violating).expect_err("9 is outside the in-set");
    assert!(
        error
            .to_string()
            .starts_with("the value violates the param's constraint"),
        "{error}"
    );
}

/// An integer param assigned `"not an integer"` must not round-trip: decoding
/// has to validate the payload against the param's type, not just parse it.
#[test]
fn an_inadmissible_assignment_payload_fails_to_decode() {
    let variable = restored(61_604, "p");
    let assignment = ParamAssignment::new_unvalidated(build_param(&variable), Value::Int(2.into()));
    let text = text_of(&assignment).replace(
        r#""value":{"int":"2"}"#,
        r#""value":{"str":"not an integer"}"#,
    );

    let error = serde_json::from_str::<ParamAssignment>(&text).expect_err("inadmissible");

    assert!(error.to_string().contains("not admissible"), "{error}");
    let data: ParamAssignmentData = serde_json::from_str(&text).expect("reads");
    let solver = Solver::new();
    let refused = data.build(&TestResolver, &ParamContext::new(&solver));
    assert!(
        matches!(
            &refused,
            Err(BuildError::Invalid(source))
                if matches!(
                    source.downcast_ref::<fhy_core::param::AssignmentError>(),
                    Some(fhy_core::param::AssignmentError::Inadmissible)
                )
        ),
        "{refused:?}"
    );
}

/// Return a strategy for domain members: integers, Booleans, short strings,
/// and tuples and frozen sets of those.
fn member_value_strategy() -> BoxedStrategy<Value> {
    let leaf = prop_oneof![
        (-3_i64..6).prop_map(|value| Value::Int(value.into())),
        any::<bool>().prop_map(Value::Bool),
        "[ab]{1,2}".prop_map(Value::Str),
    ];
    leaf.prop_recursive(2, 6, 3, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 1..3).prop_map(Value::Tuple),
            prop::collection::vec(inner, 1..3).prop_map(Value::FrozenSet),
        ]
    })
    .boxed()
}

/// Return a strategy for the six built-in domain kinds.
fn domain_strategy() -> impl Strategy<Value = ParamDomain> {
    let sign = prop_oneof![Just(Sign::Any), Just(Sign::NonNegative)];
    let zero = prop_oneof![Just(ZeroInclusion::Included), Just(ZeroInclusion::Excluded)];
    let inclusivity = prop_oneof![Just(Inclusivity::Inclusive), Just(Inclusivity::Exclusive)];
    let values = || prop::collection::vec(member_value_strategy(), 1..5);
    prop_oneof![
        (sign.clone(), zero.clone())
            .prop_map(|(sign, zero)| ParamDomain::from(IntegerDomain::new(sign, zero))),
        (inclusivity, sign, zero).prop_map(|(inclusivity, sign, zero)| {
            ParamDomain::from(IntervalIntegerDomain::new(inclusivity, sign, zero))
        }),
        Just(ParamDomain::from(RealDomain)),
        prop::collection::vec(-5_i64..5, 1..5).prop_filter_map("distinct values", |values| {
            OrdinalDomain::new(
                values
                    .into_iter()
                    .map(|value| Value::Int(value.into()))
                    .collect(),
            )
            .ok()
            .map(ParamDomain::from)
        }),
        values().prop_filter_map("distinct leaf values", |values| {
            CategoricalDomain::new(values).ok().map(ParamDomain::from)
        }),
        values().prop_filter_map("distinct members", |values| {
            PermutationDomain::new(values).ok().map(ParamDomain::from)
        }),
    ]
}

proptest::proptest! {
    /// Every built-in domain round-trips through JSON and postcard, members
    /// of kind bool, tuple and frozen set included, and its JSON re-encodes
    /// byte-identically.
    #[test]
    fn a_domain_round_trips_through_serde(domain in domain_strategy()) {
        crate::support::serde::check_serde_round_trip(&domain)?;
    }
}
