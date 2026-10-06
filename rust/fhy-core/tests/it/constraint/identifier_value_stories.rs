//! Tests for identifier values and members (`Value::Identifier`): an
//! identifier used as a constant, such as a category of a param, equal to
//! the identifier with its id and to nothing else.
//!
//! They cover equality and hashing, the canonical order, display, lifting,
//! set constraints and equations over identifiers, the wire form and the
//! opaque form 0.2.0 wrote, and the finite param domains.

use std::collections::{HashMap, HashSet};

use fhy_core::constraint::wire::{ConstraintData, ValueData};
use fhy_core::constraint::{
    Binding, Bindings, Constraint, ConstraintContext, ConstraintError, EquationConstraint, Member,
    MemberKind, Outcome, Polarity, SetConstraint, UnusableBindingReason, Value,
};
use fhy_core::expression::{BigInt, Expression, LiteralValue};
use fhy_core::identifier::Identifier;
use fhy_core::param::wire::ParamDomainData;
use fhy_core::param::{
    AssignmentError, CategoricalDomain, DomainError, OrdinalDomain, Param, ParamAssignment,
    ParamContext, ParamDomain, PermutationDomain,
};
use fhy_core::solver::Solver;
use fhy_core::term::AlphaRenaming;
use rstest::rstest;

use crate::support::constraint::{TestOpaque, int, member, member_set, text};
use crate::support::foreign::{LegacyIdentifier, LegacyResolver, WireToken};
use crate::support::hashing::hash_of;
use crate::support::serde::restored;

/// Return the identifier value of the identifier restored at `id`.
fn identifier(id: u64, name: &str) -> Value {
    Value::Identifier(restored(id, name))
}

/// Return the JSON text of `value`.
fn text_of(value: &impl serde::Serialize) -> String {
    serde_json::to_string(value).expect("encodes")
}

/// Return the in-set constraint on `variable` of `values`.
fn in_set(variable: &Identifier, values: impl IntoIterator<Item = Value>) -> SetConstraint {
    SetConstraint::new(variable.clone(), member_set(values), Polarity::In)
}

/// Return the outcome of `constraint` with `variable` bound to `value`.
fn decide(constraint: &Constraint, variable: &Identifier, value: Value) -> Outcome {
    let solver = Solver::new();
    let bindings = Bindings::from_iter([(variable.clone(), Binding::Value(value))]);
    constraint
        .evaluate(&bindings, &ConstraintContext::new(&solver))
        .expect("an identifier binding is decided")
}

// ---------------------------------------------------------------------------
// Equality and hashing
// ---------------------------------------------------------------------------

#[test]
fn identifier_values_with_one_id_are_equal_whatever_their_name_hints() {
    let left = identifier(62_000, "a");
    let right = identifier(62_000, "renamed");

    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
    assert_eq!(member(left.clone()), member(right.clone()));
    assert_eq!(hash_of(&member(left)), hash_of(&member(right)));
}

#[rstest]
#[case::another_id(identifier(62_001, "a"))]
#[case::its_name_hint_as_a_string(text("a"))]
#[case::its_id_as_an_integer(int(62_000))]
#[case::a_one_tuple_of_it(Value::Tuple(vec![identifier(62_000, "a")]))]
#[case::an_opaque_value_reporting_it(LegacyIdentifier::value(&restored(62_000, "a")))]
fn identifier_value_equals_nothing_but_an_identifier_with_its_id(#[case] other: Value) {
    let value = identifier(62_000, "a");

    assert_ne!(value, other);
    assert_ne!(other, value);
    assert_ne!(member(value), member(other));
}

#[test]
fn identifier_member_shows_its_identifier_and_kind() {
    let a = restored(62_002, "a");
    let stored = member(Value::Identifier(a.clone()));

    let MemberKind::Identifier(held) = stored.kind() else {
        panic!("expected an identifier, got {:?}", stored.kind());
    };
    assert_eq!(held, &a);
    assert_eq!(held.name_hint(), "a");
    assert_eq!(stored.kind_name(), "identifier");
}

#[test]
fn identifier_value_is_member_shaped_and_hashable_at_any_depth() {
    let nested = Value::FrozenSet(vec![Value::Tuple(vec![identifier(62_003, "a"), int(1)])]);

    assert!(identifier(62_003, "a").is_member_shaped());
    assert!(nested.is_member_shaped());
    nested.check_hashable().expect("an identifier is hashable");
    let stored = member(nested);
    assert_eq!(stored.kind_name(), "frozenset");
}

// ---------------------------------------------------------------------------
// The canonical order
// ---------------------------------------------------------------------------

#[test]
fn member_set_orders_identifiers_by_id_not_by_name_hint() {
    let members = member_set([
        identifier(1_000, "a"),
        identifier(100, "m"),
        identifier(99, "z"),
    ]);

    let ids: Vec<u64> = members
        .iter()
        .map(|member| match member.kind() {
            MemberKind::Identifier(identifier) => identifier.id(),
            other => panic!("expected an identifier, got {other:?}"),
        })
        .collect();
    assert_eq!(ids, [99, 100, 1_000]);
}

#[test]
fn member_set_places_identifiers_between_frozen_sets_and_integers() {
    let members = member_set([
        TestOpaque::token(1).into_value(),
        Value::Tuple(vec![int(2)]),
        text("a"),
        int(1),
        identifier(62_004, "a"),
        Value::FrozenSet(vec![int(5)]),
        Value::Float(1.0),
        Value::Bool(true),
    ]);

    let kinds: Vec<String> = members
        .iter()
        .map(|member| member.kind_name().into_owned())
        .collect();
    assert_eq!(
        kinds,
        [
            "bool",
            "float",
            "frozenset",
            "identifier",
            "int",
            "str",
            "tuple",
            "Token"
        ]
    );
}

#[test]
fn member_set_keeps_the_first_of_identifiers_with_one_id() {
    let members = member_set([
        identifier(62_005, "first"),
        identifier(62_006, "other"),
        identifier(62_005, "second"),
    ]);

    assert_eq!(members.len(), 2);
    assert_eq!(members.to_string(), "{first, other}");
}

#[test]
fn identifier_members_order_against_each_other_by_id() {
    let low = member(identifier(99, "z"));
    let high = member(identifier(100, "a"));

    assert!(low < high);
    assert!(member(identifier(62_007, "c")) > member(Value::FrozenSet(vec![int(9)])));
    assert!(member(identifier(62_007, "c")) < member(int(-1)));
}

// ---------------------------------------------------------------------------
// Display and lifting
// ---------------------------------------------------------------------------

#[test]
fn identifier_displays_as_its_name_hint() {
    let x = restored(62_010, "x");
    let constraint = in_set(&x, [identifier(62_011, "b"), identifier(62_012, "a")]);

    assert_eq!(identifier(62_011, "b").to_string(), "b");
    assert_eq!(member(identifier(62_011, "b")).to_string(), "b");
    assert_eq!(
        Value::Tuple(vec![identifier(62_011, "b")]).to_string(),
        "(b,)"
    );
    assert_eq!(constraint.to_string(), "x in {b, a}");
}

#[test]
fn identifier_member_does_not_lift_to_an_expression() {
    let stored = member(identifier(62_013, "a"));

    assert!(!stored.lifts_to_expression());
    assert_eq!(stored.to_literal(), None);
}

#[test]
fn set_constraint_over_an_identifier_has_no_expression() {
    let x = restored(62_014, "x");
    let constraint = in_set(&x, [int(1), identifier(62_015, "a")]);

    let error = constraint
        .to_expression()
        .expect_err("an identifier member does not lift");

    assert!(
        matches!(&error, ConstraintError::UnliftableMember(refused) if *refused == member(identifier(62_015, "a"))),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        "conversion of type identifier to an expression is not supported"
    );
}

// ---------------------------------------------------------------------------
// Constraints over identifiers
// ---------------------------------------------------------------------------

#[test]
fn set_of_identifiers_holds_a_value_by_id_only() {
    let members = member_set([identifier(62_020, "a"), identifier(62_021, "b")]);

    assert!(members.contains_value(&identifier(62_020, "renamed")));
    assert!(!members.contains_value(&identifier(62_022, "a")));
    assert!(!members.contains_value(&text("a")));
    assert!(!members.contains_value(&int(62_020)));
    assert!(members.contains(&member(identifier(62_021, "b"))));
}

#[test]
fn set_of_tuples_holds_a_tuple_holding_an_identifier() {
    let members = member_set([Value::Tuple(vec![identifier(62_023, "a"), int(1)])]);

    assert!(members.contains_value(&Value::Tuple(vec![identifier(62_023, "a"), int(1)])));
    assert!(!members.contains_value(&Value::Tuple(vec![text("a"), int(1)])));
}

#[rstest]
#[case::the_same_id(
    identifier(62_030, "other_hint"),
    Outcome::Satisfied,
    Outcome::Violated
)]
#[case::another_id(identifier(62_032, "a"), Outcome::Violated, Outcome::Satisfied)]
#[case::its_name(text("a"), Outcome::Violated, Outcome::Satisfied)]
#[case::its_id(int(62_030), Outcome::Violated, Outcome::Satisfied)]
fn set_constraint_over_identifiers_is_decided_by_id(
    #[case] bound: Value,
    #[case] in_outcome: Outcome,
    #[case] not_in_outcome: Outcome,
) {
    let x = restored(62_033, "x");
    let members = member_set([identifier(62_030, "a"), identifier(62_031, "b")]);
    let inside = Constraint::from(SetConstraint::new(x.clone(), members.clone(), Polarity::In));
    let outside = Constraint::from(SetConstraint::new(x.clone(), members, Polarity::NotIn));

    assert_eq!(decide(&inside, &x, bound.clone()), in_outcome);
    assert_eq!(decide(&outside, &x, bound), not_in_outcome);
}

#[test]
fn equation_refuses_an_identifier_binding_as_no_literal() {
    let x = restored(62_034, "x");
    let equation = Constraint::from(EquationConstraint::new(
        Expression::from(&x).equals(Expression::literal(LiteralValue::Int(BigInt::from(1)))),
    ));
    let solver = Solver::new();
    let bindings = Bindings::from_iter([(x.clone(), Binding::Value(identifier(62_035, "a")))]);

    let error = equation
        .evaluate(&bindings, &ConstraintContext::new(&solver))
        .expect_err("an identifier is no literal");

    assert!(
        matches!(
            &error,
            ConstraintError::UnusableBinding { identifier, reason: UnusableBindingReason::NotALiteral }
                if *identifier == x
        ),
        "{error:?}"
    );
}

#[test]
fn identifier_member_is_no_free_identifier_of_its_constraint() {
    let x = restored(62_036, "x");
    let constraint = Constraint::from(in_set(&x, [identifier(62_037, "a")]));

    let free = constraint
        .free_identifiers()
        .expect("a set constraint has a scope");

    assert_eq!(free, HashSet::from([x]));
}

#[test]
fn set_constraints_over_one_identifier_of_two_name_hints_are_one() {
    let x = restored(62_038, "x");
    let left = in_set(&x, [identifier(62_039, "a")]);
    let right = in_set(&x, [identifier(62_039, "renamed")]);

    assert!(left.is_structurally_equivalent(&right));
    assert_eq!(left.ordering_key(), right.ordering_key());
    assert_eq!(hash_of(&left), hash_of(&right));
}

#[test]
fn set_constraint_key_writes_an_identifier_member_by_id() {
    let x = restored(62_040, "x");

    assert_eq!(
        in_set(&x, [identifier(62_041, "a")]).ordering_key(),
        "in_set|62040|{identifier:62041}"
    );
    assert_ne!(
        in_set(&x, [identifier(62_041, "a")]).ordering_key(),
        in_set(&x, [text("a")]).ordering_key()
    );
}

#[test]
fn alpha_equivalence_of_set_constraints_compares_identifier_members_by_value() {
    let (x, y) = (restored(62_042, "x"), restored(62_043, "y"));
    let (a, b) = (restored(62_044, "a"), restored(62_045, "b"));
    let mut renaming = AlphaRenaming::new(HashMap::new()).expect("an empty renaming");
    renaming
        .enter_binders(&[x.clone(), a.clone()], &[y.clone(), b.clone()])
        .expect("distinct binders");

    let left = in_set(&x, [Value::Identifier(a.clone())]);

    assert!(left.is_alpha_equivalent_under(&in_set(&y, [Value::Identifier(a)]), &renaming));
    assert!(!left.is_alpha_equivalent_under(&in_set(&y, [Value::Identifier(b)]), &renaming));
}

// ---------------------------------------------------------------------------
// The wire form
// ---------------------------------------------------------------------------

#[test]
fn identifier_value_serializes_as_its_identifier() {
    let value = identifier(62_050, "a");
    let wire = r#"{"identifier":{"id":62050,"name_hint":"a"}}"#;

    assert_eq!(text_of(&value), wire);
    assert_eq!(text_of(&member(value.clone())), wire);
    assert_eq!(
        text_of(&ValueData::of_value(&value).expect("no opaque part")),
        wire
    );
    let decoded: Value = serde_json::from_str(wire).expect("decodes");
    assert_eq!(decoded, value);
    assert_eq!(decoded.to_string(), "a");
    let from_postcard: Value =
        postcard::from_bytes(&postcard::to_allocvec(&value).expect("encodes")).expect("decodes");
    assert_eq!(text_of(&from_postcard), wire);
}

#[test]
fn identifier_member_decodes_as_a_member() {
    let decoded: Member =
        serde_json::from_str(r#"{"identifier":{"id":62051,"name_hint":"a"}}"#).expect("decodes");

    assert_eq!(decoded, member(identifier(62_051, "a")));
}

#[test]
fn set_constraint_writes_identifier_members_in_canonical_order() {
    let x = restored(62_052, "x");
    let constraint = Constraint::from(in_set(
        &x,
        [identifier(100, "b"), int(1), identifier(99, "a")],
    ));
    let wire = concat!(
        r#"{"in_set":{"variable":{"id":62052,"name_hint":"x"},"values":["#,
        r#"{"identifier":{"id":99,"name_hint":"a"}},"#,
        r#"{"identifier":{"id":100,"name_hint":"b"}},"#,
        r#"{"int":"1"}]}}"#
    );

    assert_eq!(text_of(&constraint), wire);
    let decoded: Constraint = serde_json::from_str(wire).expect("decodes");
    assert!(decoded.is_structurally_equivalent(&constraint));
    let from_postcard: Constraint =
        postcard::from_bytes(&postcard::to_allocvec(&constraint).expect("encodes"))
            .expect("decodes");
    assert!(from_postcard.is_structurally_equivalent(&constraint));
}

#[test]
fn tuple_holding_an_identifier_serializes_it_in_place() {
    let value = Value::Tuple(vec![identifier(62_053, "a"), int(2)]);
    let wire = r#"{"tuple":[{"identifier":{"id":62053,"name_hint":"a"}},{"int":"2"}]}"#;

    assert_eq!(text_of(&value), wire);
    assert_eq!(serde_json::from_str::<Value>(wire).expect("decodes"), value);
}

#[rstest]
#[case::not_an_identifier(r#"{"identifier":"a"}"#)]
#[case::without_a_name_hint(r#"{"identifier":{"id":62054}}"#)]
#[case::an_id_past_the_cap(r#"{"identifier":{"id":18446744073709551615,"name_hint":"a"}}"#)]
fn malformed_identifier_value_is_refused(#[case] wire: &str) {
    serde_json::from_str::<Value>(wire).expect_err("a malformed identifier is refused");
    serde_json::from_str::<ValueData>(wire).expect_err("a malformed identifier is refused");
}

#[test]
fn legacy_opaque_identifier_reads_as_an_identifier() {
    let a = restored(62_060, "a");
    let wire = LegacyIdentifier::wire(&a);

    let data: ValueData = serde_json::from_str(&wire).expect("reads");
    let value = data.clone().build(&LegacyResolver).expect("resolves");
    let built_member = data.build_member(&LegacyResolver).expect("resolves");

    assert!(matches!(&value, Value::Identifier(held) if *held == a && held.name_hint() == "a"));
    assert_eq!(built_member, member(Value::Identifier(a.clone())));
    assert_eq!(
        text_of(&value),
        r#"{"identifier":{"id":62060,"name_hint":"a"}}"#
    );
}

#[test]
fn legacy_opaque_identifier_inside_a_tuple_reads_as_an_identifier() {
    let a = restored(62_061, "a");
    let wire = format!(
        r#"{{"tuple":[{},{{"int":"1"}}]}}"#,
        LegacyIdentifier::wire(&a)
    );

    let value = serde_json::from_str::<ValueData>(&wire)
        .expect("reads")
        .build(&LegacyResolver)
        .expect("resolves");

    assert_eq!(value, Value::Tuple(vec![Value::Identifier(a), int(1)]));
}

#[test]
fn legacy_set_constraint_reads_its_identifier_members_and_writes_the_new_form() {
    let (x, a, b) = (
        restored(62_062, "x"),
        restored(62_064, "a"),
        restored(62_063, "b"),
    );
    let wire = format!(
        r#"{{"not_in_set":{{"variable":{},"values":[{},{}]}}}}"#,
        text_of(&x),
        LegacyIdentifier::wire(&a),
        LegacyIdentifier::wire(&b)
    );

    let constraint = serde_json::from_str::<ConstraintData>(&wire)
        .expect("reads")
        .build(&LegacyResolver)
        .expect("resolves");

    let expected = Constraint::from(SetConstraint::new(
        x,
        member_set([Value::Identifier(a), Value::Identifier(b)]),
        Polarity::NotIn,
    ));
    assert!(constraint.is_structurally_equivalent(&expected));
    assert_eq!(text_of(&constraint), text_of(&expected));
}

#[test]
fn legacy_opaque_identifier_is_refused_without_a_resolver() {
    let wire = LegacyIdentifier::wire(&restored(62_065, "a"));

    let error = serde_json::from_str::<Value>(&wire).expect_err("no resolver");

    assert!(error.to_string().contains("`id`"), "{error}");
}

#[test]
fn opaque_part_that_reports_no_identifier_stays_opaque() {
    let wire = text_of(&WireToken::value(7));

    let value = serde_json::from_str::<ValueData>(&wire)
        .expect("reads")
        .build(&LegacyResolver)
        .expect("resolves");

    assert!(matches!(value, Value::Opaque(_)), "{value:?}");
}

// ---------------------------------------------------------------------------
// Finite param domains over identifiers
// ---------------------------------------------------------------------------

#[test]
fn categorical_domain_over_identifiers_orders_them_by_id_and_admits_by_id() {
    let domain = CategoricalDomain::new(vec![
        identifier(1_000, "a"),
        identifier(100, "b"),
        identifier(99, "c"),
    ])
    .expect("identifiers are categories");

    let names: Vec<String> = domain.values().iter().map(ToString::to_string).collect();
    assert_eq!(names, ["c", "b", "a"]);
    assert!(domain.contains_value(&identifier(100, "renamed")));
    assert!(!domain.contains_value(&text("b")));
    let domain = ParamDomain::from(domain);
    assert!(
        domain
            .is_value_admissible(&identifier(99, "c"))
            .expect("decides")
    );
    assert!(!domain.is_value_admissible(&text("c")).expect("decides"));
}

#[test]
fn categorical_domain_refuses_one_identifier_twice() {
    let error = CategoricalDomain::new(vec![identifier(62_070, "a"), identifier(62_070, "b")])
        .expect_err("one id is one category");

    assert!(
        matches!(error, DomainError::DuplicateValues(_)),
        "{error:?}"
    );
}

#[test]
fn categorical_domain_over_identifiers_round_trips_and_reads_the_legacy_form() {
    let (a, b) = (restored(62_071, "a"), restored(62_072, "b"));
    let domain = ParamDomain::from(
        CategoricalDomain::new(vec![
            Value::Identifier(b.clone()),
            Value::Identifier(a.clone()),
        ])
        .expect("identifiers are categories"),
    );
    let wire = concat!(
        r#"{"categorical":{"categories":["#,
        r#"{"identifier":{"id":62071,"name_hint":"a"}},"#,
        r#"{"identifier":{"id":62072,"name_hint":"b"}}]}}"#
    );
    let legacy = format!(
        r#"{{"categorical":{{"categories":[{},{}]}}}}"#,
        LegacyIdentifier::wire(&a),
        LegacyIdentifier::wire(&b)
    );

    assert_eq!(text_of(&domain), wire);
    let decoded: ParamDomain = serde_json::from_str(wire).expect("decodes");
    assert_eq!(text_of(&decoded), wire);
    let from_legacy = serde_json::from_str::<ParamDomainData>(&legacy)
        .expect("reads")
        .build(&LegacyResolver)
        .expect("resolves");
    assert_eq!(text_of(&from_legacy), wire);
}

#[test]
fn permutation_domain_over_identifiers_keeps_their_order_and_admits_their_permutations() {
    let domain = PermutationDomain::new(vec![
        identifier(62_082, "n"),
        identifier(62_080, "c"),
        identifier(62_081, "h"),
    ])
    .expect("identifiers are permutation members");
    let wire = concat!(
        r#"{"permutation":{"ordered_members":["#,
        r#"{"identifier":{"id":62082,"name_hint":"n"}},"#,
        r#"{"identifier":{"id":62080,"name_hint":"c"}},"#,
        r#"{"identifier":{"id":62081,"name_hint":"h"}}]}}"#
    );

    assert!(domain.is_permutation(&Value::Tuple(vec![
        identifier(62_080, "c"),
        identifier(62_081, "h"),
        identifier(62_082, "n"),
    ])));
    assert!(!domain.is_permutation(&Value::Tuple(vec![
        text("c"),
        identifier(62_081, "h"),
        identifier(62_082, "n"),
    ])));
    assert!(!domain.is_permutation(&Value::Tuple(vec![
        identifier(62_080, "c"),
        identifier(62_080, "c"),
        identifier(62_082, "n"),
    ])));
    let domain = ParamDomain::from(domain);
    assert_eq!(text_of(&domain), wire);
    assert_eq!(
        text_of(&serde_json::from_str::<ParamDomain>(wire).expect("decodes")),
        wire
    );
}

#[rstest]
#[case::two_identifiers(vec![identifier(62_090, "a"), identifier(62_091, "b")])]
#[case::an_identifier_and_an_integer(vec![int(1), identifier(62_090, "a")])]
fn ordinal_domain_refuses_identifiers_as_incomparable(#[case] values: Vec<Value>) {
    let error = OrdinalDomain::new(values).expect_err("identifiers do not order");

    assert!(
        matches!(error, DomainError::IncomparableValues),
        "{error:?}"
    );
}

#[test]
fn param_over_identifier_categories_checks_its_constraint_by_id() {
    let x = restored(62_092, "x");
    let (a, b) = (restored(62_093, "a"), restored(62_094, "b"));
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let domain = ParamDomain::from(
        CategoricalDomain::new(vec![
            Value::Identifier(a.clone()),
            Value::Identifier(b.clone()),
        ])
        .expect("identifiers are categories"),
    );
    let param = Param::new(
        domain,
        x.clone(),
        [Constraint::from(in_set(&x, [Value::Identifier(a.clone())]))],
        &context,
    )
    .expect("a param");

    ParamAssignment::new(param.clone(), Value::Identifier(a), &context)
        .expect("a satisfies the constraint");
    let violated = ParamAssignment::new(param.clone(), Value::Identifier(b), &context)
        .expect_err("b violates the constraint");
    let inadmissible =
        ParamAssignment::new(param, text("a"), &context).expect_err("a string is no category");

    assert!(
        matches!(violated, AssignmentError::ViolatedConstraint { .. }),
        "{violated:?}"
    );
    assert!(
        matches!(inadmissible, AssignmentError::Inadmissible),
        "{inadmissible:?}"
    );
}
