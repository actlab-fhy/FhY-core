//! Stories of categorical domains whose categories are tuples or frozen
//! sets: what the constructor accepts and refuses, the canonical order, the
//! other finite domains that still refuse composite values, params and
//! assignments over such a domain, and its serde form.

use fhy_core::constraint::wire::MAX_VALUE_DEPTH;
use fhy_core::constraint::{Binding, Bindings, Value};
use fhy_core::expression::Decimal;
use fhy_core::identifier::Identifier;
use fhy_core::param::wire::ParamDomainData;
use fhy_core::param::{
    AssignmentError, CategoricalDomain, DomainError, DomainKind, OrdinalDomain, Param,
    ParamAssignment, ParamContext, ParamDomain, PermutationDomain, ValueCheck,
};
use fhy_core::solver::Solver;
use rstest::rstest;

use crate::support::constraint::{int, text};
use crate::support::foreign::TestResolver;
use crate::support::param::{boolean, describe_all, float, in_set};
use crate::support::serde::check_serde_round_trip;

/// Return the tuple of the integers `values`.
fn tuple_of(values: &[i64]) -> Value {
    Value::Tuple(values.iter().copied().map(int).collect())
}

/// Return the frozen set of the integers `values`.
fn frozen_set_of(values: &[i64]) -> Value {
    Value::FrozenSet(values.iter().copied().map(int).collect())
}

/// Return the categorical domain of `values`, failing the test if it is
/// refused.
///
/// # Panics
///
/// Panics if the constructor refuses `values`.
fn build_categorical(values: Vec<Value>) -> CategoricalDomain {
    CategoricalDomain::new(values).expect("a categorical domain")
}

// ---------------------------------------------------------------------------
// What the constructor accepts
// ---------------------------------------------------------------------------

/// Test a categorical domain holds categories that are tuples, nested
/// tuples, frozen sets, or a mix of those and leaves, each of them found.
#[rstest]
#[case::tuples_of_integers(vec![tuple_of(&[4, 4]), tuple_of(&[8, 8])])]
#[case::single_tuple(vec![tuple_of(&[1])])]
#[case::empty_tuple(vec![Value::Tuple(Vec::new()), tuple_of(&[1])])]
#[case::nested_tuples(vec![
    Value::Tuple(vec![tuple_of(&[1, 2]), int(3)]),
    Value::Tuple(vec![Value::Tuple(vec![tuple_of(&[4])])]),
])]
#[case::frozen_sets(vec![frozen_set_of(&[1, 2]), frozen_set_of(&[3])])]
#[case::empty_frozen_set(vec![Value::FrozenSet(Vec::new()), frozen_set_of(&[1])])]
#[case::tuple_of_a_frozen_set(vec![Value::Tuple(vec![frozen_set_of(&[1, 2]), text("a")])])]
#[case::leaves_and_tuples(vec![int(1), text("a"), boolean(true), tuple_of(&[8, 8])])]
#[case::tuple_of_mixed_leaves(vec![Value::Tuple(vec![int(1), text("a"), boolean(false)])])]
fn categorical_domain_accepts_tuple_and_frozen_set_categories(#[case] values: Vec<Value>) {
    let domain = CategoricalDomain::new(values.clone()).expect("composite categories");

    assert_eq!(domain.values().len(), values.len());
    for value in &values {
        assert!(domain.contains_value(value), "{value:?}");
    }
}

/// Return `depth` one-element tuples nested around the integer `leaf`.
fn nest_in_tuples(leaf: i64, depth: usize) -> Value {
    (0..depth).fold(int(leaf), |value, _| Value::Tuple(vec![value]))
}

/// Test a category nested as deep as a value may be on the wire is
/// accepted and found.
#[test]
fn categorical_domain_accepts_a_category_nested_to_the_value_depth_cap() {
    let deep = nest_in_tuples(1, MAX_VALUE_DEPTH);

    let domain = build_categorical(vec![int(0), deep.clone()]);

    assert!(domain.contains_value(&deep));
    assert!(!domain.contains_value(&nest_in_tuples(1, MAX_VALUE_DEPTH - 1)));
}

/// Test a category nested deeper than a value may be on the wire is
/// refused as no leaf value, with the index of the top-level value.
#[test]
fn categorical_domain_refuses_a_category_nested_past_the_value_depth_cap() {
    let too_deep = nest_in_tuples(1, MAX_VALUE_DEPTH + 1);

    let error = CategoricalDomain::new(vec![int(0), too_deep]).expect_err("too deep");

    assert!(
        matches!(
            error,
            DomainError::NotALeafValue {
                kind: DomainKind::Categorical,
                index: 1
            }
        ),
        "{error:?}"
    );
}

/// Test a tuple category is found, and a tuple that is not a category, or
/// a leaf, is not.
#[test]
fn categorical_domain_finds_only_the_tuples_it_holds() {
    let domain = build_categorical(vec![tuple_of(&[4, 4]), tuple_of(&[8, 8]), text("a")]);

    assert!(domain.contains_value(&tuple_of(&[8, 8])));
    assert!(!domain.contains_value(&tuple_of(&[3, 3])));
    assert!(!domain.contains_value(&tuple_of(&[8])));
    assert!(!domain.contains_value(&tuple_of(&[8, 8, 8])));
    assert!(!domain.contains_value(&tuple_of(&[8, 4])));
    assert!(!domain.contains_value(&int(8)));
}

/// Test a frozen-set category is found whatever the order of its elements.
#[test]
fn categorical_domain_finds_a_frozen_set_in_any_element_order() {
    let domain = build_categorical(vec![frozen_set_of(&[1, 2, 3]), frozen_set_of(&[4])]);

    assert!(domain.contains_value(&frozen_set_of(&[3, 1, 2])));
    assert!(!domain.contains_value(&frozen_set_of(&[1, 2])));
    assert!(!domain.contains_value(&tuple_of(&[1, 2, 3])));
}

/// Test a tuple is not a frozen set of the same elements: the two are
/// distinct categories.
#[test]
fn categorical_domain_tells_a_tuple_from_a_frozen_set_of_its_elements() {
    let domain = build_categorical(vec![tuple_of(&[1, 2]), frozen_set_of(&[1, 2])]);

    assert_eq!(domain.values().len(), 2);
    assert!(domain.contains_value(&tuple_of(&[1, 2])));
    assert!(domain.contains_value(&frozen_set_of(&[2, 1])));
}

/// Test two domains built from the same categories in any input order hold
/// the same categories in the same canonical order.
#[rstest]
#[case::as_given([0, 1, 2, 3])]
#[case::reversed([3, 2, 1, 0])]
#[case::rotated([2, 3, 0, 1])]
#[case::shuffled([1, 3, 0, 2])]
fn categorical_domain_orders_composite_categories_canonically(#[case] order: [usize; 4]) {
    let categories = [
        tuple_of(&[4, 4]),
        tuple_of(&[8, 8]),
        Value::Tuple(vec![int(1), text("a")]),
        int(7),
    ];
    let reference = build_categorical(categories.to_vec());

    let shuffled = build_categorical(
        order
            .iter()
            .map(|&index| categories[index].clone())
            .collect(),
    );

    assert_eq!(
        describe_all(shuffled.values()),
        describe_all(reference.values())
    );
    assert_eq!(shuffled.values().len(), 4);
}

/// Test the domains of the same categories are structurally equivalent as
/// params' domains whatever their input order.
#[test]
fn categorical_domains_of_the_same_tuples_are_structurally_equivalent() {
    let forward = ParamDomain::from(build_categorical(vec![
        tuple_of(&[4, 4]),
        tuple_of(&[8, 8]),
    ]));
    let backward = ParamDomain::from(build_categorical(vec![
        tuple_of(&[8, 8]),
        tuple_of(&[4, 4]),
    ]));
    let other = ParamDomain::from(build_categorical(vec![tuple_of(&[4, 4])]));

    assert!(forward.is_structurally_equivalent(&backward));
    assert!(!forward.is_structurally_equivalent(&other));
}

// ---------------------------------------------------------------------------
// What the constructor refuses
// ---------------------------------------------------------------------------

/// Test a float or a decimal inside a tuple or frozen set, at any depth, is
/// refused with the index of the top-level value.
#[rstest]
#[case::float_in_a_tuple(
    vec![tuple_of(&[1, 2]), Value::Tuple(vec![int(4), float(4.5)])],
    1
)]
#[case::first_value(vec![Value::Tuple(vec![int(4), float(4.5)])], 0)]
#[case::decimal_in_a_tuple(
    vec![tuple_of(&[1, 2]), Value::Tuple(vec![int(4), Value::Decimal("4.5".parse::<Decimal>().expect("a decimal"))])],
    1
)]
#[case::float_in_a_frozen_set(vec![int(1), tuple_of(&[2]), Value::FrozenSet(vec![float(0.5)])], 2)]
#[case::float_deep_inside(
    vec![
        tuple_of(&[1]),
        tuple_of(&[2]),
        Value::Tuple(vec![Value::Tuple(vec![Value::Tuple(vec![float(1.0)])])]),
    ],
    2
)]
#[case::float_in_a_tuple_in_a_frozen_set(
    vec![frozen_set_of(&[1]), Value::FrozenSet(vec![Value::Tuple(vec![float(2.0)])])],
    1
)]
#[case::decimal_next_to_a_valid_element(
    vec![tuple_of(&[1, 2]), Value::Tuple(vec![int(1), Value::Decimal("2.5".parse::<Decimal>().expect("a decimal")), int(3)])],
    1
)]
fn categorical_domain_refuses_a_non_leaf_inside_a_tuple(
    #[case] values: Vec<Value>,
    #[case] index: usize,
) {
    let error = CategoricalDomain::new(values).expect_err("a float or a decimal inside");

    assert!(
        matches!(
            error,
            DomainError::NotALeafValue {
                kind: DomainKind::Categorical,
                index: refused
            } if refused == index
        ),
        "{error:?}"
    );
}

/// Test a categorical domain still refuses no value and equal values.
#[rstest]
#[case::no_value(Vec::new(), true)]
#[case::equal_tuples(vec![tuple_of(&[4, 4]), tuple_of(&[4, 4])], false)]
#[case::equal_frozen_sets_in_another_order(vec![frozen_set_of(&[1, 2]), frozen_set_of(&[2, 1])], false)]
#[case::equal_after_a_leaf(vec![int(1), tuple_of(&[1]), tuple_of(&[1])], false)]
fn categorical_domain_refuses_empty_and_duplicate_categories(
    #[case] values: Vec<Value>,
    #[case] is_empty: bool,
) {
    let error = CategoricalDomain::new(values).expect_err("empty or duplicate");

    if is_empty {
        assert!(
            matches!(error, DomainError::EmptyValues(DomainKind::Categorical)),
            "{error:?}"
        );
    } else {
        assert!(
            matches!(error, DomainError::DuplicateValues(DomainKind::Categorical)),
            "{error:?}"
        );
    }
}

/// Test an ordinal domain still refuses a tuple or a frozen set, as a
/// not-a-leaf value with its index.
#[rstest]
#[case::tuple(tuple_of(&[4, 4]))]
#[case::frozen_set(frozen_set_of(&[4]))]
fn ordinal_domain_still_refuses_a_composite_value(#[case] value: Value) {
    let error = OrdinalDomain::new(vec![int(1), value]).expect_err("a composite value");

    assert!(
        matches!(
            error,
            DomainError::NotALeafValue {
                kind: DomainKind::Ordinal,
                index: 1
            }
        ),
        "{error:?}"
    );
}

/// Test a permutation domain still refuses a tuple or a frozen set, as a
/// not-a-leaf value with its index.
#[rstest]
#[case::tuple(tuple_of(&[4, 4]))]
#[case::frozen_set(frozen_set_of(&[4]))]
fn permutation_domain_still_refuses_a_composite_value(#[case] value: Value) {
    let error = PermutationDomain::new(vec![int(1), value]).expect_err("a composite value");

    assert!(
        matches!(
            error,
            DomainError::NotALeafValue {
                kind: DomainKind::Permutation,
                index: 1
            }
        ),
        "{error:?}"
    );
}

// ---------------------------------------------------------------------------
// Params and assignments over a tuple domain
// ---------------------------------------------------------------------------

/// Return the param of a fresh variable over the tuple categories `(4, 4)`,
/// `(8, 8)` and `(16, 16)`, with `constraints` over the variable.
fn build_tile_param(
    constraints: impl FnOnce(&Identifier) -> Vec<fhy_core::constraint::Constraint>,
    context: &ParamContext<'_>,
) -> Param {
    let tile = Identifier::new("tile");
    let domain = ParamDomain::from(build_categorical(vec![
        tuple_of(&[4, 4]),
        tuple_of(&[8, 8]),
        tuple_of(&[16, 16]),
    ]));
    let constraints = constraints(&tile);
    Param::new(domain, tile, constraints, context).expect("a param over tuple categories")
}

/// Test an assignment of a tuple category to a param over tuple categories
/// is accepted and equal to another of the same tuple, and one outside the
/// categories is refused as inadmissible.
#[test]
fn param_assignment_accepts_a_tuple_category_and_refuses_another_tuple() {
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let param = build_tile_param(|_| Vec::new(), &context);

    let accepted = ParamAssignment::new(param.clone(), tuple_of(&[8, 8]), &context)
        .expect("(8, 8) is a category");
    let again = ParamAssignment::new(param.clone(), tuple_of(&[8, 8]), &context)
        .expect("(8, 8) is a category");
    let other = ParamAssignment::new(param.clone(), tuple_of(&[4, 4]), &context)
        .expect("(4, 4) is a category");
    let refused = ParamAssignment::new(param, tuple_of(&[3, 3]), &context);

    assert!(accepted.is_structurally_equivalent(&again));
    assert!(!accepted.is_structurally_equivalent(&other));
    assert!(
        matches!(refused, Err(AssignmentError::Inadmissible)),
        "{refused:?}"
    );
}

/// Test a value check over a tuple domain finds a category valid and any
/// other value, a leaf included, inadmissible.
#[rstest]
#[case::category(tuple_of(&[16, 16]), ValueCheck::Valid)]
#[case::other_tuple(tuple_of(&[16, 8]), ValueCheck::Inadmissible)]
#[case::leaf(int(16), ValueCheck::Inadmissible)]
#[case::frozen_set(frozen_set_of(&[16]), ValueCheck::Inadmissible)]
fn param_check_value_decides_tuple_categories(#[case] value: Value, #[case] expected: ValueCheck) {
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let param = build_tile_param(|_| Vec::new(), &context);
    let environment = param
        .environment(Binding::Value(value), &Bindings::new())
        .expect("an environment");

    let check = param.check_value(&environment, &context);

    assert!(
        matches!(&check, Ok(check) if *check == expected),
        "{check:?}"
    );
}

/// Test an in-set constraint over tuple members is decided over a tuple
/// domain: a listed category is valid, an unlisted one is violated, and a
/// value outside the domain is inadmissible.
#[test]
fn param_set_constraint_over_a_tuple_category_is_decided() {
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let param = build_tile_param(
        |tile| vec![in_set(tile, [tuple_of(&[4, 4]), tuple_of(&[8, 8])])],
        &context,
    );
    let check = |value: Value| {
        let environment = param
            .environment(Binding::Value(value), &Bindings::new())
            .expect("an environment");
        param.check_value(&environment, &context).expect("decides")
    };

    assert_eq!(check(tuple_of(&[8, 8])), ValueCheck::Valid);
    assert_eq!(
        check(tuple_of(&[16, 16])),
        ValueCheck::Violated { member: 0 }
    );
    assert_eq!(check(tuple_of(&[2, 2])), ValueCheck::Inadmissible);
}

// ---------------------------------------------------------------------------
// Serde
// ---------------------------------------------------------------------------

/// Test a domain with tuple and frozen-set categories round-trips through
/// JSON and postcard.
#[rstest]
#[case::tuples(vec![tuple_of(&[4, 4]), tuple_of(&[8, 8])])]
#[case::nested(vec![Value::Tuple(vec![tuple_of(&[1, 2]), text("a")]), tuple_of(&[3])])]
#[case::frozen_sets(vec![frozen_set_of(&[1, 2]), frozen_set_of(&[3])])]
#[case::mixed(vec![int(1), text("a"), tuple_of(&[8, 8]), frozen_set_of(&[5, 6])])]
fn categorical_domain_with_composite_categories_round_trips_through_serde(
    #[case] values: Vec<Value>,
) {
    let domain = ParamDomain::from(build_categorical(values));

    check_serde_round_trip(&domain).expect("the domain round trips");
}

/// Test the payload of a tuple categorical decodes, whatever the order of
/// its categories, and writes back in the one canonical order.
#[test]
fn categorical_domain_decodes_tuple_categories_in_any_order() {
    let forward = r#"{"categorical":{"categories":[{"tuple":[{"int":"4"},{"int":"4"}]},{"tuple":[{"int":"8"},{"int":"8"}]}]}}"#;
    let backward = r#"{"categorical":{"categories":[{"tuple":[{"int":"8"},{"int":"8"}]},{"tuple":[{"int":"4"},{"int":"4"}]}]}}"#;

    let from_forward: ParamDomain = serde_json::from_str(forward).expect("decodes");
    let from_backward: ParamDomain = serde_json::from_str(backward).expect("decodes");

    assert!(from_forward.is_structurally_equivalent(&from_backward));
    assert_eq!(
        serde_json::to_string(&from_forward).expect("encodes"),
        serde_json::to_string(&from_backward).expect("encodes")
    );
    let expected = ParamDomain::from(build_categorical(vec![
        tuple_of(&[8, 8]),
        tuple_of(&[4, 4]),
    ]));
    assert!(from_forward.is_structurally_equivalent(&expected));
}

/// Test the wire data of a domain with a tuple category builds with a
/// resolver and keeps refusing a float inside the tuple.
#[rstest]
#[case::float_in_a_tuple(
    r#"{"categorical":{"categories":[{"int":"1"},{"tuple":[{"int":"4"},{"float":"4.5"}]}]}}"#
)]
#[case::duplicate_tuples(
    r#"{"categorical":{"categories":[{"tuple":[{"int":"4"}]},{"tuple":[{"int":"4"}]}]}}"#
)]
fn categorical_domain_data_refuses_what_the_constructor_refuses(#[case] payload: &str) {
    let data: ParamDomainData = serde_json::from_str(payload).expect("the shape reads");

    let built = data.build(&TestResolver);

    assert!(
        matches!(&built, Err(fhy_core::foreign::BuildError::Invalid(_))),
        "{built:?}"
    );
    serde_json::from_str::<ParamDomain>(payload).expect_err("the plain decoding refuses it too");
}
