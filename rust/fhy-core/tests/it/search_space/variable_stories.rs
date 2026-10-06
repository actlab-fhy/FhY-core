//! Tests for variables (`PlainVariable` and the relations of
//! `Part<dyn Variable>`): construction and accessors, `==` and `Hash`, and
//! alpha and structural equivalence on their own.
//!
//! Compared on their own, a variable's name is a binder, so two variables
//! with different names can be equivalent. A param's identifier members are
//! references that correspond only through the renaming; every other member
//! compares type-strictly by value. The param's constraints compare under
//! one more frame pairing the two params' variables.

use std::borrow::Cow;
use std::collections::HashMap;

use fhy_core::constraint::Value;
use fhy_core::diagnostic::Note;
use fhy_core::foreign::{ForeignPart, Part};
use fhy_core::identifier::Identifier;
use fhy_core::param::{
    CategoricalDomain, IntegerDomain, Param, ParamContext, ParamDomain, PermutationDomain, Sign,
    ZeroInclusion,
};
use fhy_core::search_space::{PlainVariable, Variable};
use fhy_core::solver::Solver;
use fhy_core::term::{AlphaEquivalence, AlphaRenaming};
use rstest::rstest;

use crate::support::constraint::{int, text};
use crate::support::hashing::hash_of;
use crate::support::param::{at_most, in_set};
use crate::support::search_space::{
    categorical, categorical_where, chosen, compare_alpha_both_ways, int_param, int_variable,
    plain_variable,
};

/// Return the alpha comparison of `left` with `right` under the free
/// renaming of `pairs` (each `(left, right)`), and of `right` with `left`
/// under the same pairs reversed.
fn compare_alpha_under_both_ways(
    left: &Part<dyn Variable>,
    right: &Part<dyn Variable>,
    pairs: &[(&Identifier, &Identifier)],
) -> [bool; 2] {
    let forward = AlphaRenaming::new(
        pairs
            .iter()
            .map(|&(from, to)| (from.clone(), to.clone()))
            .collect::<HashMap<_, _>>(),
    )
    .expect("the pairs are injective");
    let backward = AlphaRenaming::new(
        pairs
            .iter()
            .map(|&(from, to)| (to.clone(), from.clone()))
            .collect::<HashMap<_, _>>(),
    )
    .expect("the pairs are injective");
    [
        left.is_alpha_equivalent_under(right, &forward)
            .expect("compares"),
        right
            .is_alpha_equivalent_under(left, &backward)
            .expect("compares"),
    ]
}

/// Return the structural comparison of `left` with `right` and of `right`
/// with `left`.
fn compare_structural_both_ways(
    left: &Part<dyn Variable>,
    right: &Part<dyn Variable>,
) -> [bool; 2] {
    [
        left.is_structurally_equivalent(right).expect("compares"),
        right.is_structurally_equivalent(left).expect("compares"),
    ]
}

/// Return the param over the non-negative integers whose variable is
/// fresh, bounded above by `bound` on that variable.
fn build_bounded_param(bound: i64) -> Param {
    let variable = Identifier::new("p");
    let constraint = at_most(&variable, bound);
    let solver = Solver::new();
    Param::new(
        ParamDomain::from(IntegerDomain::new(
            Sign::NonNegative,
            ZeroInclusion::Included,
        )),
        variable,
        vec![constraint],
        &ParamContext::new(&solver),
    )
    .expect("the param is valid")
}

/// Return the param over the identifiers `members` as categories.
fn build_identifier_param(members: &[&Identifier]) -> Param {
    categorical(members.iter().map(|&member| chosen(member)).collect())
}

/// Return the param over the identifiers `members`, a permutation domain.
fn build_permutation_param(members: &[&Identifier]) -> Param {
    let solver = Solver::new();
    Param::new(
        ParamDomain::from(
            PermutationDomain::new(members.iter().map(|&member| chosen(member)).collect())
                .expect("a permutation domain"),
        ),
        Identifier::new("p"),
        Vec::new(),
        &ParamContext::new(&solver),
    )
    .expect("the param is valid")
}

/// Return the param over the categories `values`, its variable fresh.
fn build_category_param(values: Vec<Value>) -> Param {
    let solver = Solver::new();
    Param::new(
        ParamDomain::from(CategoricalDomain::new(values).expect("the categories are valid")),
        Identifier::new("p"),
        Vec::new(),
        &ParamContext::new(&solver),
    )
    .expect("the param is valid")
}

// ---------------------------------------------------------------------------
// Construction and accessors
// ---------------------------------------------------------------------------

#[test]
fn plain_variable_exposes_its_name_and_param_and_holds_no_notes_by_default() {
    let name = Identifier::new("x");
    let param = int_param(&[1, 2]);

    let variable = PlainVariable::new(name.clone(), param.clone());

    assert_eq!(variable.name(), &name);
    assert_eq!(variable.param(), &param);
    assert_eq!(variable.notes(), &[] as &[Note]);
}

#[test]
fn plain_variable_kind_is_the_search_space_variable_kind() {
    let variable = PlainVariable::new(Identifier::new("x"), int_param(&[1]));

    assert_eq!(PlainVariable::KIND, "search_space.variable");
    assert_eq!(variable.kind(), Cow::Borrowed("search_space.variable"));
}

#[test]
fn plain_variable_type_name_is_plain_variable() {
    let variable = PlainVariable::new(Identifier::new("x"), int_param(&[1]));

    assert_eq!(variable.type_name(), "PlainVariable");
}

#[test]
fn with_notes_attaches_the_notes_and_keeps_the_rest() {
    let name = Identifier::new("x");
    let param = int_param(&[1, 2]);
    let notes = vec![
        Note::with_other_kind("first"),
        Note::with_other_kind("second"),
    ];

    let variable = PlainVariable::new(name.clone(), param.clone()).with_notes(notes.clone());

    assert_eq!(variable.notes(), notes.as_slice());
    assert_eq!(variable.name(), &name);
    assert_eq!(variable.param(), &param);
}

#[test]
fn with_notes_replaces_earlier_notes() {
    let replacement = vec![Note::with_other_kind("second")];

    let variable = PlainVariable::new(Identifier::new("x"), int_param(&[1]))
        .with_notes(vec![Note::with_other_kind("first")])
        .with_notes(replacement.clone());

    assert_eq!(variable.notes(), replacement.as_slice());
}

// ---------------------------------------------------------------------------
// Equality and hashing
// ---------------------------------------------------------------------------

#[test]
fn plain_variables_with_equal_fields_are_equal_and_hash_alike() {
    let name = Identifier::new("x");
    let param = int_param(&[1, 2]);
    let notes = vec![Note::with_other_kind("note")];

    let left = PlainVariable::new(name.clone(), param.clone()).with_notes(notes.clone());
    let right = PlainVariable::new(name, param).with_notes(notes);

    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
}

#[test]
fn separately_built_equal_variables_are_equal_as_parts_and_hash_alike() {
    let name = Identifier::new("x");
    let param = int_param(&[1, 2]);

    let left = plain_variable(&name, param.clone());
    let right = plain_variable(&name, param);

    assert!(!Part::ptr_eq(&left, &right));
    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
}

#[test]
fn variables_with_different_names_are_unequal() {
    let param = int_param(&[1, 2]);

    let left = PlainVariable::new(Identifier::new("x"), param.clone());
    let right = PlainVariable::new(Identifier::new("y"), param);

    assert_ne!(left, right);
    assert_ne!(hash_of(&left), hash_of(&right));
}

#[test]
fn variables_with_different_params_are_unequal() {
    let name = Identifier::new("x");

    let left = PlainVariable::new(name.clone(), int_param(&[1, 2]));
    let right = PlainVariable::new(name, int_param(&[1, 3]));

    assert_ne!(left, right);
    assert_ne!(hash_of(&left), hash_of(&right));
}

#[test]
fn variables_with_different_notes_are_unequal() {
    let name = Identifier::new("x");
    let param = int_param(&[1, 2]);

    let left = PlainVariable::new(name.clone(), param.clone())
        .with_notes(vec![Note::with_other_kind("one")]);
    let right = PlainVariable::new(name, param).with_notes(vec![Note::with_other_kind("two")]);

    assert_ne!(left, right);
    assert_ne!(hash_of(&left), hash_of(&right));
}

#[test]
fn variable_parts_with_different_fields_are_unequal() {
    let name = Identifier::new("x");
    let param = int_param(&[1, 2]);

    let base = plain_variable(&name, param.clone());

    assert_ne!(base, plain_variable(&Identifier::new("y"), param.clone()));
    assert_ne!(base, plain_variable(&name, int_param(&[1, 3])));
    assert_ne!(
        base,
        Part::new(PlainVariable::new(name, param).with_notes(vec![Note::with_other_kind("note")]))
    );
}

// ---------------------------------------------------------------------------
// Alpha equivalence on their own
// ---------------------------------------------------------------------------

#[test]
fn variables_with_distinct_names_are_alpha_equivalent_standalone() {
    let left = int_variable(&Identifier::new("x"), &[1, 2]);
    let right = int_variable(&Identifier::new("y"), &[1, 2]);

    assert_eq!(compare_alpha_both_ways(&left, &right), [true, true]);
}

#[test]
fn a_variable_is_alpha_equivalent_to_itself() {
    let variable = int_variable(&Identifier::new("x"), &[1, 2]);

    assert_eq!(
        compare_alpha_both_ways(&variable, &variable.clone()),
        [true, true]
    );
}

#[test]
fn variables_over_different_domains_are_not_alpha_equivalent() {
    let name = Identifier::new("x");
    let left = int_variable(&name, &[1, 2]);

    assert_eq!(
        compare_alpha_both_ways(&left, &int_variable(&Identifier::new("y"), &[1, 3])),
        [false, false]
    );
    assert_eq!(
        compare_alpha_both_ways(&left, &int_variable(&Identifier::new("z"), &[1, 2, 3])),
        [false, false]
    );
}

#[test]
fn variables_with_different_notes_are_not_alpha_equivalent() {
    let param = int_param(&[1, 2]);
    let left = Part::new(
        PlainVariable::new(Identifier::new("x"), param.clone())
            .with_notes(vec![Note::with_other_kind("one")]),
    );
    let right = Part::new(
        PlainVariable::new(Identifier::new("y"), param)
            .with_notes(vec![Note::with_other_kind("two")]),
    );

    assert_eq!(compare_alpha_both_ways(&left, &right), [false, false]);
}

#[test]
fn variables_with_equal_notes_are_alpha_equivalent() {
    let notes = vec![Note::with_other_kind("same")];
    let left = Part::new(
        PlainVariable::new(Identifier::new("x"), int_param(&[1, 2])).with_notes(notes.clone()),
    );
    let right =
        Part::new(PlainVariable::new(Identifier::new("y"), int_param(&[1, 2])).with_notes(notes));

    assert_eq!(compare_alpha_both_ways(&left, &right), [true, true]);
}

#[test]
fn bounded_variables_whose_param_variables_are_renamed_are_alpha_equivalent() {
    let left = plain_variable(&Identifier::new("x"), build_bounded_param(10));
    let right = plain_variable(&Identifier::new("y"), build_bounded_param(10));

    assert_eq!(compare_alpha_both_ways(&left, &right), [true, true]);
}

#[test]
fn bounded_variables_whose_bounds_differ_are_not_alpha_equivalent() {
    let left = plain_variable(&Identifier::new("x"), build_bounded_param(10));
    let right = plain_variable(&Identifier::new("y"), build_bounded_param(11));

    assert_eq!(compare_alpha_both_ways(&left, &right), [false, false]);
}

#[test]
fn a_bounded_variable_is_not_alpha_equivalent_to_an_unbounded_one() {
    let solver = Solver::new();
    let unbounded = Param::new(
        ParamDomain::from(IntegerDomain::new(
            Sign::NonNegative,
            ZeroInclusion::Included,
        )),
        Identifier::new("p"),
        Vec::new(),
        &ParamContext::new(&solver),
    )
    .expect("the param is valid");
    let left = plain_variable(&Identifier::new("x"), build_bounded_param(10));
    let right = plain_variable(&Identifier::new("y"), unbounded);

    assert_eq!(compare_alpha_both_ways(&left, &right), [false, false]);
}

#[test]
fn categorical_variables_whose_set_constraints_differ_are_not_alpha_equivalent() {
    // F-SS-001: the constraint `p in {1}` narrows the second variable only.
    let plain = categorical(vec![int(1), int(2)]);
    let narrowed = categorical_where(vec![int(1), int(2)], |p| vec![in_set(p, [int(1)])]);
    let left = plain_variable(&Identifier::new("x"), plain);
    let right = plain_variable(&Identifier::new("y"), narrowed);

    assert_eq!(compare_alpha_both_ways(&left, &right), [false, false]);
}

#[rstest]
#[case::int_and_bool(int(1), Value::Bool(true))]
#[case::int_and_string(int(1), text("1"))]
#[case::bool_and_string(Value::Bool(true), text("1"))]
fn variables_over_values_of_different_types_are_not_alpha_equivalent(
    #[case] left: Value,
    #[case] right: Value,
) {
    // F-SS-004: members other than identifiers compare type-strictly.
    let left = plain_variable(&Identifier::new("x"), build_category_param(vec![left]));
    let right = plain_variable(&Identifier::new("y"), build_category_param(vec![right]));

    assert_eq!(compare_alpha_both_ways(&left, &right), [false, false]);
}

// ---------------------------------------------------------------------------
// Identifier members are references
// ---------------------------------------------------------------------------

#[test]
fn identifier_members_correspond_only_through_the_renaming() {
    let (a, b) = (Identifier::new("a"), Identifier::new("b"));
    let left = plain_variable(&Identifier::new("x"), build_identifier_param(&[&a]));
    let right = plain_variable(&Identifier::new("y"), build_identifier_param(&[&b]));

    assert_eq!(compare_alpha_both_ways(&left, &right), [false, false]);
    assert_eq!(
        compare_alpha_under_both_ways(&left, &right, &[(&a, &b)]),
        [true, true]
    );
}

#[test]
fn identifier_members_naming_the_same_free_identifier_are_equivalent() {
    let a = Identifier::new("a");
    let left = plain_variable(&Identifier::new("x"), build_identifier_param(&[&a]));
    let right = plain_variable(&Identifier::new("y"), build_identifier_param(&[&a]));

    assert_eq!(compare_alpha_both_ways(&left, &right), [true, true]);
}

#[test]
fn a_variable_named_like_a_free_member_does_not_capture_it() {
    // F-SS-003: `Z` is free in the first variable and bound by the name `Z`
    // in the second.
    let z = Identifier::new("Z");
    let left = plain_variable(&Identifier::new("k"), build_identifier_param(&[&z]));
    let right = plain_variable(&z, build_identifier_param(&[&z]));

    assert_eq!(compare_alpha_both_ways(&left, &right), [false, false]);
}

#[test]
fn a_variable_over_its_own_name_is_alpha_equivalent_to_one_over_its_own_renamed_name() {
    let (z, w) = (Identifier::new("Z"), Identifier::new("W"));
    let left = plain_variable(&z, build_identifier_param(&[&z]));
    let right = plain_variable(&w, build_identifier_param(&[&w]));

    assert_eq!(compare_alpha_both_ways(&left, &right), [true, true]);
}

#[test]
fn categorical_identifier_domains_compare_as_a_bijection_whatever_the_id_order() {
    let (a, b) = (Identifier::new("a"), Identifier::new("b"));
    // The second side's ids are created in the opposite order, so the
    // canonical order of its members differs from the first side's.
    let b_renamed = Identifier::new("b2");
    let a_renamed = Identifier::new("a2");
    let left = plain_variable(&Identifier::new("x"), build_identifier_param(&[&a, &b]));
    let right = plain_variable(
        &Identifier::new("y"),
        build_identifier_param(&[&a_renamed, &b_renamed]),
    );

    assert_eq!(
        compare_alpha_under_both_ways(&left, &right, &[(&a, &a_renamed), (&b, &b_renamed)]),
        [true, true]
    );
    assert_eq!(
        compare_alpha_under_both_ways(&left, &right, &[(&a, &a_renamed)]),
        [false, false]
    );
}

#[test]
fn permutation_identifier_domains_compare_in_order() {
    let (a, b) = (Identifier::new("a"), Identifier::new("b"));
    let b_renamed = Identifier::new("b2");
    let a_renamed = Identifier::new("a2");
    let pairs = [(&a, &a_renamed), (&b, &b_renamed)];
    let left = plain_variable(&Identifier::new("x"), build_permutation_param(&[&a, &b]));
    let in_order = plain_variable(
        &Identifier::new("y"),
        build_permutation_param(&[&a_renamed, &b_renamed]),
    );
    let swapped = plain_variable(
        &Identifier::new("z"),
        build_permutation_param(&[&b_renamed, &a_renamed]),
    );

    assert_eq!(
        compare_alpha_under_both_ways(&left, &in_order, &pairs),
        [true, true]
    );
    assert_eq!(
        compare_alpha_under_both_ways(&left, &swapped, &pairs),
        [false, false]
    );
}

#[test]
fn set_constraint_members_that_are_identifiers_resolve_through_the_renaming() {
    // D-SS-3.
    let (a, b) = (Identifier::new("a"), Identifier::new("b"));
    let (a_renamed, b_renamed) = (Identifier::new("a2"), Identifier::new("b2"));
    let pairs = [(&a, &a_renamed), (&b, &b_renamed)];
    let over = |first: &Identifier, second: &Identifier, member: &Identifier| {
        plain_variable(
            &Identifier::new("x"),
            categorical_where(vec![chosen(first), chosen(second)], |p| {
                vec![in_set(p, [chosen(member)])]
            }),
        )
    };
    let left = over(&a, &b, &a);

    let corresponding = over(&a_renamed, &b_renamed, &a_renamed);
    let crossed = over(&a_renamed, &b_renamed, &b_renamed);

    assert_eq!(
        compare_alpha_under_both_ways(&left, &corresponding, &pairs),
        [true, true]
    );
    assert_eq!(
        compare_alpha_under_both_ways(&left, &crossed, &pairs),
        [false, false]
    );
}

// ---------------------------------------------------------------------------
// Structural equivalence
// ---------------------------------------------------------------------------

#[test]
fn variables_of_one_kind_sharing_parts_are_structurally_equivalent() {
    let name = Identifier::new("x");
    let param = int_param(&[1, 2]);
    let notes = vec![Note::with_other_kind("note")];
    let left = Part::new(PlainVariable::new(name.clone(), param.clone()).with_notes(notes.clone()));
    let right = Part::new(PlainVariable::new(name, param).with_notes(notes));

    assert_eq!(compare_structural_both_ways(&left, &right), [true, true]);
}

#[test]
fn a_variable_is_structurally_equivalent_to_its_clone() {
    let variable = int_variable(&Identifier::new("x"), &[1, 2]);

    assert_eq!(
        compare_structural_both_ways(&variable, &variable.clone()),
        [true, true]
    );
}

#[test]
fn variables_with_different_names_are_not_structurally_equivalent() {
    let param = int_param(&[1, 2]);
    let left = plain_variable(&Identifier::new("x"), param.clone());
    let right = plain_variable(&Identifier::new("y"), param);

    assert_eq!(compare_structural_both_ways(&left, &right), [false, false]);
}

#[test]
fn variables_with_different_params_are_not_structurally_equivalent() {
    let name = Identifier::new("x");
    let left = plain_variable(&name, int_param(&[1, 2]));
    let right = plain_variable(&name, int_param(&[1, 3]));

    assert_eq!(compare_structural_both_ways(&left, &right), [false, false]);
}

#[test]
fn variables_whose_params_differ_only_in_their_variable_are_not_structurally_equivalent() {
    let name = Identifier::new("x");
    let left = plain_variable(&name, build_bounded_param(10));
    let right = plain_variable(&name, build_bounded_param(10));

    assert_eq!(compare_structural_both_ways(&left, &right), [false, false]);
}

#[test]
fn variables_with_different_notes_are_not_structurally_equivalent() {
    let name = Identifier::new("x");
    let param = int_param(&[1, 2]);
    let left = Part::new(
        PlainVariable::new(name.clone(), param.clone())
            .with_notes(vec![Note::with_other_kind("one")]),
    );
    let right =
        Part::new(PlainVariable::new(name, param).with_notes(vec![Note::with_other_kind("two")]));

    assert_eq!(compare_structural_both_ways(&left, &right), [false, false]);
}

#[test]
fn categorical_variables_whose_set_constraints_differ_are_not_structurally_equivalent() {
    // F-SS-001.
    let name = Identifier::new("x");
    let left = plain_variable(&name, categorical(vec![int(1), int(2)]));
    let right = plain_variable(
        &name,
        categorical_where(vec![int(1), int(2)], |p| vec![in_set(p, [int(1)])]),
    );

    assert_eq!(compare_structural_both_ways(&left, &right), [false, false]);
}

#[rstest]
#[case::int_and_bool(int(1), Value::Bool(true))]
#[case::int_and_string(int(1), text("1"))]
fn variables_over_values_of_different_types_are_not_structurally_equivalent(
    #[case] left: Value,
    #[case] right: Value,
) {
    // F-SS-004.
    let name = Identifier::new("x");
    let left = plain_variable(&name, build_category_param(vec![left]));
    let right = plain_variable(&name, build_category_param(vec![right]));

    assert_eq!(compare_structural_both_ways(&left, &right), [false, false]);
}

#[test]
fn variables_over_different_identifier_members_are_not_structurally_equivalent() {
    let (a, b) = (Identifier::new("a"), Identifier::new("b"));
    let name = Identifier::new("x");
    let left = plain_variable(&name, build_identifier_param(&[&a]));
    let right = plain_variable(&name, build_identifier_param(&[&b]));

    assert_eq!(compare_structural_both_ways(&left, &right), [false, false]);
}
