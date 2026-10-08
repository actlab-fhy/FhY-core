//! Tests for choices (`Choice`): construction and the order of its
//! refusals, accessors, `==` and `Hash`, and alpha and structural
//! equivalence on their own.
//!
//! Every name a choice holds is a binder and distinct: its own, its
//! alternatives', their bound identifiers, and the names of their variables
//! and sub-choices at every depth. A choice compares its alternatives
//! positionally, so their order is significant.

use std::borrow::Cow;

use fhy_core::diagnostic::Note;
use fhy_core::foreign::{BoxError, ForeignPart, Part};
use fhy_core::identifier::Identifier;
use fhy_core::search_space::{Alternative, Choice, MAX_CHOICE_DEPTH, SpaceError, Variable};
use rstest::rstest;

use crate::support::hashing::hash_of;
use crate::support::search_space::{
    bare_alternative, build_choice_chain, choice_of, compare_alpha_both_ways, int_variable,
    plain_alternative,
};

/// An alternative that binds the identifiers it is given, or fails to
/// report them, and holds only the variables it is given.
#[derive(Debug)]
struct Binder {
    name: Identifier,
    variables: Vec<Part<dyn Variable>>,
    bound: Result<Vec<Identifier>, &'static str>,
}

impl Binder {
    /// Return the part `name` that binds `bound`.
    fn binding(name: &Identifier, bound: &[&Identifier]) -> Part<dyn Alternative> {
        Part::new(Self {
            name: name.clone(),
            variables: Vec::new(),
            bound: Ok(bound.iter().map(|&axis| axis.clone()).collect()),
        })
    }

    /// Return the part `name` whose bound identifiers fail with `message`.
    fn failing(name: &Identifier, message: &'static str) -> Part<dyn Alternative> {
        Part::new(Self {
            name: name.clone(),
            variables: Vec::new(),
            bound: Err(message),
        })
    }
}

impl ForeignPart for Binder {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("Binder")
    }
}

impl Alternative for Binder {
    fn kind(&self) -> Cow<'_, str> {
        Cow::Borrowed("test.binder")
    }

    fn name(&self) -> &Identifier {
        &self.name
    }

    fn variables(&self) -> &[Part<dyn Variable>] {
        &self.variables
    }

    fn bound_identifiers(&self) -> Result<Vec<Identifier>, BoxError> {
        self.bound.clone().map_err(BoxError::from)
    }
}

/// Return the structural comparison of `left` with `right` and of `right`
/// with `left`.
fn compare_structural_both_ways(left: &Choice, right: &Choice) -> [bool; 2] {
    [
        left.is_structurally_equivalent(right).expect("compares"),
        right.is_structurally_equivalent(left).expect("compares"),
    ]
}

/// Return the name `result` refuses a duplicate with.
///
/// # Panics
///
/// Panics if `result` is not [`SpaceError::DuplicateName`].
fn find_duplicate_name(result: Result<Choice, SpaceError>) -> Identifier {
    let Err(SpaceError::DuplicateName { name }) = result else {
        panic!("expected DuplicateName, got {result:?}");
    };
    name
}

/// Return the alternative `name` holding the variable `variable` over
/// `{value}`.
fn build_holding_alternative(
    name: &Identifier,
    variable: &Identifier,
    value: i64,
) -> Part<dyn Alternative> {
    plain_alternative(name, vec![int_variable(variable, &[value])], Vec::new())
}

/// Return the choice of the given names: `choice` among `first` holding
/// `first_variable` over `{1}` and `second` holding `second_variable` over
/// `{2}`, in that order, or the reverse if `swapped`.
fn build_two_way_choice(names: [&str; 5], swapped: bool) -> Choice {
    let [choice, first, first_variable, second, second_variable] = names.map(Identifier::new);
    let mut alternatives = vec![
        build_holding_alternative(&first, &first_variable, 1),
        build_holding_alternative(&second, &second_variable, 2),
    ];
    if swapped {
        alternatives.reverse();
    }
    choice_of(&choice, alternatives)
}

/// Return a two-level choice from the given names: `top` among `outer`,
/// holding `inner` whose alternative `leaf` holds `leaf_variable`, and
/// `plain`.
fn build_hierarchy(names: [&str; 6]) -> Choice {
    let [top, outer, inner, leaf, leaf_variable, plain] = names.map(Identifier::new);
    choice_of(
        &top,
        vec![
            plain_alternative(
                &outer,
                Vec::new(),
                vec![choice_of(
                    &inner,
                    vec![build_holding_alternative(&leaf, &leaf_variable, 1)],
                )],
            ),
            bare_alternative(&plain),
        ],
    )
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[test]
fn choice_new_refuses_an_empty_list_of_alternatives() {
    let name = Identifier::new("c");

    let result = Choice::new(name.clone(), Vec::new());

    let Err(SpaceError::EmptyChoice { choice }) = result else {
        panic!("expected EmptyChoice, got {result:?}");
    };
    assert_eq!(choice, name);
}

#[test]
fn choice_new_refuses_an_alternative_whose_bound_identifiers_fail() {
    let (name, failing) = (Identifier::new("c"), Identifier::new("f"));

    let result = Choice::new(
        name,
        vec![
            bare_alternative(&Identifier::new("ok")),
            Binder::failing(&failing, "no axes"),
        ],
    );

    let Err(SpaceError::Hook {
        alternative,
        source,
    }) = result
    else {
        panic!("expected Hook, got {result:?}");
    };
    assert_eq!(alternative, failing);
    assert_eq!(source.to_string(), "no axes");
}

#[test]
fn choice_new_reports_a_failing_hook_before_a_duplicate_name() {
    let (name, failing) = (Identifier::new("c"), Identifier::new("f"));

    let result = Choice::new(
        name.clone(),
        vec![
            Binder::failing(&failing, "no axes"),
            bare_alternative(&name),
        ],
    );

    let Err(SpaceError::Hook { alternative, .. }) = result else {
        panic!("expected Hook, got {result:?}");
    };
    assert_eq!(alternative, failing);
}

#[test]
fn choice_new_refuses_an_alternative_named_like_the_choice() {
    let name = Identifier::new("c");

    let result = Choice::new(name.clone(), vec![bare_alternative(&name)]);

    assert_eq!(find_duplicate_name(result), name);
}

#[test]
fn choice_new_refuses_one_identifier_for_two_alternatives() {
    // The audit's A1 shape: choice(S, S).
    let shared = Identifier::new("S");

    let result = Choice::new(
        Identifier::new("c"),
        vec![bare_alternative(&shared), bare_alternative(&shared)],
    );

    assert_eq!(find_duplicate_name(result), shared);
}

#[test]
fn choice_new_refuses_a_variable_named_like_another_alternative() {
    let (first, shared) = (Identifier::new("a"), Identifier::new("s"));

    let result = Choice::new(
        Identifier::new("c"),
        vec![
            build_holding_alternative(&first, &shared, 1),
            bare_alternative(&shared),
        ],
    );

    assert_eq!(find_duplicate_name(result), shared);
}

#[test]
fn choice_new_refuses_one_variable_name_in_two_alternatives() {
    let shared = Identifier::new("x");

    let result = Choice::new(
        Identifier::new("c"),
        vec![
            build_holding_alternative(&Identifier::new("a"), &shared, 1),
            build_holding_alternative(&Identifier::new("b"), &shared, 2),
        ],
    );

    assert_eq!(find_duplicate_name(result), shared);
}

#[test]
fn choice_new_refuses_a_name_repeated_at_depth() {
    let shared = Identifier::new("deep");
    let nested = plain_alternative(
        &Identifier::new("a"),
        Vec::new(),
        vec![choice_of(
            &Identifier::new("inner"),
            vec![build_holding_alternative(
                &Identifier::new("leaf"),
                &shared,
                1,
            )],
        )],
    );

    let result = Choice::new(
        Identifier::new("c"),
        vec![
            nested,
            build_holding_alternative(&Identifier::new("b"), &shared, 2),
        ],
    );

    assert_eq!(find_duplicate_name(result), shared);
}

#[test]
fn choice_new_refuses_a_bound_identifier_named_like_a_variable() {
    let shared = Identifier::new("axis");
    let binding = Binder::binding(&Identifier::new("a"), &[&shared]);

    let result = Choice::new(
        Identifier::new("c"),
        vec![
            binding,
            build_holding_alternative(&Identifier::new("b"), &shared, 1),
        ],
    );

    assert_eq!(find_duplicate_name(result), shared);
}

#[test]
fn choice_new_refuses_a_bound_identifier_named_like_the_choice() {
    let name = Identifier::new("c");

    let result = Choice::new(
        name.clone(),
        vec![Binder::binding(&Identifier::new("a"), &[&name])],
    );

    assert_eq!(find_duplicate_name(result), name);
}

#[test]
fn choice_new_refuses_a_bound_identifier_shared_by_two_alternatives() {
    let shared = Identifier::new("axis");

    let result = Choice::new(
        Identifier::new("c"),
        vec![
            Binder::binding(&Identifier::new("a"), &[&shared]),
            Binder::binding(&Identifier::new("b"), &[&shared]),
        ],
    );

    assert_eq!(find_duplicate_name(result), shared);
}

#[test]
fn choice_new_names_the_first_repeat_in_canonical_order() {
    let (first, second) = (Identifier::new("p"), Identifier::new("q"));

    let result = Choice::new(
        Identifier::new("c"),
        vec![
            build_holding_alternative(&Identifier::new("a1"), &first, 1),
            build_holding_alternative(&Identifier::new("a2"), &first, 1),
            build_holding_alternative(&Identifier::new("a3"), &second, 1),
            build_holding_alternative(&Identifier::new("a4"), &second, 1),
        ],
    );

    assert_eq!(find_duplicate_name(result), first);
}

#[test]
fn choice_new_accepts_choices_nested_to_the_cap() {
    let choice = build_choice_chain(MAX_CHOICE_DEPTH, Vec::new());

    choice.expect("a chain as deep as the cap builds");
}

#[test]
fn choice_new_refuses_choices_nested_past_the_cap() {
    let inner = build_choice_chain(MAX_CHOICE_DEPTH, Vec::new()).expect("within the cap");
    let name = Identifier::new("top");

    let result = Choice::new(
        name.clone(),
        vec![plain_alternative(
            &Identifier::new("holder"),
            Vec::new(),
            vec![inner],
        )],
    );

    let Err(SpaceError::ChoiceTooDeep { choice }) = &result else {
        panic!("expected ChoiceTooDeep, got {result:?}");
    };
    assert_eq!(choice, &name);
}

#[test]
fn choice_new_reports_an_empty_choice_before_its_depth() {
    let result = Choice::new(Identifier::new("c"), Vec::new());

    assert!(
        matches!(result, Err(SpaceError::EmptyChoice { .. })),
        "{result:?}"
    );
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

#[test]
fn choice_exposes_its_name_alternatives_in_order_and_no_notes_by_default() {
    let name = Identifier::new("c");
    let alternatives = vec![
        bare_alternative(&Identifier::new("a")),
        build_holding_alternative(&Identifier::new("b"), &Identifier::new("x"), 1),
        bare_alternative(&Identifier::new("d")),
    ];

    let choice = Choice::new(name.clone(), alternatives.clone()).expect("the names are distinct");

    assert_eq!(choice.name(), &name);
    assert_eq!(choice.alternatives(), alternatives.as_slice());
    assert_eq!(choice.notes(), &[] as &[Note]);
}

#[test]
fn with_notes_attaches_the_notes_and_keeps_the_rest() {
    let name = Identifier::new("c");
    let alternatives = vec![bare_alternative(&Identifier::new("a"))];
    let notes = vec![
        Note::with_other_kind("first"),
        Note::with_other_kind("second"),
    ];

    let choice = Choice::new(name.clone(), alternatives.clone())
        .expect("the names are distinct")
        .with_notes(notes.clone());

    assert_eq!(choice.notes(), notes.as_slice());
    assert_eq!(choice.name(), &name);
    assert_eq!(choice.alternatives(), alternatives.as_slice());
}

#[test]
fn with_notes_replaces_earlier_notes() {
    let replacement = vec![Note::with_other_kind("second")];

    let choice = choice_of(
        &Identifier::new("c"),
        vec![bare_alternative(&Identifier::new("a"))],
    )
    .with_notes(vec![Note::with_other_kind("first")])
    .with_notes(replacement.clone());

    assert_eq!(choice.notes(), replacement.as_slice());
}

// ---------------------------------------------------------------------------
// Equality and hashing
// ---------------------------------------------------------------------------

#[test]
fn choices_with_equal_fields_are_equal_and_hash_alike() {
    let name = Identifier::new("c");
    let alternatives = vec![
        build_holding_alternative(&Identifier::new("a"), &Identifier::new("x"), 1),
        bare_alternative(&Identifier::new("b")),
    ];
    let notes = vec![Note::with_other_kind("note")];

    let left = choice_of(&name, alternatives.clone()).with_notes(notes.clone());
    let right = choice_of(&name, alternatives).with_notes(notes);

    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
}

#[test]
fn a_clone_of_a_choice_is_equal_hashes_alike_and_is_structurally_equivalent() {
    let choice = build_two_way_choice(["c", "a", "x", "b", "y"], false);

    let clone = choice.clone();

    assert_eq!(clone, choice);
    assert_eq!(hash_of(&clone), hash_of(&choice));
    assert_eq!(compare_structural_both_ways(&choice, &clone), [true, true]);
}

#[test]
fn choices_with_different_names_are_unequal() {
    let alternatives = vec![bare_alternative(&Identifier::new("a"))];

    let left = choice_of(&Identifier::new("c"), alternatives.clone());
    let right = choice_of(&Identifier::new("d"), alternatives);

    assert_ne!(left, right);
    assert_ne!(hash_of(&left), hash_of(&right));
}

#[test]
fn choices_with_different_alternatives_are_unequal() {
    let name = Identifier::new("c");

    let left = choice_of(&name, vec![bare_alternative(&Identifier::new("a"))]);
    let right = choice_of(&name, vec![bare_alternative(&Identifier::new("b"))]);

    assert_ne!(left, right);
    assert_ne!(hash_of(&left), hash_of(&right));
}

#[test]
fn choices_with_the_same_alternatives_in_another_order_are_unequal() {
    let name = Identifier::new("c");
    let (first, second) = (
        bare_alternative(&Identifier::new("a")),
        bare_alternative(&Identifier::new("b")),
    );

    let left = choice_of(&name, vec![first.clone(), second.clone()]);
    let right = choice_of(&name, vec![second, first]);

    assert_ne!(left, right);
    assert_ne!(hash_of(&left), hash_of(&right));
}

#[test]
fn choices_with_different_notes_are_unequal() {
    let name = Identifier::new("c");
    let alternatives = vec![bare_alternative(&Identifier::new("a"))];

    let left =
        choice_of(&name, alternatives.clone()).with_notes(vec![Note::with_other_kind("one")]);
    let right = choice_of(&name, alternatives).with_notes(vec![Note::with_other_kind("two")]);

    assert_ne!(left, right);
    assert_ne!(hash_of(&left), hash_of(&right));
}

// ---------------------------------------------------------------------------
// Alpha equivalence on their own
// ---------------------------------------------------------------------------

#[test]
fn choices_with_distinct_labels_are_alpha_equivalent_standalone() {
    let left = build_two_way_choice(["c", "a", "x", "b", "y"], false);
    let right = build_two_way_choice(["d", "p", "u", "q", "w"], false);

    assert_eq!(compare_alpha_both_ways(&left, &right), [true, true]);
}

#[test]
fn a_choice_is_alpha_equivalent_to_itself() {
    let choice = build_two_way_choice(["c", "a", "x", "b", "y"], false);

    assert_eq!(
        compare_alpha_both_ways(&choice, &choice.clone()),
        [true, true]
    );
}

#[test]
fn choices_with_different_variable_domains_are_not_alpha_equivalent() {
    let build = |second_domain: i64| {
        let [choice, first, first_variable, second, second_variable] =
            ["c", "a", "x", "b", "y"].map(Identifier::new);
        choice_of(
            &choice,
            vec![
                build_holding_alternative(&first, &first_variable, 1),
                build_holding_alternative(&second, &second_variable, second_domain),
            ],
        )
    };

    assert_eq!(
        compare_alpha_both_ways(&build(2), &build(3)),
        [false, false]
    );
    assert_eq!(compare_alpha_both_ways(&build(2), &build(2)), [true, true]);
}

#[test]
fn choices_with_different_notes_are_not_alpha_equivalent() {
    let part = |name: &str, note: &str| {
        choice_of(
            &Identifier::new(name),
            vec![bare_alternative(&Identifier::new("a"))],
        )
        .with_notes(vec![Note::with_other_kind(note)])
    };

    assert_eq!(
        compare_alpha_both_ways(&part("c", "one"), &part("d", "two")),
        [false, false]
    );
    assert_eq!(
        compare_alpha_both_ways(&part("c", "same"), &part("d", "same")),
        [true, true]
    );
}

#[test]
fn alternatives_order_is_significant_for_alpha_equivalence() {
    // Option order is significant: the options are not a set.
    let left = build_two_way_choice(["c", "a", "x", "b", "y"], false);
    let right = build_two_way_choice(["d", "p", "u", "q", "w"], true);

    assert_eq!(compare_alpha_both_ways(&left, &right), [false, false]);
}

#[test]
fn alternatives_binding_as_many_identifiers_in_another_place_are_not_alpha_equivalent() {
    // Both choices hold four names, so a frame pairs them, but the first
    // binds `x` in its first alternative and the second `y` in its second:
    // the bound identifiers do not line up, whatever the hooks answer.
    let [c, a1, x, a2, d, b1, b2, y] =
        ["c", "a1", "x", "a2", "d", "b1", "b2", "y"].map(Identifier::new);
    let left = choice_of(
        &c,
        vec![Binder::binding(&a1, &[&x]), Binder::binding(&a2, &[])],
    );
    let right = choice_of(
        &d,
        vec![Binder::binding(&b1, &[]), Binder::binding(&b2, &[&y])],
    );
    let relabeled = choice_of(
        &Identifier::new("e"),
        vec![
            Binder::binding(&Identifier::new("e1"), &[&Identifier::new("z")]),
            Binder::binding(&Identifier::new("e2"), &[]),
        ],
    );

    assert_eq!(compare_alpha_both_ways(&left, &right), [false, false]);
    assert_eq!(compare_alpha_both_ways(&left, &relabeled), [true, true]);
}

#[rstest]
#[case::prefix_first(true)]
#[case::prefix_second(false)]
fn choice_is_not_equivalent_to_a_prefix(#[case] prefix_first: bool) {
    let name = Identifier::new("c");
    let (first, second) = (
        bare_alternative(&Identifier::new("a")),
        bare_alternative(&Identifier::new("b")),
    );
    let prefix = choice_of(&name, vec![first.clone()]);
    let longer = choice_of(&name, vec![first, second]);
    let fresh_prefix = choice_of(
        &Identifier::new("d"),
        vec![bare_alternative(&Identifier::new("p"))],
    );
    let fresh_longer = choice_of(
        &Identifier::new("e"),
        vec![
            bare_alternative(&Identifier::new("q")),
            bare_alternative(&Identifier::new("r")),
        ],
    );
    let ((left, right), (fresh_left, fresh_right)) = if prefix_first {
        ((prefix, longer), (fresh_prefix, fresh_longer))
    } else {
        ((longer, prefix), (fresh_longer, fresh_prefix))
    };

    assert_eq!(compare_structural_both_ways(&left, &right), [false, false]);
    assert_eq!(
        compare_alpha_both_ways(&fresh_left, &fresh_right),
        [false, false]
    );
}

#[test]
fn a_relabeled_two_level_hierarchy_is_alpha_equivalent() {
    let left = build_hierarchy(["top", "outer", "inner", "leaf", "x", "plain"]);
    let right = build_hierarchy(["top2", "outer2", "inner2", "leaf2", "x2", "plain2"]);

    assert_eq!(compare_alpha_both_ways(&left, &right), [true, true]);
}

#[test]
fn a_hierarchy_that_differs_below_the_top_level_is_not_alpha_equivalent() {
    let left = build_hierarchy(["top", "outer", "inner", "leaf", "x", "plain"]);
    let [top, outer, inner, leaf, leaf_variable, plain] =
        ["top2", "outer2", "inner2", "leaf2", "x2", "plain2"].map(Identifier::new);
    let right = choice_of(
        &top,
        vec![
            plain_alternative(
                &outer,
                Vec::new(),
                vec![choice_of(
                    &inner,
                    vec![build_holding_alternative(&leaf, &leaf_variable, 2)],
                )],
            ),
            bare_alternative(&plain),
        ],
    );

    assert_eq!(compare_alpha_both_ways(&left, &right), [false, false]);
}

// ---------------------------------------------------------------------------
// Structural equivalence
// ---------------------------------------------------------------------------

#[test]
fn choices_sharing_their_alternatives_are_structurally_equivalent() {
    let name = Identifier::new("c");
    let alternatives = vec![
        build_holding_alternative(&Identifier::new("a"), &Identifier::new("x"), 1),
        bare_alternative(&Identifier::new("b")),
    ];
    let notes = vec![Note::with_other_kind("note")];

    let left = choice_of(&name, alternatives.clone()).with_notes(notes.clone());
    let right = choice_of(&name, alternatives).with_notes(notes);

    assert_eq!(compare_structural_both_ways(&left, &right), [true, true]);
}

#[test]
fn choices_with_different_names_are_not_structurally_equivalent() {
    let alternatives = vec![bare_alternative(&Identifier::new("a"))];

    let left = choice_of(&Identifier::new("c"), alternatives.clone());
    let right = choice_of(&Identifier::new("d"), alternatives);

    assert_eq!(compare_structural_both_ways(&left, &right), [false, false]);
}

#[test]
fn choices_with_different_notes_are_not_structurally_equivalent() {
    let name = Identifier::new("c");
    let alternatives = vec![bare_alternative(&Identifier::new("a"))];

    let left =
        choice_of(&name, alternatives.clone()).with_notes(vec![Note::with_other_kind("one")]);
    let right = choice_of(&name, alternatives).with_notes(vec![Note::with_other_kind("two")]);

    assert_eq!(compare_structural_both_ways(&left, &right), [false, false]);
}

#[test]
fn choices_with_different_alternatives_are_not_structurally_equivalent() {
    let name = Identifier::new("c");

    let left = choice_of(&name, vec![bare_alternative(&Identifier::new("a"))]);
    let right = choice_of(&name, vec![bare_alternative(&Identifier::new("b"))]);

    assert_eq!(compare_structural_both_ways(&left, &right), [false, false]);
}

#[test]
fn choices_with_the_same_alternatives_in_another_order_are_not_structurally_equivalent() {
    let name = Identifier::new("c");
    let (first, second) = (
        bare_alternative(&Identifier::new("a")),
        bare_alternative(&Identifier::new("b")),
    );

    let left = choice_of(&name, vec![first.clone(), second.clone()]);
    let right = choice_of(&name, vec![second, first]);

    assert_eq!(compare_structural_both_ways(&left, &right), [false, false]);
}

#[test]
fn a_hierarchy_is_structurally_equivalent_to_itself_and_its_clone_only() {
    let choice = build_hierarchy(["top", "outer", "inner", "leaf", "x", "plain"]);
    let relabeled = build_hierarchy(["top2", "outer2", "inner2", "leaf2", "x2", "plain2"]);

    assert_eq!(compare_structural_both_ways(&choice, &choice), [true, true]);
    assert_eq!(
        compare_structural_both_ways(&choice, &choice.clone()),
        [true, true]
    );
    assert_eq!(
        compare_structural_both_ways(&choice, &relabeled),
        [false, false]
    );
}
