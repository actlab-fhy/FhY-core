//! Tests for alternatives (`PlainAlternative` and the relations of
//! `Part<dyn Alternative>`): construction with its duplicate-name check,
//! accessors, `==` and `Hash`, and alpha and structural equivalence on
//! their own.
//!
//! Compared on their own, every name an alternative holds is a binder,
//! paired in canonical order: the alternative's name, its bound
//! identifiers, then its variables' names and its sub-choices' names,
//! depth first. So a reference inside the alternative to one of those names
//! corresponds to the reference to the paired name.

use std::borrow::Cow;
use std::collections::HashMap;

use fhy_core::constraint::Value;
use fhy_core::diagnostic::Note;
use fhy_core::foreign::{ForeignPart, Part};
use fhy_core::identifier::Identifier;
use fhy_core::search_space::{Alternative, Choice, PlainAlternative, SpaceError, Variable};
use fhy_core::term::{AlphaEquivalence, AlphaRenaming};
use rstest::rstest;

use crate::support::hashing::hash_of;
use crate::support::search_space::{
    bare_alternative, categorical, choice_of, compare_alpha_both_ways, int_param, int_variable,
    plain_alternative, plain_variable,
};

/// Return the structural comparison of `left` with `right` and of `right`
/// with `left`.
fn compare_structural_both_ways(
    left: &Part<dyn Alternative>,
    right: &Part<dyn Alternative>,
) -> [bool; 2] {
    [
        left.is_structurally_equivalent(right).expect("compares"),
        right.is_structurally_equivalent(left).expect("compares"),
    ]
}

/// Return the variable `name` over the identifiers `members` as
/// categories.
fn build_variable_over(name: &Identifier, members: &[&Identifier]) -> Part<dyn Variable> {
    plain_variable(
        name,
        categorical(
            members
                .iter()
                .map(|&member| Value::Identifier(member.clone()))
                .collect(),
        ),
    )
}

/// Return the name of the duplicate `result` refuses with.
///
/// # Panics
///
/// Panics if `result` is not [`SpaceError::DuplicateName`].
fn find_duplicate_name(result: Result<PlainAlternative, SpaceError>) -> Identifier {
    let Err(SpaceError::DuplicateName { name }) = result else {
        panic!("expected DuplicateName, got {result:?}");
    };
    name
}

/// Return the choice `name` among one alternative `alternative`, holding
/// the variable `variable` over `{1}`.
fn build_sub_choice(name: &Identifier, alternative: &Identifier, variable: &Identifier) -> Choice {
    choice_of(
        name,
        vec![plain_alternative(
            alternative,
            vec![int_variable(variable, &[1])],
            Vec::new(),
        )],
    )
}

// ---------------------------------------------------------------------------
// Construction and accessors
// ---------------------------------------------------------------------------

#[test]
fn plain_alternative_exposes_its_name_variables_and_choices() {
    let name = Identifier::new("o");
    let variables = vec![
        int_variable(&Identifier::new("x"), &[1, 2]),
        int_variable(&Identifier::new("y"), &[3]),
    ];
    let choices = vec![build_sub_choice(
        &Identifier::new("c"),
        &Identifier::new("s"),
        &Identifier::new("u"),
    )];

    let alternative = PlainAlternative::new(name.clone(), variables.clone(), choices.clone())
        .expect("the names are distinct");

    assert_eq!(alternative.name(), &name);
    assert_eq!(alternative.variables(), variables.as_slice());
    assert_eq!(alternative.choices(), choices.as_slice());
}

#[test]
fn plain_alternative_holds_no_notes_and_binds_no_identifier_by_default() {
    let alternative = PlainAlternative::new(Identifier::new("o"), Vec::new(), Vec::new())
        .expect("the names are distinct");

    assert_eq!(alternative.notes(), &[] as &[Note]);
    assert_eq!(
        alternative.bound_identifiers().expect("binds"),
        Vec::<Identifier>::new()
    );
}

#[test]
fn plain_alternative_kind_is_the_search_space_alternative_kind() {
    let alternative = PlainAlternative::new(Identifier::new("o"), Vec::new(), Vec::new())
        .expect("the names are distinct");

    assert_eq!(PlainAlternative::KIND, "search_space.alternative");
    assert_eq!(
        alternative.kind(),
        Cow::Borrowed("search_space.alternative")
    );
}

#[test]
fn plain_alternative_type_name_is_plain_alternative() {
    let alternative = PlainAlternative::new(Identifier::new("o"), Vec::new(), Vec::new())
        .expect("the names are distinct");

    assert_eq!(alternative.type_name(), "PlainAlternative");
}

#[test]
fn with_notes_attaches_the_notes_and_keeps_the_rest() {
    let name = Identifier::new("o");
    let variables = vec![int_variable(&Identifier::new("x"), &[1])];
    let notes = vec![
        Note::with_other_kind("first"),
        Note::with_other_kind("second"),
    ];

    let alternative = PlainAlternative::new(name.clone(), variables.clone(), Vec::new())
        .expect("the names are distinct")
        .with_notes(notes.clone());

    assert_eq!(alternative.notes(), notes.as_slice());
    assert_eq!(alternative.name(), &name);
    assert_eq!(alternative.variables(), variables.as_slice());
}

#[test]
fn with_notes_replaces_earlier_notes() {
    let replacement = vec![Note::with_other_kind("second")];

    let alternative = PlainAlternative::new(Identifier::new("o"), Vec::new(), Vec::new())
        .expect("the names are distinct")
        .with_notes(vec![Note::with_other_kind("first")])
        .with_notes(replacement.clone());

    assert_eq!(alternative.notes(), replacement.as_slice());
}

// ---------------------------------------------------------------------------
// Duplicate names
// ---------------------------------------------------------------------------

#[test]
fn plain_alternative_new_refuses_a_variable_named_like_the_alternative() {
    let name = Identifier::new("o");

    let result = PlainAlternative::new(name.clone(), vec![int_variable(&name, &[1])], Vec::new());

    assert_eq!(find_duplicate_name(result), name);
}

#[test]
fn plain_alternative_new_refuses_two_variables_with_one_name() {
    let shared = Identifier::new("x");

    let result = PlainAlternative::new(
        Identifier::new("o"),
        vec![
            int_variable(&shared, &[1]),
            int_variable(&Identifier::new("y"), &[2]),
            int_variable(&shared, &[3]),
        ],
        Vec::new(),
    );

    assert_eq!(find_duplicate_name(result), shared);
}

#[test]
fn plain_alternative_new_refuses_a_variable_named_like_a_name_in_a_sub_choice() {
    let shared = Identifier::new("v");

    let result = PlainAlternative::new(
        Identifier::new("o"),
        vec![int_variable(&shared, &[1])],
        vec![build_sub_choice(
            &Identifier::new("c"),
            &Identifier::new("s"),
            &shared,
        )],
    );

    assert_eq!(find_duplicate_name(result), shared);
}

#[test]
fn plain_alternative_new_refuses_a_sub_choice_alternative_named_like_a_variable() {
    let shared = Identifier::new("v");

    let result = PlainAlternative::new(
        Identifier::new("o"),
        vec![int_variable(&shared, &[1])],
        vec![choice_of(
            &Identifier::new("c"),
            vec![bare_alternative(&shared)],
        )],
    );

    assert_eq!(find_duplicate_name(result), shared);
}

#[test]
fn plain_alternative_new_refuses_a_sub_choice_named_like_the_alternative() {
    let name = Identifier::new("o");

    let result = PlainAlternative::new(
        name.clone(),
        Vec::new(),
        vec![choice_of(
            &name,
            vec![bare_alternative(&Identifier::new("s"))],
        )],
    );

    assert_eq!(find_duplicate_name(result), name);
}

#[test]
fn plain_alternative_new_refuses_two_sub_choices_sharing_a_name() {
    let shared = Identifier::new("c");

    let result = PlainAlternative::new(
        Identifier::new("o"),
        Vec::new(),
        vec![
            choice_of(&shared, vec![bare_alternative(&Identifier::new("s"))]),
            choice_of(&shared, vec![bare_alternative(&Identifier::new("t"))]),
        ],
    );

    assert_eq!(find_duplicate_name(result), shared);
}

#[test]
fn plain_alternative_new_names_the_first_repeat_in_canonical_order() {
    let (first, second) = (Identifier::new("a"), Identifier::new("b"));

    let result = PlainAlternative::new(
        Identifier::new("o"),
        vec![
            int_variable(&first, &[1]),
            int_variable(&second, &[1]),
            int_variable(&first, &[1]),
            int_variable(&second, &[1]),
        ],
        Vec::new(),
    );

    assert_eq!(find_duplicate_name(result), first);
}

// ---------------------------------------------------------------------------
// Equality and hashing
// ---------------------------------------------------------------------------

/// Return the alternative `o` holding `x` over `{1}` and a sub-choice, all
/// from the given identifiers, with the shared `variable` part.
fn build_alternative(
    name: &Identifier,
    variable: &Part<dyn Variable>,
    choices: Vec<Choice>,
) -> PlainAlternative {
    PlainAlternative::new(name.clone(), vec![variable.clone()], choices)
        .expect("the names are distinct")
}

#[test]
fn plain_alternatives_with_equal_fields_are_equal_and_hash_alike() {
    let name = Identifier::new("o");
    let variable = int_variable(&Identifier::new("x"), &[1]);
    let choice = build_sub_choice(
        &Identifier::new("c"),
        &Identifier::new("s"),
        &Identifier::new("u"),
    );
    let notes = vec![Note::with_other_kind("note")];

    let left = build_alternative(&name, &variable, vec![choice.clone()]).with_notes(notes.clone());
    let right = build_alternative(&name, &variable, vec![choice]).with_notes(notes);

    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
}

#[test]
fn separately_built_equal_alternatives_are_equal_as_parts_and_hash_alike() {
    let name = Identifier::new("o");
    let variable = int_variable(&Identifier::new("x"), &[1]);

    let left = Part::new(build_alternative(&name, &variable, Vec::new()));
    let right = Part::new(build_alternative(&name, &variable, Vec::new()));

    assert!(!Part::ptr_eq(&left, &right));
    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
}

#[test]
fn alternatives_with_different_names_are_unequal() {
    let variable = int_variable(&Identifier::new("x"), &[1]);

    let left = build_alternative(&Identifier::new("o"), &variable, Vec::new());
    let right = build_alternative(&Identifier::new("p"), &variable, Vec::new());

    assert_ne!(left, right);
    assert_ne!(hash_of(&left), hash_of(&right));
}

#[test]
fn alternatives_with_different_variables_are_unequal() {
    let name = Identifier::new("o");

    let left = build_alternative(
        &name,
        &int_variable(&Identifier::new("x"), &[1]),
        Vec::new(),
    );
    let right = build_alternative(
        &name,
        &int_variable(&Identifier::new("x"), &[1]),
        Vec::new(),
    );

    assert_ne!(left, right);
    assert_ne!(hash_of(&left), hash_of(&right));
}

#[test]
fn alternatives_with_different_sub_choices_are_unequal() {
    let name = Identifier::new("o");
    let variable = int_variable(&Identifier::new("x"), &[1]);

    let left = build_alternative(
        &name,
        &variable,
        vec![build_sub_choice(
            &Identifier::new("c"),
            &Identifier::new("s"),
            &Identifier::new("u"),
        )],
    );
    let right = build_alternative(&name, &variable, Vec::new());

    assert_ne!(left, right);
    assert_ne!(hash_of(&left), hash_of(&right));
}

#[test]
fn alternatives_with_different_notes_are_unequal() {
    let name = Identifier::new("o");
    let variable = int_variable(&Identifier::new("x"), &[1]);

    let left = build_alternative(&name, &variable, Vec::new())
        .with_notes(vec![Note::with_other_kind("one")]);
    let right = build_alternative(&name, &variable, Vec::new())
        .with_notes(vec![Note::with_other_kind("two")]);

    assert_ne!(left, right);
    assert_ne!(hash_of(&left), hash_of(&right));
}

// ---------------------------------------------------------------------------
// Alpha equivalence on their own
// ---------------------------------------------------------------------------

#[test]
fn alternatives_with_distinct_labels_are_alpha_equivalent_standalone() {
    let left = plain_alternative(
        &Identifier::new("o"),
        vec![
            int_variable(&Identifier::new("x"), &[1, 2]),
            int_variable(&Identifier::new("y"), &[3]),
        ],
        Vec::new(),
    );
    let right = plain_alternative(
        &Identifier::new("p"),
        vec![
            int_variable(&Identifier::new("u"), &[1, 2]),
            int_variable(&Identifier::new("w"), &[3]),
        ],
        Vec::new(),
    );

    assert_eq!(compare_alpha_both_ways(&left, &right), [true, true]);
}

#[test]
fn bare_alternatives_with_distinct_names_are_alpha_equivalent() {
    let left = bare_alternative(&Identifier::new("o"));
    let right = bare_alternative(&Identifier::new("p"));

    assert_eq!(compare_alpha_both_ways(&left, &right), [true, true]);
}

#[test]
fn alternatives_with_different_variable_domains_are_not_alpha_equivalent() {
    let left = plain_alternative(
        &Identifier::new("o"),
        vec![int_variable(&Identifier::new("x"), &[1, 2])],
        Vec::new(),
    );
    let right = plain_alternative(
        &Identifier::new("p"),
        vec![int_variable(&Identifier::new("y"), &[1, 3])],
        Vec::new(),
    );

    assert_eq!(compare_alpha_both_ways(&left, &right), [false, false]);
}

#[test]
fn alternatives_differing_only_in_a_domain_are_not_alpha_equivalent() {
    let build = |second_domain: &[i64]| {
        plain_alternative(
            &Identifier::new("o"),
            vec![
                int_variable(&Identifier::new("x"), &[1]),
                int_variable(&Identifier::new("y"), &[2]),
                int_variable(&Identifier::new("z"), second_domain),
            ],
            Vec::new(),
        )
    };
    let left = build(&[5, 6]);

    assert_eq!(
        compare_alpha_both_ways(&left, &build(&[5, 6])),
        [true, true]
    );
    assert_eq!(
        compare_alpha_both_ways(&left, &build(&[5, 7])),
        [false, false]
    );
}

#[test]
fn alternatives_with_different_notes_are_not_alpha_equivalent() {
    let part = |name: &str, note: &str| {
        Part::new(
            PlainAlternative::new(Identifier::new(name), Vec::new(), Vec::new())
                .expect("the names are distinct")
                .with_notes(vec![Note::with_other_kind(note)]),
        ) as Part<dyn Alternative>
    };

    assert_eq!(
        compare_alpha_both_ways(&part("o", "one"), &part("p", "two")),
        [false, false]
    );
    assert_eq!(
        compare_alpha_both_ways(&part("o", "same"), &part("p", "same")),
        [true, true]
    );
}

#[test]
fn alternatives_with_variables_in_a_different_order_are_not_alpha_equivalent() {
    let left = plain_alternative(
        &Identifier::new("o"),
        vec![
            int_variable(&Identifier::new("x"), &[1]),
            int_variable(&Identifier::new("y"), &[2]),
        ],
        Vec::new(),
    );
    let right = plain_alternative(
        &Identifier::new("p"),
        vec![
            int_variable(&Identifier::new("u"), &[2]),
            int_variable(&Identifier::new("w"), &[1]),
        ],
        Vec::new(),
    );

    assert_eq!(compare_alpha_both_ways(&left, &right), [false, false]);
}

#[test]
fn names_inside_an_alternative_bind_the_references_inside_it() {
    let (o, x, y) = (
        Identifier::new("o"),
        Identifier::new("x"),
        Identifier::new("y"),
    );
    let (p, x2, y2) = (
        Identifier::new("p"),
        Identifier::new("x2"),
        Identifier::new("y2"),
    );
    let left = plain_alternative(
        &o,
        vec![int_variable(&x, &[1]), build_variable_over(&y, &[&x])],
        Vec::new(),
    );
    let renamed = plain_alternative(
        &p,
        vec![int_variable(&x2, &[1]), build_variable_over(&y2, &[&x2])],
        Vec::new(),
    );
    let free = Identifier::new("z");
    let dangling = plain_alternative(
        &p,
        vec![int_variable(&x2, &[1]), build_variable_over(&y2, &[&free])],
        Vec::new(),
    );

    assert_eq!(compare_alpha_both_ways(&left, &renamed), [true, true]);
    assert_eq!(compare_alpha_both_ways(&left, &dangling), [false, false]);
}

#[test]
fn an_alternative_with_a_relabeled_sub_choice_is_alpha_equivalent() {
    let build = |names: [&str; 6]| {
        let [outer, choice, first, second, first_var, second_var] = names.map(Identifier::new);
        plain_alternative(
            &outer,
            Vec::new(),
            vec![choice_of(
                &choice,
                vec![
                    plain_alternative(&first, vec![int_variable(&first_var, &[1])], Vec::new()),
                    plain_alternative(&second, vec![int_variable(&second_var, &[2])], Vec::new()),
                ],
            )],
        )
    };

    let left = build(["o", "c", "s", "t", "u", "w"]);
    let right = build(["o2", "c2", "s2", "t2", "u2", "w2"]);

    assert_eq!(compare_alpha_both_ways(&left, &right), [true, true]);
}

#[test]
fn an_alternative_whose_sub_choice_alternatives_are_swapped_is_not_alpha_equivalent() {
    let build = |swapped: bool| {
        let first = plain_alternative(
            &Identifier::new("s"),
            vec![int_variable(&Identifier::new("u"), &[1])],
            Vec::new(),
        );
        let second = plain_alternative(
            &Identifier::new("t"),
            vec![int_variable(&Identifier::new("w"), &[2])],
            Vec::new(),
        );
        let alternatives = if swapped {
            vec![second, first]
        } else {
            vec![first, second]
        };
        plain_alternative(
            &Identifier::new("o"),
            Vec::new(),
            vec![choice_of(&Identifier::new("c"), alternatives)],
        )
    };

    assert_eq!(
        compare_alpha_both_ways(&build(false), &build(true)),
        [false, false]
    );
}

#[test]
fn a_sub_choice_name_binds_references_inside_the_alternative() {
    let build = |names: [&str; 4]| {
        let [outer, variable, choice, inner] = names.map(Identifier::new);
        let reference = build_variable_over(&variable, &[&choice]);
        plain_alternative(
            &outer,
            vec![reference],
            vec![choice_of(&choice, vec![bare_alternative(&inner)])],
        )
    };

    let left = build(["o", "x", "c", "s"]);
    let right = build(["o2", "x2", "c2", "s2"]);

    assert_eq!(compare_alpha_both_ways(&left, &right), [true, true]);
}

#[rstest]
#[case::prefix_first(true)]
#[case::prefix_second(false)]
fn alternative_is_not_equivalent_to_a_variable_prefix(#[case] prefix_first: bool) {
    let name = Identifier::new("o");
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let (first, second) = (int_param(&[1]), int_param(&[2]));
    let order = |prefix: Part<dyn Alternative>, longer: Part<dyn Alternative>| {
        if prefix_first {
            (prefix, longer)
        } else {
            (longer, prefix)
        }
    };
    let (left, right) = order(
        plain_alternative(&name, vec![plain_variable(&x, first.clone())], Vec::new()),
        plain_alternative(
            &name,
            vec![plain_variable(&x, first), plain_variable(&y, second)],
            Vec::new(),
        ),
    );
    let (fresh_left, fresh_right) = order(
        plain_alternative(
            &Identifier::new("p"),
            vec![int_variable(&Identifier::new("a"), &[1])],
            Vec::new(),
        ),
        plain_alternative(
            &Identifier::new("q"),
            vec![
                int_variable(&Identifier::new("b"), &[1]),
                int_variable(&Identifier::new("c"), &[2]),
            ],
            Vec::new(),
        ),
    );

    assert_eq!(compare_structural_both_ways(&left, &right), [false, false]);
    assert_eq!(
        compare_alpha_both_ways(&fresh_left, &fresh_right),
        [false, false]
    );
}

#[rstest]
#[case::prefix_first(true)]
#[case::prefix_second(false)]
fn alternative_is_not_equivalent_to_a_sub_choice_prefix(#[case] prefix_first: bool) {
    let make = |extra: bool| {
        let mut choices = vec![build_sub_choice(
            &Identifier::new("c"),
            &Identifier::new("s"),
            &Identifier::new("u"),
        )];
        if extra {
            choices.push(build_sub_choice(
                &Identifier::new("d"),
                &Identifier::new("t"),
                &Identifier::new("w"),
            ));
        }
        plain_alternative(&Identifier::new("o"), Vec::new(), choices)
    };
    let (left, right) = if prefix_first {
        (make(false), make(true))
    } else {
        (make(true), make(false))
    };

    assert_eq!(compare_alpha_both_ways(&left, &right), [false, false]);
}

// ---------------------------------------------------------------------------
// Structural equivalence
// ---------------------------------------------------------------------------

#[test]
fn alternatives_sharing_all_parts_are_structurally_equivalent() {
    let name = Identifier::new("o");
    let variable = int_variable(&Identifier::new("x"), &[1]);
    let choice = build_sub_choice(
        &Identifier::new("c"),
        &Identifier::new("s"),
        &Identifier::new("u"),
    );
    let notes = vec![Note::with_other_kind("note")];
    let left = Part::new(
        build_alternative(&name, &variable, vec![choice.clone()]).with_notes(notes.clone()),
    );
    let right = Part::new(build_alternative(&name, &variable, vec![choice]).with_notes(notes));

    assert_eq!(compare_structural_both_ways(&left, &right), [true, true]);
}

#[test]
fn an_alternative_is_structurally_equivalent_to_its_clone() {
    let alternative = plain_alternative(
        &Identifier::new("o"),
        vec![int_variable(&Identifier::new("x"), &[1])],
        Vec::new(),
    );

    assert_eq!(
        compare_structural_both_ways(&alternative, &alternative.clone()),
        [true, true]
    );
}

#[test]
fn alternatives_with_different_names_are_not_structurally_equivalent() {
    let variable = int_variable(&Identifier::new("x"), &[1]);
    let left = Part::new(build_alternative(
        &Identifier::new("o"),
        &variable,
        Vec::new(),
    ));
    let right = Part::new(build_alternative(
        &Identifier::new("p"),
        &variable,
        Vec::new(),
    ));

    assert_eq!(compare_structural_both_ways(&left, &right), [false, false]);
}

#[test]
fn alternatives_with_different_variables_are_not_structurally_equivalent() {
    let name = Identifier::new("o");
    let left = Part::new(build_alternative(
        &name,
        &int_variable(&Identifier::new("x"), &[1]),
        Vec::new(),
    ));
    let right = Part::new(build_alternative(
        &name,
        &int_variable(&Identifier::new("y"), &[1]),
        Vec::new(),
    ));

    assert_eq!(compare_structural_both_ways(&left, &right), [false, false]);
}

#[test]
fn alternatives_with_different_sub_choices_are_not_structurally_equivalent() {
    let name = Identifier::new("o");
    let variable = int_variable(&Identifier::new("x"), &[1]);
    let left = Part::new(build_alternative(
        &name,
        &variable,
        vec![build_sub_choice(
            &Identifier::new("c"),
            &Identifier::new("s"),
            &Identifier::new("u"),
        )],
    ));
    let right = Part::new(build_alternative(&name, &variable, Vec::new()));

    assert_eq!(compare_structural_both_ways(&left, &right), [false, false]);
}

#[test]
fn alternatives_with_different_notes_are_not_structurally_equivalent() {
    let name = Identifier::new("o");
    let variable = int_variable(&Identifier::new("x"), &[1]);
    let left = Part::new(
        build_alternative(&name, &variable, Vec::new())
            .with_notes(vec![Note::with_other_kind("one")]),
    );
    let right = Part::new(
        build_alternative(&name, &variable, Vec::new())
            .with_notes(vec![Note::with_other_kind("two")]),
    );

    assert_eq!(compare_structural_both_ways(&left, &right), [false, false]);
}

#[rstest]
#[case::prefix_first(true)]
#[case::prefix_second(false)]
fn an_alternative_is_not_structurally_equivalent_to_one_with_an_extra_sub_choice(
    #[case] prefix_first: bool,
) {
    let name = Identifier::new("o");
    let choice = build_sub_choice(
        &Identifier::new("c"),
        &Identifier::new("s"),
        &Identifier::new("u"),
    );
    let bare = plain_alternative(&name, Vec::new(), Vec::new());
    let extended = plain_alternative(&name, Vec::new(), vec![choice]);
    let (left, right) = if prefix_first {
        (bare, extended)
    } else {
        (extended, bare)
    };

    assert_eq!(compare_structural_both_ways(&left, &right), [false, false]);
}

#[test]
fn alpha_comparison_under_a_renaming_pairs_the_free_names_too() {
    let (a, b) = (Identifier::new("a"), Identifier::new("b"));
    let left = plain_alternative(
        &Identifier::new("o"),
        vec![build_variable_over(&Identifier::new("x"), &[&a])],
        Vec::new(),
    );
    let right = plain_alternative(
        &Identifier::new("p"),
        vec![build_variable_over(&Identifier::new("y"), &[&b])],
        Vec::new(),
    );
    let renaming = AlphaRenaming::new(HashMap::from([(a, b)])).expect("injective");

    assert!(!left.is_alpha_equivalent(&right).expect("compares"));
    assert!(
        left.is_alpha_equivalent_under(&right, &renaming)
            .expect("compares")
    );
}
