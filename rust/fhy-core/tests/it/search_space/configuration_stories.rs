//! Tests for `Configuration`: the problems `Configuration::new` collects and
//! their order, completeness, the accessors, `with_entry`, `with_entries`,
//! `without_entry` and `without_entries`, `==` and `Hash`, and the
//! `ConfigurationKey`.

#![expect(
    clippy::many_single_char_names,
    reason = "the stories name decisions as the design's examples do: c, a, b, x, y"
)]

use std::collections::HashMap;

use fhy_core::constraint::Value;
use fhy_core::identifier::Identifier;
use fhy_core::param::{AssignmentError, ParamContext};
use fhy_core::search_space::{ConfigurationError, ConfigurationErrors, Space};
use rstest::rstest;

use crate::support::constraint::{int, text};
use crate::support::hashing::hash_of;
use crate::support::param::{in_set, not_in_set};
use crate::support::search_space::{
    bare_alternative, categorical, categorical_where, choice_of, chosen, condition, configure,
    forbidden, ground_solver, int_param, int_variable, plain_alternative, plain_variable,
    try_configure,
};

/// The names of the selection space: the top-level variable `t` over
/// `{1, 2, 3}`, and the choice `c` among `a` (holding `k1` over `{1, 2}`
/// and `k2` over `{1, 2}` where `k2 in {1}`) and `b` (holding `m` over
/// `{1, 2}`).
struct Selection {
    space: Space,
    t: Identifier,
    c: Identifier,
    a: Identifier,
    k1: Identifier,
    k2: Identifier,
    b: Identifier,
    m: Identifier,
}

/// Return the selection space, its names hinted as given.
fn build_selection() -> Selection {
    let [space, t, c, a, k1, k2, b, m] =
        ["selection", "t", "c", "a", "k1", "k2", "b", "m"].map(Identifier::new);
    let narrowed = categorical_where(vec![int(1), int(2)], |p| vec![in_set(p, [int(1)])]);
    let choice = choice_of(
        &c,
        vec![
            plain_alternative(
                &a,
                vec![int_variable(&k1, &[1, 2]), plain_variable(&k2, narrowed)],
                Vec::new(),
            ),
            plain_alternative(&b, vec![int_variable(&m, &[1, 2])], Vec::new()),
        ],
    );
    let space = Space::new(
        space,
        vec![int_variable(&t, &[1, 2, 3])],
        vec![choice],
        Vec::new(),
        Vec::new(),
    )
    .expect("the space is valid");
    Selection {
        space,
        t,
        c,
        a,
        k1,
        k2,
        b,
        m,
    }
}

/// Return the problems `Configuration::new` refuses `entries` with.
fn collect_problems(space: &Space, entries: Vec<(Identifier, Value)>) -> ConfigurationErrors {
    match try_configure(space, entries) {
        Ok(configuration) => panic!("expected a refusal, got {configuration:?}"),
        Err(errors) => errors,
    }
}

/// Return the context the configurations of these tests are checked
/// under, for `with_entry` and `with_entries`.
fn with_ground<T>(run: impl FnOnce(&ParamContext<'_>) -> T) -> T {
    let solver = ground_solver();
    run(&ParamContext::new(&solver))
}

// ---------------------------------------------------------------------------
// Valid configurations and completeness
// ---------------------------------------------------------------------------

#[test]
fn complete_configuration_is_valid_and_complete() {
    let s = build_selection();

    let configuration = configure(
        &s.space,
        [
            (s.t.clone(), int(3)),
            (s.c.clone(), chosen(&s.a)),
            (s.k1.clone(), int(2)),
            (s.k2.clone(), int(1)),
        ],
    );

    assert!(configuration.is_complete());
}

#[test]
fn empty_configuration_is_valid_and_incomplete() {
    let s = build_selection();

    let configuration = configure(&s.space, []);

    assert!(!configuration.is_complete());
    assert_eq!(configuration.entries().len(), 0);
}

#[test]
fn partial_configuration_is_valid_and_incomplete() {
    let s = build_selection();

    let configuration = configure(&s.space, [(s.t.clone(), int(1))]);

    assert!(!configuration.is_complete());
    assert_eq!(configuration.value(&s.t), Some(&int(1)));
}

#[test]
fn chosen_alternative_with_unassigned_variables_is_valid_and_incomplete() {
    let s = build_selection();

    let configuration = configure(
        &s.space,
        [(s.t.clone(), int(1)), (s.c.clone(), chosen(&s.a))],
    );

    assert!(!configuration.is_complete());
}

#[test]
fn configuration_leaving_only_inactive_decisions_unassigned_is_complete() {
    let s = build_selection();

    let configuration = configure(
        &s.space,
        [
            (s.t.clone(), int(1)),
            (s.c.clone(), chosen(&s.b)),
            (s.m.clone(), int(2)),
        ],
    );

    assert!(
        configuration.is_complete(),
        "k1 and k2 are inactive, so they need no value"
    );
}

#[test]
fn configuration_with_a_pending_decision_is_incomplete() {
    let [x, w] = ["x", "w"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("pending"),
        vec![int_variable(&x, &[1, 2]), int_variable(&w, &[1])],
        Vec::new(),
        vec![condition(&w, [in_set(&x, [int(1)])])],
        Vec::new(),
    )
    .expect("the space is valid");

    let configuration = configure(&space, []);

    assert!(!configuration.is_complete());
}

#[test]
fn space_without_decisions_has_a_complete_empty_configuration() {
    let space = Space::new(
        Identifier::new("nothing"),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )
    .expect("the space is valid");

    let configuration = configure(&space, []);

    assert!(configuration.is_complete());
}

// ---------------------------------------------------------------------------
// Problems
// ---------------------------------------------------------------------------

#[rstest]
#[case::an_alternative_name("alternative")]
#[case::the_space_name("space")]
#[case::a_fresh_identifier("fresh")]
fn configuration_new_refuses_an_entry_naming_no_decision(#[case] which: &str) {
    let s = build_selection();
    let name = match which {
        "alternative" => s.a.clone(),
        "space" => s.space.name().clone(),
        "fresh" => Identifier::new("fresh"),
        _ => unreachable!("unknown case {which}"),
    };

    let errors = collect_problems(&s.space, vec![(name.clone(), int(1))]);

    let [ConfigurationError::UnknownDecision { name: reported }] = errors.errors() else {
        panic!("expected one UnknownDecision, got {errors:?}");
    };
    assert_eq!(reported, &name);
}

#[rstest]
#[case::with_equal_values(1)]
#[case::with_another_admissible_value(2)]
#[case::with_an_inadmissible_value(99)]
fn configuration_new_refuses_a_second_entry_for_one_decision(#[case] second: i64) {
    let s = build_selection();

    let errors = collect_problems(
        &s.space,
        vec![(s.t.clone(), int(1)), (s.t.clone(), int(second))],
    );

    let [ConfigurationError::DuplicateEntry { name }] = errors.errors() else {
        panic!("expected one DuplicateEntry and nothing about the dropped value, got {errors:?}");
    };
    assert_eq!(name, &s.t);
}

#[test]
fn configuration_new_keeps_the_first_of_two_entries_for_one_decision() {
    let s = build_selection();

    let errors = collect_problems(
        &s.space,
        vec![(s.t.clone(), int(99)), (s.t.clone(), int(1))],
    );

    let [
        ConfigurationError::DuplicateEntry { name },
        ConfigurationError::Assignment { variable, error },
    ] = errors.errors()
    else {
        panic!("expected DuplicateEntry then the first value's Assignment, got {errors:?}");
    };
    assert_eq!((name, variable), (&s.t, &s.t));
    assert!(
        matches!(error, AssignmentError::Inadmissible),
        "got {error:?}"
    );
}

#[test]
fn unknown_alternative_is_refused() {
    let s = build_selection();
    let stranger = Identifier::new("stranger");

    let errors = collect_problems(&s.space, vec![(s.c.clone(), chosen(&stranger))]);

    let [ConfigurationError::UnknownAlternative { choice, value }] = errors.errors() else {
        panic!("expected one UnknownAlternative, got {errors:?}");
    };
    assert_eq!(choice, &s.c);
    assert_eq!(value, &chosen(&stranger));
}

#[rstest]
#[case::an_integer("integer")]
#[case::the_name_as_a_string("string")]
#[case::a_variable_name("variable")]
#[case::the_choice_itself("choice")]
fn configuration_new_refuses_a_choice_value_that_names_none_of_its_alternatives(
    #[case] which: &str,
) {
    let s = build_selection();
    let value = match which {
        "integer" => int(1),
        "string" => text("a"),
        "variable" => chosen(&s.k1),
        "choice" => chosen(&s.c),
        _ => unreachable!("unknown case {which}"),
    };

    let errors = collect_problems(&s.space, vec![(s.c.clone(), value.clone())]);

    let [
        ConfigurationError::UnknownAlternative {
            choice,
            value: reported,
        },
    ] = errors.errors()
    else {
        panic!("expected one UnknownAlternative, got {errors:?}");
    };
    assert_eq!((choice, reported), (&s.c, &value));
}

#[test]
fn value_for_an_inactive_variable_is_refused() {
    let s = build_selection();

    let errors = collect_problems(
        &s.space,
        vec![(s.c.clone(), chosen(&s.a)), (s.m.clone(), int(1))],
    );

    let [ConfigurationError::InactiveDecision { name }] = errors.errors() else {
        panic!("expected one InactiveDecision, got {errors:?}");
    };
    assert_eq!(name, &s.m);
}

#[test]
fn values_under_an_unchosen_choice_are_refused() {
    let s = build_selection();

    let errors = collect_problems(
        &s.space,
        vec![(s.k1.clone(), int(1)), (s.m.clone(), int(1))],
    );

    let names: Vec<_> = errors
        .errors()
        .iter()
        .map(|problem| match problem {
            ConfigurationError::InactiveDecision { name } => name.clone(),
            other => panic!("expected only InactiveDecision problems, got {other:?}"),
        })
        .collect();
    assert_eq!(names, vec![s.k1.clone(), s.m.clone()]);
}

#[rstest]
#[case::outside_the_domain(Value::Int(7.into()), "inadmissible")]
#[case::a_boolean_for_an_integer(Value::Bool(true), "inadmissible")]
#[case::a_string_for_an_integer(text("1"), "inadmissible")]
#[case::violating_the_params_constraint(int(2), "violated")]
fn configuration_new_refuses_a_value_its_param_refuses(
    #[case] value: Value,
    #[case] expected: &str,
) {
    let s = build_selection();

    let errors = collect_problems(
        &s.space,
        vec![(s.c.clone(), chosen(&s.a)), (s.k2.clone(), value)],
    );

    let [ConfigurationError::Assignment { variable, error }] = errors.errors() else {
        panic!("expected one Assignment, got {errors:?}");
    };
    assert_eq!(variable, &s.k2);
    match expected {
        "inadmissible" => assert!(
            matches!(error, AssignmentError::Inadmissible),
            "got {error:?}"
        ),
        "violated" => assert!(
            matches!(error, AssignmentError::ViolatedConstraint { .. }),
            "got {error:?}"
        ),
        _ => unreachable!("unknown expectation {expected}"),
    }
}

#[test]
fn refused_choice_value_leaves_its_alternatives_variables_pending() {
    let s = build_selection();

    let errors = collect_problems(
        &s.space,
        vec![(s.c.clone(), int(1)), (s.k1.clone(), int(1))],
    );

    let [
        ConfigurationError::UnknownAlternative { choice, .. },
        ConfigurationError::InactiveDecision { name },
    ] = errors.errors()
    else {
        panic!("expected UnknownAlternative then InactiveDecision, got {errors:?}");
    };
    assert_eq!((choice, name), (&s.c, &s.k1));
}

#[test]
fn refused_variable_value_counts_as_unassigned_for_conditions() {
    let [x, w] = ["x", "w"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("dropped"),
        vec![int_variable(&x, &[1, 2]), int_variable(&w, &[1])],
        Vec::new(),
        vec![condition(&w, [in_set(&x, [int(1)])])],
        Vec::new(),
    )
    .expect("the space is valid");

    let errors = collect_problems(&space, vec![(x.clone(), int(9)), (w.clone(), int(1))]);

    let [
        ConfigurationError::Assignment { variable, .. },
        ConfigurationError::InactiveDecision { name },
    ] = errors.errors()
    else {
        panic!("expected Assignment then InactiveDecision, got {errors:?}");
    };
    assert_eq!((variable, name), (&x, &w));
}

#[test]
fn configuration_new_collects_every_problem_in_the_documented_order() {
    let [t, c, a, k, b, m, w] = ["t", "c", "a", "k", "b", "m", "w"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("everything"),
        vec![int_variable(&t, &[1, 2]), int_variable(&w, &[1, 2])],
        vec![choice_of(
            &c,
            vec![
                plain_alternative(&a, vec![int_variable(&k, &[1, 2])], Vec::new()),
                plain_alternative(&b, vec![int_variable(&m, &[1, 2])], Vec::new()),
            ],
        )],
        vec![condition(&w, [in_set(&t, [int(2)])])],
        vec![forbidden([in_set(&k, [int(1)])])],
    )
    .expect("the space is valid");
    let ghost = Identifier::new("ghost");

    let errors = collect_problems(
        &space,
        vec![
            (w.clone(), int(1)),
            (ghost.clone(), int(1)),
            (k.clone(), int(1)),
            (t.clone(), int(9)),
            (c.clone(), chosen(&a)),
            (m.clone(), int(1)),
            (t.clone(), int(1)),
        ],
    );

    let found: Vec<String> = errors
        .errors()
        .iter()
        .map(|problem| match problem {
            ConfigurationError::UnknownDecision { name } => format!("unknown {}", name.name_hint()),
            ConfigurationError::DuplicateEntry { name } => {
                format!("duplicate {}", name.name_hint())
            }
            ConfigurationError::Assignment { variable, .. } => {
                format!("assignment {}", variable.name_hint())
            }
            ConfigurationError::InactiveDecision { name } => {
                format!("inactive {}", name.name_hint())
            }
            ConfigurationError::Forbidden { index } => format!("forbidden {index}"),
            other => format!("{other:?}"),
        })
        .collect();
    assert_eq!(
        found,
        [
            "unknown ghost",
            "duplicate t",
            "assignment t",
            "inactive w",
            "inactive m",
            "forbidden 0",
        ],
        "the entries in order, then the decisions in decision order (t, w, c, k, m), then the \
         clauses: t's refused value leaves w pending, and m's alternative is not chosen"
    );
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

#[test]
fn configuration_value_of_a_choice_is_its_alternatives_name() {
    let s = build_selection();

    let configuration = configure(&s.space, [(s.c.clone(), chosen(&s.b))]);

    assert_eq!(configuration.value(&s.c), Some(&chosen(&s.b)));
    assert_eq!(configuration.value(&s.m), None);
    assert_eq!(configuration.value(&Identifier::new("unknown")), None);
}

#[test]
fn configuration_alternative_returns_the_chosen_alternative() {
    let s = build_selection();

    let configuration = configure(&s.space, [(s.c.clone(), chosen(&s.b))]);

    let alternative = configuration.alternative(&s.c).expect("c is assigned");
    assert_eq!(alternative.get().name(), &s.b);
    assert_eq!(alternative, &s.space.choices()[0].alternatives()[1]);
}

#[rstest]
#[case::an_unassigned_choice("unassigned")]
#[case::a_variable("variable")]
#[case::an_unknown_name("unknown")]
fn configuration_alternative_answers_none_without_a_chosen_alternative(#[case] which: &str) {
    let s = build_selection();
    let configuration = configure(&s.space, [(s.t.clone(), int(1))]);
    let name = match which {
        "unassigned" => s.c.clone(),
        "variable" => s.t.clone(),
        "unknown" => Identifier::new("unknown"),
        _ => unreachable!("unknown case {which}"),
    };

    assert!(configuration.alternative(&name).is_none());
}

#[test]
fn configuration_entries_are_in_canonical_order_whatever_the_input_order() {
    let s = build_selection();

    let configuration = configure(
        &s.space,
        [
            (s.k2.clone(), int(1)),
            (s.c.clone(), chosen(&s.a)),
            (s.t.clone(), int(3)),
            (s.k1.clone(), int(2)),
        ],
    );

    let entries: Vec<(Identifier, Value)> = configuration
        .entries()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    assert_eq!(
        entries,
        vec![
            (s.t.clone(), int(3)),
            (s.c.clone(), chosen(&s.a)),
            (s.k1.clone(), int(2)),
            (s.k2.clone(), int(1)),
        ]
    );
}

#[test]
fn configuration_space_is_the_space_it_was_built_for() {
    let s = build_selection();

    let configuration = configure(&s.space, [(s.t.clone(), int(1))]);

    assert_eq!(configuration.space(), &s.space);
}

// ---------------------------------------------------------------------------
// with_entry and with_entries
// ---------------------------------------------------------------------------

#[test]
fn configuration_with_entry_adds_a_value_and_keeps_the_original() {
    let s = build_selection();
    let original = configure(&s.space, [(s.t.clone(), int(1))]);

    let extended = with_ground(|context| original.with_entry(s.c.clone(), chosen(&s.a), context))
        .expect("c may choose a");

    assert_eq!(extended.value(&s.c), Some(&chosen(&s.a)));
    assert_eq!(extended.value(&s.t), Some(&int(1)));
    assert_eq!(original.value(&s.c), None);
}

#[test]
fn configuration_with_entry_replaces_a_value() {
    let s = build_selection();
    let original = configure(&s.space, [(s.t.clone(), int(1))]);

    let replaced = with_ground(|context| original.with_entry(s.t.clone(), int(3), context))
        .expect("t may be 3");

    assert_eq!(replaced.value(&s.t), Some(&int(3)));
    assert_eq!(replaced.entries().len(), 1);
}

#[test]
fn configuration_with_entry_checks_the_whole_configuration() {
    let s = build_selection();
    let original = configure(
        &s.space,
        [(s.c.clone(), chosen(&s.a)), (s.k1.clone(), int(1))],
    );

    let result = with_ground(|context| original.with_entry(s.c.clone(), chosen(&s.b), context));

    let errors = result.expect_err("k1 is inactive under b");
    let [ConfigurationError::InactiveDecision { name }] = errors.errors() else {
        panic!("expected one InactiveDecision, got {errors:?}");
    };
    assert_eq!(name, &s.k1, "no entry is dropped implicitly");
}

#[test]
fn configuration_with_entry_refuses_a_value_the_param_refuses() {
    let s = build_selection();
    let original = configure(&s.space, [(s.t.clone(), int(1))]);

    let result = with_ground(|context| original.with_entry(s.t.clone(), int(4), context));

    let errors = result.expect_err("4 is not a category of t");
    assert!(
        matches!(
            errors.errors(),
            [ConfigurationError::Assignment {
                error: AssignmentError::Inadmissible,
                ..
            }]
        ),
        "got {errors:?}"
    );
}

#[test]
fn configuration_with_entries_adds_and_replaces_values() {
    let s = build_selection();
    let original = configure(&s.space, [(s.t.clone(), int(1))]);

    let extended = with_ground(|context| {
        original.with_entries(
            [
                (s.t.clone(), int(2)),
                (s.c.clone(), chosen(&s.b)),
                (s.m.clone(), int(2)),
            ],
            context,
        )
    })
    .expect("the entries are valid");

    assert_eq!(extended.value(&s.t), Some(&int(2)));
    assert_eq!(extended.value(&s.m), Some(&int(2)));
    assert!(extended.is_complete());
}

#[test]
fn configuration_with_entries_refuses_two_entries_for_one_decision() {
    let s = build_selection();
    let original = configure(&s.space, [(s.t.clone(), int(1))]);

    let result = with_ground(|context| {
        original.with_entries([(s.t.clone(), int(2)), (s.t.clone(), int(3))], context)
    });

    let errors = result.expect_err("t is given twice");
    let [ConfigurationError::DuplicateEntry { name }] = errors.errors() else {
        panic!("expected one DuplicateEntry, got {errors:?}");
    };
    assert_eq!(name, &s.t);
}

// ---------------------------------------------------------------------------
// without_entry and without_entries
// ---------------------------------------------------------------------------

#[test]
fn configuration_without_entry_removes_a_value_and_keeps_the_original() {
    let s = build_selection();
    let original = configure(
        &s.space,
        [(s.t.clone(), int(1)), (s.c.clone(), chosen(&s.b))],
    );

    let reduced = with_ground(|context| original.without_entry(s.t.clone(), context))
        .expect("t may be unassigned");

    assert_eq!(reduced.value(&s.t), None);
    assert_eq!(reduced.value(&s.c), Some(&chosen(&s.b)));
    assert_eq!(original.value(&s.t), Some(&int(1)));
}

#[test]
fn configuration_without_entries_equals_the_configuration_built_without_them() {
    let s = build_selection();
    let original = configure(
        &s.space,
        [
            (s.t.clone(), int(1)),
            (s.c.clone(), chosen(&s.a)),
            (s.k1.clone(), int(2)),
            (s.k2.clone(), int(1)),
        ],
    );
    let expected = configure(
        &s.space,
        [(s.t.clone(), int(1)), (s.c.clone(), chosen(&s.a))],
    );

    let reduced =
        with_ground(|context| original.without_entries([s.k1.clone(), s.k2.clone()], context))
            .expect("a's variables may be unassigned");

    assert_eq!(reduced, expected);
    assert_eq!(reduced.key(), expected.key());
    assert!(!reduced.is_complete());
}

#[test]
fn configuration_without_entries_then_with_entry_switches_an_alternative() {
    let s = build_selection();
    let original = configure(
        &s.space,
        [(s.c.clone(), chosen(&s.a)), (s.k1.clone(), int(1))],
    );

    let switched = with_ground(|context| {
        original
            .without_entries([s.k1.clone()], context)?
            .with_entry(s.c.clone(), chosen(&s.b), context)
    })
    .expect("with k1 dropped, c may choose b");

    assert_eq!(switched.value(&s.c), Some(&chosen(&s.b)));
    assert_eq!(switched.value(&s.k1), None);
    assert_eq!(
        switched.key(),
        configure(&s.space, [(s.c.clone(), chosen(&s.b))]).key()
    );
}

#[test]
fn configuration_without_entry_of_an_unassigned_decision_changes_nothing() {
    let s = build_selection();
    let original = configure(&s.space, [(s.t.clone(), int(1))]);

    let same = with_ground(|context| original.without_entry(s.c.clone(), context))
        .expect("c holds no value");

    assert_eq!(same, original);
    assert_eq!(same.key(), original.key());
}

#[test]
fn configuration_without_entries_removes_a_decision_named_twice_once() {
    let s = build_selection();
    let original = configure(&s.space, [(s.t.clone(), int(1))]);

    let reduced =
        with_ground(|context| original.without_entries([s.t.clone(), s.t.clone()], context))
            .expect("naming t twice removes it");

    assert_eq!(reduced.entries().len(), 0);
}

#[test]
fn configuration_without_entry_refuses_a_name_that_is_no_decision() {
    let s = build_selection();
    let original = configure(&s.space, [(s.t.clone(), int(1))]);
    let ghost = Identifier::new("ghost");

    let result = with_ground(|context| original.without_entry(ghost.clone(), context));

    let errors = result.expect_err("ghost is no decision of the space");
    let [ConfigurationError::UnknownDecision { name }] = errors.errors() else {
        panic!("expected one UnknownDecision, got {errors:?}");
    };
    assert_eq!(name, &ghost);
}

#[test]
fn configuration_without_entry_refuses_a_choice_whose_variables_hold_values() {
    let s = build_selection();
    let original = configure(
        &s.space,
        [(s.c.clone(), chosen(&s.a)), (s.k1.clone(), int(1))],
    );

    let result = with_ground(|context| original.without_entry(s.c.clone(), context));

    let errors = result.expect_err("k1 is pending once c is unassigned");
    let [ConfigurationError::InactiveDecision { name }] = errors.errors() else {
        panic!("expected one InactiveDecision, got {errors:?}");
    };
    assert_eq!(name, &s.k1, "no entry is dropped implicitly");
}

#[test]
fn configuration_without_entry_checks_conditions_anew() {
    let [x, y, w] = ["x", "y", "w"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("conditioned"),
        vec![
            int_variable(&x, &[1, 2]),
            int_variable(&y, &[1, 2]),
            int_variable(&w, &[1]),
        ],
        Vec::new(),
        vec![condition(&w, [in_set(&x, [int(1)])])],
        Vec::new(),
    )
    .expect("the space is valid");
    let unassigned_target = configure(&space, [(x.clone(), int(1)), (y.clone(), int(2))]);
    let assigned_target = configure(&space, [(x.clone(), int(1)), (w.clone(), int(1))]);

    let reduced = with_ground(|context| unassigned_target.without_entry(x.clone(), context))
        .expect("w is pending and unassigned without x");
    let refused = with_ground(|context| assigned_target.without_entry(x.clone(), context));

    assert_eq!(
        reduced.key(),
        configure(&space, [(y.clone(), int(2))]).key()
    );
    let errors = refused.expect_err("w is pending but holds a value without x");
    let [ConfigurationError::InactiveDecision { name }] = errors.errors() else {
        panic!("expected one InactiveDecision, got {errors:?}");
    };
    assert_eq!(name, &w);
}

// ---------------------------------------------------------------------------
// Equality and hashing
// ---------------------------------------------------------------------------

#[test]
fn configurations_with_equal_entries_are_equal_and_hash_alike() {
    let s = build_selection();

    let left = configure(
        &s.space,
        [(s.t.clone(), int(1)), (s.c.clone(), chosen(&s.a))],
    );
    let right = configure(
        &s.space,
        [(s.c.clone(), chosen(&s.a)), (s.t.clone(), int(1))],
    );

    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
    assert!(
        left.is_structurally_equivalent(&right)
            .expect("plain parts")
    );
}

#[test]
fn configurations_with_different_values_are_unequal() {
    let s = build_selection();

    let left = configure(&s.space, [(s.t.clone(), int(1))]);
    let right = configure(&s.space, [(s.t.clone(), int(2))]);

    assert_ne!(left, right);
    assert!(
        !left
            .is_structurally_equivalent(&right)
            .expect("plain parts")
    );
}

#[test]
fn configurations_of_spaces_built_apart_from_one_description_are_equal() {
    let [space, x] = ["space", "x"].map(Identifier::new);
    let param = int_param(&[1, 2]);
    let build = || {
        let space = Space::new(
            space.clone(),
            vec![plain_variable(&x, param.clone())],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
        .expect("the space is valid");
        configure(&space, [(x.clone(), int(1))])
    };

    let (left, right) = (build(), build());

    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
}

// ---------------------------------------------------------------------------
// Keys
// ---------------------------------------------------------------------------

/// Return a copy of the selection space with every name fresh, and the map
/// from the original names to the copy's.
fn relabel_selection(s: &Selection) -> (Selection, HashMap<Identifier, Identifier>) {
    let copy = build_selection();
    let pairs = [
        (&s.t, &copy.t),
        (&s.c, &copy.c),
        (&s.a, &copy.a),
        (&s.k1, &copy.k1),
        (&s.k2, &copy.k2),
        (&s.b, &copy.b),
        (&s.m, &copy.m),
    ];
    let map = pairs
        .into_iter()
        .map(|(left, right)| (left.clone(), right.clone()))
        .collect();
    (copy, map)
}

#[test]
fn configurations_of_relabeled_spaces_have_equal_keys() {
    let s = build_selection();
    let (copy, names) = relabel_selection(&s);
    let entries = [
        (s.t.clone(), int(3)),
        (s.c.clone(), chosen(&s.a)),
        (s.k1.clone(), int(2)),
    ];
    let renamed = entries.clone().map(|(name, value)| {
        let value = match value {
            Value::Identifier(alternative) => chosen(&names[&alternative]),
            other => other,
        };
        (names[&name].clone(), value)
    });

    let left = configure(&s.space, entries);
    let right = configure(&copy.space, renamed);

    assert_eq!(left.key(), right.key());
    assert_eq!(hash_of(&left.key()), hash_of(&right.key()));
}

#[test]
fn complete_and_incomplete_configurations_have_different_keys() {
    let s = build_selection();
    let base = [
        (s.t.clone(), int(1)),
        (s.c.clone(), chosen(&s.a)),
        (s.k1.clone(), int(1)),
    ];

    let incomplete = configure(&s.space, base.clone());
    let complete = configure(&s.space, base.into_iter().chain([(s.k2.clone(), int(1))]));

    assert!(complete.is_complete() && !incomplete.is_complete());
    assert_ne!(complete.key(), incomplete.key());
}

#[test]
fn different_values_give_different_keys() {
    let s = build_selection();

    let one = configure(&s.space, [(s.t.clone(), int(1))]);
    let two = configure(&s.space, [(s.t.clone(), int(2))]);

    assert_ne!(one.key(), two.key());
}

#[test]
fn assigning_a_choice_changes_the_key() {
    let s = build_selection();

    let unassigned = configure(&s.space, []);
    let chose_a = configure(&s.space, [(s.c.clone(), chosen(&s.a))]);
    let chose_b = configure(&s.space, [(s.c.clone(), chosen(&s.b))]);

    assert_ne!(unassigned.key(), chose_a.key());
    assert_ne!(chose_a.key(), chose_b.key());
    assert_ne!(unassigned.key(), chose_b.key());
}

#[test]
fn equal_configurations_have_equal_keys_and_hashes() {
    let s = build_selection();

    let left = configure(
        &s.space,
        [(s.t.clone(), int(2)), (s.c.clone(), chosen(&s.b))],
    );
    let right = configure(
        &s.space,
        [(s.c.clone(), chosen(&s.b)), (s.t.clone(), int(2))],
    );

    assert_eq!(left.key(), right.key());
    assert_eq!(hash_of(&left.key()), hash_of(&right.key()));
    assert_eq!(left.key().clone(), left.key());
}

#[test]
fn inactive_and_unassigned_decisions_have_different_keys() {
    let [x, w] = ["x", "w"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("gated"),
        vec![int_variable(&x, &[1, 2]), int_variable(&w, &[1])],
        Vec::new(),
        vec![condition(&w, [not_in_set(&x, [int(2)])])],
        Vec::new(),
    )
    .expect("the space is valid");
    let other = Space::new(
        Identifier::new("ungated"),
        vec![int_variable(&x, &[1, 2]), int_variable(&w, &[1])],
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )
    .expect("the space is valid");

    let inactive = configure(&space, [(x.clone(), int(2))]);
    let unassigned = configure(&other, [(x.clone(), int(2))]);

    assert_ne!(
        inactive.key(),
        unassigned.key(),
        "w is inactive in one and active but unassigned in the other"
    );
}

#[test]
fn key_of_a_value_naming_an_alternative_is_invariant_under_relabeling() {
    let build = |forward: bool| {
        let [c, first, second, v] = if forward {
            ["c", "first", "second", "v"].map(Identifier::new)
        } else {
            let v = Identifier::new("v");
            let second = Identifier::new("second");
            let first = Identifier::new("first");
            let c = Identifier::new("c");
            [c, first, second, v]
        };
        let names = vec![chosen(&first), chosen(&second)];
        let space = Space::new(
            Identifier::new("references"),
            vec![plain_variable(&v, categorical(names))],
            vec![choice_of(
                &c,
                vec![bare_alternative(&first), bare_alternative(&second)],
            )],
            Vec::new(),
            Vec::new(),
        )
        .expect("the space is valid");
        configure(
            &space,
            [(v.clone(), chosen(&first)), (c.clone(), chosen(&second))],
        )
    };

    let forward = build(true);
    let backward = build(false);

    assert_eq!(
        forward.key(),
        backward.key(),
        "the categories are the alternatives' names, minted in opposite id orders"
    );
}

#[test]
fn key_of_a_free_identifier_value_keeps_the_identifier() {
    let [left_free, right_free] = ["free", "free"].map(Identifier::new);
    let build = |free: &Identifier| {
        let v = Identifier::new("v");
        let space = Space::new(
            Identifier::new("free_values"),
            vec![plain_variable(&v, categorical(vec![chosen(free)]))],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
        .expect("the space is valid");
        configure(&space, [(v.clone(), chosen(free))])
    };

    let same = (build(&left_free), build(&left_free));
    let different = (build(&left_free), build(&right_free));

    assert_eq!(same.0.key(), same.1.key());
    assert_ne!(
        different.0.key(),
        different.1.key(),
        "a free identifier corresponds only to itself"
    );
}
