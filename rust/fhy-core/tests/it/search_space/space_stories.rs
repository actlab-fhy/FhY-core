//! Tests for `Space`: its names, its decisions in canonical and decision
//! order, the checks of `Space::new` in their documented order, the
//! conditions it keeps per target, and its `==`, `Hash` and structural
//! equivalence.

#![expect(
    clippy::many_single_char_names,
    reason = "the stories name decisions as the design's examples do: c, a, b, x, y"
)]

use std::sync::{Arc, Mutex};

use fhy_core::constraint::{Constraint, ConstraintError, EquationConstraint, Outcome, Value};
use fhy_core::diagnostic::Note;
use fhy_core::expression::Expression;
use fhy_core::foreign::Part;
use fhy_core::identifier::Identifier;
use fhy_core::param::Param;
use fhy_core::search_space::{
    Choice, Condition, Configuration, Decision, Forbidden, Space, SpaceError, Variable,
};
use rstest::rstest;

use crate::support::constraint::{Failing, FailingHook, TestCustom, int};
use crate::support::hashing::hash_of;
use crate::support::param::{at_least, in_set, less_than, literal};
use crate::support::search_space::{
    Realization, bare_alternative, categorical_where, choice_of, chooses, condition, forbidden,
    int_param, int_variable, plain_alternative, plain_variable, space_of, system,
};

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Space>();
    assert_send_sync::<Choice>();
    assert_send_sync::<Configuration>();
    assert_send_sync::<Condition>();
    assert_send_sync::<Forbidden>();
    assert_send_sync::<SpaceError>();
};

/// Return the error `Space::new` refuses its arguments with.
fn refuse_space(
    name: &Identifier,
    variables: Vec<Part<dyn Variable>>,
    choices: Vec<Choice>,
    conditions: Vec<Condition>,
    forbidden: Vec<Forbidden>,
) -> SpaceError {
    match Space::new(name.clone(), variables, choices, conditions, forbidden) {
        Ok(space) => panic!("expected a refusal, got {space:?}"),
        Err(error) => error,
    }
}

/// Return the names of `space`'s decisions, in canonical order.
fn list_decision_names(space: &Space) -> Vec<Identifier> {
    space
        .decisions()
        .map(|decision| decision.name().clone())
        .collect()
}

/// The names of a hierarchical space: top-level variables `v1`, `v2`;
/// choice `c1` with alternative `a1` holding variables `x1`, `x2` and the
/// sub-choice `s` (alternative `b1` holding `y1`), and alternative `a2`
/// holding `z1`; choice `c2` with the bare alternative `d1`.
struct Names {
    space: Identifier,
    v1: Identifier,
    v2: Identifier,
    c1: Identifier,
    a1: Identifier,
    x1: Identifier,
    x2: Identifier,
    s: Identifier,
    b1: Identifier,
    y1: Identifier,
    a2: Identifier,
    z1: Identifier,
    c2: Identifier,
    d1: Identifier,
}

impl Names {
    /// Return fresh names.
    fn new() -> Self {
        Self {
            space: Identifier::new("space"),
            v1: Identifier::new("v1"),
            v2: Identifier::new("v2"),
            c1: Identifier::new("c1"),
            a1: Identifier::new("a1"),
            x1: Identifier::new("x1"),
            x2: Identifier::new("x2"),
            s: Identifier::new("s"),
            b1: Identifier::new("b1"),
            y1: Identifier::new("y1"),
            a2: Identifier::new("a2"),
            z1: Identifier::new("z1"),
            c2: Identifier::new("c2"),
            d1: Identifier::new("d1"),
        }
    }
}

/// A hierarchical space and its names.
struct Hierarchy {
    space: Space,
    n: Names,
}

/// Return the hierarchy, its space built with the conditions and the
/// forbidden clauses the closures build over its names.
fn build_hierarchy_with(
    conditions: impl FnOnce(&Names) -> Vec<Condition>,
    forbidden_clauses: impl FnOnce(&Names) -> Vec<Forbidden>,
) -> Result<Hierarchy, SpaceError> {
    let n = Names::new();
    let conditions = conditions(&n);
    let forbidden_clauses = forbidden_clauses(&n);
    let sub = choice_of(
        &n.s,
        vec![plain_alternative(
            &n.b1,
            vec![int_variable(&n.y1, &[1, 2])],
            Vec::new(),
        )],
    );
    let c1 = choice_of(
        &n.c1,
        vec![
            plain_alternative(
                &n.a1,
                vec![
                    int_variable(&n.x1, &[1, 2, 3]),
                    int_variable(&n.x2, &[1, 2]),
                ],
                vec![sub],
            ),
            plain_alternative(&n.a2, vec![int_variable(&n.z1, &[1, 2])], Vec::new()),
        ],
    );
    let c2 = choice_of(&n.c2, vec![bare_alternative(&n.d1)]);
    let space = Space::new(
        n.space.clone(),
        vec![
            int_variable(&n.v1, &[1, 2, 3]),
            int_variable(&n.v2, &[1, 2, 3]),
        ],
        vec![c1, c2],
        conditions,
        forbidden_clauses,
    )?;
    Ok(Hierarchy { space, n })
}

/// Return the hierarchy with no condition or forbidden clause.
fn build_hierarchy() -> Hierarchy {
    build_hierarchy_with(|_| Vec::new(), |_| Vec::new()).expect("the hierarchy is valid")
}

/// Return the error the hierarchy's space is refused with, given
/// `conditions` and `forbidden_clauses`.
fn refuse_hierarchy(
    conditions: impl FnOnce(&Names) -> Vec<Condition>,
    forbidden_clauses: impl FnOnce(&Names) -> Vec<Forbidden>,
) -> SpaceError {
    match build_hierarchy_with(conditions, forbidden_clauses) {
        Ok(hierarchy) => panic!("expected a refusal, got {:?}", hierarchy.space),
        Err(error) => error,
    }
}

// ---------------------------------------------------------------------------
// Construction and accessors
// ---------------------------------------------------------------------------

#[test]
fn space_new_keeps_its_parts_in_the_order_given() {
    let (name, x, y, c, a) = (
        Identifier::new("space"),
        Identifier::new("x"),
        Identifier::new("y"),
        Identifier::new("c"),
        Identifier::new("a"),
    );
    let variables = vec![int_variable(&y, &[1]), int_variable(&x, &[1])];
    let choices = vec![choice_of(&c, vec![bare_alternative(&a)])];
    let clause = forbidden([in_set(&x, [int(1)])]);

    let space = Space::new(
        name.clone(),
        variables.clone(),
        choices.clone(),
        Vec::new(),
        vec![clause.clone()],
    )
    .expect("the space is valid");

    assert_eq!(space.name(), &name);
    assert_eq!(space.variables(), variables.as_slice());
    assert_eq!(space.choices(), choices.as_slice());
    assert_eq!(space.conditions().len(), 0);
    assert_eq!(space.forbidden(), [clause].as_slice());
    assert_eq!(space.notes().len(), 0);
}

#[test]
fn space_new_accepts_a_space_with_no_decision() {
    let space = space_of(&Identifier::new("empty"), Vec::new(), Vec::new());

    assert_eq!(space.decisions().len(), 0);
    assert_eq!(space.decision_order().len(), 0);
}

#[test]
fn space_with_notes_replaces_its_notes() {
    let space = space_of(&Identifier::new("noted"), Vec::new(), Vec::new());
    let notes = vec![
        Note::with_other_kind("first"),
        Note::with_other_kind("second"),
    ];

    let annotated = space.clone().with_notes(notes.clone());

    assert_eq!(annotated.notes(), notes.as_slice());
    assert_eq!(space.notes().len(), 0, "the original keeps its notes");
}

#[test]
fn condition_and_forbidden_keep_their_parts() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let when = system([in_set(&x, [int(1)])]);

    let guard = Condition::new(y.clone(), when.clone());
    let clause = Forbidden::new(when.clone());

    assert_eq!(guard.target(), &y);
    assert_eq!(guard.when(), &when);
    assert_eq!(clause.when(), &when);
}

// ---------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------

#[rstest]
#[case::space_name_and_a_variable("space_and_variable")]
#[case::two_top_level_variables("two_variables")]
#[case::a_variable_and_a_choice("variable_and_choice")]
#[case::a_variable_and_an_alternative("variable_and_alternative")]
#[case::a_variable_and_a_nested_variable("variable_and_nested_variable")]
#[case::a_variable_and_a_deep_name("variable_and_deep_name")]
#[case::alternatives_of_two_choices("alternatives_of_two_choices")]
#[case::a_variable_and_a_bound_identifier("variable_and_bound_identifier")]
fn space_new_refuses_a_name_used_twice(#[case] shape: &str) {
    let repeated = Identifier::new("repeated");
    let other = Identifier::new("other");
    let (space, variables, choices) = match shape {
        "space_and_variable" => (
            repeated.clone(),
            vec![int_variable(&repeated, &[1])],
            vec![],
        ),
        "two_variables" => (
            other.clone(),
            vec![int_variable(&repeated, &[1]), int_variable(&repeated, &[2])],
            vec![],
        ),
        "variable_and_choice" => (
            other.clone(),
            vec![int_variable(&repeated, &[1])],
            vec![choice_of(
                &repeated,
                vec![bare_alternative(&Identifier::new("a"))],
            )],
        ),
        "variable_and_alternative" => (
            other.clone(),
            vec![int_variable(&repeated, &[1])],
            vec![choice_of(
                &Identifier::new("c"),
                vec![bare_alternative(&repeated)],
            )],
        ),
        "variable_and_nested_variable" => (
            other.clone(),
            vec![int_variable(&repeated, &[1])],
            vec![choice_of(
                &Identifier::new("c"),
                vec![plain_alternative(
                    &Identifier::new("a"),
                    vec![int_variable(&repeated, &[1])],
                    vec![],
                )],
            )],
        ),
        "variable_and_deep_name" => (
            other.clone(),
            vec![int_variable(&repeated, &[1])],
            vec![choice_of(
                &Identifier::new("c"),
                vec![plain_alternative(
                    &Identifier::new("a"),
                    vec![],
                    vec![choice_of(
                        &Identifier::new("s"),
                        vec![bare_alternative(&repeated)],
                    )],
                )],
            )],
        ),
        "alternatives_of_two_choices" => (
            other.clone(),
            vec![],
            vec![
                choice_of(&Identifier::new("c1"), vec![bare_alternative(&repeated)]),
                choice_of(&Identifier::new("c2"), vec![bare_alternative(&repeated)]),
            ],
        ),
        "variable_and_bound_identifier" => (
            other.clone(),
            vec![int_variable(&repeated, &[1])],
            vec![choice_of(
                &Identifier::new("c"),
                vec![
                    Realization::new(&Identifier::new("a"), vec![], &[&repeated], &[], 0)
                        .into_part(),
                ],
            )],
        ),
        _ => unreachable!("unknown shape {shape}"),
    };

    let error = refuse_space(&space, variables, choices, Vec::new(), Vec::new());

    let SpaceError::DuplicateName { name } = &error else {
        panic!("expected DuplicateName, got {error:?}");
    };
    assert_eq!(name, &repeated);
}

#[test]
fn space_new_names_the_first_repeat_in_canonical_order() {
    let (first, second) = (Identifier::new("first"), Identifier::new("second"));
    let c = Identifier::new("c");
    let variables = vec![
        int_variable(&second, &[1]),
        int_variable(&first, &[1]),
        int_variable(&second, &[2]),
    ];
    let choices = vec![choice_of(&c, vec![bare_alternative(&first)])];

    let error = refuse_space(
        &Identifier::new("space"),
        variables,
        choices,
        vec![],
        vec![],
    );

    let SpaceError::DuplicateName { name } = &error else {
        panic!("expected DuplicateName, got {error:?}");
    };
    assert_eq!(
        name, &second,
        "the top-level variables come before the choices"
    );
}

#[test]
fn space_new_reads_a_param_variable_as_no_name_of_the_space() {
    let x = Identifier::new("x");
    let param = categorical_where(vec![int(1), int(2)], |_| Vec::new());
    let shared = param.variable().clone();
    let variables = vec![
        int_variable(&x, &[1]),
        plain_variable(&Identifier::new("y"), param.clone()),
        plain_variable(&Identifier::new("z"), param),
    ];

    let space = Space::new(Identifier::new("space"), variables, vec![], vec![], vec![])
        .expect("two variables may share a param, and its variable is no name of the space");

    assert!(space.decision(&shared).is_none());
}

// ---------------------------------------------------------------------------
// Decisions and their orders
// ---------------------------------------------------------------------------

#[test]
fn space_decisions_walk_the_hierarchy_in_canonical_order() {
    let h = build_hierarchy();

    let names = list_decision_names(&h.space);

    assert_eq!(
        names,
        [
            &h.n.v1, &h.n.v2, &h.n.c1, &h.n.x1, &h.n.x2, &h.n.s, &h.n.y1, &h.n.z1, &h.n.c2
        ]
        .map(Clone::clone)
        .to_vec()
    );
    assert_eq!(h.space.decisions().len(), 9);
}

#[test]
fn space_decision_finds_variables_and_choices_at_every_depth() {
    let h = build_hierarchy();

    let Some(Decision::Variable(variable)) = h.space.decision(&h.n.y1) else {
        panic!("expected the variable y1");
    };
    let Some(Decision::Choice(choice)) = h.space.decision(&h.n.s) else {
        panic!("expected the choice s");
    };

    assert_eq!(variable.get().name(), &h.n.y1);
    assert_eq!(choice.name(), &h.n.s);
    assert_eq!(choice.alternatives().len(), 1);
}

#[rstest]
#[case::an_alternative("alternative")]
#[case::a_nested_alternative("nested_alternative")]
#[case::the_space_itself("space")]
#[case::an_unknown_name("unknown")]
fn space_decision_answers_none_for_a_name_that_is_no_decision(#[case] which: &str) {
    let h = build_hierarchy();
    let name = match which {
        "alternative" => h.n.a1.clone(),
        "nested_alternative" => h.n.b1.clone(),
        "space" => h.space.name().clone(),
        "unknown" => Identifier::new("unknown"),
        _ => unreachable!("unknown case {which}"),
    };

    assert!(h.space.decision(&name).is_none());
}

#[test]
fn decision_name_is_the_variable_or_choice_name() {
    let h = build_hierarchy();

    let names: Vec<_> = [&h.n.v1, &h.n.c2]
        .into_iter()
        .map(|name| h.space.decision(name).expect("a decision").name().clone())
        .collect();

    assert_eq!(names, vec![h.n.v1.clone(), h.n.c2.clone()]);
}

#[test]
fn space_decision_order_is_canonical_without_conditions() {
    let h = build_hierarchy();

    let order = h.space.decision_order();

    assert_eq!(order, list_decision_names(&h.space).as_slice());
}

#[test]
fn space_decision_order_puts_a_decision_after_those_its_condition_names() {
    let h = build_hierarchy_with(
        |n| vec![condition(&n.v1, [in_set(&n.z1, [int(1)])])],
        |_| Vec::new(),
    )
    .expect("v1 may depend on z1");

    let order = h.space.decision_order();

    assert_eq!(
        order,
        [
            &h.n.v2, &h.n.c1, &h.n.x1, &h.n.x2, &h.n.s, &h.n.y1, &h.n.z1, &h.n.v1, &h.n.c2
        ]
        .map(Clone::clone)
        .as_slice(),
        "v1 waits for z1, and every other decision keeps its canonical place"
    );
}

#[test]
fn space_decision_order_takes_the_ready_decision_first_in_canonical_order() {
    let h = build_hierarchy_with(
        |n| vec![condition(&n.v1, [in_set(&n.v2, [int(1)])])],
        |_| Vec::new(),
    )
    .expect("v1 may depend on v2");

    let order = h.space.decision_order();

    assert_eq!(
        order[..3],
        [h.n.v2.clone(), h.n.v1.clone(), h.n.c1.clone()],
        "v1 is ready once v2 is placed, and comes before c1"
    );
}

// ---------------------------------------------------------------------------
// Conditions
// ---------------------------------------------------------------------------

#[test]
fn space_new_accepts_a_condition_on_a_choice_through_a_set_constraint() {
    let h = build_hierarchy_with(
        |n| vec![condition(&n.v1, [chooses(&n.c1, &[&n.a1])])],
        |_| Vec::new(),
    );

    let h = h.expect("a set constraint may name a choice");
    assert_eq!(h.space.conditions().len(), 1);
}

#[test]
fn space_new_accepts_an_equation_over_variables() {
    let h = build_hierarchy_with(
        |n| vec![condition(&n.v1, [less_than(&n.v2, &n.z1)])],
        |_| Vec::new(),
    );

    h.expect("an equation may name variables");
}

#[test]
fn space_new_accepts_a_condition_on_a_nested_decision() {
    let h = build_hierarchy_with(
        |n| vec![condition(&n.y1, [in_set(&n.v1, [int(2)])])],
        |_| Vec::new(),
    );

    let h = h.expect("a nested decision may have a condition");
    assert_eq!(h.space.conditions()[0].target(), &h.n.y1);
}

#[rstest]
#[case::an_alternative("alternative")]
#[case::the_space_itself("space")]
#[case::an_unknown_name("unknown")]
fn space_new_refuses_a_condition_on_a_name_that_is_no_decision(#[case] which: &str) {
    let mut target = None;
    let error = refuse_hierarchy(
        |n| {
            let name = match which {
                "alternative" => n.a2.clone(),
                "space" => n.space.clone(),
                "unknown" => Identifier::new("unknown"),
                _ => unreachable!("unknown case {which}"),
            };
            target = Some(name.clone());
            vec![condition(&name, [in_set(&n.v2, [int(1)])])]
        },
        |_| Vec::new(),
    );

    let SpaceError::UnknownConditionTarget { target: reported } = &error else {
        panic!("expected UnknownConditionTarget, got {error:?}");
    };
    assert_eq!(Some(reported), target.as_ref());
}

#[rstest]
#[case::an_unknown_identifier("unknown")]
#[case::an_alternative_as_a_variable("alternative")]
#[case::the_space_name("space")]
fn space_new_refuses_a_condition_naming_no_decision(#[case] which: &str) {
    let mut referenced = None;
    let error = refuse_hierarchy(
        |n| {
            let name = match which {
                "unknown" => Identifier::new("unknown"),
                "alternative" => n.a1.clone(),
                "space" => n.space.clone(),
                _ => unreachable!("unknown case {which}"),
            };
            referenced = Some(name.clone());
            vec![condition(&n.v1, [in_set(&name, [int(1)])])]
        },
        |_| Vec::new(),
    );

    let SpaceError::UnknownReference { name } = &error else {
        panic!("expected UnknownReference, got {error:?}");
    };
    assert_eq!(Some(name), referenced.as_ref());
}

#[rstest]
#[case::no_constraint("empty")]
#[case::a_closed_equation("closed")]
fn space_new_refuses_a_condition_naming_nothing(#[case] which: &str) {
    let mut expected = None;
    let error = refuse_hierarchy(
        |n| {
            expected = Some(n.v2.clone());
            let constraints = match which {
                "empty" => Vec::new(),
                "closed" => vec![Constraint::from(EquationConstraint::new(
                    literal(5).equals(literal(5)),
                ))],
                _ => unreachable!("unknown case {which}"),
            };
            vec![
                condition(&n.v1, [in_set(&n.v2, [int(1)])]),
                condition(&n.v2, constraints),
            ]
        },
        |_| Vec::new(),
    );

    let SpaceError::EmptyCondition { target } = &error else {
        panic!("expected EmptyCondition, got {error:?}");
    };
    assert_eq!(Some(target), expected.as_ref());
}

#[test]
fn space_new_checks_a_condition_target_before_whether_it_names_anything() {
    let error = refuse_hierarchy(
        |_| vec![condition(&Identifier::new("no_target"), [])],
        |_| Vec::new(),
    );

    assert!(
        matches!(error, SpaceError::UnknownConditionTarget { .. }),
        "got {error:?}"
    );
}

#[test]
fn space_new_reports_the_unknown_reference_with_the_smallest_id_first() {
    let mut expected = None;
    let error = refuse_hierarchy(
        |n| {
            let (low, high) = (Identifier::new("low"), Identifier::new("high"));
            expected = Some(low.clone());
            vec![condition(
                &n.v1,
                [in_set(&high, [int(1)]), in_set(&low, [int(1)])],
            )]
        },
        |_| Vec::new(),
    );

    let SpaceError::UnknownReference { name } = &error else {
        panic!("expected UnknownReference, got {error:?}");
    };
    assert_eq!(Some(name), expected.as_ref());
}

#[test]
fn space_new_reads_a_member_identifier_as_no_reference() {
    let h = build_hierarchy_with(
        |n| {
            let stranger = Identifier::new("stranger");
            vec![condition(
                &n.v1,
                [in_set(&n.v2, [Value::Identifier(stranger)])],
            )]
        },
        |_| Vec::new(),
    );

    h.expect("a member is a constant, not a reference to a decision");
}

#[test]
fn space_new_refuses_an_equation_naming_a_choice_in_a_condition() {
    let mut choice = None;
    let error = refuse_hierarchy(
        |n| {
            choice = Some(n.c2.clone());
            vec![condition(&n.v1, [at_least(&n.c2, 1)])]
        },
        |_| Vec::new(),
    );

    let SpaceError::EquationOverChoice { choice: reported } = &error else {
        panic!("expected EquationOverChoice, got {error:?}");
    };
    assert_eq!(Some(reported), choice.as_ref());
}

#[rstest]
#[case::a_misspelled_name_in_a_condition("misspelled", true)]
#[case::another_choices_alternative_in_a_condition("other_choice", true)]
#[case::a_number_in_a_forbidden_clause("number", false)]
fn space_new_refuses_a_set_constraint_over_a_choice_naming_no_alternative_of_it(
    #[case] which: &str,
    #[case] is_condition: bool,
) {
    let mut expected = None;
    let mut build = |n: &Names| {
        let member = match which {
            "misspelled" => Value::Identifier(Identifier::new("bogus")),
            "other_choice" => Value::Identifier(n.a1.clone()),
            "number" => int(1),
            _ => unreachable!("unknown case {which}"),
        };
        expected = Some((n.c2.clone(), member.clone()));
        in_set(&n.c2, [member])
    };
    let error = if is_condition {
        refuse_hierarchy(|n| vec![condition(&n.v1, [build(n)])], |_| Vec::new())
    } else {
        refuse_hierarchy(|_| Vec::new(), |n| vec![forbidden([build(n)])])
    };

    let SpaceError::UnknownAlternative { choice, value } = &error else {
        panic!("expected UnknownAlternative, got {error:?}");
    };
    let (expected_choice, expected_value) = expected.expect("the closure ran");
    assert_eq!((choice, value), (&expected_choice, &expected_value));
}

#[rstest]
#[case::the_target_itself("itself")]
#[case::a_variable_under_the_target("child")]
#[case::a_choice_under_the_target("sub_choice")]
#[case::a_variable_two_levels_down("grandchild")]
fn space_new_refuses_a_condition_naming_its_own_subtree(#[case] which: &str) {
    let mut expected = None;
    let error = refuse_hierarchy(
        |n| {
            let named = match which {
                "itself" => n.c1.clone(),
                "child" => n.x1.clone(),
                "sub_choice" => n.s.clone(),
                "grandchild" => n.y1.clone(),
                _ => unreachable!("unknown case {which}"),
            };
            expected = Some((n.c1.clone(), named.clone()));
            let constraint = if which == "sub_choice" {
                chooses(&named, &[&n.b1])
            } else if which == "itself" {
                chooses(&named, &[&n.a1])
            } else {
                in_set(&named, [int(1)])
            };
            vec![condition(&n.c1, [constraint])]
        },
        |_| Vec::new(),
    );

    let SpaceError::ConditionReferencesSubtree { target, name } = &error else {
        panic!("expected ConditionReferencesSubtree, got {error:?}");
    };
    let (expected_target, expected_name) = expected.expect("the closure ran");
    assert_eq!((target, name), (&expected_target, &expected_name));
}

#[test]
fn space_new_refuses_a_variable_conditioned_on_itself() {
    let mut expected = None;
    let error = refuse_hierarchy(
        |n| {
            expected = Some(n.v1.clone());
            vec![condition(&n.v1, [in_set(&n.v1, [int(1)])])]
        },
        |_| Vec::new(),
    );

    let SpaceError::ConditionReferencesSubtree { target, name } = &error else {
        panic!("expected ConditionReferencesSubtree, got {error:?}");
    };
    assert_eq!(Some(target), expected.as_ref());
    assert_eq!(target, name);
}

#[test]
fn space_new_reports_a_custom_constraint_whose_scope_fails() {
    let error = refuse_hierarchy(
        |n| {
            vec![Condition::new(
                n.v1.clone(),
                system([Failing(FailingHook::Scope).into_constraint()]),
            )]
        },
        |_| Vec::new(),
    );

    assert!(
        matches!(error, SpaceError::Constraint(ConstraintError::Custom(_))),
        "got {error:?}"
    );
}

#[test]
fn space_conditions_conjoin_the_conditions_on_one_target() {
    let h = build_hierarchy_with(
        |n| {
            vec![
                condition(&n.v1, [in_set(&n.v2, [int(1), int(2)])]),
                condition(&n.v1, [chooses(&n.c2, &[&n.d1])]),
            ]
        },
        |_| Vec::new(),
    )
    .expect("two conditions may share a target");

    let conditions = h.space.conditions();

    assert_eq!(conditions.len(), 1);
    assert_eq!(conditions[0].target(), &h.n.v1);
    assert_eq!(
        conditions[0].when(),
        &system([
            in_set(&h.n.v2, [int(1), int(2)]),
            chooses(&h.n.c2, &[&h.n.d1]),
        ])
    );
}

#[test]
fn space_conditions_follow_the_canonical_order_of_their_targets() {
    let h = build_hierarchy_with(
        |n| {
            vec![
                condition(&n.c2, [in_set(&n.v1, [int(1)])]),
                condition(&n.y1, [in_set(&n.v1, [int(2)])]),
                condition(&n.v2, [in_set(&n.v1, [int(3)])]),
            ]
        },
        |_| Vec::new(),
    )
    .expect("the conditions are valid");

    let targets: Vec<_> = h
        .space
        .conditions()
        .iter()
        .map(|guard| guard.target().clone())
        .collect();

    assert_eq!(
        targets,
        vec![h.n.v2.clone(), h.n.y1.clone(), h.n.c2.clone()]
    );
}

// ---------------------------------------------------------------------------
// Forbidden clauses
// ---------------------------------------------------------------------------

#[test]
fn space_new_refuses_a_forbidden_clause_naming_no_decision() {
    let error = refuse_hierarchy(
        |_| Vec::new(),
        |n| {
            vec![
                forbidden([in_set(&n.v1, [int(1)])]),
                Forbidden::new(system([])),
            ]
        },
    );

    let SpaceError::EmptyForbidden { index } = &error else {
        panic!("expected EmptyForbidden, got {error:?}");
    };
    assert_eq!(*index, 1);
}

#[test]
fn space_new_refuses_a_forbidden_clause_naming_an_unknown_identifier() {
    let mut expected = None;
    let error = refuse_hierarchy(
        |_| Vec::new(),
        |n| {
            let name = n.b1.clone();
            expected = Some(name.clone());
            vec![forbidden([
                in_set(&n.v1, [int(1)]),
                in_set(&name, [int(1)]),
            ])]
        },
    );

    let SpaceError::UnknownReference { name } = &error else {
        panic!("expected UnknownReference, got {error:?}");
    };
    assert_eq!(Some(name), expected.as_ref());
}

#[test]
fn space_new_refuses_an_equation_naming_a_choice_in_a_forbidden_clause() {
    let mut expected = None;
    let error = refuse_hierarchy(
        |_| Vec::new(),
        |n| {
            expected = Some(n.s.clone());
            vec![forbidden([
                in_set(&n.v1, [int(1)]),
                Constraint::from(EquationConstraint::new(
                    Expression::from(&n.s).equals(Expression::from(&n.v2)),
                )),
            ])]
        },
    );

    let SpaceError::EquationOverChoice { choice } = &error else {
        panic!("expected EquationOverChoice, got {error:?}");
    };
    assert_eq!(Some(choice), expected.as_ref());
}

#[test]
fn space_new_accepts_a_forbidden_clause_across_levels() {
    let h = build_hierarchy_with(
        |_| Vec::new(),
        |n| {
            vec![forbidden([
                chooses(&n.c1, &[&n.a1]),
                in_set(&n.y1, [int(2)]),
                in_set(&n.v1, [int(3)]),
            ])]
        },
    );

    let h = h.expect("a clause may name decisions at any depth");
    assert_eq!(h.space.forbidden().len(), 1);
}

// ---------------------------------------------------------------------------
// Cycles
// ---------------------------------------------------------------------------

#[test]
fn space_new_refuses_two_decisions_conditioned_on_each_other() {
    let mut expected = None;
    let error = refuse_hierarchy(
        |n| {
            expected = Some(vec![n.v1.clone(), n.v2.clone()]);
            vec![
                condition(&n.v2, [in_set(&n.v1, [int(1)])]),
                condition(&n.v1, [in_set(&n.v2, [int(1)])]),
            ]
        },
        |_| Vec::new(),
    );

    let SpaceError::CyclicDependency { cycle } = &error else {
        panic!("expected CyclicDependency, got {error:?}");
    };
    assert_eq!(Some(cycle), expected.as_ref());
}

#[test]
fn space_new_refuses_a_cycle_through_a_choice_and_its_child() {
    let mut expected = None;
    let error = refuse_hierarchy(
        |n| {
            expected = Some(vec![n.v2.clone(), n.c1.clone(), n.x1.clone()]);
            vec![
                condition(&n.c1, [in_set(&n.v2, [int(1)])]),
                condition(&n.v2, [in_set(&n.x1, [int(1)])]),
            ]
        },
        |_| Vec::new(),
    );

    let SpaceError::CyclicDependency { cycle } = &error else {
        panic!("expected CyclicDependency, got {error:?}");
    };
    assert_eq!(
        Some(cycle),
        expected.as_ref(),
        "the cycle starts at its decision first in canonical order, each next one depending on \
         the one before"
    );
}

// ---------------------------------------------------------------------------
// The order of the checks
// ---------------------------------------------------------------------------

#[test]
fn space_new_checks_names_before_conditions() {
    let x = Identifier::new("x");

    let error = refuse_space(
        &Identifier::new("space"),
        vec![int_variable(&x, &[1]), int_variable(&x, &[1])],
        Vec::new(),
        vec![condition(
            &Identifier::new("unknown"),
            [in_set(&x, [int(1)])],
        )],
        Vec::new(),
    );

    assert!(
        matches!(error, SpaceError::DuplicateName { .. }),
        "got {error:?}"
    );
}

#[test]
fn space_new_checks_a_condition_target_before_its_references() {
    let error = refuse_hierarchy(
        |_| {
            vec![condition(
                &Identifier::new("no_target"),
                [in_set(&Identifier::new("no_reference"), [int(1)])],
            )]
        },
        |_| Vec::new(),
    );

    assert!(
        matches!(error, SpaceError::UnknownConditionTarget { .. }),
        "got {error:?}"
    );
}

#[test]
fn space_new_checks_conditions_in_the_order_given() {
    let mut expected = None;
    let error = refuse_hierarchy(
        |n| {
            let (first, second) = (Identifier::new("first"), Identifier::new("second"));
            expected = Some(second.clone());
            vec![
                condition(&n.v1, [in_set(&second, [int(1)])]),
                condition(&first, [in_set(&n.v2, [int(1)])]),
            ]
        },
        |_| Vec::new(),
    );

    let SpaceError::UnknownReference { name } = &error else {
        panic!("expected the first condition's UnknownReference, got {error:?}");
    };
    assert_eq!(Some(name), expected.as_ref());
}

#[test]
fn space_new_checks_conditions_before_forbidden_clauses() {
    let error = refuse_hierarchy(
        |n| {
            vec![condition(
                &n.v1,
                [in_set(&Identifier::new("unknown"), [int(1)])],
            )]
        },
        |_| vec![Forbidden::new(system([]))],
    );

    assert!(
        matches!(error, SpaceError::UnknownReference { .. }),
        "got {error:?}"
    );
}

#[test]
fn space_new_checks_forbidden_clauses_before_cycles() {
    let error = refuse_hierarchy(
        |n| {
            vec![
                condition(&n.v2, [in_set(&n.v1, [int(1)])]),
                condition(&n.v1, [in_set(&n.v2, [int(1)])]),
            ]
        },
        |_| vec![Forbidden::new(system([]))],
    );

    assert!(
        matches!(error, SpaceError::EmptyForbidden { index: 0 }),
        "got {error:?}"
    );
}

// ---------------------------------------------------------------------------
// Equality, hashing and structural equivalence
// ---------------------------------------------------------------------------

/// Return a space of `names` rebuilt from scratch, its variables over
/// `param`, with one condition, one forbidden clause, and `notes`.
fn build_guarded(names: &[Identifier; 6], param: &Param, notes: &[&str]) -> Space {
    let [space, v, c, a, b, x] = names;
    let choice = choice_of(
        c,
        vec![
            plain_alternative(a, vec![plain_variable(x, param.clone())], Vec::new()),
            bare_alternative(b),
        ],
    );
    Space::new(
        space.clone(),
        vec![plain_variable(v, param.clone())],
        vec![choice],
        vec![condition(v, [chooses(c, &[a])])],
        vec![forbidden([in_set(x, [int(2)]), in_set(v, [int(2)])])],
    )
    .expect("the space is valid")
    .with_notes(
        notes
            .iter()
            .map(|&note| Note::with_other_kind(note))
            .collect(),
    )
}

/// Return the six names of a guarded space.
fn build_guarded_names() -> [Identifier; 6] {
    ["space", "v", "c", "a", "b", "x"].map(Identifier::new)
}

#[test]
fn spaces_built_apart_from_equal_parts_are_equal_and_hash_alike() {
    let names = build_guarded_names();
    let param = int_param(&[1, 2]);

    let left = build_guarded(&names, &param, &["note"]);
    let right = build_guarded(&names, &param, &["note"]);

    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
    assert!(
        left.is_structurally_equivalent(&right)
            .expect("plain parts")
    );
}

#[test]
fn space_clone_is_equal_and_structurally_equivalent() {
    let space = build_guarded(&build_guarded_names(), &int_param(&[1, 2]), &[]);

    let copy = space.clone();

    assert_eq!(copy, space);
    assert!(
        copy.is_structurally_equivalent(&space)
            .expect("plain parts")
    );
}

#[rstest]
#[case::another_name("name")]
#[case::another_variable_name("variable")]
#[case::another_alternative_name("alternative")]
#[case::other_notes("notes")]
fn spaces_differing_in_one_name_are_unequal_and_not_structurally_equivalent(#[case] field: &str) {
    let names = build_guarded_names();
    let param = int_param(&[1, 2]);
    let mut changed = names.clone();
    let mut notes = ["note"];
    match field {
        "name" => changed[0] = Identifier::new("space"),
        "variable" => changed[1] = Identifier::new("v"),
        "alternative" => changed[4] = Identifier::new("b"),
        "notes" => notes = ["another note"],
        _ => unreachable!("unknown field {field}"),
    }

    let left = build_guarded(&names, &param, &["note"]);
    let right = build_guarded(&changed, &param, &notes);

    assert_ne!(left, right);
    assert!(
        !left
            .is_structurally_equivalent(&right)
            .expect("plain parts")
    );
    assert!(
        !right
            .is_structurally_equivalent(&left)
            .expect("plain parts")
    );
}

#[test]
fn spaces_differing_in_a_condition_are_unequal_and_not_structurally_equivalent() {
    let param = int_param(&[1, 2, 3]);
    let [space, v, c, a, b, x] = build_guarded_names();
    let build = |chosen: &Identifier| {
        let choice = choice_of(
            &c,
            vec![
                plain_alternative(&a, vec![plain_variable(&x, param.clone())], Vec::new()),
                bare_alternative(&b),
            ],
        );
        Space::new(
            space.clone(),
            vec![plain_variable(&v, param.clone())],
            vec![choice],
            vec![condition(&v, [chooses(&c, &[chosen])])],
            Vec::new(),
        )
        .expect("the space is valid")
    };

    let (left, right) = (build(&a), build(&b));

    assert_ne!(left, right);
    assert!(
        !left
            .is_structurally_equivalent(&right)
            .expect("plain parts")
    );
}

#[test]
fn spaces_differing_in_a_forbidden_clause_are_unequal_and_not_structurally_equivalent() {
    let param = int_param(&[1, 2, 3]);
    let [space, v, ..] = build_guarded_names();
    let build = |value: i64| {
        Space::new(
            space.clone(),
            vec![plain_variable(&v, param.clone())],
            Vec::new(),
            Vec::new(),
            vec![forbidden([in_set(&v, [int(value)])])],
        )
        .expect("the space is valid")
    };

    let (left, right) = (build(1), build(2));

    assert_ne!(left, right);
    assert!(
        !left
            .is_structurally_equivalent(&right)
            .expect("plain parts")
    );
}

#[test]
fn spaces_given_their_conditions_in_another_order_are_equal() {
    let param = int_param(&[1, 2, 3]);
    let n = Names::new();
    let build = |reversed: bool| {
        let mut conditions = vec![
            condition(&n.v1, [in_set(&n.v2, [int(1)])]),
            condition(&n.c2, [in_set(&n.v1, [int(2)])]),
        ];
        if reversed {
            conditions.reverse();
        }
        Space::new(
            n.space.clone(),
            vec![
                plain_variable(&n.v1, param.clone()),
                plain_variable(&n.v2, param.clone()),
            ],
            vec![choice_of(&n.c2, vec![bare_alternative(&n.d1)])],
            conditions,
            Vec::new(),
        )
        .expect("the space is valid")
    };

    let (left, right) = (build(false), build(true));

    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
}

#[test]
fn space_built_with_an_undecidable_custom_condition_is_accepted() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let h = build_hierarchy_with(
        |n| {
            vec![condition(
                &n.v1,
                [TestCustom::build(
                    "custom",
                    Expression::from(&n.v2),
                    Outcome::Undecided,
                    &log,
                )],
            )]
        },
        |_| Vec::new(),
    );

    let h = h.expect("a custom constraint naming a decision is a valid condition");
    assert!(
        log.lock().expect("the log").is_empty(),
        "building evaluates nothing"
    );
    assert_eq!(h.space.conditions().len(), 1);
}
