//! Tests for forbidden clauses in configurations: a clause applies once
//! every decision it names is active and assigned, and then must be
//! violated; a clause naming an inactive decision never applies; and an
//! undecided or failing clause is a problem.

#![expect(
    clippy::many_single_char_names,
    reason = "the stories name decisions as the design's examples do: c, a, b, x, y"
)]

use std::sync::{Arc, Mutex};

use fhy_core::constraint::{ConstraintError, Outcome};
use fhy_core::expression::Expression;
use fhy_core::identifier::Identifier;
use fhy_core::param::ParamContext;
use fhy_core::search_space::{Configuration, ConfigurationError, Space};
use rstest::rstest;

use crate::support::constraint::{TestCustom, int};
use crate::support::param::{at_least, failing_simplifier_solver, in_set, not_in_set};
use crate::support::search_space::{
    bare_alternative, choice_of, chooses, chosen, configure, forbidden, ground_solver,
    int_variable, natural_param, plain_alternative, plain_variable, try_configure,
};

/// A space of the variable `x` over `{1, 2, 3}` and the choice `c` among
/// `a` (holding `y` over `{1, 2}`) and `b`, with one forbidden clause:
/// `x in {1}` together with `y in {2}`.
struct Clause {
    space: Space,
    x: Identifier,
    c: Identifier,
    a: Identifier,
    b: Identifier,
    y: Identifier,
}

/// Return the space with the forbidden clause.
fn build_clause() -> Clause {
    let [x, c, a, b, y] = ["x", "c", "a", "b", "y"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("clause"),
        vec![int_variable(&x, &[1, 2, 3])],
        vec![choice_of(
            &c,
            vec![
                plain_alternative(&a, vec![int_variable(&y, &[1, 2])], Vec::new()),
                bare_alternative(&b),
            ],
        )],
        Vec::new(),
        vec![forbidden([in_set(&x, [int(1)]), in_set(&y, [int(2)])])],
    )
    .expect("the space is valid");
    Clause {
        space,
        x,
        c,
        a,
        b,
        y,
    }
}

#[test]
fn configuration_new_refuses_a_forbidden_combination() {
    let s = build_clause();

    let errors = try_configure(
        &s.space,
        [
            (s.x.clone(), int(1)),
            (s.c.clone(), chosen(&s.a)),
            (s.y.clone(), int(2)),
        ],
    )
    .expect_err("the combination is forbidden");

    let [ConfigurationError::Forbidden { index }] = errors.errors() else {
        panic!("expected one Forbidden, got {errors:?}");
    };
    assert_eq!(*index, 0);
}

#[rstest]
#[case::the_first_member_violated(2, 2)]
#[case::the_second_member_violated(1, 1)]
#[case::both_members_violated(3, 1)]
fn configuration_new_accepts_a_combination_violating_the_clause(
    #[case] x_value: i64,
    #[case] y_value: i64,
) {
    let s = build_clause();

    let configuration = configure(
        &s.space,
        [
            (s.x.clone(), int(x_value)),
            (s.c.clone(), chosen(&s.a)),
            (s.y.clone(), int(y_value)),
        ],
    );

    assert!(configuration.is_complete());
}

#[test]
fn clause_naming_an_inactive_decision_does_not_apply() {
    let s = build_clause();

    let configuration = configure(
        &s.space,
        [(s.x.clone(), int(1)), (s.c.clone(), chosen(&s.b))],
    );

    assert!(
        configuration.is_complete(),
        "y is inactive, so the clause never applies"
    );
}

#[test]
fn clause_naming_an_unassigned_active_decision_does_not_apply_yet() {
    let s = build_clause();

    let configuration = configure(
        &s.space,
        [(s.x.clone(), int(1)), (s.c.clone(), chosen(&s.a))],
    );

    assert!(!configuration.is_complete());
    let completed =
        configuration.with_entry(s.y.clone(), int(2), &ParamContext::new(&ground_solver()));
    let errors = completed.expect_err("completing it takes the forbidden combination");
    assert!(
        matches!(
            errors.errors(),
            [ConfigurationError::Forbidden { index: 0 }]
        ),
        "got {errors:?}"
    );
}

#[test]
fn clause_naming_a_pending_decision_does_not_apply_yet() {
    let s = build_clause();

    let configuration = configure(&s.space, [(s.x.clone(), int(1))]);

    assert_eq!(
        configuration.activity(&s.y),
        Some(fhy_core::search_space::Activity::Pending)
    );
    assert_eq!(configuration.value(&s.x), Some(&int(1)));
}

#[test]
fn clause_over_a_choice_forbids_an_alternative_with_a_value() {
    let [x, c, a, b] = ["x", "c", "a", "b"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("choice_clause"),
        vec![int_variable(&x, &[1, 2])],
        vec![choice_of(
            &c,
            vec![bare_alternative(&a), bare_alternative(&b)],
        )],
        Vec::new(),
        vec![forbidden([chooses(&c, &[&a]), in_set(&x, [int(2)])])],
    )
    .expect("the space is valid");

    let refused = try_configure(&space, [(c.clone(), chosen(&a)), (x.clone(), int(2))]);
    let accepted = configure(&space, [(c.clone(), chosen(&b)), (x.clone(), int(2))]);

    let errors = refused.expect_err("a with x = 2 is forbidden");
    assert!(
        matches!(
            errors.errors(),
            [ConfigurationError::Forbidden { index: 0 }]
        ),
        "got {errors:?}"
    );
    assert_eq!(accepted.value(&c), Some(&chosen(&b)));
}

#[test]
fn positive_rule_is_written_as_a_forbidden_negation() {
    let [x, y] = ["x", "y"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("implication"),
        vec![int_variable(&x, &[1, 2]), int_variable(&y, &[1, 2])],
        Vec::new(),
        Vec::new(),
        vec![forbidden([in_set(&x, [int(1)]), not_in_set(&y, [int(2)])])],
    )
    .expect("the space is valid");

    let keeps_the_rule = configure(&space, [(x.clone(), int(1)), (y.clone(), int(2))]);
    let outside_the_rule = configure(&space, [(x.clone(), int(2)), (y.clone(), int(1))]);
    let breaks_the_rule = try_configure(&space, [(x.clone(), int(1)), (y.clone(), int(1))]);

    assert!(keeps_the_rule.is_complete());
    assert!(outside_the_rule.is_complete());
    let errors = breaks_the_rule.expect_err("x = 1 needs y = 2");
    assert!(
        matches!(
            errors.errors(),
            [ConfigurationError::Forbidden { index: 0 }]
        ),
        "got {errors:?}"
    );
}

#[test]
fn every_holding_clause_is_reported_in_order() {
    let [x, y] = ["x", "y"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("several"),
        vec![int_variable(&x, &[1, 2]), int_variable(&y, &[1, 2])],
        Vec::new(),
        Vec::new(),
        vec![
            forbidden([in_set(&y, [int(1)])]),
            forbidden([in_set(&x, [int(2)])]),
            forbidden([in_set(&x, [int(1)]), in_set(&y, [int(1)])]),
        ],
    )
    .expect("the space is valid");

    let errors = try_configure(&space, [(x.clone(), int(1)), (y.clone(), int(1))])
        .expect_err("two clauses hold");

    let indices: Vec<usize> = errors
        .errors()
        .iter()
        .map(|problem| match problem {
            ConfigurationError::Forbidden { index } => *index,
            other => panic!("expected only Forbidden problems, got {other:?}"),
        })
        .collect();
    assert_eq!(indices, vec![0, 2]);
}

#[test]
fn clause_with_an_equation_over_a_number_is_decided() {
    let [n, m] = ["n", "m"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("numeric_clause"),
        vec![
            plain_variable(&n, natural_param()),
            plain_variable(&m, natural_param()),
        ],
        Vec::new(),
        Vec::new(),
        vec![forbidden([at_least(&n, 5), in_set(&m, [int(0)])])],
    )
    .expect("an equation may name a variable");

    let small = configure(&space, [(n.clone(), int(4)), (m.clone(), int(0))]);
    let large = try_configure(&space, [(n.clone(), int(5)), (m.clone(), int(0))]);

    assert!(small.is_complete());
    let errors = large.expect_err("n >= 5 with m = 0 is forbidden");
    assert!(
        matches!(
            errors.errors(),
            [ConfigurationError::Forbidden { index: 0 }]
        ),
        "got {errors:?}"
    );
}

#[test]
fn undecided_clause_is_a_problem_naming_its_index() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let [x, y] = ["x", "y"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("undecided_clause"),
        vec![int_variable(&x, &[1, 2]), int_variable(&y, &[1, 2])],
        Vec::new(),
        Vec::new(),
        vec![
            forbidden([in_set(&y, [int(2)])]),
            forbidden([TestCustom::build(
                "undecided",
                Expression::from(&x),
                Outcome::Undecided,
                &log,
            )]),
        ],
    )
    .expect("the space is valid");

    let errors = try_configure(&space, [(x.clone(), int(1)), (y.clone(), int(1))])
        .expect_err("an undecided clause is refused");

    assert!(
        matches!(
            errors.errors(),
            [ConfigurationError::UndecidedForbidden { index: 1 }]
        ),
        "got {errors:?}"
    );
}

#[test]
fn undecided_clause_naming_an_unassigned_decision_is_not_evaluated() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let x = Identifier::new("x");
    let space = Space::new(
        Identifier::new("unevaluated_clause"),
        vec![int_variable(&x, &[1, 2])],
        Vec::new(),
        Vec::new(),
        vec![forbidden([TestCustom::build(
            "watched",
            Expression::from(&x),
            Outcome::Undecided,
            &log,
        )])],
    )
    .expect("the space is valid");

    let empty = configure(&space, []);

    assert!(!empty.is_complete());
    assert!(log.lock().expect("the log").is_empty());
}

#[test]
fn failing_clause_is_a_problem_naming_its_index() {
    let n = Identifier::new("n");
    let space = Space::new(
        Identifier::new("failing_clause"),
        vec![plain_variable(&n, natural_param())],
        Vec::new(),
        Vec::new(),
        vec![forbidden([at_least(&n, 5)])],
    )
    .expect("the space is valid");
    let solver = failing_simplifier_solver();

    let result = Configuration::new(&space, [(n.clone(), int(1))], &ParamContext::new(&solver));

    let errors = result.expect_err("the simplifier fails");
    let [ConfigurationError::FailedForbidden { index, error }] = errors.errors() else {
        panic!("expected one FailedForbidden, got {errors:?}");
    };
    assert_eq!(*index, 0);
    assert!(matches!(error, ConstraintError::Solve(_)), "got {error:?}");
}

#[test]
fn clause_is_checked_after_the_decisions_and_with_their_problems() {
    let s = build_clause();

    let errors = try_configure(
        &s.space,
        [
            (s.x.clone(), int(1)),
            (s.c.clone(), chosen(&s.a)),
            (s.y.clone(), int(2)),
            (s.b.clone(), int(1)),
        ],
    )
    .expect_err("an unknown entry and a forbidden combination");

    let [
        ConfigurationError::UnknownDecision { name },
        ConfigurationError::Forbidden { index: 0 },
    ] = errors.errors()
    else {
        panic!("expected UnknownDecision then Forbidden, got {errors:?}");
    };
    assert_eq!(name, &s.b);
}
