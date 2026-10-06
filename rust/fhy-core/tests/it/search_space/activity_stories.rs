//! Tests for the activity of decisions in a configuration: the hierarchy
//! (a decision exists while its choice chooses its alternative), the
//! conditions (a decision exists while its condition holds), and the
//! pending state between them, through `Configuration::activity` and the
//! problems `Configuration::new` reports.

#![expect(
    clippy::many_single_char_names,
    reason = "the stories name decisions as the design's examples do: c, a, b, x, y"
)]

use std::sync::{Arc, Mutex};

use fhy_core::constraint::{ConstraintError, Outcome};
use fhy_core::expression::Expression;
use fhy_core::identifier::Identifier;
use fhy_core::param::ParamContext;
use fhy_core::search_space::{Activity, Configuration, ConfigurationError, Space};
use rstest::rstest;

use crate::support::constraint::{TestCustom, int, text};
use crate::support::param::{at_least, failing_simplifier_solver, in_set, not_in_set};
use crate::support::search_space::{
    bare_alternative, categorical, choice_of, chooses, chosen, condition, configure, int_variable,
    natural_param, plain_alternative, plain_variable, try_configure,
};

/// Return the activity of `name` in `configuration`.
///
/// # Panics
///
/// Panics if the space has no decision `name`.
fn activity_of(configuration: &Configuration, name: &Identifier) -> Activity {
    configuration
        .activity(name)
        .unwrap_or_else(|| panic!("{name:?} is a decision"))
}

/// A space of one choice `c` among `a`, holding `x` and the sub-choice `s`
/// (alternative `b`, holding `y`), and the bare `other`.
struct Nested {
    space: Space,
    c: Identifier,
    a: Identifier,
    other: Identifier,
    x: Identifier,
    s: Identifier,
    b: Identifier,
    y: Identifier,
}

/// Return the nested space.
fn build_nested() -> Nested {
    let names = ["nested", "c", "a", "other", "x", "s", "b", "y"].map(Identifier::new);
    let [space, c, a, other, x, s, b, y] = names;
    let sub = choice_of(
        &s,
        vec![plain_alternative(
            &b,
            vec![int_variable(&y, &[1, 2])],
            Vec::new(),
        )],
    );
    let choice = choice_of(
        &c,
        vec![
            plain_alternative(&a, vec![int_variable(&x, &[1, 2])], vec![sub]),
            bare_alternative(&other),
        ],
    );
    let space = Space::new(space, Vec::new(), vec![choice], Vec::new(), Vec::new())
        .expect("the space is valid");
    Nested {
        space,
        c,
        a,
        other,
        x,
        s,
        b,
        y,
    }
}

// ---------------------------------------------------------------------------
// The hierarchy
// ---------------------------------------------------------------------------

#[test]
fn top_level_decision_without_a_condition_is_active() {
    let (v, c, a) = (
        Identifier::new("v"),
        Identifier::new("c"),
        Identifier::new("a"),
    );
    let space = Space::new(
        Identifier::new("flat"),
        vec![int_variable(&v, &[1])],
        vec![choice_of(&c, vec![bare_alternative(&a)])],
        Vec::new(),
        Vec::new(),
    )
    .expect("the space is valid");

    let empty = configure(&space, []);

    assert_eq!(activity_of(&empty, &v), Activity::Active);
    assert_eq!(activity_of(&empty, &c), Activity::Active);
}

#[test]
fn decision_under_an_unassigned_choice_is_pending() {
    let n = build_nested();

    let empty = configure(&n.space, []);

    assert_eq!(activity_of(&empty, &n.c), Activity::Active);
    assert_eq!(activity_of(&empty, &n.x), Activity::Pending);
    assert_eq!(activity_of(&empty, &n.s), Activity::Pending);
    assert_eq!(activity_of(&empty, &n.y), Activity::Pending);
}

#[test]
fn decision_under_its_chosen_alternative_is_active() {
    let n = build_nested();

    let chose_a = configure(&n.space, [(n.c.clone(), chosen(&n.a))]);

    assert_eq!(activity_of(&chose_a, &n.x), Activity::Active);
    assert_eq!(activity_of(&chose_a, &n.s), Activity::Active);
    assert_eq!(
        activity_of(&chose_a, &n.y),
        Activity::Pending,
        "y waits for the sub-choice s"
    );
}

#[test]
fn decision_under_a_chosen_sub_alternative_is_active() {
    let n = build_nested();

    let chose_b = configure(
        &n.space,
        [(n.c.clone(), chosen(&n.a)), (n.s.clone(), chosen(&n.b))],
    );

    assert_eq!(activity_of(&chose_b, &n.y), Activity::Active);
}

#[test]
fn decisions_under_another_alternative_are_inactive_at_every_depth() {
    let n = build_nested();

    let chose_other = configure(&n.space, [(n.c.clone(), chosen(&n.other))]);

    assert_eq!(activity_of(&chose_other, &n.c), Activity::Active);
    assert_eq!(activity_of(&chose_other, &n.x), Activity::Inactive);
    assert_eq!(activity_of(&chose_other, &n.s), Activity::Inactive);
    assert_eq!(activity_of(&chose_other, &n.y), Activity::Inactive);
}

#[rstest]
#[case::an_alternative("alternative")]
#[case::the_space("space")]
#[case::an_unknown_name("unknown")]
fn configuration_activity_answers_none_for_a_name_that_is_no_decision(#[case] which: &str) {
    let n = build_nested();
    let configuration = configure(&n.space, [(n.c.clone(), chosen(&n.a))]);
    let name = match which {
        "alternative" => n.a.clone(),
        "space" => n.space.name().clone(),
        "unknown" => Identifier::new("unknown"),
        _ => unreachable!("unknown case {which}"),
    };

    assert_eq!(configuration.activity(&name), None);
}

// ---------------------------------------------------------------------------
// Conditions
// ---------------------------------------------------------------------------

/// A flat space: the choice `c` among `a` and `b`, the variable `x` over
/// `{1, 2, 3}`, and the variable `w` with the condition `guard` builds.
struct Guarded {
    space: Space,
    c: Identifier,
    a: Identifier,
    b: Identifier,
    x: Identifier,
    w: Identifier,
}

/// Return the guarded space, `w`'s condition built by `guard` over `c`,
/// `a`, `b` and `x`.
fn build_guarded(
    guard: impl FnOnce(
        &Identifier,
        &Identifier,
        &Identifier,
        &Identifier,
    ) -> Vec<fhy_core::constraint::Constraint>,
) -> Guarded {
    let [c, a, b, x, w] = ["c", "a", "b", "x", "w"].map(Identifier::new);
    let constraints = guard(&c, &a, &b, &x);
    let space = Space::new(
        Identifier::new("guarded"),
        vec![int_variable(&x, &[1, 2, 3]), int_variable(&w, &[1, 2])],
        vec![choice_of(
            &c,
            vec![bare_alternative(&a), bare_alternative(&b)],
        )],
        vec![condition(&w, constraints)],
        Vec::new(),
    )
    .expect("the space is valid");
    Guarded {
        space,
        c,
        a,
        b,
        x,
        w,
    }
}

#[test]
fn condition_on_a_choice_reads_the_chosen_alternative() {
    let g = build_guarded(|c, a, _, _| vec![chooses(c, &[a])]);

    let chose_a = configure(&g.space, [(g.c.clone(), chosen(&g.a))]);
    let chose_b = configure(&g.space, [(g.c.clone(), chosen(&g.b))]);

    assert_eq!(activity_of(&chose_a, &g.w), Activity::Active);
    assert_eq!(activity_of(&chose_b, &g.w), Activity::Inactive);
}

#[test]
fn condition_naming_an_unassigned_active_decision_leaves_its_target_pending() {
    let g = build_guarded(|c, a, _, _| vec![chooses(c, &[a])]);

    let empty = configure(&g.space, []);

    assert_eq!(activity_of(&empty, &g.c), Activity::Active);
    assert_eq!(activity_of(&empty, &g.w), Activity::Pending);
}

#[rstest]
#[case::a_satisfying_value(1, Activity::Active)]
#[case::another_satisfying_value(2, Activity::Active)]
#[case::a_violating_value(3, Activity::Inactive)]
fn condition_on_a_variable_reads_its_value(#[case] value: i64, #[case] expected: Activity) {
    let g = build_guarded(|_, _, _, x| vec![in_set(x, [int(1), int(2)])]);

    let configuration = configure(&g.space, [(g.x.clone(), int(value))]);

    assert_eq!(activity_of(&configuration, &g.w), expected);
}

#[test]
fn condition_holds_only_while_every_member_holds() {
    let g = build_guarded(|c, a, _, x| vec![chooses(c, &[a]), not_in_set(x, [int(3)])]);

    let both = configure(
        &g.space,
        [(g.c.clone(), chosen(&g.a)), (g.x.clone(), int(1))],
    );
    let one = configure(
        &g.space,
        [(g.c.clone(), chosen(&g.a)), (g.x.clone(), int(3))],
    );

    assert_eq!(activity_of(&both, &g.w), Activity::Active);
    assert_eq!(activity_of(&one, &g.w), Activity::Inactive);
}

#[test]
fn conditions_on_one_target_conjoin() {
    let [c, a, b, x, w] = ["c", "a", "b", "x", "w"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("conjoined"),
        vec![int_variable(&x, &[1, 2, 3]), int_variable(&w, &[1, 2])],
        vec![choice_of(
            &c,
            vec![bare_alternative(&a), bare_alternative(&b)],
        )],
        vec![
            condition(&w, [chooses(&c, &[&a])]),
            condition(&w, [in_set(&x, [int(1)])]),
        ],
        Vec::new(),
    )
    .expect("the space is valid");

    let both = configure(&space, [(c.clone(), chosen(&a)), (x.clone(), int(1))]);
    let first_only = configure(&space, [(c.clone(), chosen(&a)), (x.clone(), int(2))]);

    assert_eq!(activity_of(&both, &w), Activity::Active);
    assert_eq!(activity_of(&first_only, &w), Activity::Inactive);
}

#[test]
fn condition_naming_an_inactive_decision_is_false() {
    let n = build_nested();
    let w = Identifier::new("w");
    let space = Space::new(
        Identifier::new("reads_a_child"),
        vec![int_variable(&w, &[1])],
        vec![n.space.choices()[0].clone()],
        vec![condition(&w, [not_in_set(&n.x, [int(1)])])],
        Vec::new(),
    )
    .expect("the space is valid");

    let chose_other = configure(&space, [(n.c.clone(), chosen(&n.other))]);

    assert_eq!(
        activity_of(&chose_other, &w),
        Activity::Inactive,
        "x is inactive, so the condition is false whatever x's value would be"
    );
}

#[test]
fn inactive_wins_over_pending() {
    let n = build_nested();
    let gate = Identifier::new("gate");
    let space = Space::new(
        Identifier::new("inactive_over_pending"),
        vec![int_variable(&gate, &[1, 2])],
        vec![n.space.choices()[0].clone()],
        vec![condition(&n.y, [in_set(&gate, [int(1)])])],
        Vec::new(),
    )
    .expect("the space is valid");

    let configuration = configure(&space, [(gate.clone(), int(2))]);

    assert_eq!(
        activity_of(&configuration, &n.y),
        Activity::Inactive,
        "y's choices are unassigned, but its condition is already false"
    );
}

#[test]
fn pending_parent_keeps_a_target_whose_condition_holds_pending() {
    let n = build_nested();
    let gate = Identifier::new("gate");
    let space = Space::new(
        Identifier::new("pending_parent"),
        vec![int_variable(&gate, &[1, 2])],
        vec![n.space.choices()[0].clone()],
        vec![condition(&n.y, [in_set(&gate, [int(1)])])],
        Vec::new(),
    )
    .expect("the space is valid");

    let configuration = configure(&space, [(gate.clone(), int(1))]);

    assert_eq!(activity_of(&configuration, &n.y), Activity::Pending);
}

#[test]
fn condition_reads_a_decision_declared_after_its_target() {
    let [late, early] = ["late", "early"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("declared_later"),
        vec![int_variable(&early, &[1, 2]), int_variable(&late, &[1, 2])],
        Vec::new(),
        vec![condition(&early, [in_set(&late, [int(2)])])],
        Vec::new(),
    )
    .expect("the space is valid");

    let configuration = configure(&space, [(late.clone(), int(2)), (early.clone(), int(1))]);

    assert_eq!(activity_of(&configuration, &early), Activity::Active);
    assert_eq!(configuration.value(&early), Some(&int(1)));
}

#[rstest]
#[case::above_the_bound(3, Activity::Active)]
#[case::at_the_bound(2, Activity::Active)]
#[case::below_the_bound(1, Activity::Inactive)]
fn condition_with_an_equation_over_a_number_is_decided(
    #[case] value: i64,
    #[case] expected: Activity,
) {
    let [n, w] = ["n", "w"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("numeric"),
        vec![plain_variable(&n, natural_param()), int_variable(&w, &[1])],
        Vec::new(),
        vec![condition(&w, [at_least(&n, 2)])],
        Vec::new(),
    )
    .expect("an equation may name a variable");

    let configuration = configure(&space, [(n.clone(), int(value))]);

    assert_eq!(activity_of(&configuration, &w), expected);
}

#[test]
fn decision_activated_by_a_condition_chain_follows_each_link() {
    let [p, q, r] = ["p", "q", "r"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("chain"),
        vec![
            int_variable(&p, &[1, 2]),
            int_variable(&q, &[1, 2]),
            int_variable(&r, &[1]),
        ],
        Vec::new(),
        vec![
            condition(&q, [in_set(&p, [int(1)])]),
            condition(&r, [in_set(&q, [int(1)])]),
        ],
        Vec::new(),
    )
    .expect("the space is valid");

    let deactivated = configure(&space, [(p.clone(), int(2))]);
    let pending = configure(&space, [(p.clone(), int(1))]);
    let active = configure(&space, [(p.clone(), int(1)), (q.clone(), int(1))]);

    assert_eq!(activity_of(&deactivated, &q), Activity::Inactive);
    assert_eq!(
        activity_of(&deactivated, &r),
        Activity::Inactive,
        "r's condition names q, which is inactive"
    );
    assert_eq!(activity_of(&pending, &r), Activity::Pending);
    assert_eq!(activity_of(&active, &r), Activity::Active);
}

// ---------------------------------------------------------------------------
// Undecided and failing conditions
// ---------------------------------------------------------------------------

#[test]
fn undecided_condition_is_a_problem_and_leaves_its_target_pending() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let [x, w] = ["x", "w"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("undecided"),
        vec![int_variable(&x, &[1, 2]), int_variable(&w, &[1, 2])],
        Vec::new(),
        vec![condition(
            &w,
            [TestCustom::build(
                "undecided",
                Expression::from(&x),
                Outcome::Undecided,
                &log,
            )],
        )],
        Vec::new(),
    )
    .expect("the space is valid");

    let errors = try_configure(&space, [(x.clone(), int(1))]).expect_err("undecided is refused");

    let [ConfigurationError::UndecidedCondition { target }] = errors.errors() else {
        panic!("expected one UndecidedCondition, got {errors:?}");
    };
    assert_eq!(target, &w);
}

#[test]
fn undecided_condition_also_refuses_an_entry_for_its_target() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let [x, w] = ["x", "w"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("undecided_target"),
        vec![int_variable(&x, &[1, 2]), int_variable(&w, &[1, 2])],
        Vec::new(),
        vec![condition(
            &w,
            [TestCustom::build(
                "undecided",
                Expression::from(&x),
                Outcome::Undecided,
                &log,
            )],
        )],
        Vec::new(),
    )
    .expect("the space is valid");

    let errors = try_configure(&space, [(x.clone(), int(1)), (w.clone(), int(1))])
        .expect_err("undecided is refused");

    let [
        ConfigurationError::UndecidedCondition { target },
        ConfigurationError::InactiveDecision { name },
    ] = errors.errors()
    else {
        panic!("expected UndecidedCondition then InactiveDecision, got {errors:?}");
    };
    assert_eq!((target, name), (&w, &w));
}

#[test]
fn condition_naming_an_unassigned_decision_is_not_evaluated() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let [x, w] = ["x", "w"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("not_evaluated"),
        vec![int_variable(&x, &[1, 2]), int_variable(&w, &[1, 2])],
        Vec::new(),
        vec![condition(
            &w,
            [TestCustom::build(
                "watched",
                Expression::from(&x),
                Outcome::Undecided,
                &log,
            )],
        )],
        Vec::new(),
    )
    .expect("the space is valid");

    let empty = configure(&space, []);

    assert_eq!(activity_of(&empty, &w), Activity::Pending);
    assert!(
        log.lock().expect("the log").is_empty(),
        "the condition names an unassigned decision, so it is not evaluated"
    );
}

#[test]
fn failing_condition_is_a_problem_naming_its_target() {
    let [n, w] = ["n", "w"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("failing"),
        vec![int_variable(&n, &[1, 2, 3]), int_variable(&w, &[1])],
        Vec::new(),
        vec![condition(&w, [at_least(&n, 2)])],
        Vec::new(),
    )
    .expect("the space is valid");
    let solver = failing_simplifier_solver();

    let result = Configuration::new(&space, [(n.clone(), int(3))], &ParamContext::new(&solver));

    let errors = result.expect_err("the simplifier fails");
    let [ConfigurationError::FailedCondition { target, error }] = errors.errors() else {
        panic!("expected one FailedCondition, got {errors:?}");
    };
    assert_eq!(target, &w);
    assert!(matches!(error, ConstraintError::Solve(_)), "got {error:?}");
}

#[test]
fn condition_over_a_refused_value_is_not_evaluated() {
    let [n, w] = ["n", "w"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("refused_value"),
        vec![plain_variable(&n, natural_param()), int_variable(&w, &[1])],
        Vec::new(),
        vec![condition(&w, [at_least(&n, 2)])],
        Vec::new(),
    )
    .expect("the space is valid");
    let solver = failing_simplifier_solver();

    let result = Configuration::new(&space, [(n.clone(), int(3))], &ParamContext::new(&solver));

    let errors = result.expect_err("the simplifier fails the value check");
    let [ConfigurationError::Assignment { variable, .. }] = errors.errors() else {
        panic!("expected the value's problem alone, got {errors:?}");
    };
    assert_eq!(
        variable, &n,
        "n's refused value counts as unassigned, so w's condition is pending and not evaluated"
    );
}

#[test]
fn equation_condition_over_a_string_variable_fails_to_evaluate() {
    let [label, w] = ["label", "w"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("textual"),
        vec![
            plain_variable(&label, categorical(vec![text("tile"), text("walk")])),
            int_variable(&w, &[1]),
        ],
        Vec::new(),
        vec![condition(&w, [at_least(&label, 2)])],
        Vec::new(),
    )
    .expect("an equation may name any variable; its evaluation decides the rest");

    let errors =
        try_configure(&space, [(label.clone(), text("tile"))]).expect_err("a string is no number");

    let [ConfigurationError::FailedCondition { target, error }] = errors.errors() else {
        panic!("expected one FailedCondition, got {errors:?}");
    };
    assert_eq!(target, &w);
    assert!(
        matches!(error, ConstraintError::UnusableBinding { identifier, .. } if identifier == &label),
        "got {error:?}"
    );
}
