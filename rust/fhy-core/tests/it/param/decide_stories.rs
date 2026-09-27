//! Stories of the decision procedures: evaluation under bindings,
//! feasibility, subsets and implication, with their enumerations,
//! screening and downgrades.

use fhy_core::param::{Sign, ZeroInclusion};
use std::sync::Arc;

use fhy_core::constraint::{Binding, Bindings, Constraint, ConstraintError, Outcome, Value};
use fhy_core::expression::{Expression, SymbolType};
use fhy_core::identifier::Identifier;
use fhy_core::param::{
    CategoricalDomain, IntegerDomain, NoParamObserver, OrdinalDomain, ParamContext, ParamDomain,
    ParamEvent, ParamObserver, PermutationDomain, RealDomain, Side, are_all_constraints_satisfied,
    compute_constraint_implication_subset, evaluate_constraints,
};
use fhy_core::solver::{QueryKind, SatResult, SolveError};

use crate::support::constraint::ConstraintKey;
use crate::support::constraint::{TestCustom, TestValueError, int, text};
use crate::support::param::{
    EvaluatingSimplifier, RecordedParamEvent, RecordingParamObserver, above, at_least, at_most,
    build_solver, context, failing_simplifier_solver, float, in_set, ints, less_than, not_in_set,
    real_solver, reference, scripted_solver, unknown,
};
use crate::support::solver::quoted_symbol;

fn integer() -> ParamDomain {
    ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included))
}

fn bind_value(x: &Identifier, value: Value) -> Bindings {
    Bindings::from_iter([(x.clone(), Binding::Value(value))])
}

// ---------------------------------------------------------------------------
// Evaluation under bindings
// ---------------------------------------------------------------------------

#[test]
fn evaluation_is_satisfied_when_every_member_is() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let constraints = [at_least(&x, 0), in_set(&x, ints([1, 2, 3]))];

    let evaluation = evaluate_constraints(
        &constraints,
        &bind_value(&x, int(2)),
        &context(&solver, &observer),
    )
    .expect("evaluates");

    assert_eq!(evaluation.outcome(), Outcome::Satisfied);
    assert_eq!(evaluation.deciding_member(), None);
}

#[test]
fn evaluation_stops_at_the_first_violated_member() {
    let x = Identifier::new("x");
    let simplifier = EvaluatingSimplifier::new();
    let solver = build_solver(&simplifier, None);
    let observer = RecordingParamObserver::default();
    let constraints = [at_least(&x, 0), at_most(&x, 1), at_least(&x, 5)];

    let evaluation = evaluate_constraints(
        &constraints,
        &bind_value(&x, int(3)),
        &context(&solver, &observer),
    )
    .expect("evaluates");

    assert_eq!(evaluation.outcome(), Outcome::Violated);
    assert_eq!(evaluation.deciding_member(), Some(1));
    assert_eq!(simplifier.inputs().len(), 2);
}

#[test]
fn evaluation_lets_a_later_violation_decide_over_an_undecided_member() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let dependent = less_than(&x, &y);
    let constraints = [dependent.clone(), at_least(&x, 5)];

    let evaluation = evaluate_constraints(
        &constraints,
        &bind_value(&x, int(3)),
        &context(&solver, &observer),
    )
    .expect("evaluates");

    assert_eq!(evaluation.outcome(), Outcome::Violated);
    assert_eq!(evaluation.deciding_member(), Some(1));
    assert!(
        observer
            .events()
            .contains(&RecordedParamEvent::UndecidedMember(dependent.key()))
    );
}

#[test]
fn evaluation_names_the_first_undecided_member() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let constraints = [at_least(&x, 0), less_than(&x, &y), in_set(&y, ints([1]))];

    let evaluation = evaluate_constraints(
        &constraints,
        &bind_value(&x, int(3)),
        &context(&solver, &observer),
    )
    .expect("evaluates");

    assert_eq!(evaluation.outcome(), Outcome::Undecided);
    assert_eq!(evaluation.deciding_member(), Some(1));
    let events = observer.events();
    assert!(events.contains(&RecordedParamEvent::Member(
        constraints[1].key(),
        "residual".to_owned()
    )));
    assert!(events.contains(&RecordedParamEvent::Member(
        constraints[2].key(),
        "unbound".to_owned()
    )));
}

#[test]
fn evaluation_counts_an_undecidable_failure_as_undecided() {
    let x = Identifier::new("x");
    let solver = failing_simplifier_solver();
    let observer = RecordingParamObserver::default();
    let failing = at_least(&x, 0);
    let constraints = [failing.clone(), in_set(&x, ints([5]))];

    let evaluation = evaluate_constraints(
        &constraints,
        &bind_value(&x, int(3)),
        &context(&solver, &observer),
    )
    .expect("the failure is undecidable");

    assert_eq!(evaluation.outcome(), Outcome::Violated);
    assert_eq!(evaluation.deciding_member(), Some(1));
    let events = observer.events();
    assert!(events.contains(&RecordedParamEvent::BridgeFailed(failing.key())));
    assert!(events.contains(&RecordedParamEvent::UndecidedMember(failing.key())));
}

#[test]
fn evaluation_propagates_a_failure_the_observer_does_not_judge_undecidable() {
    let x = Identifier::new("x");
    let solver = failing_simplifier_solver();
    let observer = RecordingParamObserver::strict();

    let error = evaluate_constraints(
        &[at_least(&x, 0)],
        &bind_value(&x, int(3)),
        &context(&solver, &observer),
    )
    .expect_err("the failure propagates");

    assert!(matches!(error, ConstraintError::Solve(_)), "{error:?}");
}

/// An observer that ignores events and keeps the default judgement.
struct SilentObserver;

impl ParamObserver for SilentObserver {
    fn notify(&self, _event: &ParamEvent<'_>) {}
}

#[test]
fn the_default_judgement_counts_only_a_backend_failure_undecidable() {
    let backend = ConstraintError::Solve(SolveError::Backend {
        backend: "recording".to_owned(),
        source: Box::new(TestValueError("cannot lower".to_owned())),
    });
    let refused = ConstraintError::Solve(SolveError::NoCapableBackend(QueryKind::Simplification));
    let custom = ConstraintError::Custom(Box::new(TestValueError("no".to_owned())));

    for observer in [&NoParamObserver as &dyn ParamObserver, &SilentObserver] {
        assert!(observer.is_undecidable(&backend));
        assert!(!observer.is_undecidable(&refused));
        assert!(!observer.is_undecidable(&custom));
    }
}

#[test]
fn a_context_without_an_observer_counts_a_backend_failure_as_undecided() {
    let x = Identifier::new("x");
    let solver = failing_simplifier_solver();

    let evaluation = evaluate_constraints(
        &[at_least(&x, 0), in_set(&x, ints([5]))],
        &bind_value(&x, int(3)),
        &ParamContext::new(&solver),
    )
    .expect("the failure is undecidable");

    assert_eq!(evaluation.outcome(), Outcome::Violated);
    assert_eq!(evaluation.deciding_member(), Some(1));
}

#[test]
fn evaluation_propagates_an_unusable_binding() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();

    let error = evaluate_constraints(
        &[at_least(&x, 0)],
        &bind_value(&x, Value::Tuple(Vec::new())),
        &context(&solver, &observer),
    )
    .expect_err("a tuple is no literal");

    assert!(
        matches!(error, ConstraintError::UnusableBinding { .. }),
        "{error:?}"
    );
}

#[test]
fn all_satisfied_stops_at_the_first_member_not_satisfied() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let simplifier = EvaluatingSimplifier::new();
    let solver = build_solver(&simplifier, None);
    let observer = RecordingParamObserver::default();
    let constraints = [less_than(&x, &y), at_least(&x, 0)];

    let is_satisfied = are_all_constraints_satisfied(
        &constraints,
        &bind_value(&x, int(3)),
        &context(&solver, &observer),
    )
    .expect("evaluates");

    assert!(!is_satisfied);
    assert_eq!(simplifier.inputs().len(), 1);
}

#[test]
fn all_satisfied_propagates_every_failure() {
    let x = Identifier::new("x");
    let solver = failing_simplifier_solver();
    let observer = RecordingParamObserver::default();

    are_all_constraints_satisfied(
        &[at_least(&x, 0)],
        &bind_value(&x, int(3)),
        &context(&solver, &observer),
    )
    .expect_err("the failure propagates");
}

// ---------------------------------------------------------------------------
// Feasibility of finite domains
// ---------------------------------------------------------------------------

#[test]
fn finite_feasibility_enumerates_the_values_without_the_solver() {
    let x = Identifier::new("x");
    let (solver, smt) = scripted_solver(SatResult::Unsat);
    let observer = RecordingParamObserver::default();
    let domain = ParamDomain::from(OrdinalDomain::new(ints([1, 2, 3])).expect("ordinal"));
    let context = context(&solver, &observer);

    let open = [not_in_set(&x, ints([1, 2]))];
    let closed = [not_in_set(&x, ints([1, 2, 3]))];

    assert_eq!(
        domain
            .has_feasible_value(Side::new(&open, &x), &context)
            .expect("decides"),
        Outcome::Satisfied
    );
    assert_eq!(
        domain
            .has_feasible_value(Side::new(&closed, &x), &context)
            .expect("decides"),
        Outcome::Violated
    );
    assert!(smt.checks().is_empty());
}

#[test]
fn permutation_feasibility_enumerates_the_permutations() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let domain = ParamDomain::from(PermutationDomain::new(ints([1, 2])).expect("permutation"));
    let context = context(&solver, &observer);
    let identity = Value::Tuple(ints([1, 2]));
    let swap = Value::Tuple(ints([2, 1]));

    let one_left = [not_in_set(&x, [identity.clone()])];
    let none_left = [not_in_set(&x, [identity, swap])];

    assert_eq!(
        domain
            .has_feasible_value(Side::new(&one_left, &x), &context)
            .expect("decides"),
        Outcome::Satisfied
    );
    assert_eq!(
        domain
            .has_feasible_value(Side::new(&none_left, &x), &context)
            .expect("decides"),
        Outcome::Violated
    );
}

/// A permutation domain of ten members has 3,628,800 permutations; with an
/// in-set constraint of one, feasibility and subset enumerate that one
/// candidate (F2-038), where they walked every permutation.
#[test]
fn a_permutation_param_with_a_singleton_in_set_decides_at_n_10() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let members: Vec<i64> = (0..10).collect();
    let domain =
        ParamDomain::from(PermutationDomain::new(ints(members.clone())).expect("permutation"));
    let reversed = Value::Tuple(ints(members.iter().rev().copied()));
    let only_reversed = [in_set(&x, [reversed.clone(), int(99)])];
    let excluded = [
        in_set(&x, [reversed.clone()]),
        not_in_set(&x, [reversed.clone()]),
    ];
    let other_side = [in_set(&y, [reversed])];

    let start = std::time::Instant::now();
    let feasible = domain
        .has_feasible_value(Side::new(&only_reversed, &x), &context)
        .expect("decides");
    let infeasible = domain
        .has_feasible_value(Side::new(&excluded, &x), &context)
        .expect("decides");
    let subset = domain
        .feasibility_subset(
            Side::new(&only_reversed, &x),
            &domain,
            Side::new(&other_side, &y),
            &context,
        )
        .expect("decides");
    let elapsed = start.elapsed();

    assert_eq!(feasible, Outcome::Satisfied);
    assert_eq!(infeasible, Outcome::Violated);
    assert_eq!(subset, Outcome::Satisfied);
    assert!(
        elapsed < std::time::Duration::from_millis(100),
        "{elapsed:?}"
    );
}

// ---------------------------------------------------------------------------
// Feasibility of numeric domains
// ---------------------------------------------------------------------------

#[test]
fn numeric_feasibility_enumerates_in_set_candidates() {
    let x = Identifier::new("x");
    let (solver, smt) = scripted_solver(SatResult::Unsat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let open = [
        in_set(&x, ints([1, 2, 3, 4, 5])),
        not_in_set(&x, ints([4])),
        above(&x, 3),
    ];
    let closed = [in_set(&x, ints([1, 2, 3, 4, 5])), above(&x, 10)];

    assert_eq!(
        integer()
            .has_feasible_value(Side::new(&open, &x), &context)
            .expect("decides"),
        Outcome::Satisfied
    );
    assert_eq!(
        integer()
            .has_feasible_value(Side::new(&closed, &x), &context)
            .expect("decides"),
        Outcome::Violated
    );
    assert!(smt.checks().is_empty());
}

#[test]
fn numeric_feasibility_intersects_every_in_set_constraint() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Unsat);
    let observer = RecordingParamObserver::default();
    let constraints = [in_set(&x, ints([1, 2, 3])), in_set(&x, ints([3, 4]))];

    let outcome = integer()
        .has_feasible_value(Side::new(&constraints, &x), &context(&solver, &observer))
        .expect("decides");

    assert_eq!(outcome, Outcome::Satisfied);
}

#[test]
fn numeric_feasibility_refuses_candidates_the_domain_does_not_admit() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let constraints = [in_set(&x, ints([1, 2]))];

    let outcome = ParamDomain::from(RealDomain)
        .has_feasible_value(Side::new(&constraints, &x), &context(&solver, &observer))
        .expect("decides");

    assert_eq!(outcome, Outcome::Violated);
}

#[test]
fn numeric_feasibility_reports_candidates_a_dependent_equation_leaves_undecided() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let constraints = [in_set(&x, ints([2, 1])), less_than(&x, &y)];

    let outcome = integer()
        .has_feasible_value(Side::new(&constraints, &x), &context(&solver, &observer))
        .expect("decides");

    assert_eq!(outcome, Outcome::Undecided);
    assert_eq!(
        observer.param_events(),
        [RecordedParamEvent::EnumerationUndecided(
            x,
            vec!["int:1".to_owned(), "int:2".to_owned()]
        )]
    );
}

#[test]
fn numeric_feasibility_asks_the_solver_about_the_screened_system() {
    let x = Identifier::new("x");
    let (solver, smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let constraints = [at_least(&x, 1), at_most(&x, 10)];

    let outcome = integer()
        .has_feasible_value(Side::new(&constraints, &x), &context(&solver, &observer))
        .expect("decides");

    assert_eq!(outcome, Outcome::Satisfied);
    let script = smt.only_script();
    assert!(script.contains(&format!("(declare-const {} Int)", quoted_symbol(&x))));
    assert!(observer.param_events().is_empty());
}

#[test]
fn numeric_feasibility_downgrades_a_satisfied_answer_on_an_inexact_system() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let dependent = less_than(&x, &y);
    let constraints = [at_least(&x, 0), dependent.clone()];

    let outcome = integer()
        .has_feasible_value(Side::new(&constraints, &x), &context(&solver, &observer))
        .expect("decides");

    assert_eq!(outcome, Outcome::Undecided);
    assert_eq!(
        observer.param_events(),
        [
            RecordedParamEvent::Screened(dependent.key(), x.clone(), "dependent_scope".to_owned()),
            RecordedParamEvent::SatisfiedOnInexactSystem(x),
        ]
    );
}

#[test]
fn numeric_feasibility_keeps_a_violated_answer_on_an_inexact_system() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let (solver, _smt) = scripted_solver(SatResult::Unsat);
    let observer = RecordingParamObserver::default();
    let constraints = [at_least(&x, 0), less_than(&x, &y)];

    let outcome = integer()
        .has_feasible_value(Side::new(&constraints, &x), &context(&solver, &observer))
        .expect("decides");

    assert_eq!(outcome, Outcome::Violated);
}

#[test]
fn numeric_feasibility_reports_a_solver_that_gives_up() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(unknown());
    let observer = RecordingParamObserver::default();
    let constraints = [at_least(&x, 0)];

    let outcome = integer()
        .has_feasible_value(Side::new(&constraints, &x), &context(&solver, &observer))
        .expect("decides");

    assert_eq!(outcome, Outcome::Undecided);
    assert!(
        observer
            .events()
            .contains(&RecordedParamEvent::Question(1, "gave_up".to_owned()))
    );
    assert_eq!(
        observer.param_events(),
        [RecordedParamEvent::SatisfiabilityUndecided(x)]
    );
}

#[test]
fn real_feasibility_downgrades_a_violation_resting_on_a_float_member() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Unsat);
    let observer = RecordingParamObserver::default();
    let constraints = [not_in_set(&x, [float(0.5)])];

    let outcome = ParamDomain::from(RealDomain)
        .has_feasible_value(Side::new(&constraints, &x), &context(&solver, &observer))
        .expect("decides");

    assert_eq!(outcome, Outcome::Undecided);
    assert_eq!(
        observer.param_events(),
        [RecordedParamEvent::ViolatedUnderKindConflation(x)]
    );
}

#[test]
fn screening_narrows_a_not_in_set_constraint_to_its_liftable_members() {
    let x = Identifier::new("x");
    let (solver, smt) = scripted_solver(SatResult::Unsat);
    let observer = RecordingParamObserver::default();
    let mixed = not_in_set(&x, [int(3), text("a")]);
    let constraints = [mixed.clone()];

    let outcome = integer()
        .has_feasible_value(Side::new(&constraints, &x), &context(&solver, &observer))
        .expect("decides");

    assert_eq!(outcome, Outcome::Violated);
    assert!(smt.only_script().contains("(distinct"));
    assert_eq!(
        observer.param_events(),
        [RecordedParamEvent::Screened(
            mixed.key(),
            x,
            "narrowed:[\"int:3\"]:[\"str:a\"]".to_owned()
        )]
    );
}

#[test]
fn screening_drops_a_not_in_set_constraint_without_a_liftable_member() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let strings = not_in_set(&x, [text("a")]);
    let constraints = [strings.clone()];

    let outcome = integer()
        .has_feasible_value(Side::new(&constraints, &x), &context(&solver, &observer))
        .expect("decides");

    assert_eq!(outcome, Outcome::Undecided);
    assert_eq!(
        observer.param_events()[0],
        RecordedParamEvent::Screened(strings.key(), x, "no_liftable_member".to_owned())
    );
}

#[test]
fn screening_drops_a_set_constraint_on_another_variable() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let foreign = not_in_set(&y, ints([1]));
    let constraints = [foreign.clone()];

    let outcome = integer()
        .has_feasible_value(Side::new(&constraints, &x), &context(&solver, &observer))
        .expect("decides");

    assert_eq!(outcome, Outcome::Undecided);
    assert_eq!(
        observer.param_events()[0],
        RecordedParamEvent::Screened(foreign.key(), x, "foreign_variable".to_owned())
    );
}

#[test]
fn screening_drops_a_custom_constraint_silently() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let log = Arc::new(std::sync::Mutex::new(Vec::new()));
    let custom = TestCustom::build(
        "custom",
        reference(&x).greater_equal(Expression::from(&x)),
        Outcome::Satisfied,
        &log,
    );

    let outcome = integer()
        .has_feasible_value(Side::new(&[custom], &x), &context(&solver, &observer))
        .expect("decides");

    assert_eq!(outcome, Outcome::Undecided);
    assert_eq!(
        observer.param_events(),
        [RecordedParamEvent::SatisfiedOnInexactSystem(x)]
    );
}

// ---------------------------------------------------------------------------
// Subsets of finite domains
// ---------------------------------------------------------------------------

#[test]
fn finite_subset_enumerates_the_values_that_are_valid_on_the_own_side() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let own = ParamDomain::from(OrdinalDomain::new(ints([1, 2, 3])).expect("ordinal"));
    let own_constraints = [in_set(&x, ints([1, 2]))];
    let wide = ParamDomain::from(OrdinalDomain::new(ints([1, 2])).expect("ordinal"));
    let narrow = ParamDomain::from(OrdinalDomain::new(ints([1])).expect("ordinal"));
    let categorical =
        ParamDomain::from(CategoricalDomain::new(ints([1, 2, 3])).expect("categorical"));

    let own_side = Side::new(&own_constraints, &x);
    assert_eq!(
        own.feasibility_subset(own_side, &wide, Side::new(&[], &y), &context)
            .expect("decides"),
        Outcome::Satisfied
    );
    assert_eq!(
        own.feasibility_subset(own_side, &narrow, Side::new(&[], &y), &context)
            .expect("decides"),
        Outcome::Violated
    );
    assert_eq!(
        own.feasibility_subset(own_side, &categorical, Side::new(&[], &y), &context)
            .expect("decides"),
        Outcome::Violated
    );
}

#[test]
fn permutation_subset_needs_as_many_members() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let own = ParamDomain::from(PermutationDomain::new(ints([1, 2])).expect("permutation"));
    let same = ParamDomain::from(PermutationDomain::new(ints([2, 1])).expect("permutation"));
    let longer = ParamDomain::from(PermutationDomain::new(ints([1, 2, 3])).expect("permutation"));
    let constraints = [not_in_set(&x, [Value::Tuple(ints([2, 1]))])];

    assert_eq!(
        own.feasibility_subset(Side::new(&[], &x), &same, Side::new(&[], &x), &context)
            .expect("decides"),
        Outcome::Satisfied
    );
    assert_eq!(
        own.feasibility_subset(
            Side::new(&[], &x),
            &same,
            Side::new(&constraints, &x),
            &context
        )
        .expect("decides"),
        Outcome::Violated
    );
    assert_eq!(
        own.feasibility_subset(Side::new(&[], &x), &longer, Side::new(&[], &x), &context)
            .expect("decides"),
        Outcome::Violated
    );
}

// ---------------------------------------------------------------------------
// Subsets of numeric domains
// ---------------------------------------------------------------------------

#[test]
fn numeric_subset_of_another_sort_is_violated_without_asking() {
    let x = Identifier::new("x");
    let (solver, smt) = scripted_solver(SatResult::Unsat);
    let observer = RecordingParamObserver::default();

    let outcome = integer()
        .feasibility_subset(
            Side::new(&[], &x),
            &ParamDomain::from(RealDomain),
            Side::new(&[], &x),
            &context(&solver, &observer),
        )
        .expect("decides");

    assert_eq!(outcome, Outcome::Violated);
    assert!(smt.checks().is_empty());
}

#[test]
fn implication_enumerates_the_own_in_set_candidates() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let (solver, smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let own = [in_set(&x, ints([1, 2]))];
    let loose = [at_least(&y, 1)];
    let tight = [at_least(&y, 2)];

    let decide = |other: &[Constraint]| {
        compute_constraint_implication_subset(
            &integer(),
            Side::new(&own, &x),
            &integer(),
            Side::new(other, &y),
            SymbolType::Int,
            &context,
        )
        .expect("decides")
    };

    assert_eq!(decide(&loose), Outcome::Satisfied);
    assert_eq!(decide(&tight), Outcome::Violated);
    assert!(smt.checks().is_empty());
}

#[test]
fn implication_skips_own_candidates_decided_out() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let own = [in_set(&x, ints([1, 2])), at_least(&x, 2)];
    let other = [at_least(&y, 2)];

    let outcome = compute_constraint_implication_subset(
        &integer(),
        Side::new(&own, &x),
        &integer(),
        Side::new(&other, &y),
        SymbolType::Int,
        &context(&solver, &observer),
    )
    .expect("decides");

    assert_eq!(outcome, Outcome::Satisfied);
}

#[test]
fn implication_reports_candidates_undecided_on_either_side() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let z = Identifier::new("z");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let own = [in_set(&x, ints([1]))];
    let other = [less_than(&y, &z)];

    let outcome = compute_constraint_implication_subset(
        &integer(),
        Side::new(&own, &x),
        &integer(),
        Side::new(&other, &y),
        SymbolType::Int,
        &context(&solver, &observer),
    )
    .expect("decides");

    assert_eq!(outcome, Outcome::Undecided);
    assert_eq!(
        observer.param_events(),
        [RecordedParamEvent::SubsetEnumerationUndecided(
            x,
            y,
            vec!["int:1".to_owned()]
        )]
    );
}

#[test]
fn implication_finds_a_witness_outside_the_other_in_set_candidates() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let (solver, smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let own = [at_least(&x, 0)];
    let other = [in_set(&y, ints([1, 2]))];

    let outcome = compute_constraint_implication_subset(
        &integer(),
        Side::new(&own, &x),
        &integer(),
        Side::new(&other, &y),
        SymbolType::Int,
        &context(&solver, &observer),
    )
    .expect("decides");

    assert_eq!(outcome, Outcome::Violated);
    let script = smt.only_script();
    assert!(!script.contains(&quoted_symbol(&x)));
    assert!(script.contains("var_"));
    assert_eq!(
        observer.param_events()[0],
        RecordedParamEvent::WitnessOutside(x, 2)
    );
}

#[test]
fn implication_asks_the_solver_when_no_witness_is_found() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let (solver, smt) = scripted_solver(SatResult::Unsat);
    let observer = RecordingParamObserver::default();
    let own = [at_least(&x, 0)];
    let other = [in_set(&y, ints([1, 2]))];

    let outcome = compute_constraint_implication_subset(
        &integer(),
        Side::new(&own, &x),
        &integer(),
        Side::new(&other, &y),
        SymbolType::Int,
        &context(&solver, &observer),
    )
    .expect("decides");

    assert_eq!(outcome, Outcome::Satisfied);
    assert_eq!(smt.checks().len(), 2);
}

#[test]
fn implication_skips_the_witness_on_an_inexact_own_side() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let z = Identifier::new("z");
    let (solver, smt) = scripted_solver(SatResult::Unsat);
    let observer = RecordingParamObserver::default();
    let own = [at_least(&x, 0), less_than(&x, &z)];
    let other = [in_set(&y, ints([1, 2]))];

    compute_constraint_implication_subset(
        &integer(),
        Side::new(&own, &x),
        &integer(),
        Side::new(&other, &y),
        SymbolType::Int,
        &context(&solver, &observer),
    )
    .expect("decides");

    assert_eq!(smt.checks().len(), 1);
}

#[test]
fn implication_moves_both_sides_onto_one_fresh_variable() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let (solver, smt) = scripted_solver(SatResult::Unsat);
    let observer = RecordingParamObserver::default();
    let own = [at_least(&x, 2)];
    let other = [at_least(&y, 0)];

    let outcome = compute_constraint_implication_subset(
        &integer(),
        Side::new(&own, &x),
        &integer(),
        Side::new(&other, &y),
        SymbolType::Int,
        &context(&solver, &observer),
    )
    .expect("decides");

    assert_eq!(outcome, Outcome::Satisfied);
    let script = smt.only_script();
    assert!(!script.contains(&quoted_symbol(&x)));
    assert!(!script.contains(&quoted_symbol(&y)));
    assert_eq!(script.matches("(declare-const").count(), 1);
}

#[test]
fn implication_reads_a_counterexample_as_violated() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let own = [at_least(&x, 0)];
    let other = [at_least(&y, 2)];

    let outcome = compute_constraint_implication_subset(
        &integer(),
        Side::new(&own, &x),
        &integer(),
        Side::new(&other, &y),
        SymbolType::Int,
        &context(&solver, &observer),
    )
    .expect("decides");

    assert_eq!(outcome, Outcome::Violated);
}

#[test]
fn implication_reports_a_solver_that_gives_up() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let (solver, _smt) = scripted_solver(unknown());
    let observer = RecordingParamObserver::default();

    let outcome = compute_constraint_implication_subset(
        &integer(),
        Side::new(&[at_least(&x, 0)], &x),
        &integer(),
        Side::new(&[at_least(&y, 0)], &y),
        SymbolType::Int,
        &context(&solver, &observer),
    )
    .expect("decides");

    assert_eq!(outcome, Outcome::Undecided);
    assert_eq!(
        observer.param_events(),
        [RecordedParamEvent::ImplicationUndecided(x, y)]
    );
}

#[test]
fn implication_downgrades_a_violation_from_an_inexact_antecedent() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let z = Identifier::new("z");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let own = [at_least(&x, 0), less_than(&x, &z)];
    let other = [at_least(&y, 2)];

    let outcome = compute_constraint_implication_subset(
        &integer(),
        Side::new(&own, &x),
        &integer(),
        Side::new(&other, &y),
        SymbolType::Int,
        &context(&solver, &observer),
    )
    .expect("decides");

    assert_eq!(outcome, Outcome::Undecided);
    assert!(
        observer
            .param_events()
            .contains(&RecordedParamEvent::ImplicationDowngraded(
                Outcome::Violated,
                x,
                y
            ))
    );
}

#[test]
fn implication_downgrades_a_satisfaction_into_an_inexact_consequent() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let z = Identifier::new("z");
    let (solver, _smt) = scripted_solver(SatResult::Unsat);
    let observer = RecordingParamObserver::default();
    let own = [at_least(&x, 2)];
    let other = [at_least(&y, 0), less_than(&y, &z)];

    let outcome = compute_constraint_implication_subset(
        &integer(),
        Side::new(&own, &x),
        &integer(),
        Side::new(&other, &y),
        SymbolType::Int,
        &context(&solver, &observer),
    )
    .expect("decides");

    assert_eq!(outcome, Outcome::Undecided);
    assert!(
        observer
            .param_events()
            .contains(&RecordedParamEvent::ImplicationDowngraded(
                Outcome::Satisfied,
                x,
                y
            ))
    );
}

#[test]
fn real_implication_downgrades_a_satisfaction_resting_on_a_float_member() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let (solver, _smt) = scripted_solver(SatResult::Unsat);
    let observer = RecordingParamObserver::default();
    let own = [not_in_set(&x, [float(0.5)])];
    let other: [Constraint; 0] = [];
    let real = ParamDomain::from(RealDomain);

    let outcome = compute_constraint_implication_subset(
        &real,
        Side::new(&own, &x),
        &real,
        Side::new(&other, &y),
        SymbolType::Real,
        &context(&solver, &observer),
    )
    .expect("decides");

    assert_eq!(outcome, Outcome::Undecided);
}

#[test]
fn integer_implication_leaves_a_float_member_to_the_membership_screen() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let (solver, smt) = scripted_solver(SatResult::Unsat);
    let observer = RecordingParamObserver::default();
    let own = [not_in_set(&x, [float(0.5)])];

    let outcome = compute_constraint_implication_subset(
        &integer(),
        Side::new(&own, &x),
        &integer(),
        Side::new(&[], &y),
        SymbolType::Int,
        &context(&solver, &observer),
    )
    .expect("decides");

    // A float member of an INT variable's set is a hazard the constraint
    // layer's membership screen refuses (R2-040 as revised), so the
    // question is undecided without a downgrade.
    assert_eq!(outcome, Outcome::Undecided);
    assert!(smt.checks().is_empty());
    assert!(
        observer
            .events()
            .contains(&RecordedParamEvent::Question(1, "refused".to_owned()))
    );
    assert_eq!(
        observer.param_events(),
        [RecordedParamEvent::ImplicationUndecided(x, y)]
    );
}

// ---------------------------------------------------------------------------
// With a real solver
// ---------------------------------------------------------------------------

#[test]
fn real_solver_decides_feasibility_and_subsets_of_intervals() {
    let Some(solver) = real_solver() else {
        return;
    };
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let inner = [at_least(&x, 2), at_most(&x, 8)];
    let outer = [at_least(&y, 0), at_most(&y, 10)];
    let empty = [at_least(&x, 5), at_most(&x, 3)];

    assert_eq!(
        integer()
            .has_feasible_value(Side::new(&inner, &x), &context)
            .expect("decides"),
        Outcome::Satisfied
    );
    assert_eq!(
        integer()
            .has_feasible_value(Side::new(&empty, &x), &context)
            .expect("decides"),
        Outcome::Violated
    );
    assert_eq!(
        integer()
            .feasibility_subset(
                Side::new(&inner, &x),
                &integer(),
                Side::new(&outer, &y),
                &context
            )
            .expect("decides"),
        Outcome::Satisfied
    );
    assert_eq!(
        integer()
            .feasibility_subset(
                Side::new(&outer, &y),
                &integer(),
                Side::new(&inner, &x),
                &context
            )
            .expect("decides"),
        Outcome::Violated
    );
}

// ---------------------------------------------------------------------------
// The domain's own restriction (F2-021)
// ---------------------------------------------------------------------------

fn natural() -> ParamDomain {
    ParamDomain::from(IntegerDomain::new(
        Sign::NonNegative,
        ZeroInclusion::Included,
    ))
}

fn positive() -> ParamDomain {
    ParamDomain::from(IntegerDomain::new(
        Sign::NonNegative,
        ZeroInclusion::Excluded,
    ))
}

#[test]
fn a_natural_domain_does_not_admit_a_negative_value() {
    assert_eq!(natural().is_value_admissible(&int(-5)).ok(), Some(false));
    assert_eq!(natural().is_value_admissible(&int(0)).ok(), Some(true));
    assert_eq!(positive().is_value_admissible(&int(0)).ok(), Some(false));
    assert_eq!(positive().is_value_admissible(&int(1)).ok(), Some(true));
    assert_eq!(integer().is_value_admissible(&int(-5)).ok(), Some(true));
    let interval_natural = ParamDomain::from(fhy_core::param::IntervalIntegerDomain::new(
        fhy_core::param::Inclusivity::Inclusive,
        Sign::NonNegative,
        ZeroInclusion::Included,
    ));
    assert_eq!(
        interval_natural.is_value_admissible(&int(-1)).ok(),
        Some(false)
    );
}

#[test]
fn value_set_subsets_respect_the_sign_restriction() {
    let solver = fhy_core::solver::Solver::new();
    let context = ParamContext::new(&solver);
    let real = ParamDomain::from(RealDomain);

    for (own, other, expected) in [
        (integer(), natural(), false),
        (natural(), integer(), true),
        (positive(), natural(), true),
        (natural(), positive(), false),
        (natural(), natural(), true),
        (natural(), real.clone(), false),
        (real.clone(), real, true),
    ] {
        assert_eq!(
            own.is_value_set_subset(&other, &context).ok(),
            Some(expected),
            "{own} within {other}"
        );
    }
}

#[test]
fn natural_feasibility_folds_in_the_sign_bound() {
    let Some(solver) = real_solver() else {
        return;
    };
    let x = Identifier::new("x");
    let constraints = [at_most(&x, -1)];
    let context = ParamContext::new(&solver);

    let outcome = natural()
        .has_feasible_value(Side::new(&constraints, &x), &context)
        .expect("decides");

    assert_eq!(outcome, Outcome::Violated);
    assert_eq!(
        integer()
            .has_feasible_value(Side::new(&constraints, &x), &context)
            .expect("decides"),
        Outcome::Satisfied
    );
    let param = fhy_core::param::Param::new(natural(), x.clone(), constraints.to_vec(), &context)
        .expect("a bound");
    assert_eq!(
        param.check_feasibility(&context).ok(),
        Some(Outcome::Violated)
    );
}

#[test]
fn the_integers_are_no_feasibility_subset_of_the_naturals() {
    let Some(solver) = real_solver() else {
        return;
    };
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let context = ParamContext::new(&solver);

    let integers_in_naturals = integer()
        .feasibility_subset(Side::new(&[], &x), &natural(), Side::new(&[], &y), &context)
        .expect("decides");
    let naturals_in_integers = natural()
        .feasibility_subset(Side::new(&[], &x), &integer(), Side::new(&[], &y), &context)
        .expect("decides");
    let implication = compute_constraint_implication_subset(
        &integer(),
        Side::new(&[], &x),
        &natural(),
        Side::new(&[], &y),
        SymbolType::Int,
        &context,
    )
    .expect("decides");

    assert_ne!(integers_in_naturals, Outcome::Satisfied);
    assert_eq!(integers_in_naturals, Outcome::Violated);
    assert_eq!(naturals_in_integers, Outcome::Satisfied);
    assert_eq!(implication, Outcome::Violated);
}

#[test]
fn the_param_path_holds_the_restriction_once() {
    let Some(solver) = real_solver() else {
        return;
    };
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let context = ParamContext::new(&solver);
    let naturals = fhy_core::param::Param::new(natural(), x, [], &context).expect("a param");
    let integers = fhy_core::param::Param::new(integer(), y, [], &context).expect("a param");

    assert_eq!(
        naturals.check_subset(&integers, &context).ok(),
        Some(Outcome::Satisfied)
    );
    assert_eq!(
        integers.check_subset(&naturals, &context).ok(),
        Some(Outcome::Violated)
    );
    assert_eq!(
        naturals.check_feasibility(&context).ok(),
        Some(Outcome::Satisfied)
    );
    assert_eq!(naturals.constraints().len(), 1);
}
