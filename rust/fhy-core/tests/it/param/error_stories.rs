//! Compile-level pins that each public `param` operation returns only its
//! family's error: `DomainError` for the finite domains' constructors,
//! `ParamBuildError` for building and narrowing params, `AssignmentError`
//! for the value checks, `IntervalError` for interval arithmetic,
//! `ParamError` for the questions and the set algebra, and
//! `ConstraintError` for evaluating constraints.
//!
//! Each binding names its error type, so a signature that widens fails to
//! compile; each call also runs.

use fhy_core::constraint::{Binding, Bindings, ConstraintError, Outcome, Value};
use fhy_core::expression::{LiteralValue, SymbolType};
use fhy_core::identifier::Identifier;
use fhy_core::param::{
    AssignmentError, BoundSide, CategoricalDomain, DomainError, Evaluation, Inclusivity,
    IntegerDomain, IntervalError, IntervalProfile, Operand, OrdinalDomain, Param, ParamAssignment,
    ParamBuildError, ParamContext, ParamDomain, ParamError, PermutationDomain, Side, Sign,
    ValueCheck, ZeroInclusion, are_all_constraints_satisfied, check_bounds_are_ordered,
    compute_constraint_implication_subset, evaluate_constraints,
};
use fhy_core::solver::{SatResult, Solver};
use fhy_core::term::AlphaEquivalence;

use crate::support::constraint::int;
use crate::support::param::{RecordingParamObserver, at_least, context, ints, scripted_solver};

fn integer() -> ParamDomain {
    ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included))
}

#[test]
fn the_finite_domain_constructors_return_domain_errors() {
    let ordinal: Result<OrdinalDomain, DomainError> = OrdinalDomain::new(ints([2, 1]));
    let categorical: Result<CategoricalDomain, DomainError> = CategoricalDomain::new(Vec::new());
    let permutation: Result<PermutationDomain, DomainError> = PermutationDomain::new(ints([1, 1]));

    ordinal.expect("orders");
    assert!(matches!(categorical, Err(DomainError::EmptyValues(_))));
    assert!(matches!(permutation, Err(DomainError::DuplicateValues(_))));
}

#[test]
fn building_and_narrowing_a_param_return_build_errors() {
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let x = Identifier::new("x");

    let built: Result<Param, ParamBuildError> =
        Param::new(integer(), x.clone(), [at_least(&x, 0)], &context);
    let param = built.expect("builds");
    let narrowed: Result<Param, ParamBuildError> = param.with_constraint(at_least(&x, 1), &context);
    let replaced: Result<Param, ParamBuildError> = param.with_constraints([], &context);
    let bounded: Result<Param, ParamBuildError> =
        param.with_bound(&LiteralValue::from(3), BoundSide::Upper, true, &context);
    let validated: Result<(), ParamBuildError> = param.validate_constraint(&at_least(&x, 2));
    let ordered: Result<(), ParamBuildError> = check_bounds_are_ordered(
        &LiteralValue::from(2),
        &LiteralValue::from(1),
        Inclusivity::Inclusive,
        Inclusivity::Inclusive,
    );
    let refused: Result<(), ParamBuildError> = integer().validate_constraint(&at_least(&x, 0), &x);
    let implied: Result<Vec<_>, ParamBuildError> = integer().implied_constraints(&x);

    narrowed.expect("narrows");
    replaced.expect("replaces");
    bounded.expect("bounds");
    validated.expect("validates");
    assert!(matches!(ordered, Err(ParamBuildError::UnorderedBounds)));
    refused.expect("allowed");
    assert!(implied.expect("implies").is_empty());
}

#[test]
fn the_value_checks_return_assignment_errors() {
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let x = Identifier::new("x");
    let param = Param::new(integer(), x, [], &context).expect("builds");

    let environment: Result<Bindings, AssignmentError> =
        param.environment(Binding::Value(int(1)), &Bindings::new());
    let environment = environment.expect("binds");
    let admissible: Result<bool, AssignmentError> =
        param.is_value_admissible(&Binding::Value(int(1)));
    let evaluation: Result<Evaluation, AssignmentError> =
        param.evaluate_constraints(&environment, &context);
    let check: Result<ValueCheck, AssignmentError> = param.check_value(&environment, &context);
    let assigned: Result<ParamAssignment, AssignmentError> =
        ParamAssignment::new(param.clone(), Value::Bool(true), &context);
    let restored: Result<ParamAssignment, AssignmentError> =
        ParamAssignment::restore(param, int(2), &context);

    assert!(admissible.expect("answers"));
    assert_eq!(evaluation.expect("evaluates").outcome(), Outcome::Satisfied);
    assert!(matches!(check, Ok(ValueCheck::Valid)));
    assert!(matches!(assigned, Err(AssignmentError::Inadmissible)));
    restored.expect("restores");
}

#[test]
fn interval_arithmetic_returns_interval_errors() {
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let x = Identifier::new("x");
    let param = Param::new(integer(), x, [], &context).expect("builds");
    let other = Operand::Param(param.clone());

    let results: [Result<Param, IntervalError>; 5] = [
        param.checked_add(&other, &context),
        param.checked_sub(&other, &context),
        param.checked_mul(&other, &context),
        param.checked_reverse_sub(&other, &context),
        param.checked_neg(&context),
    ];

    for result in results {
        assert!(matches!(result, Err(IntervalError::NotAnIntervalOperand)));
    }
}

#[test]
fn the_questions_and_the_set_algebra_return_param_errors() {
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let left = Param::new(integer(), x.clone(), [at_least(&x, 0)], &context).expect("builds");
    let right = Param::new(integer(), y.clone(), [at_least(&y, 0)], &context).expect("builds");
    let side = Side::new(left.constraints(), left.variable());

    let sort: Result<Option<SymbolType>, ParamError> = left.symbol_type();
    let feasible: Result<Outcome, ParamError> = left.check_feasibility(&context);
    let subset: Result<Outcome, ParamError> = left.check_subset(&right, &context);
    let value_subset: Result<bool, ParamError> = left.is_value_set_subset(&right, &context);
    let union: Result<Param, ParamError> = left.union(&right, Identifier::new("u"), &context);
    let intersection: Result<Param, ParamError> =
        left.intersection(&right, Identifier::new("i"), &context);
    let alpha: Result<bool, ParamError> = left.is_alpha_equivalent(&right);
    let admissible: Result<bool, ParamError> = integer().is_value_admissible(&int(1));
    let profile: Result<Option<IntervalProfile>, ParamError> = integer().interval_profile();
    let implication: Result<Outcome, ParamError> = compute_constraint_implication_subset(
        &integer(),
        side,
        &integer(),
        Side::new(right.constraints(), right.variable()),
        SymbolType::Int,
        &context,
    );

    assert_eq!(sort.expect("answers"), Some(SymbolType::Int));
    feasible.expect("decides");
    subset.expect("decides");
    assert!(value_subset.expect("answers"));
    assert!(matches!(union, Err(ParamError::UnsupportedUnion(_))));
    intersection.expect("intersects");
    assert!(alpha.expect("compares"));
    assert!(admissible.expect("answers"));
    assert!(profile.expect("answers").is_some());
    implication.expect("decides");
}

#[test]
fn evaluating_constraints_returns_constraint_errors() {
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let x = Identifier::new("x");
    let mut bindings = Bindings::new();
    bindings.insert(x.clone(), Binding::Value(int(1)));

    let evaluation: Result<Evaluation, ConstraintError> =
        evaluate_constraints(&[at_least(&x, 0)], &bindings, &context);
    let satisfied: Result<bool, ConstraintError> =
        are_all_constraints_satisfied(&[at_least(&x, 0)], &bindings, &context);

    assert_eq!(evaluation.expect("evaluates").outcome(), Outcome::Satisfied);
    assert!(satisfied.expect("evaluates"));
}
