//! Stories of how a param decides a member that is an integer bound of its
//! own variable: exactly, by comparing the integers, with no solver, so a
//! solver holding no simplifier still assigns a bounded param, and no
//! member event is reported for the bound.

use fhy_core::constraint::{
    Binding, Bindings, Constraint, ConstraintError, EquationConstraint, Outcome, Value,
};
use fhy_core::expression::{BigInt, Expression, LiteralValue};
use fhy_core::identifier::Identifier;
use fhy_core::param::{
    AssignmentError, BoundSide, IntegerDomain, Param, ParamAssignment, ParamContext, ParamDomain,
    RealDomain, Sign, ValueCheck, ZeroInclusion,
};
use fhy_core::solver::{GroundSimplifier, SolveError, Solver};
use rstest::rstest;

use crate::support::constraint::ConstraintKey;
use crate::support::param::{
    EvaluatingSimplifier, RecordedParamEvent, RecordingParamObserver, at_least, at_most,
    build_solver, context, literal, reference,
};
use crate::support::solver::RecordingSimplifier;

/// Return the integer value of the decimal text `digits`.
///
/// # Panics
///
/// Panics if `digits` is no integer text.
fn big_value(digits: &str) -> Value {
    Value::Int(digits.parse::<BigInt>().expect("an integer text"))
}

/// Return the integer literal of the decimal text `digits`.
///
/// # Panics
///
/// Panics if `digits` is no integer text.
fn big_literal(digits: &str) -> LiteralValue {
    LiteralValue::Int(digits.parse::<BigInt>().expect("an integer text"))
}

/// Return the domain of the integers, or of the natural numbers.
fn integer_domain(is_non_negative: bool) -> ParamDomain {
    ParamDomain::from(IntegerDomain::new(
        if is_non_negative {
            Sign::NonNegative
        } else {
            Sign::Any
        },
        ZeroInclusion::Included,
    ))
}

/// Return the param of a fresh variable over the integers (or the natural
/// numbers) narrowed by `bounds`, each `(side, inclusive, bound text)`.
///
/// # Panics
///
/// Panics if the param is refused.
fn build_bounded_param(
    is_non_negative: bool,
    bounds: &[(BoundSide, bool, &str)],
    context: &ParamContext<'_>,
) -> Param {
    let param = Param::new(
        integer_domain(is_non_negative),
        Identifier::new("x"),
        Vec::new(),
        context,
    )
    .expect("a param");
    bounds
        .iter()
        .try_fold(param, |param, &(side, is_inclusive, bound)| {
            param.with_bound(&big_literal(bound), side, is_inclusive, context)
        })
        .expect("bounds the domain allows")
}

/// Return the param of the integers narrowed by `constraint`, a constraint
/// over its variable built by `build`.
///
/// # Panics
///
/// Panics if the param is refused.
fn build_param_with_constraint(
    build: impl FnOnce(&Identifier) -> Constraint,
    context: &ParamContext<'_>,
) -> Param {
    let x = Identifier::new("x");
    let constraint = build(&x);
    Param::new(integer_domain(false), x, [constraint], context).expect("a param")
}

/// Return the environment binding the variable of `param` to `value`.
fn bind_variable(param: &Param, value: Value) -> Bindings {
    param
        .environment(Binding::Value(value), &Bindings::new())
        .expect("an environment")
}

/// Return the equation `expression` as a constraint.
fn equation(expression: Expression) -> Constraint {
    Constraint::from(EquationConstraint::new(expression))
}

// ---------------------------------------------------------------------------
// The motivating bounds, with no simplifier
// ---------------------------------------------------------------------------

/// Test a natural param bounded to `16..=4095` is assigned the values in
/// the range and refuses the values just outside it, by its violated
/// constraint, under a solver with no simplifier.
#[rstest]
#[case::lower_bound(16, true)]
#[case::inside(2955, true)]
#[case::upper_bound(4095, true)]
#[case::below_the_lower_bound(15, false)]
#[case::above_the_upper_bound(4096, false)]
#[case::zero(0, false)]
#[case::far_above(1_000_000, false)]
fn param_assignment_decides_inclusive_bounds_without_a_simplifier(
    #[case] value: i64,
    #[case] is_accepted: bool,
) {
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let param = build_bounded_param(
        true,
        &[
            (BoundSide::Lower, true, "16"),
            (BoundSide::Upper, true, "4095"),
        ],
        &context,
    );

    let assignment = ParamAssignment::new(param, Value::Int(BigInt::from(value)), &context);

    if is_accepted {
        assignment.expect("the value is inside the bounds");
    } else {
        assert!(
            matches!(assignment, Err(AssignmentError::ViolatedConstraint { .. })),
            "{assignment:?}"
        );
    }
}

/// Test restoring an assignment decides the bounds the same way.
#[rstest]
#[case::inside(2955, true)]
#[case::above_the_upper_bound(4096, false)]
fn param_assignment_restore_decides_bounds_without_a_simplifier(
    #[case] value: i64,
    #[case] is_accepted: bool,
) {
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let param = build_bounded_param(
        true,
        &[
            (BoundSide::Lower, true, "16"),
            (BoundSide::Upper, true, "4095"),
        ],
        &context,
    );

    let assignment = ParamAssignment::restore(param, Value::Int(BigInt::from(value)), &context);

    if is_accepted {
        let assignment = assignment.expect("the value is inside the bounds");
        assert_eq!(assignment.value(), &Value::Int(BigInt::from(value)));
    } else {
        assert!(
            matches!(assignment, Err(AssignmentError::ViolatedConstraint { .. })),
            "{assignment:?}"
        );
    }
}

/// Test a bounded param's value check is valid inside the bounds and
/// violated outside, without a simplifier.
#[rstest]
#[case::inside(20, true)]
#[case::below(10, false)]
fn param_check_value_decides_bounds_without_a_simplifier(
    #[case] value: i64,
    #[case] is_valid: bool,
) {
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let param = build_bounded_param(true, &[(BoundSide::Lower, true, "16")], &context);
    let environment = bind_variable(&param, Value::Int(BigInt::from(value)));

    let check = param.check_value(&environment, &context);

    match check {
        Ok(ValueCheck::Valid) => assert!(is_valid),
        Ok(ValueCheck::Violated { member }) => {
            assert!(!is_valid);
            assert!(member < param.constraints().len());
        }
        other => panic!("expected Valid or Violated, got {other:?}"),
    }
}

/// Test the evaluation of a bounded param's constraints is satisfied inside
/// the bounds, and violated at a member outside them, without a simplifier.
#[test]
fn param_evaluate_constraints_decides_bounds_without_a_simplifier() {
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let param = build_bounded_param(
        false,
        &[
            (BoundSide::Lower, true, "-5"),
            (BoundSide::Upper, false, "5"),
        ],
        &context,
    );

    let inside = param
        .evaluate_constraints(&bind_variable(&param, big_value("4")), &context)
        .expect("decides");
    let outside = param
        .evaluate_constraints(&bind_variable(&param, big_value("5")), &context)
        .expect("decides");

    assert_eq!(inside.outcome(), Outcome::Satisfied);
    assert_eq!(inside.deciding_member(), None);
    assert_eq!(outside.outcome(), Outcome::Violated);
    let member = outside.deciding_member().expect("a violated member");
    assert!(member < param.constraints().len());
}

// ---------------------------------------------------------------------------
// Every side and inclusivity
// ---------------------------------------------------------------------------

/// Test each bound side, inclusivity, sign and size of bound is decided
/// exactly, at the bound and on each side of it.
#[rstest]
#[case::lower_inclusive_at(false, BoundSide::Lower, true, "5", "5", true)]
#[case::lower_inclusive_above(false, BoundSide::Lower, true, "5", "6", true)]
#[case::lower_inclusive_below(false, BoundSide::Lower, true, "5", "4", false)]
#[case::lower_exclusive_at(false, BoundSide::Lower, false, "5", "5", false)]
#[case::lower_exclusive_above(false, BoundSide::Lower, false, "5", "6", true)]
#[case::lower_exclusive_below(false, BoundSide::Lower, false, "5", "4", false)]
#[case::upper_inclusive_at(false, BoundSide::Upper, true, "5", "5", true)]
#[case::upper_inclusive_below(false, BoundSide::Upper, true, "5", "4", true)]
#[case::upper_inclusive_above(false, BoundSide::Upper, true, "5", "6", false)]
#[case::upper_exclusive_at(false, BoundSide::Upper, false, "5", "5", false)]
#[case::upper_exclusive_below(false, BoundSide::Upper, false, "5", "4", true)]
#[case::upper_exclusive_above(false, BoundSide::Upper, false, "5", "6", false)]
#[case::negative_lower_at(false, BoundSide::Lower, true, "-3", "-3", true)]
#[case::negative_lower_below(false, BoundSide::Lower, true, "-3", "-4", false)]
#[case::negative_lower_far_above(false, BoundSide::Lower, true, "-3", "100", true)]
#[case::negative_lower_exclusive_at(false, BoundSide::Lower, false, "-3", "-3", false)]
#[case::negative_upper_at(false, BoundSide::Upper, true, "-3", "-3", true)]
#[case::negative_upper_above(false, BoundSide::Upper, true, "-3", "0", false)]
#[case::negative_upper_exclusive_at(false, BoundSide::Upper, false, "-3", "-3", false)]
#[case::negative_upper_exclusive_below(false, BoundSide::Upper, false, "-3", "-4", true)]
#[case::zero_lower_inclusive_at(false, BoundSide::Lower, true, "0", "0", true)]
#[case::zero_upper_exclusive_at(false, BoundSide::Upper, false, "0", "0", false)]
#[case::natural_lower_at(true, BoundSide::Lower, true, "7", "7", true)]
#[case::natural_lower_below(true, BoundSide::Lower, true, "7", "6", false)]
#[case::natural_upper_exclusive_below(true, BoundSide::Upper, false, "7", "6", true)]
#[case::natural_upper_exclusive_at(true, BoundSide::Upper, false, "7", "7", false)]
#[case::big_lower_at(
    false,
    BoundSide::Lower,
    true,
    "10000000000000000000000000000000000000000",
    "10000000000000000000000000000000000000000",
    true
)]
#[case::big_lower_just_below(
    false,
    BoundSide::Lower,
    true,
    "10000000000000000000000000000000000000000",
    "9999999999999999999999999999999999999999",
    false
)]
#[case::big_lower_exclusive_at(
    false,
    BoundSide::Lower,
    false,
    "10000000000000000000000000000000000000000",
    "10000000000000000000000000000000000000000",
    false
)]
#[case::big_upper_at(
    false,
    BoundSide::Upper,
    true,
    "10000000000000000000000000000000000000000",
    "10000000000000000000000000000000000000000",
    true
)]
#[case::big_upper_just_above(
    false,
    BoundSide::Upper,
    true,
    "10000000000000000000000000000000000000000",
    "10000000000000000000000000000000000000001",
    false
)]
#[case::big_negative_upper_below(
    false,
    BoundSide::Upper,
    false,
    "-10000000000000000000000000000000000000000",
    "-10000000000000000000000000000000000000001",
    true
)]
#[case::value_beyond_i64(
    false,
    BoundSide::Upper,
    true,
    "5",
    "-99999999999999999999999999",
    true
)]
fn param_decides_a_bound_exactly_without_a_simplifier(
    #[case] is_non_negative: bool,
    #[case] side: BoundSide,
    #[case] is_inclusive: bool,
    #[case] bound: &str,
    #[case] value: &str,
    #[case] is_accepted: bool,
) {
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let param = build_bounded_param(is_non_negative, &[(side, is_inclusive, bound)], &context);

    let assignment = ParamAssignment::new(param, big_value(value), &context);

    if is_accepted {
        assignment.expect("the value satisfies the bound");
    } else {
        assert!(
            matches!(assignment, Err(AssignmentError::ViolatedConstraint { .. })),
            "{assignment:?}"
        );
    }
}

/// Test a bound written with the literal on the left of the comparison is
/// decided as the same bound written with the variable on the left.
#[rstest]
#[case::literal_less_equal_variable_at(|x: &Identifier| literal(16).less_equal(reference(x)), 16, true)]
#[case::literal_less_equal_variable_below(|x: &Identifier| literal(16).less_equal(reference(x)), 15, false)]
#[case::literal_less_variable_at(|x: &Identifier| literal(16).less(reference(x)), 16, false)]
#[case::literal_less_variable_above(|x: &Identifier| literal(16).less(reference(x)), 17, true)]
#[case::literal_greater_equal_variable_at(|x: &Identifier| literal(4095).greater_equal(reference(x)), 4095, true)]
#[case::literal_greater_equal_variable_above(|x: &Identifier| literal(4095).greater_equal(reference(x)), 4096, false)]
#[case::literal_greater_variable_at(|x: &Identifier| literal(4095).greater(reference(x)), 4095, false)]
#[case::literal_greater_variable_below(|x: &Identifier| literal(4095).greater(reference(x)), 4094, true)]
#[case::negative_literal_on_the_left(|x: &Identifier| literal(-4).less_equal(reference(x)), -4, true)]
#[case::negative_literal_on_the_left_below(|x: &Identifier| literal(-4).less_equal(reference(x)), -5, false)]
fn param_decides_a_bound_with_the_literal_on_the_left(
    #[case] build: fn(&Identifier) -> Expression,
    #[case] value: i64,
    #[case] is_accepted: bool,
) {
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let param = build_param_with_constraint(|x| equation(build(x)), &context);

    let assignment = ParamAssignment::new(param, Value::Int(BigInt::from(value)), &context);

    if is_accepted {
        assignment.expect("the value satisfies the bound");
    } else {
        assert!(
            matches!(assignment, Err(AssignmentError::ViolatedConstraint { .. })),
            "{assignment:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// What stays with the solver
// ---------------------------------------------------------------------------

/// Test a member that is no bound of the variable still goes to the solver,
/// which, holding no simplifier, refuses the question.
#[rstest]
#[case::modulo(|x: &Identifier| reference(x).floor_mod(4).equals(0))]
#[case::bound_on_a_sum(|x: &Identifier| (reference(x) + 1).greater_equal(literal(16)))]
#[case::bound_with_a_non_integer_literal(|x: &Identifier| reference(x).greater_equal(1.5_f64))]
#[case::equality_with_a_literal(|x: &Identifier| reference(x).equals(16))]
#[case::inequality_with_a_literal(|x: &Identifier| reference(x).not_equals(16))]
fn param_assignment_still_asks_the_solver_about_a_non_bound_member(
    #[case] build: fn(&Identifier) -> Expression,
) {
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let param = build_param_with_constraint(|x| equation(build(x)), &context);

    let assignment = ParamAssignment::new(param, Value::Int(BigInt::from(16)), &context);

    assert!(
        matches!(
            &assignment,
            Err(AssignmentError::Constraint(ConstraintError::Solve(
                SolveError::NoCapableBackend(_)
            )))
        ),
        "{assignment:?}"
    );
}

/// Test a param holding a bound and a non-bound member decides the bound
/// exactly and still asks the solver about the other member.
#[test]
fn param_assignment_asks_the_solver_only_about_the_non_bound_member() {
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let x = Identifier::new("x");
    let modulo = equation(reference(&x).floor_mod(4).equals(0));
    let param = Param::new(
        integer_domain(true),
        x.clone(),
        [modulo, at_least(&x, 16)],
        &context,
    )
    .expect("a param");

    let assignment = ParamAssignment::new(param, Value::Int(BigInt::from(20)), &context);

    assert!(
        matches!(
            &assignment,
            Err(AssignmentError::Constraint(ConstraintError::Solve(
                SolveError::NoCapableBackend(_)
            )))
        ),
        "{assignment:?}"
    );
}

/// Test a bound of a variable bound to a non-integer value is not decided
/// by integer comparison: it goes to the solver.
#[test]
fn param_assignment_does_not_decide_a_bound_of_a_real_value_by_integers() {
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let x = Identifier::new("x");
    let param = Param::new(
        ParamDomain::from(RealDomain),
        x.clone(),
        [at_least(&x, 1)],
        &context,
    )
    .expect("a param");

    let assignment = ParamAssignment::new(param, Value::Float(2.0), &context);

    assert!(
        matches!(
            &assignment,
            Err(AssignmentError::Constraint(ConstraintError::Solve(
                SolveError::NoCapableBackend(_)
            )))
        ),
        "{assignment:?}"
    );
}

// ---------------------------------------------------------------------------
// No solver call and no member event for a bound
// ---------------------------------------------------------------------------

/// Test a bounded param's value check asks its simplifier nothing, and
/// reports no event, whatever the simplifier would answer.
#[rstest]
#[case::inside(20, ValueCheck::Valid)]
#[case::outside(10, ValueCheck::Violated { member: 0 })]
fn param_check_value_never_asks_the_simplifier_about_a_bound(
    #[case] value: i64,
    #[case] expected: ValueCheck,
) {
    let simplifier = RecordingSimplifier::failing("must not be asked");
    let solver = simplifier.solver();
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let x = Identifier::new("x");
    let param = Param::new(
        integer_domain(false),
        x.clone(),
        [at_least(&x, 16)],
        &context,
    )
    .expect("a param");
    let environment = bind_variable(&param, Value::Int(BigInt::from(value)));

    let check = param.check_value(&environment, &context);

    assert!(
        matches!(&check, Ok(check) if *check == expected),
        "{check:?}"
    );
    assert_eq!(simplifier.inputs(), Vec::<Expression>::new());
    assert_eq!(observer.events(), Vec::<RecordedParamEvent>::new());
}

/// Test a param of several bounds and one other member asks the simplifier
/// about the other member alone.
#[test]
fn param_check_value_asks_the_simplifier_about_the_non_bound_member_alone() {
    let simplifier = EvaluatingSimplifier::new();
    let solver = build_solver(&simplifier, None);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let x = Identifier::new("x");
    let modulo = equation(reference(&x).floor_mod(4).equals(0));
    let param = Param::new(
        integer_domain(true),
        x.clone(),
        [modulo, at_least(&x, 16), at_most(&x, 4095)],
        &context,
    )
    .expect("a param");
    let environment = bind_variable(&param, Value::Int(BigInt::from(20)));

    let check = param.check_value(&environment, &context);

    assert!(matches!(check, Ok(ValueCheck::Valid)), "{check:?}");
    assert_eq!(simplifier.inputs().len(), 1);
}

/// Test a bound member reports no member event under the ground simplifier,
/// and a member that is no bound reports its residual.
#[test]
fn param_evaluate_constraints_reports_member_events_for_a_non_bound_member_only() {
    let solver = Solver::new().with_simplifier(GroundSimplifier::new());
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let bound = at_least(&x, 16);
    let other = equation(reference(&x).less(reference(&y)));
    let param = Param::new(
        integer_domain(false),
        x.clone(),
        [bound.clone(), other.clone()],
        &ParamContext::new(&solver),
    )
    .expect("a param");
    let environment = bind_variable(&param, Value::Int(BigInt::from(20)));

    let evaluation = param
        .evaluate_constraints(&environment, &context)
        .expect("decides");

    assert_eq!(evaluation.outcome(), Outcome::Undecided);
    let events = observer.events();
    assert!(
        events.contains(&RecordedParamEvent::Member(
            other.key(),
            "residual".to_owned()
        )),
        "{events:?}"
    );
    assert!(
        !events.iter().any(
            |event| matches!(event, RecordedParamEvent::Member(key, _) if *key == bound.key())
        ),
        "{events:?}"
    );
}

/// Test a bound member is decided without reporting a failing simplifier as
/// an undecided member, which the default judgement would otherwise count.
#[test]
fn param_evaluate_constraints_counts_no_undecided_member_for_a_bound() {
    let simplifier = RecordingSimplifier::failing("cannot lower");
    let solver = simplifier.solver();
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let x = Identifier::new("x");
    let param = Param::new(
        integer_domain(false),
        x.clone(),
        [at_least(&x, 16)],
        &context,
    )
    .expect("a param");
    let environment = bind_variable(&param, Value::Int(BigInt::from(20)));

    let evaluation = param
        .evaluate_constraints(&environment, &context)
        .expect("decides");

    assert_eq!(evaluation.outcome(), Outcome::Satisfied);
    assert_eq!(evaluation.deciding_member(), None);
    assert_eq!(observer.events(), Vec::<RecordedParamEvent>::new());
}
