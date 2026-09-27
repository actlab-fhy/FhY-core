//! Stories of params: construction and canonical constraints, bounds and
//! their gates, value checks, questions, the set algebra, interval
//! arithmetic, equivalence, and assignments.

use fhy_core::constraint::{Binding, Bindings, Constraint, Outcome, Value};
use fhy_core::expression::builtins::BuiltinConstant;
use fhy_core::expression::{BigInt, Decimal, LiteralValue};
use fhy_core::identifier::Identifier;
use fhy_core::param::{
    BoundSide, CategoricalDomain, DomainKind, IntegerDomain, IntervalIntegerDomain, Operand,
    OrdinalDomain, Param, ParamAssignment, ParamContext, ParamDomain, ParamError, RealDomain,
    ValueCheck, check_bounds_are_ordered,
};
use fhy_core::param::{Inclusivity, Sign, ZeroInclusion};
use fhy_core::solver::{SatResult, Solver};
use rstest::rstest;

use crate::support::constraint::ConstraintKey;
use crate::support::constraint::{int, text};
use crate::support::lambda::Alpha;
use crate::support::param::{
    RecordingParamObserver, above, at_least, at_most, context, float, in_set, ints, less_than,
    not_in_set, scripted_solver,
};

fn integer_domain() -> ParamDomain {
    ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included))
}

fn natural_domain(zero_included: bool) -> ParamDomain {
    ParamDomain::from(IntegerDomain::new(
        Sign::NonNegative,
        ZeroInclusion::included_if(zero_included),
    ))
}

fn interval_domain(prefer_inclusive: bool, non_negative: bool, zero_included: bool) -> ParamDomain {
    ParamDomain::from(IntervalIntegerDomain::new(
        Inclusivity::inclusive_if(prefer_inclusive),
        Sign::non_negative_if(non_negative),
        ZeroInclusion::included_if(zero_included),
    ))
}

fn integer(value: i64) -> LiteralValue {
    LiteralValue::Int(BigInt::from(value))
}

/// Return the param over `domain` of a fresh variable bounded to
/// `[lower, upper]`, inclusive.
fn between(domain: ParamDomain, lower: i64, upper: i64, context: &ParamContext<'_>) -> Param {
    Param::new(domain, Identifier::new("param"), Vec::new(), context)
        .and_then(|param| param.with_bound(&integer(lower), BoundSide::Lower, true, context))
        .and_then(|param| param.with_bound(&integer(upper), BoundSide::Upper, true, context))
        .expect("a bounded param")
}

/// Return the ordering keys of `param`'s constraints, which name each
/// constraint's variable by id.
fn keys(param: &Param) -> Vec<String> {
    param.constraints().iter().map(ConstraintKey::key).collect()
}

/// Return the effective bounds `(min, max)` of an integer param's
/// constraints, read by evaluating membership at each integer in `range`.
fn members(
    param: &Param,
    range: std::ops::RangeInclusive<i64>,
    context: &ParamContext<'_>,
) -> Vec<i64> {
    range
        .filter(|value| {
            let environment = param
                .environment(Binding::Value(int(*value)), &Bindings::new())
                .expect("no bindings");
            param.check_value(&environment, context).expect("decides") == ValueCheck::Valid
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

#[test]
fn param_refuses_a_native_constant_as_its_variable() {
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let pi = BuiltinConstant::Pi.identifier().clone();

    let error = Param::new(
        integer_domain(),
        pi.clone(),
        Vec::new(),
        &context(&solver, &observer),
    )
    .expect_err("a native constant");

    assert!(matches!(error, ParamError::NativeConstantVariable(variable) if variable == pi));
}

#[test]
fn param_refuses_a_constraint_outside_its_scope() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();

    let error = Param::new(
        integer_domain(),
        x,
        vec![at_least(&y, 0)],
        &context(&solver, &observer),
    )
    .expect_err("out of scope");

    assert!(matches!(error, ParamError::OutOfScope { .. }));
}

#[test]
fn param_accepts_a_dependent_constraint_whose_scope_holds_its_variable() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();

    let param = Param::new(
        integer_domain(),
        x.clone(),
        vec![less_than(&x, &y)],
        &context(&solver, &observer),
    )
    .expect("a dependent constraint attaches");

    assert_eq!(keys(&param), [less_than(&x, &y).key()]);
}

#[test]
fn param_refuses_a_constraint_its_domain_forbids() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let domain = ParamDomain::from(OrdinalDomain::new(ints([1, 2])).expect("ordinal"));

    let error = Param::new(
        domain,
        x.clone(),
        vec![at_least(&x, 0)],
        &context(&solver, &observer),
    )
    .expect_err("an equation");

    assert!(matches!(
        error,
        ParamError::ForbiddenConstraintKind(DomainKind::Ordinal)
    ));
}

#[test]
fn param_drops_equivalent_constraints_and_orders_them_canonically() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();

    let param = Param::new(
        integer_domain(),
        x.clone(),
        vec![at_most(&x, 9), at_least(&x, 1), at_most(&x, 9)],
        &context(&solver, &observer),
    )
    .expect("a param");

    let mut expected = vec![at_most(&x, 9).key(), at_least(&x, 1).key()];
    expected.sort();
    assert_eq!(keys(&param), expected);
}

#[test]
fn param_appends_the_domain_s_implied_constraints_once() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);

    let implied =
        Param::new(natural_domain(true), x.clone(), Vec::new(), &context).expect("a param");
    let given = Param::new(
        natural_domain(true),
        x.clone(),
        vec![at_least(&x, 0)],
        &context,
    )
    .expect("a param");
    let positive =
        Param::new(natural_domain(false), x.clone(), Vec::new(), &context).expect("a param");

    assert_eq!(keys(&implied), [at_least(&x, 0).key()]);
    assert_eq!(keys(&given), [at_least(&x, 0).key()]);
    assert_eq!(keys(&positive), [above(&x, 0).key()]);
}

#[test]
fn adding_an_equivalent_constraint_returns_the_same_param() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let param =
        Param::new(integer_domain(), x.clone(), vec![at_least(&x, 1)], &context).expect("a param");

    let same = param
        .with_constraint(at_least(&x, 1), &context)
        .expect("adds");
    let other = param
        .with_constraint(at_most(&x, 5), &context)
        .expect("adds");

    assert!(Param::ptr_eq(&param, &same));
    assert!(!Param::ptr_eq(&param, &other));
    assert_eq!(other.constraints().len(), 2);
}

// ---------------------------------------------------------------------------
// Bounds and their gates
// ---------------------------------------------------------------------------

#[rstest]
#[case::lower_zero_negative(BoundSide::Lower, true, true, -1, "lower bound must be non-negative")]
#[case::lower_zero_exclusive(
    BoundSide::Lower,
    true,
    false,
    0,
    "lower bound must be at least 1 if zero is included and bound is exclusive"
)]
#[case::lower_positive_inclusive(
    BoundSide::Lower,
    false,
    true,
    0,
    "lower bound must be at least 1 when zero is not included"
)]
#[case::lower_positive_exclusive(BoundSide::Lower, false, false, -1, "lower bound must be non-negative when zero is not included and bound is exclusive")]
#[case::upper_zero_inclusive(BoundSide::Upper, true, true, -1, "upper bound must be non-negative when zero is included")]
#[case::upper_zero_exclusive(
    BoundSide::Upper,
    true,
    false,
    0,
    "upper bound must be at least 1 if zero is included and bound is exclusive"
)]
#[case::upper_positive_inclusive(
    BoundSide::Upper,
    false,
    true,
    0,
    "upper bound must be at least 1 when zero is not included"
)]
#[case::upper_positive_exclusive(
    BoundSide::Upper,
    false,
    false,
    1,
    "upper bound must be at least 2 when zero is not included and bound is exclusive"
)]
fn natural_gate_refuses_a_bound_literal_the_naturals_do_not_admit(
    #[case] side: BoundSide,
    #[case] zero_included: bool,
    #[case] is_inclusive: bool,
    #[case] bound: i64,
    #[case] message: &str,
) {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let param =
        Param::new(natural_domain(zero_included), x, Vec::new(), &context).expect("a param");

    let error = param
        .with_bound(&integer(bound), side, is_inclusive, &context)
        .expect_err("refused");

    assert!(matches!(error, ParamError::NaturalBound { .. }));
    assert_eq!(error.to_string(), message);
    param
        .with_bound(&integer(bound + 1), side, is_inclusive, &context)
        .expect("the next literal is admitted");
}

#[test]
fn natural_gate_judges_integer_bounds_of_non_negative_profiles_only() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let natural =
        Param::new(natural_domain(true), x.clone(), Vec::new(), &context).expect("a param");
    let interval_natural = Param::new(
        interval_domain(true, true, true),
        x.clone(),
        Vec::new(),
        &context,
    )
    .expect("a param");
    let plain = Param::new(integer_domain(), x, Vec::new(), &context).expect("a param");

    natural
        .with_bound(&LiteralValue::Float(-0.5), BoundSide::Lower, true, &context)
        .expect("a float bound is not judged");
    assert!(matches!(
        interval_natural.with_bound(&integer(-1), BoundSide::Lower, true, &context),
        Err(ParamError::NaturalBound { .. })
    ));
    plain
        .with_bound(&integer(-1), BoundSide::Lower, true, &context)
        .expect("an integer domain has no gate");
}

#[rstest]
#[case::ordered(integer(1), integer(2), true, true, true)]
#[case::equal_inclusive(integer(2), integer(2), true, true, true)]
#[case::equal_exclusive(integer(2), integer(2), true, false, false)]
#[case::reversed(integer(3), integer(2), true, true, false)]
#[case::decimal_below_float(
    LiteralValue::Decimal("0.1".parse::<Decimal>().expect("decimal")),
    LiteralValue::Float(0.1),
    false,
    false,
    true
)]
#[case::float_above_decimal(
    LiteralValue::Float(0.1),
    LiteralValue::Decimal("0.1".parse::<Decimal>().expect("decimal")),
    true,
    true,
    false
)]
#[case::rounding_decimal(
    LiteralValue::Decimal("0.10000000000000000001".parse::<Decimal>().expect("decimal")),
    LiteralValue::Float(0.1),
    true,
    true,
    true
)]
#[case::big_integer_and_float(
    LiteralValue::Int(BigInt::from(9_007_199_254_740_993_i64)),
    LiteralValue::Float(9_007_199_254_740_992.0),
    true,
    true,
    false
)]
#[case::infinity(LiteralValue::Float(f64::NEG_INFINITY), integer(0), false, false, true)]
#[case::nan(LiteralValue::Float(f64::NAN), integer(0), true, true, true)]
fn bounds_are_ordered_exactly(
    #[case] lower: LiteralValue,
    #[case] upper: LiteralValue,
    #[case] is_lower_inclusive: bool,
    #[case] is_upper_inclusive: bool,
    #[case] is_ordered: bool,
) {
    let result = check_bounds_are_ordered(
        &lower,
        &upper,
        Inclusivity::inclusive_if(is_lower_inclusive),
        Inclusivity::inclusive_if(is_upper_inclusive),
    );

    assert_eq!(result.is_ok(), is_ordered, "{result:?}");
    if let Err(error) = result {
        assert_eq!(
            error.to_string(),
            "lower bound must be less than or equal to upper bound"
        );
    }
}

// ---------------------------------------------------------------------------
// Value checks
// ---------------------------------------------------------------------------

#[test]
fn environment_refuses_bindings_of_the_param_s_own_variable() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let param = Param::new(
        integer_domain(),
        x.clone(),
        Vec::new(),
        &context(&solver, &observer),
    )
    .expect("a param");
    let bindings = Bindings::from_iter([(x, Binding::Value(int(1)))]);

    assert!(matches!(
        param.environment(Binding::Value(int(2)), &bindings),
        Err(ParamError::BindingsBindVariable(_))
    ));
}

#[test]
fn value_check_names_the_violated_or_undecided_member() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let param = Param::new(
        integer_domain(),
        x.clone(),
        vec![at_least(&x, 0), less_than(&x, &y), at_most(&x, 10)],
        &context,
    )
    .expect("a param");
    let check = |value: Value, bindings: &Bindings| {
        let environment = param
            .environment(Binding::Value(value), bindings)
            .expect("an environment");
        param.check_value(&environment, &context).expect("decides")
    };
    let position = |constraint: &Constraint| {
        param
            .constraints()
            .iter()
            .position(|member| member.is_structurally_equivalent(constraint))
            .expect("a member")
    };

    assert_eq!(check(text("a"), &Bindings::new()), ValueCheck::Inadmissible);
    assert_eq!(
        check(int(11), &Bindings::new()),
        ValueCheck::Violated {
            member: position(&at_most(&x, 10))
        }
    );
    assert_eq!(
        check(int(5), &Bindings::new()),
        ValueCheck::Undecided {
            member: position(&less_than(&x, &y))
        }
    );
    let bound = Bindings::from_iter([(y, Binding::Value(int(9)))]);
    assert_eq!(check(int(5), &bound), ValueCheck::Valid);
}

#[test]
fn value_check_of_a_finite_param_asks_no_solver() {
    let x = Identifier::new("x");
    let solver = Solver::new();
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let domain =
        ParamDomain::from(CategoricalDomain::new(vec![text("a"), text("b")]).expect("categorical"));
    let param = Param::new(
        domain,
        x.clone(),
        vec![not_in_set(&x, [text("b")])],
        &context,
    )
    .expect("a param");
    let check = |value: &str| {
        let environment = param
            .environment(Binding::Value(text(value)), &Bindings::new())
            .expect("an environment");
        param.check_value(&environment, &context).expect("decides")
    };

    assert_eq!(check("a"), ValueCheck::Valid);
    assert_eq!(check("b"), ValueCheck::Violated { member: 0 });
    assert_eq!(check("c"), ValueCheck::Inadmissible);
}

// ---------------------------------------------------------------------------
// Questions and the set algebra
// ---------------------------------------------------------------------------

#[test]
fn questions_are_the_domain_s() {
    let x = Identifier::new("x");
    let (solver, smt) = scripted_solver(SatResult::Unsat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let numeric = between(integer_domain(), 0, 10, &context);
    let finite = Param::new(
        ParamDomain::from(OrdinalDomain::new(ints([1, 2])).expect("ordinal")),
        x.clone(),
        vec![in_set(&x, ints([2]))],
        &context,
    )
    .expect("a param");

    assert_eq!(
        numeric.check_feasibility(&context).expect("decides"),
        Outcome::Violated
    );
    assert_eq!(
        finite.check_feasibility(&context).expect("decides"),
        Outcome::Satisfied
    );
    assert_eq!(
        finite.check_subset(&numeric, &context).expect("decides"),
        Outcome::Violated
    );
    assert!(
        numeric
            .is_value_set_subset(&between(integer_domain(), 5, 6, &context), &context)
            .expect("answers")
    );
    assert_eq!(smt.checks().len(), 1);
}

#[test]
fn union_of_finite_params_bakes_their_effective_values() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let z = Identifier::new("z");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let ordinal = |values: Vec<Value>, variable: &Identifier, constraints: Vec<Constraint>| {
        Param::new(
            ParamDomain::from(OrdinalDomain::new(values).expect("ordinal")),
            variable.clone(),
            constraints,
            &context,
        )
        .expect("a param")
    };
    let left = ordinal(ints([1, 2, 3]), &x, vec![not_in_set(&x, ints([1]))]);
    let right = ordinal(ints([5]), &y, Vec::new());

    let union = left.union(&right, z.clone(), &context).expect("unions");

    assert_eq!(union.variable(), &z);
    assert!(union.constraints().is_empty());
    let ParamDomain::Ordinal(domain) = union.domain() else {
        panic!("an ordinal domain");
    };
    assert_eq!(domain.values().len(), 3);
}

#[test]
fn union_of_numeric_params_is_unsupported() {
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let left = between(integer_domain(), 0, 1, &context);

    let error = left
        .union(&left, Identifier::new("z"), &context)
        .expect_err("unsupported");

    assert!(matches!(
        error,
        ParamError::UnsupportedUnion(DomainKind::Integer)
    ));
}

#[test]
fn intersection_is_refused_when_provably_empty() {
    let (feasible, _smt) = scripted_solver(SatResult::Sat);
    let (infeasible, _other) = scripted_solver(SatResult::Unsat);
    let observer = RecordingParamObserver::default();
    let left_context = context(&feasible, &observer);
    let left = between(integer_domain(), 0, 10, &left_context);
    let right = between(integer_domain(), 5, 20, &left_context);

    let result = left
        .intersection(&right, Identifier::new("z"), &left_context)
        .expect("feasible");
    let error = left
        .intersection(
            &right,
            Identifier::new("z"),
            &context(&infeasible, &observer),
        )
        .expect_err("infeasible");

    assert_eq!(result.constraints().len(), 4);
    assert!(matches!(error, ParamError::EmptyParamIntersection));
}

#[test]
fn intersection_recasts_an_integer_param_against_an_interval_one() {
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let interval = between(interval_domain(true, false, true), 0, 10, &context);
    let plain = between(integer_domain(), 5, 20, &context);
    let y = Identifier::new("y");
    let non_bound = Param::new(
        integer_domain(),
        y.clone(),
        vec![less_than(&y, &Identifier::new("w"))],
        &context,
    )
    .expect("a param");

    let result = plain
        .intersection(&interval, Identifier::new("z"), &context)
        .expect("intersects");
    let error = interval
        .intersection(&non_bound, Identifier::new("z"), &context)
        .expect_err("a non-bound constraint");

    assert_eq!(result.domain().kind(), DomainKind::IntervalInteger);
    assert!(matches!(error, ParamError::NonBoundOperand(None)));
}

// ---------------------------------------------------------------------------
// Interval arithmetic
// ---------------------------------------------------------------------------

#[test]
fn arithmetic_combines_the_effective_intervals() {
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let left = between(interval_domain(true, false, true), 0, 10, &context);
    let right = between(interval_domain(true, false, true), -3, 4, &context);
    let operand = Operand::Param(right.clone());

    let sum = left.checked_add(&operand, &context).expect("adds");
    let difference = left.checked_sub(&operand, &context).expect("subtracts");
    let product = left.checked_mul(&operand, &context).expect("multiplies");
    let negation = left.checked_neg(&context).expect("negates");

    assert_eq!(
        members(&sum, -20..=20, &context),
        (-3..=14).collect::<Vec<_>>()
    );
    assert_eq!(
        members(&difference, -20..=20, &context),
        (-4..=13).collect::<Vec<_>>()
    );
    assert_eq!(
        members(&product, -50..=50, &context),
        (-30..=40).collect::<Vec<_>>()
    );
    assert_eq!(
        members(&negation, -20..=20, &context),
        (-10..=0).collect::<Vec<_>>()
    );
    for result in [&sum, &difference, &product, &negation] {
        assert_ne!(result.variable(), left.variable());
        assert_ne!(result.variable(), right.variable());
    }
}

#[test]
fn arithmetic_renders_bounds_as_the_left_operand_prefers() {
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let exclusive = between(interval_domain(false, false, true), 0, 10, &context);
    let other = between(interval_domain(true, false, true), 1, 1, &context);

    let sum = exclusive
        .checked_add(&Operand::Param(other), &context)
        .expect("adds");

    let v = sum.variable();
    let mut expected = vec![above(v, 0).key(), {
        let below = Constraint::from(fhy_core::constraint::EquationConstraint::new(
            fhy_core::expression::Expression::from(v)
                .less(fhy_core::expression::Expression::literal(integer(12))),
        ));
        below.key()
    }];
    expected.sort();
    assert_eq!(keys(&sum), expected);
}

#[test]
fn arithmetic_keeps_a_natural_result_when_both_operands_are() {
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let positive = between(interval_domain(true, true, false), 1, 5, &context);
    let natural = between(interval_domain(true, true, true), 0, 2, &context);
    let plain = between(interval_domain(true, false, true), 0, 2, &context);

    let sum = positive
        .checked_add(&Operand::Param(natural.clone()), &context)
        .expect("adds");
    let product = positive
        .checked_mul(&Operand::Param(natural), &context)
        .expect("multiplies");
    let widened = positive
        .checked_add(&Operand::Param(plain), &context)
        .expect("adds");

    let profile = |param: &Param| {
        param
            .domain()
            .interval_profile()
            .expect("native")
            .expect("a profile")
    };
    assert!(profile(&sum).non_negative && !profile(&sum).zero_included);
    assert!(profile(&product).non_negative && profile(&product).zero_included);
    assert!(!profile(&widened).non_negative);
}

#[test]
fn arithmetic_keeps_unbounded_ends_and_zero_absorbs_them() {
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let unbounded = Param::new(
        interval_domain(true, false, true),
        Identifier::new("param"),
        Vec::new(),
        &context,
    )
    .and_then(|param| param.with_bound(&integer(1), BoundSide::Lower, true, &context))
    .expect("a param");

    let product = unbounded
        .checked_mul(&Operand::Integer(BigInt::from(0)), &context)
        .expect("multiplies");
    let sum = unbounded
        .checked_add(&Operand::Integer(BigInt::from(2)), &context)
        .expect("adds");

    assert_eq!(members(&product, -5..=5, &context), [0]);
    assert_eq!(members(&sum, -5..=5, &context), [3, 4, 5]);
}

#[test]
fn arithmetic_coerces_an_integer_param_of_bounds_and_refuses_other_pairs() {
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let plain = between(integer_domain(), 0, 3, &context);
    let interval = between(interval_domain(true, false, true), 1, 1, &context);
    let ordinal = Param::new(
        ParamDomain::from(OrdinalDomain::new(ints([1])).expect("ordinal")),
        Identifier::new("o"),
        Vec::new(),
        &context,
    )
    .expect("a param");

    let sum = plain
        .checked_add(&Operand::Param(interval.clone()), &context)
        .expect("adds");
    let declined = plain.checked_add(&Operand::Param(plain.clone()), &context);
    let reversed = interval
        .checked_reverse_sub(&Operand::Integer(BigInt::from(10)), &context)
        .expect("subtracts");

    assert_eq!(members(&sum, -5..=10, &context), [1, 2, 3, 4]);
    assert!(
        matches!(declined, Err(ParamError::NotAnIntervalOperand)),
        "{declined:?}"
    );
    assert_eq!(members(&reversed, -5..=15, &context), [9]);
    assert!(matches!(
        interval.checked_add(&Operand::Param(ordinal), &context),
        Err(ParamError::UnsupportedOperand)
    ));
    assert!(matches!(
        plain.checked_neg(&context),
        Err(ParamError::NotAnIntervalOperand)
    ));
}

#[test]
fn arithmetic_refuses_an_empty_interval() {
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let empty = between(interval_domain(true, false, true), 5, 3, &context);

    let error = empty.checked_neg(&context).expect_err("empty");

    assert!(matches!(error, ParamError::EmptyInterval(_)));
}

// ---------------------------------------------------------------------------
// Equivalence and assignments
// ---------------------------------------------------------------------------

#[test]
fn params_are_equivalent_structurally_and_under_a_renaming() {
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let left = between(natural_domain(true), 1, 10, &context);
    let right = between(natural_domain(true), 1, 10, &context);
    let wider = between(natural_domain(true), 1, 11, &context);
    let real = Param::new(
        ParamDomain::from(RealDomain),
        Identifier::new("r"),
        Vec::new(),
        &context,
    )
    .expect("a param");

    assert!(left.is_structurally_equivalent(&left));
    assert!(!left.is_structurally_equivalent(&right));
    assert!(left.alpha_equivalent(&right));
    assert!(!left.alpha_equivalent(&wider));
    assert!(!left.alpha_equivalent(&real));
}

#[test]
fn assignment_checks_its_value_and_restores_an_undecided_one() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let param = Param::new(
        integer_domain(),
        x.clone(),
        vec![at_most(&x, 10), less_than(&x, &y)],
        &context,
    )
    .expect("a param");

    assert!(matches!(
        ParamAssignment::new(param.clone(), text("a"), &context),
        Err(ParamError::Inadmissible)
    ));
    assert!(matches!(
        ParamAssignment::new(param.clone(), int(11), &context),
        Err(ParamError::ViolatedConstraint { .. })
    ));
    assert!(matches!(
        ParamAssignment::new(param.clone(), int(5), &context),
        Err(ParamError::UnverifiedConstraint { .. })
    ));
    let restored =
        ParamAssignment::restore(param.clone(), int(5), &context).expect("undecided is accepted");
    assert!(matches!(
        ParamAssignment::restore(param.clone(), int(11), &context),
        Err(ParamError::ViolatedConstraint { .. })
    ));
    assert!(
        restored
            .is_structurally_equivalent(&ParamAssignment::new_unvalidated(param.clone(), int(5)))
    );
    assert!(
        !restored.is_structurally_equivalent(&ParamAssignment::new_unvalidated(param, float(5.0)))
    );
}

#[test]
fn assignment_values_compare_type_strictly() {
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let domain = ParamDomain::from(
        CategoricalDomain::new(vec![int(1), Value::Bool(true)]).expect("categorical"),
    );
    let param = Param::new(domain, Identifier::new("c"), Vec::new(), &context).expect("a param");

    let one = ParamAssignment::new(param.clone(), int(1), &context).expect("admissible");
    let truth = ParamAssignment::new(param, Value::Bool(true), &context).expect("admissible");

    assert!(!one.is_structurally_equivalent(&truth));
    assert!(one.alpha_equivalent(&one.clone()));
}

#[test]
fn checked_add_of_a_non_interval_param_is_an_error() {
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let plain = between(integer_domain(), 0, 3, &context);
    let other = Operand::Param(plain.clone());

    for result in [
        plain.checked_add(&other, &context),
        plain.checked_sub(&other, &context),
        plain.checked_mul(&other, &context),
        plain.checked_reverse_sub(&Operand::Integer(BigInt::from(1)), &context),
        plain.checked_neg(&context),
    ] {
        assert!(
            matches!(result, Err(ParamError::NotAnIntervalOperand)),
            "{result:?}"
        );
    }
}
