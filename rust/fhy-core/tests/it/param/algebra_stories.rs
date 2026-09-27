//! Stories of the domains' union and intersection.

use fhy_core::param::{Inclusivity, Sign, ZeroInclusion};
use std::sync::Arc;

use fhy_core::constraint::{Constraint, Outcome, Polarity, Value};
use fhy_core::identifier::Identifier;
use fhy_core::param::{
    CategoricalDomain, DomainKind, IntegerDomain, IntervalIntegerDomain, OrdinalDomain,
    ParamDomain, ParamError, PermutationDomain, RealDomain, SetOperation, Side,
};
use fhy_core::solver::SatResult;

use crate::support::constraint::{TestCustom, int, text};
use crate::support::param::{
    RecordingParamObserver, at_least, context, describe_all, in_set, ints, less_than, not_in_set,
    reference, scripted_solver,
};

fn ordinal(values: Vec<Value>) -> ParamDomain {
    ParamDomain::from(OrdinalDomain::new(values).expect("an ordinal domain"))
}

fn categorical(values: Vec<Value>) -> ParamDomain {
    ParamDomain::from(CategoricalDomain::new(values).expect("a categorical domain"))
}

fn values_of(domain: &ParamDomain) -> Vec<String> {
    match domain {
        ParamDomain::Ordinal(domain) => describe_all(domain.values()),
        ParamDomain::Categorical(domain) => describe_all(domain.values()),
        ParamDomain::Permutation(domain) => describe_all(domain.values()),
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Union
// ---------------------------------------------------------------------------

#[test]
fn union_of_ordinal_domains_bakes_both_effective_value_sets() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let z = Identifier::new("z");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let own_constraints = [not_in_set(&x, ints([1]))];
    let other_constraints = [in_set(&y, ints([5, 6]))];

    let (domain, constraints) = ordinal(ints([1, 2, 3]))
        .union(
            Side::new(&own_constraints, &x),
            &ordinal(ints([3, 4, 5, 6])),
            Side::new(&other_constraints, &y),
            &z,
            &context(&solver, &observer),
        )
        .expect("unions")
        .expect("the ordinal kind represents a union");

    assert_eq!(values_of(&domain), ["int:2", "int:3", "int:5", "int:6"]);
    assert!(constraints.is_empty());
}

#[test]
fn union_of_categorical_domains_keeps_distinct_kinds_apart() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();

    let (domain, _) = categorical(vec![int(1), text("a")])
        .union(
            Side::new(&[], &x),
            &categorical(vec![Value::Bool(true), int(1)]),
            Side::new(&[], &x),
            &x,
            &context(&solver, &observer),
        )
        .expect("unions")
        .expect("the categorical kind represents a union");

    assert_eq!(values_of(&domain), ["bool:true", "int:1", "str:a"]);
}

#[test]
fn union_is_not_represented_by_the_other_kinds() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    for domain in [
        ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included)),
        ParamDomain::from(IntervalIntegerDomain::new(
            Inclusivity::Inclusive,
            Sign::Any,
            ZeroInclusion::Included,
        )),
        ParamDomain::from(RealDomain),
        ParamDomain::from(PermutationDomain::new(ints([1])).expect("permutation")),
    ] {
        let union = domain
            .union(
                Side::new(&[], &x),
                &domain,
                Side::new(&[], &x),
                &x,
                &context,
            )
            .expect("answers");
        assert!(union.is_none(), "{:?}", domain.kind());
    }
}

#[test]
fn union_refuses_a_domain_of_another_kind() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();

    let error = ordinal(ints([1]))
        .union(
            Side::new(&[], &x),
            &categorical(ints([1])),
            Side::new(&[], &x),
            &x,
            &context(&solver, &observer),
        )
        .expect_err("another kind");

    assert!(matches!(
        error,
        ParamError::KindMismatch {
            operation: SetOperation::Union,
            own: DomainKind::Ordinal,
            other: DomainKind::Categorical,
        }
    ));
}

#[test]
fn union_of_two_empty_value_sets_is_refused() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let none = [in_set(&x, ints([9]))];

    let error = ordinal(ints([1]))
        .union(
            Side::new(&none, &x),
            &ordinal(ints([2])),
            Side::new(&none, &x),
            &x,
            &context(&solver, &observer),
        )
        .expect_err("empty");

    assert!(matches!(error, ParamError::EmptyUnion(DomainKind::Ordinal)));
    assert!(error.to_string().contains("is empty"));
}

#[test]
fn union_of_ordinal_values_that_do_not_order_is_refused() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();

    let error = ordinal(ints([1]))
        .union(
            Side::new(&[], &x),
            &ordinal(vec![text("a")]),
            Side::new(&[], &x),
            &x,
            &context(&solver, &observer),
        )
        .expect_err("incomparable");

    assert!(matches!(error, ParamError::IncomparableValues));
}

// ---------------------------------------------------------------------------
// Intersection
// ---------------------------------------------------------------------------

#[test]
fn intersection_of_finite_domains_bakes_the_common_effective_values() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let own_constraints = [not_in_set(&x, ints([3]))];

    let (domain, constraints) = categorical(ints([1, 2, 3, 4]))
        .intersection(
            Side::new(&own_constraints, &x),
            &categorical(ints([2, 3, 4, 5])),
            Side::new(&[], &x),
            &x,
            &context(&solver, &observer),
        )
        .expect("intersects");

    assert_eq!(values_of(&domain), ["int:2", "int:4"]);
    assert!(constraints.is_empty());
}

#[test]
fn intersection_of_disjoint_finite_domains_is_refused() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();

    let error = ordinal(ints([1]))
        .intersection(
            Side::new(&[], &x),
            &ordinal(vec![Value::Float(1.0)]),
            Side::new(&[], &x),
            &x,
            &context(&solver, &observer),
        )
        .expect_err("type-strictly disjoint");

    assert!(matches!(
        error,
        ParamError::EmptyIntersection(DomainKind::Ordinal)
    ));
}

#[test]
fn intersection_of_permutation_domains_keeps_the_members_and_conjoins() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let z = Identifier::new("z");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let own = ParamDomain::from(PermutationDomain::new(ints([1, 2])).expect("permutation"));
    let other = ParamDomain::from(PermutationDomain::new(ints([2, 1])).expect("permutation"));
    let own_constraints = [not_in_set(&x, [Value::Tuple(ints([1, 2]))])];
    let other_constraints = [not_in_set(&y, [Value::Tuple(ints([2, 1]))])];

    let (domain, constraints) = own
        .intersection(
            Side::new(&own_constraints, &x),
            &other,
            Side::new(&other_constraints, &y),
            &z,
            &context(&solver, &observer),
        )
        .expect("intersects");

    assert_eq!(values_of(&domain), ["int:1", "int:2"]);
    assert_eq!(constraints.len(), 2);
    for constraint in &constraints {
        let Constraint::Set(set) = constraint else {
            panic!("a set constraint");
        };
        assert_eq!(set.variable(), &z);
        assert_eq!(set.polarity(), Polarity::NotIn);
    }
}

#[test]
fn intersection_of_permutation_domains_over_different_members_is_refused() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let own = ParamDomain::from(PermutationDomain::new(ints([1, 2])).expect("permutation"));
    let other = ParamDomain::from(PermutationDomain::new(ints([1, 3])).expect("permutation"));

    let error = own
        .intersection(
            Side::new(&[], &x),
            &other,
            Side::new(&[], &x),
            &x,
            &context(&solver, &observer),
        )
        .expect_err("different members");

    assert!(matches!(error, ParamError::DifferentPermutationMembers));
}

#[test]
fn intersection_of_integer_domains_merges_the_restrictions() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);
    let merge = |left: IntegerDomain, right: IntegerDomain| {
        let (domain, _) = ParamDomain::from(left)
            .intersection(
                Side::new(&[], &x),
                &ParamDomain::from(right),
                Side::new(&[], &x),
                &x,
                &context,
            )
            .expect("intersects");
        let ParamDomain::Integer(domain) = domain else {
            panic!("an integer domain");
        };
        (domain.is_non_negative(), domain.is_zero_included())
    };

    assert_eq!(
        merge(
            IntegerDomain::new(Sign::Any, ZeroInclusion::Included),
            IntegerDomain::new(Sign::Any, ZeroInclusion::Included)
        ),
        (false, true)
    );
    assert_eq!(
        merge(
            IntegerDomain::new(Sign::NonNegative, ZeroInclusion::Included),
            IntegerDomain::new(Sign::Any, ZeroInclusion::Included)
        ),
        (true, true)
    );
    assert_eq!(
        merge(
            IntegerDomain::new(Sign::NonNegative, ZeroInclusion::Included),
            IntegerDomain::new(Sign::NonNegative, ZeroInclusion::Excluded)
        ),
        (true, false)
    );
}

#[test]
fn intersection_of_interval_domains_keeps_the_own_rendering_preference() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();

    let (domain, _) = ParamDomain::from(IntervalIntegerDomain::new(
        Inclusivity::Exclusive,
        Sign::Any,
        ZeroInclusion::Included,
    ))
    .intersection(
        Side::new(&[], &x),
        &ParamDomain::from(IntervalIntegerDomain::new(
            Inclusivity::Inclusive,
            Sign::NonNegative,
            ZeroInclusion::Excluded,
        )),
        Side::new(&[], &x),
        &x,
        &context(&solver, &observer),
    )
    .expect("intersects");

    assert_eq!(
        domain.interval_profile().expect("native").map(|profile| (
            profile.prefer_inclusive,
            profile.non_negative,
            profile.zero_included
        )),
        Some((false, true, false))
    );
}

#[test]
fn intersection_of_numeric_domains_rescopes_both_sides_onto_the_result() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let z = Identifier::new("z");
    let w = Identifier::new("w");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let own_constraints = [at_least(&x, 0), less_than(&x, &y), in_set(&x, ints([1, 2]))];
    let other_constraints = [less_than(&y, &w)];

    let (_, constraints) =
        ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included))
            .intersection(
                Side::new(&own_constraints, &x),
                &ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included)),
                Side::new(&other_constraints, &y),
                &z,
                &context(&solver, &observer),
            )
            .expect("intersects");

    let keys: Vec<String> = constraints.iter().map(Constraint::ordering_key).collect();
    assert_eq!(
        keys,
        [
            at_least(&z, 0).ordering_key(),
            Constraint::from(fhy_core::constraint::EquationConstraint::new(
                reference(&z).less(reference(&z))
            ))
            .ordering_key(),
            in_set(&z, ints([1, 2])).ordering_key(),
            less_than(&z, &w).ordering_key(),
        ]
    );
}

#[test]
fn intersection_refuses_a_set_constraint_on_another_variable() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let z = Identifier::new("z");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let foreign = [in_set(&y, ints([1]))];

    let error = ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included))
        .intersection(
            Side::new(&foreign, &x),
            &ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included)),
            Side::new(&[], &x),
            &z,
            &context(&solver, &observer),
        )
        .expect_err("scoped elsewhere");

    assert!(matches!(error, ParamError::Rescope { .. }));
    assert!(error.to_string().contains("scoped"));
}

#[test]
fn intersection_refuses_to_rescope_a_custom_constraint() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let log = Arc::new(std::sync::Mutex::new(Vec::new()));
    let custom = [TestCustom::build(
        "custom",
        reference(&x).greater_equal(reference(&x)),
        Outcome::Satisfied,
        &log,
    )];

    let error = ParamDomain::from(RealDomain)
        .intersection(
            Side::new(&custom, &x),
            &ParamDomain::from(RealDomain),
            Side::new(&[], &x),
            &x,
            &context(&solver, &observer),
        )
        .expect_err("a custom constraint");

    assert!(matches!(error, ParamError::UnexpectedConstraintKind));
}

#[test]
fn intersection_refuses_a_domain_of_another_kind() {
    let x = Identifier::new("x");
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();

    let error = ParamDomain::from(IntegerDomain::new(Sign::Any, ZeroInclusion::Included))
        .intersection(
            Side::new(&[], &x),
            &ParamDomain::from(IntervalIntegerDomain::new(
                Inclusivity::Inclusive,
                Sign::Any,
                ZeroInclusion::Included,
            )),
            Side::new(&[], &x),
            &x,
            &context(&solver, &observer),
        )
        .expect_err("another kind");

    assert!(matches!(
        error,
        ParamError::KindMismatch {
            operation: SetOperation::Intersection,
            own: DomainKind::Integer,
            other: DomainKind::IntervalInteger,
        }
    ));
}
