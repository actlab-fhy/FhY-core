//! Stories of a custom domain: the procedures reach it through its hooks,
//! and its failures propagate.

use fhy_core::constraint::Outcome;
use fhy_core::expression::SymbolType;
use fhy_core::identifier::Identifier;
use fhy_core::param::{IntegerDomain, OrdinalDomain, ParamDomain, ParamError, Side};
use fhy_core::solver::SatResult;

use crate::support::constraint::int;
use crate::support::param::{
    EvenDomain, RecordingParamObserver, at_least, context, in_set, ints, scripted_solver,
};

#[test]
fn custom_domain_answers_through_its_hooks() {
    let x = Identifier::new("x");
    let (domain, handle) = EvenDomain::build(false);
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);

    assert_eq!(
        domain.symbol_type().expect("answers"),
        Some(SymbolType::Int)
    );
    assert!(domain.is_value_admissible(&int(4)).expect("answers"));
    assert!(!domain.is_value_admissible(&int(3)).expect("answers"));
    domain
        .validate_constraint(&at_least(&x, 0), &x)
        .expect("allows");
    assert_eq!(domain.implied_constraints(&x).expect("answers").len(), 1);
    assert_eq!(domain.interval_profile().expect("answers"), None);
    assert_eq!(
        domain
            .has_feasible_value(Side::new(&[], &x), &context)
            .expect("answers"),
        Outcome::Satisfied
    );
    assert!(
        domain
            .union(
                Side::new(&[], &x),
                &domain,
                Side::new(&[], &x),
                &x,
                &context
            )
            .expect("answers")
            .is_none()
    );
    assert!(domain.is_structurally_equivalent(&domain));

    assert_eq!(
        handle.calls(),
        [
            "symbol_type",
            "is_value_admissible",
            "is_value_admissible",
            "validate_constraint",
            "implied_constraints",
            "interval_profile",
            "has_feasible_value",
            "union",
            "is_structurally_equivalent",
        ]
    );
}

#[test]
fn numeric_subset_asks_a_custom_other_side_its_sort_and_its_values() {
    let x = Identifier::new("x");
    let y = Identifier::new("y");
    let (other, handle) = EvenDomain::build(false);
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let own_constraints = [in_set(&x, ints([2, 3]))];

    let outcome = ParamDomain::from(IntegerDomain::new(false, true))
        .feasibility_subset(
            Side::new(&own_constraints, &x),
            &other,
            Side::new(&[], &y),
            &context(&solver, &observer),
        )
        .expect("decides");

    assert_eq!(outcome, Outcome::Violated);
    assert_eq!(
        handle.calls(),
        ["symbol_type", "is_value_admissible", "is_value_admissible"]
    );
}

#[test]
fn finite_domain_never_asks_a_custom_other_side() {
    let x = Identifier::new("x");
    let (other, handle) = EvenDomain::build(false);
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let own = ParamDomain::from(OrdinalDomain::new(ints([1])).expect("ordinal"));

    let outcome = own
        .feasibility_subset(
            Side::new(&[], &x),
            &other,
            Side::new(&[], &x),
            &context(&solver, &observer),
        )
        .expect("decides");

    assert_eq!(outcome, Outcome::Violated);
    assert!(!own.is_structurally_equivalent(&other));
    assert!(handle.calls().is_empty());
}

#[test]
fn custom_domain_failures_propagate() {
    let x = Identifier::new("x");
    let (domain, _handle) = EvenDomain::build(true);
    let (solver, _smt) = scripted_solver(SatResult::Sat);
    let observer = RecordingParamObserver::default();
    let context = context(&solver, &observer);

    assert!(matches!(domain.symbol_type(), Err(ParamError::Custom(_))));
    assert!(matches!(
        domain.is_value_admissible(&int(2)),
        Err(ParamError::Custom(_))
    ));
    assert!(matches!(
        domain.has_feasible_value(Side::new(&[], &x), &context),
        Err(ParamError::Custom(_))
    ));
    assert!(matches!(
        ParamDomain::from(IntegerDomain::new(false, true)).is_value_set_subset(&domain),
        Err(ParamError::Custom(_))
    ));
}
