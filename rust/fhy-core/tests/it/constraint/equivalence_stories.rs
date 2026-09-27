//! Tests for the structural and alpha equivalence of constraints.
//!
//! The cases are ported from `test_structural_equivalence.py`.

use std::collections::HashMap;

use fhy_core::constraint::{Constraint, EquationConstraint, Polarity, SetConstraint, Value};
use fhy_core::expression::Expression;
use fhy_core::identifier::Identifier;
use fhy_core::term::{AlphaEquivalence, AlphaRenaming};
use rstest::rstest;

use crate::support::constraint::{TestOpaque, int, int_set, member_set};

fn renaming(pairs: &[(&Identifier, &Identifier)]) -> AlphaRenaming {
    AlphaRenaming::new(
        pairs
            .iter()
            .map(|(left, right)| ((*left).clone(), (*right).clone()))
            .collect::<HashMap<_, _>>(),
    )
    .expect("an injective renaming")
}

#[test]
fn equations_are_structurally_equivalent_by_their_expressions() {
    let x = Identifier::new("x");
    let left = EquationConstraint::new(Expression::from(x.clone()).less(1));
    let right = EquationConstraint::new(Expression::from(x.clone()).less(1));
    let other = EquationConstraint::new(Expression::from(x).less(2));

    assert!(left.is_structurally_equivalent(&right));
    assert!(!left.is_structurally_equivalent(&other));
}

#[test]
fn equations_are_alpha_equivalent_under_a_renaming_of_their_identifiers() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let left = Constraint::from(EquationConstraint::new(Expression::from(x.clone()).less(1)));
    let right = Constraint::from(EquationConstraint::new(Expression::from(y.clone()).less(1)));

    assert!(left.is_alpha_equivalent_under(&right, &renaming(&[(&x, &y)])));
    assert!(!left.is_alpha_equivalent(&right));
    assert!(!left.is_structurally_equivalent(&right));
}

#[test]
fn set_constraints_are_equivalent_whatever_order_their_members_were_given_in() {
    let x = Identifier::new("x");
    let left = SetConstraint::new(x.clone(), int_set([1, 2, 3]), Polarity::In);
    let right = SetConstraint::new(x, int_set([3, 1, 2]), Polarity::In);

    assert!(left.is_structurally_equivalent(&right));
}

#[rstest]
#[case::a_boolean_for_an_integer(Value::Bool(true))]
#[case::a_float_for_an_integer(Value::Float(1.0))]
fn set_constraints_compare_their_members_type_strictly(#[case] other: Value) {
    let x = Identifier::new("x");
    let left = SetConstraint::new(x.clone(), member_set([int(1)]), Polarity::In);
    let right = SetConstraint::new(x, member_set([other]), Polarity::In);

    assert!(!left.is_structurally_equivalent(&right));
}

#[test]
fn set_constraints_of_another_polarity_or_variable_are_not_equivalent() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let left = SetConstraint::new(x.clone(), int_set([1]), Polarity::In);

    assert!(!left.is_structurally_equivalent(&SetConstraint::new(
        x,
        int_set([1]),
        Polarity::NotIn
    )));
    assert!(!left.is_structurally_equivalent(&SetConstraint::new(y, int_set([1]), Polarity::In)));
}

#[test]
fn set_constraints_are_alpha_equivalent_when_their_variables_correspond() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let left = SetConstraint::new(x.clone(), int_set([1]), Polarity::NotIn);
    let right = SetConstraint::new(y.clone(), int_set([1]), Polarity::NotIn);
    let other_members = SetConstraint::new(y.clone(), int_set([2]), Polarity::NotIn);

    assert!(left.is_alpha_equivalent_under(&right, &renaming(&[(&x, &y)])));
    assert!(!left.is_alpha_equivalent_under(&right, &AlphaRenaming::default()));
    assert!(!left.is_alpha_equivalent_under(&other_members, &renaming(&[(&x, &y)])));
}

#[test]
fn set_constraints_with_colliding_opaque_members_are_equivalent_in_either_order() {
    let x = Identifier::new("x");
    let build = |first: i64, second: i64| {
        SetConstraint::new(
            x.clone(),
            member_set([
                TestOpaque::colliding(first).into_value(),
                TestOpaque::colliding(second).into_value(),
            ]),
            Polarity::In,
        )
    };

    assert!(build(1, 2).is_structurally_equivalent(&build(2, 1)));
    assert!(!build(1, 2).is_structurally_equivalent(&build(1, 3)));
}

#[test]
fn constraints_of_different_kinds_are_not_equivalent() {
    let x = Identifier::new("x");
    let equation = Constraint::from(EquationConstraint::new(
        Expression::from(x.clone()).equals(1),
    ));
    let set = Constraint::from(SetConstraint::new(x, int_set([1]), Polarity::In));

    assert!(!equation.is_structurally_equivalent(&set));
    assert!(!equation.is_alpha_equivalent(&set));
    assert!(equation.is_structurally_equivalent(&equation.clone()));
}
