//! Tests for the canonical ordering keys: equal for structurally equivalent
//! constraints, distinct otherwise, and computed on a small stack for a
//! deep expression.
//!
//! The cases are ported from `test_ordering_key.py`.

use fhy_core::constraint::{Constraint, EquationConstraint, Polarity, SetConstraint, Value};
use fhy_core::expression::{Decimal, Expression, LiteralValue};
use fhy_core::identifier::Identifier;
use rstest::rstest;

use crate::support::constraint::{TestOpaque, int, int_set, member_set, text};
use crate::support::expression::{build_deep_sum, build_parsed_literal};
use crate::support::stack::{SMALL_STACK_DEPTH, run_on_small_stack};

fn equation(expression: Expression) -> Constraint {
    Constraint::from(EquationConstraint::new(expression))
}

fn decimal(text: &str) -> Expression {
    Expression::literal(text.parse::<Decimal>().expect("a decimal text"))
}

#[test]
fn keys_start_with_the_kind() {
    let x = Identifier::new("x");

    assert!(
        equation(Expression::from(x.clone()).less(1))
            .ordering_key()
            .starts_with("equation|")
    );
    assert!(
        Constraint::from(SetConstraint::new(x.clone(), int_set([1]), Polarity::In))
            .ordering_key()
            .starts_with("in_set|")
    );
    assert!(
        Constraint::from(SetConstraint::new(x, int_set([1]), Polarity::NotIn))
            .ordering_key()
            .starts_with("not_in_set|")
    );
}

#[rstest]
#[case::an_integer_and_its_padded_text(Expression::literal(5), build_parsed_literal("05"))]
#[case::a_decimal_and_its_padded_text(decimal("1.5"), build_parsed_literal("1.50"))]
#[case::two_nans(Expression::literal(f64::NAN), Expression::literal(f64::NAN))]
#[case::two_zeros(Expression::literal(-0.0), Expression::literal(0.0))]
fn equal_literals_key_alike(#[case] left: Expression, #[case] right: Expression) {
    let x = Identifier::new("x");
    let reference = Expression::from(x);

    assert_eq!(
        equation(reference.equals(left)).ordering_key(),
        equation(reference.equals(right)).ordering_key()
    );
}

#[test]
fn equations_built_apart_over_one_identifier_key_alike() {
    let x = Identifier::new("x");
    let build = || {
        equation(Expression::all([
            Expression::from(x.clone()).greater(0),
            Expression::from(x.clone()).less(10),
        ]))
    };

    assert_eq!(build().ordering_key(), build().ordering_key());
}

#[rstest]
#[case::another_comparison(Expression::from(Identifier::new("x")).less(1), Expression::from(Identifier::new("x")).less_equal(1))]
#[case::another_identifier_with_the_same_name(Expression::from(Identifier::new("x")).less(1), Expression::from(Identifier::new("x")).less(1))]
#[case::a_float_and_a_decimal(Expression::literal(true).equals(Expression::literal(1.5)), Expression::literal(true).equals(decimal("1.5")))]
#[case::an_integer_and_a_boolean(Expression::literal(true).equals(Expression::literal(1)), Expression::literal(true).equals(Expression::literal(true)))]
#[case::and_and_or(Expression::literal(true).and(Expression::literal(false)), Expression::literal(true).or(Expression::literal(false)))]
#[case::flat_and_nested(
    Expression::all([Expression::literal(true), Expression::literal(true), Expression::literal(true)]),
    Expression::all([Expression::literal(true), Expression::all([Expression::literal(true), Expression::literal(true)])])
)]
#[case::decimals_apart_in_the_thirtieth_digit(
    Expression::literal(true).equals(decimal("0.100000000000000000000000000001")),
    Expression::literal(true).equals(decimal("0.100000000000000000000000000002"))
)]
fn different_equations_key_apart(#[case] left: Expression, #[case] right: Expression) {
    assert_ne!(
        equation(left).ordering_key(),
        equation(right).ordering_key()
    );
}

#[test]
fn set_keys_hold_the_polarity_the_variable_and_the_members() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let key = |variable: &Identifier, members: &[i64], polarity| {
        SetConstraint::new(variable.clone(), int_set(members.iter().copied()), polarity)
            .ordering_key()
    };

    assert_eq!(
        key(&x, &[1, 2], Polarity::In),
        key(&x, &[2, 1, 2], Polarity::In)
    );
    assert_ne!(
        key(&x, &[1, 2], Polarity::In),
        key(&x, &[1, 2], Polarity::NotIn)
    );
    assert_ne!(
        key(&x, &[1, 2], Polarity::In),
        key(&y, &[1, 2], Polarity::In)
    );
    assert_ne!(
        key(&x, &[1, 2], Polarity::In),
        key(&x, &[1, 3], Polarity::In)
    );
}

#[rstest]
#[case::an_integer_and_a_boolean(int(1), Value::Bool(true))]
#[case::an_integer_and_a_float(int(1), Value::Float(1.0))]
#[case::an_integer_and_its_text(int(1), text("1"))]
#[case::a_tuple_and_a_frozen_set(Value::Tuple(vec![int(1)]), Value::FrozenSet(vec![int(1)]))]
#[case::a_string_holding_a_separator(text("a,b"), Value::Tuple(vec![text("a"), text("b")]))]
fn different_members_key_apart(#[case] left: Value, #[case] right: Value) {
    let x = Identifier::new("x");
    let key = |value: Value| {
        SetConstraint::new(x.clone(), member_set([value]), Polarity::In).ordering_key()
    };

    assert_ne!(key(left), key(right));
}

#[test]
fn colliding_opaque_members_key_alike_in_either_order() {
    let x = Identifier::new("x");
    let key = |first: i64, second: i64| {
        SetConstraint::new(
            x.clone(),
            member_set([
                TestOpaque::colliding(first).into_value(),
                TestOpaque::colliding(second).into_value(),
            ]),
            Polarity::In,
        )
        .ordering_key()
    };

    assert_eq!(key(1, 2), key(2, 1));
}

#[test]
fn equation_key_is_computed_on_a_small_stack() {
    let key = run_on_small_stack(|| {
        let x = Expression::from(Identifier::new("x"));
        let deep = build_deep_sum(&x, SMALL_STACK_DEPTH).less(0);
        let key = equation(deep).ordering_key();
        key.len()
    });

    assert!(key > SMALL_STACK_DEPTH);
}

#[test]
fn literal_values_the_key_distinguishes_are_unequal_literals() {
    let pairs = [
        (LiteralValue::Int(1.into()), LiteralValue::Float(1.0)),
        (
            LiteralValue::Float(1.0),
            LiteralValue::Decimal("1".parse().expect("a decimal")),
        ),
    ];
    for (left, right) in pairs {
        assert_ne!(left, right);
        assert_ne!(
            equation(Expression::literal(true).equals(Expression::literal(left))).ordering_key(),
            equation(Expression::literal(true).equals(Expression::literal(right))).ordering_key()
        );
    }
}
