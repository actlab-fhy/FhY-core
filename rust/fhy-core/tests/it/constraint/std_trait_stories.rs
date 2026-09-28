//! Stories of the std traits of values, members, bindings, constraints and
//! systems: `==` is structural equivalence, `Hash` agrees with it, and
//! `Display` writes them for people.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use fhy_core::constraint::{
    Binding, Constraint, ConstraintSystem, EquationConstraint, Outcome, Polarity, SetConstraint,
    Value,
};
use fhy_core::expression::{BigInt, Decimal, Expression};
use fhy_core::identifier::Identifier;
use rstest::rstest;

use crate::support::constraint::{TestCustom, TestOpaque, int, int_set, member, member_set, text};
use crate::support::hashing::hash_of;

/// Assert that `==` on every pair of `items` is `equivalent`, and that
/// equal items hash alike and are found in a `HashSet` of the others.
fn assert_agrees_with<T: Eq + std::hash::Hash + std::fmt::Debug>(
    items: &[T],
    equivalent: impl Fn(&T, &T) -> bool,
) {
    let set: HashSet<&T> = items.iter().collect();
    for left in items {
        assert!(set.contains(left), "{left:?} is found");
        for right in items {
            assert_eq!(
                left == right,
                equivalent(left, right),
                "{left:?} == {right:?}"
            );
            if left == right {
                assert_eq!(
                    hash_of(left),
                    hash_of(right),
                    "{left:?} hashes as {right:?}"
                );
            }
        }
    }
}

fn float(value: f64) -> Value {
    Value::Float(value)
}

#[rstest]
#[case::equal_ints(int(1), int(1), true)]
#[case::different_ints(int(1), int(2), false)]
#[case::bool_is_no_int(Value::Bool(true), int(1), false)]
#[case::float_is_no_int(float(1.0), int(1), false)]
#[case::signed_zeros(float(-0.0), float(0.0), true)]
#[case::nan_equals_nan(float(f64::NAN), float(f64::NAN), true)]
#[case::decimal_is_no_float(
    Value::Decimal("0.5".parse::<Decimal>().expect("a decimal")),
    float(0.5),
    false
)]
#[case::strings(text("a"), text("a"), true)]
#[case::tuples_in_order(Value::Tuple(vec![int(1), int(2)]), Value::Tuple(vec![int(1), int(2)]), true)]
#[case::tuples_out_of_order(Value::Tuple(vec![int(1), int(2)]), Value::Tuple(vec![int(2), int(1)]), false)]
#[case::sets_in_any_order_and_repeats(
    Value::FrozenSet(vec![int(1), int(2), int(2)]),
    Value::FrozenSet(vec![int(2), int(1)]),
    true
)]
#[case::sets_of_other_values(
    Value::FrozenSet(vec![int(1)]),
    Value::FrozenSet(vec![int(1), int(2)]),
    false
)]
#[case::opaque_values_through_eq_part(
    TestOpaque::token(1).into_value(),
    TestOpaque::token(1).into_value(),
    true
)]
#[case::unequal_opaque_values(
    TestOpaque::token(1).into_value(),
    TestOpaque::token(2).into_value(),
    false
)]
fn values_compare_structurally_and_type_strictly(
    #[case] left: Value,
    #[case] right: Value,
    #[case] is_equal: bool,
) {
    assert_eq!(left == right, is_equal);
    assert_eq!(right == left, is_equal);
    if is_equal {
        assert_eq!(hash_of(&left), hash_of(&right));
    }
}

#[test]
fn a_value_is_found_in_a_hash_set_of_equal_values() {
    let values = [
        int(1),
        Value::Int(BigInt::from(1)),
        float(0.0),
        float(-0.0),
        float(f64::NAN),
        text("a"),
        Value::FrozenSet(vec![int(2), int(1), int(1)]),
        Value::FrozenSet(vec![int(1), int(2)]),
        Value::Tuple(vec![int(1)]),
        TestOpaque::token(3).into_value(),
        TestOpaque::token(3).into_value(),
    ];

    let set: HashSet<Value> = values.iter().cloned().collect();

    assert_eq!(set.len(), 7);
    assert!(values.iter().all(|value| set.contains(value)));
}

#[rstest]
#[case::bool(Value::Bool(true), "true")]
#[case::int(int(-3), "-3")]
#[case::float(float(0.5), "0.5")]
#[case::string(text("a\"b"), r#""a\"b""#)]
#[case::pair(Value::Tuple(vec![int(1), text("x")]), r#"(1, "x")"#)]
#[case::single(Value::Tuple(vec![int(1)]), "(1,)")]
#[case::empty_tuple(Value::Tuple(Vec::new()), "()")]
#[case::set(Value::FrozenSet(vec![int(1), int(2)]), "{1, 2}")]
#[case::opaque(TestOpaque::token(1).into_value(), "<Token>")]
fn a_value_displays_for_people(#[case] value: Value, #[case] expected: &str) {
    assert_eq!(value.to_string(), expected);
}

#[test]
fn members_and_member_sets_are_eq_hash_and_display() {
    let left = member_set([int(2), int(1), text("a")]);
    let right = member_set([text("a"), int(1), int(2)]);

    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
    assert_eq!(left.to_string(), r#"{1, 2, "a"}"#);
    assert_eq!(member(float(-0.0)), member(float(0.0)));
    assert_eq!(hash_of(&member(float(-0.0))), hash_of(&member(float(0.0))));
    assert_eq!(
        member(Value::FrozenSet(vec![int(2), int(1)])).to_string(),
        "{1, 2}"
    );
    assert_eq!(
        member(TestOpaque::colliding(1).into_value()),
        member(TestOpaque::colliding(1).into_value())
    );
    assert_ne!(
        member(TestOpaque::colliding(1).into_value()),
        member(TestOpaque::colliding(2).into_value())
    );
}

#[test]
fn bindings_compare_expressions_and_values_apart() {
    let one = Binding::Expression(Expression::literal(1));

    assert_eq!(one, Binding::Expression(Expression::literal(1)));
    assert_ne!(one, Binding::Value(int(1)));
    assert_eq!(Binding::Value(int(1)), Binding::Value(int(1)));
    assert_eq!(
        hash_of(&Binding::Value(float(-0.0))),
        hash_of(&Binding::Value(float(0.0)))
    );
    assert_eq!(one.to_string(), "1");
    assert_eq!(Binding::Value(text("a")).to_string(), r#""a""#);
}

/// Return constraints over `x`: pairs built apart, and constraints of each
/// kind that differ.
fn constraints(x: &Identifier) -> Vec<Constraint> {
    let log = Arc::new(Mutex::new(Vec::new()));
    let reference = Expression::from(x.clone());
    vec![
        EquationConstraint::new(reference.clone().less(1)).into(),
        EquationConstraint::new(reference.clone().less(1)).into(),
        EquationConstraint::new(reference.clone().less(2)).into(),
        SetConstraint::new(x.clone(), int_set([1, 2]), Polarity::In).into(),
        SetConstraint::new(x.clone(), int_set([2, 1]), Polarity::In).into(),
        SetConstraint::new(x.clone(), int_set([1, 2]), Polarity::NotIn).into(),
        TestCustom::build("a", reference.clone().less(1), Outcome::Satisfied, &log),
        TestCustom::build("a", reference.clone().less(1), Outcome::Satisfied, &log),
        TestCustom::build("b", reference.less(1), Outcome::Satisfied, &log),
    ]
}

#[test]
fn constraints_equal_as_they_are_structurally_equivalent() {
    let x = Identifier::new("x");

    assert_agrees_with(&constraints(&x), Constraint::is_structurally_equivalent);
}

#[test]
fn equation_and_set_constraints_are_eq_on_their_own() {
    let x = Identifier::new("x");
    let equations = [
        EquationConstraint::new(Expression::from(x.clone()).less(1)),
        EquationConstraint::new(Expression::from(x.clone()).less(1)),
        EquationConstraint::new(Expression::from(x.clone()).less(2)),
    ];
    let sets = [
        SetConstraint::new(x.clone(), member_set([int(1), text("a")]), Polarity::In),
        SetConstraint::new(x, member_set([text("a"), int(1)]), Polarity::In),
        SetConstraint::new(Identifier::new("x"), int_set([1]), Polarity::In),
    ];

    assert_agrees_with(&equations, EquationConstraint::is_structurally_equivalent);
    assert_agrees_with(&sets, SetConstraint::is_structurally_equivalent);
}

#[test]
fn a_constraint_displays_for_people() {
    let x = Identifier::new("x");
    let [equation, _, _, set, _, not_in, custom, ..] =
        <[Constraint; 9]>::try_from(constraints(&x)).expect("nine constraints");

    assert_eq!(equation.to_string(), "(x < 1)");
    assert_eq!(set.to_string(), "x in {1, 2}");
    assert_eq!(not_in.to_string(), "x not in {1, 2}");
    assert_eq!(custom.to_string(), "<TestCustom>");
}

#[test]
fn systems_equal_as_their_canonical_members_do() {
    let x = Identifier::new("x");
    let [less_one, _, less_two, set, ..] =
        <[Constraint; 9]>::try_from(constraints(&x)).expect("nine constraints");
    let systems = [
        ConstraintSystem::new([less_one.clone(), set.clone()]).expect("a system"),
        ConstraintSystem::new([set.clone(), less_one.clone()]).expect("a system"),
        ConstraintSystem::new([less_two, set]).expect("a system"),
        ConstraintSystem::new([less_one]).expect("a system"),
        ConstraintSystem::default(),
    ];

    assert_agrees_with(&systems, ConstraintSystem::is_structurally_equivalent);
    assert_eq!(systems[0], systems[1]);
    assert_eq!(systems[3].to_string(), "(x < 1)");
    assert_eq!(systems[4].to_string(), "true");
    assert!(systems[0].to_string().contains(" and "));
}
