//! Decision rules of constraints that only the Python suites, or neither
//! suite, pinned (F2-028): membership of opaque values alone and inside
//! containers, the position a rebound identifier keeps, and a system with
//! an undecided member.

use fhy_core::constraint::{
    Binding, Bindings, Constraint, ConstraintContext, ConstraintSystem, MemberSet, Outcome,
    Polarity, SetConstraint, Value,
};
use fhy_core::identifier::Identifier;
use fhy_core::solver::Solver;
use proptest::prelude::*;
use rstest::rstest;

use crate::support::constraint::{TestOpaque, int, member};

fn token(payload: i64) -> Value {
    TestOpaque::token(payload).into_value()
}

fn colliding(payload: i64) -> Value {
    TestOpaque::colliding(payload).into_value()
}

/// Return the outcome of `bound` against the in-set constraint of `members`.
fn membership(members: Vec<Value>, bound: Value) -> Outcome {
    let x = Identifier::new("x");
    let constraint = SetConstraint::new(
        x.clone(),
        members.into_iter().map(member).collect::<MemberSet>(),
        Polarity::In,
    );
    let solver = Solver::new();
    constraint
        .evaluate(
            &Bindings::from_iter([(x, Binding::Value(bound))]),
            &ConstraintContext::new(&solver),
        )
        .expect("a member-shaped, hashable value is decided")
}

#[rstest]
#[case::alone(vec![token(1), int(1)], token(1), Outcome::Satisfied)]
#[case::another_payload(vec![token(1)], token(2), Outcome::Violated)]
#[case::an_integer_is_no_token(vec![token(1)], int(1), Outcome::Violated)]
#[case::colliding_keys_are_not_equal(vec![colliding(1)], colliding(2), Outcome::Violated)]
#[case::colliding_and_equal(vec![colliding(2), colliding(1)], colliding(1), Outcome::Satisfied)]
#[case::inside_a_tuple(
    vec![Value::Tuple(vec![token(1), int(2)])],
    Value::Tuple(vec![token(1), int(2)]),
    Outcome::Satisfied
)]
#[case::inside_a_tuple_in_another_position(
    vec![Value::Tuple(vec![token(1), int(2)])],
    Value::Tuple(vec![int(2), token(1)]),
    Outcome::Violated
)]
#[case::inside_a_frozenset_in_any_order(
    vec![Value::FrozenSet(vec![token(1), token(2)])],
    Value::FrozenSet(vec![token(2), token(1), token(2)]),
    Outcome::Satisfied
)]
#[case::inside_a_frozenset_with_another_token(
    vec![Value::FrozenSet(vec![token(1), token(2)])],
    Value::FrozenSet(vec![token(1), token(3)]),
    Outcome::Violated
)]
fn opaque_membership_is_decided_by_the_value_s_equality(
    #[case] members: Vec<Value>,
    #[case] bound: Value,
    #[case] expected: Outcome,
) {
    assert_eq!(membership(members, bound), expected);
}

/// A value built from small integers and tokens, of which some collide.
fn opaque_value_strategy() -> BoxedStrategy<Value> {
    let leaf = prop_oneof![
        (0_i64..3).prop_map(int),
        (0_i64..3).prop_map(token),
        (0_i64..3).prop_map(colliding),
    ];
    leaf.prop_recursive(2, 6, 3, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..3).prop_map(Value::Tuple),
            prop::collection::vec(inner, 0..3).prop_map(Value::FrozenSet),
        ]
    })
    .boxed()
}

/// Return whether `left` equals `right` type-strictly, a token by its type
/// and payload, by a direct recursive comparison.
fn is_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Int(left), Value::Int(right)) => left == right,
        (Value::Opaque(left), Value::Opaque(right)) => left == right,
        (Value::Tuple(left), Value::Tuple(right)) => {
            left.len() == right.len() && left.iter().zip(right).all(|(l, r)| is_equal(l, r))
        }
        (Value::FrozenSet(left), Value::FrozenSet(right)) => {
            left.iter()
                .all(|value| right.iter().any(|other| is_equal(value, other)))
                && right
                    .iter()
                    .all(|value| left.iter().any(|other| is_equal(value, other)))
        }
        _ => false,
    }
}

proptest! {
    /// Membership with opaque members, alone and nested, agrees with a
    /// type-strict reference that compares tokens by their own equality.
    #[test]
    fn opaque_set_membership_agrees_with_a_type_strict_reference(
        values in prop::collection::vec(opaque_value_strategy(), 0..5),
        bound in opaque_value_strategy(),
    ) {
        let expected = values.iter().any(|value| is_equal(value, &bound));

        let outcome = membership(values, bound);

        prop_assert_eq!(outcome, if expected { Outcome::Satisfied } else { Outcome::Violated });
    }

    /// Binding an identifier again replaces its binding in place: the
    /// bindings list each identifier once, in the order first bound, with
    /// the last value.
    #[test]
    fn rebinding_an_identifier_keeps_its_position(
        operations in prop::collection::vec((0_usize..4, 0_i64..10), 0..20),
    ) {
        let names: Vec<Identifier> = (0..4).map(|index| Identifier::new(&format!("v{index}"))).collect();
        let mut bindings = Bindings::new();
        let mut model: Vec<(usize, i64)> = Vec::new();
        for (name, value) in operations {
            bindings.insert(names[name].clone(), Binding::Value(int(value)));
            match model.iter_mut().find(|(held, _)| *held == name) {
                Some(entry) => entry.1 = value,
                None => model.push((name, value)),
            }
        }

        let listed: Vec<(Identifier, Value)> = bindings
            .iter()
            .map(|(identifier, binding)| {
                let Binding::Value(value) = binding else { panic!("a value binding") };
                (identifier.clone(), value.clone())
            })
            .collect();
        let expected: Vec<(Identifier, Value)> = model
            .iter()
            .map(|(name, value)| (names[*name].clone(), int(*value)))
            .collect();
        prop_assert_eq!(bindings.len(), expected.len());
        prop_assert_eq!(listed, expected);
    }

    /// A system with one member left undecided, a set constraint on an
    /// unbound identifier, is undecided unless a decided member is
    /// violated.
    #[test]
    fn a_system_with_one_undecided_member_is_undecided_unless_violated(
        members in prop::collection::vec((prop::collection::vec(0_i64..5, 1..4), any::<bool>()), 0..4),
        value in 0_i64..5,
    ) {
        let (x, unbound) = (Identifier::new("x"), Identifier::new("unbound"));
        let mut constraints: Vec<Constraint> = members
            .iter()
            .map(|(values, is_in)| {
                Constraint::from(SetConstraint::new(
                    x.clone(),
                    values.iter().copied().map(|v| member(int(v))).collect::<MemberSet>(),
                    if *is_in { Polarity::In } else { Polarity::NotIn },
                ))
            })
            .collect();
        constraints.push(Constraint::from(SetConstraint::new(
            unbound,
            [member(int(0))].into_iter().collect::<MemberSet>(),
            Polarity::In,
        )));
        let system = ConstraintSystem::new(constraints).expect("keys");
        let solver = Solver::new();

        let outcome = system
            .evaluate(
                &Bindings::from_iter([(x, Binding::Value(int(value)))]),
                &ConstraintContext::new(&solver),
            )
            .expect("decides");

        let is_violated = members
            .iter()
            .any(|(values, is_in)| values.contains(&value) != *is_in);
        prop_assert_eq!(outcome, if is_violated { Outcome::Violated } else { Outcome::Undecided });
    }
}
