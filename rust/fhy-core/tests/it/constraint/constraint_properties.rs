//! Properties of constraints: membership agrees with a reference
//! type-strict matcher, a member set does not depend on the order of its
//! members, and the ordering key is equal exactly on structural
//! equivalence.

use fhy_core::constraint::{
    Binding, Bindings, Constraint, ConstraintContext, EquationConstraint, Member, MemberKind,
    MemberSet, Outcome, Polarity, SetConstraint, Value,
};
use fhy_core::identifier::Identifier;
use fhy_core::solver::Solver;
use proptest::prelude::*;

use crate::support::constraint::{int, member, text};
use crate::support::expression::build_expression_strategy;

/// Return a strategy for member-shaped values: small integers, Booleans,
/// short strings, floats with repeats of one value, and tuples of those.
fn build_value_strategy() -> BoxedStrategy<Value> {
    let leaf = prop_oneof![
        (-5_i64..10).prop_map(int),
        any::<bool>().prop_map(Value::Bool),
        "[abc]{0,2}".prop_map(|string: String| text(&string)),
        prop::sample::select(vec![0.0, -0.0, 0.5, 1.0, -2.0]).prop_map(Value::Float),
    ];
    leaf.prop_recursive(2, 8, 3, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..3).prop_map(Value::Tuple),
            prop::collection::vec(inner, 0..3).prop_map(Value::FrozenSet),
        ]
    })
    .boxed()
}

/// Return whether `left` and `right` are equal type-strictly, by a direct
/// recursive comparison of the two values.
fn is_type_strictly_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Bool(left), Value::Bool(right)) => left == right,
        (Value::Int(left), Value::Int(right)) => left == right,
        (Value::Float(left), Value::Float(right)) => {
            // Python's `==` of floats: zeros equal, NaNs unequal.
            left.partial_cmp(right) == Some(std::cmp::Ordering::Equal)
        }
        (Value::Str(left), Value::Str(right)) => left == right,
        (Value::Tuple(left), Value::Tuple(right)) => {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| is_type_strictly_equal(left, right))
        }
        (Value::FrozenSet(left), Value::FrozenSet(right)) => {
            left.iter().all(|value| {
                right
                    .iter()
                    .any(|other| is_type_strictly_equal(value, other))
            }) && right.iter().all(|value| {
                left.iter()
                    .any(|other| is_type_strictly_equal(value, other))
            })
        }
        _ => false,
    }
}

/// Return the canonical sequence of `set`, as debug texts.
fn describe(set: &MemberSet) -> Vec<String> {
    set.iter()
        .map(|member| format!("{:?}", member.kind()))
        .collect()
}

proptest! {
    #[test]
    fn set_membership_agrees_with_a_type_strict_reference(
        values in prop::collection::vec(build_value_strategy(), 0..6),
        bound in build_value_strategy(),
    ) {
        let x = Identifier::new("x");
        let members: MemberSet = values.iter().cloned().map(member).collect();
        let constraint = SetConstraint::new(x.clone(), members, Polarity::In);
        let solver = Solver::new();
        let bindings = Bindings::from_iter([(x, Binding::Value(bound.clone()))]);

        let outcome = constraint
            .evaluate(&bindings, &ConstraintContext::new(&solver))
            .expect("a member-shaped value is decided");

        let expected = values.iter().any(|value| is_type_strictly_equal(value, &bound));
        prop_assert_eq!(outcome, if expected { Outcome::Satisfied } else { Outcome::Violated });
    }

    #[test]
    fn member_set_does_not_depend_on_the_order_of_its_members(
        values in prop::collection::vec(build_value_strategy(), 0..8),
        seed in any::<u64>(),
    ) {
        let mut shuffled = values.clone();
        let length = shuffled.len();
        if length > 1 {
            for index in 0..length {
                let other = usize::try_from(seed.rotate_left(u32::try_from(index).unwrap_or(0)) % length as u64)
                    .unwrap_or(0);
                shuffled.swap(index, other);
            }
        }
        let forward: MemberSet = values.into_iter().map(member).collect();
        let backward: MemberSet = shuffled.into_iter().map(member).collect();

        prop_assert_eq!(describe(&forward), describe(&backward));
        prop_assert!(forward == backward);
    }

    #[test]
    fn member_set_holds_no_two_equal_members_in_canonical_order(
        values in prop::collection::vec(build_value_strategy(), 0..8),
    ) {
        let set: MemberSet = values.into_iter().map(member).collect();
        let members: Vec<&Member> = set.iter().collect();

        for pair in members.windows(2) {
            prop_assert!(pair[0] < pair[1], "{:?} then {:?}", pair[0].kind(), pair[1].kind());
        }
        prop_assert!(members.iter().all(|member| !matches!(member.kind(), MemberKind::Float(value) if value == 0.0 && value.is_sign_negative())));
    }

    #[test]
    fn set_key_is_equal_exactly_for_equivalent_set_constraints(
        left in prop::collection::vec(build_value_strategy(), 0..4),
        right in prop::collection::vec(build_value_strategy(), 0..4),
        same_polarity in any::<bool>(),
    ) {
        let x = Identifier::new("x");
        let left = Constraint::from(SetConstraint::new(
            x.clone(),
            left.into_iter().map(member).collect(),
            Polarity::In,
        ));
        let right = Constraint::from(SetConstraint::new(
            x,
            right.into_iter().map(member).collect(),
            if same_polarity { Polarity::In } else { Polarity::NotIn },
        ));

        prop_assert_eq!(
            left.ordering_key() == right.ordering_key(),
            left.is_structurally_equivalent(&right)
        );
    }

    #[test]
    fn equation_key_is_equal_exactly_for_equivalent_equations(
        left in build_expression_strategy(true),
        right in build_expression_strategy(true),
    ) {
        let left = Constraint::from(EquationConstraint::new(left));
        let right = Constraint::from(EquationConstraint::new(right));

        prop_assert_eq!(
            left.ordering_key() == right.ordering_key(),
            left.is_structurally_equivalent(&right)
        );
        prop_assert_eq!(left.ordering_key(), left.clone().ordering_key());
    }
}
