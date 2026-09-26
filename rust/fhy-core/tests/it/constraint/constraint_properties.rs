//! Properties of constraints: membership agrees with a reference
//! type-strict matcher, a member set does not depend on the order of its
//! members, the ordering key is equal exactly on structural equivalence,
//! and a system's satisfiability agrees with brute force.
//!
//! The satisfiability property runs on the z3 backend under the `z3`
//! feature, and otherwise when `FHY_SMT_SOLVER` names an SMT-LIB2
//! executable, such as `z3 -in`; without either, it passes trivially.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use fhy_core::constraint::{
    Binding, Bindings, Constraint, ConstraintContext, ConstraintSystem, EquationConstraint, Member,
    MemberKind, MemberSet, Outcome, Polarity, SetConstraint, Value,
};
use fhy_core::expression::{Expression, SymbolType};
use fhy_core::identifier::Identifier;
use fhy_core::solver::{CheckLimits, SmtSolver, Solver};
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

/// Return the SMT solver the solver-backed property runs on, if any.
#[cfg(feature = "z3")]
#[expect(
    clippy::unnecessary_wraps,
    reason = "without the feature, there may be no backend"
)]
fn property_backend() -> Option<Arc<dyn SmtSolver>> {
    Some(Arc::new(fhy_core::solver::Z3Solver::new()))
}

/// Return the SMT solver the solver-backed property runs on, if any.
#[cfg(not(feature = "z3"))]
fn property_backend() -> Option<Arc<dyn SmtSolver>> {
    let configured = std::env::var("FHY_SMT_SOLVER").ok()?;
    let mut words = configured.split_whitespace();
    let program = words.next()?;
    Some(Arc::new(
        fhy_core::solver::SmtLib2Process::new(program).with_args(words),
    ))
}

/// A member of the generated systems over one integer identifier.
#[derive(Debug, Clone)]
enum Bound {
    AtLeast(i64),
    Below(i64),
    In(Vec<i64>),
    NotIn(Vec<i64>),
}

impl Bound {
    /// Return whether `value` satisfies the member.
    fn holds(&self, value: i64) -> bool {
        match self {
            Self::AtLeast(bound) => value >= *bound,
            Self::Below(bound) => value < *bound,
            Self::In(members) => members.contains(&value),
            Self::NotIn(members) => !members.contains(&value),
        }
    }

    /// Return the member as a constraint on `x`.
    fn to_constraint(&self, x: &Identifier) -> Constraint {
        let reference = Expression::from(x.clone());
        let set = |members: &[i64], polarity| {
            Constraint::from(SetConstraint::new(
                x.clone(),
                members.iter().map(|value| member(int(*value))).collect(),
                polarity,
            ))
        };
        match self {
            Self::AtLeast(bound) => {
                Constraint::from(EquationConstraint::new(reference.greater_equal(*bound)))
            }
            Self::Below(bound) => Constraint::from(EquationConstraint::new(reference.less(*bound))),
            Self::In(members) => set(members, Polarity::In),
            Self::NotIn(members) => set(members, Polarity::NotIn),
        }
    }
}

/// Return a strategy for members whose constants lie in `[-5, 5]`, so a
/// satisfiable system has a solution in `[-10, 10]`.
fn build_bound_strategy() -> BoxedStrategy<Bound> {
    let constant = -5_i64..=5;
    prop_oneof![
        constant.clone().prop_map(Bound::AtLeast),
        constant.clone().prop_map(Bound::Below),
        prop::collection::vec(constant.clone(), 0..4).prop_map(Bound::In),
        prop::collection::vec(constant, 0..4).prop_map(Bound::NotIn),
    ]
    .boxed()
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

    #[test]
    fn system_satisfiability_agrees_with_brute_force(
        bounds in prop::collection::vec(build_bound_strategy(), 1..5),
    ) {
        let Some(backend) = property_backend() else {
            return Ok(());
        };
        let x = Identifier::new("x");
        let system = ConstraintSystem::new(bounds.iter().map(|bound| bound.to_constraint(&x)));
        let solver = Solver::new().with_shared_smt_solver(backend);
        let symbol_types = HashMap::from([(x, SymbolType::Int)]);
        let limits = CheckLimits::new().with_timeout(Duration::from_secs(2));

        let outcome = system
            .check_satisfiability(&symbol_types, limits, &ConstraintContext::new(&solver))
            .expect("the question is answered");

        let expected = (-10..=10).any(|value| bounds.iter().all(|bound| bound.holds(value)));
        if outcome != Outcome::Undecided {
            prop_assert_eq!(
                outcome,
                if expected { Outcome::Satisfied } else { Outcome::Violated }
            );
        }
    }
}
