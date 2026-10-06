//! Properties of constraints: membership agrees with a reference
//! type-strict matcher, a member set does not depend on the order of its
//! members, the ordering key is equal exactly on structural equivalence,
//! and a system's satisfiability agrees with brute force; and of values
//! and members, identifiers among them: equality agrees with hashing and
//! the canonical order, an identifier equals no string or integer, and the
//! wire form round-trips, from the opaque form 0.2.0 wrote included.
//!
//! The satisfiability property runs on the z3 backend under the `z3`
//! feature, and otherwise when `FHY_SMT_SOLVER` names an SMT-LIB2
//! executable, such as `z3 -in`; without either, it passes trivially.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use fhy_core::constraint::wire::ValueData;
use fhy_core::constraint::{
    Binding, Bindings, Constraint, ConstraintContext, ConstraintSystem, EquationConstraint, Member,
    MemberKind, MemberSet, Outcome, Polarity, SetConstraint, Value,
};
use fhy_core::expression::{Expression, SymbolType};
use fhy_core::identifier::Identifier;
use fhy_core::solver::{CheckLimits, SmtSolver, Solver};
use proptest::prelude::*;

use crate::support::constraint::ConstraintKey;
use crate::support::constraint::{describe, int, member, text};
use crate::support::expression::build_expression_strategy;
use crate::support::foreign::{LEGACY_IDENTIFIER, LegacyResolver};
use crate::support::hashing::hash_of;
use crate::support::serde::check_serde_round_trip;

/// The identifiers values are drawn from: one name hint per id, the names
/// shared with the strings drawn and the ids with the integers, and two ids
/// whose decimal texts order opposite to the ids.
static IDENTIFIERS: std::sync::LazyLock<Vec<Identifier>> = std::sync::LazyLock::new(|| {
    [(1, "a"), (9, "b"), (10, "c"), (99, "ab"), (100, "x")]
        .into_iter()
        .map(|(id, name)| Identifier::try_restore(id, name).expect("below the cap"))
        .collect()
});

/// Return a strategy for member-shaped values: small integers, Booleans,
/// short strings, floats with repeats of one value, identifiers whose
/// names and ids the strings and integers reuse, and tuples of those.
fn build_value_strategy() -> BoxedStrategy<Value> {
    let leaf = prop_oneof![
        (-5_i64..10).prop_map(int),
        any::<bool>().prop_map(Value::Bool),
        "[abc]{0,2}".prop_map(|string: String| text(&string)),
        prop::sample::select(vec![0.0, -0.0, 0.5, 1.0, -2.0]).prop_map(Value::Float),
        prop::sample::select(IDENTIFIERS.clone()).prop_map(Value::Identifier),
    ];
    leaf.prop_recursive(2, 8, 3, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..3).prop_map(Value::Tuple),
            prop::collection::vec(inner, 0..3).prop_map(Value::FrozenSet),
        ]
    })
    .boxed()
}

/// Return a strategy for system members over `x` and `y`: equations,
/// set constraints over generated values, and set constraints over opaque
/// values whose keys all collide.
fn build_system_member_strategy() -> BoxedStrategy<Constraint> {
    static VARIABLES: std::sync::LazyLock<[Identifier; 2]> =
        std::sync::LazyLock::new(|| [Identifier::new("x"), Identifier::new("y")]);
    let variable = prop::sample::select(VARIABLES.to_vec());
    let polarity = prop::sample::select(vec![Polarity::In, Polarity::NotIn]);
    prop_oneof![
        build_expression_strategy(true)
            .prop_map(|expression| Constraint::from(EquationConstraint::new(expression))),
        (
            variable.clone(),
            polarity.clone(),
            prop::collection::vec(build_value_strategy(), 1..3)
        )
            .prop_map(|(variable, polarity, values)| {
                Constraint::from(SetConstraint::new(
                    variable,
                    crate::support::constraint::member_set(values),
                    polarity,
                ))
            }),
        (variable, polarity, 0_i64..3).prop_map(|(variable, polarity, payload)| {
            Constraint::from(SetConstraint::new(
                variable,
                crate::support::constraint::member_set([
                    crate::support::constraint::TestOpaque::colliding(payload).into_value(),
                ]),
                polarity,
            ))
        }),
    ]
    .boxed()
}

/// Return a strategy for built-in constraints over `x` and `y`: equations
/// and set constraints over generated values.
fn build_built_in_constraint_strategy() -> BoxedStrategy<Constraint> {
    static VARIABLES: std::sync::LazyLock<[Identifier; 2]> =
        std::sync::LazyLock::new(|| [Identifier::new("x"), Identifier::new("y")]);
    prop_oneof![
        build_expression_strategy(true)
            .prop_map(|expression| Constraint::from(EquationConstraint::new(expression))),
        (
            prop::sample::select(VARIABLES.to_vec()),
            prop::sample::select(vec![Polarity::In, Polarity::NotIn]),
            prop::collection::vec(build_value_strategy(), 1..3),
        )
            .prop_map(|(variable, polarity, values)| {
                Constraint::from(SetConstraint::new(
                    variable,
                    crate::support::constraint::member_set(values),
                    polarity,
                ))
            }),
    ]
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
        (Value::Identifier(left), Value::Identifier(right)) => left.id() == right.id(),
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
            left.key() == right.key(),
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
            left.key() == right.key(),
            left.is_structurally_equivalent(&right)
        );
        prop_assert_eq!(left.key(), left.key());
    }

    /// Keys are equal exactly when equations are structurally equivalent,
    /// over pairs whose sharing differs: each side is a generated tree with
    /// one identifier replaced by a generated subtree through `substitute`,
    /// which shares the subtree at every place the identifier occurs, and
    /// the right side is also compared unshared.
    #[test]
    fn keys_are_equal_exactly_when_constraints_are_structurally_equivalent(
        left in build_expression_strategy(true),
        right in build_expression_strategy(true),
        replacement in build_expression_strategy(true),
    ) {
        let substituted = |tree: &Expression| {
            let mapping: HashMap<Identifier, Expression> = tree
                .free_identifiers()
                .into_iter()
                .take(1)
                .map(|identifier| (identifier, replacement.clone()))
                .collect();
            tree.substitute(&mapping).unwrap_or_else(|_| tree.clone())
        };
        let (left, right) = (substituted(&left), substituted(&right));
        let unshared = crate::support::expression::copy_deeply(&right);
        let [left, right, unshared] = [left, right, unshared]
            .map(|tree| Constraint::from(EquationConstraint::new(tree)));

        prop_assert_eq!(
            left.key() == right.key(),
            left.is_structurally_equivalent(&right)
        );
        prop_assert_eq!(right.key(), unshared.key());
    }

    /// A system's member keys, its equivalence and its hash do not depend
    /// on the order its members are given in, even when opaque members'
    /// keys collide: the members draw equations, set constraints over
    /// generated values, and set constraints over opaque values whose keys
    /// are all one.
    #[test]
    fn system_order_and_equivalence_do_not_depend_on_the_input_order(
        (members, shuffled) in prop::collection::vec(build_system_member_strategy(), 1..7)
            .prop_flat_map(|members| (Just(members.clone()), Just(members).prop_shuffle())),
    ) {
        let forward = ConstraintSystem::new(members).expect("no key fails");
        let backward = ConstraintSystem::new(shuffled).expect("no key fails");
        let keys = |system: &ConstraintSystem| -> Vec<String> {
            system.constraints().iter().map(ConstraintKey::key).collect()
        };

        prop_assert_eq!(keys(&forward), keys(&backward));
        prop_assert!(forward.is_structurally_equivalent(&backward));
        prop_assert_eq!(hash_of(&forward), hash_of(&backward));
    }

    /// `Ord` for constraints is a total order that agrees with equivalence:
    /// equal exactly for equivalent constraints, antisymmetric, transitive,
    /// and the order of their keys.
    #[test]
    fn constraint_order_is_total_and_agrees_with_equivalence(
        a in build_built_in_constraint_strategy(),
        b in build_built_in_constraint_strategy(),
        c in build_built_in_constraint_strategy(),
    ) {
        use std::cmp::Ordering;
        prop_assert_eq!(a.cmp(&b) == Ordering::Equal, a == b);
        prop_assert_eq!(a.cmp(&b), b.cmp(&a).reverse());
        prop_assert_eq!(a.partial_cmp(&b), Some(a.cmp(&b)));
        prop_assert_eq!(a.cmp(&b), a.key().cmp(&b.key()));
        prop_assert_eq!(a.cmp(&a.clone()), Ordering::Equal);
        if a <= b && b <= c {
            prop_assert!(a <= c);
        }
    }

    #[test]
    fn system_satisfiability_agrees_with_brute_force(
        bounds in prop::collection::vec(build_bound_strategy(), 1..5),
    ) {
        let Some(backend) = property_backend() else {
            return Ok(());
        };
        let x = Identifier::new("x");
        let system = ConstraintSystem::new(bounds.iter().map(|bound| bound.to_constraint(&x))).expect("every member has a key");
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

/// Return `wire` with every identifier written as 0.2.0 wrote it, an opaque
/// part of type id `id` whose payload is the identifier's JSON text.
fn to_legacy_form(wire: serde_json::Value) -> serde_json::Value {
    use serde_json::Value as Json;
    match wire {
        Json::Object(fields) => Json::Object(
            fields
                .into_iter()
                .map(|(tag, inner)| {
                    if tag == "identifier" {
                        let payload = inner.to_string();
                        (
                            "opaque".to_owned(),
                            serde_json::json!({"type_id": LEGACY_IDENTIFIER, "data": payload}),
                        )
                    } else {
                        (tag, to_legacy_form(inner))
                    }
                })
                .collect(),
        ),
        Json::Array(elements) => Json::Array(elements.into_iter().map(to_legacy_form).collect()),
        other => other,
    }
}

proptest! {
    /// A value and its member, identifiers at any depth included,
    /// round-trip through JSON and postcard, and their JSON re-encodes
    /// byte-identically.
    #[test]
    fn values_and_members_round_trip_through_serde(value in build_value_strategy()) {
        check_serde_round_trip(&value)?;
        check_serde_round_trip(&member(value))?;
    }

    /// Writing each identifier of a value's wire form as 0.2.0's opaque part
    /// reads back, through a resolver whose part reports the identifier, as
    /// the same value, which writes the current form again.
    #[test]
    fn legacy_opaque_identifiers_read_as_identifiers(value in build_value_strategy()) {
        let wire: serde_json::Value = serde_json::to_value(&value).expect("encodes");
        let legacy = to_legacy_form(wire.clone());

        let data: ValueData = serde_json::from_value(legacy).expect("reads");
        let decoded = data.clone().build(&LegacyResolver).expect("resolves");
        let decoded_member = data.build_member(&LegacyResolver).expect("resolves");

        prop_assert_eq!(&decoded, &value);
        prop_assert_eq!(serde_json::to_value(&decoded).expect("encodes"), wire);
        prop_assert_eq!(decoded_member, member(value));
    }

    /// Value equality is reflexive and symmetric and agrees with hashing.
    #[test]
    fn value_equality_is_an_equivalence_that_agrees_with_hashing(
        left in build_value_strategy(),
        right in build_value_strategy(),
    ) {
        prop_assert_eq!(&left, &left.clone());
        prop_assert_eq!(left == right, right == left);
        prop_assert_eq!(left == right, is_type_strictly_equal(&left, &right));
        if left == right {
            prop_assert_eq!(hash_of(&left), hash_of(&right));
        }
    }

    /// The canonical member order is total, antisymmetric and transitive,
    /// equal exactly on equal members, and agrees with hashing.
    #[test]
    fn member_order_is_total_and_agrees_with_equality_and_hashing(
        a in build_value_strategy(),
        b in build_value_strategy(),
        c in build_value_strategy(),
    ) {
        use std::cmp::Ordering;
        let [a, b, c] = [a, b, c].map(member);
        let ab = a.partial_cmp(&b);

        prop_assert!(ab.is_some(), "{:?} and {:?} have no order", a.kind(), b.kind());
        prop_assert_eq!(ab, b.partial_cmp(&a).map(Ordering::reverse));
        prop_assert_eq!(ab == Some(Ordering::Equal), a == b);
        if a == b {
            prop_assert_eq!(hash_of(&a), hash_of(&b));
        }
        if a <= b && b <= c {
            prop_assert!(a <= c);
        }
    }

    /// An identifier equals no string and no integer, its name and id
    /// included, as a value, as a member and in a set.
    #[test]
    fn identifier_never_equals_a_string_or_an_integer(
        identifier in prop::sample::select(IDENTIFIERS.clone()),
        other_text in "[abcx]{0,2}",
        other_int in 0_i64..120,
    ) {
        let value = Value::Identifier(identifier.clone());
        let set = MemberSet::new([member(value.clone())]);
        let id = i64::try_from(identifier.id()).expect("a small id");

        for other in [
            text(&other_text),
            text(identifier.name_hint()),
            int(other_int),
            int(id),
        ] {
            prop_assert_ne!(&value, &other);
            prop_assert_ne!(member(value.clone()), member(other.clone()));
            prop_assert!(!set.contains_value(&other));
        }
    }
}
