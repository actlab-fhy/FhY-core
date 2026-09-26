//! Properties of the domains: the ordinal order, and the finite domains'
//! questions against brute force.

use std::collections::BTreeSet;

use fhy_core::constraint::{Constraint, MemberKind, Outcome, Value};
use fhy_core::identifier::Identifier;
use fhy_core::param::{CategoricalDomain, OrdinalDomain, ParamDomain, Side};
use fhy_core::solver::SatResult;
use proptest::prelude::*;

use crate::support::constraint::int;
use crate::support::param::{
    RecordingParamObserver, boolean, context, float, in_set, not_in_set, scripted_solver,
};

/// A number of the ordinal order, and its value as a float.
fn number() -> impl Strategy<Value = (Value, f64)> {
    prop_oneof![
        (-20_i32..20).prop_map(|value| (int(i64::from(value)), f64::from(value))),
        (-40_i32..40).prop_map(|half| {
            let value = f64::from(half) / 2.0;
            (float(value), value)
        }),
        any::<bool>().prop_map(|value| (boolean(value), f64::from(u8::from(value)))),
    ]
}

/// Return the numbers of `values`, keeping the first of type-strictly equal
/// ones.
fn distinct(values: Vec<(Value, f64)>) -> Vec<(Value, f64)> {
    let mut seen = BTreeSet::new();
    values
        .into_iter()
        .filter(|(value, number)| {
            let kind = match value {
                Value::Bool(_) => 0,
                Value::Float(_) => 1,
                _ => 2,
            };
            seen.insert((kind, number.to_bits()))
        })
        .collect()
}

/// Return the number a member of the generated domains denotes.
fn as_number(kind: MemberKind<'_>) -> f64 {
    match kind {
        MemberKind::Bool(value) => f64::from(u8::from(value)),
        MemberKind::Int(value) => value.to_string().parse().expect("a small integer"),
        MemberKind::Float(value) => value,
        _ => f64::NAN,
    }
}

/// A set constraint of the generated finite questions.
#[derive(Debug, Clone)]
struct SetBound {
    values: Vec<i64>,
    is_in: bool,
}

impl SetBound {
    fn build(&self, variable: &Identifier) -> Constraint {
        let values = self.values.iter().copied().map(int);
        if self.is_in {
            in_set(variable, values)
        } else {
            not_in_set(variable, values)
        }
    }

    fn holds(&self, value: i64) -> bool {
        self.values.contains(&value) == self.is_in
    }
}

fn set_bounds() -> impl Strategy<Value = Vec<SetBound>> {
    prop::collection::vec(
        (prop::collection::vec(0_i64..8, 0..5), any::<bool>())
            .prop_map(|(values, is_in)| SetBound { values, is_in }),
        0..4,
    )
}

fn domain_values() -> impl Strategy<Value = Vec<i64>> {
    prop::collection::btree_set(0_i64..8, 1..6).prop_map(|values| values.into_iter().collect())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn ordinal_order_is_ascending_and_independent_of_the_input_order(
        values in prop::collection::vec(number(), 1..8),
        seed in any::<u64>(),
    ) {
        let values = distinct(values);
        let mut shuffled = values.clone();
        let length = shuffled.len();
        for index in 0..length {
            let other = usize::try_from((seed.wrapping_mul(index as u64 + 1)) % length as u64)
                .expect("an index");
            shuffled.swap(index, other);
        }
        let domain = OrdinalDomain::new(values.into_iter().map(|(value, _)| value).collect())
            .expect("numbers order");
        let again = OrdinalDomain::new(shuffled.into_iter().map(|(value, _)| value).collect())
            .expect("numbers order");

        let numbers: Vec<f64> = domain.values().iter().map(|member| as_number(member.kind())).collect();
        prop_assert!(numbers.windows(2).all(|pair| pair[0] <= pair[1]));
        prop_assert!(domain.values() == again.values());
    }

    #[test]
    fn finite_feasibility_agrees_with_brute_force(
        values in domain_values(),
        bounds in set_bounds(),
    ) {
        let x = Identifier::new("x");
        let (solver, _smt) = scripted_solver(SatResult::Sat);
        let observer = RecordingParamObserver::default();
        let constraints: Vec<Constraint> = bounds.iter().map(|bound| bound.build(&x)).collect();
        let domain = ParamDomain::from(
            CategoricalDomain::new(values.iter().copied().map(int).collect()).expect("categorical"),
        );

        let outcome = domain
            .has_feasible_value(Side::new(&constraints, &x), &context(&solver, &observer))
            .expect("decides");

        let expected = values
            .iter()
            .any(|value| bounds.iter().all(|bound| bound.holds(*value)));
        prop_assert_eq!(outcome, if expected { Outcome::Satisfied } else { Outcome::Violated });
    }

    #[test]
    fn finite_subset_agrees_with_brute_force(
        own_values in domain_values(),
        own_bounds in set_bounds(),
        other_values in domain_values(),
        other_bounds in set_bounds(),
    ) {
        let x = Identifier::new("x");
        let y = Identifier::new("y");
        let (solver, _smt) = scripted_solver(SatResult::Sat);
        let observer = RecordingParamObserver::default();
        let own_constraints: Vec<Constraint> = own_bounds.iter().map(|bound| bound.build(&x)).collect();
        let other_constraints: Vec<Constraint> =
            other_bounds.iter().map(|bound| bound.build(&y)).collect();
        let build = |values: &[i64]| {
            ParamDomain::from(
                OrdinalDomain::new(values.iter().copied().map(int).collect()).expect("ordinal"),
            )
        };

        let outcome = build(&own_values)
            .feasibility_subset(
                Side::new(&own_constraints, &x),
                &build(&other_values),
                Side::new(&other_constraints, &y),
                &context(&solver, &observer),
            )
            .expect("decides");

        let is_valid = |values: &[i64], bounds: &[SetBound], value: i64| {
            values.contains(&value) && bounds.iter().all(|bound| bound.holds(value))
        };
        let expected = own_values
            .iter()
            .filter(|value| is_valid(&own_values, &own_bounds, **value))
            .all(|value| is_valid(&other_values, &other_bounds, *value));
        prop_assert_eq!(outcome, if expected { Outcome::Satisfied } else { Outcome::Violated });
    }
}
