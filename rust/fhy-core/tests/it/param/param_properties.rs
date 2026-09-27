//! Properties of the domains: the ordinal order, and the finite domains'
//! questions against brute force.

use fhy_core::param::{Inclusivity, Sign, ZeroInclusion};
use std::collections::BTreeSet;

use fhy_core::constraint::{Binding, Bindings, Constraint, MemberKind, Outcome, Value};
use fhy_core::expression::{BigInt, LiteralValue};
use fhy_core::identifier::Identifier;
use fhy_core::param::{
    BoundSide, CategoricalDomain, IntervalIntegerDomain, Operand, OrdinalDomain, Param,
    ParamContext, ParamDomain, Side, ValueCheck,
};
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

/// Return whether the interval param `param` admits `value`.
fn admits(param: &Param, value: i64, context: &ParamContext<'_>) -> bool {
    let environment = param
        .environment(Binding::Value(int(value)), &Bindings::new())
        .expect("no bindings");
    param.check_value(&environment, context).expect("decides") == ValueCheck::Valid
}

/// Return the interval param `[lower, upper]` over a fresh variable.
fn interval_param(
    lower: i64,
    upper: i64,
    prefer_inclusive: bool,
    context: &ParamContext<'_>,
) -> Param {
    Param::new(
        ParamDomain::from(IntervalIntegerDomain::new(
            Inclusivity::inclusive_if(prefer_inclusive),
            Sign::Any,
            ZeroInclusion::Included,
        )),
        Identifier::new("param"),
        Vec::new(),
        context,
    )
    .and_then(|param| {
        param.with_bound(
            &LiteralValue::Int(BigInt::from(lower)),
            BoundSide::Lower,
            true,
            context,
        )
    })
    .and_then(|param| {
        param.with_bound(
            &LiteralValue::Int(BigInt::from(upper)),
            BoundSide::Upper,
            true,
            context,
        )
    })
    .expect("an interval param")
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn interval_arithmetic_is_the_exact_hull_of_the_pairwise_results(
        (left_lower, left_upper) in (-4_i64..4).prop_flat_map(|lower| (Just(lower), lower..5)),
        (right_lower, right_upper) in (-4_i64..4).prop_flat_map(|lower| (Just(lower), lower..5)),
        prefer_inclusive in any::<bool>(),
    ) {
        let (solver, _smt) = scripted_solver(SatResult::Sat);
        let observer = RecordingParamObserver::default();
        let context = context(&solver, &observer);
        let left = interval_param(left_lower, left_upper, prefer_inclusive, &context);
        let right = interval_param(right_lower, right_upper, true, &context);
        let operand = Operand::Param(right);
        let results = [
            (left.checked_add(&operand, &context).expect("adds"), 0),
            (left.checked_sub(&operand, &context).expect("subtracts"), 1),
            (left.checked_mul(&operand, &context).expect("multiplies"), 2),
        ];
        for (result, operation) in &results {
            let pairwise: Vec<i64> = (left_lower..=left_upper)
                .flat_map(|a| (right_lower..=right_upper).map(move |b| match operation {
                    0 => a + b,
                    1 => a - b,
                    _ => a * b,
                }))
                .collect();
            let least = *pairwise.iter().min().expect("a pair");
            let greatest = *pairwise.iter().max().expect("a pair");
            for value in -30..=30 {
                prop_assert_eq!(admits(result, value, &context), (least..=greatest).contains(&value));
            }
        }
    }
}

/// Return the permutations of `0..n`, each as a list of positions.
fn all_permutations(n: i64) -> Vec<Vec<i64>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut result = Vec::new();
    for shorter in all_permutations(n - 1) {
        for position in 0..=shorter.len() {
            let mut longer = shorter.clone();
            longer.insert(position, n - 1);
            result.push(longer);
        }
    }
    result
}

/// Return the tuple value of `permutation`.
fn permutation_value(permutation: &[i64]) -> Value {
    Value::Tuple(permutation.iter().map(|value| int(*value)).collect())
}

/// A permutation domain's side: in-set constraints, each over some of the
/// permutations and possibly a non-permutation, and not-in-set ones.
#[derive(Debug, Clone)]
struct PermutationSide {
    in_sets: Vec<Vec<usize>>,
    not_in_sets: Vec<Vec<usize>>,
    with_stray: bool,
}

fn permutation_side() -> impl Strategy<Value = PermutationSide> {
    let picks = || proptest::collection::vec(proptest::collection::vec(0_usize..120, 0..6), 0..3);
    (picks(), picks(), any::<bool>()).prop_map(|(in_sets, not_in_sets, with_stray)| {
        PermutationSide {
            in_sets,
            not_in_sets,
            with_stray,
        }
    })
}

impl PermutationSide {
    fn constraints(&self, x: &Identifier, permutations: &[Vec<i64>]) -> Vec<Constraint> {
        let pick = |indices: &[usize]| -> Vec<Value> {
            indices
                .iter()
                .map(|index| permutation_value(&permutations[index % permutations.len()]))
                .collect()
        };
        let mut constraints = Vec::new();
        for indices in &self.in_sets {
            let mut members = pick(indices);
            if self.with_stray {
                members.push(Value::Tuple(vec![int(7)]));
            }
            if !members.is_empty() {
                constraints.push(in_set(x, members));
            }
        }
        for indices in &self.not_in_sets {
            let members = pick(indices);
            if !members.is_empty() {
                constraints.push(not_in_set(x, members));
            }
        }
        constraints
    }

    /// Return whether `permutation` satisfies the side, by brute force.
    fn admits(&self, permutation: usize, permutations: &[Vec<i64>]) -> bool {
        let holds = |indices: &Vec<usize>| {
            indices
                .iter()
                .any(|index| index % permutations.len() == permutation)
        };
        self.in_sets
            .iter()
            .filter(|indices| !indices.is_empty() || self.with_stray)
            .all(holds)
            && !self.not_in_sets.iter().any(holds)
    }
}

proptest! {
    /// Permutation feasibility and subset, which enumerate the in-set
    /// candidates when there are some, agree with a walk over every
    /// permutation (F2-038).
    #[test]
    fn permutation_questions_agree_with_brute_force(
        n in 1_i64..=5,
        own in permutation_side(),
        other in permutation_side(),
    ) {
        let (x, y) = (Identifier::new("x"), Identifier::new("y"));
        let (solver, _smt) = scripted_solver(SatResult::Sat);
        let observer = RecordingParamObserver::default();
        let context = context(&solver, &observer);
        let permutations = all_permutations(n);
        let domain = ParamDomain::from(
            fhy_core::param::PermutationDomain::new((0..n).map(int).collect()).expect("members"),
        );
        let own_constraints = own.constraints(&x, &permutations);
        let other_constraints = other.constraints(&y, &permutations);

        let feasible = domain
            .has_feasible_value(Side::new(&own_constraints, &x), &context)
            .expect("decides");
        let subset = domain
            .feasibility_subset(
                Side::new(&own_constraints, &x),
                &domain,
                Side::new(&other_constraints, &y),
                &context,
            )
            .expect("decides");

        let own_admits: Vec<usize> =
            (0..permutations.len()).filter(|index| own.admits(*index, &permutations)).collect();
        prop_assert_eq!(
            feasible,
            if own_admits.is_empty() { Outcome::Violated } else { Outcome::Satisfied }
        );
        let is_subset = own_admits.iter().all(|index| other.admits(*index, &permutations));
        prop_assert_eq!(
            subset,
            if is_subset { Outcome::Satisfied } else { Outcome::Violated }
        );
    }
}
