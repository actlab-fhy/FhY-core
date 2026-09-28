//! Stories of the std traits of domains, params and assignments: `==` is
//! structural equivalence, `Hash` agrees with it, and `Display` writes them
//! for people.

use std::collections::HashSet;

use fhy_core::constraint::Value;
use fhy_core::identifier::Identifier;
use fhy_core::param::{
    CategoricalDomain, Inclusivity, IntegerDomain, IntervalIntegerDomain, OrdinalDomain, Param,
    ParamAssignment, ParamContext, ParamDomain, PermutationDomain, RealDomain, Sign, ZeroInclusion,
};
use fhy_core::solver::Solver;
use rstest::rstest;

use crate::support::constraint::{int, text};
use crate::support::hashing::hash_of;
use crate::support::param::{EvenDomain, at_least, at_most, float, ints};

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

fn integer(sign: Sign, zero: ZeroInclusion) -> ParamDomain {
    ParamDomain::from(IntegerDomain::new(sign, zero))
}

fn ordinal(values: Vec<Value>) -> ParamDomain {
    ParamDomain::from(OrdinalDomain::new(values).expect("an ordinal domain"))
}

fn categorical(values: Vec<Value>) -> ParamDomain {
    ParamDomain::from(CategoricalDomain::new(values).expect("a categorical domain"))
}

fn permutation(values: Vec<Value>) -> ParamDomain {
    ParamDomain::from(PermutationDomain::new(values).expect("a permutation domain"))
}

#[test]
fn domains_equal_as_they_are_structurally_equivalent() {
    let (even, _calls) = EvenDomain::build(false);
    let domains = [
        integer(Sign::Any, ZeroInclusion::Included),
        integer(Sign::Any, ZeroInclusion::Excluded),
        integer(Sign::NonNegative, ZeroInclusion::Excluded),
        ParamDomain::from(IntervalIntegerDomain::new(
            Inclusivity::Exclusive,
            Sign::Any,
            ZeroInclusion::Included,
        )),
        ParamDomain::from(RealDomain),
        ordinal(ints([1, 2])),
        ordinal(ints([2, 1])),
        ordinal(ints([1, 3])),
        categorical(vec![text("a"), text("b")]),
        categorical(vec![text("b"), text("a")]),
        permutation(ints([1, 2])),
        even,
        EvenDomain::build(false).0,
    ];

    assert_agrees_with(&domains, ParamDomain::is_structurally_equivalent);
    assert_eq!(domains[0], domains[1]);
    assert_eq!(domains[5], domains[6]);
    assert_ne!(domains[5], domains[7]);
    assert_eq!(domains[8], domains[9]);
    assert_eq!(domains[11], domains[12]);
}

#[rstest]
#[case::integer(integer(Sign::Any, ZeroInclusion::Included), "integer")]
#[case::natural(
    integer(Sign::NonNegative, ZeroInclusion::Included),
    "non-negative integer"
)]
#[case::positive(
    integer(Sign::NonNegative, ZeroInclusion::Excluded),
    "positive integer"
)]
#[case::interval(
    ParamDomain::from(IntervalIntegerDomain::new(
        Inclusivity::Exclusive,
        Sign::NonNegative,
        ZeroInclusion::Included,
    )),
    "non-negative interval integer (exclusive bounds)"
)]
#[case::real(ParamDomain::from(RealDomain), "real")]
#[case::ordinal(ordinal(ints([2, 1])), "ordinal (1, 2)")]
#[case::one_ordinal(ordinal(ints([2])), "ordinal (2,)")]
#[case::categorical(categorical(vec![text("b"), text("a")]), r#"categorical {"a", "b"}"#)]
#[case::permutation(permutation(ints([1, 2])), "permutation of (1, 2)")]
#[case::custom(EvenDomain::build(false).0, "<EvenDomain>")]
fn a_domain_displays_for_people(#[case] domain: ParamDomain, #[case] expected: &str) {
    assert_eq!(domain.to_string(), expected);
}

#[test]
fn params_equal_as_they_are_structurally_equivalent() {
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let build = |domain: ParamDomain, variable: &Identifier, bounded: bool| {
        let constraints = if bounded {
            vec![at_least(variable, 1), at_most(variable, 9)]
        } else {
            Vec::new()
        };
        Param::new(domain, variable.clone(), constraints, &context).expect("a param")
    };
    let any = || integer(Sign::Any, ZeroInclusion::Included);
    let params = [
        build(any(), &x, true),
        build(any(), &x, true),
        build(any(), &x, false),
        build(any(), &y, true),
        build(ParamDomain::from(RealDomain), &x, true),
    ];

    assert_agrees_with(&params, Param::is_structurally_equivalent);
    assert_eq!(params[0], params[1]);
    assert_eq!(params[2].to_string(), "x: integer");
    assert_eq!(
        params[0].to_string(),
        format!("x: integer where {}", params[0].constraint_system())
    );
}

#[test]
fn assignments_equal_by_their_params_and_values() {
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    let x = Identifier::new("x");
    let param = Param::new(
        ParamDomain::from(RealDomain),
        x.clone(),
        Vec::new(),
        &context,
    )
    .expect("a param");
    let other =
        Param::new(ParamDomain::from(RealDomain), x, Vec::new(), &context).expect("a param");
    let assignments = [
        ParamAssignment::new_unvalidated(param.clone(), float(0.0)),
        ParamAssignment::new_unvalidated(other.clone(), float(-0.0)),
        ParamAssignment::new_unvalidated(param.clone(), float(1.0)),
        ParamAssignment::new_unvalidated(param.clone(), int(1)),
    ];

    assert_agrees_with(&assignments, ParamAssignment::is_structurally_equivalent);
    assert_eq!(assignments[0], assignments[1]);
    assert_eq!(assignments[3].to_string(), "x = 1");
    // `==` is an equivalence: a NaN value equals itself, which the
    // type-strict value equality of `is_structurally_equivalent` denies.
    let nan = ParamAssignment::new_unvalidated(param, float(f64::NAN));
    assert_eq!(nan, nan.clone());
    assert!(!nan.is_structurally_equivalent(&nan));
    assert_eq!(
        hash_of(&nan),
        hash_of(&ParamAssignment::new_unvalidated(other, float(f64::NAN)))
    );
}
