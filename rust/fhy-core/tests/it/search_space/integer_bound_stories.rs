//! Tests for searching a space whose variables are integers bounded by
//! their params, under a solver without a simplifier: the bounds decide a
//! value exactly, so counting, sampling and `Configuration::new` agree;
//! a constraint no bound decides needs a simplifier and fails the same way
//! everywhere.

use std::num::NonZeroU32;

use fhy_core::constraint::{EquationConstraint, Value};
use fhy_core::expression::{BigInt, LiteralValue};
use fhy_core::identifier::Identifier;
use fhy_core::param::{
    AssignmentError, BoundSide, IntegerDomain, Param, ParamContext, ParamDomain, Sign,
    ZeroInclusion,
};
use fhy_core::search_space::{
    Cardinality, Configuration, ConfigurationError, RandomOracle, Rng, Space, TraceError,
};
use fhy_core::solver::Solver;
use num_bigint::BigUint;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};
use rstest::rstest;

use crate::support::constraint::int;
use crate::support::param::{literal, reference};
use crate::support::search_space::{configure, ground_solver, plain_variable, space_of};

/// Return a positive attempt count.
fn attempts(count: u32) -> NonZeroU32 {
    NonZeroU32::new(count).expect("a positive count")
}

/// Return the param over the integers of `sign`, bounded below by
/// `lower` and above by `upper`, each `(value, is_inclusive)`.
///
/// # Panics
///
/// Panics if the param is refused.
fn build_interval_param(sign: Sign, lower: (i64, bool), upper: (i64, bool)) -> Param {
    let solver = Solver::new();
    let context = ParamContext::new(&solver);
    Param::new(
        ParamDomain::from(IntegerDomain::new(sign, ZeroInclusion::Included)),
        Identifier::new("n"),
        Vec::new(),
        &context,
    )
    .and_then(|param| {
        param.with_bound(
            &LiteralValue::Int(BigInt::from(lower.0)),
            BoundSide::Lower,
            lower.1,
            &context,
        )
    })
    .and_then(|param| {
        param.with_bound(
            &LiteralValue::Int(BigInt::from(upper.0)),
            BoundSide::Upper,
            upper.1,
            &context,
        )
    })
    .expect("the bounded param is valid")
}

/// Return the space of one variable `k` over `param`.
fn build_one_variable_space(k: &Identifier, param: Param) -> Space {
    space_of(
        &Identifier::new("bounded"),
        vec![plain_variable(k, param)],
        Vec::new(),
    )
}

/// Return the space of one variable `k` over the natural numbers from 16 to
/// 4095, a byte address knob.
fn build_address_space(k: &Identifier) -> Space {
    build_one_variable_space(
        k,
        build_interval_param(Sign::NonNegative, (16, true), (4095, true)),
    )
}

/// Return the space of one variable `k` over the integers from 0 to 40
/// that are multiples of 4: the bounds are decided by the param, the
/// multiple by a solver.
fn build_multiples_space(k: &Identifier) -> Space {
    let param = build_interval_param(Sign::Any, (0, true), (40, true));
    let multiple_of_four = fhy_core::constraint::Constraint::from(EquationConstraint::new(
        reference(param.variable())
            .floor_mod(literal(4))
            .equals(literal(0)),
    ));
    let solver = Solver::new();
    let param = param
        .with_constraint(multiple_of_four, &ParamContext::new(&solver))
        .expect("the constraint is valid");
    build_one_variable_space(k, param)
}

/// Return the problem `result` stopped with: the one assignment of `k`.
///
/// # Panics
///
/// Panics if `result` is not a `TraceError::Configuration` holding one
/// problem, an assignment error of a constraint that failed to evaluate.
fn constraint_failure<T: std::fmt::Debug>(result: Result<T, TraceError>, k: &Identifier) {
    let Err(TraceError::Configuration(errors)) = result else {
        panic!("expected a configuration error, got {result:?}");
    };
    let [
        ConfigurationError::Assignment {
            variable,
            error: AssignmentError::Constraint(_),
        },
    ] = errors.errors()
    else {
        panic!("expected one failed constraint, got {errors:?}");
    };
    assert_eq!(variable, k);
}

// ---------------------------------------------------------------------------
// A natural number bounded on both sides, under a solver without a simplifier
// ---------------------------------------------------------------------------

/// Test the count of a natural number bounded from 16 to 4095 is exact,
/// closed form, under a solver that has no simplifier: the guard that
/// counting never asked the solver.
#[test]
fn space_cardinality_of_a_bounded_natural_is_exact_without_a_simplifier() {
    let space = build_address_space(&Identifier::new("address"));
    let solver = Solver::new();

    let count = space.cardinality(&ParamContext::new(&solver), 10_000);

    assert_eq!(
        count.expect("the count succeeds"),
        Cardinality::Exact(BigUint::from(4080_u32))
    );
}

/// Test sampling the bounded natural under a solver without a simplifier
/// draws a value inside the bounds, as the count says there are.
#[test]
fn space_sample_of_a_bounded_natural_draws_a_value_in_its_bounds_without_a_simplifier() {
    let k = Identifier::new("address");
    let space = build_address_space(&k);
    let solver = Solver::new();

    for seed in 0..8 {
        let recorded = space
            .sample(&mut RandomOracle::new(seed), &ParamContext::new(&solver))
            .expect("a value inside the bounds is admissible");

        let configuration = recorded.configuration().expect("a run over a space");
        let Some(Value::Int(value)) = configuration.value(&k) else {
            panic!("an integer value, got {:?}", configuration.value(&k));
        };
        assert!(
            (BigInt::from(16)..=BigInt::from(4095)).contains(value),
            "seed {seed} drew {value}"
        );
    }
}

/// Test a uniform draw over the bounded natural under a solver without a
/// simplifier finds a configuration in the bounds.
#[test]
fn space_sample_uniform_of_a_bounded_natural_draws_a_value_in_its_bounds_without_a_simplifier() {
    let k = Identifier::new("address");
    let space = build_address_space(&k);
    let solver = Solver::new();

    let recorded = space
        .sample_uniform(&mut Rng::new(5), &ParamContext::new(&solver), attempts(16))
        .expect("the space has configurations");

    let configuration = recorded.configuration().expect("a run over a space");
    let Some(Value::Int(value)) = configuration.value(&k) else {
        panic!("an integer value, got {:?}", configuration.value(&k));
    };
    assert!((BigInt::from(16)..=BigInt::from(4095)).contains(value));
}

/// Test a configuration accepts a value inside the bounds, at either edge
/// or between, under a solver without a simplifier.
#[rstest]
#[case::lower_edge(16)]
#[case::inside(2955)]
#[case::upper_edge(4095)]
fn configuration_new_accepts_a_value_inside_the_bounds_without_a_simplifier(#[case] value: i64) {
    let k = Identifier::new("address");
    let space = build_address_space(&k);
    let solver = Solver::new();

    let configuration = Configuration::new(
        &space,
        [(k.clone(), int(value))],
        &ParamContext::new(&solver),
    )
    .expect("a value inside the bounds is accepted");

    assert_eq!(configuration.value(&k), Some(&int(value)));
}

/// Test a configuration refuses a value one past either bound as a
/// violated constraint, not as a failure of the solver.
#[rstest]
#[case::below_the_lower_bound(15)]
#[case::above_the_upper_bound(4096)]
#[case::zero(0)]
fn configuration_new_refuses_a_value_outside_the_bounds_without_a_simplifier(#[case] value: i64) {
    let k = Identifier::new("address");
    let space = build_address_space(&k);
    let solver = Solver::new();

    let result = Configuration::new(
        &space,
        [(k.clone(), int(value))],
        &ParamContext::new(&solver),
    );

    let errors = result.expect_err("a value outside the bounds is refused");
    let [
        ConfigurationError::Assignment {
            variable,
            error: AssignmentError::ViolatedConstraint { .. },
        },
    ] = errors.errors()
    else {
        panic!("expected one violated constraint, got {errors:?}");
    };
    assert_eq!(variable, &k);
}

// ---------------------------------------------------------------------------
// A constraint no bound decides, under a solver without a simplifier
// ---------------------------------------------------------------------------

/// Test sampling a variable whose param holds `n % 4 == 0` under a solver
/// without a simplifier stops with the failed constraint, not with a dead
/// end that says no value is admissible.
#[test]
fn space_sample_over_a_constraint_no_bound_decides_fails_without_a_simplifier() {
    let k = Identifier::new("multiple");
    let space = build_multiples_space(&k);
    let solver = Solver::new();

    let result = space.sample(&mut RandomOracle::new(0), &ParamContext::new(&solver));

    constraint_failure(result, &k);
}

/// Test counting such a space under a solver without a simplifier stops
/// with the failed constraint, not with a count of none.
#[test]
fn space_cardinality_over_a_constraint_no_bound_decides_fails_without_a_simplifier() {
    let k = Identifier::new("multiple");
    let space = build_multiples_space(&k);
    let solver = Solver::new();

    let result = space.cardinality(&ParamContext::new(&solver), 1_000);

    constraint_failure(result, &k);
}

/// Test mutating a configuration of such a space, built under a ground
/// simplifier, stops with the failed constraint when the mutation runs
/// under a solver without a simplifier.
#[test]
fn space_mutate_over_a_constraint_no_bound_decides_fails_without_a_simplifier() {
    let k = Identifier::new("multiple");
    let space = build_multiples_space(&k);
    let configuration = configure(&space, [(k.clone(), int(8))]);
    let solver = Solver::new();

    let result = space.mutate(
        &configuration,
        &mut Rng::new(0),
        &ParamContext::new(&solver),
        attempts(4),
    );

    constraint_failure(result, &k);
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

/// The integers a bounded param's values are searched among: the bounds
/// stay inside `[-6, 6]`, so every value a bound admits is inside.
const SEARCHED: std::ops::RangeInclusive<i64> = -9..=9;

/// Return a strategy of the two bounds of a param, each a value in
/// `[-6, 6]` and whether it is inclusive.
fn generate_bounds() -> impl Strategy<Value = ((i64, bool), (i64, bool))> {
    ((-6_i64..=6, any::<bool>()), (-6_i64..=6, any::<bool>()))
}

/// Return how many integers of `SEARCHED` lie within `lower` and `upper`,
/// by comparing, as no part of the crate does.
fn count_in_bounds(lower: (i64, bool), upper: (i64, bool)) -> usize {
    SEARCHED
        .filter(|&value| {
            (value > lower.0 || (lower.1 && value == lower.0))
                && (value < upper.0 || (upper.1 && value == upper.0))
        })
        .count()
}

/// Return how many integers of `SEARCHED` a configuration of `space` over
/// `k` accepts, under `context`.
fn count_accepted(space: &Space, k: &Identifier, context: &ParamContext<'_>) -> usize {
    SEARCHED
        .filter(|&value| Configuration::new(space, [(k.clone(), int(value))], context).is_ok())
        .count()
}

proptest! {
    #![proptest_config(Config::with_cases(96))]

    /// Test the count of an integer param bounded on both sides equals the
    /// number of values `Configuration::new` accepts, under a solver
    /// without a simplifier and under a ground simplifier, and equals the
    /// number of integers between the bounds.
    #[test]
    fn cardinality_of_a_bounded_integer_equals_the_values_a_configuration_accepts(
        (lower, upper) in generate_bounds(),
    ) {
        let k = Identifier::new("k");
        let space = build_one_variable_space(&k, build_interval_param(Sign::Any, lower, upper));
        let plain = Solver::new();
        let ground = ground_solver();

        for solver in [&plain, &ground] {
            let context = ParamContext::new(solver);
            let count = space.cardinality(&context, 1_000).expect("the count succeeds");
            let accepted = count_accepted(&space, &k, &context);

            prop_assert_eq!(&count, &Cardinality::Exact(BigUint::from(accepted)));
            prop_assert_eq!(accepted, count_in_bounds(lower, upper));
        }
    }
}

/// The cases the guard draws.
const GUARD_CASES: usize = 256;

/// Return `GUARD_CASES` pairs of bounds drawn by a runner with a fixed
/// seed.
fn draw_bounds() -> Vec<((i64, bool), (i64, bool))> {
    let strategy = generate_bounds();
    let mut runner = TestRunner::new_with_rng(
        Config::default(),
        TestRng::deterministic_rng(RngAlgorithm::ChaCha),
    );
    (0..GUARD_CASES)
        .map(|_| {
            strategy
                .new_tree(&mut runner)
                .expect("the strategy draws")
                .current()
        })
        .collect()
}

/// Test most generated bounds enclose some integer, and some enclose none:
/// the property compares non-zero counts, and the empty count too.
#[test]
fn many_generated_bounds_enclose_integers_and_some_enclose_none() {
    let bounds = draw_bounds();

    let enclosing = bounds
        .iter()
        .filter(|&&(lower, upper)| count_in_bounds(lower, upper) > 0)
        .count();
    let empty = bounds
        .iter()
        .filter(|&&(lower, upper)| count_in_bounds(lower, upper) == 0)
        .count();
    let exclusive = bounds
        .iter()
        .filter(|&&(lower, upper)| !lower.1 && !upper.1)
        .count();

    assert!(
        enclosing * 10 >= GUARD_CASES * 3,
        "{enclosing} of {GUARD_CASES} bounds enclose an integer; the equality of the count \
         and the accepted values is exercised only on those"
    );
    assert!(empty >= 1, "no bounds of {GUARD_CASES} enclose none");
    assert!(
        exclusive >= 1,
        "no bounds of {GUARD_CASES} are both exclusive"
    );
}
