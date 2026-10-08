//! Tests for `non_dominated`, the Pareto front of measurements: the
//! design's table, the order kept, ties, failures left out, directions and
//! reported objectives, the error for different objectives, and a property
//! against the pairwise `dominates` of the input.

use fhy_core::identifier::Identifier;
use fhy_core::search_space::{Direction, Measurement, MeasurementError, Objective, non_dominated};
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};

use crate::support::measurement::{objective, tiling_key};
use crate::support::search::{build_tiling_space, record_tiling_run};

/// Return the successful measurement of the minimized `latency` and
/// `energy` with the values `latency` and `energy`.
fn measure_two(latency: f64, energy: f64) -> Measurement {
    Measurement::ok(
        tiling_key(0),
        vec![
            (objective("latency", Direction::Minimize), latency),
            (objective("energy", Direction::Minimize), energy),
        ],
    )
    .expect("finite values")
}

/// Return the positions in `measurements` of each of `front`, by identity.
fn locate(measurements: &[Measurement], front: &[&Measurement]) -> Vec<usize> {
    front
        .iter()
        .map(|kept| {
            measurements
                .iter()
                .position(|measurement| std::ptr::eq(measurement, *kept))
                .expect("the front holds input measurements")
        })
        .collect()
}

/// Return the front of `measurements` as positions in it.
///
/// # Panics
///
/// Panics if the measurements are not comparable.
fn locate_front(measurements: &[Measurement]) -> Vec<usize> {
    let front = non_dominated(measurements).expect("comparable measurements");
    locate(measurements, &front)
}

/// Test the design's table: of `(1, 2)`, `(2, 1)`, `(2, 2)` and a failure,
/// minimizing both, the front is the first two, in order.
#[test]
fn non_dominated_keeps_the_pareto_front_in_input_order() {
    let measurements = [
        measure_two(1.0, 2.0),
        measure_two(2.0, 1.0),
        measure_two(2.0, 2.0),
        Measurement::failed(tiling_key(1), "crashed"),
    ];

    let front = locate_front(&measurements);

    assert_eq!(front, [0, 1]);
}

/// Test no measurement has an empty front.
#[test]
fn non_dominated_of_nothing_is_empty() {
    let front = non_dominated(&[]).expect("nothing to compare");

    assert_eq!(front, Vec::<&Measurement>::new());
}

/// Test measurements that all did not succeed have an empty front, and
/// no error though they hold no objectives.
#[test]
fn non_dominated_leaves_out_every_measurement_that_did_not_succeed() {
    let measurements = [
        Measurement::infeasible(tiling_key(0), "rejected"),
        Measurement::failed(tiling_key(1), "crashed"),
        Measurement::timeout(tiling_key(2)),
    ];

    let front = locate_front(&measurements);

    assert_eq!(front, Vec::<usize>::new());
}

/// Test a failure among successes is left out, wherever it stands.
#[test]
fn non_dominated_leaves_out_a_failure_among_successes() {
    let measurements = [
        Measurement::timeout(tiling_key(2)),
        measure_two(3.0, 3.0),
        Measurement::failed(tiling_key(1), "crashed"),
    ];

    let front = locate_front(&measurements);

    assert_eq!(front, [1]);
}

/// Test two measurements of equal values both stay: neither dominates the
/// other.
#[test]
fn non_dominated_keeps_both_of_two_equal_measurements() {
    let measurements = [measure_two(1.0, 1.0), measure_two(1.0, 1.0)];

    let front = locate_front(&measurements);

    assert_eq!(front, [0, 1]);
}

/// Test two equal measurements both go when a third dominates them.
#[test]
fn non_dominated_drops_equal_measurements_a_third_dominates() {
    let measurements = [
        measure_two(2.0, 2.0),
        measure_two(2.0, 2.0),
        measure_two(1.0, 1.0),
    ];

    let front = locate_front(&measurements);

    assert_eq!(front, [2]);
}

/// Test the front keeps the input order, not the order of the values.
#[test]
fn non_dominated_keeps_the_input_order_of_incomparable_measurements() {
    let measurements = [
        measure_two(3.0, 1.0),
        measure_two(1.0, 3.0),
        measure_two(2.0, 2.0),
    ];

    let front = locate_front(&measurements);

    assert_eq!(front, [0, 1, 2]);
}

/// Test a maximized objective is better higher.
#[test]
fn non_dominated_follows_a_maximized_objective() {
    let throughput = objective("throughput", Direction::Maximize);
    let measure = |value: f64| {
        Measurement::ok(tiling_key(0), vec![(throughput.clone(), value)]).expect("finite")
    };
    let measurements = [measure(1.0), measure(5.0), measure(3.0)];

    let front = locate_front(&measurements);

    assert_eq!(front, [1]);
}

/// Test measurements differing only in a reported objective are both kept,
/// and one dominated in a compared objective goes whatever it reports.
#[test]
fn non_dominated_ignores_a_reported_objective() {
    let latency = objective("latency", Direction::Minimize);
    let power = objective("power", Direction::Report);
    let measure = |latency_value: f64, power_value: f64| {
        Measurement::ok(
            tiling_key(0),
            vec![
                (latency.clone(), latency_value),
                (power.clone(), power_value),
            ],
        )
        .expect("finite")
    };
    let measurements = [measure(1.0, 100.0), measure(1.0, 0.0), measure(2.0, -50.0)];

    let front = locate_front(&measurements);

    assert_eq!(front, [0, 1]);
}

/// Test two successful measurements over different objectives are refused.
#[test]
fn non_dominated_refuses_successes_over_different_objectives() {
    let measurements = [
        measure_two(1.0, 1.0),
        Measurement::ok(
            tiling_key(0),
            vec![(objective("latency", Direction::Maximize), 1.0)],
        )
        .expect("finite"),
    ];

    let result = non_dominated(&measurements);

    assert!(
        matches!(result, Err(MeasurementError::DifferentObjectives)),
        "{result:?}"
    );
}

/// Test measurements of runs and of configurations are compared alike:
/// the key does not take part.
#[test]
fn non_dominated_compares_measurements_of_runs_and_configurations_alike() {
    let tiling = build_tiling_space();
    let (trace, _) = record_tiling_run(&tiling, 17, &Identifier::new("buffer"));
    let latency = objective("latency", Direction::Minimize);
    let measurements = [
        Measurement::ok(tiling_key(0), vec![(latency.clone(), 2.0)]).expect("finite"),
        Measurement::ok(trace.key(), vec![(latency, 1.0)]).expect("finite"),
    ];

    let front = locate_front(&measurements);

    assert_eq!(front, [1]);
}

// ---------------------------------------------------------------------------
// Property
// ---------------------------------------------------------------------------

/// The objectives of the generated populations, in their directions.
fn build_objectives(directions: [Direction; 3]) -> [Objective; 3] {
    let names = ["a", "b", "c"];
    [0, 1, 2].map(|index| objective(names[index], directions[index]))
}

/// Return a direction.
fn generate_direction() -> impl Strategy<Value = Direction> {
    prop_oneof![
        Just(Direction::Minimize),
        Just(Direction::Maximize),
        Just(Direction::Report),
    ]
}

/// A generated population: the directions of the three objectives, and
/// per member its values, or none for a measurement that failed.
type Population = ([Direction; 3], Vec<Option<[u8; 3]>>);

/// Return a strategy of populations of up to eight members with values in
/// `0..4`, so ties and dominance are common.
fn generate_population() -> impl Strategy<Value = Population> {
    (
        prop::array::uniform3(generate_direction()),
        prop::collection::vec(
            prop::option::weighted(0.85, prop::array::uniform3(0_u8..4)),
            0..8,
        ),
    )
}

/// Return the measurements of `population`.
fn build_population(population: &Population) -> Vec<Measurement> {
    let objectives = build_objectives(population.0);
    population
        .1
        .iter()
        .map(|member| match member {
            Some(values) => Measurement::ok(
                tiling_key(0),
                objectives
                    .iter()
                    .cloned()
                    .zip(values.iter().copied().map(f64::from))
                    .collect(),
            )
            .expect("finite values"),
            None => Measurement::failed(tiling_key(0), "crashed"),
        })
        .collect()
}

/// Return the positions of the successful measurements of `measurements`
/// no other successful one dominates, by the pairwise `dominates`.
fn compute_reference_front(measurements: &[Measurement]) -> Vec<usize> {
    let successes: Vec<&Measurement> = measurements.iter().filter(|m| m.is_ok()).collect();
    measurements
        .iter()
        .enumerate()
        .filter(|(_, measurement)| {
            measurement.is_ok()
                && !successes.iter().any(|other| {
                    other
                        .dominates(measurement)
                        .expect("measurements over the same objectives")
                })
        })
        .map(|(position, _)| position)
        .collect()
}

proptest! {
    #![proptest_config(Config::with_cases(256))]

    /// Test the front is exactly the successful measurements no successful
    /// one dominates, in input order: nothing returned is dominated, and
    /// every success left out is dominated by a returned one.
    ///
    /// Oracle: the pairwise `dominates` of the input.
    #[test]
    fn non_dominated_is_the_undominated_successes_in_order(population in generate_population()) {
        let measurements = build_population(&population);

        let front = locate_front(&measurements);

        prop_assert_eq!(&front, &compute_reference_front(&measurements));
        for (position, measurement) in measurements.iter().enumerate() {
            if measurement.is_ok() && !front.contains(&position) {
                prop_assert!(front.iter().any(|&kept| measurements[kept]
                    .dominates(measurement)
                    .expect("same objectives")));
            }
        }
    }
}

/// The populations the guard draws.
const GUARD_CASES: usize = 256;

/// Test the generated populations often hold a dominated success and a
/// failure, and often have a front of two or more, so the property
/// compares non-trivial fronts.
#[test]
fn generated_populations_reach_dominated_successes_failures_and_wide_fronts() {
    let strategy = generate_population();
    let mut runner = TestRunner::new_with_rng(
        Config::default(),
        TestRng::deterministic_rng(RngAlgorithm::ChaCha),
    );
    let populations: Vec<Population> = (0..GUARD_CASES)
        .map(|_| {
            strategy
                .new_tree(&mut runner)
                .expect("the strategy draws")
                .current()
        })
        .collect();

    let mut dominated = 0;
    let mut with_failure = 0;
    let mut wide = 0;
    for population in &populations {
        let measurements = build_population(population);
        let front = compute_reference_front(&measurements);
        let successes = measurements.iter().filter(|m| m.is_ok()).count();
        dominated += usize::from(successes > front.len());
        with_failure += usize::from(successes < measurements.len());
        wide += usize::from(front.len() >= 2);
    }

    assert!(
        dominated * 4 >= GUARD_CASES,
        "{dominated} with a dominated success"
    );
    assert!(
        with_failure * 4 >= GUARD_CASES,
        "{with_failure} with a failure"
    );
    assert!(
        wide * 4 >= GUARD_CASES,
        "{wide} with a front of two or more"
    );
}
