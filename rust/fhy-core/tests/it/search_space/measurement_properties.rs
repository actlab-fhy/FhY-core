//! Properties of `Measurement::dominates` over generated measurements of
//! three objectives with generated directions: a strict partial order
//! (irreflexive, asymmetric, transitive), consistent with the directions
//! (reversing every direction reverses dominance; over one objective it is
//! the direction's strict order), blind to reported objectives and to the
//! order of the values; and measurements round-trip through JSON and
//! postcard. Values come from a small set, so ties are common.

use fhy_core::search_space::{Direction, Measurement, Objective};
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};

use crate::support::measurement::{objective, tiling_key};
use crate::support::serde::check_serde_round_trip;

/// The names of the generated measurements' objectives.
const NAMES: [&str; 3] = ["a", "b", "c"];

/// Return a direction.
fn any_direction() -> impl Strategy<Value = Direction> {
    prop_oneof![
        Just(Direction::Minimize),
        Just(Direction::Maximize),
        Just(Direction::Report),
    ]
}

/// Return the values of one measurement: small integers, so ties are
/// common, as floats.
fn any_values() -> impl Strategy<Value = [f64; 3]> {
    prop::array::uniform3((0_u8..4).prop_map(f64::from))
}

/// Return the objectives `NAMES` in `directions`.
fn objectives(directions: [Direction; 3]) -> [Objective; 3] {
    [0, 1, 2].map(|index| objective(NAMES[index], directions[index]))
}

/// Return the successful measurement of `values` over `objectives`, in the
/// order of `order`'s indices.
fn measure(objectives: &[Objective; 3], values: [f64; 3], order: [usize; 3]) -> Measurement {
    Measurement::ok(
        tiling_key(0),
        order
            .iter()
            .map(|&index| (objectives[index].clone(), values[index]))
            .collect(),
    )
    .expect("finite values of distinct objectives")
}

/// Return `direction` reversed: minimize and maximize swapped.
fn reversed(direction: Direction) -> Direction {
    match direction {
        Direction::Minimize => Direction::Maximize,
        Direction::Maximize => Direction::Minimize,
        Direction::Report => Direction::Report,
    }
}

/// Return whether `left` dominates `right`, both successful over the same
/// objectives.
fn dominates(left: &Measurement, right: &Measurement) -> bool {
    left.dominates(right)
        .unwrap_or_else(|error| panic!("comparable measurements: {error}"))
}

proptest! {
    #![proptest_config(Config::with_cases(256))]

    /// Test no measurement dominates itself.
    #[test]
    fn dominance_is_irreflexive(directions in prop::array::uniform3(any_direction()), values in any_values()) {
        let objectives = objectives(directions);
        let measurement = measure(&objectives, values, [0, 1, 2]);

        prop_assert!(!dominates(&measurement, &measurement));
    }

    /// Test two measurements never dominate each other.
    #[test]
    fn dominance_is_asymmetric(
        directions in prop::array::uniform3(any_direction()),
        left in any_values(),
        right in any_values(),
    ) {
        let objectives = objectives(directions);
        let left = measure(&objectives, left, [0, 1, 2]);
        let right = measure(&objectives, right, [0, 1, 2]);

        prop_assert!(!(dominates(&left, &right) && dominates(&right, &left)));
    }

    /// Test dominance is transitive.
    #[test]
    fn dominance_is_transitive(
        directions in prop::array::uniform3(any_direction()),
        first in any_values(),
        second in any_values(),
        third in any_values(),
    ) {
        let objectives = objectives(directions);
        let [first, second, third] =
            [first, second, third].map(|values| measure(&objectives, values, [0, 1, 2]));

        if dominates(&first, &second) && dominates(&second, &third) {
            prop_assert!(dominates(&first, &third));
        }
    }

    /// Test reversing every direction reverses dominance.
    #[test]
    fn reversing_the_directions_reverses_dominance(
        directions in prop::array::uniform3(any_direction()),
        left in any_values(),
        right in any_values(),
    ) {
        let forward = objectives(directions);
        let backward = objectives(directions.map(reversed));

        let ahead = dominates(&measure(&forward, left, [0, 1, 2]), &measure(&forward, right, [0, 1, 2]));
        let behind = dominates(&measure(&backward, right, [0, 1, 2]), &measure(&backward, left, [0, 1, 2]));

        prop_assert_eq!(ahead, behind);
    }

    /// Test over one compared objective, dominance is the direction's
    /// strict order: lower when minimizing, higher when maximizing.
    #[test]
    fn dominance_over_one_objective_is_the_directions_order(
        maximize in any::<bool>(),
        left in 0_u8..4,
        right in 0_u8..4,
    ) {
        let direction = if maximize { Direction::Maximize } else { Direction::Minimize };
        let objectives = objectives([direction, Direction::Report, Direction::Report]);
        let [left_value, right_value] = [left, right].map(f64::from);
        let left_measurement = measure(&objectives, [left_value, 0.0, 0.0], [0, 1, 2]);
        let right_measurement = measure(&objectives, [right_value, 9.0, 9.0], [0, 1, 2]);

        let expected = if maximize { left > right } else { left < right };
        prop_assert_eq!(dominates(&left_measurement, &right_measurement), expected);
    }

    /// Test dominance ignores reported values and the order values are
    /// given in.
    #[test]
    fn dominance_ignores_reported_values_and_value_order(
        directions in prop::array::uniform3(any_direction()),
        left in any_values(),
        right in any_values(),
        reported in any_values(),
        order in Just([0_usize, 1, 2]).prop_shuffle(),
    ) {
        let objectives = objectives(directions);
        let changed = [0, 1, 2].map(|index| {
            if directions[index] == Direction::Report { reported[index] } else { left[index] }
        });
        let order: [usize; 3] = [order[0], order[1], order[2]];

        let base = dominates(&measure(&objectives, left, [0, 1, 2]), &measure(&objectives, right, [0, 1, 2]));
        let varied = dominates(&measure(&objectives, changed, order), &measure(&objectives, right, [2, 1, 0]));

        prop_assert_eq!(base, varied);
    }

    /// Test a measurement round-trips through JSON and postcard.
    #[test]
    fn measurement_round_trips_through_json_and_postcard(
        directions in prop::array::uniform3(any_direction()),
        values in prop::array::uniform3(-1e9_f64..1e9),
        order in Just([0_usize, 1, 2]).prop_shuffle(),
    ) {
        let objectives = objectives(directions);
        let measurement = measure(&objectives, values, [order[0], order[1], order[2]]);

        check_serde_round_trip(&measurement)?;
    }
}

/// Test a fixed sample of the generated pairs holds both dominating and
/// non-dominating pairs with a compared objective, so the properties above
/// are not vacuous.
#[test]
fn the_generated_pairs_dominate_often_enough() {
    let strategy = (
        prop::array::uniform3(any_direction()),
        any_values(),
        any_values(),
    );
    let mut runner = TestRunner::new_with_rng(
        Config::default(),
        TestRng::from_seed(RngAlgorithm::ChaCha, &[7; 32]),
    );
    let mut dominating = 0;
    let mut compared = 0;
    for _ in 0..200 {
        let (directions, left, right) = strategy
            .new_tree(&mut runner)
            .expect("a case is generated")
            .current();
        if directions
            .iter()
            .all(|direction| *direction == Direction::Report)
        {
            continue;
        }
        compared += 1;
        let objectives = objectives(directions);
        if dominates(
            &measure(&objectives, left, [0, 1, 2]),
            &measure(&objectives, right, [0, 1, 2]),
        ) {
            dominating += 1;
        }
    }

    assert!(
        compared >= 150,
        "{compared} pairs with a compared objective"
    );
    assert!(dominating >= 30, "{dominating} dominating pairs");
    assert!(
        compared - dominating >= 30,
        "{} non-dominating pairs",
        compared - dominating
    );
}
