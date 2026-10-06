//! Tests for `Direction` and `Objective`: their names, equality and text,
//! and `Objective::compare` per direction, NaN included (F-SS-023).

use std::cmp::Ordering;

use fhy_core::search_space::{Direction, MeasurementError, Objective};
use rstest::rstest;

use crate::support::hashing::hash_of;
use crate::support::measurement::objective;

/// Return the value of `values` that `objective` ranks best, the first of
/// equals: a running best that takes a value only when it is better.
fn running_best(objective: &Objective, values: &[f64]) -> f64 {
    let mut best = values[0];
    for &value in &values[1..] {
        if objective.compare(value, best) == Some(Ordering::Greater) {
            best = value;
        }
    }
    best
}

/// Test a direction's name and text.
#[rstest]
#[case::minimize(Direction::Minimize, "minimize")]
#[case::maximize(Direction::Maximize, "maximize")]
#[case::report(Direction::Report, "report")]
fn direction_names_itself(#[case] direction: Direction, #[case] name: &str) {
    assert_eq!(direction.as_str(), name);
    assert_eq!(direction.to_string(), name);
}

/// Test an objective keeps its name and direction.
#[test]
fn objective_keeps_its_name_and_direction() {
    let latency = Objective::new("latency_cycles", Direction::Minimize).expect("named");

    assert_eq!(latency.name(), "latency_cycles");
    assert_eq!(latency.direction(), Direction::Minimize);
}

/// Test an objective needs a name.
#[test]
fn objective_refuses_an_empty_name() {
    let error = Objective::new("", Direction::Maximize).expect_err("an empty name");

    assert_eq!(error, MeasurementError::EmptyName);
}

/// Test an objective's name may be any non-empty text, spaces and
/// multi-byte characters included.
#[rstest]
#[case::space(" ")]
#[case::multibyte("débit µs")]
fn objective_takes_any_non_empty_name(#[case] name: &str) {
    assert_eq!(objective(name, Direction::Report).name(), name);
}

/// Test an objective displays as its name and direction.
#[test]
fn objective_displays_its_name_and_direction() {
    assert_eq!(
        objective("throughput", Direction::Maximize).to_string(),
        "throughput (maximize)"
    );
}

/// Test objectives are equal, and hash alike, when their names and
/// directions are.
#[test]
fn objectives_are_equal_by_name_and_direction() {
    let left = objective("bytes", Direction::Minimize);
    let right = objective("bytes", Direction::Minimize);

    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
    assert_ne!(left, objective("bytes", Direction::Maximize));
    assert_ne!(left, objective("bytes_moved", Direction::Minimize));
}

/// Test minimizing prefers the lower value.
///
/// Ports MOGA-VM
/// `test_records.py::test_best_returns_the_lowest_scored_feasible_record_with_earliest_tie_break`
/// (the comparison only).
#[test]
fn minimize_prefers_the_lower_value() {
    let latency = objective("latency", Direction::Minimize);

    assert_eq!(latency.compare(1.0, 2.0), Some(Ordering::Greater));
    assert_eq!(latency.compare(2.0, 1.0), Some(Ordering::Less));
}

/// Test maximizing prefers the higher value.
#[test]
fn maximize_prefers_the_higher_value() {
    let throughput = objective("throughput", Direction::Maximize);

    assert_eq!(throughput.compare(2.0, 1.0), Some(Ordering::Greater));
    assert_eq!(throughput.compare(1.0, 2.0), Some(Ordering::Less));
}

/// Test equal values tie in either direction, `0.0` and `-0.0` included.
#[rstest]
#[case::minimize(Direction::Minimize)]
#[case::maximize(Direction::Maximize)]
fn equal_values_tie(#[case] direction: Direction) {
    let measured = objective("m", direction);

    assert_eq!(measured.compare(3.5, 3.5), Some(Ordering::Equal));
    assert_eq!(measured.compare(0.0, -0.0), Some(Ordering::Equal));
}

/// Test a reported objective is never compared.
#[rstest]
#[case::lower(1.0, 2.0)]
#[case::equal(2.0, 2.0)]
#[case::nan(f64::NAN, 1.0)]
fn report_is_never_compared(#[case] left: f64, #[case] right: f64) {
    assert_eq!(
        objective("note", Direction::Report).compare(left, right),
        None
    );
}

/// Test the infinities order as numbers do.
#[test]
fn infinities_order_as_numbers() {
    let latency = objective("latency", Direction::Minimize);
    let throughput = objective("throughput", Direction::Maximize);

    assert_eq!(
        latency.compare(f64::NEG_INFINITY, 0.0),
        Some(Ordering::Greater)
    );
    assert_eq!(latency.compare(f64::INFINITY, 0.0), Some(Ordering::Less));
    assert_eq!(
        throughput.compare(f64::INFINITY, 0.0),
        Some(Ordering::Greater)
    );
}

/// Test a NaN loses to every number, in either direction (F-SS-023).
#[rstest]
#[case::minimize_zero(Direction::Minimize, 0.0)]
#[case::minimize_huge(Direction::Minimize, f64::MAX)]
#[case::minimize_infinite(Direction::Minimize, f64::INFINITY)]
#[case::maximize_zero(Direction::Maximize, 0.0)]
#[case::maximize_tiny(Direction::Maximize, f64::MIN)]
#[case::maximize_infinite(Direction::Maximize, f64::NEG_INFINITY)]
fn a_nan_loses_to_every_number(#[case] direction: Direction, #[case] number: f64) {
    let measured = objective("m", direction);

    assert_eq!(measured.compare(f64::NAN, number), Some(Ordering::Less));
    assert_eq!(measured.compare(number, f64::NAN), Some(Ordering::Greater));
}

/// Test two NaNs tie.
#[rstest]
#[case::minimize(Direction::Minimize)]
#[case::maximize(Direction::Maximize)]
fn two_nans_tie(#[case] direction: Direction) {
    assert_eq!(
        objective("m", direction).compare(f64::NAN, f64::NAN),
        Some(Ordering::Equal)
    );
}

/// Test a NaN first in a run never blocks a later best (F-SS-023: MOGA-VM's
/// history kept a first NaN as its best forever).
#[rstest]
#[case::minimize_after_nan(Direction::Minimize, &[f64::NAN, 3.0, 1.0, 2.0], 1.0)]
#[case::minimize_between(Direction::Minimize, &[4.0, f64::NAN, 2.0], 2.0)]
#[case::minimize_nans_first(Direction::Minimize, &[f64::NAN, f64::NAN, 5.0], 5.0)]
#[case::maximize_after_nan(Direction::Maximize, &[f64::NAN, 1.0, 3.0, 2.0], 3.0)]
fn a_nan_never_blocks_a_later_best(
    #[case] direction: Direction,
    #[case] values: &[f64],
    #[case] best: f64,
) {
    assert_eq!(running_best(&objective("m", direction), values), best);
}
