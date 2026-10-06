//! Tests for `Measurement`: the constructors and every refusal, the values
//! kept in order, notes, equality, `dominates`, the error texts, and a
//! `Measurer` measuring the configurations of a space.

use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use fhy_core::constraint::Value;
use fhy_core::diagnostic::Note;
use fhy_core::foreign::BoxError;
use fhy_core::identifier::Identifier;
use fhy_core::search_space::{
    Configuration, ConfigurationKey, Direction, Measurement, MeasurementError, MeasurementStatus,
    Measurer, Objective,
};
use rstest::rstest;

use crate::support::measurement::{measured, objective, tiling_key};
use crate::support::search::{build_tiling_space, with_context};

/// Return the hash of `value`.
fn hash_of<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Return latency (minimize), throughput (maximize) and a reported note.
fn three_objectives() -> [Objective; 3] {
    [
        objective("latency", Direction::Minimize),
        objective("throughput", Direction::Maximize),
        objective("power", Direction::Report),
    ]
}

/// Return the successful measurement of tiling configuration 0 with
/// `latency`, `throughput` and `power`.
fn measure_three(latency: f64, throughput: f64, power: f64) -> Measurement {
    let [l, t, p] = three_objectives();
    measured(0, &[(&l, latency), (&t, throughput), (&p, power)])
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

/// Test a successful measurement keeps its key, its status and its values
/// in the order given.
///
/// Ports MOGA-VM `test_records.py::test_a_score_on_a_lowered_record_is_valid`.
#[test]
fn an_ok_measurement_keeps_its_values() {
    let [latency, throughput, power] = three_objectives();
    let values = vec![
        (throughput.clone(), 7.5),
        (latency.clone(), 1532.0),
        (power.clone(), 0.25),
    ];

    let measurement = Measurement::ok(tiling_key(2), values.clone()).expect("valid values");

    assert_eq!(measurement.key(), &tiling_key(2));
    assert_eq!(measurement.status(), &MeasurementStatus::Ok);
    assert!(measurement.is_ok());
    assert_eq!(measurement.values(), values.as_slice());
    assert_eq!(measurement.value("latency"), Some(1532.0));
    assert_eq!(measurement.value("power"), Some(0.25));
    assert_eq!(measurement.value("area"), None);
    assert_eq!(measurement.notes(), &[]);
}

/// Test a measurement that did not succeed holds no value.
///
/// Ports MOGA-VM `test_records.py::test_a_score_on_an_infeasible_record_is_invalid`.
#[rstest]
#[case::infeasible(
    Measurement::infeasible(tiling_key(1), "rejected by placement"),
    MeasurementStatus::Infeasible { reason: "rejected by placement".to_owned() }
)]
#[case::failed(
    Measurement::failed(tiling_key(1), "the simulator crashed"),
    MeasurementStatus::Failed { reason: "the simulator crashed".to_owned() }
)]
#[case::timeout(Measurement::timeout(tiling_key(1)), MeasurementStatus::Timeout)]
fn a_failed_measurement_holds_no_values(
    #[case] measurement: Measurement,
    #[case] status: MeasurementStatus,
) {
    assert_eq!(measurement.key(), &tiling_key(1));
    assert_eq!(measurement.status(), &status);
    assert!(!measurement.is_ok());
    assert_eq!(measurement.values(), &[]);
    assert_eq!(measurement.value("latency"), None);
}

/// Test a successful measurement needs a value.
#[test]
fn an_ok_measurement_refuses_no_values() {
    let error = Measurement::ok(tiling_key(0), Vec::new()).expect_err("no values");

    assert_eq!(error, MeasurementError::NoValues);
}

/// Test two values for one objective name are refused, whatever their
/// directions.
#[rstest]
#[case::same_direction(Direction::Minimize)]
#[case::other_direction(Direction::Maximize)]
fn an_ok_measurement_refuses_a_repeated_objective(#[case] second: Direction) {
    let values = vec![
        (objective("latency", Direction::Minimize), 1.0),
        (objective("bytes", Direction::Minimize), 2.0),
        (objective("latency", second), 3.0),
    ];

    let error = Measurement::ok(tiling_key(0), values).expect_err("a repeated objective");

    assert_eq!(
        error,
        MeasurementError::RepeatedObjective {
            name: "latency".to_owned()
        }
    );
}

/// Test a NaN or an infinite value is refused, naming its objective
/// (F-SS-023).
#[rstest]
#[case::nan(f64::NAN)]
#[case::infinity(f64::INFINITY)]
#[case::negative_infinity(f64::NEG_INFINITY)]
fn an_ok_measurement_refuses_a_non_finite_value(#[case] value: f64) {
    let values = vec![
        (objective("latency", Direction::Minimize), 1.0),
        (objective("bytes", Direction::Report), value),
    ];

    let error = Measurement::ok(tiling_key(0), values).expect_err("a non-finite value");

    assert_eq!(
        error,
        MeasurementError::NonFiniteValue {
            objective: "bytes".to_owned()
        }
    );
}

/// Test a repeated objective is reported before a non-finite value.
#[test]
fn a_repeated_objective_is_reported_before_a_non_finite_value() {
    let values = vec![
        (objective("latency", Direction::Minimize), f64::NAN),
        (objective("latency", Direction::Minimize), 1.0),
    ];

    let error = Measurement::ok(tiling_key(0), values).expect_err("refused");

    assert_eq!(
        error,
        MeasurementError::RepeatedObjective {
            name: "latency".to_owned()
        }
    );
}

/// Test the extreme finite values are kept exactly.
#[rstest]
#[case::max(f64::MAX)]
#[case::min(f64::MIN)]
#[case::smallest_positive(f64::MIN_POSITIVE)]
#[case::subnormal(5e-324)]
fn an_ok_measurement_keeps_extreme_finite_values(#[case] value: f64) {
    let bytes = objective("bytes", Direction::Minimize);

    let measurement = measured(0, &[(&bytes, value)]);

    assert_eq!(
        measurement.value("bytes").map(f64::to_bits),
        Some(value.to_bits())
    );
}

/// Test `-0.0` is kept as `0.0`, so equal measurements hash alike.
#[test]
fn a_negative_zero_is_kept_as_zero() {
    let bytes = objective("bytes", Direction::Minimize);

    let negative = measured(0, &[(&bytes, -0.0)]);
    let positive = measured(0, &[(&bytes, 0.0)]);

    let kept = negative.value("bytes").expect("held");
    assert_eq!(kept.to_bits(), 0.0_f64.to_bits());
    assert_eq!(negative, positive);
    assert_eq!(hash_of(&negative), hash_of(&positive));
}

// ---------------------------------------------------------------------------
// Notes and equality
// ---------------------------------------------------------------------------

/// Test `with_notes` replaces the notes, in order, leaving the original
/// as it was.
#[test]
fn with_notes_replaces_the_notes() {
    let original = Measurement::failed(tiling_key(0), "crashed")
        .with_notes(vec![Note::with_other_kind("old")]);
    let notes = vec![
        Note::with_other_kind("first"),
        Note::with_other_kind("second"),
    ];

    let replaced = original.clone().with_notes(notes.clone());

    assert_eq!(replaced.notes(), notes.as_slice());
    assert_eq!(replaced.status(), original.status());
    assert_eq!(original.notes(), [Note::with_other_kind("old")].as_slice());
}

/// Test measurements built alike are equal and hash alike.
#[test]
fn measurements_built_alike_are_equal() {
    let left = measure_three(10.0, 2.0, 1.0);
    let right = measure_three(10.0, 2.0, 1.0);

    assert_eq!(left, right);
    assert_eq!(hash_of(&left), hash_of(&right));
}

/// Test measurements differing in any field are unequal.
#[test]
fn measurements_differing_anywhere_are_unequal() {
    let [latency, throughput, _] = three_objectives();
    let base = measured(0, &[(&latency, 1.0), (&throughput, 2.0)]);

    let others = [
        measured(1, &[(&latency, 1.0), (&throughput, 2.0)]),
        measured(0, &[(&latency, 1.5), (&throughput, 2.0)]),
        measured(0, &[(&throughput, 2.0), (&latency, 1.0)]),
        base.clone().with_notes(vec![Note::with_other_kind("n")]),
        Measurement::failed(tiling_key(0), "a"),
    ];

    for other in &others {
        assert_ne!(&base, other);
    }
    assert_ne!(
        Measurement::failed(tiling_key(0), "a"),
        Measurement::failed(tiling_key(0), "b")
    );
    assert_ne!(
        Measurement::failed(tiling_key(0), "a"),
        Measurement::infeasible(tiling_key(0), "a")
    );
}

// ---------------------------------------------------------------------------
// Dominance
// ---------------------------------------------------------------------------

/// Test a measurement better on one objective and as good on the others
/// dominates, and not the reverse.
#[rstest]
#[case::better_latency(measure_three(5.0, 2.0, 9.0), measure_three(10.0, 2.0, 1.0))]
#[case::better_throughput(measure_three(10.0, 3.0, 1.0), measure_three(10.0, 2.0, 1.0))]
#[case::better_on_both(measure_three(5.0, 3.0, 1.0), measure_three(10.0, 2.0, 1.0))]
fn a_better_measurement_dominates(#[case] better: Measurement, #[case] worse: Measurement) {
    assert_eq!(better.dominates(&worse), Ok(true));
    assert_eq!(worse.dominates(&better), Ok(false));
}

/// Test neither of two measurements that trade objectives off dominates.
#[test]
fn a_trade_off_dominates_neither_way() {
    let faster = measure_three(5.0, 1.0, 1.0);
    let wider = measure_three(10.0, 2.0, 1.0);

    assert_eq!(faster.dominates(&wider), Ok(false));
    assert_eq!(wider.dominates(&faster), Ok(false));
}

/// Test equal measurements, and measurements differing only in a reported
/// objective, do not dominate.
#[rstest]
#[case::equal(measure_three(5.0, 2.0, 1.0), measure_three(5.0, 2.0, 1.0))]
#[case::report_only(measure_three(5.0, 2.0, 1.0), measure_three(5.0, 2.0, 100.0))]
fn ties_do_not_dominate(#[case] left: Measurement, #[case] right: Measurement) {
    assert_eq!(left.dominates(&right), Ok(false));
    assert_eq!(right.dominates(&left), Ok(false));
}

/// Test measurements of reported objectives only never dominate.
#[test]
fn reported_objectives_alone_never_dominate() {
    let power = objective("power", Direction::Report);

    let low = measured(0, &[(&power, 1.0)]);
    let high = measured(1, &[(&power, 2.0)]);

    assert_eq!(low.dominates(&high), Ok(false));
    assert_eq!(high.dominates(&low), Ok(false));
}

/// Test the objectives are matched by name, whatever order each
/// measurement holds them in.
#[test]
fn dominance_matches_objectives_in_any_order() {
    let [latency, throughput, _] = three_objectives();
    let better = measured(0, &[(&latency, 5.0), (&throughput, 2.0)]);
    let worse = measured(1, &[(&throughput, 2.0), (&latency, 10.0)]);

    assert_eq!(better.dominates(&worse), Ok(true));
    assert_eq!(worse.dominates(&better), Ok(false));
}

/// Test measurements over different objectives are not compared.
#[rstest]
#[case::other_name(objective("bytes", Direction::Minimize))]
#[case::other_direction(objective("latency", Direction::Maximize))]
fn dominance_refuses_different_objectives(#[case] other: Objective) {
    let latency = objective("latency", Direction::Minimize);
    let left = measured(0, &[(&latency, 5.0)]);
    let right = measured(1, &[(&other, 10.0)]);

    assert_eq!(
        left.dominates(&right),
        Err(MeasurementError::DifferentObjectives)
    );
    assert_eq!(
        right.dominates(&left),
        Err(MeasurementError::DifferentObjectives)
    );
}

/// Test a measurement holding an extra objective is not compared.
#[test]
fn dominance_refuses_an_extra_objective() {
    let [latency, throughput, _] = three_objectives();
    let left = measured(0, &[(&latency, 5.0)]);
    let right = measured(1, &[(&latency, 10.0), (&throughput, 1.0)]);

    assert_eq!(
        left.dominates(&right),
        Err(MeasurementError::DifferentObjectives)
    );
    assert_eq!(
        right.dominates(&left),
        Err(MeasurementError::DifferentObjectives)
    );
}

/// Test only successful measurements are compared, before their
/// objectives are.
#[rstest]
#[case::infeasible(Measurement::infeasible(tiling_key(3), "rejected"))]
#[case::failed(Measurement::failed(tiling_key(3), "crashed"))]
#[case::timeout(Measurement::timeout(tiling_key(3)))]
fn dominance_refuses_a_measurement_that_did_not_succeed(#[case] failing: Measurement) {
    let ok = measure_three(5.0, 2.0, 1.0);

    assert_eq!(ok.dominates(&failing), Err(MeasurementError::NotOk));
    assert_eq!(failing.dominates(&ok), Err(MeasurementError::NotOk));
    assert_eq!(failing.dominates(&failing), Err(MeasurementError::NotOk));
}

// ---------------------------------------------------------------------------
// Error text
// ---------------------------------------------------------------------------

/// Test each refusal's text.
#[rstest]
#[case::empty_name(MeasurementError::EmptyName, "an objective needs a name")]
#[case::no_values(
    MeasurementError::NoValues,
    "a successful measurement needs at least one value"
)]
#[case::repeated(
    MeasurementError::RepeatedObjective { name: "latency".to_owned() },
    "the objective \"latency\" has two values"
)]
#[case::non_finite(
    MeasurementError::NonFiniteValue { objective: "bytes".to_owned() },
    "the value of the objective \"bytes\" is not finite"
)]
#[case::unexpected(
    MeasurementError::UnexpectedValues,
    "a measurement that did not succeed holds no values"
)]
#[case::not_ok(MeasurementError::NotOk, "only successful measurements are compared")]
#[case::different(
    MeasurementError::DifferentObjectives,
    "the measurements are over different objectives"
)]
fn measurement_error_text(#[case] error: MeasurementError, #[case] text: &str) {
    assert_eq!(error.to_string(), text);
}

// ---------------------------------------------------------------------------
// A measurer
// ---------------------------------------------------------------------------

/// A measurer of the tiling space's configurations: its latency is the
/// unroll times 10 minus the tile, and the flat layout is infeasible.
struct TilingMeasurer {
    objectives: [Objective; 1],
    /// The unroll variable's name.
    unroll: Identifier,
    /// The tile variable's name.
    tile: Identifier,
}

impl Measurer<Configuration> for TilingMeasurer {
    fn objectives(&self) -> &[Objective] {
        &self.objectives
    }

    fn measure(
        &mut self,
        key: &ConfigurationKey,
        subject: &Configuration,
    ) -> Result<Measurement, BoxError> {
        let read = |value: &Value| match value {
            Value::Int(integer) => i32::try_from(integer).map_err(BoxError::from),
            _ => Err(BoxError::from("an integer value")),
        };
        let unroll = read(
            subject
                .value(&self.unroll)
                .ok_or("the unroll is unassigned")?,
        )?;
        let Some(tile) = subject.value(&self.tile) else {
            return Ok(Measurement::infeasible(key.clone(), "the flat layout"));
        };
        let tile = read(tile)?;
        let latency = f64::from(unroll * 10 - tile);
        Ok(Measurement::ok(
            key.clone(),
            vec![(self.objectives[0].clone(), latency)],
        )?)
    }
}

/// Test a measurer over a space's configurations gives measurements whose
/// best, by the objective, is the configuration the measurer favors, and
/// which dominates every other successful one.
#[test]
fn a_measurer_ranks_the_configurations_of_a_space() {
    let tiling = build_tiling_space();
    let mut measurer = TilingMeasurer {
        objectives: [objective("latency", Direction::Minimize)],
        unroll: tiling.t.clone(),
        tile: tiling.x.clone(),
    };
    let configurations: Vec<Configuration> =
        with_context(|context| tiling.space.enumerate(context).collect::<Result<_, _>>())
            .expect("the tiling space enumerates");

    let measurements: Vec<Measurement> = configurations
        .iter()
        .map(|configuration| {
            let measurer: &mut dyn Measurer<Configuration> = &mut measurer;
            measurer.measure(&configuration.key(), configuration)
        })
        .collect::<Result<_, _>>()
        .expect("the measurer has no fault");

    let ok: Vec<&Measurement> = measurements.iter().filter(|m| m.is_ok()).collect();
    assert_eq!(ok.len(), 3);
    assert_eq!(measurements.len() - ok.len(), 2);
    let best = ok
        .iter()
        .copied()
        .reduce(|best, candidate| {
            let latency = &measurer.objectives()[0];
            let ordering = latency.compare(
                candidate.value("latency").unwrap_or(f64::NAN),
                best.value("latency").unwrap_or(f64::NAN),
            );
            if ordering == Some(Ordering::Greater) {
                candidate
            } else {
                best
            }
        })
        .expect("a successful measurement");
    assert_eq!(best.value("latency"), Some(7.0));
    for other in ok.iter().filter(|other| ***other != *best) {
        assert_eq!(best.dominates(other), Ok(true));
    }
}
