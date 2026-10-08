//! Tests for `MeasurementKey`: a measurement keyed by a configuration or
//! by a run's trace, the keys' conversions and comparisons, measurements
//! of a run through every constructor, and their wire form tagged
//! `trace`.

use std::collections::HashMap;

use fhy_core::foreign::NoForeign;
use fhy_core::identifier::Identifier;
use fhy_core::search_space::wire::MeasurementData;
use fhy_core::search_space::{
    ConfigurationKey, Direction, Measurement, MeasurementKey, MeasurementStatus, Objective,
    TraceKey,
};
use rstest::rstest;
use serde_json::json;

use crate::support::hashing::hash_of;
use crate::support::measurement::{objective, tiling_key};
use crate::support::search::{build_tiling_space, record_tiling_run};
use crate::support::serde::{check_serde_round_trip, json_of};

/// Return the key of a run of the tiling space that placed its buffer at
/// `address`, and the key of that run's configuration.
fn record_keys(address: u64) -> (TraceKey, ConfigurationKey) {
    let tiling = build_tiling_space();
    let (trace, configuration) = record_tiling_run(&tiling, address, &Identifier::new("buffer"));
    (trace.key(), configuration.key())
}

/// Return the objective `latency`, minimized.
fn latency() -> Objective {
    objective("latency", Direction::Minimize)
}

/// Return the successful measurement of `key` with the latency `value`.
fn measure_latency(key: impl Into<MeasurementKey>, value: f64) -> Measurement {
    Measurement::ok(key, vec![(latency(), value)]).expect("a finite value")
}

/// Return the configuration key `measurement` holds.
///
/// # Panics
///
/// Panics if the measurement is of a run.
fn configuration_key_of(measurement: &Measurement) -> ConfigurationKey {
    match measurement.key() {
        MeasurementKey::Configuration(key) => key.clone(),
        MeasurementKey::Trace(key) => panic!("expected a configuration key, got {key:?}"),
    }
}

// ---------------------------------------------------------------------------
// Conversions and comparisons
// ---------------------------------------------------------------------------

/// Test a configuration key converts to the configuration variant, equal
/// to that key and to no trace key.
#[test]
fn measurement_key_from_a_configuration_key_is_that_configuration() {
    let (trace_key, configuration_key) = record_keys(17);

    let key = MeasurementKey::from(configuration_key.clone());

    assert_eq!(
        key,
        MeasurementKey::Configuration(configuration_key.clone())
    );
    assert_eq!(key, configuration_key);
    assert_ne!(key, trace_key);
}

/// Test a trace key converts to the trace variant, equal to that key and
/// to no configuration key.
#[test]
fn measurement_key_from_a_trace_key_is_that_trace() {
    let (trace_key, configuration_key) = record_keys(17);

    let key = MeasurementKey::from(trace_key.clone());

    assert_eq!(key, MeasurementKey::Trace(trace_key.clone()));
    assert_eq!(key, trace_key);
    assert_ne!(key, configuration_key);
}

/// Test a measurement key compares unequal to a key of the same variant
/// holding another key.
#[test]
fn measurement_key_compares_the_key_it_holds() {
    let (low_trace, low_configuration) = record_keys(17);
    let (high_trace, _) = record_keys(18);

    let low = MeasurementKey::from(low_trace);

    assert_ne!(low, high_trace);
    assert_ne!(MeasurementKey::from(low_configuration), tiling_key(3));
}

/// Test two runs that placed an address differently are two entries of a
/// cache keyed by measurement key, though their configuration keys are
/// one entry.
#[test]
fn measurement_keys_of_runs_differing_in_a_dynamic_step_are_distinct() {
    let (low_trace, low_configuration) = record_keys(17);
    let (high_trace, high_configuration) = record_keys(18);

    let by_run: HashMap<MeasurementKey, f64> = [
        (MeasurementKey::from(low_trace), 1.0),
        (MeasurementKey::from(high_trace), 2.0),
    ]
    .into_iter()
    .collect();
    let by_configuration: HashMap<MeasurementKey, f64> = [
        (MeasurementKey::from(low_configuration), 1.0),
        (MeasurementKey::from(high_configuration), 2.0),
    ]
    .into_iter()
    .collect();

    assert_eq!(by_run.len(), 2);
    assert_eq!(by_configuration.len(), 1);
}

/// Test the keys of one run recorded twice, with fresh subjects, are one
/// measurement key with one hash.
#[test]
fn measurement_keys_of_one_run_recorded_twice_are_equal() {
    let (first, _) = record_keys(40);
    let (second, _) = record_keys(40);

    let first = MeasurementKey::from(first);
    let second = MeasurementKey::from(second);

    assert_eq!(first, second);
    assert_eq!(hash_of(&first), hash_of(&second));
}

// ---------------------------------------------------------------------------
// Measurements of a run
// ---------------------------------------------------------------------------

/// Test every constructor keeps a run's trace key as the measurement's
/// key, with its status.
#[rstest]
#[case::ok(|key: TraceKey| measure_latency(key, 3.0), MeasurementStatus::Ok)]
#[case::infeasible(
    |key: TraceKey| Measurement::infeasible(key, "no placement"),
    MeasurementStatus::Infeasible { reason: "no placement".to_owned() }
)]
#[case::failed(
    |key: TraceKey| Measurement::failed(key, "crashed"),
    MeasurementStatus::Failed { reason: "crashed".to_owned() }
)]
#[case::timeout(Measurement::timeout, MeasurementStatus::Timeout)]
fn measurement_of_a_run_keeps_its_trace_key(
    #[case] build: fn(TraceKey) -> Measurement,
    #[case] status: MeasurementStatus,
) {
    let (trace_key, _) = record_keys(17);

    let measurement = build(trace_key.clone());

    assert_eq!(measurement.key(), &MeasurementKey::Trace(trace_key));
    assert_eq!(measurement.status(), &status);
}

/// Test measurements of the two runs of one configuration differ by key,
/// and a measurement of the configuration differs from both.
#[test]
fn measurements_of_runs_and_of_their_configuration_differ_by_key() {
    let (low_trace, configuration) = record_keys(17);
    let (high_trace, _) = record_keys(18);

    let low = measure_latency(low_trace, 1.0);
    let high = measure_latency(high_trace, 1.0);
    let of_configuration = measure_latency(configuration.clone(), 1.0);

    assert_ne!(low, high);
    assert_ne!(low, of_configuration);
    assert_eq!(configuration_key_of(&of_configuration), configuration);
}

// ---------------------------------------------------------------------------
// Wire form
// ---------------------------------------------------------------------------

/// Test a measurement of a run writes its key tagged `trace`, holding the
/// trace key's own form.
#[test]
fn measurement_of_a_run_serializes_its_key_tagged_trace() {
    let (trace_key, _) = record_keys(17);
    let measurement = Measurement::timeout(trace_key.clone());

    let written = json_of(&measurement);

    assert_eq!(written["key"], json!({"trace": json_of(&trace_key)}));
}

/// Test a measurement key writes the configuration variant tagged
/// `configuration` and the trace variant tagged `trace`.
#[test]
fn measurement_key_serializes_tagged_by_its_variant() {
    let (trace_key, configuration_key) = record_keys(17);

    let by_configuration = json_of(&MeasurementKey::Configuration(configuration_key.clone()));
    let by_trace = json_of(&MeasurementKey::Trace(trace_key.clone()));

    assert_eq!(
        by_configuration,
        json!({"configuration": json_of(&configuration_key)})
    );
    assert_eq!(by_trace, json!({"trace": json_of(&trace_key)}));
}

/// Test a measurement key of either variant round-trips through JSON and
/// postcard.
#[rstest]
#[case::trace(|(trace_key, _): (TraceKey, ConfigurationKey)| MeasurementKey::Trace(trace_key))]
#[case::configuration(
    |(_, configuration_key): (TraceKey, ConfigurationKey)| {
        MeasurementKey::Configuration(configuration_key)
    }
)]
fn measurement_key_round_trips_through_json_and_postcard(
    #[case] select: fn((TraceKey, ConfigurationKey)) -> MeasurementKey,
) {
    let key = select(record_keys(17));

    let result = check_serde_round_trip(&key);

    result.unwrap_or_else(|failure| panic!("{failure}"));
}

/// Test a measurement of a run, of every status, round-trips through JSON
/// and postcard by `Measurement`'s own decoding.
#[rstest]
#[case::ok(|key: TraceKey| measure_latency(key, 3.0))]
#[case::infeasible(|key: TraceKey| Measurement::infeasible(key, "no placement"))]
#[case::failed(|key: TraceKey| Measurement::failed(key, "crashed"))]
#[case::timeout(Measurement::timeout)]
fn measurement_of_a_run_round_trips_through_json_and_postcard(
    #[case] build: fn(TraceKey) -> Measurement,
) {
    let (trace_key, _) = record_keys(17);
    let measurement = build(trace_key);

    let result = check_serde_round_trip(&measurement);

    result.unwrap_or_else(|failure| panic!("{failure}"));
}

/// Test the wire data of a measurement of a run writes what the
/// measurement writes and builds back to it.
#[test]
fn measurement_data_of_a_run_round_trips() {
    let (trace_key, _) = record_keys(17);
    let measurement = measure_latency(trace_key, 3.0);

    let data = MeasurementData::of(&measurement).expect("a wire form");
    let text = serde_json::to_string(&data).expect("encodes");
    let decoded: MeasurementData = serde_json::from_str(&text).expect("decodes");
    let built = decoded.build(&NoForeign).expect("no opaque value");

    assert_eq!(text, serde_json::to_string(&measurement).expect("encodes"));
    assert_eq!(built, measurement);
}

/// Test decoding a measurement refuses a trace key holding a coordinate
/// its step's domain does not contain, by both decodings.
#[test]
fn measurement_decoding_refuses_a_trace_key_with_a_coordinate_outside_its_domain() {
    let text = json!({
        "key": {"trace": {"steps": [{
            "kind": "k",
            "decision": null,
            "domain": {"strided": [{"start": "0", "stop": "4", "stride": "1"}]},
            "coordinate": {"index": 4},
        }]}},
        "status": "timeout",
        "values": [],
        "notes": [],
    })
    .to_string();

    serde_json::from_str::<Measurement>(&text).expect_err("the coordinate is outside");
    serde_json::from_str::<MeasurementData>(&text).expect_err("the coordinate is outside");
}
