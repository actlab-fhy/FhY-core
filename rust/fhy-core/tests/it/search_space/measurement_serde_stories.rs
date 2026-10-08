//! Tests for the wire forms of `Direction`, `Objective` and `Measurement`:
//! the shapes pinned, the key tagged as a configuration's or a trace's,
//! round trips through JSON and postcard, and the refusals of a payload
//! outside the constructors' rules.

use fhy_core::diagnostic::Note;
use fhy_core::foreign::NoForeign;
use fhy_core::search_space::wire::MeasurementData;
use fhy_core::search_space::{Direction, Measurement, MeasurementError, MeasurementKey, Objective};
use rstest::rstest;
use serde::Serialize;
use serde_json::json;

use crate::support::measurement::{measured, objective, tiling_key};
use crate::support::param::Level;
use crate::support::search_space::{categorical, configure, plain_variable, space_of};
use crate::support::serde::check_serde_round_trip;

/// Return the JSON value `value` serializes to.
fn json_of<T: Serialize>(value: &T) -> serde_json::Value {
    serde_json::to_value(value).expect("the value serializes")
}

/// Return the JSON payload of a measurement of tiling configuration 0 with
/// the status `status` and the values `values`.
fn payload(status: &serde_json::Value, values: &serde_json::Value) -> String {
    json!({
        "key": {"configuration": json_of(&tiling_key(0))},
        "status": status,
        "values": values,
        "notes": [],
    })
    .to_string()
}

/// Return the error a payload decoding as a measurement fails with, as
/// text.
fn decoding_error(text: &str) -> String {
    serde_json::from_str::<Measurement>(text)
        .expect_err("the payload is refused")
        .to_string()
}

/// Test a direction is written as its name.
#[rstest]
#[case::minimize(Direction::Minimize, "minimize")]
#[case::maximize(Direction::Maximize, "maximize")]
#[case::report(Direction::Report, "report")]
fn direction_serializes_as_its_name(#[case] direction: Direction, #[case] name: &str) {
    assert_eq!(json_of(&direction), json!(name));
    check_serde_round_trip(&direction).unwrap_or_else(|failure| panic!("{failure}"));
}

/// Test an objective is written as its name and direction.
#[test]
fn objective_serializes_its_name_and_direction() {
    let latency = objective("latency_cycles", Direction::Minimize);

    assert_eq!(
        json_of(&latency),
        json!({"name": "latency_cycles", "direction": "minimize"})
    );
}

/// Test an objective round-trips through JSON and postcard.
#[rstest]
#[case::minimize(objective("latency", Direction::Minimize))]
#[case::report(objective("débit µs", Direction::Report))]
fn objective_round_trips_through_json_and_postcard(#[case] objective: Objective) {
    check_serde_round_trip(&objective).unwrap_or_else(|failure| panic!("{failure}"));
}

/// Test an objective's payload with an empty name, an unknown direction or
/// another field is refused.
#[rstest]
#[case::empty_name(r#"{"name": "", "direction": "minimize"}"#)]
#[case::unknown_direction(r#"{"name": "n", "direction": "lower"}"#)]
#[case::extra_field(r#"{"name": "n", "direction": "minimize", "unit": "s"}"#)]
#[case::missing_direction(r#"{"name": "n"}"#)]
fn objective_decoding_refuses_a_malformed_payload(#[case] text: &str) {
    serde_json::from_str::<Objective>(text).expect_err("a malformed objective is refused");
}

/// Test an empty name read back names the refusal.
#[test]
fn objective_decoding_names_an_empty_name() {
    let error = serde_json::from_str::<Objective>(r#"{"name": "", "direction": "minimize"}"#)
        .expect_err("refused");

    assert!(
        error
            .to_string()
            .contains(&MeasurementError::EmptyName.to_string()),
        "{error}"
    );
}

/// Test a successful measurement is written with its key, status, values in
/// order and notes.
#[test]
fn ok_measurement_serializes_its_fields() {
    let latency = objective("latency_cycles", Direction::Minimize);
    let throughput = objective("throughput", Direction::Maximize);
    let measurement = measured(0, &[(&latency, 1532.0), (&throughput, 0.5)])
        .with_notes(vec![Note::with_other_kind("warm cache")]);

    let written = json_of(&measurement);

    assert_eq!(
        written,
        json!({
            "key": {"configuration": json_of(&tiling_key(0))},
            "status": "ok",
            "values": [
                {"objective": {"name": "latency_cycles", "direction": "minimize"}, "value": 1532.0},
                {"objective": {"name": "throughput", "direction": "maximize"}, "value": 0.5},
            ],
            "notes": [json_of(&Note::with_other_kind("warm cache"))],
        })
    );
}

/// Test a failing status is written tagged, with no values.
#[rstest]
#[case::infeasible(
    Measurement::infeasible(tiling_key(1), "placement"),
    json!({"infeasible": {"reason": "placement"}})
)]
#[case::failed(
    Measurement::failed(tiling_key(1), "crash"),
    json!({"failed": {"reason": "crash"}})
)]
#[case::timeout(Measurement::timeout(tiling_key(1)), json!("timeout"))]
fn failing_measurement_serializes_its_status(
    #[case] measurement: Measurement,
    #[case] status: serde_json::Value,
) {
    let written = json_of(&measurement);

    assert_eq!(written["status"], status);
    assert_eq!(written["values"], json!([]));
    assert_eq!(
        written["key"],
        json!({"configuration": json_of(&tiling_key(1))})
    );
}

/// Test every kind of measurement round-trips through JSON and postcard.
#[rstest]
#[case::ok(measured(
    2,
    &[
        (&objective("latency", Direction::Minimize), 1.25),
        (&objective("bytes", Direction::Report), 1e300),
    ],
))]
#[case::ok_with_notes(
    measured(3, &[(&objective("latency", Direction::Minimize), -7.0)])
        .with_notes(vec![Note::with_other_kind("n")])
)]
#[case::infeasible(Measurement::infeasible(tiling_key(4), "rejected by validation"))]
#[case::failed(Measurement::failed(tiling_key(4), ""))]
#[case::timeout(Measurement::timeout(tiling_key(4)))]
fn measurement_round_trips_through_json_and_postcard(#[case] measurement: Measurement) {
    check_serde_round_trip(&measurement).unwrap_or_else(|failure| panic!("{failure}"));
}

/// Test the wire data of a measurement builds back the same measurement.
#[test]
fn measurement_data_builds_back_the_measurement() {
    let measurement = measured(1, &[(&objective("latency", Direction::Minimize), 3.0)]);

    let data = MeasurementData::of(&measurement).expect("the key has no opaque value");
    let built = data.build(&NoForeign).expect("the data is valid");

    assert_eq!(built, measurement);
}

/// Test a payload breaking a constructor's rule is refused with the
/// constructor's error.
#[rstest]
#[case::ok_without_values(payload(&json!("ok"), &json!([])), MeasurementError::NoValues)]
#[case::repeated_objective(
    payload(&json!("ok"), &json!([
        {"objective": {"name": "a", "direction": "minimize"}, "value": 1.0},
        {"objective": {"name": "a", "direction": "maximize"}, "value": 2.0},
    ])),
    MeasurementError::RepeatedObjective { name: "a".to_owned() }
)]
#[case::values_on_a_failure(
    payload(&json!({"failed": {"reason": "r"}}), &json!([
        {"objective": {"name": "a", "direction": "minimize"}, "value": 1.0},
    ])),
    MeasurementError::UnexpectedValues
)]
#[case::values_on_a_timeout(
    payload(&json!("timeout"), &json!([
        {"objective": {"name": "a", "direction": "minimize"}, "value": 1.0},
    ])),
    MeasurementError::UnexpectedValues
)]
fn measurement_decoding_refuses_what_a_constructor_refuses(
    #[case] text: String,
    #[case] error: MeasurementError,
) {
    let refused = decoding_error(&text);

    assert!(refused.contains(&error.to_string()), "{refused}");
}

/// Test a payload of another shape is refused.
#[rstest]
#[case::unknown_status(payload(&json!("crashed"), &json!([])))]
#[case::untagged_reason(payload(&json!({"reason": "r"}), &json!([])))]
#[case::value_as_text(payload(&json!("ok"), &json!([
    {"objective": {"name": "a", "direction": "minimize"}, "value": "1.0"},
])))]
#[case::extra_field(json!({
    "key": {"configuration": json_of(&tiling_key(0))}, "status": "timeout", "values": [],
    "notes": [], "space": null,
}).to_string())]
fn measurement_decoding_refuses_another_shape(#[case] text: String) {
    serde_json::from_str::<Measurement>(&text).expect_err("another shape is refused");
}

/// Return the postcard bytes of a successful measurement of tiling
/// configuration 0 whose one value, of `latency`, is `value`, written
/// without the constructor's checks.
fn forge_postcard(value: f64) -> Vec<u8> {
    /// The status's first variant, `ok`, as postcard numbers variants.
    #[derive(Serialize)]
    enum Status {
        Ok,
    }
    #[derive(Serialize)]
    struct Entry {
        objective: Objective,
        value: f64,
    }
    #[derive(Serialize)]
    struct Forged {
        key: MeasurementKey,
        status: Status,
        values: Vec<Entry>,
        notes: Vec<Note>,
    }
    postcard::to_allocvec(&Forged {
        key: MeasurementKey::Configuration(tiling_key(0)),
        status: Status::Ok,
        values: vec![Entry {
            objective: objective("latency", Direction::Minimize),
            value,
        }],
        notes: Vec::new(),
    })
    .expect("the forged payload encodes")
}

/// Test the forged postcard payload is a measurement's when its value is
/// finite: the control of the refusal below.
#[test]
fn a_forged_postcard_measurement_with_a_finite_value_decodes() {
    let decoded: Measurement =
        postcard::from_bytes(&forge_postcard(2.5)).expect("a finite value decodes");

    assert_eq!(
        decoded,
        measured(0, &[(&objective("latency", Direction::Minimize), 2.5)])
    );
}

/// Test a non-finite value read through postcard, which carries one, is
/// refused.
#[rstest]
#[case::nan(f64::NAN)]
#[case::infinity(f64::INFINITY)]
#[case::negative_infinity(f64::NEG_INFINITY)]
fn measurement_decoding_refuses_a_non_finite_value(#[case] value: f64) {
    let error = postcard::from_bytes::<Measurement>(&forge_postcard(value))
        .expect_err("a non-finite value");

    assert!(
        matches!(error, postcard::Error::SerdeDeCustom),
        "the refusal is the constructor's, not the format's: {error:?}"
    );
}

/// Test a measurement of a key holding an opaque value with no foreign
/// form cannot be written.
#[test]
fn measurement_of_an_opaque_key_value_cannot_be_written() {
    let [name, level] = ["levels", "level"].map(fhy_core::identifier::Identifier::new);
    let space = space_of(
        &name,
        vec![plain_variable(
            &level,
            categorical(vec![Level::value(1), Level::value(2)]),
        )],
        Vec::new(),
    );
    let key = configure(&space, [(level, Level::value(2))]).key();
    let measurement = Measurement::timeout(key);

    MeasurementData::of(&measurement).expect_err("the opaque value has no foreign form");
    serde_json::to_string(&measurement).expect_err("the measurement cannot be written");
}
