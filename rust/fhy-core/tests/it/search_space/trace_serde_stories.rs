//! Tests for the wire form of a trace: the pinned JSON shape of static and
//! dynamic steps, opaque values written as `null`, round trips, and the
//! refusals of reading.

use fhy_core::constraint::Value;
use fhy_core::identifier::Identifier;
use fhy_core::search_space::{ChoiceDomain, Coordinate, Recorder, StepDomain, Trace, TraceStep};
use rstest::rstest;
use serde_json::json;

use crate::support::constraint::{TestOpaque, int};
use crate::support::search::{
    ScriptedOracle, TilingSpace, build_tiling_space, index, kind, order, order_of, strided,
    with_context,
};
use crate::support::serde::{check_serde_round_trip, json_of};

/// Return a trace over the tiling space: `t = 1`, `c = a`, a dynamic
/// address step answered 17, and `x = 2`.
fn record_tiling_trace(tiling: &TilingSpace) -> Trace {
    let mut recorder = Recorder::over(&tiling.space);
    let mut oracle = ScriptedOracle::new([index(0), index(0), index(17), index(1)]);
    with_context(|context| {
        recorder.decide(&tiling.t, &mut oracle, context)?;
        recorder.decide(&tiling.c, &mut oracle, context)?;
        recorder.decide_dynamic(
            &kind("moga.cir.address"),
            &tiling.x,
            &strided(&[(0, 64)]),
            &mut oracle,
            context,
        )?;
        recorder.decide(&tiling.x, &mut oracle, context)
    })
    .expect("admissible answers");
    recorder.trace()
}

/// Test the JSON shape of static and dynamic steps: kind, subject,
/// decision, signature, coordinate and value, with the names the space
/// binds written by position among its names.
#[test]
fn trace_serializes_to_the_pinned_shape() {
    let tiling = build_tiling_space();
    let trace = record_tiling_trace(&tiling);

    let written = json_of(&trace);

    let expected = json!({"steps": [
        {
            "kind": "search_space.variable",
            "subject": json_of(&tiling.t),
            "decision": 0,
            "domain": {"choice": [{"value": {"int": "1"}}, {"value": {"int": "2"}}]},
            "coordinate": {"index": 0},
            "value": {"int": "1"},
        },
        {
            "kind": "search_space.choice",
            "subject": json_of(&tiling.c),
            "decision": 1,
            "domain": {"choice": [{"bound": 3}, {"bound": 5}]},
            "coordinate": {"index": 0},
            "value": json_of(&Value::Identifier(tiling.a.clone())),
        },
        {
            "kind": "moga.cir.address",
            "subject": json_of(&tiling.x),
            "decision": null,
            "domain": {"strided": [{"start": "0", "stop": "64", "stride": "1"}]},
            "coordinate": {"index": 17},
            "value": {"int": "17"},
        },
        {
            "kind": "search_space.variable",
            "subject": json_of(&tiling.x),
            "decision": 2,
            "domain": {"choice": [
                {"value": {"int": "1"}}, {"value": {"int": "2"}}, {"value": {"int": "3"}},
            ]},
            "coordinate": {"index": 1},
            "value": {"int": "2"},
        },
    ]});
    assert_eq!(written, expected);
}

/// Test an order coordinate is tagged `order` and holds its positions, and free
/// identifiers in a signature are written as `identifier`.
#[test]
fn trace_writes_an_order_coordinate_as_an_array() {
    let [i, j, k] = ["i", "j", "k"].map(Identifier::new);
    let step = TraceStep::dynamic(
        kind("walk"),
        i.clone(),
        &order_of(&[&i, &j, &k]),
        order(&[1, 2, 0]),
    )
    .expect("a permutation");

    let written = json_of(&Trace::new(vec![step]));

    assert_eq!(
        written["steps"][0]["coordinate"],
        json!({"order": [1, 2, 0]})
    );
    assert_eq!(
        written["steps"][0]["domain"],
        json!({"order": ["identifier", "identifier", "identifier"]})
    );
}

/// Test an opaque value is written as `null`, in the value and in the
/// signature, and reads back as no value.
#[test]
fn trace_writes_an_opaque_value_as_null() {
    let domain = StepDomain::from(
        ChoiceDomain::new(vec![
            TestOpaque::token(1).into_value(),
            TestOpaque::token(2).into_value(),
        ])
        .expect("distinct tokens"),
    );
    let step = TraceStep::dynamic(
        kind("moga.cir.option"),
        Identifier::new("vertex"),
        &domain,
        index(1),
    )
    .expect("in range");
    let trace = Trace::new(vec![step]);

    let text = serde_json::to_string(&trace).expect("the trace serializes");
    let decoded: Trace = serde_json::from_str(&text).expect("the text decodes");

    let written: serde_json::Value = serde_json::from_str(&text).expect("JSON");
    assert_eq!(written["steps"][0]["value"], serde_json::Value::Null);
    assert_eq!(
        written["steps"][0]["domain"],
        json!({"choice": ["opaque", "opaque"]})
    );
    assert_eq!(decoded.steps()[0].value(), None);
    assert_eq!(decoded.steps()[0].coordinate(), &index(1));
    assert_eq!(decoded.steps()[0].signature(), trace.steps()[0].signature());
}

/// Test a trace of plain and identifier values round-trips through JSON to
/// an equal trace.
#[test]
fn trace_round_trips_through_json() {
    let tiling = build_tiling_space();
    let trace = record_tiling_trace(&tiling);

    let text = serde_json::to_string(&trace).expect("the trace serializes");
    let decoded: Trace = serde_json::from_str(&text).expect("the text decodes");

    assert_eq!(decoded, trace);
}

/// Test a decoded trace replays into the configuration it records.
#[test]
fn trace_decoded_from_json_replays_into_its_configuration() {
    let tiling = build_tiling_space();
    let trace = record_tiling_trace(&tiling);
    let text = serde_json::to_string(&trace).expect("the trace serializes");
    let decoded: Trace = serde_json::from_str(&text).expect("the text decodes");

    let configuration = with_context(|context| tiling.space.replay(&decoded, context))
        .expect("the static steps describe a configuration");

    assert_eq!(configuration.value(&tiling.x), Some(&int(2)));
    assert_eq!(configuration.value(&tiling.t), Some(&int(1)));
}

/// Test an empty trace is written as no steps.
#[test]
fn trace_empty_serializes_as_no_steps() {
    let written = json_of(&Trace::default());

    assert_eq!(written, json!({"steps": []}));
}

/// Return the JSON text of a trace of one dynamic step of `kind` over
/// the one run `[start, stop)`, answered with `coordinate`.
fn build_step_text(kind: &str, start: i64, stop: i64, coordinate: &serde_json::Value) -> String {
    json!({"steps": [{
        "kind": kind,
        "subject": json_of(&Identifier::new("s")),
        "decision": null,
        "domain": {"strided": [{
            "start": start.to_string(),
            "stop": stop.to_string(),
            "stride": "1",
        }]},
        "coordinate": coordinate,
        "value": null,
    }]})
    .to_string()
}

/// Test a well-formed step decodes, the control of the refusals below.
#[test]
fn trace_decoding_reads_a_well_formed_step() {
    let text = build_step_text("k", 0, 4, &json!({"index": 3}));

    let decoded: Trace = serde_json::from_str(&text).expect("a valid step");

    assert_eq!(decoded.steps()[0].coordinate(), &index(3));
    assert_eq!(decoded.steps()[0].kind().as_str(), "k");
}

/// Test reading refuses a coordinate its signature does not contain, an
/// empty kind, and an empty run.
#[rstest]
#[case::past_the_end(build_step_text("k", 0, 4, &json!({"index": 4})))]
#[case::wrong_shape(build_step_text("k", 0, 4, &json!({"order": [0]})))]
#[case::untagged(build_step_text("k", 0, 4, &json!(3)))]
#[case::empty_kind(build_step_text("", 0, 4, &json!({"index": 0})))]
#[case::empty_run(build_step_text("k", 4, 4, &json!({"index": 0})))]
fn trace_decoding_refuses_a_malformed_step(#[case] text: String) {
    let result = serde_json::from_str::<Trace>(&text);

    result.expect_err("a malformed step is refused");
}

/// Test a coordinate round-trips through JSON and postcard.
#[rstest]
#[case::index(index(7))]
#[case::large_index(index(u64::MAX))]
#[case::order(order(&[2, 0, 1]))]
#[case::empty_order(order(&[]))]
fn coordinate_round_trips_through_json_and_postcard(#[case] coordinate: Coordinate) {
    let result = check_serde_round_trip(&coordinate);

    result.unwrap_or_else(|failure| panic!("{failure}"));
}

/// Test a coordinate is written tagged by its shape.
#[rstest]
#[case::index(index(7), json!({"index": 7}))]
#[case::order(order(&[2, 0, 1]), json!({"order": [2, 0, 1]}))]
fn coordinate_serializes_tagged_by_its_shape(
    #[case] coordinate: Coordinate,
    #[case] expected: serde_json::Value,
) {
    let written = json_of(&coordinate);

    assert_eq!(written, expected);
}

/// Test static and dynamic steps round-trip through JSON and postcard.
#[test]
fn trace_step_round_trips_through_json_and_postcard() {
    let tiling = build_tiling_space();
    let trace = record_tiling_trace(&tiling);

    let failures: Vec<String> = trace
        .steps()
        .iter()
        .filter_map(|step| check_serde_round_trip(step).err())
        .map(|failure| failure.to_string())
        .collect();

    assert!(failures.is_empty(), "{failures:?}");
}

/// Test a trace round-trips through JSON and postcard.
#[test]
fn trace_round_trips_through_json_and_postcard() {
    let tiling = build_tiling_space();
    let trace = record_tiling_trace(&tiling);

    let result = check_serde_round_trip(&trace);

    result.unwrap_or_else(|failure| panic!("{failure}"));
}

/// Test a step serializes as it does inside a trace.
#[test]
fn trace_step_serializes_as_inside_a_trace() {
    let tiling = build_tiling_space();
    let trace = record_tiling_trace(&tiling);

    let written = json_of(&trace.steps()[2]);

    assert_eq!(written, json_of(&trace)["steps"][2]);
}
