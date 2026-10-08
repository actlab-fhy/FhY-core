//! Tests for `Trace::key` and `TraceKey`: what a key drops (subjects and
//! values) and what it keeps (kind, decision position, signature,
//! coordinate), its accessors, and its wire form.

use fhy_core::identifier::Identifier;
use fhy_core::search_space::{Coordinate, StepDomain, Trace, TraceKey, TraceStep};
use rstest::rstest;
use serde_json::json;

use crate::support::hashing::hash_of;
use crate::support::search::{
    build_tiling_space, index, kind, order, order_of, record_tiling_run, strided,
};
use crate::support::serde::check_serde_round_trip;

/// Return the JSON value `value` serializes to.
fn json_of<T: serde::Serialize>(value: &T) -> serde_json::Value {
    serde_json::to_value(value).expect("the value serializes")
}

// ---------------------------------------------------------------------------
// What a key is of
// ---------------------------------------------------------------------------

/// Test two runs that answered alike over one space, their dynamic subjects
/// minted anew for each run, are different traces with one key.
#[test]
fn trace_key_ignores_the_subjects_a_run_minted() {
    let tiling = build_tiling_space();

    let (first, _) = record_tiling_run(&tiling, 17, &Identifier::new("buffer"));
    let (second, _) = record_tiling_run(&tiling, 17, &Identifier::new("buffer"));

    assert_ne!(first, second, "the dynamic subjects differ");
    assert_eq!(first.key(), second.key());
    assert_eq!(hash_of(&first.key()), hash_of(&second.key()));
}

/// Test two runs that placed a dynamic address differently have different
/// trace keys, though their configurations, and so their configuration
/// keys, are equal.
#[test]
fn trace_key_differs_on_a_dynamic_coordinate_where_the_configuration_key_does_not() {
    let tiling = build_tiling_space();
    let subject = Identifier::new("buffer");

    let (low, low_configuration) = record_tiling_run(&tiling, 17, &subject);
    let (high, high_configuration) = record_tiling_run(&tiling, 18, &subject);

    assert_eq!(low_configuration.key(), high_configuration.key());
    assert_ne!(low.key(), high.key());
}

/// Test the keys of two runs over spaces built apart, whose identifiers are
/// all different, are equal when the runs answered alike.
#[test]
fn trace_key_is_equal_across_alpha_equivalent_spaces() {
    let (left, right) = (build_tiling_space(), build_tiling_space());

    let (left_trace, _) = record_tiling_run(&left, 9, &Identifier::new("buffer"));
    let (right_trace, _) = record_tiling_run(&right, 9, &Identifier::new("buffer"));

    assert_ne!(left_trace, right_trace, "the subjects differ");
    assert_eq!(left_trace.key(), right_trace.key());
}

/// Build the one-step trace of kind `kind_name` over `domain`, answered
/// `answer`, about a fresh subject.
fn build_dynamic_trace(kind_name: &str, domain: &StepDomain, answer: Coordinate) -> Trace {
    Trace::new(vec![
        TraceStep::dynamic(kind(kind_name), Identifier::new("s"), domain, answer)
            .expect("the answer is in the domain"),
    ])
}

/// Test a key tells apart the kind of a step, the domain it was asked over
/// and the order the steps were asked in.
#[test]
fn trace_key_differs_on_the_kind_the_domain_and_the_order_of_the_steps() {
    let base = build_dynamic_trace("address", &strided(&[(0, 8)]), index(3));
    let other_kind = build_dynamic_trace("option", &strided(&[(0, 8)]), index(3));
    let other_domain = build_dynamic_trace("address", &strided(&[(8, 16)]), index(3));
    let (first, second) = (
        build_dynamic_trace("address", &strided(&[(0, 8)]), index(3)),
        build_dynamic_trace("option", &strided(&[(0, 4)]), index(1)),
    );
    let forward = Trace::new([first.steps(), second.steps()].concat());
    let backward = Trace::new([second.steps(), first.steps()].concat());

    assert_ne!(base.key(), other_kind.key());
    assert_ne!(base.key(), other_domain.key());
    assert_ne!(forward.key(), backward.key());
}

/// Test a key keeps the coordinate of an order step, as the ordering a run
/// chose.
#[test]
fn trace_key_differs_on_the_ordering_an_order_step_chose() {
    let [i, j, k] = ["i", "j", "k"].map(Identifier::new);
    let domain = order_of(&[&i, &j, &k]);

    let forward = build_dynamic_trace("walk", &domain, order(&[0, 1, 2]));
    let reversed = build_dynamic_trace("walk", &domain, order(&[2, 1, 0]));

    assert_ne!(forward.key(), reversed.key());
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

/// Test a key has a step per step of its trace, and its coordinates are the
/// trace's, in order.
#[test]
fn trace_key_counts_and_lists_the_coordinates_of_its_steps() {
    let tiling = build_tiling_space();
    let (trace, _) = record_tiling_run(&tiling, 17, &Identifier::new("buffer"));

    let key = trace.key();

    assert_eq!(key.len(), 4);
    assert!(!key.is_empty());
    assert_eq!(key.coordinates().len(), 4);
    assert_eq!(
        key.coordinates().collect::<Vec<_>>(),
        [&index(0), &index(0), &index(17), &index(1)]
    );
    assert_eq!(
        key.coordinates().collect::<Vec<_>>(),
        trace.coordinates().collect::<Vec<_>>()
    );
}

/// Test the key of an empty trace has no step.
#[test]
fn trace_key_of_an_empty_trace_is_empty() {
    let key = Trace::default().key();

    assert!(key.is_empty());
    assert_eq!(key.len(), 0);
    assert_eq!(key.coordinates().len(), 0);
    assert_eq!(key, Trace::new(Vec::new()).key());
}

// ---------------------------------------------------------------------------
// Wire form
// ---------------------------------------------------------------------------

/// Test a key is written as its steps' kind, decision, domain and
/// coordinate, the decision `null` for a dynamic step, and neither the
/// subject nor the value.
#[test]
fn trace_key_serializes_to_the_pinned_shape() {
    let tiling = build_tiling_space();
    let (trace, _) = record_tiling_run(&tiling, 17, &Identifier::new("buffer"));

    let written = json_of(&trace.key());

    let expected = json!({"steps": [
        {
            "kind": "search_space.variable",
            "decision": 0,
            "domain": {"choice": [{"value": {"int": "1"}}, {"value": {"int": "2"}}]},
            "coordinate": {"index": 0},
        },
        {
            "kind": "search_space.choice",
            "decision": 1,
            "domain": {"choice": [{"bound": 3}, {"bound": 5}]},
            "coordinate": {"index": 0},
        },
        {
            "kind": "moga.cir.address",
            "decision": null,
            "domain": {"strided": [{"start": "0", "stop": "64", "stride": "1"}]},
            "coordinate": {"index": 17},
        },
        {
            "kind": "search_space.variable",
            "decision": 2,
            "domain": {"choice": [
                {"value": {"int": "1"}}, {"value": {"int": "2"}}, {"value": {"int": "3"}},
            ]},
            "coordinate": {"index": 1},
        },
    ]});
    assert_eq!(written, expected);
}

/// Test a key is its trace's JSON without each step's subject and value.
#[test]
fn trace_key_is_the_traces_json_without_subjects_and_values() {
    let tiling = build_tiling_space();
    let (trace, _) = record_tiling_run(&tiling, 40, &Identifier::new("buffer"));

    let mut expected = json_of(&trace);
    for step in expected["steps"].as_array_mut().expect("steps") {
        let step = step.as_object_mut().expect("a step");
        step.remove("subject");
        step.remove("value");
    }

    assert_eq!(json_of(&trace.key()), expected);
}

/// Test the key of an empty trace is written as no steps.
#[test]
fn trace_key_of_an_empty_trace_serializes_as_no_steps() {
    assert_eq!(json_of(&Trace::default().key()), json!({"steps": []}));
}

/// Test a key, of static and dynamic steps, round-trips through JSON and
/// postcard.
#[test]
fn trace_key_round_trips_through_json_and_postcard() {
    let tiling = build_tiling_space();
    let (trace, _) = record_tiling_run(&tiling, 63, &Identifier::new("buffer"));

    let result = check_serde_round_trip(&trace.key());

    result.unwrap_or_else(|failure| panic!("{failure}"));
}

/// Test the key of an order step and of an empty trace round-trip through
/// JSON and postcard.
#[rstest]
#[case::empty(Trace::default())]
#[case::order(build_dynamic_trace("walk", &order_of(&[&Identifier::new("i"), &Identifier::new("j")]), order(&[1, 0])))]
fn trace_key_of_other_runs_round_trips_through_json_and_postcard(#[case] trace: Trace) {
    let result = check_serde_round_trip(&trace.key());

    result.unwrap_or_else(|failure| panic!("{failure}"));
}

/// Return the JSON text of a key of one dynamic step of `kind_name` over
/// the one run `[start, stop)`, answered with `coordinate`.
fn build_key_text(
    kind_name: &str,
    start: i64,
    stop: i64,
    coordinate: &serde_json::Value,
) -> String {
    json!({"steps": [{
        "kind": kind_name,
        "decision": null,
        "domain": {"strided": [{
            "start": start.to_string(),
            "stop": stop.to_string(),
            "stride": "1",
        }]},
        "coordinate": coordinate,
    }]})
    .to_string()
}

/// Test a well-formed key decodes: the control of the refusals below.
#[test]
fn trace_key_decoding_reads_a_well_formed_step() {
    let text = build_key_text("k", 0, 4, &json!({"index": 3}));

    let key: TraceKey = serde_json::from_str(&text).expect("a valid key");

    assert_eq!(key.len(), 1);
    assert_eq!(key.coordinates().collect::<Vec<_>>(), [&index(3)]);
}

/// Test reading refuses a coordinate its signature does not contain, an
/// untagged coordinate and an empty kind.
#[rstest]
#[case::past_the_end(build_key_text("k", 0, 4, &json!({"index": 4})))]
#[case::wrong_shape(build_key_text("k", 0, 4, &json!({"order": [0]})))]
#[case::untagged(build_key_text("k", 0, 4, &json!(3)))]
#[case::empty_kind(build_key_text("", 0, 4, &json!({"index": 0})))]
fn trace_key_decoding_refuses_a_malformed_step(#[case] text: String) {
    let result = serde_json::from_str::<TraceKey>(&text);

    result.expect_err("a malformed key is refused");
}
