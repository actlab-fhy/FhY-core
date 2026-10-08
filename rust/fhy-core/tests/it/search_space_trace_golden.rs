//! Replay of the search-stream corpus: each case, recorded from the MOGA-VM
//! oracle with the port's answers beside the oracle's, is run through the
//! Rust port, which must answer as the corpus's port answers say: the
//! domains it builds or refuses and their coordinates and admission, the
//! replays it accepts or refuses, where a replayed stream is refused, and
//! the cardinality of extracted spaces.
//!
//! The corpus is recorded by
//! `rust/fhy-core/tests/golden/record_trace_cases.py` and committed at
//! `rust/fhy-core/tests/golden/trace_cases.json`. A case's port answers
//! agree with the oracle's unless it names the divergence that changes
//! them.

use fhy_core::constraint::Value;
use fhy_core::expression::BigInt;
use fhy_core::identifier::Identifier;
use fhy_core::search_space::{
    Cardinality, Choice, ChoiceDomain, Coordinate, OrderDomain, Recorder, ReplayError,
    ReplayOracle, StepDomain, StepDomainError, StridedDomain, StridedRun, Trace, TraceError,
    TraceStep,
};
use num_bigint::BigUint;
use serde_json::Value as Json;

use crate::support::constraint::{int, text};
use crate::support::search::{kind, permutation_param, with_context};
use crate::support::search_space::{
    bare_alternative, choice_of, int_variable, plain_alternative, plain_variable, space_of,
};

const GOLDEN_JSON: &str = include_str!("../golden/trace_cases.json");

/// Return the parsed corpus.
fn read_corpus() -> Json {
    serde_json::from_str(GOLDEN_JSON).expect("the corpus is JSON")
}

/// Return the cases of the family `family`.
fn read_family(family: &str) -> Vec<Json> {
    read_corpus()[family]
        .as_array()
        .unwrap_or_else(|| panic!("the corpus has no {family}"))
        .clone()
}

/// Return the integer `json` holds.
fn read_u64(json: &Json) -> u64 {
    json.as_u64()
        .unwrap_or_else(|| panic!("expected an integer, got {json}"))
}

/// Return the value of the member `json`, a label naming `labels`' entry.
fn read_value(json: &Json, labels: &[Identifier]) -> Value {
    let (kind, payload) = json
        .as_object()
        .and_then(|object| object.iter().next())
        .unwrap_or_else(|| panic!("expected a member, got {json}"));
    match kind.as_str() {
        "int" => int(payload.as_i64().expect("an integer")),
        "str" => text(payload.as_str().expect("a string")),
        "bool" => Value::Bool(payload.as_bool().expect("a Boolean")),
        "float" => Value::Float(payload.as_f64().expect("a float")),
        "label" => {
            Value::Identifier(labels[usize::try_from(read_u64(payload)).expect("small")].clone())
        }
        other => panic!("unknown member kind {other}"),
    }
}

/// Return the value of a probe of a strided domain: an integer, or a
/// Boolean.
fn read_probe(json: &Json) -> Value {
    json.as_bool().map_or_else(
        || int(json.as_i64().expect("an integer probe")),
        Value::Bool,
    )
}

/// Return the strided runs `json` describes as `[start, stop, step]`.
fn read_runs(json: &Json) -> Vec<StridedRun> {
    json.as_array()
        .expect("runs")
        .iter()
        .map(|run| {
            let [start, stop, stride] = [0, 1, 2].map(|at| run[at].as_i64().expect("an integer"));
            StridedRun::new(
                BigInt::from(start),
                BigInt::from(stop),
                BigUint::from(u64::try_from(stride).expect("a positive stride")),
            )
            .expect("a recorded run is valid")
        })
        .collect()
}

/// Return the domain `spec` describes, its labels fresh identifiers.
fn build_domain(spec: &Json) -> StepDomain {
    let labels: Vec<Identifier> = (0..8).map(|_| Identifier::new("label")).collect();
    if spec["shape"] == "choice" {
        let values = spec["values"]
            .as_array()
            .expect("values")
            .iter()
            .map(|value| read_value(value, &labels))
            .collect();
        StepDomain::from(ChoiceDomain::new(values).expect("a recorded replay domain is valid"))
    } else {
        StepDomain::from(StridedDomain::new(read_runs(&spec["runs"])).expect("valid runs"))
    }
}

/// Return the error name the corpus gives `error`.
fn name_domain_error(error: &StepDomainError) -> &'static str {
    match error {
        StepDomainError::EmptyChoice => "empty_choice",
        StepDomainError::RepeatedValue { .. } => "repeated_value",
        StepDomainError::EmptyOrder => "empty_order",
        _ => "other",
    }
}

/// Test every case without a divergence answers as the oracle does, so the
/// corpus's port answers are the oracle's wherever no divergence explains
/// them.
#[test]
fn every_untagged_case_agrees_with_the_oracle() {
    let corpus = read_corpus();

    let disagreeing: Vec<String> = ["domains", "replays", "streams", "extractions"]
        .iter()
        .flat_map(|family| corpus[*family].as_array().expect("a family").iter())
        .filter(|case| case["divergence"].is_null() && case["oracle"] != case["port"])
        .map(|case| case["name"].to_string())
        .collect();

    assert!(disagreeing.is_empty(), "{disagreeing:?}");
}

/// Test the corpus holds every family and the cases its divergences tag.
#[test]
fn the_corpus_holds_every_family_and_divergence() {
    let corpus = read_corpus();

    let tags: Vec<&str> = ["domains", "replays", "streams"]
        .iter()
        .flat_map(|family| corpus[*family].as_array().expect("a family").iter())
        .filter_map(|case| case["divergence"].as_str())
        .collect();

    for family in ["domains", "replays", "streams", "extractions"] {
        assert!(
            corpus[family]
                .as_array()
                .is_some_and(|cases| cases.len() >= 5),
            "{family}"
        );
    }
    for tag in [
        "replay-compares-signatures",
        "finish-refuses-shorter-stream",
        "type-strict-distinct-choices",
        "empty-order-domain-refused",
    ] {
        assert!(tags.contains(&tag), "no case of {tag}");
    }
}

/// Test every domain case builds or is refused, and answers its
/// coordinates and admission, as recorded.
#[test]
fn every_domain_case_replays_as_recorded() {
    for case in read_family("domains") {
        match case["shape"].as_str().expect("a shape") {
            "choice" => replay_choice_case(&case),
            "order" => replay_order_case(&case),
            "strided" => replay_strided_case(&case),
            other => panic!("{}: unknown shape {other}", case["name"]),
        }
    }
}

/// Check the choice domain case `case` against its recorded answers.
fn replay_choice_case(case: &Json) {
    let name = &case["name"];
    let port = &case["port"];
    let values: Vec<Value> = case["spec"]["values"]
        .as_array()
        .expect("values")
        .iter()
        .map(|value| read_value(value, &[]))
        .collect();
    let built = ChoiceDomain::new(values.clone());
    if port["outcome"] == "refused" {
        let error = built.expect_err("the case is refused");
        assert_eq!(name_domain_error(&error), port["error"], "{name}");
        return;
    }
    let domain = built.unwrap_or_else(|error| panic!("{name}: {error}"));
    assert_eq!(
        domain.cardinality(),
        read_u64(&port["cardinality"]),
        "{name}"
    );
    let coordinates: Vec<Json> = (0..domain.cardinality())
        .map(|index| {
            let value = domain.value_at(index).expect("in range");
            Json::from(domain.coordinate_of(value).expect("its own value"))
        })
        .collect();
    assert_eq!(Json::from(coordinates), port["coordinates"], "{name}");
    let admits: Vec<Json> = case["spec"]["probes"]
        .as_array()
        .expect("probes")
        .iter()
        .map(|probe| Json::from(domain.admits(&read_value(probe, &[]))))
        .collect();
    assert_eq!(Json::from(admits), port["admits"], "{name}");
}

/// Check the order domain case `case` against its recorded answers.
fn replay_order_case(case: &Json) {
    let name = &case["name"];
    let port = &case["port"];
    let size = read_u64(&case["spec"]["size"]);
    let elements: Vec<Value> = (0..size)
        .map(|_| Value::Identifier(Identifier::new("element")))
        .collect();
    let built = OrderDomain::new(elements);
    if port["outcome"] == "refused" {
        let error = built.expect_err("the case is refused");
        assert_eq!(name_domain_error(&error), port["error"], "{name}");
        return;
    }
    let domain = built.unwrap_or_else(|error| panic!("{name}: {error}"));
    assert_eq!(
        domain.cardinality(),
        BigUint::from(read_u64(&port["cardinality"])),
        "{name}"
    );
    for (positions, ordering) in case["spec"]["permutations"]
        .as_array()
        .expect("permutations")
        .iter()
        .zip(port["orderings"].as_array().expect("orderings"))
    {
        let positions: Vec<u32> = positions
            .as_array()
            .expect("positions")
            .iter()
            .map(|position| u32::try_from(read_u64(position)).expect("small"))
            .collect();
        let value = domain.value_at(&positions).expect("a permutation");
        let coordinate = domain.coordinate_of(&value).expect("an ordering");
        let found: Vec<Json> = coordinate
            .iter()
            .map(|&position| Json::from(position))
            .collect();
        assert_eq!(&Json::from(found), ordering, "{name}");
    }
}

/// Check the strided domain case `case` against its recorded answers.
fn replay_strided_case(case: &Json) {
    let name = &case["name"];
    let port = &case["port"];
    let domain = StridedDomain::new(read_runs(&case["spec"]["runs"]))
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    assert_eq!(
        domain.cardinality(),
        read_u64(&port["cardinality"]),
        "{name}"
    );
    let values: Vec<Json> = (0..domain.cardinality())
        .map(|index| {
            let value = domain.value_at(index).expect("in range");
            Json::from(i64::try_from(value).expect("a small address"))
        })
        .collect();
    assert_eq!(Json::from(values), port["values"], "{name}");
    let admits: Vec<Json> = case["spec"]["probes"]
        .as_array()
        .expect("probes")
        .iter()
        .map(|probe| Json::from(domain.admits(&read_probe(probe))))
        .collect();
    assert_eq!(Json::from(admits), port["admits"], "{name}");
}

/// Test every replay case replays or is refused as recorded.
#[test]
fn every_replay_case_replays_as_recorded() {
    for case in read_family("replays") {
        let name = &case["name"];
        let recorded_domain = build_domain(&case["recorded"]);
        let offered = build_domain(&case["offered"]);
        let coordinate = Coordinate::Index(read_u64(&case["coordinate"]));
        let step = TraceStep::dynamic(
            kind("moga.cir.option"),
            Identifier::new("s"),
            &recorded_domain,
            coordinate,
        )
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        let mut replay = ReplayOracle::new(Trace::new(vec![step]));
        let mut recorder = Recorder::new();

        let result = with_context(|context| {
            recorder.decide_dynamic(
                &kind("moga.cir.option"),
                &Identifier::new("s"),
                &offered,
                &mut replay,
                context,
            )
        });

        if case["port"]["outcome"] == "refused" {
            let Err(TraceError::Oracle { source, .. }) = result else {
                panic!("{name}: expected a refused replay, got {result:?}");
            };
            assert!(
                matches!(
                    source.downcast_ref::<ReplayError>(),
                    Some(ReplayError::DomainMismatch { position: 0 })
                ),
                "{name}: {source}"
            );
        } else {
            let answer = result.unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(
                answer,
                Coordinate::Index(read_u64(&case["port"]["coordinate"])),
                "{name}"
            );
        }
    }
}

/// Test every stream case is refused where recorded, and refused at
/// `finish` where it asked fewer steps than recorded.
#[test]
fn every_stream_case_replays_as_recorded() {
    for case in read_family("streams") {
        let name = &case["name"];
        let domain =
            StepDomain::from(ChoiceDomain::new(vec![int(0), int(1), int(2)]).expect("valid"));
        let steps: Vec<TraceStep> = (0..read_u64(&case["recorded"]))
            .map(|index| {
                TraceStep::dynamic(
                    kind("moga.cir.address"),
                    Identifier::new("s"),
                    &domain,
                    Coordinate::Index(index % 3),
                )
                .expect("in range")
            })
            .collect();
        let mut replay = ReplayOracle::new(Trace::new(steps));
        let mut recorder = Recorder::new();
        let mut refused_at = Json::Null;

        for position in 0..read_u64(&case["asked"]) {
            let result = with_context(|context| {
                recorder.decide_dynamic(
                    &kind("moga.cir.address"),
                    &Identifier::new("s"),
                    &domain,
                    &mut replay,
                    context,
                )
            });
            if let Err(error) = result {
                let TraceError::Oracle { source, .. } = &error else {
                    panic!("{name}: expected a replay refusal, got {error}");
                };
                assert!(
                    matches!(
                        source.downcast_ref::<ReplayError>(),
                        Some(ReplayError::Exhausted { .. })
                    ),
                    "{name}: {source}"
                );
                refused_at = Json::from(position);
                break;
            }
        }
        let unconsumed_at = if refused_at.is_null() {
            match replay.finish() {
                Ok(()) => Json::Null,
                Err(ReplayError::Unconsumed { position }) => Json::from(position),
                Err(error) => panic!("{name}: {error}"),
            }
        } else {
            Json::Null
        };

        assert_eq!(refused_at, case["port"]["refused_at"], "{name}");
        assert_eq!(
            unconsumed_at, case["port"]["unconsumed_at_finish"],
            "{name}"
        );
    }
}

/// Test every extracted space's port, a space of one choice per entry whose
/// alternatives hold its option's axes as variables, has exactly the
/// cardinality recorded.
#[test]
fn every_extraction_case_counts_as_recorded() {
    for case in read_family("extractions") {
        let name = &case["name"];
        let choices: Vec<Choice> = case["entries"]
            .as_array()
            .expect("entries")
            .iter()
            .map(|options| {
                let alternatives = options
                    .as_array()
                    .expect("options")
                    .iter()
                    .map(|axes| {
                        let variables: Vec<_> = axes
                            .as_array()
                            .expect("axes")
                            .iter()
                            .map(|axis| {
                                if let Some(size) = axis.get("choice") {
                                    let values: Vec<i64> = (0..i64::try_from(read_u64(size))
                                        .expect("small"))
                                        .collect();
                                    int_variable(&Identifier::new("tile"), &values)
                                } else {
                                    let levels: Vec<Identifier> = (0..read_u64(&axis["order"]))
                                        .map(|_| Identifier::new("level"))
                                        .collect();
                                    plain_variable(
                                        &Identifier::new("walk"),
                                        permutation_param(&levels.iter().collect::<Vec<_>>()),
                                    )
                                }
                            })
                            .collect();
                        if variables.is_empty() {
                            bare_alternative(&Identifier::new("option"))
                        } else {
                            plain_alternative(&Identifier::new("option"), variables, Vec::new())
                        }
                    })
                    .collect();
                choice_of(&Identifier::new("entry"), alternatives)
            })
            .collect();
        let space = space_of(&Identifier::new("extracted"), Vec::new(), choices);

        let cardinality = with_context(|context| space.cardinality(context, 1_000_000))
            .unwrap_or_else(|error| panic!("{name}: {error}"));

        assert_eq!(
            cardinality,
            Cardinality::Exact(BigUint::from(read_u64(&case["port"]["cardinality"]))),
            "{name}"
        );
    }
}
