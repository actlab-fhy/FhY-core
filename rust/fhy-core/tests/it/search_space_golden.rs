//! Replay of the search-space corpus: each case's decision points, recorded
//! from the MOGA-VM oracle with the port's answers beside the oracle's,
//! built as spaces and configurations through the Rust port, which must
//! answer as the corpus says: what it builds or refuses, the structural and
//! alpha verdicts in both directions, and the V2 texts of what it builds,
//! byte for byte.
//!
//! The corpus is recorded by
//! `rust/fhy-core/tests/golden/record_search_space_cases.py` and committed
//! at `rust/fhy-core/tests/golden/search_space_cases.json`. A case's answers
//! agree with the oracle's unless it names the divergence that changes
//! them.

use std::collections::HashMap;

use fhy_core::constraint::{Constraint, Polarity, SetConstraint, Value};
use fhy_core::foreign::Part;
use fhy_core::identifier::Identifier;
use fhy_core::param::{CategoricalDomain, Param, ParamContext, ParamDomain};
use fhy_core::search_space::{
    Alternative, Choice, Configuration, PlainAlternative, PlainVariable, Space, Variable,
};
use fhy_core::solver::Solver;
use fhy_core::term::AlphaEquivalence;
use serde_json::Value as Json;

use crate::support::constraint::{int, member_set};
use crate::support::search_space::ground_solver;
use crate::support::serde::restored;

const GOLDEN_JSON: &str = include_str!("../golden/search_space_cases.json");

/// The fewest cases the committed corpus holds.
const MIN_CASES: usize = 100;

/// The probes the committed corpus must hold, the audit's counterexamples
/// among them.
const PROBES: [&str; 8] = [
    "a1_one_name_for_two_alternatives",
    "a1b_one_knob_twice_in_an_option",
    "a2_one_knob_in_two_alternatives",
    "a5_constraint_of_a_categorical_knob",
    "a6_one_and_true_as_categories",
    "a7_free_category_and_bound_name",
    "v3_one_knob_assigned_twice",
    "v6_no_alternative",
];

/// What the port makes of one decision point.
enum Built {
    /// The configuration of the point's selection, in its space.
    Configuration(Configuration),
    /// The space is refused.
    SpaceRefused,
    /// The space is built and the configuration refused.
    ConfigurationRefused,
}

impl Built {
    /// Return the outcome's name in the corpus.
    fn outcome(&self) -> &'static str {
        match self {
            Self::Configuration(_) => "built",
            Self::SpaceRefused => "space_refused",
            Self::ConfigurationRefused => "configuration_refused",
        }
    }
}

/// Return the integer `json` holds.
fn index(json: &Json) -> usize {
    usize::try_from(json.as_u64().expect("an index")).expect("a small index")
}

/// Return the value of the member `json` with `pool`'s identifiers.
fn member_value(json: &Json, pool: &[Identifier]) -> Value {
    if let Some(label) = json.get("label") {
        return Value::Identifier(pool[index(label)].clone());
    }
    if let Some(value) = json.get("bool") {
        return Value::Bool(value.as_bool().expect("a Boolean"));
    }
    int(json["int"].as_i64().expect("an integer"))
}

/// Return the values of the members `json` lists.
fn member_values(json: &Json, pool: &[Identifier]) -> Vec<Value> {
    json.as_array()
        .expect("a list of members")
        .iter()
        .map(|member| member_value(member, pool))
        .collect()
}

/// Return the param of the knob `json`, whose variable is its param's
/// label.
fn build_param(json: &Json, pool: &[Identifier]) -> Param {
    let variable = pool[index(&json["param"])].clone();
    let constraints: Vec<Constraint> = json["kept"]
        .as_array()
        .map(|_| {
            Constraint::from(SetConstraint::new(
                variable.clone(),
                member_set(member_values(&json["kept"], pool)),
                Polarity::In,
            ))
        })
        .into_iter()
        .collect();
    let solver = Solver::new();
    Param::new(
        ParamDomain::from(
            CategoricalDomain::new(member_values(&json["categories"], pool))
                .expect("the categories are valid"),
        ),
        variable,
        constraints,
        &ParamContext::new(&solver),
    )
    .expect("the param is valid")
}

/// Return what the port makes of the point `json`.
fn build_point(json: &Json, pool: &[Identifier]) -> Built {
    let name = |key: &str| pool[index(&json[key])].clone();
    let mut alternatives: Vec<Part<dyn Alternative>> = Vec::new();
    for option in json["options"].as_array().expect("a list of options") {
        let variables: Vec<Part<dyn Variable>> = option["knobs"]
            .as_array()
            .expect("a list of knobs")
            .iter()
            .map(|knob| {
                Part::new(PlainVariable::new(
                    pool[index(&knob["name"])].clone(),
                    build_param(knob, pool),
                ))
            })
            .collect();
        let Ok(alternative) =
            PlainAlternative::new(pool[index(&option["name"])].clone(), variables, Vec::new())
        else {
            return Built::SpaceRefused;
        };
        alternatives.push(Part::new(alternative));
    }
    let Ok(choice) = Choice::new(name("choice"), alternatives) else {
        return Built::SpaceRefused;
    };
    let Ok(space) = Space::new(
        name("space"),
        Vec::new(),
        vec![choice],
        Vec::new(),
        Vec::new(),
    ) else {
        return Built::SpaceRefused;
    };
    let mut entries: Vec<(Identifier, Value)> = Vec::new();
    if let Some(selection) = json["selection"].as_object() {
        entries.push((
            name("choice"),
            Value::Identifier(pool[index(&selection["option"])].clone()),
        ));
        for pair in selection["values"].as_array().expect("a list of values") {
            entries.push((pool[index(&pair[0])].clone(), member_value(&pair[1], pool)));
        }
    }
    let solver = ground_solver();
    match Configuration::new(&space, entries, &ParamContext::new(&solver)) {
        Ok(configuration) => Built::Configuration(configuration),
        Err(_) => Built::ConfigurationRefused,
    }
}

/// Return whether `left` and `right` are related by `relation`, in each
/// direction.
fn both_ways(
    left: &Configuration,
    right: &Configuration,
    relation: impl Fn(&Configuration, &Configuration) -> bool,
) -> Json {
    Json::from(vec![relation(left, right), relation(right, left)])
}

/// Return the outcome the oracle's answer `outcome` stands for when the
/// port's is compared with it: built or refused.
fn as_built_or_refused(outcome: &Json) -> &str {
    match outcome.as_str().expect("an outcome") {
        "built" => "built",
        _ => "refused",
    }
}

/// Replay one case, returning what it found wrong.
fn replay_case(case: &Json) -> Vec<String> {
    let name = case["name"].as_str().expect("a name");
    let base = case["id_base"].as_u64().expect("an id base");
    let pool: Vec<Identifier> = case["labels"]
        .as_array()
        .expect("a list of labels")
        .iter()
        .zip(base..)
        .map(|(label, id)| restored(id, label.as_str().expect("a label")))
        .collect();
    let (oracle, port, wire) = (&case["oracle"], &case["port"], &case["wire"]);
    let mut problems = Vec::new();

    let mut built = HashMap::new();
    for side in ["left", "right"] {
        if case[side].is_null() {
            continue;
        }
        let point = build_point(&case[side], &pool);
        if port[side] != point.outcome() {
            problems.push(format!(
                "{name}: the {side} point is {}, not {}",
                point.outcome(),
                port[side]
            ));
        }
        if let Built::Configuration(configuration) = &point {
            for (key, text) in [
                (
                    format!("{side}_space"),
                    serde_json::to_string(configuration.space()),
                ),
                (
                    format!("{side}_configuration"),
                    serde_json::to_string(configuration),
                ),
            ] {
                let text = text.expect("a built point serializes");
                if wire[&key] != text.as_str() {
                    problems.push(format!("{name}: {key} writes\n{text}\nnot\n{}", wire[&key]));
                }
            }
            let decoded: Configuration = serde_json::from_str(
                wire[format!("{side}_configuration")]
                    .as_str()
                    .unwrap_or("null"),
            )
            .expect("the golden configuration decodes");
            if &decoded != configuration {
                problems.push(format!(
                    "{name}: the {side} configuration decodes differently"
                ));
            }
        }
        built.insert(side, point);
    }

    if let (Some(Built::Configuration(left)), Some(Built::Configuration(right))) =
        (built.get("left"), built.get("right"))
    {
        let structural = both_ways(left, right, |left, right| {
            left.is_structurally_equivalent(right)
                .expect("plain parts compare")
        });
        let alpha = both_ways(left, right, |left, right| {
            left.is_alpha_equivalent(right)
                .expect("plain parts compare")
        });
        if port["structural"] != structural || port["alpha"] != alpha {
            problems.push(format!(
                "{name}: structural {structural} and alpha {alpha}, not {} and {}",
                port["structural"], port["alpha"]
            ));
        }
    }

    if case["divergence"].is_null() {
        let sides_agree = ["left", "right"].into_iter().all(|side| {
            case[side].is_null()
                || as_built_or_refused(&oracle[side]) == as_built_or_refused(&port[side])
        });
        if !sides_agree
            || oracle.get("structural") != port.get("structural")
            || oracle.get("alpha") != port.get("alpha")
        {
            problems.push(format!(
                "{name}: names no divergence, but the oracle answers {oracle} and the port {port}"
            ));
        }
    }
    problems
}

/// Replay every case of `document`, returning how many it holds.
fn replay_document(document: &Json) -> usize {
    let cases = document["cases"].as_array().expect("a list of cases");
    let problems: Vec<String> = cases.iter().flat_map(replay_case).collect();
    assert!(problems.is_empty(), "{}", problems.join("\n\n"));
    cases.len()
}

#[test]
fn every_committed_case_replays_as_recorded() {
    let document: Json = serde_json::from_str(GOLDEN_JSON).expect("the corpus is JSON");

    let count = replay_document(&document);

    assert!(count >= MIN_CASES, "only {count} cases");
    let names: Vec<&str> = document["cases"]
        .as_array()
        .expect("a list of cases")
        .iter()
        .map(|case| case["name"].as_str().expect("a name"))
        .collect();
    for probe in PROBES {
        assert!(names.contains(&probe), "the corpus lacks the probe {probe}");
    }
}

#[test]
#[ignore = "requires an expanded corpus recorded from the MOGA-VM oracle"]
fn every_expanded_case_replays_as_recorded() {
    let path = std::env::var("FHY_SEARCH_SPACE_CORPUS")
        .expect("FHY_SEARCH_SPACE_CORPUS names the expanded corpus");
    let text = std::fs::read_to_string(&path).expect("the expanded corpus reads");
    let document: Json = serde_json::from_str(&text).expect("the corpus is JSON");

    let count = replay_document(&document);

    assert!(count > 0, "the expanded corpus holds no case");
}
