//! Tests for `GuidedOracle`: which steps a guide answers (a static step by
//! canonical position, a dynamic step first-in-first-out per kind), which
//! go to the fallback (an inadmissible or mismatched guide step, a step
//! the guide lacks), and that the fallback's error stops the run as it
//! would unguided.

use fhy_core::constraint::Value;
use fhy_core::identifier::Identifier;
use fhy_core::search_space::{
    Configuration, Coordinate, GuidedOracle, RandomOracle, Recorded, Recorder, SearchOracle, Space,
    StepDomain, Trace, TraceError,
};
use rstest::rstest;

use crate::support::constraint::int;
use crate::support::param::in_set;
use crate::support::search::{
    OracleRefusal, RefusingOracle, ScriptedOracle, build_tiling_space, index, kind, strided,
    tiling_configurations, tiling_entries, with_context,
};
use crate::support::search_space::{configure, forbidden, int_variable};

/// Return the space of the variables `names`, each over `domain`, with the
/// clause forbidding `forbidden_values` of the variable at `forbidden_at`,
/// if given.
fn build_flat_space(
    names: &[Identifier],
    domain: &[i64],
    forbidden_at: Option<(usize, &[i64])>,
) -> Space {
    let clauses = forbidden_at
        .into_iter()
        .map(|(at, values)| forbidden([in_set(&names[at], values.iter().copied().map(int))]))
        .collect();
    Space::new(
        Identifier::new("flat"),
        names
            .iter()
            .map(|name| int_variable(name, domain))
            .collect(),
        Vec::new(),
        Vec::new(),
        clauses,
    )
    .expect("the flat space is valid")
}

/// Return the names of three fresh variables.
fn build_names() -> [Identifier; 3] {
    ["x", "y", "z"].map(Identifier::new)
}

/// Return the trace of the complete configuration of `space` giving `names`
/// the values `values`.
fn record_flat_guide(space: &Space, names: &[Identifier], values: &[i64]) -> Trace {
    let configuration = configure(
        space,
        names.iter().cloned().zip(values.iter().copied().map(int)),
    );
    with_context(|context| configuration.trace(context)).expect("the configuration has a trace")
}

/// Return the run of `space` answered by `oracle`.
fn sample_with(space: &Space, oracle: &mut GuidedOracle<&mut ScriptedOracle>) -> Recorded {
    with_context(|context| space.sample(oracle, context)).expect("the run completes")
}

/// Return the value `configuration` gives `name`.
fn value_of(configuration: &Configuration, name: &Identifier) -> Value {
    configuration
        .value(name)
        .cloned()
        .expect("the decision is assigned")
}

/// Return the dynamic steps `asks` (a kind and its domain) answered by
/// `oracle`, in order.
fn ask_dynamic(oracle: &mut dyn SearchOracle, asks: &[(&str, StepDomain)]) -> Vec<Coordinate> {
    let subject = Identifier::new("s");
    let mut recorder = Recorder::new();
    with_context(|context| {
        asks.iter()
            .map(|(name, domain)| {
                recorder.decide_dynamic(&kind(name), &subject, domain, oracle, context)
            })
            .collect::<Result<Vec<_>, _>>()
    })
    .expect("the run completes")
}

/// Return the trace of dynamic steps `steps` (a kind and its domain)
/// answered by `answers`.
fn record_dynamic_guide(steps: &[(&str, StepDomain)], answers: &[u64]) -> Trace {
    let subject = Identifier::new("s");
    let mut recorder = Recorder::new();
    let mut oracle = ScriptedOracle::new(answers.iter().copied().map(index));
    with_context(|context| {
        for (name, domain) in steps {
            recorder
                .decide_dynamic(&kind(name), &subject, domain, &mut oracle, context)
                .expect("the script answers inside the domain");
        }
    });
    recorder.trace()
}

/// Return the position the oracle failure `result` stopped at, naming
/// `OracleRefusal` as its source.
///
/// # Panics
///
/// Panics if the run did not stop as a refusal.
fn refusal_position(result: Result<Recorded, TraceError>) -> usize {
    match result {
        Err(TraceError::Oracle { position, source }) => {
            assert!(
                source.downcast_ref::<OracleRefusal>().is_some(),
                "expected the refusal, got {source:?}"
            );
            position
        }
        other => panic!("expected an oracle failure, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Static steps
// ---------------------------------------------------------------------------

/// Test the trace of a complete configuration guides a run back to it
/// without asking the fallback.
#[rstest]
#[case::inner_value(0)]
#[case::last_inner_value(2)]
#[case::flat_alternative(3)]
#[case::inactive_inner_variable(4)]
fn guided_oracle_guides_a_run_back_to_the_configuration_of_its_trace(#[case] which: usize) {
    let tiling = build_tiling_space();
    let original = configure(
        &tiling.space,
        tiling_entries(&tiling, &tiling_configurations(&tiling)[which]),
    );
    let guide = with_context(|context| original.trace(context)).expect("a trace");
    let mut fallback = ScriptedOracle::new([]);

    let run = sample_with(&tiling.space, &mut GuidedOracle::new(&guide, &mut fallback));

    let configuration = run.configuration().expect("a run over a space");
    assert_eq!(configuration.key(), original.key());
    assert_eq!(
        configuration.entries().collect::<Vec<_>>(),
        original.entries().collect::<Vec<_>>()
    );
    assert_eq!(fallback.seen, Vec::new());
}

/// Test a guide that answers every step never reaches a fallback that
/// refuses.
#[test]
fn guided_oracle_does_not_reach_a_refusing_fallback_when_the_guide_answers_every_step() {
    let tiling = build_tiling_space();
    let original = configure(
        &tiling.space,
        tiling_entries(&tiling, &tiling_configurations(&tiling)[1]),
    );
    let guide = with_context(|context| original.trace(context)).expect("a trace");

    let run = with_context(|context| {
        tiling
            .space
            .sample(&mut GuidedOracle::new(&guide, RefusingOracle), context)
    })
    .expect("the guide answers every step");

    let configuration = run.configuration().expect("a run over a space");
    assert_eq!(configuration.key(), original.key());
}

/// Test a trace over one space guides a run over a separately built,
/// alpha-equivalent space by canonical position, not by name.
#[test]
fn guided_oracle_guides_a_run_over_a_renamed_space_by_position() {
    let first = build_tiling_space();
    let second = build_tiling_space();
    let original = configure(
        &first.space,
        tiling_entries(&first, &tiling_configurations(&first)[2]),
    );
    let guide = with_context(|context| original.trace(context)).expect("a trace");
    let mut fallback = ScriptedOracle::new([]);

    let run = sample_with(&second.space, &mut GuidedOracle::new(&guide, &mut fallback));

    let expected = configure(
        &second.space,
        tiling_entries(&second, &tiling_configurations(&second)[2]),
    );
    let configuration = run.configuration().expect("a run over a space");
    assert_eq!(configuration.key(), expected.key());
    assert_eq!(configuration.key(), original.key());
    assert_eq!(fallback.seen, Vec::new());
}

/// Test a guide step whose value the run's space forbids goes to the
/// fallback, and every other step is answered by the guide.
#[test]
fn guided_oracle_asks_the_fallback_for_an_inadmissible_guide_step() {
    let names = build_names();
    let guide_space = build_flat_space(&names, &[1, 2, 3], None);
    let run_names = build_names();
    let run_space = build_flat_space(&run_names, &[1, 2, 3], Some((1, &[2])));
    let guide = record_flat_guide(&guide_space, &names, &[3, 2, 1]);
    let mut fallback = ScriptedOracle::new([index(0)]);

    let run = sample_with(&run_space, &mut GuidedOracle::new(&guide, &mut fallback));

    let configuration = run.configuration().expect("a run over a space");
    assert_eq!(value_of(configuration, &run_names[0]), int(3));
    assert_eq!(value_of(configuration, &run_names[1]), int(1));
    assert_eq!(value_of(configuration, &run_names[2]), int(1));
    let asked: Vec<_> = fallback
        .seen
        .iter()
        .map(|step| step.decision.clone())
        .collect();
    assert_eq!(asked, vec![Some(run_names[1].clone())]);
}

/// Test a guide step recorded over a different domain goes to the
/// fallback, though its coordinate names an admissible value of the run's
/// domain.
#[rstest]
#[case::smaller_domain(&[1, 2])]
#[case::other_member(&[1, 2, 4])]
fn guided_oracle_asks_the_fallback_for_a_guide_step_over_another_domain(
    #[case] run_domain: &[i64],
) {
    let names = build_names();
    let guide_space = build_flat_space(&names, &[1, 2, 3], None);
    let run_names = build_names();
    let run_space = Space::new(
        Identifier::new("flat"),
        vec![
            int_variable(&run_names[0], &[1, 2, 3]),
            int_variable(&run_names[1], run_domain),
            int_variable(&run_names[2], &[1, 2, 3]),
        ],
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )
    .expect("the run's space is valid");
    let guide = record_flat_guide(&guide_space, &names, &[2, 2, 3]);
    let mut fallback = ScriptedOracle::new([index(0)]);

    let run = sample_with(&run_space, &mut GuidedOracle::new(&guide, &mut fallback));

    let configuration = run.configuration().expect("a run over a space");
    assert_eq!(value_of(configuration, &run_names[0]), int(2));
    assert_eq!(value_of(configuration, &run_names[1]), int(1));
    assert_eq!(value_of(configuration, &run_names[2]), int(3));
    let asked: Vec<_> = fallback
        .seen
        .iter()
        .map(|step| step.decision.clone())
        .collect();
    assert_eq!(asked, vec![Some(run_names[1].clone())]);
}

// ---------------------------------------------------------------------------
// Dynamic steps
// ---------------------------------------------------------------------------

/// Test dynamic steps take the guide's answers first-in-first-out per
/// kind, whatever order the kinds are asked in.
#[test]
fn guided_oracle_answers_dynamic_steps_in_the_guides_order_per_kind() {
    let domain = strided(&[(0, 64)]);
    let guide = record_dynamic_guide(
        &[
            ("k1", domain.clone()),
            ("k2", domain.clone()),
            ("k1", domain.clone()),
        ],
        &[10, 20, 30],
    );
    let mut fallback = ScriptedOracle::new([]);

    let answers = ask_dynamic(
        &mut GuidedOracle::new(&guide, &mut fallback),
        &[
            ("k2", domain.clone()),
            ("k1", domain.clone()),
            ("k1", domain),
        ],
    );

    assert_eq!(answers, vec![index(20), index(10), index(30)]);
    assert_eq!(fallback.seen, Vec::new());
}

/// Test a dynamic step of a kind the guide has run out of goes to the
/// fallback.
#[test]
fn guided_oracle_asks_the_fallback_for_a_dynamic_step_beyond_the_guide() {
    let domain = strided(&[(0, 64)]);
    let guide = record_dynamic_guide(&[("k1", domain.clone()), ("k1", domain.clone())], &[10, 30]);
    let mut fallback = ScriptedOracle::new([index(7)]);

    let answers = ask_dynamic(
        &mut GuidedOracle::new(&guide, &mut fallback),
        &[
            ("k1", domain.clone()),
            ("k1", domain.clone()),
            ("k1", domain),
        ],
    );

    assert_eq!(answers, vec![index(10), index(30), index(7)]);
    let asked: Vec<_> = fallback.seen.iter().map(|step| step.kind.clone()).collect();
    assert_eq!(asked, vec!["k1".to_owned()]);
}

/// Test a dynamic step of a kind the guide never asked goes to the
/// fallback.
#[test]
fn guided_oracle_asks_the_fallback_for_a_dynamic_step_of_an_unguided_kind() {
    let domain = strided(&[(0, 64)]);
    let guide = record_dynamic_guide(&[("k1", domain.clone())], &[10]);
    let mut fallback = ScriptedOracle::new([index(5)]);

    let answers = ask_dynamic(
        &mut GuidedOracle::new(&guide, &mut fallback),
        &[("k2", domain)],
    );

    assert_eq!(answers, vec![index(5)]);
    let asked: Vec<_> = fallback.seen.iter().map(|step| step.kind.clone()).collect();
    assert_eq!(asked, vec!["k2".to_owned()]);
}

/// Test a guide step over another domain goes to the fallback but is
/// still taken, so the next step of its kind takes the guide's next step.
#[test]
fn guided_oracle_consumes_a_dynamic_guide_step_over_another_domain() {
    let wide = strided(&[(0, 64)]);
    let narrow = strided(&[(0, 32)]);
    let guide = record_dynamic_guide(&[("k1", wide.clone()), ("k1", wide.clone())], &[10, 30]);
    let mut fallback = ScriptedOracle::new([index(5)]);

    let answers = ask_dynamic(
        &mut GuidedOracle::new(&guide, &mut fallback),
        &[("k1", narrow), ("k1", wide)],
    );

    assert_eq!(answers, vec![index(5), index(30)]);
    assert_eq!(fallback.seen.len(), 1);
    assert_eq!(fallback.seen[0].cardinality, 32u32.into());
}

// ---------------------------------------------------------------------------
// The fallback
// ---------------------------------------------------------------------------

/// Test an empty guide leaves every step to the fallback: the run is the
/// fallback's alone.
#[rstest]
#[case::zero(0)]
#[case::one(1)]
#[case::wide(0xDEAD_BEEF)]
fn guided_oracle_with_an_empty_guide_runs_as_its_fallback_alone(#[case] seed: u64) {
    let tiling = build_tiling_space();

    let guided = with_context(|context| {
        tiling.space.sample(
            &mut GuidedOracle::new(&Trace::default(), RandomOracle::new(seed)),
            context,
        )
    })
    .expect("a run");
    let alone = with_context(|context| tiling.space.sample(&mut RandomOracle::new(seed), context))
        .expect("a run");

    assert_eq!(guided.trace(), alone.trace());
    assert_eq!(
        guided.configuration().expect("a run over a space").key(),
        alone.configuration().expect("a run over a space").key()
    );
}

/// Test the fallback's error stops a guided run as it stops an unguided
/// one.
#[test]
fn guided_oracle_with_an_empty_guide_stops_with_its_fallbacks_error() {
    let tiling = build_tiling_space();

    let guided = with_context(|context| {
        tiling.space.sample(
            &mut GuidedOracle::new(&Trace::default(), RefusingOracle),
            context,
        )
    });
    let alone = with_context(|context| tiling.space.sample(&mut RefusingOracle, context));

    assert_eq!(refusal_position(guided), 0);
    assert_eq!(refusal_position(alone), 0);
}

/// Test `fallback` returns the oracle given, in the state its asks left.
#[test]
fn guided_oracle_fallback_returns_the_fallback_given() {
    let mut guided = GuidedOracle::new(&Trace::default(), ScriptedOracle::new([index(3)]));
    let answers = ask_dynamic(&mut guided, &[("k", strided(&[(0, 64)]))]);

    let fallback = guided.fallback();

    assert_eq!(answers, vec![index(3)]);
    assert_eq!(fallback.seen.len(), 1);
    assert_eq!(fallback.seen[0].kind, "k");
}

/// Test `into_fallback` returns the oracle given, in the state its asks
/// left.
#[test]
fn guided_oracle_into_fallback_returns_the_fallback_given() {
    let mut guided = GuidedOracle::new(&Trace::default(), ScriptedOracle::new([index(3)]));
    ask_dynamic(&mut guided, &[("k", strided(&[(0, 64)]))]);

    let fallback = guided.into_fallback();

    assert_eq!(fallback.seen.len(), 1);
    assert_eq!(fallback.seen[0].kind, "k");
}
