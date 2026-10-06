//! Tests for the oracles this module ships: `RandomOracle`'s draws and
//! their stream, `ReplayOracle`'s checks and `finish`, and
//! `ExhaustiveOracle`'s paths over successive runs.

use std::collections::BTreeSet;

use fhy_core::constraint::Value;
use fhy_core::identifier::Identifier;
use fhy_core::search_space::{
    ChoiceDomain, Coordinate, ExhaustiveOracle, RandomOracle, Recorder, ReplayError, ReplayOracle,
    Rng, SearchOracle, StepDomain, Trace, TraceError,
};
use rstest::rstest;

use crate::support::constraint::int;
use crate::support::param::in_set;
use crate::support::search::{
    ScriptedOracle, build_tiling_space, index, indices, int_choices, kind, order, order_of,
    strided, with_context,
};
use crate::support::search_space::{forbidden, int_variable, space_of};

/// Run one stream of dynamic steps, all of kind `k` about the subject
/// `s`, over `domains`, answered by `oracle`; return the answers.
fn run_stream(
    oracle: &mut dyn SearchOracle,
    domains: &[StepDomain],
) -> Result<Vec<Coordinate>, TraceError> {
    let subject = Identifier::new("s");
    let mut recorder = Recorder::new();
    with_context(|context| {
        domains
            .iter()
            .map(|domain| recorder.decide_dynamic(&kind("k"), &subject, domain, oracle, context))
            .collect()
    })
}

/// Run one stream over `domains` and return its trace.
fn record_stream(oracle: &mut dyn SearchOracle, domains: &[StepDomain]) -> Trace {
    let subject = Identifier::new("s");
    let mut recorder = Recorder::new();
    with_context(|context| {
        for domain in domains {
            recorder
                .decide_dynamic(&kind("k"), &subject, domain, oracle, context)
                .expect("the oracle answers inside the domain");
        }
    });
    recorder.trace()
}

/// Return the `ReplayError` a run stopped with.
///
/// # Panics
///
/// Panics if the run stopped otherwise.
fn replay_error(result: Result<Vec<Coordinate>, TraceError>) -> ReplayError {
    match result {
        Err(TraceError::Oracle { source, .. }) => match source.downcast::<ReplayError>() {
            Ok(error) => *error,
            Err(source) => panic!("expected a replay error, got {source:?}"),
        },
        other => panic!("expected an oracle failure, got {other:?}"),
    }
}

/// Return the stream the replay tests record: a choice of three, a run of
/// 64 addresses, and an order of three.
fn build_mixed_stream() -> Vec<StepDomain> {
    let [i, j, k] = ["i", "j", "k"].map(Identifier::new);
    vec![
        int_choices(&[10, 20, 30]),
        strided(&[(0, 64)]),
        order_of(&[&i, &j, &k]),
    ]
}

// ---------------------------------------------------------------------------
// RandomOracle
// ---------------------------------------------------------------------------

/// Test one seed reproduces a whole stream across every domain shape.
#[test]
fn random_oracle_reproduces_a_stream_from_its_seed() {
    let mut domains = build_mixed_stream();
    domains.push(strided(&[(0, 64)]));

    let first = run_stream(&mut RandomOracle::new(1234), &domains).expect("answers");
    let second = run_stream(&mut RandomOracle::new(1234), &domains).expect("answers");

    assert_eq!(first, second);
}

/// Test two seeds disagree on a wide domain: eight draws from 4096
/// addresses agree with probability `4096^-8`.
#[test]
fn random_oracle_differs_across_seeds() {
    let domains = vec![strided(&[(0, 4096)]); 8];

    let first = run_stream(&mut RandomOracle::new(1), &domains).expect("answers");
    let second = run_stream(&mut RandomOracle::new(2), &domains).expect("answers");

    assert_ne!(first, second);
}

/// Test 200 draws of a choice of three reach every choice.
#[test]
fn random_oracle_covers_every_choice() {
    let domains = vec![int_choices(&[10, 20, 30]); 200];

    let answers = run_stream(&mut RandomOracle::new(7), &domains).expect("answers");

    let drawn: BTreeSet<u64> = indices(&answers).into_iter().collect();
    assert_eq!(drawn, BTreeSet::from([0, 1, 2]));
}

/// Test draws reach both runs of a split strided domain.
#[test]
fn random_oracle_reaches_every_run() {
    let domains = vec![strided(&[(0, 4), (1000, 1004)]); 200];

    let answers = run_stream(&mut RandomOracle::new(11), &domains).expect("answers");

    let drawn: BTreeSet<u64> = indices(&answers).into_iter().collect();
    assert_eq!(drawn, (0..8).collect::<BTreeSet<_>>());
}

/// Test draws reach every permutation of a small order domain.
#[test]
fn random_oracle_draws_every_permutation() {
    let [i, j, k] = ["i", "j", "k"].map(Identifier::new);
    let domains = vec![order_of(&[&i, &j, &k]); 120];

    let answers = run_stream(&mut RandomOracle::new(3), &domains).expect("answers");

    let drawn: BTreeSet<Coordinate> = answers.into_iter().collect();
    assert_eq!(drawn.len(), 6);
}

/// Test a random oracle's index draws are its generator's `below`.
#[test]
fn random_oracle_draws_indices_with_below() {
    let domains = vec![int_choices(&[1, 2, 3]); 8];

    let answers = run_stream(&mut RandomOracle::new(42), &domains).expect("answers");

    assert_eq!(indices(&answers), [2, 0, 0, 1, 0, 2, 0, 2]);
}

/// Test a random oracle draws an ordering by shuffling the positions.
#[test]
fn random_oracle_draws_orders_by_shuffling_positions() {
    let elements = ["a", "b", "c", "d"].map(Identifier::new);
    let domains = vec![order_of(&elements.iter().collect::<Vec<_>>())];

    let answers = run_stream(&mut RandomOracle::new(0), &domains).expect("answers");

    assert_eq!(answers, [order(&[2, 0, 1, 3])]);
}

/// Test an oracle from a generator draws as one seeded with its seed, and
/// its generator advances with each draw.
#[test]
fn random_oracle_from_rng_draws_from_that_generator() {
    let domains = vec![int_choices(&[1, 2, 3]); 4];
    let mut oracle = RandomOracle::from_rng(Rng::new(42));

    let answers = run_stream(&mut oracle, &domains).expect("answers");

    let mut expected = Rng::new(42);
    let three = std::num::NonZeroU64::new(3).expect("positive");
    let expected_draws: Vec<u64> = (0..4).map(|_| expected.below(three)).collect();
    assert_eq!(indices(&answers), expected_draws);
    assert_eq!(oracle.rng(), &expected);
}

/// Test a random oracle draws only admissible coordinates: after `t = 2`,
/// the tiling space's choice can only take `a`.
#[rstest]
#[case::seed_0(0)]
#[case::seed_1(1)]
#[case::seed_2(2)]
#[case::seed_3(3)]
#[case::seed_4(4)]
#[case::seed_5(5)]
fn random_oracle_never_draws_an_inadmissible_coordinate(#[case] seed: u64) {
    let tiling = build_tiling_space();
    let mut recorder = Recorder::over(&tiling.space);
    let mut setup = ScriptedOracle::new([index(1)]);
    let mut random = RandomOracle::new(seed);

    let choice = with_context(|context| {
        recorder.decide(&tiling.t, &mut setup, context)?;
        recorder.decide(&tiling.c, &mut random, context)
    })
    .expect("the random oracle draws an admissible alternative");

    assert_eq!(choice, Value::Identifier(tiling.a.clone()));
}

/// Test a step whose every value is forbidden is a dead end for the random
/// oracle, which the run reports as such.
#[test]
fn random_oracle_reports_a_dead_end() {
    let [name, k] = ["closed", "k"].map(Identifier::new);
    let space = fhy_core::search_space::Space::new(
        name,
        vec![int_variable(&k, &[1, 2])],
        Vec::new(),
        Vec::new(),
        vec![forbidden([in_set(&k, [int(1), int(2)])])],
    )
    .expect("a space forbidding every value of k");
    let mut recorder = Recorder::over(&space);
    let mut random = RandomOracle::new(0);

    let result = with_context(|context| recorder.decide(&k, &mut random, context));

    assert!(
        matches!(&result, Err(TraceError::DeadEnd { decision }) if *decision == k),
        "{result:?}"
    );
}

// ---------------------------------------------------------------------------
// ReplayOracle
// ---------------------------------------------------------------------------

/// Test a replay answers a recorded stream exactly, and is exhausted and
/// finishes afterwards.
#[test]
fn replay_reproduces_a_stream() {
    let domains = build_mixed_stream();
    let trace = record_stream(&mut RandomOracle::new(99), &domains);
    let mut replay = ReplayOracle::new(trace.clone());

    let answers = run_stream(&mut replay, &domains).expect("the stream replays");

    assert_eq!(answers, trace.coordinates().cloned().collect::<Vec<_>>());
    assert!(replay.is_exhausted());
    assert_eq!(replay.trace(), &trace);
    replay.finish().expect("every recorded step was asked");
}

/// Test a replay refuses a step of another kind.
#[test]
fn replay_refuses_another_kind() {
    let domain = int_choices(&[1, 2]);
    let trace = record_stream(&mut RandomOracle::new(1), &[domain.clone()]);
    let mut replay = ReplayOracle::new(trace);
    let mut recorder = Recorder::new();

    let result = with_context(|context| {
        recorder
            .decide_dynamic(
                &kind("other"),
                &Identifier::new("s"),
                &domain,
                &mut replay,
                context,
            )
            .map(|answer| vec![answer])
    });

    assert!(
        matches!(
            replay_error(result),
            ReplayError::KindMismatch { position: 0 }
        ),
        "the kinds differ"
    );
}

/// Test a replay refuses a step over a domain of another shape, another
/// size, a moved run of the same size, or reordered plain choices.
#[rstest]
#[case::another_shape(int_choices(&[1, 2, 3]), strided(&[(0, 3)]))]
#[case::another_size(strided(&[(0, 64)]), strided(&[(0, 32)]))]
#[case::moved_run(strided(&[(0, 64)]), strided(&[(64, 128)]))]
#[case::reordered_plain_choices(int_choices(&[1, 2, 3]), int_choices(&[3, 2, 1]))]
fn replay_refuses_another_domain(#[case] recorded: StepDomain, #[case] offered: StepDomain) {
    let trace = record_stream(&mut RandomOracle::new(1), &[recorded]);

    let result = run_stream(&mut ReplayOracle::new(trace), &[offered]);

    assert!(
        matches!(
            replay_error(result),
            ReplayError::DomainMismatch { position: 0 }
        ),
        "the domains differ"
    );
}

/// Test a replay answers a domain of fresh identifiers of the same size:
/// identifiers count by position only.
#[test]
fn replay_answers_fresh_identifier_choices_of_one_size() {
    let fresh = || {
        StepDomain::from(
            ChoiceDomain::new(
                (0..4)
                    .map(|_| Value::Identifier(Identifier::new("option")))
                    .collect(),
            )
            .expect("fresh identifiers are distinct"),
        )
    };
    let trace = record_stream(&mut RandomOracle::new(5), &[fresh()]);

    let answers =
        run_stream(&mut ReplayOracle::new(trace.clone()), &[fresh()]).expect("the stream replays");

    assert_eq!(answers, trace.coordinates().cloned().collect::<Vec<_>>());
}

/// Test a replay does not compare a dynamic step's subject.
#[test]
fn replay_ignores_a_dynamic_steps_subject() {
    let domain = int_choices(&[1, 2, 3]);
    let trace = record_stream(&mut RandomOracle::new(4), &[domain.clone()]);
    let mut replay = ReplayOracle::new(trace.clone());
    let mut recorder = Recorder::new();

    let answer = with_context(|context| {
        recorder.decide_dynamic(
            &kind("k"),
            &Identifier::new("another"),
            &domain,
            &mut replay,
            context,
        )
    })
    .expect("the subject is not compared");

    assert_eq!(Some(&answer), trace.coordinates().next());
}

/// Test steps with one subject replay in the order they were asked.
#[test]
fn replay_answers_repeated_subjects_in_ask_order() {
    let pool = int_choices(&[0, 1]);
    let trace = Trace::new(vec![
        fhy_core::search_space::TraceStep::dynamic(
            kind("k"),
            Identifier::new("s"),
            &pool,
            index(0),
        )
        .expect("in range"),
        fhy_core::search_space::TraceStep::dynamic(
            kind("k"),
            Identifier::new("s"),
            &pool,
            index(1),
        )
        .expect("in range"),
    ]);

    let answers = run_stream(&mut ReplayOracle::new(trace), &[pool.clone(), pool])
        .expect("the stream replays");

    assert_eq!(answers, [index(0), index(1)]);
}

/// Test a replay refuses a stream longer than the trace.
#[test]
fn replay_refuses_a_longer_stream() {
    let domain = int_choices(&[1, 2]);
    let trace = record_stream(&mut RandomOracle::new(1), &[domain.clone()]);

    let result = run_stream(&mut ReplayOracle::new(trace), &[domain.clone(), domain]);

    assert!(
        matches!(replay_error(result), ReplayError::Exhausted { position: 1 }),
        "one step too many"
    );
}

/// Test finishing a replay refuses a stream shorter than the trace, naming
/// the first step it never asked.
#[test]
fn replay_finish_refuses_a_shorter_stream() {
    let domains = build_mixed_stream();
    let trace = record_stream(&mut RandomOracle::new(8), &domains);
    let mut replay = ReplayOracle::new(trace);
    run_stream(&mut replay, &domains[..1]).expect("the first step replays");

    let result = replay.finish();

    assert!(
        matches!(result, Err(ReplayError::Unconsumed { position: 1 })),
        "{result:?}"
    );
}

/// Test a replay refuses a static step for a dynamic one, and a static step
/// of another decision.
#[test]
fn replay_refuses_another_decision() {
    let [name, u, v] = ["pair", "u", "v"].map(Identifier::new);
    let space = space_of(
        &name,
        vec![int_variable(&u, &[1, 2]), int_variable(&v, &[1, 2])],
        Vec::new(),
    );
    let mut recorder = Recorder::over(&space);
    with_context(|context| recorder.decide(&u, &mut RandomOracle::new(0), context))
        .expect("u is decided");
    let trace = recorder.trace();

    let mut by_variable = Recorder::over(&space);
    let other_variable = with_context(|context| {
        by_variable
            .decide(&v, &mut ReplayOracle::new(trace.clone()), context)
            .map(|_| Vec::new())
    });
    let mut dynamic = Recorder::new();
    let as_dynamic = with_context(|context| {
        dynamic
            .decide_dynamic(
                &kind("search_space.variable"),
                &u,
                &int_choices(&[1, 2]),
                &mut ReplayOracle::new(trace.clone()),
                context,
            )
            .map(|answer| vec![answer])
    });

    assert!(
        matches!(
            replay_error(other_variable),
            ReplayError::DecisionMismatch { position: 0 }
        ),
        "another decision"
    );
    assert!(
        matches!(
            replay_error(as_dynamic),
            ReplayError::DecisionMismatch { position: 0 }
        ),
        "a static step replayed as a dynamic one"
    );
}

/// Test a replay refuses a recorded answer this run does not admit.
#[test]
fn replay_refuses_an_answer_this_run_does_not_admit() {
    let tiling = build_tiling_space();
    let record = |answers: Vec<Coordinate>, decisions: &[&Identifier]| {
        let mut recorder = Recorder::over(&tiling.space);
        let mut oracle = ScriptedOracle::new(answers);
        with_context(|context| {
            for decision in decisions {
                recorder
                    .decide(decision, &mut oracle, context)
                    .expect("an admissible answer");
            }
        });
        recorder.trace()
    };
    let with_t_two = record(vec![index(1)], &[&tiling.t]);
    let with_b = record(vec![index(0), index(1)], &[&tiling.t, &tiling.c]);
    let spliced = Trace::new(vec![
        with_t_two.steps()[0].clone(),
        with_b.steps()[1].clone(),
    ]);
    let mut recorder = Recorder::over(&tiling.space);
    let mut replay = ReplayOracle::new(spliced);

    let result = with_context(|context| {
        recorder.decide(&tiling.t, &mut replay, context)?;
        recorder
            .decide(&tiling.c, &mut replay, context)
            .map(|_| Vec::new())
    });

    assert!(
        matches!(
            replay_error(result),
            ReplayError::Inadmissible { position: 1 }
        ),
        "b is forbidden with t = 2"
    );
}

// ---------------------------------------------------------------------------
// ExhaustiveOracle
// ---------------------------------------------------------------------------

/// Run `oracle` over successive runs of `stream`, which asks its steps
/// given the answers so far, until `advance` says every path was taken;
/// return each finished run's answers.
fn run_every_path(
    oracle: &mut ExhaustiveOracle,
    stream: impl Fn(&mut Recorder, &mut ExhaustiveOracle) -> Result<Vec<Coordinate>, TraceError>,
) -> Vec<Vec<Coordinate>> {
    let mut paths = Vec::new();
    loop {
        let mut recorder = Recorder::new();
        match stream(&mut recorder, oracle) {
            Ok(path) => paths.push(path),
            Err(error) if ExhaustiveOracle::is_backtrack(&error) => {}
            Err(error) => panic!("the stream failed: {error}"),
        }
        if !oracle.advance() {
            return paths;
        }
    }
}

/// Test successive runs of a fixed stream take every path once, in
/// lexicographic order of their coordinates.
#[test]
fn exhaustive_oracle_takes_every_path_of_a_fixed_stream_in_order() {
    let mut oracle = ExhaustiveOracle::new();
    let domains = [int_choices(&[1, 2]), int_choices(&[1, 2, 3])];

    let paths = run_every_path(&mut oracle, |_, oracle| run_stream(oracle, &domains));

    let expected: Vec<Vec<Coordinate>> = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
        .iter()
        .map(|path| path.iter().map(|&position| index(position)).collect())
        .collect();
    assert_eq!(paths, expected);
}

/// Test a conditional stream's paths: a second step only after the first
/// answers 0.
#[test]
fn exhaustive_oracle_takes_every_path_of_a_conditional_stream() {
    let mut oracle = ExhaustiveOracle::new();
    let first = int_choices(&[1, 2]);
    let second = int_choices(&[1, 2, 3]);
    let subject = Identifier::new("s");

    let paths = run_every_path(&mut oracle, |recorder, oracle| {
        with_context(|context| {
            let head = recorder.decide_dynamic(&kind("k"), &subject, &first, oracle, context)?;
            if head == index(1) {
                return Ok(vec![head]);
            }
            let tail = recorder.decide_dynamic(&kind("k"), &subject, &second, oracle, context)?;
            Ok(vec![head, tail])
        })
    });

    assert_eq!(
        paths,
        [
            vec![index(0), index(0)],
            vec![index(0), index(1)],
            vec![index(0), index(2)],
            vec![index(1)],
        ]
    );
}

/// Test an order step's paths are its permutations in lexicographic order.
#[test]
fn exhaustive_oracle_orders_permutations_lexicographically() {
    let [i, j, k] = ["i", "j", "k"].map(Identifier::new);
    let domains = [order_of(&[&i, &j, &k])];
    let mut oracle = ExhaustiveOracle::new();

    let paths = run_every_path(&mut oracle, |_, oracle| run_stream(oracle, &domains));

    let expected: Vec<Vec<Coordinate>> = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ]
    .iter()
    .map(|positions| vec![order(positions)])
    .collect();
    assert_eq!(paths, expected);
}

/// Test a step with no admissible coordinate abandons the run as a
/// backtrack, after which no path is left.
#[test]
fn exhaustive_oracle_backtracks_over_an_exhausted_branch() {
    let [name, k] = ["closed", "k"].map(Identifier::new);
    let space = fhy_core::search_space::Space::new(
        name,
        vec![int_variable(&k, &[1, 2])],
        Vec::new(),
        Vec::new(),
        vec![forbidden([in_set(&k, [int(1), int(2)])])],
    )
    .expect("a space forbidding every value of k");
    let mut oracle = ExhaustiveOracle::new();
    let mut recorder = Recorder::over(&space);

    let result = with_context(|context| recorder.decide(&k, &mut oracle, context));

    let error = result.expect_err("no admissible value");
    assert!(ExhaustiveOracle::is_backtrack(&error), "{error}");
    assert!(!oracle.advance());
}

/// Test a backtrack is told apart from any other failure.
#[test]
fn exhaustive_oracle_is_backtrack_is_false_for_other_failures() {
    let error = TraceError::NoSpace;

    assert!(!ExhaustiveOracle::is_backtrack(&error));
}

/// Test a stream that changes a replayed step's domain between runs is
/// refused: it is not deterministic.
#[test]
fn exhaustive_oracle_refuses_a_nondeterministic_stream() {
    let mut oracle = ExhaustiveOracle::new();
    run_stream(&mut oracle, &[int_choices(&[1, 2]), int_choices(&[1, 2])]).expect("the first run");
    assert!(oracle.advance());

    let result = run_stream(&mut oracle, &[strided(&[(0, 2)])]);

    assert!(
        matches!(
            replay_error(result),
            ReplayError::DomainMismatch { position: 0 }
        ),
        "the first step's domain changed"
    );
}

/// Test a random oracle finds the one admissible value of a wide domain,
/// whether its first draws hit it or it lists the admissible coordinates.
#[rstest]
#[case::seed_0(0)]
#[case::seed_1(1)]
#[case::seed_2(2)]
#[case::seed_3(3)]
#[case::seed_4(4)]
#[case::seed_5(5)]
#[case::seed_6(6)]
#[case::seed_7(7)]
fn random_oracle_finds_the_one_admissible_value(#[case] seed: u64) {
    let [name, k] = ["narrow", "k"].map(Identifier::new);
    let values: Vec<Value> = (0..200).map(int).collect();
    let param =
        crate::support::search_space::categorical_where(values, |p| vec![in_set(p, [int(137)])]);
    let space = space_of(
        &name,
        vec![crate::support::search_space::plain_variable(&k, param)],
        Vec::new(),
    );
    let mut recorder = Recorder::over(&space);

    let value = with_context(|context| recorder.decide(&k, &mut RandomOracle::new(seed), context))
        .expect("the one admissible value is found");

    assert_eq!(value, int(137));
}
