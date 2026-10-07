//! Tests for what a `Space` does on its own: `sample`, `replay` and
//! `Configuration::trace`, `enumerate`, `cardinality`, `sample_uniform`
//! and `mutate`.

use std::collections::HashSet;
use std::num::NonZeroU32;

use fhy_core::constraint::Value;
use fhy_core::foreign::Part;
use fhy_core::identifier::Identifier;
use fhy_core::param::{Param, ParamContext};
use fhy_core::search_space::{
    Cardinality, Configuration, ConfigurationKey, ExhaustiveOracle, RandomOracle, Recorder,
    ReplayError, Rng, Space, Trace, TraceError, Variable,
};
use fhy_core::solver::Solver;
use num_bigint::BigUint;
use rstest::rstest;

use crate::support::constraint::int;
use crate::support::param::{EvenDomain, in_set};
use crate::support::search::{
    ScriptedOracle, TilingSpace, build_tiling_space, index, tiling_configurations, tiling_entries,
    unbounded_space, with_context,
};
use crate::support::search_space::{
    bare_alternative, choice_of, chosen, condition, configure, forbidden, int_variable,
    plain_alternative, plain_variable, space_of,
};

/// Return a positive attempt count.
fn attempts(count: u32) -> NonZeroU32 {
    NonZeroU32::new(count).expect("a positive count")
}

/// Return the keys of the tiling space's five complete configurations, in
/// lexicographic order of their coordinates.
fn tiling_keys(tiling: &TilingSpace) -> Vec<ConfigurationKey> {
    tiling_configurations(tiling)
        .iter()
        .map(|configuration| configure(&tiling.space, tiling_entries(tiling, configuration)).key())
        .collect()
}

/// Return the space of one choice between `a`, holding a variable over
/// `{1, 2, 3}`, and `b`, holding nothing: four configurations.
fn build_choice_space() -> Space {
    let [name, c, a, x, b] = ["choosing", "c", "a", "x", "b"].map(Identifier::new);
    space_of(
        &name,
        Vec::new(),
        vec![choice_of(
            &c,
            vec![
                plain_alternative(&a, vec![int_variable(&x, &[1, 2, 3])], Vec::new()),
                bare_alternative(&b),
            ],
        )],
    )
}

/// Return the space forbidding every value of its one variable `k`.
fn build_closed_space(k: &Identifier) -> Space {
    Space::new(
        Identifier::new("closed"),
        vec![int_variable(k, &[1, 2])],
        Vec::new(),
        Vec::new(),
        vec![forbidden([in_set(k, [int(1), int(2)])])],
    )
    .expect("a space forbidding every value of k")
}

/// Return the space of one variable `even` whose param has a custom domain
/// and which offers no search domain.
fn build_custom_space(even: &Identifier) -> Space {
    let solver = Solver::new();
    let (domain, _calls) = EvenDomain::build(false);
    let param = Param::new(
        domain,
        Identifier::new("e"),
        Vec::new(),
        &ParamContext::new(&solver),
    )
    .expect("a custom-domain param");
    space_of(
        &Identifier::new("custom"),
        vec![plain_variable(even, param)],
        Vec::new(),
    )
}

// ---------------------------------------------------------------------------
// sample
// ---------------------------------------------------------------------------

/// Test sampling asks every active decision in decision order and gives a
/// complete configuration with its trace.
#[test]
fn space_sample_asks_every_active_decision_in_decision_order() {
    let tiling = build_tiling_space();
    let mut oracle = ScriptedOracle::new([index(0), index(0), index(1)]);

    let recorded = with_context(|context| tiling.space.sample(&mut oracle, context))
        .expect("admissible answers");

    let configuration = recorded.configuration().expect("a run over a space");
    assert!(configuration.is_complete());
    assert_eq!(
        configuration.key(),
        configure(
            &tiling.space,
            [
                (tiling.t.clone(), int(1)),
                (tiling.c.clone(), chosen(&tiling.a)),
                (tiling.x.clone(), int(2)),
            ]
        )
        .key()
    );
    let asked: Vec<Option<Identifier>> = oracle
        .seen
        .iter()
        .map(|step| step.decision.clone())
        .collect();
    assert_eq!(
        asked,
        [
            Some(tiling.t.clone()),
            Some(tiling.c.clone()),
            Some(tiling.x.clone())
        ]
    );
    assert_eq!(recorded.trace().len(), 3);
}

/// Test sampling skips a decision that is inactive when it is reached.
#[test]
fn space_sample_skips_inactive_decisions() {
    let tiling = build_tiling_space();
    let mut oracle = ScriptedOracle::new([index(1), index(0)]);

    let recorded = with_context(|context| tiling.space.sample(&mut oracle, context))
        .expect("admissible answers");

    let configuration = recorded.configuration().expect("a run over a space");
    assert!(configuration.is_complete());
    assert_eq!(configuration.value(&tiling.x), None);
    assert_eq!(oracle.seen.len(), 2);
}

/// Test sampling a variable with no finite domain is refused.
#[test]
fn space_sample_refuses_a_variable_without_a_finite_domain() {
    let [name, n] = ["counting", "n"].map(Identifier::new);
    let space = unbounded_space(&name, &n);

    let result = with_context(|context| space.sample(&mut RandomOracle::new(0), context));

    assert!(
        matches!(&result, Err(TraceError::NotEnumerable { decision }) if *decision == n),
        "{result:?}"
    );
}

// ---------------------------------------------------------------------------
// replay and Configuration::trace
// ---------------------------------------------------------------------------

/// Test a configuration's trace replays into the same configuration.
#[test]
fn configuration_trace_replays_into_the_configuration() {
    let tiling = build_tiling_space();
    let configuration = configure(
        &tiling.space,
        tiling_entries(&tiling, &(1, tiling.a.clone(), Some(3))),
    );

    let replayed = with_context(|context| {
        let trace = configuration.trace(context)?;
        Ok::<_, TraceError>((trace.clone(), tiling.space.replay(&trace, context)))
    })
    .expect("a finite configuration has a trace");

    let (trace, replayed) = replayed;
    assert_eq!(
        replayed.expect("the trace replays").key(),
        configuration.key()
    );
    assert_eq!(
        trace.coordinates().cloned().collect::<Vec<_>>(),
        [index(0), index(0), index(2)]
    );
    assert_eq!(
        trace
            .steps()
            .iter()
            .map(fhy_core::search_space::TraceStep::decision)
            .collect::<Vec<_>>(),
        [Some(0), Some(1), Some(2)]
    );
}

/// Test replay reads static steps by decision, whatever order they were
/// asked in.
#[test]
fn space_replay_reads_steps_in_any_order() {
    let tiling = build_tiling_space();
    let configuration = configure(
        &tiling.space,
        tiling_entries(&tiling, &(1, tiling.a.clone(), Some(2))),
    );
    let trace = with_context(|context| configuration.trace(context)).expect("a trace");
    let reversed = Trace::new(trace.steps().iter().rev().cloned().collect());

    let replayed = with_context(|context| tiling.space.replay(&reversed, context))
        .expect("the steps describe a configuration");

    assert_eq!(replayed.key(), configuration.key());
}

/// Test replay skips dynamic steps.
#[test]
fn space_replay_skips_dynamic_steps() {
    let tiling = build_tiling_space();
    let configuration = configure(
        &tiling.space,
        tiling_entries(&tiling, &(1, tiling.b.clone(), None)),
    );
    let trace = with_context(|context| configuration.trace(context)).expect("a trace");
    let dynamic = fhy_core::search_space::TraceStep::dynamic(
        crate::support::search::kind("moga.cir.address"),
        Identifier::new("value"),
        &crate::support::search::strided(&[(0, 8)]),
        index(3),
    )
    .expect("in range");
    let mut steps = trace.steps().to_vec();
    steps.insert(1, dynamic);

    let replayed = with_context(|context| tiling.space.replay(&Trace::new(steps), context))
        .expect("the static steps describe a configuration");

    assert_eq!(replayed.key(), configuration.key());
}

/// Test replay into a relabeled copy of the space gives the configuration
/// of that copy with an equal key.
#[test]
fn space_replay_onto_a_relabeled_copy_gives_an_equal_key() {
    let original = build_tiling_space();
    let copy = build_tiling_space();
    let configuration = configure(
        &original.space,
        tiling_entries(&original, &(1, original.a.clone(), Some(1))),
    );
    let trace = with_context(|context| configuration.trace(context)).expect("a trace");

    let replayed = with_context(|context| copy.space.replay(&trace, context))
        .expect("the copy has the same shape");

    assert_eq!(replayed.key(), configuration.key());
    assert_eq!(replayed.space(), &copy.space);
    assert_eq!(replayed.value(&copy.x), Some(&int(1)));
}

/// Test replay refuses a trace missing the step of an active decision.
#[test]
fn space_replay_refuses_a_missing_step() {
    let tiling = build_tiling_space();
    let configuration = configure(
        &tiling.space,
        tiling_entries(&tiling, &(1, tiling.a.clone(), Some(2))),
    );
    let trace = with_context(|context| configuration.trace(context)).expect("a trace");
    let partial = Trace::new(trace.steps()[..2].to_vec());

    let result = with_context(|context| tiling.space.replay(&partial, context));

    assert!(
        matches!(&result, Err(ReplayError::MissingStep { decision }) if *decision == tiling.x),
        "{result:?}"
    );
}

/// Test replay refuses two steps for one decision.
#[test]
fn space_replay_refuses_a_repeated_step() {
    let tiling = build_tiling_space();
    let configuration = configure(
        &tiling.space,
        tiling_entries(&tiling, &(1, tiling.b.clone(), None)),
    );
    let trace = with_context(|context| configuration.trace(context)).expect("a trace");
    let mut steps = trace.steps().to_vec();
    steps.push(steps[0].clone());

    let result = with_context(|context| tiling.space.replay(&Trace::new(steps), context));

    assert!(
        matches!(&result, Err(ReplayError::RepeatedStep { decision }) if *decision == tiling.t),
        "{result:?}"
    );
}

/// Test replay refuses a step for a decision the other steps make
/// inactive.
#[test]
fn space_replay_refuses_a_step_for_an_inactive_decision() {
    let tiling = build_tiling_space();
    let with_x = configure(
        &tiling.space,
        tiling_entries(&tiling, &(1, tiling.a.clone(), Some(2))),
    );
    let with_b = configure(
        &tiling.space,
        tiling_entries(&tiling, &(1, tiling.b.clone(), None)),
    );
    let (x_trace, b_trace) = with_context(|context| {
        Ok::<_, TraceError>((with_x.trace(context)?, with_b.trace(context)?))
    })
    .expect("traces");
    let mut steps = b_trace.steps().to_vec();
    steps.push(x_trace.steps()[2].clone());

    let result = with_context(|context| tiling.space.replay(&Trace::new(steps), context));

    assert!(
        matches!(result, Err(ReplayError::DecisionMismatch { .. })),
        "{result:?}"
    );
}

/// Test replay refuses a step whose domain differs from the decision's.
#[test]
fn space_replay_refuses_a_step_over_another_domain() {
    let [name, k] = ["single", "k"].map(Identifier::new);
    let narrow = space_of(&name, vec![int_variable(&k, &[1, 2])], Vec::new());
    let [wide_name, wide_k] = ["single", "k"].map(Identifier::new);
    let wide = space_of(
        &wide_name,
        vec![int_variable(&wide_k, &[1, 2, 3])],
        Vec::new(),
    );
    let configuration = configure(&narrow, [(k.clone(), int(2))]);
    let trace = with_context(|context| configuration.trace(context)).expect("a trace");

    let result = with_context(|context| wide.replay(&trace, context));

    assert!(
        matches!(result, Err(ReplayError::DomainMismatch { position: 0 })),
        "{result:?}"
    );
}

/// Test the trace of a configuration assigning an unbounded variable is
/// refused.
#[test]
fn configuration_trace_refuses_a_variable_without_a_finite_domain() {
    let [name, n] = ["counting", "n"].map(Identifier::new);
    let space = unbounded_space(&name, &n);
    let configuration = configure(&space, [(n.clone(), int(7))]);

    let result = with_context(|context| configuration.trace(context));

    assert!(
        matches!(&result, Err(TraceError::NotEnumerable { decision }) if *decision == n),
        "{result:?}"
    );
}

// ---------------------------------------------------------------------------
// enumerate
// ---------------------------------------------------------------------------

/// Test enumeration yields every complete configuration once, in
/// lexicographic order of coordinates in decision order.
#[test]
fn space_enumerate_yields_every_configuration_in_order() {
    let tiling = build_tiling_space();

    let keys: Vec<ConfigurationKey> = with_context(|context| {
        tiling
            .space
            .enumerate(context)
            .map(|configuration| configuration.map(|configuration| configuration.key()))
            .collect::<Result<_, _>>()
    })
    .expect("a finite space enumerates");

    assert_eq!(keys, tiling_keys(&tiling));
}

/// Test an empty space has one configuration, the empty one.
#[test]
fn space_enumerate_of_an_empty_space_yields_the_empty_configuration() {
    let space = space_of(&Identifier::new("empty"), Vec::new(), Vec::new());

    let configurations: Vec<Configuration> =
        with_context(|context| space.enumerate(context).collect::<Result<_, _>>())
            .expect("an empty space enumerates");

    assert_eq!(configurations.len(), 1);
    assert!(configurations[0].is_complete());
}

/// Test a space forbidding everything yields nothing.
#[test]
fn space_enumerate_of_a_closed_space_yields_nothing() {
    let space = build_closed_space(&Identifier::new("k"));

    let count = with_context(|context| space.enumerate(context).count());

    assert_eq!(count, 0);
}

/// Test enumeration of a space with no finite domain yields one refusal
/// and ends.
#[test]
fn space_enumerate_refuses_a_variable_without_a_finite_domain_once() {
    let [name, n] = ["counting", "n"].map(Identifier::new);
    let space = unbounded_space(&name, &n);

    let items: Vec<Result<Configuration, TraceError>> =
        with_context(|context| space.enumerate(context).collect());

    assert_eq!(items.len(), 1);
    assert!(
        matches!(&items[0], Err(TraceError::NotEnumerable { decision }) if *decision == n),
        "{:?}",
        items[0]
    );
}

/// Test enumerating a bounded integer variable yields each integer of its
/// interval.
#[test]
fn space_enumerate_walks_a_bounded_integer_interval() {
    let [name, n] = ["bounded", "n"].map(Identifier::new);
    let space = space_of(
        &name,
        vec![plain_variable(
            &n,
            crate::support::search::bounded_param(-1, 2),
        )],
        Vec::new(),
    );

    let values: Vec<Value> = with_context(|context| {
        space
            .enumerate(context)
            .map(|configuration| {
                configuration
                    .map(|configuration| configuration.value(&n).cloned().expect("assigned"))
            })
            .collect::<Result<_, _>>()
    })
    .expect("a bounded space enumerates");

    assert_eq!(values, [int(-1), int(0), int(1), int(2)]);
}

// ---------------------------------------------------------------------------
// cardinality
// ---------------------------------------------------------------------------

/// Test the counts of spaces counted in closed form and by enumeration.
#[rstest]
#[case::choice(build_choice_space(), 4_u32)]
#[case::independent_variables(
    space_of(
        &Identifier::new("pair"),
        vec![int_variable(&Identifier::new("u"), &[1, 2]), int_variable(&Identifier::new("v"), &[1, 2, 3])],
        Vec::new(),
    ),
    6_u32
)]
#[case::conditional(build_tiling_space().space, 5_u32)]
#[case::empty(space_of(&Identifier::new("empty"), Vec::new(), Vec::new()), 1_u32)]
#[case::closed(build_closed_space(&Identifier::new("k")), 0_u32)]
#[case::constrained_variable(
    space_of(
        &Identifier::new("narrowed"),
        vec![plain_variable(
            &Identifier::new("k"),
            crate::support::search_space::categorical_where(vec![int(1), int(2), int(3)], |p| {
                vec![in_set(p, [int(1), int(2)])]
            }),
        )],
        Vec::new(),
    ),
    2_u32
)]
#[case::permutation(
    space_of(
        &Identifier::new("ordered"),
        vec![plain_variable(
            &Identifier::new("order"),
            crate::support::search::permutation_param(&[
                &Identifier::new("i"),
                &Identifier::new("j"),
                &Identifier::new("k"),
            ]),
        )],
        Vec::new(),
    ),
    6_u32
)]
#[case::bounded_interval(
    space_of(
        &Identifier::new("bounded"),
        vec![plain_variable(&Identifier::new("n"), crate::support::search::bounded_param(1, 10))],
        Vec::new(),
    ),
    10_u32
)]
fn space_cardinality_counts_complete_configurations(#[case] space: Space, #[case] count: u32) {
    let cardinality = with_context(|context| space.cardinality(context, 1_000));

    assert_eq!(
        cardinality.expect("the count succeeds"),
        Cardinality::Exact(BigUint::from(count))
    );
}

/// Test a count that runs out of budget gives a lower bound.
#[test]
fn space_cardinality_gives_a_lower_bound_when_the_budget_runs_out() {
    let tiling = build_tiling_space();

    let cardinality = with_context(|context| tiling.space.cardinality(context, 2));

    let Ok(Cardinality::AtLeast(bound)) = cardinality else {
        panic!("expected a lower bound, got {cardinality:?}");
    };
    assert!(bound <= BigUint::from(2_u8), "{bound}");
    assert!(bound >= BigUint::from(1_u8), "{bound}");
}

/// Test a variable over unbounded integers makes the count unbounded.
#[test]
fn space_cardinality_of_an_unbounded_variable_is_unbounded() {
    let [name, n] = ["counting", "n"].map(Identifier::new);
    let space = unbounded_space(&name, &n);

    let cardinality = with_context(|context| space.cardinality(context, 1_000));

    assert_eq!(
        cardinality.expect("the count succeeds"),
        Cardinality::Unbounded { decision: n }
    );
}

/// Test a custom domain with no search domain makes the count unknown.
#[test]
fn space_cardinality_of_a_custom_domain_is_unknown() {
    let even = Identifier::new("even");
    let space = build_custom_space(&even);

    let cardinality = with_context(|context| space.cardinality(context, 1_000));

    assert_eq!(
        cardinality.expect("the count succeeds"),
        Cardinality::Unknown { decision: even }
    );
}

/// Test components combine: an empty component makes the count zero even
/// beside an unbounded one, and an unbounded one beside a finite one is
/// unbounded.
#[test]
fn space_cardinality_combines_components() {
    let [n, k, u] = ["n", "k", "u"].map(Identifier::new);
    let unbounded = plain_variable(&n, crate::support::search_space::natural_param());
    let closed = Space::new(
        Identifier::new("closed_and_unbounded"),
        vec![unbounded.clone(), int_variable(&k, &[1, 2])],
        Vec::new(),
        Vec::new(),
        vec![forbidden([in_set(&k, [int(1), int(2)])])],
    )
    .expect("a valid space");
    let open = space_of(
        &Identifier::new("finite_and_unbounded"),
        vec![unbounded, int_variable(&u, &[1, 2])],
        Vec::new(),
    );

    let counts = with_context(|context| {
        Ok::<_, TraceError>([
            closed.cardinality(context, 1_000)?,
            open.cardinality(context, 1_000)?,
        ])
    })
    .expect("the counts succeed");

    assert_eq!(
        counts,
        [
            Cardinality::Exact(BigUint::from(0_u8)),
            Cardinality::Unbounded { decision: n },
        ]
    );
}

/// Test a condition linking two top-level decisions puts them in one
/// component, counted by enumeration.
#[test]
fn space_cardinality_counts_a_conditioned_pair_together() {
    let [name, p, q] = ["linked", "p", "q"].map(Identifier::new);
    let space = Space::new(
        name,
        vec![int_variable(&p, &[1, 2]), int_variable(&q, &[1, 2, 3])],
        Vec::new(),
        vec![condition(&q, [in_set(&p, [int(1)])])],
        Vec::new(),
    )
    .expect("a valid space");

    let cardinality = with_context(|context| space.cardinality(context, 1_000));

    assert_eq!(
        cardinality.expect("the count succeeds"),
        Cardinality::Exact(BigUint::from(4_u8))
    );
}

// ---------------------------------------------------------------------------
// sample_uniform
// ---------------------------------------------------------------------------

/// Test a uniform draw is one of the space's complete configurations, and
/// one seed draws the same one.
#[test]
fn space_sample_uniform_draws_a_complete_configuration() {
    let tiling = build_tiling_space();
    let keys: HashSet<ConfigurationKey> = tiling_keys(&tiling).into_iter().collect();

    let draw = |seed| {
        with_context(|context| {
            tiling
                .space
                .sample_uniform(&mut Rng::new(seed), context, attempts(1_000))
        })
        .expect("the space has configurations")
    };
    let first = draw(17);
    let again = draw(17);

    let configuration = first.configuration().expect("a run over a space");
    assert!(configuration.is_complete());
    assert!(keys.contains(&configuration.key()));
    assert_eq!(
        configuration.key(),
        again.configuration().expect("a run over a space").key()
    );
    let replayed = with_context(|context| tiling.space.replay(first.trace(), context))
        .expect("the trace describes the configuration");
    assert_eq!(replayed.key(), configuration.key());
}

/// Test a space whose every point is refused exhausts its attempts.
#[test]
fn space_sample_uniform_exhausts_its_attempts_on_a_closed_space() {
    let space = build_closed_space(&Identifier::new("k"));

    let result =
        with_context(|context| space.sample_uniform(&mut Rng::new(0), context, attempts(3)));

    assert!(
        matches!(result, Err(TraceError::AttemptsExhausted { attempts: 3 })),
        "{result:?}"
    );
}

/// Test a uniform draw needs every domain finite.
#[test]
fn space_sample_uniform_refuses_a_variable_without_a_finite_domain() {
    let [name, n] = ["counting", "n"].map(Identifier::new);
    let space = unbounded_space(&name, &n);

    let result =
        with_context(|context| space.sample_uniform(&mut Rng::new(0), context, attempts(5)));

    assert!(
        matches!(&result, Err(TraceError::NotEnumerable { decision }) if *decision == n),
        "{result:?}"
    );
}

// ---------------------------------------------------------------------------
// mutate
// ---------------------------------------------------------------------------

/// Test a mutation of two independent variables changes exactly one of
/// them.
#[rstest]
#[case::seed_0(0)]
#[case::seed_1(1)]
#[case::seed_2(2)]
#[case::seed_3(3)]
fn space_mutate_changes_exactly_one_independent_decision(#[case] seed: u64) {
    let [name, u, v] = ["pair", "u", "v"].map(Identifier::new);
    let space = space_of(
        &name,
        vec![int_variable(&u, &[1, 2, 3]), int_variable(&v, &[1, 2, 3])],
        Vec::new(),
    );
    let configuration = configure(&space, [(u.clone(), int(2)), (v.clone(), int(2))]);

    let mutated = with_context(|context| {
        space.mutate(&configuration, &mut Rng::new(seed), context, attempts(16))
    })
    .expect("a mutation");

    let mutated = mutated.configuration().expect("a run over a space").clone();
    let changed = [&u, &v]
        .iter()
        .filter(|name| mutated.value(name) != configuration.value(name))
        .count();
    assert_eq!(changed, 1);
    assert!(mutated.is_complete());
}

/// Test a mutation of a choice repairs what the change activates: choosing
/// the other alternative of the choice space activates or drops its
/// variable, and the result is complete.
#[test]
fn space_mutate_repairs_what_the_change_activates() {
    let space = build_choice_space();
    let c = space.decision_order()[0].clone();
    let configuration = with_context(|context| {
        space
            .sample(&mut ScriptedOracle::new([index(1)]), context)
            .map(|recorded| recorded.configuration().cloned().expect("over a space"))
    })
    .expect("choosing b");

    let mutated = with_context(|context| {
        space.mutate(&configuration, &mut Rng::new(0), context, attempts(16))
    })
    .expect("a mutation");

    let mutated = mutated.configuration().expect("a run over a space");
    assert!(mutated.is_complete());
    assert_ne!(mutated.value(&c), configuration.value(&c));
    assert_eq!(mutated.entries().len(), 2);
}

/// Test mutation refuses an incomplete configuration, one of another
/// space, and one with nothing to change.
#[test]
fn space_mutate_refuses_what_it_cannot_mutate() {
    let tiling = build_tiling_space();
    let incomplete = configure(&tiling.space, [(tiling.t.clone(), int(1))]);
    let other = build_tiling_space();
    let foreign = configure(
        &other.space,
        tiling_entries(&other, &(1, other.b.clone(), None)),
    );
    let [name, only] = ["fixed", "only"].map(Identifier::new);
    let fixed = space_of(&name, vec![int_variable(&only, &[7])], Vec::new());
    let single = configure(&fixed, [(only.clone(), int(7))]);

    let results = with_context(|context| {
        [
            tiling
                .space
                .mutate(&incomplete, &mut Rng::new(0), context, attempts(4)),
            tiling
                .space
                .mutate(&foreign, &mut Rng::new(0), context, attempts(4)),
            fixed.mutate(&single, &mut Rng::new(0), context, attempts(4)),
        ]
    });

    let [incomplete, foreign, single] = results;
    assert!(
        matches!(incomplete, Err(TraceError::Incomplete)),
        "{incomplete:?}"
    );
    assert!(
        matches!(foreign, Err(TraceError::OtherSpace)),
        "{foreign:?}"
    );
    assert!(
        matches!(single, Err(TraceError::NothingToMutate)),
        "{single:?}"
    );
}

/// Test a mutation's trace replays into the mutated configuration.
#[test]
fn space_mutate_returns_the_trace_of_the_mutated_configuration() {
    let tiling = build_tiling_space();
    let configuration = configure(
        &tiling.space,
        tiling_entries(&tiling, &(1, tiling.a.clone(), Some(1))),
    );

    let mutated = with_context(|context| {
        tiling
            .space
            .mutate(&configuration, &mut Rng::new(5), context, attempts(16))
    })
    .expect("a mutation");

    let replayed = with_context(|context| tiling.space.replay(mutated.trace(), context))
        .expect("the trace describes the configuration");
    assert_eq!(
        replayed.key(),
        mutated.configuration().expect("a run over a space").key()
    );
}

/// Test enumeration through the exhaustive oracle and the space's own
/// enumeration agree.
#[test]
fn exhaustive_oracle_over_sample_enumerates_the_space() {
    let tiling = build_tiling_space();
    let mut oracle = ExhaustiveOracle::new();
    let mut keys = Vec::new();

    loop {
        let result = with_context(|context| tiling.space.sample(&mut oracle, context));
        match result {
            Ok(recorded) => keys.push(recorded.configuration().expect("over a space").key()),
            Err(error) if ExhaustiveOracle::is_backtrack(&error) => {}
            Err(error) => panic!("sampling failed: {error}"),
        }
        if !oracle.advance() {
            break;
        }
    }

    assert_eq!(keys, tiling_keys(&tiling));
}

/// Test a recorder over a space and the space's sample agree on what a
/// script answers.
#[test]
fn space_sample_agrees_with_a_recorder_over_the_space() {
    let tiling = build_tiling_space();
    let mut recorder = Recorder::over(&tiling.space);
    let script = [index(0), index(1)];

    let by_recorder = with_context(|context| {
        let mut oracle = ScriptedOracle::new(script.clone());
        recorder.decide(&tiling.t, &mut oracle, context)?;
        recorder.decide(&tiling.c, &mut oracle, context)?;
        Ok::<_, TraceError>(recorder.trace())
    })
    .expect("admissible answers");
    let by_sample = with_context(|context| {
        tiling
            .space
            .sample(&mut ScriptedOracle::new(script.clone()), context)
    })
    .expect("admissible answers");

    assert_eq!(&by_recorder, by_sample.trace());
}

/// Test a mutation of an ordering swaps two of its positions.
#[rstest]
#[case::seed_0(0)]
#[case::seed_1(1)]
#[case::seed_2(2)]
fn space_mutate_swaps_two_positions_of_an_ordering(#[case] seed: u64) {
    let [name, order, i, j, k, l] = ["ordered", "order", "i", "j", "k", "l"].map(Identifier::new);
    let space = space_of(
        &name,
        vec![plain_variable(
            &order,
            crate::support::search::permutation_param(&[&i, &j, &k, &l]),
        )],
        Vec::new(),
    );
    let original = Value::Tuple([&i, &j, &k, &l].map(chosen).to_vec());
    let configuration = configure(&space, [(order.clone(), original.clone())]);

    let mutated = with_context(|context| {
        space.mutate(&configuration, &mut Rng::new(seed), context, attempts(16))
    })
    .expect("a mutation");

    let Some(Value::Tuple(elements)) = mutated
        .configuration()
        .expect("a run over a space")
        .value(&order)
        .cloned()
    else {
        panic!("the ordering is a tuple");
    };
    let Value::Tuple(before) = original else {
        panic!("the original is a tuple");
    };
    let moved: Vec<usize> = (0..4)
        .filter(|&position| elements[position] != before[position])
        .collect();
    assert_eq!(moved.len(), 2, "{elements:?}");
    assert_eq!(elements[moved[0]], before[moved[1]]);
    assert_eq!(elements[moved[1]], before[moved[0]]);
}

/// Test a seeded random sample of the tiling space draws the pinned
/// coordinates and leaves its generator where it is pinned to: per seed,
/// the trace's coordinates and the generator's next number after the run.
///
/// The tiling space's forbidden clause makes some draws inadmissible, so
/// the pin covers the redraws as well as the answers.
#[test]
fn space_sample_with_a_random_oracle_follows_the_pinned_stream() {
    let tiling = build_tiling_space();

    let observed: Vec<(Vec<u64>, u64)> = (0..12)
        .map(|seed| {
            let mut oracle = RandomOracle::new(seed);
            let recorded = with_context(|context| tiling.space.sample(&mut oracle, context))
                .expect("the tiling space is sampled");
            let coordinates = recorded
                .trace()
                .coordinates()
                .map(|coordinate| match coordinate {
                    fhy_core::search_space::Coordinate::Index(index) => *index,
                    fhy_core::search_space::Coordinate::Order(_) => u64::MAX,
                })
                .collect();
            (coordinates, oracle.rng().clone().next_u64())
        })
        .collect();

    assert_eq!(
        observed,
        PINNED_TILING_SAMPLES.map(|(c, n)| (c.to_vec(), n)).to_vec()
    );
}

/// What `space_sample_with_a_random_oracle_follows_the_pinned_stream`
/// expects, per seed from 0.
const PINNED_TILING_SAMPLES: [(&[u64], u64); 12] = [
    (&[1, 0], 0x06C4_5D18_8009_454F),
    (&[1, 0], 0x71BB_54D8_D101_B5B9),
    (&[1, 0], 0x58BC_3CB3_7BC7_B2B3),
    (&[0, 1], 0x9CEB_E8A6_D050_DD01),
    (&[0, 1], 0xDBEF_19FC_8E7B_845F),
    (&[0, 1], 0x3B92_D3F0_106B_C147),
    (&[1, 0], 0x0E6C_7D03_72AA_2F46),
    (&[0, 0, 2], 0x953A_EB70_673E_29CB),
    (&[1, 0], 0x5FF7_6408_568A_C010),
    (&[1, 0], 0xC8E9_8CD6_9731_6060),
    (&[0, 1], 0x2187_6E7A_2AEC_4A3D),
    (&[0, 0, 1], 0x812E_6299_272E_6DF0),
];

// ---------------------------------------------------------------------------
// A variable whose bounds enclose no integer
// ---------------------------------------------------------------------------

/// Return the plain variable `name` over the integers at least 5 and at
/// most 3: none.
fn build_empty_variable(name: &Identifier) -> Part<dyn Variable> {
    plain_variable(name, crate::support::search::bounded_param(5, 3))
}

/// Return the space of one choice between `with_empty`, holding the empty
/// variable `empty` and a variable over `{1, 2}`, and `bare`, holding
/// nothing, active only while the variable `switch` over `{1, 2}` is 1:
/// two configurations, `switch = 2` and `switch = 1` choosing `bare`.
fn build_space_around_an_empty_variable(empty: &Identifier) -> Space {
    let [name, switch, choice, with_empty, other, bare] = [
        "around_empty",
        "switch",
        "choice",
        "with_empty",
        "other",
        "bare",
    ]
    .map(Identifier::new);
    let alternative = plain_alternative(
        &with_empty,
        vec![build_empty_variable(empty), int_variable(&other, &[1, 2])],
        Vec::new(),
    );
    Space::new(
        name,
        vec![int_variable(&switch, &[1, 2])],
        vec![choice_of(
            &choice,
            vec![alternative, bare_alternative(&bare)],
        )],
        vec![condition(&choice, [in_set(&switch, [int(1)])])],
        Vec::new(),
    )
    .expect("a valid space")
}

/// Test a variable whose bounds enclose no integer has no configuration,
/// counted in closed form, and beside an unbounded variable in one
/// alternative makes that alternative count none.
#[test]
fn space_cardinality_of_an_empty_interval_is_zero() {
    let alone = space_of(
        &Identifier::new("empty_interval"),
        vec![build_empty_variable(&Identifier::new("n"))],
        Vec::new(),
    );
    let [choice, with_empty, empty, unbounded, bare] =
        ["choice", "with_empty", "empty", "unbounded", "bare"].map(Identifier::new);
    let beside_unbounded = space_of(
        &Identifier::new("empty_beside_unbounded"),
        Vec::new(),
        vec![choice_of(
            &choice,
            vec![
                plain_alternative(
                    &with_empty,
                    vec![
                        build_empty_variable(&empty),
                        plain_variable(&unbounded, crate::support::search_space::natural_param()),
                    ],
                    Vec::new(),
                ),
                bare_alternative(&bare),
            ],
        )],
    );

    let counts = with_context(|context| {
        Ok::<_, TraceError>([
            alone.cardinality(context, 1_000)?,
            beside_unbounded.cardinality(context, 1_000)?,
        ])
    })
    .expect("the counts succeed");

    assert_eq!(
        counts,
        [
            Cardinality::Exact(BigUint::from(0_u8)),
            Cardinality::Exact(BigUint::from(1_u8)),
        ]
    );
}

/// Test an empty variable counted by enumeration is a dead branch: the
/// other branches still count.
#[test]
fn space_cardinality_by_enumeration_skips_an_empty_variable() {
    let space = build_space_around_an_empty_variable(&Identifier::new("n"));

    let cardinality = with_context(|context| space.cardinality(context, 1_000));

    assert_eq!(
        cardinality.expect("the count succeeds"),
        Cardinality::Exact(BigUint::from(2_u8))
    );
}

/// Test sampling an active empty variable is a dead end.
#[test]
fn space_sample_of_an_empty_variable_is_a_dead_end() {
    let n = Identifier::new("n");
    let space = space_of(
        &Identifier::new("empty_interval"),
        vec![build_empty_variable(&n)],
        Vec::new(),
    );

    let result = with_context(|context| space.sample(&mut RandomOracle::new(0), context));

    assert!(
        matches!(&result, Err(TraceError::DeadEnd { decision }) if *decision == n),
        "{result:?}"
    );
}

/// Test enumeration takes an empty variable as a dead branch and yields
/// the configurations of the other branches.
#[test]
fn space_enumerate_skips_an_empty_variable() {
    let n = Identifier::new("n");
    let space = build_space_around_an_empty_variable(&n);

    let configurations: Vec<Configuration> =
        with_context(|context| space.enumerate(context).collect::<Result<_, _>>())
            .expect("the space enumerates");

    assert_eq!(configurations.len(), 2);
    assert!(
        configurations
            .iter()
            .all(|configuration| configuration.is_complete() && configuration.value(&n).is_none())
    );
}

/// Test a uniform draw never takes an empty variable, yet reaches every
/// configuration that leaves it inactive.
#[test]
fn space_sample_uniform_draws_around_an_empty_variable() {
    let space = build_space_around_an_empty_variable(&Identifier::new("n"));

    let keys: HashSet<ConfigurationKey> = (0..32)
        .map(|seed| {
            with_context(|context| space.sample_uniform(&mut Rng::new(seed), context, attempts(64)))
                .expect("the space has configurations")
                .configuration()
                .expect("a run over a space")
                .key()
        })
        .collect();

    assert_eq!(keys.len(), 2);
}

/// Test a uniform draw over a space whose every configuration needs an
/// empty variable exhausts its attempts, as over a closed space.
#[test]
fn space_sample_uniform_exhausts_its_attempts_on_an_empty_variable() {
    let space = space_of(
        &Identifier::new("empty_interval"),
        vec![build_empty_variable(&Identifier::new("n"))],
        Vec::new(),
    );

    let result =
        with_context(|context| space.sample_uniform(&mut Rng::new(0), context, attempts(3)));

    assert!(
        matches!(result, Err(TraceError::AttemptsExhausted { attempts: 3 })),
        "{result:?}"
    );
}
