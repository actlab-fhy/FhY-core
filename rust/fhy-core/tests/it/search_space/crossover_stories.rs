//! Tests for `Space::crossover`: identical parents, the documented draw
//! order of the parent picks, disjoint alternatives, a forbidden
//! combination repaired, incomplete parents, a parent of another space,
//! exhausted attempts, and the determinism and replay of a seeded
//! crossover.

use std::num::{NonZeroU32, NonZeroU64};

use fhy_core::constraint::Value;
use fhy_core::identifier::Identifier;
use fhy_core::search_space::{Configuration, Recorded, Rng, Space, TraceError};
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};
use rstest::rstest;

use crate::support::constraint::int;
use crate::support::param::in_set;
use crate::support::search::{
    build_tiling_space, tiling_configurations, tiling_entries, with_context,
};
use crate::support::search_space::{
    choice_of, chosen, configure, forbidden, int_variable, plain_alternative, space_of,
    try_configure,
};

/// Return a positive attempt count.
fn attempts(count: u32) -> NonZeroU32 {
    NonZeroU32::new(count).expect("a positive count")
}

/// Return the crossover of `first` and `second` in `space` drawn from the
/// seed `seed`, with `count` attempts.
fn cross_seeded(
    space: &Space,
    first: &Configuration,
    second: &Configuration,
    seed: u64,
    count: u32,
) -> Result<Recorded, TraceError> {
    with_context(|context| {
        space.crossover(first, second, &mut Rng::new(seed), context, attempts(count))
    })
}

/// Return the crossover of `first` and `second`, which must succeed.
///
/// # Panics
///
/// Panics if the crossover fails or its run is over no space.
fn cross_complete(
    space: &Space,
    first: &Configuration,
    second: &Configuration,
    seed: u64,
) -> Configuration {
    cross_seeded(space, first, second, seed, 32)
        .expect("the parents cross")
        .configuration()
        .expect("a run over a space")
        .clone()
}

/// Return the value `configuration` gives `name`, `None` if it gives none.
fn value_of(configuration: &Configuration, name: &Identifier) -> Option<Value> {
    configuration.value(name).cloned()
}

/// Return the space of the variables `names`, each over `{1, 2, 3}`, and
/// no clause.
fn build_flat_space(names: &[Identifier]) -> Space {
    space_of(
        &Identifier::new("flat"),
        names
            .iter()
            .map(|name| int_variable(name, &[1, 2, 3]))
            .collect(),
        Vec::new(),
    )
}

/// Return the first `count` draws of `Rng::new(seed).below(2)`.
fn draw_picks(seed: u64, count: usize) -> Vec<u64> {
    let mut rng = Rng::new(seed);
    let two = NonZeroU64::new(2).expect("two is positive");
    (0..count).map(|_| rng.below(two)).collect()
}

/// Return the space of `x` and `y` over `{1, 2}` forbidding `x = 1` with
/// `y = 1`.
fn build_clause_space(x: &Identifier, y: &Identifier) -> Space {
    Space::new(
        Identifier::new("clause"),
        vec![int_variable(x, &[1, 2]), int_variable(y, &[1, 2])],
        Vec::new(),
        Vec::new(),
        vec![forbidden([in_set(x, [int(1)]), in_set(y, [int(1)])])],
    )
    .expect("the clause space is valid")
}

/// Return the space of `x` over `{1}` whose only value is forbidden: it
/// has no complete configuration.
fn build_closed_space(x: &Identifier) -> Space {
    Space::new(
        Identifier::new("closed"),
        vec![int_variable(x, &[1])],
        Vec::new(),
        Vec::new(),
        vec![forbidden([in_set(x, [int(1)])])],
    )
    .expect("the closed space is valid")
}

// ---------------------------------------------------------------------------
// Parents
// ---------------------------------------------------------------------------

/// Test crossing a configuration with itself gives it back, whatever the
/// seed.
#[rstest]
#[case::seed_0(0)]
#[case::seed_1(1)]
#[case::seed_2(2)]
#[case::seed_99(99)]
fn space_crossover_of_identical_parents_gives_the_parent(#[case] seed: u64) {
    let tiling = build_tiling_space();
    let parent = configure(
        &tiling.space,
        tiling_entries(&tiling, &tiling_configurations(&tiling)[1]),
    );

    let child = cross_complete(&tiling.space, &parent, &parent, seed);

    assert_eq!(child.key(), parent.key());
    assert_eq!(
        child.entries().collect::<Vec<_>>(),
        parent.entries().collect::<Vec<_>>()
    );
}

/// Test each decision takes the parent its pick draws, the picks being
/// `rng.below(2)` per decision in canonical order: the first parent on 0.
#[rstest]
#[case::seed_0(0)]
#[case::seed_1(1)]
#[case::seed_2(2)]
#[case::seed_3(3)]
#[case::seed_4(4)]
#[case::seed_5(5)]
#[case::seed_6(6)]
#[case::seed_7(7)]
fn space_crossover_picks_each_parent_by_the_documented_draws(#[case] seed: u64) {
    let names = ["x", "y", "z"].map(Identifier::new);
    let space = build_flat_space(&names);
    let first = configure(&space, names.iter().cloned().zip([1, 1, 1].map(int)));
    let second = configure(&space, names.iter().cloned().zip([2, 2, 2].map(int)));
    let picks = draw_picks(seed, 3);

    let child = cross_complete(&space, &first, &second, seed);

    let expected: Vec<Option<Value>> = picks
        .iter()
        .map(|&pick| Some(int(if pick == 0 { 1 } else { 2 })))
        .collect();
    let actual: Vec<Option<Value>> = names.iter().map(|name| value_of(&child, name)).collect();
    assert_eq!(actual, expected);
}

/// Test parents choosing different alternatives give a complete child
/// whose chosen alternative's variable holds the value of the parent that
/// chose it.
#[test]
fn space_crossover_of_disjoint_alternatives_keeps_each_alternatives_own_value() {
    let [name, layout, left, right, left_value, right_value] =
        ["disjoint", "c", "a", "b", "x", "y"].map(Identifier::new);
    let choice = choice_of(
        &layout,
        vec![
            plain_alternative(
                &left,
                vec![int_variable(&left_value, &[1, 2, 3])],
                Vec::new(),
            ),
            plain_alternative(
                &right,
                vec![int_variable(&right_value, &[1, 2, 3])],
                Vec::new(),
            ),
        ],
    );
    let space = space_of(&name, Vec::new(), vec![choice]);
    let first = configure(
        &space,
        [
            (layout.clone(), chosen(&left)),
            (left_value.clone(), int(1)),
        ],
    );
    let second = configure(
        &space,
        [
            (layout.clone(), chosen(&right)),
            (right_value.clone(), int(3)),
        ],
    );

    let outcomes: Vec<(Configuration, [Option<Value>; 3])> = (0..32)
        .map(|seed| {
            let child = cross_complete(&space, &first, &second, seed);
            let values = [&layout, &left_value, &right_value].map(|name| value_of(&child, name));
            (child, values)
        })
        .collect();

    let took_first = [Some(chosen(&left)), Some(int(1)), None];
    let took_second = [Some(chosen(&right)), None, Some(int(3))];
    assert!(
        outcomes.iter().all(|(child, values)| child.is_complete()
            && (*values == took_first || *values == took_second)),
        "{:?}",
        outcomes
            .iter()
            .map(|(_, values)| values)
            .collect::<Vec<_>>()
    );
    assert!(outcomes.iter().any(|(_, values)| *values == took_first));
    assert!(outcomes.iter().any(|(_, values)| *values == took_second));
}

/// Test a combination the space forbids, which the picks can make, is
/// repaired: no child is the forbidden one, every child is complete, and
/// both parents' own combinations still occur.
#[test]
fn space_crossover_repairs_a_forbidden_combination() {
    let [x, y] = ["x", "y"].map(Identifier::new);
    let space = build_clause_space(&x, &y);
    let first = configure(&space, [(x.clone(), int(1)), (y.clone(), int(2))]);
    let second = configure(&space, [(x.clone(), int(2)), (y.clone(), int(1))]);

    let children: Vec<(bool, [Option<Value>; 2])> = (0..64)
        .map(|seed| {
            let child = cross_complete(&space, &first, &second, seed);
            (
                child.is_complete(),
                [&x, &y].map(|name| value_of(&child, name)),
            )
        })
        .collect();

    let forbidden_pair = [Some(int(1)), Some(int(1))];
    assert!(children.iter().all(|(complete, _)| *complete));
    assert!(children.iter().all(|(_, values)| *values != forbidden_pair));
    assert!(
        children
            .iter()
            .any(|(_, values)| *values == [Some(int(1)), Some(int(2))])
    );
    assert!(
        children
            .iter()
            .any(|(_, values)| *values == [Some(int(2)), Some(int(1))])
    );
}

/// Test parents holding only some entries give a complete child that
/// inherits the entries it can.
#[test]
fn space_crossover_of_incomplete_parents_inherits_what_a_parent_assigns() {
    let names = ["x", "y", "z"].map(Identifier::new);
    let space = build_flat_space(&names);
    let partial = configure(&space, [(names[0].clone(), int(1))]);
    let empty = configure(&space, []);

    let children: Vec<Configuration> = (0..16)
        .flat_map(|seed| {
            [
                cross_complete(&space, &partial, &empty, seed),
                cross_complete(&space, &empty, &partial, seed),
            ]
        })
        .collect();

    assert!(children.iter().all(Configuration::is_complete));
    assert!(
        children
            .iter()
            .all(|child| value_of(child, &names[0]) == Some(int(1)))
    );
}

/// Test a decision neither parent assigns is drawn from its admissible
/// values only, and from all of them: here `z`, whose value 3 is
/// forbidden, takes both 1 and 2 and never 3, and the unconstrained `y`
/// takes all three.
#[test]
fn space_crossover_draws_an_unassigned_decision_among_its_admissible_values() {
    let names = ["x", "y", "z"].map(Identifier::new);
    let space = Space::new(
        Identifier::new("drawn"),
        names
            .iter()
            .map(|name| int_variable(name, &[1, 2, 3]))
            .collect(),
        Vec::new(),
        Vec::new(),
        vec![forbidden([in_set(&names[2], [int(3)])])],
    )
    .expect("the space is valid");
    let partial = configure(&space, [(names[0].clone(), int(1))]);
    let empty = configure(&space, []);

    let children: Vec<Configuration> = (0..64)
        .map(|seed| cross_complete(&space, &partial, &empty, seed))
        .collect();

    let seen = |name: &Identifier, value: i64| {
        children
            .iter()
            .any(|child| value_of(child, name) == Some(int(value)))
    };
    assert!(seen(&names[1], 1) && seen(&names[1], 2) && seen(&names[1], 3));
    assert!(seen(&names[2], 1) && seen(&names[2], 2));
    assert!(!seen(&names[2], 3));
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Test a parent of another space is refused, in either position.
#[rstest]
#[case::first_parent_foreign(true)]
#[case::second_parent_foreign(false)]
fn space_crossover_refuses_a_parent_of_another_space(#[case] first_is_foreign: bool) {
    let tiling = build_tiling_space();
    let other = build_tiling_space();
    let entries = &tiling_configurations(&tiling)[0];
    let own = configure(&tiling.space, tiling_entries(&tiling, entries));
    let foreign = configure(
        &other.space,
        tiling_entries(&other, &tiling_configurations(&other)[0]),
    );
    let (first, second) = if first_is_foreign {
        (&foreign, &own)
    } else {
        (&own, &foreign)
    };

    let result = cross_seeded(&tiling.space, first, second, 0, 1);

    let Err(TraceError::OtherSpace) = result else {
        panic!("expected OtherSpace, got {result:?}")
    };
}

/// Test a space with no complete configuration exhausts the attempts
/// given, and names them.
#[rstest]
#[case::one_attempt(1)]
#[case::three_attempts(3)]
fn space_crossover_exhausts_its_attempts_on_a_closed_space(#[case] count: u32) {
    let x = Identifier::new("x");
    let space = build_closed_space(&x);
    let empty = configure(&space, []);

    let result = cross_seeded(&space, &empty, &empty, 0, count);

    assert!(
        matches!(result, Err(TraceError::AttemptsExhausted { attempts }) if attempts == count),
        "{result:?}"
    );
}

// ---------------------------------------------------------------------------
// Seeded results
// ---------------------------------------------------------------------------

/// Test the same seed and parents give the same configuration and trace.
#[rstest]
#[case::seed_0(0)]
#[case::seed_5(5)]
#[case::seed_41(41)]
fn space_crossover_is_deterministic_for_a_seed(#[case] seed: u64) {
    let tiling = build_tiling_space();
    let first = configure(
        &tiling.space,
        tiling_entries(&tiling, &tiling_configurations(&tiling)[2]),
    );
    let second = configure(
        &tiling.space,
        tiling_entries(&tiling, &tiling_configurations(&tiling)[3]),
    );

    let once = cross_seeded(&tiling.space, &first, &second, seed, 16).expect("the parents cross");
    let again = cross_seeded(&tiling.space, &first, &second, seed, 16).expect("the parents cross");

    assert_eq!(once.trace(), again.trace());
    assert_eq!(
        once.configuration().expect("a run over a space").key(),
        again.configuration().expect("a run over a space").key()
    );
}

/// Test the trace a crossover returns replays into the configuration it
/// returns.
#[rstest]
#[case::seed_0(0)]
#[case::seed_5(5)]
#[case::seed_41(41)]
fn space_crossover_returns_a_trace_that_replays_into_its_configuration(#[case] seed: u64) {
    let tiling = build_tiling_space();
    let first = configure(
        &tiling.space,
        tiling_entries(&tiling, &tiling_configurations(&tiling)[0]),
    );
    let second = configure(
        &tiling.space,
        tiling_entries(&tiling, &tiling_configurations(&tiling)[4]),
    );

    let crossed =
        cross_seeded(&tiling.space, &first, &second, seed, 16).expect("the parents cross");

    let replayed = with_context(|context| tiling.space.replay(crossed.trace(), context))
        .expect("the trace describes the configuration");
    assert_eq!(
        replayed.key(),
        crossed.configuration().expect("a run over a space").key()
    );
}

// ---------------------------------------------------------------------------
// The property
// ---------------------------------------------------------------------------

/// A generated clause: it forbids the configurations giving the variable
/// at `first` a value of `first_values` and the one at `second` a value of
/// `second_values`.
#[derive(Debug, Clone)]
struct ClauseModel {
    first: usize,
    first_values: Vec<i64>,
    second: usize,
    second_values: Vec<i64>,
}

/// A generated space: two to four top-level variables over small integer
/// domains, and at most one clause that leaves a configuration.
#[derive(Debug, Clone)]
struct Model {
    domains: Vec<Vec<i64>>,
    clause: Option<ClauseModel>,
}

/// A generated crossover: a space, the seeds of its two parents' random
/// oracles, and the seed of the crossover.
#[derive(Debug, Clone)]
struct Scenario {
    model: Model,
    first_seed: u64,
    second_seed: u64,
    crossover_seed: u64,
}

/// Return the values of `domain` whose positions `mask` sets, or its first
/// value if it sets none.
fn select_values(domain: &[i64], mask: u8) -> Vec<i64> {
    let chosen: Vec<i64> = domain
        .iter()
        .enumerate()
        .filter(|(position, _)| mask >> position & 1 == 1)
        .map(|(_, &value)| value)
        .collect();
    if chosen.is_empty() {
        vec![domain[0]]
    } else {
        chosen
    }
}

/// Return whether the configuration giving the variables `point`, by
/// position, is forbidden by `clause`.
fn is_excluded(clause: &ClauseModel, point: &[i64]) -> bool {
    clause.first_values.contains(&point[clause.first])
        && clause.second_values.contains(&point[clause.second])
}

/// Return the cartesian product of `domains`.
fn list_points(domains: &[Vec<i64>]) -> Vec<Vec<i64>> {
    domains.iter().fold(vec![Vec::new()], |points, domain| {
        points
            .iter()
            .flat_map(|point| {
                domain.iter().map(|&value| {
                    let mut extended = point.clone();
                    extended.push(value);
                    extended
                })
            })
            .collect()
    })
}

/// Return the model of `domains` and the raw clause, with the clause
/// dropped if it would leave no configuration.
fn build_model(domains: Vec<Vec<i64>>, raw: Option<(usize, usize, u8, u8)>) -> Model {
    let count = domains.len();
    let clause = raw.map(|(first, offset, first_mask, second_mask)| {
        let second = (first + offset) % count;
        ClauseModel {
            first,
            first_values: select_values(&domains[first], first_mask),
            second,
            second_values: select_values(&domains[second], second_mask),
        }
    });
    let clause = clause.filter(|clause| {
        list_points(&domains)
            .iter()
            .any(|point| !is_excluded(clause, point))
    });
    Model { domains, clause }
}

/// Return the strategy of generated crossovers.
fn generate_scenario() -> impl Strategy<Value = Scenario> {
    let domains = prop::collection::vec(prop::sample::subsequence(vec![1, 2, 3, 4], 2..=4), 2..=4);
    let model = domains
        .prop_flat_map(|domains| {
            let count = domains.len();
            let clause = proptest::option::of((0..count, 1..count, any::<u8>(), any::<u8>()));
            (Just(domains), clause)
        })
        .prop_map(|(domains, raw)| build_model(domains, raw));
    (model, any::<u64>(), any::<u64>(), any::<u64>()).prop_map(
        |(model, first_seed, second_seed, crossover_seed)| Scenario {
            model,
            first_seed,
            second_seed,
            crossover_seed,
        },
    )
}

/// Return the space of `model`, with fresh names, and the names.
///
/// # Panics
///
/// Panics if the space is refused.
fn build_model_space(model: &Model) -> (Space, Vec<Identifier>) {
    let names: Vec<Identifier> = model.domains.iter().map(|_| Identifier::new("v")).collect();
    let clauses = model
        .clause
        .iter()
        .map(|clause| {
            forbidden([
                in_set(
                    &names[clause.first],
                    clause.first_values.iter().copied().map(int),
                ),
                in_set(
                    &names[clause.second],
                    clause.second_values.iter().copied().map(int),
                ),
            ])
        })
        .collect();
    let space = Space::new(
        Identifier::new("generated"),
        names
            .iter()
            .zip(&model.domains)
            .map(|(name, domain)| int_variable(name, domain))
            .collect(),
        Vec::new(),
        Vec::new(),
        clauses,
    )
    .expect("the generated space is valid");
    (space, names)
}

/// Return a complete configuration of `space` drawn uniformly with a
/// generator seeded with `seed`: not by a random oracle, which can reach
/// a dead end the clause makes.
///
/// # Panics
///
/// Panics if the space has no configuration.
fn sample_parent(space: &Space, seed: u64) -> Configuration {
    with_context(|context| space.sample_uniform(&mut Rng::new(seed), context, attempts(1_000)))
        .expect("the generated space has a configuration")
        .configuration()
        .expect("a run over a space")
        .clone()
}

proptest! {
    /// Test a crossover of two complete parents is complete, accepted by
    /// `Configuration::new`, and, with no clause, takes each variable's
    /// value from a parent.
    #[test]
    fn space_crossover_gives_a_valid_complete_child(scenario in generate_scenario()) {
        let (space, names) = build_model_space(&scenario.model);
        let first = sample_parent(&space, scenario.first_seed);
        let second = sample_parent(&space, scenario.second_seed);

        let crossed = cross_seeded(&space, &first, &second, scenario.crossover_seed, 16)
            .expect("two complete parents cross");

        let child = crossed.configuration().expect("a run over a space");
        prop_assert!(child.is_complete());
        let entries: Vec<(Identifier, Value)> = child
            .entries()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        let accepted = try_configure(&space, entries).expect("the child is a valid configuration");
        prop_assert_eq!(accepted.key(), child.key());
        if scenario.model.clause.is_none() {
            for name in &names {
                let value = value_of(child, name);
                prop_assert!(
                    value == value_of(&first, name) || value == value_of(&second, name),
                    "{:?} is neither parent's value of {:?}", value, name
                );
            }
        }
    }
}

/// The cases each guard draws.
const GUARD_CASES: usize = 256;

/// Return `GUARD_CASES` scenarios drawn by a runner with a fixed seed, so
/// a guard's count is the same on every run.
fn draw_scenarios() -> Vec<Scenario> {
    let strategy = generate_scenario();
    let mut runner = TestRunner::new_with_rng(
        Config::default(),
        TestRng::deterministic_rng(RngAlgorithm::ChaCha),
    );
    (0..GUARD_CASES)
        .map(|_| {
            strategy
                .new_tree(&mut runner)
                .expect("the strategy draws")
                .current()
        })
        .collect()
}

/// Return the number of variables the two parents of `scenario` differ on.
fn count_differences(scenario: &Scenario) -> usize {
    let (space, names) = build_model_space(&scenario.model);
    let first = sample_parent(&space, scenario.first_seed);
    let second = sample_parent(&space, scenario.second_seed);
    names
        .iter()
        .filter(|name| value_of(&first, name) != value_of(&second, name))
        .count()
}

/// Test the strategy behind the crossover property (a guard of the
/// strategy, not of `crossover`) draws parents that differ on two or more
/// variables often enough for the picks to matter.
#[test]
fn crossover_scenarios_often_have_parents_differing_on_two_variables() {
    let scenarios = draw_scenarios();

    let count = scenarios
        .iter()
        .filter(|scenario| count_differences(scenario) >= 2)
        .count();

    assert!(
        count * 10 >= GUARD_CASES * 4,
        "{count} of {GUARD_CASES} scenarios have parents differing on two or more \
         variables; the picks are exercised only on those"
    );
}

/// Test the strategy draws spaces with a forbidden clause, and spaces
/// without one, each often enough (a guard of the strategy).
#[test]
fn crossover_scenarios_often_have_a_clause_and_often_have_none() {
    let scenarios = draw_scenarios();

    let with_clause = scenarios
        .iter()
        .filter(|scenario| scenario.model.clause.is_some())
        .count();

    assert!(
        with_clause * 4 >= GUARD_CASES,
        "{with_clause} of {GUARD_CASES} scenarios have a clause; repair is exercised only on those"
    );
    assert!(
        (GUARD_CASES - with_clause) * 4 >= GUARD_CASES,
        "{} of {GUARD_CASES} scenarios have no clause; the from-a-parent check runs only on those",
        GUARD_CASES - with_clause
    );
}

/// Test the strategy draws clauses that exclude a combination the parents'
/// picks can reach: a point of the product that the clause forbids (a
/// guard of the strategy).
#[test]
fn crossover_scenarios_often_have_a_clause_that_excludes_a_point() {
    let scenarios = draw_scenarios();

    let count = scenarios
        .iter()
        .filter(|scenario| {
            scenario.model.clause.as_ref().is_some_and(|clause| {
                list_points(&scenario.model.domains)
                    .iter()
                    .any(|point| is_excluded(clause, point))
            })
        })
        .count();

    assert!(
        count * 5 >= GUARD_CASES,
        "{count} of {GUARD_CASES} scenarios have a clause that excludes a point"
    );
}
