//! Properties of searching a space, checked against a reference evaluator
//! over generated spaces: enumeration and counting give exactly the
//! complete configurations; a configuration's trace replays into it, also
//! in a relabeled copy of its space; sampling, uniform sampling and
//! mutation give complete configurations of the space; and traces
//! round-trip through JSON. Statistical tests with fixed seeds check that
//! uniform sampling is uniform over the configurations of a conditional
//! space, and that per-step sampling is uniform per step.

use std::collections::HashSet;
use std::num::NonZeroU32;

use fhy_core::constraint::{Constraint, Value};
use fhy_core::identifier::Identifier;
use fhy_core::search_space::{
    Cardinality, Choice, Configuration, ConfigurationKey, RandomOracle, Rng, Space, Trace,
    TraceError,
};
use num_bigint::BigUint;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};

use crate::support::constraint::int;
use crate::support::param::in_set;
use crate::support::search::{
    build_tiling_space, tiling_configurations, tiling_entries, with_context,
};
use crate::support::search_space::{
    bare_alternative, choice_of, chooses, chosen, condition, forbidden, int_variable,
    plain_alternative, try_configure,
};

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------

/// A generated variable: its distinct integer values.
#[derive(Debug, Clone)]
struct VariableModel {
    values: Vec<i64>,
}

/// A generated choice: per alternative, the variable it holds, if any.
#[derive(Debug, Clone)]
struct ChoiceModel {
    alternatives: Vec<Option<VariableModel>>,
}

/// A set constraint over a top-level decision: the decision, by position
/// among the top-level decisions (variables first), and a non-zero mask of
/// the values (for a variable) or alternatives (for a choice) it keeps.
#[derive(Debug, Clone, Copy)]
struct ClauseModel {
    decision: usize,
    mask: u8,
}

/// A generated space: top-level variables and choices, at most one
/// condition on a top-level decision naming an earlier one, and at most one
/// forbidden clause naming two top-level decisions.
#[derive(Debug, Clone)]
struct Model {
    variables: Vec<VariableModel>,
    choices: Vec<ChoiceModel>,
    condition: Option<(usize, ClauseModel)>,
    forbidden: Option<(ClauseModel, ClauseModel)>,
}

impl Model {
    /// Return the number of top-level decisions.
    fn top_count(&self) -> usize {
        self.variables.len() + self.choices.len()
    }

    /// Return the number of values or alternatives of the top-level
    /// decision `decision`.
    fn arity(&self, decision: usize) -> usize {
        if decision < self.variables.len() {
            self.variables[decision].values.len()
        } else {
            self.choices[decision - self.variables.len()]
                .alternatives
                .len()
        }
    }
}

/// The names of a space built from a model.
struct Names {
    /// Each top-level decision's name, variables first.
    tops: Vec<Identifier>,
    /// Per choice, each alternative's name.
    alternatives: Vec<Vec<Identifier>>,
    /// Per choice, per alternative, its variable's name.
    nested: Vec<Vec<Option<Identifier>>>,
}

/// Return the space of `model`, with fresh names, and the names.
///
/// # Panics
///
/// Panics if the space is refused.
fn build_space(model: &Model) -> (Space, Names) {
    let tops: Vec<Identifier> = (0..model.top_count())
        .map(|_| Identifier::new("d"))
        .collect();
    let variables = model
        .variables
        .iter()
        .zip(&tops)
        .map(|(variable, name)| int_variable(name, &variable.values))
        .collect();
    let mut alternatives = Vec::new();
    let mut nested = Vec::new();
    let choices: Vec<Choice> = model
        .choices
        .iter()
        .zip(&tops[model.variables.len()..])
        .map(|(choice, name)| {
            let names: Vec<Identifier> = choice
                .alternatives
                .iter()
                .map(|_| Identifier::new("a"))
                .collect();
            let inner: Vec<Option<Identifier>> = choice
                .alternatives
                .iter()
                .map(|variable| variable.as_ref().map(|_| Identifier::new("v")))
                .collect();
            let parts = choice
                .alternatives
                .iter()
                .zip(&names)
                .zip(&inner)
                .map(
                    |((variable, alternative), variable_name)| match (variable, variable_name) {
                        (Some(variable), Some(variable_name)) => plain_alternative(
                            alternative,
                            vec![int_variable(variable_name, &variable.values)],
                            Vec::new(),
                        ),
                        _ => bare_alternative(alternative),
                    },
                )
                .collect();
            alternatives.push(names);
            nested.push(inner);
            choice_of(name, parts)
        })
        .collect();
    let names = Names {
        tops,
        alternatives,
        nested,
    };
    let conditions = model
        .condition
        .iter()
        .map(|(target, clause)| {
            condition(&names.tops[*target], [build_clause(model, &names, *clause)])
        })
        .collect();
    let clauses = model
        .forbidden
        .iter()
        .map(|(left, right)| {
            forbidden([
                build_clause(model, &names, *left),
                build_clause(model, &names, *right),
            ])
        })
        .collect();
    let space = Space::new(
        Identifier::new("space"),
        variables,
        choices,
        conditions,
        clauses,
    )
    .unwrap_or_else(|error| panic!("refused {model:?}: {error}"));
    (space, names)
}

/// Return the set constraint of `clause`.
fn build_clause(model: &Model, names: &Names, clause: ClauseModel) -> Constraint {
    let kept: Vec<usize> = (0..model.arity(clause.decision))
        .filter(|&position| clause.mask & (1 << position) != 0)
        .collect();
    if clause.decision < model.variables.len() {
        let values = &model.variables[clause.decision].values;
        in_set(
            &names.tops[clause.decision],
            kept.iter().map(|&position| int(values[position])),
        )
    } else {
        let choice = clause.decision - model.variables.len();
        let alternatives: Vec<&Identifier> = kept
            .iter()
            .map(|&position| &names.alternatives[choice][position])
            .collect();
        chooses(&names.tops[clause.decision], &alternatives)
    }
}

// ---------------------------------------------------------------------------
// The reference evaluator
// ---------------------------------------------------------------------------

/// A complete configuration of a model: per top-level decision, its
/// position (a value's or an alternative's) or `None` when inactive, and per
/// choice the position of its nested variable's value, if active.
type Point = (Vec<Option<usize>>, Vec<Option<usize>>);

/// Return whether `clause` holds of the top-level positions `tops`; a
/// clause over an inactive decision does not hold.
fn holds(clause: ClauseModel, tops: &[Option<usize>]) -> bool {
    tops[clause.decision].is_some_and(|position| clause.mask & (1 << position) != 0)
}

/// Return every complete configuration of `model`, each once.
fn compute_reference_points(model: &Model) -> Vec<Point> {
    let mut raw: Vec<Vec<(usize, Option<usize>)>> = vec![Vec::new()];
    for decision in 0..model.top_count() {
        let mut next = Vec::new();
        for prefix in &raw {
            for position in 0..model.arity(decision) {
                let nested_values = if decision < model.variables.len() {
                    vec![None]
                } else {
                    match &model.choices[decision - model.variables.len()].alternatives[position] {
                        Some(variable) => (0..variable.values.len()).map(Some).collect(),
                        None => vec![None],
                    }
                };
                for nested in nested_values {
                    let mut point = prefix.clone();
                    point.push((position, nested));
                    next.push(point);
                }
            }
        }
        raw = next;
    }
    let mut seen = HashSet::new();
    raw.into_iter()
        .filter_map(|assignment| {
            let mut tops: Vec<Option<usize>> = assignment
                .iter()
                .map(|&(position, _)| Some(position))
                .collect();
            let mut nested: Vec<Option<usize>> = assignment[model.variables.len()..]
                .iter()
                .map(|&(_, nested)| nested)
                .collect();
            if let Some((target, clause)) = model.condition {
                if !holds(clause, &tops) {
                    tops[target] = None;
                    if target >= model.variables.len() {
                        nested[target - model.variables.len()] = None;
                    }
                }
            }
            if let Some((left, right)) = model.forbidden {
                if holds(left, &tops) && holds(right, &tops) {
                    return None;
                }
            }
            let point = (tops, nested);
            seen.insert(point.clone()).then_some(point)
        })
        .collect()
}

/// Return the entries of `point` in the space of `model` named `names`.
fn build_entries(model: &Model, names: &Names, (tops, nested): &Point) -> Vec<(Identifier, Value)> {
    let mut entries = Vec::new();
    for (decision, position) in tops.iter().enumerate() {
        let Some(position) = *position else {
            continue;
        };
        if decision < model.variables.len() {
            entries.push((
                names.tops[decision].clone(),
                int(model.variables[decision].values[position]),
            ));
            continue;
        }
        let choice = decision - model.variables.len();
        entries.push((
            names.tops[decision].clone(),
            chosen(&names.alternatives[choice][position]),
        ));
        if let (Some(value), Some(Some(variable_name)), Some(variable)) = (
            nested[choice],
            names.nested[choice].get(position),
            &model.choices[choice].alternatives[position],
        ) {
            entries.push((variable_name.clone(), int(variable.values[value])));
        }
    }
    entries
}

/// Return the reference configurations of `model` in its space `space`.
///
/// # Panics
///
/// Panics if a reference configuration is refused, which would make the
/// reference evaluator disagree with `Configuration::new`.
fn build_reference_configurations(
    model: &Model,
    space: &Space,
    names: &Names,
) -> Vec<Configuration> {
    compute_reference_points(model)
        .iter()
        .map(|point| {
            try_configure(space, build_entries(model, names, point)).unwrap_or_else(|errors| {
                panic!("the reference point {point:?} is refused: {errors}")
            })
        })
        .collect()
}

/// Return the keys of `configurations`.
fn collect_keys(configurations: &[Configuration]) -> HashSet<ConfigurationKey> {
    configurations.iter().map(Configuration::key).collect()
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// Return a strategy of variables: one to three distinct values.
fn generate_variable() -> impl Strategy<Value = VariableModel> {
    prop::sample::subsequence(vec![1_i64, 2, 3], 1..=3).prop_map(|values| VariableModel { values })
}

/// Return a strategy of choices: one to three alternatives, each holding a
/// variable or nothing.
fn generate_choice() -> impl Strategy<Value = ChoiceModel> {
    prop::collection::vec(prop::option::of(generate_variable()), 1..=3)
        .prop_map(|alternatives| ChoiceModel { alternatives })
}

/// Return the non-zero mask `raw` makes over `arity` positions.
fn reduce_mask(raw: u8, arity: usize) -> u8 {
    let full = u8::try_from((1_u16 << arity) - 1).expect("at most three positions");
    raw % full + 1
}

/// Return a strategy of models.
fn generate_model() -> impl Strategy<Value = Model> {
    (
        prop::collection::vec(generate_variable(), 0..=2),
        prop::collection::vec(generate_choice(), 0..=2),
        (prop::bool::weighted(0.75), any::<(usize, usize, u8)>()),
        (prop::bool::weighted(0.75), any::<(usize, usize, u8, u8)>()),
    )
        .prop_map(|(variables, choices, condition, clause)| {
            let mut model = Model {
                variables,
                choices,
                condition: None,
                forbidden: None,
            };
            let count = model.top_count();
            let (has_condition, (target, reference, mask)) = condition;
            if has_condition && count >= 2 {
                let target = 1 + target % (count - 1);
                let decision = reference % target;
                model.condition = Some((
                    target,
                    ClauseModel {
                        decision,
                        mask: reduce_mask(mask, model.arity(decision)),
                    },
                ));
            }
            let (has_clause, (left, right, left_mask, right_mask)) = clause;
            if has_clause && count >= 2 {
                let left = left % count;
                let right = (left + 1 + right % (count - 1)) % count;
                model.forbidden = Some((
                    ClauseModel {
                        decision: left,
                        mask: reduce_mask(left_mask, model.arity(left)),
                    },
                    ClauseModel {
                        decision: right,
                        mask: reduce_mask(right_mask, model.arity(right)),
                    },
                ));
            }
            model
        })
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(Config::with_cases(96))]

    /// Test enumeration yields exactly the reference configurations, each
    /// once, and counting gives their number.
    #[test]
    fn enumeration_and_count_match_the_reference(model in generate_model()) {
        let (space, names) = build_space(&model);
        let expected = collect_keys(&build_reference_configurations(&model, &space, &names));

        let (enumerated, cardinality) = with_context(|context| {
            let enumerated: Vec<Configuration> = space
                .enumerate(context)
                .collect::<Result<_, _>>()
                .expect("a finite space enumerates");
            (enumerated, space.cardinality(context, 1_000_000).expect("the count succeeds"))
        });

        let keys = collect_keys(&enumerated);
        prop_assert_eq!(keys.len(), enumerated.len(), "a configuration was yielded twice");
        prop_assert!(enumerated.iter().all(Configuration::is_complete));
        prop_assert_eq!(&keys, &expected);
        prop_assert_eq!(cardinality, Cardinality::Exact(BigUint::from(expected.len())));
    }

    /// Test every complete configuration's trace replays into it, and into
    /// the corresponding configuration of a relabeled copy of its space.
    #[test]
    fn a_configurations_trace_replays_into_it_and_into_a_relabeled_copy(model in generate_model()) {
        let (space, names) = build_space(&model);
        let (copy, _) = build_space(&model);
        let configurations = build_reference_configurations(&model, &space, &names);

        for configuration in &configurations {
            let (own, relabeled) = with_context(|context| {
                let trace = configuration.trace(context).expect("a finite configuration has a trace");
                (space.replay(&trace, context), copy.replay(&trace, context))
            });

            prop_assert_eq!(own.expect("the trace replays").key(), configuration.key());
            let relabeled = relabeled.expect("the copy has the same shape");
            prop_assert_eq!(relabeled.key(), configuration.key());
            prop_assert_eq!(relabeled.space(), &copy);
        }
    }

    /// Test sampling with a random oracle gives a reference configuration
    /// whose trace replays into it, or reaches a dead end.
    #[test]
    fn random_sampling_gives_a_reference_configuration(model in generate_model(), seed in any::<u64>()) {
        let (space, names) = build_space(&model);
        let expected = collect_keys(&build_reference_configurations(&model, &space, &names));

        let result = with_context(|context| space.sample(&mut RandomOracle::new(seed), context));

        match result {
            Ok(recorded) => {
                let configuration = recorded.configuration().expect("over a space");
                prop_assert!(expected.contains(&configuration.key()));
                let replayed = with_context(|context| space.replay(recorded.trace(), context))
                    .expect("the trace replays");
                prop_assert_eq!(replayed.key(), configuration.key());
            }
            Err(TraceError::DeadEnd { .. }) => {}
            Err(error) => prop_assert!(false, "sampling failed: {error}"),
        }
    }

    /// Test uniform sampling gives a reference configuration, or exhausts
    /// its attempts only when there is none.
    #[test]
    fn uniform_sampling_gives_a_reference_configuration(model in generate_model(), seed in any::<u64>()) {
        let (space, names) = build_space(&model);
        let expected = collect_keys(&build_reference_configurations(&model, &space, &names));

        let result = with_context(|context| {
            space.sample_uniform(&mut Rng::new(seed), context, NonZeroU32::new(10_000).expect("positive"))
        });

        if expected.is_empty() {
            prop_assert!(matches!(result, Err(TraceError::AttemptsExhausted { .. })), "{result:?}");
        } else {
            let recorded = result.expect("a space with configurations is sampled");
            prop_assert!(expected.contains(&recorded.configuration().expect("over a space").key()));
        }
    }

    /// Test a mutation of a complete configuration is another reference
    /// configuration, unless nothing can change.
    #[test]
    fn mutation_gives_another_reference_configuration(
        model in generate_model(),
        pick in any::<prop::sample::Index>(),
        seed in any::<u64>(),
    ) {
        let (space, names) = build_space(&model);
        let configurations = build_reference_configurations(&model, &space, &names);
        prop_assume!(!configurations.is_empty());
        let expected = collect_keys(&configurations);
        let original = &configurations[pick.index(configurations.len())];

        let result = with_context(|context| {
            space.mutate(original, &mut Rng::new(seed), context, NonZeroU32::new(64).expect("positive"))
        });

        match result {
            Ok(recorded) => {
                let key = recorded.configuration().expect("over a space").key();
                prop_assert!(expected.contains(&key));
                prop_assert_ne!(key, original.key());
            }
            Err(TraceError::NothingToMutate) => {}
            Err(error) => prop_assert!(false, "mutation failed: {error}"),
        }
    }

    /// Test a configuration's trace round-trips through JSON.
    #[test]
    fn a_configurations_trace_round_trips_through_json(
        model in generate_model(),
        pick in any::<prop::sample::Index>(),
    ) {
        let (space, names) = build_space(&model);
        let configurations = build_reference_configurations(&model, &space, &names);
        prop_assume!(!configurations.is_empty());
        let configuration = &configurations[pick.index(configurations.len())];
        let trace = with_context(|context| configuration.trace(context)).expect("a trace");

        let text = serde_json::to_string(&trace).expect("the trace serializes");
        let decoded: Trace = serde_json::from_str(&text).expect("the text decodes");

        prop_assert_eq!(decoded, trace);
    }
}

// ---------------------------------------------------------------------------
// Uniformity, with fixed seeds
// ---------------------------------------------------------------------------

/// Return the chi-square statistic of `counts` against the probabilities
/// `expected`, over `total` draws.
fn compute_chi_square(counts: &[u32], expected: &[f64], total: u32) -> f64 {
    counts
        .iter()
        .zip(expected)
        .map(|(&count, &probability)| {
            let expected = probability * f64::from(total);
            let difference = f64::from(count) - expected;
            difference * difference / expected
        })
        .sum()
}

/// Return how often each of the tiling space's five configurations is
/// drawn by `draw` over `total` draws, in their lexicographic order.
fn count_tiling_draws(total: u32, mut draw: impl FnMut(u32) -> ConfigurationKey) -> Vec<u32> {
    let tiling = build_tiling_space();
    let keys: Vec<ConfigurationKey> = tiling_configurations(&tiling)
        .iter()
        .map(|configuration| {
            try_configure(&tiling.space, tiling_entries(&tiling, configuration))
                .expect("a tiling configuration")
                .key()
        })
        .collect();
    let mut counts = vec![0_u32; keys.len()];
    for attempt in 0..total {
        let key = draw(attempt);
        let position = keys
            .iter()
            .position(|known| *known == key)
            .unwrap_or_else(|| panic!("the draw {key:?} is no configuration of the space"));
        counts[position] += 1;
    }
    counts
}

/// Test uniform sampling is uniform over the conditional, forbidding tiling
/// space: over 5000 draws from one generator, the chi-square statistic of
/// the five configurations' counts stays under 18.47, the 0.001 critical
/// value with four degrees of freedom. Its relaxation has eight points, and
/// the configuration whose `x` the condition drops is the image of three.
#[test]
fn uniform_sampling_is_uniform_over_a_conditional_space() {
    let tiling = build_tiling_space();
    let mut rng = Rng::new(31_337);
    let total = 5_000;

    let counts = count_tiling_draws(total, |_| {
        with_context(|context| {
            tiling.space.sample_uniform(
                &mut rng,
                context,
                NonZeroU32::new(1_000).expect("positive"),
            )
        })
        .expect("the space has configurations")
        .configuration()
        .expect("over a space")
        .key()
    });

    let statistic = compute_chi_square(&counts, &[0.2; 5], total);
    assert!(statistic < 18.47, "chi-square {statistic} over {counts:?}");
}

/// Test per-step sampling is uniform per step, not per configuration: on
/// the tiling space it draws `t` evenly, then among the admissible
/// alternatives evenly, then `x` evenly, so the five configurations come
/// with probabilities 1/12, 1/12, 1/12, 1/4 and 1/2; over 6000 draws the
/// chi-square statistic against them stays under 18.47.
#[test]
fn per_step_sampling_is_uniform_per_step() {
    let tiling = build_tiling_space();
    let mut rng = Rng::new(4_242);
    let total = 6_000;

    let counts = count_tiling_draws(total, |_| {
        let mut oracle = RandomOracle::from_rng(rng.split());
        with_context(|context| tiling.space.sample(&mut oracle, context))
            .expect("per-step sampling never dead-ends here")
            .configuration()
            .expect("over a space")
            .key()
    });

    let twelfth = 1.0 / 12.0;
    let statistic = compute_chi_square(&counts, &[twelfth, twelfth, twelfth, 0.25, 0.5], total);
    assert!(statistic < 18.47, "chi-square {statistic} over {counts:?}");
}

// ---------------------------------------------------------------------------
// Non-vacuity guards
// ---------------------------------------------------------------------------

/// The cases each guard draws.
const GUARD_CASES: usize = 256;

/// Return `GUARD_CASES` models drawn by a runner with a fixed seed, so a
/// guard's count is the same on every run.
fn draw_models() -> Vec<Model> {
    let strategy = generate_model();
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

/// Return whether `model`'s condition makes its target inactive in some
/// complete configuration.
fn has_deactivating_condition(model: &Model) -> bool {
    model.condition.is_some_and(|(target, _)| {
        compute_reference_points(model)
            .iter()
            .any(|(tops, _)| tops[target].is_none())
    })
}

/// Return whether `model`'s forbidden clause excludes some point.
fn has_excluding_clause(model: &Model) -> bool {
    model.forbidden.is_some() && {
        let mut open = model.clone();
        open.forbidden = None;
        compute_reference_points(&open).len() > compute_reference_points(model).len()
    }
}

#[test]
fn most_generated_models_have_two_or_more_configurations() {
    let models = draw_models();

    let count = models
        .iter()
        .filter(|model| compute_reference_points(model).len() >= 2)
        .count();

    assert!(
        count * 10 >= GUARD_CASES * 6,
        "{count} of {GUARD_CASES} models have two or more configurations; replay, \
         mutation and the relabeled copy are exercised only on those"
    );
}

#[test]
fn many_generated_conditions_deactivate_their_target() {
    let models = draw_models();

    let count = models
        .iter()
        .filter(|model| has_deactivating_condition(model))
        .count();

    assert!(
        count * 5 >= GUARD_CASES,
        "{count} of {GUARD_CASES} models have a condition that deactivates its target; \
         uniform sampling's acceptance is exercised only on those"
    );
}

#[test]
fn many_generated_clauses_exclude_a_configuration() {
    let models = draw_models();

    let count = models
        .iter()
        .filter(|model| has_excluding_clause(model))
        .count();

    assert!(
        count * 5 >= GUARD_CASES,
        "{count} of {GUARD_CASES} models have a forbidden clause that excludes a \
         configuration; admissibility filtering is exercised only on those"
    );
}

#[test]
fn some_generated_models_have_no_configuration() {
    let models = draw_models();

    let count = models
        .iter()
        .filter(|model| compute_reference_points(model).is_empty())
        .count();

    assert!(
        count >= 1,
        "no model of {GUARD_CASES} has an empty space; uniform sampling's exhaustion \
         is not exercised"
    );
}
