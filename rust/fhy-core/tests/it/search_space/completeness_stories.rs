//! Tests for per-choice completeness and cheap one-entry building:
//! `Configuration::is_complete_under`, and `with_entry` and `with_entries`
//! not checking the values a configuration holds again, with a property
//! that building one entry at a time agrees with `Configuration::new`.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fhy_core::constraint::{
    Bindings, Constraint, ConstraintContext, CustomConstraint, Outcome, Value,
};
use fhy_core::expression::Expression;
use fhy_core::foreign::{BoxError, ForeignPart, Part};
use fhy_core::identifier::Identifier;
use fhy_core::param::{IntegerDomain, Param, ParamContext, ParamDomain, Sign, ZeroInclusion};
use fhy_core::search_space::{Configuration, ConfigurationError, ConfigurationErrors, Space};
use fhy_core::solver::Solver;
use fhy_core::term::AlphaRenaming;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};
use rstest::rstest;

use crate::support::constraint::int;
use crate::support::param::{in_set, literal, reference};
use crate::support::search::with_context;
use crate::support::search_space::{
    bare_alternative, choice_of, chooses, chosen, condition, configure, forbidden, ground_solver,
    int_variable, odd_natural_param, plain_alternative, plain_variable, space_of, try_configure,
};

// -- helpers ----------------------------------------------------------------

/// A value an entry of the status space gives: an integer, or the name of
/// the alternative a choice chooses.
#[derive(Debug, Clone, Copy)]
enum Pick {
    Int(i64),
    Alt(&'static str),
}

/// The status space: the variables `v1`, `v2` over `{1, 2}`, `v2` active
/// only while `v1` is 1; the choice `c1` among `a1` holding `x1`, `a2`
/// holding `x2`, and the bare `a3`; and the choice `outer` among `deep`,
/// holding the choice `inner` among `leaf`, holding `w`, and the bare
/// `hollow`, and `flat`, holding `f`. Every variable ranges over `{1, 2}`.
struct Status {
    space: Space,
    names: HashMap<&'static str, Identifier>,
}

impl Status {
    /// Return the status space, its names fresh.
    fn new() -> Self {
        let labels = [
            "status", "v1", "v2", "c1", "a1", "a2", "a3", "x1", "x2", "outer", "deep", "inner",
            "leaf", "hollow", "w", "flat", "f",
        ];
        let names: HashMap<&'static str, Identifier> = labels
            .into_iter()
            .map(|label| (label, Identifier::new(label)))
            .collect();
        let n = |label: &str| &names[label];
        let inner = choice_of(
            n("inner"),
            vec![
                plain_alternative(n("leaf"), vec![int_variable(n("w"), &[1, 2])], Vec::new()),
                bare_alternative(n("hollow")),
            ],
        );
        let space = Space::new(
            n("status").clone(),
            vec![
                int_variable(n("v1"), &[1, 2]),
                int_variable(n("v2"), &[1, 2]),
            ],
            vec![
                choice_of(
                    n("c1"),
                    vec![
                        plain_alternative(
                            n("a1"),
                            vec![int_variable(n("x1"), &[1, 2])],
                            Vec::new(),
                        ),
                        plain_alternative(
                            n("a2"),
                            vec![int_variable(n("x2"), &[1, 2])],
                            Vec::new(),
                        ),
                        bare_alternative(n("a3")),
                    ],
                ),
                choice_of(
                    n("outer"),
                    vec![
                        plain_alternative(n("deep"), Vec::new(), vec![inner]),
                        plain_alternative(
                            n("flat"),
                            vec![int_variable(n("f"), &[1, 2])],
                            Vec::new(),
                        ),
                    ],
                ),
            ],
            vec![condition(n("v2"), [in_set(n("v1"), [int(1)])])],
            Vec::new(),
        )
        .expect("the status space is valid");
        Self { space, names }
    }

    /// Return the name labelled `label`.
    fn name(&self, label: &str) -> &Identifier {
        &self.names[label]
    }

    /// Return the value `pick` stands for.
    fn value_of(&self, pick: Pick) -> Value {
        match pick {
            Pick::Int(number) => int(number),
            Pick::Alt(label) => chosen(self.name(label)),
        }
    }

    /// Return the configuration of the space with `picks`.
    fn configure(&self, picks: &[(&str, Pick)]) -> Configuration {
        configure(
            &self.space,
            picks
                .iter()
                .map(|&(label, pick)| (self.name(label).clone(), self.value_of(pick))),
        )
    }
}

// -- is_complete_under ------------------------------------------------------

/// Test a variable is complete under itself when it is assigned or
/// inactive, and not when it is active and unassigned.
#[rstest]
#[case::unassigned_active_variable(&[], "v1", Some(false))]
#[case::assigned_variable(&[("v1", Pick::Int(1))], "v1", Some(true))]
#[case::unassigned_variable_whose_condition_holds(&[("v1", Pick::Int(1))], "v2", Some(false))]
#[case::variable_whose_condition_fails(&[("v1", Pick::Int(2))], "v2", Some(true))]
#[case::variable_under_an_alternative_not_chosen(&[("c1", Pick::Alt("a2"))], "x1", Some(true))]
fn is_complete_under_a_variable_is_whether_it_is_assigned_or_inactive(
    #[case] picks: &[(&str, Pick)],
    #[case] label: &str,
    #[case] expected: Option<bool>,
) {
    let status = Status::new();
    let configuration = status.configure(picks);

    let complete = configuration.is_complete_under(status.name(label));

    assert_eq!(complete, expected);
}

/// Test a choice is complete under itself when it is assigned and every
/// decision under its chosen alternative is assigned or inactive.
#[rstest]
#[case::unassigned_choice(&[], "c1", Some(false))]
#[case::chosen_with_an_unassigned_variable(&[("c1", Pick::Alt("a1"))], "c1", Some(false))]
#[case::chosen_with_its_variable_assigned(
    &[("c1", Pick::Alt("a1")), ("x1", Pick::Int(1))], "c1", Some(true)
)]
#[case::variable_of_another_alternative_does_not_count(
    &[("c1", Pick::Alt("a2")), ("x2", Pick::Int(2))], "c1", Some(true)
)]
#[case::chosen_alternative_holding_nothing(&[("c1", Pick::Alt("a3"))], "c1", Some(true))]
fn is_complete_under_a_choice_is_whether_its_subtree_is_decided(
    #[case] picks: &[(&str, Pick)],
    #[case] label: &str,
    #[case] expected: Option<bool>,
) {
    let status = Status::new();
    let configuration = status.configure(picks);

    let complete = configuration.is_complete_under(status.name(label));

    assert_eq!(complete, expected);
}

/// Test nested choices are complete under themselves only when the
/// variable at the bottom of the chosen path is assigned, at every depth.
#[rstest]
#[case::outer_chosen_inner_unassigned(
    &[("outer", Pick::Alt("deep"))], "outer", Some(false)
)]
#[case::outer_chosen_inner_unassigned_inner(
    &[("outer", Pick::Alt("deep"))], "inner", Some(false)
)]
#[case::both_chosen_leaf_unassigned_outer(
    &[("outer", Pick::Alt("deep")), ("inner", Pick::Alt("leaf"))], "outer", Some(false)
)]
#[case::both_chosen_leaf_unassigned_inner(
    &[("outer", Pick::Alt("deep")), ("inner", Pick::Alt("leaf"))], "inner", Some(false)
)]
#[case::leaf_assigned_outer(
    &[("outer", Pick::Alt("deep")), ("inner", Pick::Alt("leaf")), ("w", Pick::Int(1))],
    "outer", Some(true)
)]
#[case::leaf_assigned_inner(
    &[("outer", Pick::Alt("deep")), ("inner", Pick::Alt("leaf")), ("w", Pick::Int(1))],
    "inner", Some(true)
)]
#[case::inner_chooses_the_empty_alternative_outer(
    &[("outer", Pick::Alt("deep")), ("inner", Pick::Alt("hollow"))], "outer", Some(true)
)]
#[case::inner_chooses_the_empty_alternative_inner(
    &[("outer", Pick::Alt("deep")), ("inner", Pick::Alt("hollow"))], "inner", Some(true)
)]
#[case::inner_under_the_alternative_not_chosen(
    &[("outer", Pick::Alt("flat"))], "inner", Some(true)
)]
#[case::other_alternative_chosen_with_its_variable_open(
    &[("outer", Pick::Alt("flat"))], "outer", Some(false)
)]
fn is_complete_under_a_nested_choice_looks_at_every_depth(
    #[case] picks: &[(&str, Pick)],
    #[case] label: &str,
    #[case] expected: Option<bool>,
) {
    let status = Status::new();
    let configuration = status.configure(picks);

    let complete = configuration.is_complete_under(status.name(label));

    assert_eq!(complete, expected);
}

/// Test a name that is no decision of the space, an unknown one or an
/// alternative's, has no answer.
#[rstest]
#[case::unknown_name(Identifier::new("ghost"))]
#[case::alternative_name(Identifier::new("a1"))]
fn is_complete_under_a_name_that_is_no_decision_is_none(#[case] stranger: Identifier) {
    let status = Status::new();
    let configuration = status.configure(&[("c1", Pick::Alt("a1"))]);

    let complete = configuration.is_complete_under(&stranger);
    let alternative = configuration.is_complete_under(status.name("a1"));

    assert_eq!(complete, None);
    assert_eq!(alternative, None);
}

/// Test on a complete configuration every decision is complete under
/// itself.
#[test]
fn is_complete_under_every_decision_of_a_complete_configuration_is_true() {
    let status = Status::new();
    let configuration = status.configure(&[
        ("v1", Pick::Int(1)),
        ("v2", Pick::Int(1)),
        ("c1", Pick::Alt("a1")),
        ("x1", Pick::Int(1)),
        ("outer", Pick::Alt("deep")),
        ("inner", Pick::Alt("leaf")),
        ("w", Pick::Int(2)),
    ]);

    let answers: Vec<Option<bool>> = status
        .space
        .decisions()
        .map(|decision| configuration.is_complete_under(decision.name()))
        .collect();

    assert!(configuration.is_complete());
    assert_eq!(answers, vec![Some(true); answers.len()]);
}

// -- with_entry does not check held values again ------------------------------

/// Return the space of the variable `odd`, over the odd naturals, and the
/// variable `other` over `{1, 2}`.
fn build_odd_space(odd: &Identifier, other: &Identifier) -> Space {
    space_of(
        &Identifier::new("oddity"),
        vec![
            plain_variable(odd, odd_natural_param()),
            int_variable(other, &[1, 2]),
        ],
        Vec::new(),
    )
}

/// Test `with_entry` does not check the values the configuration holds
/// again: a value only a simplifier could check, held from a context that
/// had one, stays when another decision is given a value under a context
/// that has none.
#[test]
fn with_entry_keeps_a_held_value_a_weaker_context_could_not_check() {
    let [odd, other] = ["odd", "other"].map(Identifier::new);
    let space = build_odd_space(&odd, &other);
    let held = configure(&space, [(odd.clone(), int(3))]);
    let weaker = Solver::new();

    let extended = held
        .with_entry(other.clone(), int(1), &ParamContext::new(&weaker))
        .expect("the held value is not checked again");

    assert_eq!(extended.value(&odd), Some(&int(3)));
    assert_eq!(extended.value(&other), Some(&int(1)));
}

/// Test `with_entries` does not check the held values again either.
#[test]
fn with_entries_keeps_a_held_value_a_weaker_context_could_not_check() {
    let [odd, other] = ["odd", "other"].map(Identifier::new);
    let space = build_odd_space(&odd, &other);
    let held = configure(&space, [(odd.clone(), int(3))]);
    let weaker = Solver::new();

    let extended = held
        .with_entries([(other.clone(), int(2))], &ParamContext::new(&weaker))
        .expect("the held value is not checked again");

    assert_eq!(extended.value(&odd), Some(&int(3)));
    assert_eq!(extended.value(&other), Some(&int(2)));
}

/// Guard: the new entry's own value is still checked, so under the weaker
/// context a value for the odd variable is refused as an assignment error.
/// It passes before and after the change.
#[test]
fn with_entry_guard_still_checks_the_new_value() {
    let [odd, other] = ["odd", "other"].map(Identifier::new);
    let space = build_odd_space(&odd, &other);
    let held = configure(&space, []);
    let weaker = Solver::new();

    let result = held.with_entry(odd.clone(), int(3), &ParamContext::new(&weaker));

    let Err(errors) = result else {
        panic!("expected a refusal, got {result:?}");
    };
    let [ConfigurationError::Assignment { variable, .. }] = errors.errors() else {
        panic!("expected one assignment error, got {errors:?}");
    };
    assert_eq!(variable, &odd);
}

// -- work stays linear --------------------------------------------------------

/// A custom constraint that every value satisfies and that counts its
/// evaluations.
#[derive(Debug)]
struct CountingConstraint {
    variable: Identifier,
    evaluations: Arc<AtomicUsize>,
}

impl ForeignPart for CountingConstraint {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("CountingConstraint")
    }
}

impl CustomConstraint for CountingConstraint {
    fn free_identifiers(&self) -> Result<HashSet<Identifier>, BoxError> {
        Ok(HashSet::from([self.variable.clone()]))
    }

    fn evaluate(
        &self,
        _bindings: &Bindings,
        _context: &ConstraintContext<'_>,
    ) -> Result<Outcome, BoxError> {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        Ok(Outcome::Satisfied)
    }

    fn to_expression(&self) -> Result<Expression, BoxError> {
        Ok(reference(&self.variable)
            .floor_mod(literal(1))
            .equals(literal(0)))
    }

    fn ordering_key(&self) -> Result<Cow<'_, str>, BoxError> {
        Ok(Cow::Borrowed("counting"))
    }

    fn is_alpha_equivalent_under(
        &self,
        other: &dyn CustomConstraint,
        _renaming: &AlphaRenaming,
    ) -> Result<bool, BoxError> {
        Ok(other.as_any().downcast_ref::<Self>().is_some())
    }
}

/// Return the param over the naturals whose one constraint counts its
/// evaluations in `evaluations`.
///
/// # Panics
///
/// Panics if the param is refused.
fn build_counting_param(evaluations: &Arc<AtomicUsize>) -> Param {
    let variable = Identifier::new("n");
    let counting = Constraint::Custom(Part::new(CountingConstraint {
        variable: variable.clone(),
        evaluations: Arc::clone(evaluations),
    }));
    let solver = Solver::new();
    Param::new(
        ParamDomain::from(IntegerDomain::new(
            Sign::NonNegative,
            ZeroInclusion::Included,
        )),
        variable,
        vec![counting],
        &ParamContext::new(&solver),
    )
    .expect("the param is valid")
}

/// Test building a configuration one entry at a time evaluates a param's
/// constraint a number of times linear in the entries, not once per held
/// value per call.
#[test]
fn with_entry_one_at_a_time_checks_work_linear_in_the_entries() {
    const ENTRIES: usize = 150;
    let evaluations = Arc::new(AtomicUsize::new(0));
    let names: Vec<Identifier> = (0..ENTRIES)
        .map(|position| Identifier::new(&format!("n{position}")))
        .collect();
    let space = space_of(
        &Identifier::new("wide"),
        names
            .iter()
            .map(|name| plain_variable(name, build_counting_param(&evaluations)))
            .collect(),
        Vec::new(),
    );
    let solver = ground_solver();
    let context = ParamContext::new(&solver);
    let mut configuration = configure(&space, []);
    let before = evaluations.load(Ordering::Relaxed);

    for name in &names {
        configuration = configuration
            .with_entry(name.clone(), int(1), &context)
            .expect("every value is admitted");
    }

    let spent = evaluations.load(Ordering::Relaxed) - before;
    assert!(configuration.is_complete());
    assert!(
        spent <= 4 * ENTRIES,
        "{spent} evaluations for {ENTRIES} entries given one at a time"
    );
}

// -- activity through conditions at depth -------------------------------------

/// What building a configuration gives: the configuration, or the text of
/// the errors that refused it.
#[derive(Debug, PartialEq)]
enum Built {
    Configuration(Configuration),
    Refused(String),
}

/// Return what `result` built.
fn describe_built(result: &Result<Configuration, ConfigurationErrors>) -> Built {
    match result {
        Ok(configuration) => Built::Configuration(configuration.clone()),
        Err(errors) => Built::Refused(format!("{errors:?}")),
    }
}

/// The chain space: the choice `g` among the bare alternatives `p` and
/// `q`; `m` over `{1, 2}`, active only while `g` chose `p`; and `n` over
/// `{1, 2}`, active only while `m` is 1.
struct Chain {
    space: Space,
    names: HashMap<&'static str, Identifier>,
}

impl Chain {
    /// Return the chain space, its names fresh.
    fn new() -> Self {
        let names: HashMap<&'static str, Identifier> = ["chain", "g", "p", "q", "m", "n"]
            .into_iter()
            .map(|label| (label, Identifier::new(label)))
            .collect();
        let n = |label: &str| &names[label];
        let space = Space::new(
            n("chain").clone(),
            vec![int_variable(n("m"), &[1, 2]), int_variable(n("n"), &[1, 2])],
            vec![choice_of(
                n("g"),
                vec![bare_alternative(n("p")), bare_alternative(n("q"))],
            )],
            vec![
                condition(n("m"), [chooses(n("g"), &[n("p")])]),
                condition(n("n"), [in_set(n("m"), [int(1)])]),
            ],
            Vec::new(),
        )
        .expect("the chain space is valid");
        Self { space, names }
    }

    /// Return the entries `picks` stand for.
    fn entries(&self, picks: &[(&str, Pick)]) -> Vec<(Identifier, Value)> {
        picks
            .iter()
            .map(|&(label, pick)| {
                let value = match pick {
                    Pick::Int(number) => int(number),
                    Pick::Alt(alternative) => chosen(&self.names[alternative]),
                };
                (self.names[label].clone(), value)
            })
            .collect()
    }
}

/// Guard: `with_entry` changing a decision that gates others through
/// conditions at depth gives exactly what `Configuration::new` gives for
/// the same entries, the same configuration or the same errors. It passes
/// before the change: it guards the incremental check against dropping an
/// activity change.
#[rstest]
#[case::choice_switch_deactivates_both_held_values(
    &[("g", Pick::Alt("p")), ("m", Pick::Int(1)), ("n", Pick::Int(1))],
    ("g", Pick::Alt("q")), true
)]
#[case::first_gate_closing_deactivates_the_second_held_value(
    &[("g", Pick::Alt("p")), ("m", Pick::Int(1)), ("n", Pick::Int(1))],
    ("m", Pick::Int(2)), true
)]
#[case::first_gate_opening_activates_the_second(
    &[("g", Pick::Alt("p")), ("m", Pick::Int(2))],
    ("m", Pick::Int(1)), false
)]
#[case::choice_switch_activates_the_chain(
    &[("g", Pick::Alt("q"))],
    ("g", Pick::Alt("p")), false
)]
#[case::value_for_an_active_end_of_the_chain(
    &[("g", Pick::Alt("p")), ("m", Pick::Int(1))],
    ("n", Pick::Int(2)), false
)]
#[case::value_for_an_inactive_end_of_the_chain(
    &[("g", Pick::Alt("p")), ("m", Pick::Int(2))],
    ("n", Pick::Int(1)), true
)]
fn with_entry_guard_agrees_with_new_when_activity_changes_at_depth(
    #[case] start: &[(&str, Pick)],
    #[case] change: (&str, Pick),
    #[case] is_refused: bool,
) {
    let chain = Chain::new();
    let held = configure(&chain.space, chain.entries(start));
    let (name, value) = chain.entries(&[change]).remove(0);
    let mut all = chain.entries(start);
    match all.iter_mut().find(|(held_name, _)| *held_name == name) {
        Some(entry) => entry.1 = value.clone(),
        None => all.push((name.clone(), value.clone())),
    }

    let stepped = with_context(|context| held.with_entry(name, value, context));

    let fresh = try_configure(&chain.space, all);
    let built = describe_built(&fresh);
    assert_eq!(describe_built(&stepped), built);
    assert_eq!(matches!(built, Built::Refused(_)), is_refused, "{built:?}");
}

// -- property: one at a time agrees with new -----------------------------------

/// A generated space and a list of entries for it: up to three top-level
/// variables over small domains, optionally a choice of two alternatives
/// each holding a variable, optionally a condition on one decision and a
/// forbidden clause over the first two variables.
///
/// Decisions are numbered: the top-level variables first, then the choice,
/// then the variable of its first alternative, then that of its second.
#[derive(Debug, Clone)]
struct Model {
    /// The size `k` of each top-level variable's domain `{1..=k}`.
    top_sizes: Vec<i64>,
    /// The sizes of the domains of the two alternatives' variables, when
    /// the choice is present.
    nested_sizes: Option<(i64, i64)>,
    /// The raw selector of the condition's target among the decisions it
    /// can gate, when there is a condition: the first top-level variable
    /// gates it, being 1.
    condition: Option<usize>,
    /// The values the first two variables take together, when forbidden.
    clause: Option<(i64, i64)>,
    /// The order the decisions are given in; indices beyond the model's
    /// decisions are skipped.
    order: Vec<usize>,
    /// The draw behind each decision's value, or none: from `0..20`, where
    /// 17 and above give a variable a value outside its domain, and a
    /// choice a value that is no alternative.
    values: Vec<Option<i64>>,
}

/// The names of a built model's decisions, in the model's numbering, and
/// the names of the choice's alternatives.
struct ModelNames {
    decisions: Vec<Identifier>,
    alternatives: [Identifier; 2],
}

impl Model {
    /// Return the number of decisions.
    fn decision_count(&self) -> usize {
        self.top_sizes.len() + self.nested_sizes.map_or(0, |_| 3)
    }
}

/// Return the space of `model` and its names.
fn build_space(model: &Model) -> (Space, ModelNames) {
    let tops = model.top_sizes.len();
    let decisions: Vec<Identifier> = (0..6)
        .map(|position| Identifier::new(&format!("d{position}")))
        .collect();
    let alternatives = [Identifier::new("alt0"), Identifier::new("alt1")];
    let domain = |size: i64| (1..=size).collect::<Vec<_>>();
    let variables = model
        .top_sizes
        .iter()
        .enumerate()
        .map(|(position, &size)| int_variable(&decisions[position], &domain(size)))
        .collect();
    let choices = model
        .nested_sizes
        .map(|(first, second)| {
            let holder = |slot: usize, size: i64| {
                plain_alternative(
                    &alternatives[slot],
                    vec![int_variable(&decisions[tops + 1 + slot], &domain(size))],
                    Vec::new(),
                )
            };
            vec![choice_of(
                &decisions[tops],
                vec![holder(0, first), holder(1, second)],
            )]
        })
        .unwrap_or_default();
    let conditions = model
        .condition
        .map(|selector| {
            let mut targets: Vec<usize> = (1..tops).collect();
            if model.nested_sizes.is_some() {
                targets.extend([tops + 1, tops + 2]);
            }
            let target = targets[selector % targets.len()];
            vec![condition(
                &decisions[target],
                [in_set(&decisions[0], [int(1)])],
            )]
        })
        .unwrap_or_default();
    let clauses = model
        .clause
        .map(|(first, second)| {
            vec![forbidden([
                in_set(&decisions[0], [int(first)]),
                in_set(&decisions[1], [int(second)]),
            ])]
        })
        .unwrap_or_default();
    let space = Space::new(
        Identifier::new("generated"),
        variables,
        choices,
        conditions,
        clauses,
    )
    .expect("the generated space is valid");
    (
        space,
        ModelNames {
            decisions,
            alternatives,
        },
    )
}

/// Return the value of the decision numbered `decision` that the draw `raw`
/// stands for: for a variable, one of its domain below 17, 0 at 17, and one
/// above its domain after; for a choice, an alternative below 17 and an
/// integer after.
fn build_model_value(model: &Model, names: &ModelNames, decision: usize, raw: i64) -> Value {
    let tops = model.top_sizes.len();
    let nested = model.nested_sizes.unwrap_or((1, 1));
    let size = match decision.checked_sub(tops) {
        None => model.top_sizes[decision],
        Some(1) => nested.0,
        Some(_) => nested.1,
    };
    if model.nested_sizes.is_some() && decision == tops {
        return if raw < 17 {
            chosen(&names.alternatives[usize::from(raw % 2 == 1)])
        } else {
            int(raw)
        };
    }
    match raw {
        0..=16 => int(1 + raw % size),
        17 => int(0),
        _ => int(size + 1),
    }
}

/// Return the entries of `model` in the order it gives them: each decision
/// it has a value for, once.
fn build_model_entries(model: &Model, names: &ModelNames) -> Vec<(Identifier, Value)> {
    model
        .order
        .iter()
        .filter(|&&decision| decision < model.decision_count())
        .filter_map(|&decision| {
            let raw = model.values[decision]?;
            Some((
                names.decisions[decision].clone(),
                build_model_value(model, names, decision, raw),
            ))
        })
        .collect()
}

/// What refused an entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Refusal {
    Assignment,
    Inactive,
    Forbidden,
    UnknownAlternative,
}

/// What building a model's entries one at a time did.
#[derive(Debug, Default)]
struct Walk {
    accepted: usize,
    refusals: Vec<Refusal>,
}

/// Fold `with_entry` over the entries of `model`, one at a time, and check
/// each step against `Configuration::new` over the same prefix, stopping at
/// the first refusal.
///
/// # Errors
///
/// Returns the first step the two disagree on.
fn walk_entries(model: &Model) -> Result<Walk, TestCaseError> {
    let (space, names) = build_space(model);
    let entries = build_model_entries(model, &names);
    let mut configuration = configure(&space, []);
    let mut walk = Walk::default();
    for (step, (name, value)) in entries.iter().enumerate() {
        let stepped =
            with_context(|context| configuration.with_entry(name.clone(), value.clone(), context));
        let fresh = try_configure(&space, entries[..=step].to_vec());
        prop_assert_eq!(
            describe_built(&stepped),
            describe_built(&fresh),
            "step {}",
            step
        );
        match stepped {
            Ok(next) => {
                configuration = next;
                walk.accepted += 1;
            }
            Err(errors) => {
                walk.refusals = errors
                    .errors()
                    .iter()
                    .filter_map(|error| match error {
                        ConfigurationError::Assignment { .. } => Some(Refusal::Assignment),
                        ConfigurationError::InactiveDecision { .. } => Some(Refusal::Inactive),
                        ConfigurationError::Forbidden { .. } => Some(Refusal::Forbidden),
                        ConfigurationError::UnknownAlternative { .. } => {
                            Some(Refusal::UnknownAlternative)
                        }
                        _ => None,
                    })
                    .collect();
                break;
            }
        }
    }
    Ok(walk)
}

/// Return the order of the six decision numbers after `swaps`, each
/// exchanging two positions of the natural order, so the order is mostly
/// the natural one, which gives decisions after those they depend on, and
/// sometimes not.
fn shuffle_order(swaps: &[(usize, usize)]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..6).collect();
    for &(left, right) in swaps {
        order.swap(left, right);
    }
    order
}

/// Return the strategy of models.
fn generate_model() -> impl Strategy<Value = Model> {
    (
        prop::collection::vec(1..=3i64, 2..=3),
        prop::option::of((1..=3i64, 1..=3i64)),
        prop::option::of(0..5usize),
        prop::option::of((1..=2i64, 1..=2i64)),
        prop::collection::vec((0..6usize, 0..6usize), 0..=2),
        prop::collection::vec(prop::option::weighted(0.85, 0..20i64), 6),
    )
        .prop_map(
            |(top_sizes, nested_sizes, condition, clause, swaps, values)| Model {
                top_sizes,
                nested_sizes,
                condition,
                clause,
                order: shuffle_order(&swaps),
                values,
            },
        )
}

proptest! {
    /// Test building the entries one at a time with `with_entry` agrees
    /// with `Configuration::new` over the same entries at every step:
    /// both build the same configuration, or both refuse with the same
    /// errors.
    #[test]
    fn building_entries_one_at_a_time_agrees_with_building_them_at_once(
        model in generate_model()
    ) {
        walk_entries(&model)?;
    }
}

// ---------------------------------------------------------------------------
// Non-vacuity guards
// ---------------------------------------------------------------------------

/// The cases the guard draws.
const GUARD_CASES: usize = 256;

/// Return `GUARD_CASES` models drawn by a runner with a fixed seed, so the
/// guard's counts are the same on every run.
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

/// Test the strategy reaches both accepted and refused entries, with
/// conditions and forbidden clauses in play, so the property is not
/// checking only trivial walks.
#[test]
fn generated_models_reach_accepted_and_refused_entries_conditions_and_clauses() {
    let models = draw_models();
    let walks: Vec<Walk> = models
        .iter()
        .map(|model| walk_entries(model).expect("the walk agrees with new"))
        .collect();

    let accepting = walks.iter().filter(|walk| walk.accepted >= 2).count();
    let refused = walks
        .iter()
        .filter(|walk| !walk.refusals.is_empty())
        .count();
    let with_refusal = |wanted: Refusal| {
        walks
            .iter()
            .filter(|walk| walk.refusals.contains(&wanted))
            .count()
    };
    let conditions = models
        .iter()
        .filter(|model| model.condition.is_some())
        .count();
    let clauses = models.iter().filter(|model| model.clause.is_some()).count();

    assert!(
        accepting * 2 >= GUARD_CASES,
        "{accepting} of {GUARD_CASES} walks accept two entries"
    );
    assert!(
        refused * 4 >= GUARD_CASES,
        "{refused} of {GUARD_CASES} walks end in a refusal"
    );
    assert!(
        with_refusal(Refusal::Assignment) >= 10,
        "{} walks refuse a value outside a domain",
        with_refusal(Refusal::Assignment)
    );
    assert!(
        with_refusal(Refusal::Inactive) >= 5,
        "{} walks refuse a value for an inactive decision",
        with_refusal(Refusal::Inactive)
    );
    assert!(
        with_refusal(Refusal::Forbidden) >= 5,
        "{} walks reach a forbidden combination",
        with_refusal(Refusal::Forbidden)
    );
    assert!(
        with_refusal(Refusal::UnknownAlternative) >= 3,
        "{} walks give a choice a value that is no alternative",
        with_refusal(Refusal::UnknownAlternative)
    );
    assert!(
        conditions * 4 >= GUARD_CASES && clauses * 4 >= GUARD_CASES,
        "{conditions} models with a condition, {clauses} with a clause, of {GUARD_CASES}"
    );
}
