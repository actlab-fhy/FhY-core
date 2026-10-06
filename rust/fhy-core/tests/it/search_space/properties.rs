//! Properties of spaces and configurations over generated hierarchies:
//! the equivalence laws (reflexive, symmetric under label reuse, structural
//! implies alpha, invariant under relabeling, broken by a perturbation, no
//! constraint blindness, no capture, no conflation of `1` and `true`),
//! activity against a brute-force reference evaluator, keys, and serde
//! round trips.
//!
//! A space is generated as a [`Model`]: plain variables and choices, every
//! name a label index in canonical order (the space's own name is label
//! 0), categorical domains over integers, Booleans and identifiers (labels
//! of the space, which are references to its names, or free identifiers),
//! conditions and forbidden clauses of set constraints. A condition names
//! only decisions before its target in canonical order, so every model is
//! acyclic. The model is built into a [`Space`] with a table of
//! identifiers per label, so relabeling a space is building its model with
//! another table.

use std::collections::HashSet;

use fhy_core::constraint::{Constraint, Polarity, SetConstraint, Value};
use fhy_core::foreign::Part;
use fhy_core::identifier::Identifier;
use fhy_core::param::{CategoricalDomain, Param, ParamContext, ParamDomain};
use fhy_core::search_space::{
    Activity, Alternative, Choice, Condition, Configuration, ConfigurationError, Forbidden,
    PlainAlternative, PlainVariable, Space, SpaceError, Variable,
};
use fhy_core::solver::Solver;
use fhy_core::term::AlphaEquivalence;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};

use crate::support::constraint::{int, member_set};
use crate::support::hashing::hash_of;
use crate::support::search_space::{ground_solver, system};
use crate::support::serde::check_serde_round_trip;

/// The labels a model may name, and so the size of a label table.
const LABEL_COUNT: usize = 96;
/// The free identifiers a model's members may name.
const FREE_COUNT: usize = 3;

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------

/// A member of a generated categorical domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Member {
    Int(i64),
    Bool(bool),
    /// The identifier of a label: a reference to a name of the space when
    /// the label is used, and otherwise an identifier the table holds.
    Label(usize),
    /// A free identifier, the same in every table.
    Free(usize),
}

/// A generated plain variable: its categories, distinct, and the positions
/// of the categories its param's set constraint keeps, if it has one.
#[derive(Debug, Clone)]
struct VariableModel {
    members: Vec<Member>,
    narrowed: Option<Vec<usize>>,
}

/// A generated plain alternative.
#[derive(Debug, Clone)]
struct AlternativeModel {
    variables: Vec<VariableModel>,
    choices: Vec<ChoiceModel>,
}

/// A generated choice.
#[derive(Debug, Clone)]
struct ChoiceModel {
    alternatives: Vec<AlternativeModel>,
}

/// One set constraint of a condition or a forbidden clause: the decision
/// it names, by canonical position, and a mask of the categories (for a
/// variable) or alternatives (for a choice) it keeps.
#[derive(Debug, Clone, Copy)]
struct ClauseModel {
    decision: usize,
    mask: u8,
}

/// A generated space.
#[derive(Debug, Clone)]
struct Model {
    variables: Vec<VariableModel>,
    choices: Vec<ChoiceModel>,
    /// Each condition's target, by canonical position, and its members.
    conditions: Vec<(usize, Vec<ClauseModel>)>,
    forbidden: Vec<Vec<ClauseModel>>,
}

/// What a decision of a model is.
#[derive(Debug, Clone)]
enum DecisionKind {
    Variable(VariableModel),
    /// A choice, with its alternatives' labels.
    Choice(Vec<usize>),
}

/// A decision of a model, in canonical order.
#[derive(Debug, Clone)]
struct DecisionModel {
    label: usize,
    kind: DecisionKind,
    /// The decision's choice, by canonical position, and the position of
    /// the alternative it is under.
    parent: Option<(usize, usize)>,
}

impl Model {
    /// Return the decisions in canonical order, and the number of labels
    /// the model uses (its names, the space's own included).
    fn decisions(&self) -> (Vec<DecisionModel>, usize) {
        let mut decisions = Vec::new();
        let mut next_label = 1;
        for variable in &self.variables {
            decisions.push(DecisionModel {
                label: next_label,
                kind: DecisionKind::Variable(variable.clone()),
                parent: None,
            });
            next_label += 1;
        }
        for choice in &self.choices {
            walk_choice(choice, None, &mut decisions, &mut next_label);
        }
        (decisions, next_label)
    }
}

/// Append `choice`'s decisions to `decisions`, in canonical order, under
/// `parent`, numbering its names from `next_label`.
fn walk_choice(
    choice: &ChoiceModel,
    parent: Option<(usize, usize)>,
    decisions: &mut Vec<DecisionModel>,
    next_label: &mut usize,
) {
    let position = decisions.len();
    let label = *next_label;
    *next_label += 1;
    decisions.push(DecisionModel {
        label,
        kind: DecisionKind::Choice(Vec::new()),
        parent,
    });
    let mut alternative_labels = Vec::new();
    for (index, alternative) in choice.alternatives.iter().enumerate() {
        alternative_labels.push(*next_label);
        *next_label += 1;
        for variable in &alternative.variables {
            decisions.push(DecisionModel {
                label: *next_label,
                kind: DecisionKind::Variable(variable.clone()),
                parent: Some((position, index)),
            });
            *next_label += 1;
        }
        for sub in &alternative.choices {
            walk_choice(sub, Some((position, index)), decisions, next_label);
        }
    }
    decisions[position].kind = DecisionKind::Choice(alternative_labels);
}

/// The identifiers a model is built with: one per label, and the free
/// identifiers.
#[derive(Debug, Clone)]
struct Table {
    labels: Vec<Identifier>,
    free: Vec<Identifier>,
    /// One param variable per label, so that two builds of one model with
    /// one table have structurally equal params.
    param_variables: Vec<Identifier>,
}

impl Table {
    /// Return a table of fresh identifiers.
    fn fresh() -> Self {
        Self {
            labels: (0..LABEL_COUNT)
                .map(|index| Identifier::new(&format!("l{index}")))
                .collect(),
            free: (0..FREE_COUNT)
                .map(|index| Identifier::new(&format!("f{index}")))
                .collect(),
            param_variables: (0..LABEL_COUNT)
                .map(|index| Identifier::new(&format!("p{index}")))
                .collect(),
        }
    }

    /// Return this table with the first `used` labels replaced by fresh
    /// identifiers, minted in reverse order when `reversed`, and every
    /// other identifier kept.
    fn relabeled(&self, used: usize, reversed: bool) -> Self {
        let mut fresh: Vec<Identifier> = Vec::with_capacity(used);
        if reversed {
            for index in (0..used).rev() {
                fresh.push(Identifier::new(&format!("r{index}")));
            }
            fresh.reverse();
        } else {
            for index in 0..used {
                fresh.push(Identifier::new(&format!("r{index}")));
            }
        }
        let mut labels = self.labels.clone();
        labels[..used].clone_from_slice(&fresh);
        Self {
            labels,
            free: self.free.clone(),
            param_variables: self.param_variables.clone(),
        }
    }

    /// Return this table with its labels drawn from `pool` by `choices`,
    /// label `i` becoming `pool[choices[i] % pool.len()]`.
    fn pooled(&self, pool: &[Identifier], choices: &[usize]) -> Self {
        let labels = (0..LABEL_COUNT)
            .map(|index| pool[choices[index % choices.len()] % pool.len()].clone())
            .collect();
        Self {
            labels,
            ..self.clone()
        }
    }

    /// Return the value of `member`.
    fn value(&self, member: Member) -> Value {
        match member {
            Member::Int(value) => int(value),
            Member::Bool(value) => Value::Bool(value),
            Member::Label(index) => Value::Identifier(self.labels[index].clone()),
            Member::Free(index) => Value::Identifier(self.free[index].clone()),
        }
    }
}

/// Return the param of `variable`, whose own variable is `param_variable`.
fn build_param(variable: &VariableModel, table: &Table, param_variable: &Identifier) -> Param {
    let mut values: Vec<Value> = Vec::new();
    for &member in &variable.members {
        let value = table.value(member);
        if !values.contains(&value) {
            values.push(value);
        }
    }
    let constraints: Vec<Constraint> = variable
        .narrowed
        .iter()
        .map(|kept| {
            Constraint::from(SetConstraint::new(
                param_variable.clone(),
                member_set(
                    kept.iter()
                        .map(|&index| table.value(variable.members[index])),
                ),
                Polarity::In,
            ))
        })
        .collect();
    let solver = Solver::new();
    Param::new(
        ParamDomain::from(CategoricalDomain::new(values).expect("the members are distinct leaves")),
        param_variable.clone(),
        constraints,
        &ParamContext::new(&solver),
    )
    .expect("the param is valid")
}

/// Builds a model's parts, numbering names in canonical order.
struct Builder<'a> {
    table: &'a Table,
    next_label: usize,
}

impl Builder<'_> {
    /// Return the next label's identifier, and its index.
    fn take_label(&mut self) -> (Identifier, usize) {
        let label = self.next_label;
        self.next_label += 1;
        (self.table.labels[label].clone(), label)
    }

    /// Return the plain variable of `variable`.
    fn variable(&mut self, variable: &VariableModel) -> Part<dyn Variable> {
        let (name, label) = self.take_label();
        let param = build_param(variable, self.table, &self.table.param_variables[label]);
        Part::new(PlainVariable::new(name, param))
    }

    /// Return the choice of `choice`.
    fn choice(&mut self, choice: &ChoiceModel) -> Result<Choice, SpaceError> {
        let (name, _) = self.take_label();
        let mut alternatives: Vec<Part<dyn Alternative>> = Vec::new();
        for alternative in &choice.alternatives {
            let (alternative_name, _) = self.take_label();
            let variables = alternative
                .variables
                .iter()
                .map(|variable| self.variable(variable))
                .collect();
            let choices = alternative
                .choices
                .iter()
                .map(|sub| self.choice(sub))
                .collect::<Result<Vec<_>, _>>()?;
            alternatives.push(Part::new(PlainAlternative::new(
                alternative_name,
                variables,
                choices,
            )?));
        }
        Choice::new(name, alternatives)
    }
}

/// Return the set constraint of `clause` over `decisions`.
fn build_clause(clause: ClauseModel, decisions: &[DecisionModel], table: &Table) -> Constraint {
    let decision = &decisions[clause.decision];
    let values: Vec<Value> = match &decision.kind {
        DecisionKind::Variable(variable) => variable
            .members
            .iter()
            .map(|&member| table.value(member))
            .collect(),
        DecisionKind::Choice(labels) => labels
            .iter()
            .map(|&label| Value::Identifier(table.labels[label].clone()))
            .collect(),
    };
    let kept = values
        .into_iter()
        .enumerate()
        .filter(|(index, _)| clause.mask & (1 << index) != 0)
        .map(|(_, value)| value);
    Constraint::from(SetConstraint::new(
        table.labels[decision.label].clone(),
        member_set(kept),
        Polarity::In,
    ))
}

/// Return the space of `model` built with `table`.
fn build_space(model: &Model, table: &Table) -> Result<Space, SpaceError> {
    let (decisions, _) = model.decisions();
    let mut builder = Builder {
        table,
        next_label: 1,
    };
    let variables = model
        .variables
        .iter()
        .map(|variable| builder.variable(variable))
        .collect();
    let choices = model
        .choices
        .iter()
        .map(|choice| builder.choice(choice))
        .collect::<Result<Vec<_>, _>>()?;
    let conditions = model
        .conditions
        .iter()
        .map(|(target, clauses)| {
            Condition::new(
                table.labels[decisions[*target].label].clone(),
                system(
                    clauses
                        .iter()
                        .map(|&clause| build_clause(clause, &decisions, table)),
                ),
            )
        })
        .collect();
    let forbidden = model
        .forbidden
        .iter()
        .map(|clauses| {
            Forbidden::new(system(
                clauses
                    .iter()
                    .map(|&clause| build_clause(clause, &decisions, table)),
            ))
        })
        .collect();
    Space::new(
        table.labels[0].clone(),
        variables,
        choices,
        conditions,
        forbidden,
    )
}

/// Return the space of `model` built with `table`, which must succeed.
fn build_valid(model: &Model, table: &Table) -> Space {
    build_space(model, table).unwrap_or_else(|error| panic!("refused {model:?}: {error}"))
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// Return a strategy of distinct categories.
fn members() -> impl Strategy<Value = Vec<Member>> {
    let member = prop_oneof![
        3 => (1_i64..=3).prop_map(Member::Int),
        1 => any::<bool>().prop_map(Member::Bool),
        2 => (0..LABEL_COUNT).prop_map(Member::Label),
        1 => (0..FREE_COUNT).prop_map(Member::Free),
    ];
    prop::collection::vec(member, 1..=4).prop_map(|members| {
        let mut seen = HashSet::new();
        members
            .into_iter()
            .filter(|member| seen.insert(*member))
            .collect()
    })
}

/// Return a strategy of variables.
fn variable() -> impl Strategy<Value = VariableModel> {
    (members(), any::<Option<u8>>()).prop_map(|(members, narrowing)| {
        let narrowed = narrowing.map(|mask| {
            (0..members.len())
                .filter(|index| mask & (1 << index) != 0)
                .collect()
        });
        VariableModel { members, narrowed }
    })
}

/// Return a strategy of choices nested at most `depth` levels below.
fn choice(depth: u32) -> BoxedStrategy<ChoiceModel> {
    let leaf = prop::collection::vec(
        prop::collection::vec(variable(), 0..=2).prop_map(|variables| AlternativeModel {
            variables,
            choices: Vec::new(),
        }),
        1..=3,
    )
    .prop_map(|alternatives| ChoiceModel { alternatives });
    if depth == 0 {
        return leaf.boxed();
    }
    prop::collection::vec(
        (
            prop::collection::vec(variable(), 0..=2),
            prop::collection::vec(choice(depth - 1), 0..=1),
        )
            .prop_map(|(variables, choices)| AlternativeModel { variables, choices }),
        1..=3,
    )
    .prop_map(|alternatives| ChoiceModel { alternatives })
    .boxed()
}

/// Return a strategy of models with conditions and forbidden clauses.
fn model() -> impl Strategy<Value = Model> {
    (
        prop::collection::vec(variable(), 0..=3),
        prop::collection::vec(choice(1), 0..=2),
    )
        .prop_flat_map(|(variables, choices)| {
            let structure = Model {
                variables,
                choices,
                conditions: Vec::new(),
                forbidden: Vec::new(),
            };
            let count = structure.decisions().0.len();
            let clauses = |count: usize| {
                prop::collection::vec(
                    (0..count.max(1), any::<u8>())
                        .prop_map(|(decision, mask)| ClauseModel { decision, mask }),
                    1..=2,
                )
            };
            let conditions = prop::collection::vec(
                (1..count.max(2)).prop_flat_map(move |target| {
                    (
                        Just(target),
                        prop::collection::vec(
                            (0..target, any::<u8>())
                                .prop_map(|(decision, mask)| ClauseModel { decision, mask }),
                            1..=2,
                        ),
                    )
                }),
                0..=if count >= 2 { 2 } else { 0 },
            );
            let forbidden =
                prop::collection::vec(clauses(count), 0..=if count >= 1 { 2 } else { 0 });
            (Just(structure), conditions, forbidden).prop_map(
                |(mut model, conditions, forbidden)| {
                    model.conditions = conditions;
                    model.forbidden = forbidden;
                    model
                },
            )
        })
}

/// Return a strategy of assignments of `count` decisions: per decision,
/// none or a raw index reduced modulo its categories or alternatives.
fn raw_assignment() -> impl Strategy<Value = Vec<Option<u8>>> {
    prop::collection::vec(prop::option::weighted(0.7, any::<u8>()), 0..=24)
}

// ---------------------------------------------------------------------------
// The reference evaluator
// ---------------------------------------------------------------------------

/// Return the number of categories or alternatives of `decision`.
fn arity(decision: &DecisionModel) -> usize {
    match &decision.kind {
        DecisionKind::Variable(variable) => variable.members.len(),
        DecisionKind::Choice(labels) => labels.len(),
    }
}

/// Return the assignment `raw` gives the model's decisions: each index
/// reduced modulo the decision's arity.
fn reduce(decisions: &[DecisionModel], raw: &[Option<u8>]) -> Vec<Option<usize>> {
    decisions
        .iter()
        .enumerate()
        .map(|(position, decision)| {
            raw.get(position)
                .copied()
                .flatten()
                .map(|index| usize::from(index) % arity(decision))
        })
        .collect()
}

/// Return whether the set constraint `clause` holds under `assignment`,
/// whose named decision is assigned.
fn holds(clause: ClauseModel, assignment: &[Option<usize>]) -> bool {
    let index = assignment[clause.decision].expect("the decision is assigned");
    clause.mask & (1 << index) != 0
}

/// Return what the decisions `clauses` name say of a condition or clause:
/// `Inactive` if one is inactive, else `Pending` if one is pending or
/// unassigned, else `Active` if every member holds and `Inactive` if not.
fn judge(clauses: &[ClauseModel], activity: &[Activity], assignment: &[Option<usize>]) -> Activity {
    if clauses
        .iter()
        .any(|clause| activity[clause.decision] == Activity::Inactive)
    {
        return Activity::Inactive;
    }
    if clauses.iter().any(|clause| {
        activity[clause.decision] == Activity::Pending || assignment[clause.decision].is_none()
    }) {
        return Activity::Pending;
    }
    if clauses.iter().all(|&clause| holds(clause, assignment)) {
        Activity::Active
    } else {
        Activity::Inactive
    }
}

/// Return each decision's activity under `assignment`, by the documented
/// rules, in canonical order: every decision a decision depends on comes
/// before it in canonical order in a generated model.
fn reference_activity(
    model: &Model,
    decisions: &[DecisionModel],
    assignment: &[Option<usize>],
) -> Vec<Activity> {
    let mut activity: Vec<Activity> = Vec::with_capacity(decisions.len());
    for (position, decision) in decisions.iter().enumerate() {
        let from_parent = match decision.parent {
            None => Activity::Active,
            Some((choice, alternative)) => match activity[choice] {
                Activity::Inactive => Activity::Inactive,
                Activity::Pending => Activity::Pending,
                Activity::Active => match assignment[choice] {
                    None => Activity::Pending,
                    Some(chosen) if chosen == alternative => Activity::Active,
                    Some(_) => Activity::Inactive,
                },
            },
        };
        let clauses: Vec<ClauseModel> = model
            .conditions
            .iter()
            .filter(|(target, _)| *target == position)
            .flat_map(|(_, clauses)| clauses.iter().copied())
            .collect();
        let from_condition = if clauses.is_empty() || from_parent == Activity::Inactive {
            Activity::Active
        } else {
            judge(&clauses, &activity, assignment)
        };
        activity.push(match (from_parent, from_condition) {
            (Activity::Inactive, _) | (_, Activity::Inactive) => Activity::Inactive,
            (Activity::Pending, _) | (_, Activity::Pending) => Activity::Pending,
            (Activity::Active, Activity::Active) => Activity::Active,
        });
    }
    activity
}

/// Return whether `assignment` is a valid configuration of `model`: every
/// assigned decision active with a value its param keeps, and no forbidden
/// clause applying and holding.
fn reference_is_valid(
    model: &Model,
    decisions: &[DecisionModel],
    assignment: &[Option<usize>],
    activity: &[Activity],
) -> bool {
    let entries_valid = decisions.iter().enumerate().all(|(position, decision)| {
        let Some(index) = assignment[position] else {
            return true;
        };
        activity[position] == Activity::Active
            && match &decision.kind {
                DecisionKind::Variable(variable) => variable
                    .narrowed
                    .as_ref()
                    .is_none_or(|kept| kept.contains(&index)),
                DecisionKind::Choice(_) => true,
            }
    });
    entries_valid
        && model
            .forbidden
            .iter()
            .all(|clauses| judge(clauses, activity, assignment) != Activity::Active)
}

/// Return `assignment` with every entry the reference evaluator would
/// refuse dropped, in canonical order, so that only forbidden clauses can
/// refuse it.
fn repair(
    model: &Model,
    decisions: &[DecisionModel],
    assignment: &[Option<usize>],
) -> Vec<Option<usize>> {
    let mut repaired: Vec<Option<usize>> = vec![None; decisions.len()];
    for position in 0..decisions.len() {
        repaired[position] = assignment[position];
        let activity = reference_activity(model, decisions, &repaired);
        let keeps = repaired[position].is_some_and(|index| {
            activity[position] == Activity::Active
                && match &decisions[position].kind {
                    DecisionKind::Variable(variable) => variable
                        .narrowed
                        .as_ref()
                        .is_none_or(|kept| kept.contains(&index)),
                    DecisionKind::Choice(_) => true,
                }
        });
        if !keeps {
            repaired[position] = None;
        }
    }
    repaired
}

/// Return the entries of `assignment` with `table`'s identifiers.
fn entries(
    decisions: &[DecisionModel],
    assignment: &[Option<usize>],
    table: &Table,
) -> Vec<(Identifier, Value)> {
    decisions
        .iter()
        .zip(assignment)
        .filter_map(|(decision, index)| {
            let index = (*index)?;
            let value = match &decision.kind {
                DecisionKind::Variable(variable) => table.value(variable.members[index]),
                DecisionKind::Choice(labels) => {
                    Value::Identifier(table.labels[labels[index]].clone())
                }
            };
            Some((table.labels[decision.label].clone(), value))
        })
        .collect()
}

/// Return the configuration of `space` with `entries`.
fn try_configuration(
    space: &Space,
    entries: Vec<(Identifier, Value)>,
) -> Result<Configuration, fhy_core::search_space::ConfigurationErrors> {
    let solver = ground_solver();
    Configuration::new(space, entries, &ParamContext::new(&solver))
}

/// Return whether `left` and `right` are alpha-equivalent, in each
/// direction.
fn alpha<T: AlphaEquivalence>(left: &T, right: &T) -> [bool; 2]
where
    T::Error: std::fmt::Debug,
{
    [
        left.is_alpha_equivalent(right)
            .expect("the comparison succeeds"),
        right
            .is_alpha_equivalent(left)
            .expect("the comparison succeeds"),
    ]
}

// ---------------------------------------------------------------------------
// Perturbations
// ---------------------------------------------------------------------------

/// A change to a model that makes it inequivalent to the original.
#[derive(Debug, Clone, Copy)]
enum Perturbation {
    /// Add the category 4, which no generated domain holds, to a variable.
    AddCategory(usize),
    /// Add a set constraint to a variable's param, or drop the one it has.
    ToggleNarrowing(usize),
    /// Drop every condition on one target.
    DropConditions(usize),
    /// Drop the last forbidden clause.
    DropForbidden,
}

/// Return `model` with `perturbation` applied, or `None` when the model
/// has nothing it applies to.
fn perturb(model: &Model, perturbation: Perturbation) -> Option<Model> {
    let mut changed = model.clone();
    let mut variables: Vec<&mut VariableModel> = Vec::new();
    collect_variables(&mut changed.variables, &mut changed.choices, &mut variables);
    match perturbation {
        Perturbation::AddCategory(index) => {
            let count = variables.len();
            let variable = variables.into_iter().nth(index % count.max(1))?;
            variable.members.push(Member::Int(4));
        }
        Perturbation::ToggleNarrowing(index) => {
            let count = variables.len();
            let variable = variables.into_iter().nth(index % count.max(1))?;
            variable.narrowed = match variable.narrowed {
                Some(_) => None,
                None => Some(vec![0]),
            };
        }
        Perturbation::DropConditions(index) => {
            let targets: Vec<usize> = changed
                .conditions
                .iter()
                .map(|(target, _)| *target)
                .collect();
            let target = *targets.get(index % targets.len().max(1))?;
            changed.conditions.retain(|(other, _)| *other != target);
        }
        Perturbation::DropForbidden => {
            changed.forbidden.pop()?;
        }
    }
    Some(changed)
}

/// Collect mutable references to every variable of a model's parts.
fn collect_variables<'a>(
    variables: &'a mut [VariableModel],
    choices: &'a mut [ChoiceModel],
    into: &mut Vec<&'a mut VariableModel>,
) {
    into.extend(variables.iter_mut());
    for choice in choices {
        for alternative in &mut choice.alternatives {
            collect_variables(&mut alternative.variables, &mut alternative.choices, into);
        }
    }
}

/// Return a strategy of pairs of models: a model and itself, half the
/// time, and otherwise the model and a perturbation of it, which is itself
/// when the perturbation does not apply.
fn model_pair() -> impl Strategy<Value = (Model, Model)> {
    (model(), prop::option::of(perturbation())).prop_map(|(model, change)| {
        let other = change
            .and_then(|change| perturb(&model, change))
            .unwrap_or_else(|| model.clone());
        (model, other)
    })
}

/// Return a strategy of perturbations.
fn perturbation() -> impl Strategy<Value = Perturbation> {
    prop_oneof![
        any::<usize>().prop_map(Perturbation::AddCategory),
        any::<usize>().prop_map(Perturbation::ToggleNarrowing),
        any::<usize>().prop_map(Perturbation::DropConditions),
        Just(Perturbation::DropForbidden),
    ]
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn space_equivalences_are_reflexive(model in model()) {
        let space = build_valid(&model, &Table::fresh());

        prop_assert!(space.is_structurally_equivalent(&space).expect("plain parts"));
        prop_assert_eq!(alpha(&space, &space), [true, true]);
    }

    #[test]
    fn relabeled_space_is_alpha_equivalent_and_not_structurally(
        model in model(),
        reversed in any::<bool>(),
    ) {
        let table = Table::fresh();
        let (_, used) = model.decisions();
        let relabeled = table.relabeled(used, reversed);

        let left = build_valid(&model, &table);
        let right = build_valid(&model, &relabeled);

        prop_assert_eq!(alpha(&left, &right), [true, true]);
        prop_assert!(!left.is_structurally_equivalent(&right).expect("plain parts"));
    }

    #[test]
    fn spaces_built_apart_from_one_model_and_table_are_structurally_equivalent(model in model()) {
        let table = Table::fresh();

        let left = build_valid(&model, &table);
        let right = build_valid(&model, &table);

        prop_assert!(left.is_structurally_equivalent(&right).expect("plain parts"));
        prop_assert_eq!(&left, &right);
        prop_assert_eq!(hash_of(&left), hash_of(&right));
    }

    #[test]
    fn structural_equivalence_implies_alpha_equivalence(
        (left_model, right_model) in model_pair(),
    ) {
        let table = Table::fresh();
        let left = build_valid(&left_model, &table);
        let right = build_valid(&right_model, &table);

        let structural = left.is_structurally_equivalent(&right).expect("plain parts");

        prop_assert_eq!(
            structural,
            right.is_structurally_equivalent(&left).expect("plain parts")
        );
        if structural {
            prop_assert_eq!(alpha(&left, &right), [true, true]);
        }
    }

    #[test]
    fn alpha_equivalence_is_symmetric_when_labels_are_reused(
        left_model in model(),
        right_model in model(),
        pool_size in 2_usize..6,
        left_choices in prop::collection::vec(0_usize..6, 1..=LABEL_COUNT),
        right_choices in prop::collection::vec(0_usize..6, 1..=LABEL_COUNT),
    ) {
        let table = Table::fresh();
        let pool: Vec<Identifier> = (0..pool_size).map(|index| Identifier::new(&format!("s{index}"))).collect();
        let (left_table, right_table) = (table.pooled(&pool, &left_choices), table.pooled(&pool, &right_choices));
        let built = [(&left_model, &left_table), (&right_model, &right_table)].map(|(model, table)| {
            let (_, used) = model.decisions();
            let names = &table.labels[..used];
            let repeated: HashSet<&Identifier> = names
                .iter()
                .enumerate()
                .filter(|(position, name)| names[..*position].contains(name))
                .map(|(_, name)| name)
                .collect();
            (build_space(model, table), repeated)
        });

        let mut spaces = Vec::new();
        for (result, repeated) in built {
            match result {
                Ok(space) => {
                    prop_assert!(repeated.is_empty(), "a space repeating a name was built");
                    spaces.push(space);
                }
                Err(SpaceError::DuplicateName { name }) => {
                    prop_assert!(repeated.contains(&name), "{name:?} is not repeated");
                }
                Err(other) => prop_assert!(false, "unexpected refusal {other}"),
            }
        }

        if let [left, right] = spaces.as_slice() {
            let [forward, backward] = alpha(left, right);
            prop_assert_eq!(forward, backward);
            prop_assert_eq!(
                left.is_structurally_equivalent(right).expect("plain parts"),
                right.is_structurally_equivalent(left).expect("plain parts")
            );
        }
    }

    #[test]
    fn a_perturbation_breaks_alpha_equivalence(
        model in model(),
        change in perturbation(),
    ) {
        let Some(changed) = perturb(&model, change) else {
            return Ok(());
        };
        let table = Table::fresh();
        let (_, used) = changed.decisions();

        let left = build_valid(&model, &table);
        let right = build_valid(&changed, &table.relabeled(used, false));

        prop_assert_eq!(alpha(&left, &right), [false, false]);
    }

    #[test]
    fn a_free_member_never_corresponds_to_a_bound_name(
        model in model(),
        target in any::<usize>(),
        name in any::<usize>(),
    ) {
        let (decisions, used) = model.decisions();
        let variables: Vec<usize> = decisions
            .iter()
            .enumerate()
            .filter(|(_, decision)| matches!(decision.kind, DecisionKind::Variable(_)))
            .map(|(position, _)| position)
            .collect();
        let Some(&position) = variables.get(target % variables.len().max(1)) else {
            return Ok(());
        };
        let bound = 1 + name % (used - 1).max(1);
        let mut left_model = model.clone();
        let mut right_model = model.clone();
        set_first_member(&mut left_model, position, Member::Free(0), &[]);
        set_first_member(&mut right_model, position, Member::Label(bound), &[]);
        let table = Table::fresh();
        let mut right_table = table.relabeled(used, false);
        right_table.labels[bound] = table.free[0].clone();

        let left = build_valid(&left_model, &table);
        let right = build_valid(&right_model, &right_table);

        prop_assert_eq!(
            alpha(&left, &right),
            [false, false],
            "the left member is free and the right one, the same identifier, is a name"
        );
    }

    #[test]
    fn integer_and_boolean_members_never_correspond(model in model(), target in any::<usize>()) {
        let (decisions, used) = model.decisions();
        let variables: Vec<usize> = decisions
            .iter()
            .enumerate()
            .filter(|(_, decision)| matches!(decision.kind, DecisionKind::Variable(_)))
            .map(|(position, _)| position)
            .collect();
        let Some(&position) = variables.get(target % variables.len().max(1)) else {
            return Ok(());
        };
        let mut left_model = model.clone();
        let mut right_model = model.clone();
        set_first_member(&mut left_model, position, Member::Int(1), &[Member::Bool(true)]);
        set_first_member(&mut right_model, position, Member::Bool(true), &[Member::Int(1)]);
        let table = Table::fresh();

        let left = build_valid(&left_model, &table);
        let right = build_valid(&right_model, &table.relabeled(used, false));

        prop_assert_eq!(alpha(&left, &right), [false, false]);
    }

    #[test]
    fn activity_agrees_with_the_reference_evaluator(
        model in model(),
        raw in raw_assignment(),
        repaired in any::<bool>(),
    ) {
        let table = Table::fresh();
        let space = build_valid(&model, &table);
        let (decisions, _) = model.decisions();
        let mut assignment = reduce(&decisions, &raw);
        if repaired {
            assignment = repair(&model, &decisions, &assignment);
        }
        let activity = reference_activity(&model, &decisions, &assignment);
        let expected_valid = reference_is_valid(&model, &decisions, &assignment, &activity);

        let result = try_configuration(&space, entries(&decisions, &assignment, &table));

        match result {
            Ok(configuration) => {
                prop_assert!(expected_valid, "accepted an invalid configuration");
                for (decision, expected) in decisions.iter().zip(&activity) {
                    prop_assert_eq!(
                        configuration.activity(&table.labels[decision.label]),
                        Some(*expected)
                    );
                }
                let complete = activity
                    .iter()
                    .zip(&assignment)
                    .all(|(state, value)| *state == Activity::Inactive || value.is_some());
                prop_assert_eq!(configuration.is_complete(), complete);
            }
            Err(errors) => {
                prop_assert!(!expected_valid, "refused a valid configuration: {errors}");
                for problem in errors.errors() {
                    prop_assert!(
                        matches!(
                            problem,
                            ConfigurationError::InactiveDecision { .. }
                                | ConfigurationError::Assignment { .. }
                                | ConfigurationError::Forbidden { .. }
                        ),
                        "a generated configuration cannot cause {problem:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn corresponding_configurations_of_relabeled_spaces_have_equal_keys(
        model in model(),
        raw in raw_assignment(),
        reversed in any::<bool>(),
    ) {
        let table = Table::fresh();
        let (decisions, used) = model.decisions();
        let relabeled = table.relabeled(used, reversed);
        let assignment = repair(&model, &decisions, &reduce(&decisions, &raw));
        let left_space = build_valid(&model, &table);
        let right_space = build_valid(&model, &relabeled);

        let left = try_configuration(&left_space, entries(&decisions, &assignment, &table));
        let right = try_configuration(&right_space, entries(&decisions, &assignment, &relabeled));

        match (left, right) {
            (Ok(left), Ok(right)) => {
                prop_assert_eq!(left.key(), right.key());
                prop_assert_eq!(hash_of(&left.key()), hash_of(&right.key()));
                prop_assert_eq!(alpha(&left, &right), [true, true]);
            }
            (Err(_), Err(_)) => {}
            (left, right) => prop_assert!(
                false,
                "corresponding configurations are accepted alike: {left:?} and {right:?}"
            ),
        }
    }

    #[test]
    fn keys_of_one_space_are_equal_exactly_when_configurations_are(
        model in model(),
        first in raw_assignment(),
        second in raw_assignment(),
    ) {
        let table = Table::fresh();
        let (decisions, _) = model.decisions();
        let space = build_valid(&model, &table);
        let build = |raw: &[Option<u8>]| {
            let assignment = repair(&model, &decisions, &reduce(&decisions, raw));
            try_configuration(&space, entries(&decisions, &assignment, &table)).ok()
        };

        if let (Some(left), Some(right)) = (build(&first), build(&second)) {
            prop_assert_eq!(left.key() == right.key(), left == right);
            if left.key() == right.key() {
                prop_assert_eq!(hash_of(&left.key()), hash_of(&right.key()));
                prop_assert_eq!(hash_of(&left), hash_of(&right));
            }
        }
    }

    #[test]
    fn spaces_and_configurations_round_trip_through_serde(
        model in model(),
        raw in raw_assignment(),
    ) {
        let table = Table::fresh();
        let (decisions, _) = model.decisions();
        let space = build_valid(&model, &table);
        let assignment = repair(&model, &decisions, &reduce(&decisions, &raw));

        check_serde_round_trip(&space)?;
        if let Ok(configuration) = try_configuration(&space, entries(&decisions, &assignment, &table)) {
            check_serde_round_trip(&configuration)?;
        }
    }
}

/// Replace the first category of the variable at canonical `position` of
/// `model` with `member`, dropping any other category equal to it or to one
/// of `rivals`.
fn set_first_member(model: &mut Model, position: usize, member: Member, rivals: &[Member]) {
    let (decisions, _) = model.decisions();
    let target_label = decisions[position].label;
    let mut variables: Vec<&mut VariableModel> = Vec::new();
    collect_variables(&mut model.variables, &mut model.choices, &mut variables);
    let labels: Vec<usize> = decisions
        .iter()
        .filter(|decision| matches!(decision.kind, DecisionKind::Variable(_)))
        .map(|decision| decision.label)
        .collect();
    let index = labels
        .iter()
        .position(|&label| label == target_label)
        .expect("the position is a variable");
    let variable: &mut VariableModel = variables.swap_remove(index);
    variable
        .members
        .retain(|other| *other != member && !rivals.contains(other));
    if variable.members.is_empty() {
        variable.members.push(member);
    } else {
        variable.members[0] = member;
    }
    variable.narrowed = None;
}

// ---------------------------------------------------------------------------
// Non-vacuity guards
// ---------------------------------------------------------------------------

/// The cases each guard draws.
const GUARD_CASES: usize = 256;

/// Return `GUARD_CASES` values of `strategy`, drawn by a runner with a
/// fixed seed, so a guard's count is the same on every run.
fn draw<S: Strategy>(strategy: &S) -> Vec<S::Value> {
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

/// Return whether `model` holds a variable at some depth.
fn has_variable(model: &Model) -> bool {
    model
        .decisions()
        .0
        .iter()
        .any(|decision| matches!(decision.kind, DecisionKind::Variable(_)))
}

/// Return whether the repaired configuration of `raw` is accepted for the
/// space of `model`.
fn is_repaired_configuration_accepted(model: &Model, raw: &[Option<u8>]) -> bool {
    let table = Table::fresh();
    let (decisions, _) = model.decisions();
    let space = build_valid(model, &table);
    let assignment = repair(model, &decisions, &reduce(&decisions, raw));
    try_configuration(&space, entries(&decisions, &assignment, &table)).is_ok()
}

#[test]
fn most_generated_models_hold_a_variable() {
    let models = draw(&model());

    let count = models.iter().filter(|model| has_variable(model)).count();

    assert!(
        count * 10 >= GUARD_CASES * 6,
        "{count} of {GUARD_CASES} models hold a variable; a model lacks one only when it \
         has no top-level variable and no alternative holds one"
    );
}

#[test]
fn most_perturbations_apply() {
    let drawn = draw(&(model(), perturbation()));

    let count = drawn
        .iter()
        .filter(|(model, change)| perturb(model, *change).is_some())
        .count();

    assert!(
        count * 2 >= GUARD_CASES,
        "{count} of {GUARD_CASES} perturbations apply; two of the four kinds need only a \
         variable"
    );
}

#[test]
fn many_repaired_configurations_are_accepted() {
    let drawn = draw(&(model(), raw_assignment()));

    let count = drawn
        .iter()
        .filter(|(model, raw)| is_repaired_configuration_accepted(model, raw))
        .count();

    assert!(
        count * 4 >= GUARD_CASES,
        "{count} of {GUARD_CASES} repaired configurations are accepted; only a holding \
         forbidden clause refuses one, and a third of the models have none"
    );
}

#[test]
fn many_repaired_configuration_pairs_are_both_accepted() {
    let drawn = draw(&(model(), raw_assignment(), raw_assignment()));

    let count = drawn
        .iter()
        .filter(|(model, first, second)| {
            is_repaired_configuration_accepted(model, first)
                && is_repaired_configuration_accepted(model, second)
        })
        .count();

    assert!(
        count * 5 >= GUARD_CASES,
        "{count} of {GUARD_CASES} pairs are both accepted; both are whenever the model \
         has no forbidden clause"
    );
}

#[test]
fn many_model_pairs_are_structurally_equivalent() {
    let drawn = draw(&model_pair());

    let count = drawn
        .iter()
        .filter(|(left, right)| {
            let table = Table::fresh();
            build_valid(left, &table)
                .is_structurally_equivalent(&build_valid(right, &table))
                .expect("plain parts")
        })
        .count();

    assert!(
        count * 10 >= GUARD_CASES * 3,
        "{count} of {GUARD_CASES} pairs are structurally equivalent; half are a model and \
         itself"
    );
}
