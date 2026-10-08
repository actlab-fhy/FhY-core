//! [`Space`]: the decisions of a search, with the [`Condition`]s on when
//! each is active and the [`Forbidden`] combinations of their values.
//!
//! This module owns the space's invariants: the names, the scopes of the
//! conditions and forbidden clauses, and the canonical and decision
//! orders.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::constraint::{Constraint, ConstraintSystem, Value};
use crate::diagnostic::Note;
use crate::foreign::Part;
use crate::identifier::Identifier;
use crate::term::{AlphaEquivalence, AlphaRenaming};

use super::choice::{Choice, first_repeat};
use super::equivalence::{is_space_alpha_equivalent, is_space_structurally_equivalent};
use super::error::{EquivalenceError, SpaceError};
use super::step::member_value;
use super::variable::Variable;

/// When the decision [`target`](Self::target) is active: while the
/// constraints [`when`](Self::when) holds, read under the values of the
/// decisions it names.
///
/// A condition names at least one decision, names a choice only in a set
/// constraint whose members are the identifier values of the choice's
/// alternatives' names, and names no decision under its target.
/// [`Space::new`] checks these. `==` and `Hash` compare the target
/// and the constraints.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Condition {
    target: Identifier,
    when: ConstraintSystem,
}

impl Condition {
    /// Return the condition that `target` is active only while `when`
    /// holds.
    #[must_use]
    pub fn new(target: Identifier, when: ConstraintSystem) -> Self {
        Self { target, when }
    }

    /// Return the decision the condition is on.
    #[must_use]
    pub fn target(&self) -> &Identifier {
        &self.target
    }

    /// Return the constraints that must hold.
    #[must_use]
    pub fn when(&self) -> &ConstraintSystem {
        &self.when
    }
}

/// A combination of values no configuration may take: one that satisfies
/// the constraints [`when`](Self::when) once every decision they name is
/// active and assigned.
///
/// A positive rule `p` across decisions is written as the forbidden clause
/// `not p`. A forbidden clause names a choice only in a set constraint
/// whose members are the identifier values of the choice's alternatives'
/// names, and names at least one decision. [`Space::new`] checks both. `==` and `Hash` compare the
/// constraints.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Forbidden {
    when: ConstraintSystem,
}

impl Forbidden {
    /// Return the clause forbidding every configuration that satisfies
    /// `when`.
    #[must_use]
    pub fn new(when: ConstraintSystem) -> Self {
        Self { when }
    }

    /// Return the constraints a configuration must not satisfy.
    #[must_use]
    pub fn when(&self) -> &ConstraintSystem {
        &self.when
    }
}

/// One decision of a [`Space`]: a variable or a choice.
#[expect(
    clippy::exhaustive_enums,
    reason = "a decision is a variable or a choice, and callers match both"
)]
#[derive(Debug, Clone, Copy)]
pub enum Decision<'a> {
    /// A variable.
    Variable(&'a Part<dyn Variable>),
    /// A choice.
    Choice(&'a Choice),
}

impl<'a> Decision<'a> {
    /// Return the decision's name.
    #[must_use]
    pub fn name(&self) -> &'a Identifier {
        match self {
            Self::Variable(variable) => variable.get().name(),
            Self::Choice(choice) => choice.name(),
        }
    }
}

/// A search space: top-level variables and choices, the conditions on when
/// each decision is active, and the forbidden combinations of values.
///
/// Its choices nest at most [`MAX_CHOICE_DEPTH`](super::MAX_CHOICE_DEPTH)
/// levels, which [`Choice::new`] checks.
///
/// Building one checks, in order, that:
///
/// 1. every name it holds is distinct: its own, and every decision's,
///    alternative's and bound identifier's at every depth, read in
///    canonical order;
/// 2. each condition's target is a decision; the condition names at
///    least one decision, and every identifier it names is a decision
///    outside the target's subtree (the target and everything under its
///    alternatives); none of its equations names a choice; and each of
///    its set constraints over a choice holds only its alternatives'
///    names;
/// 3. each forbidden clause names at least one decision, only decisions,
///    no choice in an equation, and only a choice's alternatives' names in
///    a set constraint over it;
/// 4. no decision depends on itself, a decision depending on its choice
///    and on every decision its condition names.
///
/// Conditions on one target conjoin: the space keeps one condition per
/// target, holding all their constraints, in canonical order of the
/// targets. Forbidden clauses keep their order.
///
/// Cloning one shares it. `==` and `Hash` compare the name, the parts
/// (through their `eq_part`), the conditions, the forbidden clauses and
/// the notes. Its wire form is `{"identifier", "variables", "choices",
/// "conditions", "forbidden", "notes"}` (see [`wire`](super::wire)).
#[derive(Debug, Clone)]
pub struct Space(Arc<SpaceInner>);

#[derive(Debug, Clone)]
struct SpaceInner {
    name: Identifier,
    variables: Vec<Part<dyn Variable>>,
    choices: Vec<Choice>,
    /// One condition per target, in canonical order of the targets.
    conditions: Vec<Condition>,
    forbidden: Vec<Forbidden>,
    notes: Vec<Note>,
    /// Every name the space holds, its own first, then in canonical order.
    labels: Vec<Identifier>,
    /// The position of each name in `labels`.
    label_positions: HashMap<Identifier, usize>,
    /// The decisions, in canonical order.
    nodes: Vec<Node>,
    /// The canonical position of each decision, by name.
    positions: HashMap<Identifier, usize>,
    /// The decisions' names in decision order.
    order: Vec<Identifier>,
    /// The decisions' canonical positions in decision order.
    order_positions: Vec<usize>,
    /// The canonical positions of the decisions each forbidden clause
    /// names, ascending.
    forbidden_references: Vec<Vec<usize>>,
    /// Per decision, by canonical position, the positions of the
    /// forbidden clauses naming it, ascending.
    forbidden_naming: Vec<Vec<usize>>,
    /// Per decision, by canonical position, the canonical positions of the
    /// decisions that depend on it directly, ascending: those under its
    /// alternatives' top level, and the targets of the conditions naming
    /// it.
    dependents: Vec<Vec<usize>>,
    /// Per decision, by canonical position, its rank in decision order.
    order_ranks: Vec<usize>,
}

/// One decision of a space, in canonical order.
#[derive(Debug, Clone)]
struct Node {
    part: NodePart,
    /// The canonical position of the decision's choice and the position of
    /// the alternative it is under, for a decision under an alternative.
    parent: Option<(usize, usize)>,
    /// The canonical position after the decision's subtree: the decisions
    /// under it are the ones between it and this position.
    subtree_end: usize,
    /// The position of the decision's condition in the space's conditions,
    /// and the canonical positions of the decisions it names, ascending.
    condition: Option<(usize, Vec<usize>)>,
}

/// The variable or choice of a [`Node`].
#[derive(Debug, Clone)]
enum NodePart {
    Variable(Part<dyn Variable>),
    Choice(Choice),
}

impl Node {
    /// Return the decision.
    fn decision(&self) -> Decision<'_> {
        match &self.part {
            NodePart::Variable(variable) => Decision::Variable(variable),
            NodePart::Choice(choice) => Decision::Choice(choice),
        }
    }
}

impl Space {
    /// Return the space named `name` holding the top-level `variables` and
    /// `choices`, the `conditions` and the `forbidden` clauses, with no
    /// notes.
    ///
    /// # Errors
    ///
    /// The first problem found, checking in the order the type documents:
    /// [`SpaceError::DuplicateName`]; for each condition in order,
    /// [`SpaceError::UnknownConditionTarget`],
    /// [`SpaceError::UnknownReference`],
    /// [`SpaceError::EquationOverChoice`],
    /// [`SpaceError::UnknownAlternative`],
    /// [`SpaceError::EmptyCondition`] and
    /// [`SpaceError::ConditionReferencesSubtree`]; for each forbidden
    /// clause in order, [`SpaceError::EmptyForbidden`],
    /// [`SpaceError::UnknownReference`],
    /// [`SpaceError::EquationOverChoice`] and
    /// [`SpaceError::UnknownAlternative`]; then
    /// [`SpaceError::CyclicDependency`]. Within one condition or clause,
    /// the names it refers to are checked in the order of their ids, and
    /// its set constraints over choices in its order, each member in the
    /// set's.
    /// [`SpaceError::Constraint`] reports a custom constraint whose scope
    /// or key fails.
    pub fn new(
        name: Identifier,
        variables: Vec<Part<dyn Variable>>,
        choices: Vec<Choice>,
        conditions: Vec<Condition>,
        forbidden: Vec<Forbidden>,
    ) -> Result<Self, SpaceError> {
        build(name, variables, choices, conditions, forbidden).map(|inner| Self(Arc::new(inner)))
    }

    /// Return this space with `notes` in place of its notes.
    #[must_use]
    pub fn with_notes(self, notes: Vec<Note>) -> Self {
        let mut inner = Arc::unwrap_or_clone(self.0);
        inner.notes = notes;
        Self(Arc::new(inner))
    }

    /// Return the space's name.
    #[must_use]
    pub fn name(&self) -> &Identifier {
        &self.0.name
    }

    /// Return the top-level variables, in order.
    #[must_use]
    pub fn variables(&self) -> &[Part<dyn Variable>] {
        &self.0.variables
    }

    /// Return the top-level choices, in order.
    #[must_use]
    pub fn choices(&self) -> &[Choice] {
        &self.0.choices
    }

    /// Return the conditions, one per target, in canonical order of their
    /// targets.
    #[must_use]
    pub fn conditions(&self) -> &[Condition] {
        &self.0.conditions
    }

    /// Return the forbidden clauses, in the order given.
    #[must_use]
    pub fn forbidden(&self) -> &[Forbidden] {
        &self.0.forbidden
    }

    /// Return the notes.
    #[must_use]
    pub fn notes(&self) -> &[Note] {
        &self.0.notes
    }

    /// Return every decision, at every depth, in canonical order.
    pub fn decisions(&self) -> impl ExactSizeIterator<Item = Decision<'_>> + '_ {
        self.0.nodes.iter().map(Node::decision)
    }

    /// Return the decision named `name`, if the space has one.
    #[must_use]
    pub fn decision(&self, name: &Identifier) -> Option<Decision<'_>> {
        self.0
            .positions
            .get(name)
            .map(|&position| self.0.nodes[position].decision())
    }

    /// Return every decision's name in decision order: each after its
    /// choice and after every decision its condition names, and otherwise
    /// in canonical order. Without conditions it is the canonical order.
    #[must_use]
    pub fn decision_order(&self) -> &[Identifier] {
        &self.0.order
    }

    /// Return this space with each of `variables` and `choices` put at the
    /// top level: in place of the top-level variable, or choice, of the
    /// same name, keeping its slot, or after the last top-level variable,
    /// or choice, in the order given. The name, the conditions, the
    /// forbidden clauses and the notes are kept, and the space is checked
    /// as [`new`](Self::new) checks one.
    ///
    /// Every decision before the first one the edit changes, in canonical
    /// order, keeps its canonical position, and so do the steps of a trace
    /// over them: appending choices keeps every existing position, while
    /// appending or replacing a variable, or replacing a choice by one
    /// whose subtree holds another number of decisions, moves the
    /// decisions after it.
    ///
    /// # Errors
    ///
    /// Returns what [`new`](Self::new) returns for the edited space, such
    /// as [`SpaceError::DuplicateName`] for a variable named as a top-level
    /// choice, or [`SpaceError::UnknownReference`] for a condition naming a
    /// decision a replaced choice no longer holds.
    pub fn with_decisions(
        &self,
        variables: Vec<Part<dyn Variable>>,
        choices: Vec<Choice>,
    ) -> Result<Self, SpaceError> {
        let mut edited_variables = self.0.variables.clone();
        for variable in variables {
            let name = variable.get().name();
            match edited_variables
                .iter()
                .position(|held| held.get().name() == name)
            {
                Some(slot) => edited_variables[slot] = variable,
                None => edited_variables.push(variable),
            }
        }
        let mut edited_choices = self.0.choices.clone();
        for choice in choices {
            match edited_choices
                .iter()
                .position(|held| held.name() == choice.name())
            {
                Some(slot) => edited_choices[slot] = choice,
                None => edited_choices.push(choice),
            }
        }
        self.rebuild(edited_variables, edited_choices, self.0.conditions.clone())
    }

    /// Return this space without the top-level decisions `names`, and
    /// without the conditions on them or on any decision under them. The
    /// name, the other conditions, the forbidden clauses and the notes are
    /// kept, and the space is checked as [`new`](Self::new) checks one. A
    /// name given twice is removed once.
    ///
    /// The decisions before the first one removed, in canonical order,
    /// keep their canonical positions; the decisions after it move.
    ///
    /// # Errors
    ///
    /// Returns [`SpaceError::NotTopLevelDecision`] for the first of `names`
    /// that is no top-level decision of the space, and what
    /// [`new`](Self::new) returns for the edited space, such as
    /// [`SpaceError::UnknownReference`] for a condition or a forbidden
    /// clause that names a removed decision.
    pub fn without_decisions(
        &self,
        names: impl IntoIterator<Item = Identifier>,
    ) -> Result<Self, SpaceError> {
        let mut removed: HashSet<usize> = HashSet::new();
        for name in names {
            match self.position(&name) {
                Some(position) if self.0.nodes[position].parent.is_none() => {
                    removed.insert(position);
                }
                _ => return Err(SpaceError::NotTopLevelDecision { name }),
            }
        }
        let is_removed = |name: &Identifier| {
            self.position(name)
                .is_some_and(|position| removed.contains(&position))
        };
        let is_under_removed = |position: usize| {
            removed
                .iter()
                .any(|&top| (top..self.0.nodes[top].subtree_end).contains(&position))
        };
        let variables = self
            .0
            .variables
            .iter()
            .filter(|variable| !is_removed(variable.get().name()))
            .cloned()
            .collect();
        let choices = self
            .0
            .choices
            .iter()
            .filter(|choice| !is_removed(choice.name()))
            .cloned()
            .collect();
        let conditions = self
            .0
            .conditions
            .iter()
            .filter(|condition| {
                self.position(condition.target())
                    .is_none_or(|target| !is_under_removed(target))
            })
            .cloned()
            .collect();
        self.rebuild(variables, choices, conditions)
    }

    /// Return the space of this one's name, forbidden clauses and notes,
    /// holding `variables`, `choices` and `conditions`, checked as
    /// [`new`](Self::new) checks one.
    fn rebuild(
        &self,
        variables: Vec<Part<dyn Variable>>,
        choices: Vec<Choice>,
        conditions: Vec<Condition>,
    ) -> Result<Self, SpaceError> {
        let space = Self::new(
            self.0.name.clone(),
            variables,
            choices,
            conditions,
            self.0.forbidden.clone(),
        )?;
        Ok(space.with_notes(self.0.notes.clone()))
    }

    /// Return whether `other` is the same space up to identity: equal
    /// names and notes, structurally equivalent variables and choices in
    /// order, and structurally equivalent conditions and forbidden
    /// clauses.
    ///
    /// # Errors
    ///
    /// Returns [`EquivalenceError::Extension`] for a hook that fails.
    pub fn is_structurally_equivalent(&self, other: &Self) -> Result<bool, EquivalenceError> {
        is_space_structurally_equivalent(self, other)
    }
}

impl PartialEq for Space {
    /// Compare the names, the variables and choices in order (parts
    /// through their `eq_part`), the conditions, the forbidden clauses and
    /// the notes.
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
            || (self.0.name == other.0.name
                && self.0.variables == other.0.variables
                && self.0.choices == other.0.choices
                && self.0.conditions == other.0.conditions
                && self.0.forbidden == other.0.forbidden
                && self.0.notes == other.0.notes)
    }
}

impl Eq for Space {}

impl Hash for Space {
    /// Feed what `==` compares, parts through their `hash_part`.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.name.hash(state);
        self.0.variables.hash(state);
        self.0.choices.hash(state);
        self.0.conditions.hash(state);
        self.0.forbidden.hash(state);
        self.0.notes.hash(state);
    }
}

impl AlphaEquivalence for Space {
    type Error = EquivalenceError;

    /// Compare the two spaces with every name each holds paired as a
    /// binder on top of `renaming`, in canonical order, after the spaces'
    /// own names. They must have one shape: as many top-level variables
    /// and choices, and at every depth as many alternatives, variables,
    /// sub-choices and bound identifiers. Under that frame: every part
    /// corresponds to `other`'s at its position, conditions correspond by
    /// the position of their targets, forbidden clauses in order, and the
    /// notes are equal. Two systems of constraints correspond when their
    /// members pair up, in any order: equations and custom constraints
    /// through their alpha equivalence, and set constraints by polarity,
    /// variable and members, an identifier member by
    /// [`AlphaRenaming::is_corresponding`].
    ///
    /// # Errors
    ///
    /// Returns [`EquivalenceError::Extension`] for a hook that fails and
    /// [`EquivalenceError::Constraint`] for a custom constraint that fails.
    fn is_alpha_equivalent_under(
        &self,
        other: &Self,
        renaming: &AlphaRenaming,
    ) -> Result<bool, EquivalenceError> {
        is_space_alpha_equivalent(self, other, renaming)
    }
}

/// The read-only views of a space's decision graph that the configuration
/// and the equivalence walk read.
impl Space {
    /// Return every name the space holds, its own first, then in canonical
    /// order.
    pub(super) fn labels(&self) -> &[Identifier] {
        &self.0.labels
    }

    /// Return the position of `name` among the space's names, if it is
    /// one.
    pub(super) fn label_position(&self, name: &Identifier) -> Option<usize> {
        self.0.label_positions.get(name).copied()
    }

    /// Return the number of decisions.
    pub(super) fn decision_count(&self) -> usize {
        self.0.nodes.len()
    }

    /// Return the canonical position of the decision `name`.
    pub(super) fn position(&self, name: &Identifier) -> Option<usize> {
        self.0.positions.get(name).copied()
    }

    /// Return the decision at canonical `position`.
    pub(super) fn decision_at(&self, position: usize) -> Decision<'_> {
        self.0.nodes[position].decision()
    }

    /// Return the canonical position of the choice of the decision at
    /// `position`, and the position of the alternative it is under.
    pub(super) fn parent_at(&self, position: usize) -> Option<(usize, usize)> {
        self.0.nodes[position].parent
    }

    /// Return the condition of the decision at `position`, and the
    /// canonical positions of the decisions it names.
    pub(super) fn condition_with_references_at(
        &self,
        position: usize,
    ) -> Option<(&ConstraintSystem, &[usize])> {
        self.0.nodes[position]
            .condition
            .as_ref()
            .map(|(index, references)| (self.0.conditions[*index].when(), references.as_slice()))
    }

    /// Return the condition of the decision at `position`.
    pub(super) fn condition_at(&self, position: usize) -> Option<&ConstraintSystem> {
        self.condition_with_references_at(position)
            .map(|(system, _)| system)
    }

    /// Return the canonical positions of the decisions in decision order.
    pub(super) fn order_positions(&self) -> &[usize] {
        &self.0.order_positions
    }

    /// Return the canonical positions of the decisions the forbidden clause
    /// at `index` names.
    pub(super) fn forbidden_references(&self, index: usize) -> &[usize] {
        &self.0.forbidden_references[index]
    }

    /// Return the positions of the forbidden clauses naming the decision at
    /// `position`, ascending.
    pub(super) fn forbidden_naming(&self, position: usize) -> &[usize] {
        &self.0.forbidden_naming[position]
    }

    /// Return the canonical positions of the decisions that depend on the
    /// decision at `position` directly, through its choice or their
    /// condition, ascending.
    pub(super) fn dependents_at(&self, position: usize) -> &[usize] {
        &self.0.dependents[position]
    }

    /// Return the rank of the decision at `position` in decision order.
    pub(super) fn order_rank(&self, position: usize) -> usize {
        self.0.order_ranks[position]
    }

    /// Return the canonical position after the subtree of the decision at
    /// `position`: the decisions under it are the ones between the two.
    pub(super) fn subtree_end_at(&self, position: usize) -> usize {
        self.0.nodes[position].subtree_end
    }
}

/// The conditions on one target, gathered: their members, and the
/// canonical positions of the decisions they name.
type Gathered = (Vec<Constraint>, BTreeSet<usize>);

/// Return the data of the space `Space::new` builds, checked as it
/// documents.
fn build(
    name: Identifier,
    variables: Vec<Part<dyn Variable>>,
    choices: Vec<Choice>,
    conditions: Vec<Condition>,
    forbidden: Vec<Forbidden>,
) -> Result<SpaceInner, SpaceError> {
    let labels = collect_labels(&name, &variables, &choices)?;
    let mut nodes = lay_out(&variables, &choices);
    let positions: HashMap<Identifier, usize> = nodes
        .iter()
        .enumerate()
        .map(|(position, node)| (node.decision().name().clone(), position))
        .collect();
    let gathered = gather_conditions(conditions, &positions, &nodes)?;
    let forbidden_references = forbidden
        .iter()
        .enumerate()
        .map(|(index, clause)| check_forbidden(index, clause, &positions, &nodes))
        .collect::<Result<Vec<_>, _>>()?;
    let merged = merge_conditions(gathered, &mut nodes)?;
    let (order_positions, dependents) = find_decision_order(&nodes)?;
    let mut order_ranks = vec![0; nodes.len()];
    for (rank, &position) in order_positions.iter().enumerate() {
        order_ranks[position] = rank;
    }
    let mut forbidden_naming = vec![Vec::new(); nodes.len()];
    for (index, references) in forbidden_references.iter().enumerate() {
        for &reference in references {
            forbidden_naming[reference].push(index);
        }
    }
    let label_positions = labels
        .iter()
        .enumerate()
        .map(|(position, label)| (label.clone(), position))
        .collect();
    Ok(SpaceInner {
        name,
        variables,
        choices,
        conditions: merged,
        forbidden,
        notes: Vec::new(),
        labels,
        label_positions,
        order: order_positions
            .iter()
            .map(|&position| nodes[position].decision().name().clone())
            .collect(),
        order_positions,
        nodes,
        positions,
        forbidden_references,
        forbidden_naming,
        dependents,
        order_ranks,
    })
}

/// Return the space's names, its own first, refusing a repeated one.
fn collect_labels(
    name: &Identifier,
    variables: &[Part<dyn Variable>],
    choices: &[Choice],
) -> Result<Vec<Identifier>, SpaceError> {
    let mut labels = vec![name.clone()];
    labels.extend(
        variables
            .iter()
            .map(|variable| variable.get().name().clone()),
    );
    for choice in choices {
        labels.extend(choice.labels().iter().cloned());
    }
    match first_repeat(&labels) {
        Some(repeated) => Err(SpaceError::DuplicateName {
            name: repeated.clone(),
        }),
        None => Ok(labels),
    }
}

/// Return the decisions of the top-level `variables` and `choices`, in
/// canonical order, with no condition yet.
fn lay_out(variables: &[Part<dyn Variable>], choices: &[Choice]) -> Vec<Node> {
    let mut nodes = Vec::new();
    for variable in variables {
        nodes.push(Node {
            part: NodePart::Variable(variable.clone()),
            parent: None,
            subtree_end: nodes.len() + 1,
            condition: None,
        });
    }
    for choice in choices {
        push_choice(&mut nodes, choice, None);
    }
    nodes
}

/// Return each decision's gathered conditions, by canonical position,
/// checking each condition's target and the names it refers to, at least
/// one.
fn gather_conditions(
    conditions: Vec<Condition>,
    positions: &HashMap<Identifier, usize>,
    nodes: &[Node],
) -> Result<Vec<Option<Gathered>>, SpaceError> {
    let mut gathered: Vec<Option<Gathered>> = vec![None; nodes.len()];
    for condition in conditions {
        let Some(&target) = positions.get(&condition.target) else {
            return Err(SpaceError::UnknownConditionTarget {
                target: condition.target,
            });
        };
        let references = find_references(&condition.when, positions, nodes)?;
        if references.is_empty() {
            return Err(SpaceError::EmptyCondition {
                target: condition.target,
            });
        }
        let subtree = target..nodes[target].subtree_end;
        if let Some(&inside) = references
            .iter()
            .find(|&position| subtree.contains(position))
        {
            return Err(SpaceError::ConditionReferencesSubtree {
                target: condition.target,
                name: nodes[inside].decision().name().clone(),
            });
        }
        let (members, named) =
            gathered[target].get_or_insert_with(|| (Vec::new(), BTreeSet::new()));
        members.extend(condition.when.constraints().iter().cloned());
        named.extend(references);
    }
    Ok(gathered)
}

/// Return the canonical positions of the decisions the forbidden clause at
/// `index` names, ascending, checking that it names at least one and only
/// decisions.
fn check_forbidden(
    index: usize,
    clause: &Forbidden,
    positions: &HashMap<Identifier, usize>,
    nodes: &[Node],
) -> Result<Vec<usize>, SpaceError> {
    if free_identifiers(clause.when())?.is_empty() {
        return Err(SpaceError::EmptyForbidden { index });
    }
    let mut references = find_references(clause.when(), positions, nodes)?;
    references.sort_unstable();
    Ok(references)
}

/// Return one condition per target, in canonical order of the targets,
/// from the gathered conditions, and point each target's node at its
/// condition.
fn merge_conditions(
    gathered: Vec<Option<Gathered>>,
    nodes: &mut [Node],
) -> Result<Vec<Condition>, SpaceError> {
    let mut merged = Vec::new();
    for (target, entry) in gathered.into_iter().enumerate() {
        let Some((members, named)) = entry else {
            continue;
        };
        let when = ConstraintSystem::new(members).map_err(SpaceError::Constraint)?;
        nodes[target].condition = Some((merged.len(), named.into_iter().collect()));
        merged.push(Condition::new(
            nodes[target].decision().name().clone(),
            when,
        ));
    }
    Ok(merged)
}

/// Append the decisions of `choice` to `nodes`, in canonical order, under
/// `parent`.
fn push_choice(nodes: &mut Vec<Node>, choice: &Choice, parent: Option<(usize, usize)>) {
    let position = nodes.len();
    nodes.push(Node {
        part: NodePart::Choice(choice.clone()),
        parent,
        subtree_end: position + 1,
        condition: None,
    });
    for (index, alternative) in choice.alternatives().iter().enumerate() {
        let alternative = alternative.get();
        for variable in alternative.variables() {
            nodes.push(Node {
                part: NodePart::Variable(variable.clone()),
                parent: Some((position, index)),
                subtree_end: nodes.len() + 1,
                condition: None,
            });
        }
        for sub in alternative.choices() {
            push_choice(nodes, sub, Some((position, index)));
        }
    }
    nodes[position].subtree_end = nodes.len();
}

/// Return the identifiers the members of `system` name.
fn free_identifiers(system: &ConstraintSystem) -> Result<HashSet<Identifier>, SpaceError> {
    let mut names = HashSet::new();
    for constraint in system.constraints() {
        names.extend(
            constraint
                .free_identifiers()
                .map_err(SpaceError::Constraint)?,
        );
    }
    Ok(names)
}

/// Return the canonical positions of the decisions `system` names, in the
/// order of their ids, refusing a name that is no decision, an equation
/// that names a choice, and a set constraint over a choice holding a
/// member that is none of its alternatives' names.
fn find_references(
    system: &ConstraintSystem,
    positions: &HashMap<Identifier, usize>,
    nodes: &[Node],
) -> Result<Vec<usize>, SpaceError> {
    let mut names: Vec<Identifier> = free_identifiers(system)?.into_iter().collect();
    names.sort_by_key(Identifier::id);
    let mut references = Vec::with_capacity(names.len());
    for name in &names {
        let Some(&position) = positions.get(name) else {
            return Err(SpaceError::UnknownReference { name: name.clone() });
        };
        references.push(position);
    }
    let mut choices_in_equations: Vec<&Identifier> = Vec::new();
    for constraint in system.constraints() {
        if let Constraint::Equation(equation) = constraint {
            for name in equation.free_identifiers() {
                if let Some(&position) = positions.get(&name) {
                    if let NodePart::Choice(choice) = &nodes[position].part {
                        choices_in_equations.push(choice.name());
                    }
                }
            }
        }
    }
    if let Some(choice) = choices_in_equations
        .into_iter()
        .min_by_key(|name| name.id())
    {
        return Err(SpaceError::EquationOverChoice {
            choice: choice.clone(),
        });
    }
    for constraint in system.constraints() {
        let Constraint::Set(set) = constraint else {
            continue;
        };
        let Some(&position) = positions.get(set.variable()) else {
            continue;
        };
        let NodePart::Choice(choice) = &nodes[position].part else {
            continue;
        };
        for member in set.members() {
            let value = member_value(member);
            let is_alternative = matches!(&value, Value::Identifier(name)
                if choice.alternatives().iter().any(|alternative| alternative.get().name() == name));
            if !is_alternative {
                return Err(SpaceError::UnknownAlternative {
                    choice: choice.name().clone(),
                    value,
                });
            }
        }
    }
    Ok(references)
}

/// Return the decisions' canonical positions in decision order and, per
/// decision, the canonical positions of the decisions depending on it
/// directly, ascending; or the cycle that prevents an order.
fn find_decision_order(nodes: &[Node]) -> Result<(Vec<usize>, Vec<Vec<usize>>), SpaceError> {
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    let mut waiting: Vec<usize> = vec![0; nodes.len()];
    for (position, node) in nodes.iter().enumerate() {
        let mut dependencies: BTreeSet<usize> = BTreeSet::new();
        if let Some((choice, _)) = node.parent {
            dependencies.insert(choice);
        }
        if let Some((_, references)) = &node.condition {
            dependencies.extend(references.iter().copied());
        }
        waiting[position] = dependencies.len();
        for dependency in dependencies {
            dependents[dependency].push(position);
        }
    }
    let mut ready: BTreeSet<usize> = (0..nodes.len())
        .filter(|&position| waiting[position] == 0)
        .collect();
    let mut order = Vec::with_capacity(nodes.len());
    while let Some(position) = ready.pop_first() {
        order.push(position);
        for &dependent in &dependents[position] {
            waiting[dependent] -= 1;
            if waiting[dependent] == 0 {
                ready.insert(dependent);
            }
        }
    }
    if order.len() == nodes.len() {
        return Ok((order, dependents));
    }
    let placed: HashSet<usize> = order.into_iter().collect();
    let cycle = (0..nodes.len())
        .filter(|position| !placed.contains(position))
        .find_map(|start| find_cycle(start, &dependents, &placed))
        .expect("decisions left unplaced lie on or after a cycle");
    Err(SpaceError::CyclicDependency {
        cycle: cycle
            .into_iter()
            .map(|position| nodes[position].decision().name().clone())
            .collect(),
    })
}

/// Return the shortest cycle through `start` along `dependents` among the
/// unplaced decisions, each decision depending on the one before it, or
/// `None` when `start` lies on no cycle.
fn find_cycle(
    start: usize,
    dependents: &[Vec<usize>],
    placed: &HashSet<usize>,
) -> Option<Vec<usize>> {
    let mut previous: HashMap<usize, usize> = HashMap::new();
    let mut queue = VecDeque::from([start]);
    while let Some(position) = queue.pop_front() {
        for &dependent in &dependents[position] {
            if placed.contains(&dependent) {
                continue;
            }
            if dependent == start {
                let mut cycle = vec![position];
                let mut current = position;
                while let Some(&before) = previous.get(&current) {
                    cycle.push(before);
                    current = before;
                }
                cycle.reverse();
                return Some(cycle);
            }
            if let std::collections::hash_map::Entry::Vacant(entry) = previous.entry(dependent) {
                entry.insert(position);
                queue.push_back(dependent);
            }
        }
    }
    None
}
