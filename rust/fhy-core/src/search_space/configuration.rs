//! [`Configuration`]: a point of a space, checked against it, and its
//! [`ConfigurationKey`].
//!
//! This module owns a configuration's invariant: it is valid for its
//! space.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::mem;
use std::sync::Arc;

use crate::constraint::{Binding, Bindings, ConstraintError, ConstraintSystem, Outcome, Value};
use crate::foreign::Part;
use crate::identifier::Identifier;
use crate::param::{ParamAssignment, ParamContext};
use crate::solver::Solver;
use crate::term::{AlphaEquivalence, AlphaRenaming};

use super::alternative::Alternative;
use super::choice::Choice;
use super::chunked::Chunked;
use super::equivalence::{do_spaces_correspond, do_values_correspond, space_frame};
use super::error::{ConfigurationError, ConfigurationErrors, EquivalenceError};
use super::space::{Decision, Space};
use super::variable::Variable;

/// Whether a decision exists in a configuration.
#[expect(
    clippy::exhaustive_enums,
    reason = "a decision is active, inactive or not yet known to be either"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Activity {
    /// The decision exists: its parent is active and its condition holds.
    Active,
    /// The decision does not exist: its choice chose another alternative
    /// or is itself inactive, or its condition is false.
    Inactive,
    /// Whether the decision exists depends on a decision that is active
    /// but unassigned.
    Pending,
}

/// A point of a [`Space`]: a value for some of its active decisions,
/// checked against the space when it is built, so every configuration is
/// valid.
///
/// A variable's value is one its param admits, and a choice's value is the
/// [identifier value](Value::Identifier) of the chosen alternative's name.
/// A configuration may leave active decisions unassigned;
/// [`is_complete`](Self::is_complete) says whether it does.
///
/// Building one checks the entries in three passes and reports every
/// problem it finds, in this order:
///
/// 1. the entries in the order given: one that names no decision of the
///    space ([`UnknownDecision`](super::ConfigurationError::UnknownDecision)),
///    and one that names a decision an earlier entry named
///    ([`DuplicateEntry`](super::ConfigurationError::DuplicateEntry));
/// 2. the decisions in [decision order](Space::decision_order), each
///    after every decision it depends on. Its activity is found first:
///    [inactive](Activity::Inactive) when its parent is, or when its
///    choice chose another alternative, or when its condition names an
///    inactive decision or is violated; otherwise
///    [pending](Activity::Pending) when its choice is pending or active
///    but unassigned, or when its condition names a pending or unassigned
///    decision; otherwise [active](Activity::Active). A condition that
///    evaluates to undecided
///    ([`UndecidedCondition`](super::ConfigurationError::UndecidedCondition))
///    or fails ([`FailedCondition`](super::ConfigurationError::FailedCondition))
///    leaves its target pending. Then an entry for a decision that is not
///    active is refused
///    ([`InactiveDecision`](super::ConfigurationError::InactiveDecision));
///    a choice's value must name one of its alternatives
///    ([`UnknownAlternative`](super::ConfigurationError::UnknownAlternative));
///    and a variable's value must be assignable to its param, as
///    [`ParamAssignment::new`](crate::param::ParamAssignment::new) checks
///    it ([`Assignment`](super::ConfigurationError::Assignment));
/// 3. the forbidden clauses in order: one whose decisions are all active
///    and assigned must be violated
///    ([`Forbidden`](super::ConfigurationError::Forbidden)), and must not
///    be undecided
///    ([`UndecidedForbidden`](super::ConfigurationError::UndecidedForbidden))
///    or fail ([`FailedForbidden`](super::ConfigurationError::FailedForbidden)).
///    A clause naming an inactive decision does not apply, and one naming
///    a pending or unassigned decision does not apply yet.
///
/// An entry refused in one pass counts as unassigned in every check after
/// it. So a condition or forbidden clause naming a variable whose value is
/// refused is not evaluated: its target is pending, or the clause does not
/// apply yet, and only the value's problem is reported for it. Conditions and clauses are evaluated with each decision they name
/// bound to its value, through the context's constraint context.
///
/// Cloning one shares it. `==` and `Hash` compare the spaces, as
/// [`Space`]'s `==` does, and the values.
#[derive(Debug, Clone)]
pub struct Configuration(Arc<ConfigurationInner>);

#[derive(Debug)]
struct ConfigurationInner {
    space: Space,
    /// Each decision's value, in canonical order.
    values: Chunked<Option<Value>>,
    /// Each decision's activity, in canonical order.
    activities: Chunked<Activity>,
    /// Each choice's chosen alternative's position, in canonical order.
    chosen: Chunked<Option<usize>>,
}

impl Configuration {
    /// Return the configuration of `space` giving each decision named in
    /// `entries` its value, checked as the type documents under
    /// `context`.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigurationErrors`] holding every problem found, in the
    /// order the type documents.
    pub fn new(
        space: &Space,
        entries: impl IntoIterator<Item = (Identifier, Value)>,
        context: &ParamContext<'_>,
    ) -> Result<Self, ConfigurationErrors> {
        check(space, entries, context, ValueCheck::New)
    }

    /// Return the configuration of `space` built as [`new`](Self::new)
    /// builds it, except that a variable's value is checked as
    /// [`ParamAssignment::restore`](crate::param::ParamAssignment::restore)
    /// checks it: a constraint of its param the value leaves undecided is
    /// accepted. Decoding a configuration builds it this way.
    pub(super) fn restore(
        space: &Space,
        entries: impl IntoIterator<Item = (Identifier, Value)>,
        context: &ParamContext<'_>,
    ) -> Result<Self, ConfigurationErrors> {
        check(space, entries, context, ValueCheck::Restore)
    }

    /// Return the configuration of `space` assigning nothing.
    ///
    /// It is valid by construction: [`Space::new`] refuses a condition or
    /// a forbidden clause that names no decision, so with every decision
    /// unassigned none of them is evaluated, and the context, which only
    /// evaluation reads, is never asked.
    pub(super) fn empty(space: &Space) -> Self {
        let solver = Solver::new();
        check(space, [], &ParamContext::new(&solver), ValueCheck::New).expect(
            "every condition and forbidden clause names a decision, so a configuration \
             assigning nothing evaluates none of them and is valid",
        )
    }

    /// Return this configuration with the decision `name` given `value`,
    /// in place of its value if it has one, checked as
    /// [`with_entries`](Self::with_entries) checks one entry.
    ///
    /// No entry is dropped implicitly: switching a choice to another
    /// alternative while the variables of the one it chose hold values is
    /// refused, since they are no longer active
    /// ([`InactiveDecision`](super::ConfigurationError::InactiveDecision)).
    /// Remove their values first, with
    /// [`without_entries`](Self::without_entries): to switch the choice
    /// `layout` from the alternative holding the variable `tile` to the
    /// alternative `flat`, ask for
    /// `configuration.without_entries([tile], context)?.with_entry(layout,
    /// Value::Identifier(flat), context)`.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigurationErrors`] holding every problem found.
    pub fn with_entry(
        &self,
        name: Identifier,
        value: Value,
        context: &ParamContext<'_>,
    ) -> Result<Self, ConfigurationErrors> {
        self.with_entries([(name, value)], context)
    }

    /// Return this configuration with each decision named in `entries`
    /// given its value, in place of its value if it has one, checked as
    /// [`new`](Self::new) checks a configuration, except that the values
    /// this configuration holds are not checked against their params
    /// again, since its own check accepted them. Two of `entries` naming
    /// one decision are a
    /// [`DuplicateEntry`](super::ConfigurationError::DuplicateEntry).
    ///
    /// Activity, conditions and forbidden clauses are checked anew only
    /// where the entries can change them: for the decisions that depend
    /// on an entry's decision, through its choice or its condition, at any
    /// depth, and for the forbidden clauses naming one of those
    /// decisions. Its cost grows with the entries and what depends on
    /// them, and with copying the configuration's values, not with
    /// checking every value, so building a configuration one entry at a
    /// time stays cheap.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigurationErrors`] holding every problem found.
    pub fn with_entries(
        &self,
        entries: impl IntoIterator<Item = (Identifier, Value)>,
        context: &ParamContext<'_>,
    ) -> Result<Self, ConfigurationErrors> {
        let mut checker = Checker::extending(self, context);
        checker.read_entries(entries);
        checker.run_incrementally()
    }

    /// Return this configuration without the value of the decision
    /// `name`, checked as [`new`](Self::new) checks a configuration. A
    /// decision that holds no value is left as it is.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigurationErrors`] holding every problem found: an
    /// [`UnknownDecision`](super::ConfigurationError::UnknownDecision) if
    /// the space has no decision `name`, and an
    /// [`InactiveDecision`](super::ConfigurationError::InactiveDecision)
    /// for a decision that holds a value but is no longer active without
    /// `name`'s, such as a variable of the alternative a removed choice's
    /// value chose.
    pub fn without_entry(
        &self,
        name: Identifier,
        context: &ParamContext<'_>,
    ) -> Result<Self, ConfigurationErrors> {
        self.without_entries([name], context)
    }

    /// Return this configuration without the values of the decisions
    /// `names`, checked as [`new`](Self::new) checks a configuration: the
    /// result is the configuration `new` builds from the entries left. A
    /// decision that holds no value, or that `names` names again, is left
    /// as it is.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigurationErrors`] holding every problem found, as
    /// [`without_entry`](Self::without_entry) does, the names that are no
    /// decision of the space first.
    pub fn without_entries(
        &self,
        names: impl IntoIterator<Item = Identifier>,
        context: &ParamContext<'_>,
    ) -> Result<Self, ConfigurationErrors> {
        let mut checker = Checker::new(
            &self.0.space,
            self.0.values.clone(),
            context,
            ValueCheck::New,
        );
        checker.remove_entries(names);
        checker.run_fully()
    }

    /// Return this configuration with the decision `name` given `value`,
    /// checked as [`with_entry`](Self::with_entry) checks it: a search run
    /// grows its configuration this way, one step at a time.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigurationErrors`] holding every problem found.
    pub(super) fn extended(
        &self,
        name: Identifier,
        value: Value,
        context: &ParamContext<'_>,
    ) -> Result<Self, ConfigurationErrors> {
        self.with_entries([(name, value)], context)
    }

    /// Return the space the configuration is a point of.
    #[must_use]
    pub fn space(&self) -> &Space {
        &self.0.space
    }

    /// Return the value of the decision `name`, or `None` if it is
    /// unassigned or the space has no such decision.
    #[must_use]
    pub fn value(&self, name: &Identifier) -> Option<&Value> {
        let position = self.0.space.position(name)?;
        self.0.values.get(position).as_ref()
    }

    /// Return the alternative the choice `choice` chose, or `None` if it is
    /// unassigned or the space has no such choice.
    #[must_use]
    pub fn alternative(&self, choice: &Identifier) -> Option<&Part<dyn Alternative>> {
        let position = self.0.space.position(choice)?;
        let index = (*self.0.chosen.get(position))?;
        match self.0.space.decision_at(position) {
            Decision::Choice(choice) => choice.alternatives().get(index),
            Decision::Variable(_) => None,
        }
    }

    /// Return each assigned decision's name and value, in canonical order.
    pub fn entries(&self) -> impl ExactSizeIterator<Item = (&Identifier, &Value)> + '_ {
        Entries {
            space: &self.0.space,
            values: &self.0.values,
            position: 0,
            remaining: self.0.values.iter().flatten().count(),
        }
    }

    /// Return the activity of the decision `name`, or `None` if the space
    /// has no such decision.
    #[must_use]
    pub fn activity(&self, name: &Identifier) -> Option<Activity> {
        self.0
            .space
            .position(name)
            .map(|position| *self.0.activities.get(position))
    }

    /// Return whether every decision is assigned or inactive.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.0
            .activities
            .iter()
            .zip(self.0.values.iter())
            .all(|(activity, value)| *activity == Activity::Inactive || value.is_some())
    }

    /// Return whether the decision `name` and every decision under it, at
    /// every depth, are assigned or inactive, or `None` if the space has no
    /// such decision.
    ///
    /// For a choice this says whether the configuration has decided all of
    /// its subtree: a choice that is assigned and complete under itself is
    /// fully selected, one that is assigned but not complete under itself
    /// is partly selected. A decision under an alternative the choice did
    /// not choose is inactive, so it counts as decided.
    #[must_use]
    pub fn is_complete_under(&self, name: &Identifier) -> Option<bool> {
        let space = &self.0.space;
        let position = space.position(name)?;
        Some((position..space.subtree_end_at(position)).all(|position| {
            *self.0.activities.get(position) == Activity::Inactive
                || self.0.values.get(position).is_some()
        }))
    }

    /// Return the configuration's key within its space.
    #[must_use]
    pub fn key(&self) -> ConfigurationKey {
        let space = &self.0.space;
        let entries = (0..space.decision_count())
            .map(|position| {
                if *self.0.activities.get(position) == Activity::Inactive {
                    return KeyEntry::Inactive;
                }
                if let Some(index) = *self.0.chosen.get(position) {
                    return KeyEntry::Alternative(index);
                }
                match self.0.values.get(position) {
                    None => KeyEntry::Unassigned,
                    Some(value) => KeyEntry::Value(KeyValue::of(value, space)),
                }
            })
            .collect();
        ConfigurationKey(entries)
    }

    /// Return whether `other` has structurally equivalent spaces, as
    /// [`Space::is_structurally_equivalent`] compares them, and equal
    /// values.
    ///
    /// # Errors
    ///
    /// Returns [`EquivalenceError::Extension`] for a hook that fails.
    pub fn is_structurally_equivalent(&self, other: &Self) -> Result<bool, EquivalenceError> {
        Ok(self.0.space.is_structurally_equivalent(&other.0.space)?
            && self.0.values == other.0.values)
    }
}

impl PartialEq for Configuration {
    /// Compare the spaces, as [`Space`]'s `==` does, and the values.
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
            || (self.0.space == other.0.space && self.0.values == other.0.values)
    }
}

impl Eq for Configuration {}

impl Hash for Configuration {
    /// Feed the space and the values.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.space.hash(state);
        self.0.values.hash(state);
    }
}

impl AlphaEquivalence for Configuration {
    type Error = EquivalenceError;

    /// Compare the spaces as [`Space`]'s alpha equivalence does, then each
    /// decision's value under the spaces' frame: both unassigned, or the
    /// alternatives at one position for a choice, or corresponding values
    /// for a variable, an identifier in a value by
    /// [`AlphaRenaming::is_corresponding`] and anything else by `==`.
    ///
    /// # Errors
    ///
    /// Returns what the spaces' comparison returns.
    fn is_alpha_equivalent_under(
        &self,
        other: &Self,
        renaming: &AlphaRenaming,
    ) -> Result<bool, EquivalenceError> {
        let (left, right) = (&self.0, &other.0);
        let Some(frame) = space_frame(&left.space, &right.space, renaming) else {
            return Ok(false);
        };
        if !do_spaces_correspond(&left.space, &right.space, &frame)? {
            return Ok(false);
        }
        Ok((0..left.space.decision_count()).all(|position| {
            match (left.values.get(position), right.values.get(position)) {
                (None, None) => true,
                (Some(left_value), Some(right_value)) => match *left.chosen.get(position) {
                    Some(index) => *right.chosen.get(position) == Some(index),
                    None => do_values_correspond(left_value, right_value, &frame),
                },
                _ => false,
            }
        }))
    }
}

/// The identity of a [`Configuration`] within its space, the same for
/// configurations of spaces that differ only in their names: one entry per
/// decision, in canonical order.
///
/// A decision's entry says that it is inactive, or unassigned (active or
/// pending), or, for a choice, the position of the chosen alternative, or,
/// for a variable, its value with every identifier the space binds written
/// as its position among the space's names. An identifier the space does
/// not bind stays itself, since only the identity renaming relates the
/// free identifiers of two spaces compared on their own.
///
/// So two configurations of alpha-equivalent spaces whose values
/// correspond have equal keys, and keys of one space are equal exactly
/// when the configurations are. A key means nothing outside its space:
/// a cache keyed by configurations holds the space beside the key. `==`
/// and `Hash` compare the entries.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConfigurationKey(Arc<[KeyEntry]>);

impl ConfigurationKey {
    /// Return the key of `entries`, one per decision in canonical order.
    pub(super) fn from_entries(entries: Vec<KeyEntry>) -> Self {
        Self(entries.into())
    }

    /// Return the entries, one per decision in canonical order.
    pub(super) fn entries(&self) -> &[KeyEntry] {
        &self.0
    }
}

/// One decision's entry in a [`ConfigurationKey`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum KeyEntry {
    Inactive,
    Unassigned,
    Alternative(usize),
    Value(KeyValue),
}

/// A variable's value in a [`ConfigurationKey`]: a [`Value`] whose bound
/// identifiers are written as their positions among the space's names.
///
/// `==` and `Hash` follow [`Value`]'s: type-strict, a frozen set's elements
/// in any order.
#[derive(Debug, Clone)]
pub(super) enum KeyValue {
    Leaf(Value),
    Bound(usize),
    Tuple(Vec<KeyValue>),
    FrozenSet(Vec<KeyValue>),
}

impl PartialEq for KeyValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Leaf(left), Self::Leaf(right)) => left == right,
            (Self::Bound(left), Self::Bound(right)) => left == right,
            (Self::Tuple(left), Self::Tuple(right)) => left == right,
            (Self::FrozenSet(left), Self::FrozenSet(right)) => {
                left.iter().all(|value| right.contains(value))
                    && right.iter().all(|value| left.contains(value))
            }
            _ => false,
        }
    }
}

impl Eq for KeyValue {}

impl Hash for KeyValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        mem::discriminant(self).hash(state);
        match self {
            Self::Leaf(value) => value.hash(state),
            Self::Bound(position) => position.hash(state),
            Self::Tuple(values) => values.hash(state),
            Self::FrozenSet(values) => {
                let mut hashes: Vec<u64> = values
                    .iter()
                    .map(|value| {
                        let mut hasher = DefaultHasher::new();
                        value.hash(&mut hasher);
                        hasher.finish()
                    })
                    .collect();
                hashes.sort_unstable();
                hashes.dedup();
                hashes.hash(state);
            }
        }
    }
}

impl KeyValue {
    /// Return the key value of `value`, each identifier `space` binds
    /// written as its position among the space's names.
    fn of(value: &Value, space: &Space) -> Self {
        match value {
            Value::Identifier(identifier) => match space.label_position(identifier) {
                Some(position) => Self::Bound(position),
                None => Self::Leaf(value.clone()),
            },
            Value::Tuple(values) => {
                Self::Tuple(values.iter().map(|value| Self::of(value, space)).collect())
            }
            Value::FrozenSet(values) => {
                Self::FrozenSet(values.iter().map(|value| Self::of(value, space)).collect())
            }
            _ => Self::Leaf(value.clone()),
        }
    }
}

/// How a configuration's variable values are checked against their
/// params.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueCheck {
    /// As [`ParamAssignment::new`] checks them.
    New,
    /// As [`ParamAssignment::restore`] checks them.
    Restore,
}

/// Which decisions a check looks at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scope {
    /// Every decision and forbidden clause, and every value, given or held.
    Full,
    /// The decisions the entries can change and the forbidden clauses
    /// naming one that changed; a held value, which the configuration's
    /// own check accepted, is not checked against its param again.
    Incremental,
}

/// The state of one check of a configuration's entries: each decision's
/// value, activity and chosen alternative, in canonical order, and the
/// problems found so far.
struct Checker<'a> {
    space: &'a Space,
    context: &'a ParamContext<'a>,
    value_check: ValueCheck,
    scope: Scope,
    values: Chunked<Option<Value>>,
    /// The canonical positions of the decisions an entry gave a value.
    given: HashSet<usize>,
    activities: Chunked<Activity>,
    chosen: Chunked<Option<usize>>,
    problems: Vec<ConfigurationError>,
}

/// Return the configuration of `space` holding `entries`, checked as
/// [`Configuration`] documents.
fn check(
    space: &Space,
    entries: impl IntoIterator<Item = (Identifier, Value)>,
    context: &ParamContext<'_>,
    value_check: ValueCheck,
) -> Result<Configuration, ConfigurationErrors> {
    let mut checker = Checker::new(
        space,
        Chunked::filled(space.decision_count(), &None),
        context,
        value_check,
    );
    checker.read_entries(entries);
    checker.run_fully()
}

impl<'a> Checker<'a> {
    /// Return the checker of a configuration of `space` holding `values`
    /// (each decision's value, in canonical order, or none), checking
    /// every decision, before any entry is read.
    fn new(
        space: &'a Space,
        values: Chunked<Option<Value>>,
        context: &'a ParamContext<'a>,
        value_check: ValueCheck,
    ) -> Self {
        let count = space.decision_count();
        Checker {
            space,
            context,
            value_check,
            scope: Scope::Full,
            values,
            given: HashSet::new(),
            activities: Chunked::filled(count, &Activity::Pending),
            chosen: Chunked::filled(count, &None),
            problems: Vec::new(),
        }
    }

    /// Return the checker of `base` grown by entries, checking only what
    /// the entries can change, before any entry is read.
    fn extending(base: &'a Configuration, context: &'a ParamContext<'a>) -> Self {
        let inner = &base.0;
        Checker {
            space: &inner.space,
            context,
            value_check: ValueCheck::New,
            scope: Scope::Incremental,
            values: inner.values.clone(),
            given: HashSet::new(),
            activities: inner.activities.clone(),
            chosen: inner.chosen.clone(),
            problems: Vec::new(),
        }
    }

    /// Check every decision and forbidden clause, and return the
    /// configuration or every problem found.
    fn run_fully(mut self) -> Result<Configuration, ConfigurationErrors> {
        for &position in self.space.order_positions() {
            self.check_decision(position);
        }
        for index in 0..self.space.forbidden().len() {
            self.check_forbidden_clause(index);
        }
        self.finish()
    }

    /// Check the decisions the entries can change, in decision order: each
    /// given one, and each depending on a decision whose activity, chosen
    /// alternative or value changed; then the forbidden clauses naming a
    /// decision that changed, in order. Return the configuration or every
    /// problem found.
    ///
    /// The configuration grown is valid, so a decision no entry reaches
    /// keeps its activity and value, and a clause naming none of the
    /// changed decisions keeps its outcome.
    fn run_incrementally(mut self) -> Result<Configuration, ConfigurationErrors> {
        let space = self.space;
        let mut queued: HashSet<usize> = self.given.clone();
        let mut queue: BinaryHeap<Reverse<(usize, usize)>> = queued
            .iter()
            .map(|&position| Reverse((space.order_rank(position), position)))
            .collect();
        let mut changed: Vec<usize> = Vec::new();
        while let Some(Reverse((_, position))) = queue.pop() {
            let before = self.read_state(position);
            self.check_decision(position);
            if !self.given.contains(&position) && self.read_state(position) == before {
                continue;
            }
            changed.push(position);
            for &dependent in space.dependents_at(position) {
                if queued.insert(dependent) {
                    queue.push(Reverse((space.order_rank(dependent), dependent)));
                }
            }
        }
        let mut clauses: Vec<usize> = changed
            .iter()
            .flat_map(|&position| space.forbidden_naming(position).iter().copied())
            .collect();
        clauses.sort_unstable();
        clauses.dedup();
        for index in clauses {
            self.check_forbidden_clause(index);
        }
        self.finish()
    }

    /// Return what a dependent of the decision at `position` reads of it:
    /// its activity, its chosen alternative and whether it holds a value.
    fn read_state(&self, position: usize) -> (Activity, Option<usize>, bool) {
        (
            *self.activities.get(position),
            *self.chosen.get(position),
            self.values.get(position).is_some(),
        )
    }

    /// Remove the value of each decision `names` names, refusing a name
    /// that is no decision.
    fn remove_entries(&mut self, names: impl IntoIterator<Item = Identifier>) {
        for name in names {
            match self.space.position(&name) {
                Some(position) => {
                    self.values.take(position);
                }
                None => self
                    .problems
                    .push(ConfigurationError::UnknownDecision { name }),
            }
        }
    }

    /// Give each entry's decision its value, refusing an entry that names
    /// no decision or one an earlier entry named.
    fn read_entries(&mut self, entries: impl IntoIterator<Item = (Identifier, Value)>) {
        for (name, value) in entries {
            let Some(position) = self.space.position(&name) else {
                self.problems
                    .push(ConfigurationError::UnknownDecision { name });
                continue;
            };
            if !self.given.insert(position) {
                self.problems
                    .push(ConfigurationError::DuplicateEntry { name });
                continue;
            }
            self.values.set(position, Some(value));
        }
    }

    /// Find the activity of the decision at `position` and check its value,
    /// dropping a refused one.
    fn check_decision(&mut self, position: usize) {
        let activity = self.find_activity(position);
        if *self.activities.get(position) != activity {
            self.activities.set(position, activity);
        }
        if self.values.get(position).is_none() {
            return;
        }
        let name = self.space.decision_at(position).name();
        if activity != Activity::Active {
            self.problems
                .push(ConfigurationError::InactiveDecision { name: name.clone() });
            self.values.take(position);
            self.forget_chosen(position);
            return;
        }
        let is_held = self.scope == Scope::Incremental && !self.given.contains(&position);
        if is_held {
            return;
        }
        let Some(value) = self.values.take(position) else {
            return;
        };
        self.forget_chosen(position);
        let accepted = match self.space.decision_at(position) {
            Decision::Choice(choice) => self.check_choice_value(position, choice, value),
            Decision::Variable(variable) => self.check_variable_value(variable, value),
        };
        if accepted.is_some() {
            self.values.set(position, accepted);
        }
    }

    /// Forget the alternative the choice at `position` chose, if any.
    fn forget_chosen(&mut self, position: usize) {
        if self.chosen.get(position).is_some() {
            self.chosen.set(position, None);
        }
    }

    /// Return `value` if it names an alternative of `choice`, at
    /// `position`, recording which.
    fn check_choice_value(
        &mut self,
        position: usize,
        choice: &Choice,
        value: Value,
    ) -> Option<Value> {
        let index = match &value {
            Value::Identifier(identifier) => choice
                .alternatives()
                .iter()
                .position(|alternative| alternative.get().name() == identifier),
            _ => None,
        };
        let Some(index) = index else {
            self.problems.push(ConfigurationError::UnknownAlternative {
                choice: choice.name().clone(),
                value,
            });
            return None;
        };
        self.chosen.set(position, Some(index));
        Some(value)
    }

    /// Return `value` if `variable`'s param takes it.
    fn check_variable_value(
        &mut self,
        variable: &Part<dyn Variable>,
        value: Value,
    ) -> Option<Value> {
        let variable = variable.get();
        let param = variable.param().clone();
        let assigned = match self.value_check {
            ValueCheck::New => ParamAssignment::new(param, value, self.context),
            ValueCheck::Restore => ParamAssignment::restore(param, value, self.context),
        };
        match assigned {
            Ok(assignment) => Some(assignment.value().clone()),
            Err(error) => {
                self.problems.push(ConfigurationError::Assignment {
                    variable: variable.name().clone(),
                    error,
                });
                None
            }
        }
    }

    /// Return the activity of the decision at `position`, given the
    /// decisions before it in decision order, reporting an undecided or
    /// failing condition.
    fn find_activity(&mut self, position: usize) -> Activity {
        let from_parent = match self.space.parent_at(position) {
            None => Activity::Active,
            Some((choice, alternative)) => match *self.activities.get(choice) {
                Activity::Active => match *self.chosen.get(choice) {
                    None => Activity::Pending,
                    Some(index) if index == alternative => Activity::Active,
                    Some(_) => Activity::Inactive,
                },
                other => other,
            },
        };
        if from_parent == Activity::Inactive {
            return Activity::Inactive;
        }
        let Some((when, references)) = self.space.condition_with_references_at(position) else {
            return from_parent;
        };
        if references
            .iter()
            .any(|&reference| *self.activities.get(reference) == Activity::Inactive)
        {
            return Activity::Inactive;
        }
        let Some(bindings) =
            bind_if_decided(self.space, references, &self.activities, &self.values)
        else {
            return Activity::Pending;
        };
        let target = self.space.decision_at(position).name();
        let from_condition = match evaluate(when, &bindings, self.context) {
            Ok(Outcome::Satisfied) => Activity::Active,
            Ok(Outcome::Violated) => Activity::Inactive,
            Ok(Outcome::Undecided) => {
                self.problems.push(ConfigurationError::UndecidedCondition {
                    target: target.clone(),
                });
                Activity::Pending
            }
            Err(error) => {
                self.problems.push(ConfigurationError::FailedCondition {
                    target: target.clone(),
                    error,
                });
                Activity::Pending
            }
        };
        match (from_parent, from_condition) {
            (_, Activity::Inactive) => Activity::Inactive,
            (Activity::Active, Activity::Active) => Activity::Active,
            _ => Activity::Pending,
        }
    }

    /// Check the forbidden clause at `index`, if its decisions are all
    /// active and assigned.
    fn check_forbidden_clause(&mut self, index: usize) {
        let clause = &self.space.forbidden()[index];
        let references = self.space.forbidden_references(index);
        let Some(bindings) =
            bind_if_decided(self.space, references, &self.activities, &self.values)
        else {
            return;
        };
        match evaluate(clause.when(), &bindings, self.context) {
            Ok(Outcome::Satisfied) => {
                self.problems.push(ConfigurationError::Forbidden { index });
            }
            Ok(Outcome::Violated) => {}
            Ok(Outcome::Undecided) => {
                self.problems
                    .push(ConfigurationError::UndecidedForbidden { index });
            }
            Err(error) => self
                .problems
                .push(ConfigurationError::FailedForbidden { index, error }),
        }
    }

    /// Return the configuration, or every problem found.
    fn finish(self) -> Result<Configuration, ConfigurationErrors> {
        if !self.problems.is_empty() {
            return Err(ConfigurationErrors::new(self.problems));
        }
        Ok(Configuration(Arc::new(ConfigurationInner {
            space: self.space.clone(),
            values: self.values,
            activities: self.activities,
            chosen: self.chosen,
        })))
    }
}

/// Return the bindings of the decisions at `references` to their values,
/// or `None` when one is inactive, pending or unassigned, so that what
/// names them cannot be decided yet.
fn bind_if_decided(
    space: &Space,
    references: &[usize],
    activities: &Chunked<Activity>,
    values: &Chunked<Option<Value>>,
) -> Option<Bindings> {
    references
        .iter()
        .map(|&reference| {
            if *activities.get(reference) != Activity::Active {
                return None;
            }
            let value = values.get(reference).clone()?;
            Some((
                space.decision_at(reference).name().clone(),
                Binding::Value(value),
            ))
        })
        .collect()
}

/// Return the outcome of `system` under `bindings`.
fn evaluate(
    system: &ConstraintSystem,
    bindings: &Bindings,
    context: &ParamContext<'_>,
) -> Result<Outcome, ConstraintError> {
    system.evaluate(bindings, context.constraint_context())
}

/// The assigned decisions of a configuration, in canonical order.
struct Entries<'a> {
    space: &'a Space,
    values: &'a Chunked<Option<Value>>,
    /// The canonical position to look at next.
    position: usize,
    remaining: usize,
}

impl<'a> Iterator for Entries<'a> {
    type Item = (&'a Identifier, &'a Value);

    fn next(&mut self) -> Option<Self::Item> {
        while self.position < self.values.len() {
            let position = self.position;
            self.position += 1;
            if let Some(value) = self.values.get(position) {
                self.remaining -= 1;
                return Some((self.space.decision_at(position).name(), value));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for Entries<'_> {}
