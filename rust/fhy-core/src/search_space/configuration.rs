//! [`Configuration`]: a point of a space, checked against it, and its
//! [`ConfigurationKey`].
//!
//! This module owns a configuration's invariant: it is valid for its
//! space.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::mem;
use std::sync::Arc;

use crate::constraint::{Binding, Bindings, ConstraintSystem, Outcome, Value};
use crate::foreign::Part;
use crate::identifier::Identifier;
use crate::param::{ParamAssignment, ParamContext};
use crate::term::{AlphaEquivalence, AlphaRenaming};

use super::alternative::Alternative;
use super::equivalence::{do_spaces_correspond, do_values_correspond, space_frame};
use super::error::{ConfigurationError, ConfigurationErrors, EquivalenceError};
use super::space::{Decision, Space};

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
/// it. Conditions and clauses are evaluated with each decision they name
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
    values: Vec<Option<Value>>,
    /// Each decision's activity, in canonical order.
    activities: Vec<Activity>,
    /// Each choice's chosen alternative's position, in canonical order.
    chosen: Vec<Option<usize>>,
    /// The canonical positions of the assigned decisions, ascending.
    assigned: Vec<usize>,
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
        check(space, Vec::new(), entries, context, ValueCheck::New)
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
        check(space, Vec::new(), entries, context, ValueCheck::Restore)
    }

    /// Return this configuration with the decision `name` given `value`,
    /// in place of its value if it has one, checked as
    /// [`new`](Self::new) checks a configuration.
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
    /// [`new`](Self::new) checks a configuration. Two of `entries` naming
    /// one decision are a
    /// [`DuplicateEntry`](super::ConfigurationError::DuplicateEntry).
    ///
    /// # Errors
    ///
    /// Returns [`ConfigurationErrors`] holding every problem found.
    pub fn with_entries(
        &self,
        entries: impl IntoIterator<Item = (Identifier, Value)>,
        context: &ParamContext<'_>,
    ) -> Result<Self, ConfigurationErrors> {
        check(
            &self.0.space,
            self.0.values.clone(),
            entries,
            context,
            ValueCheck::New,
        )
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
        self.0.values[position].as_ref()
    }

    /// Return the alternative the choice `choice` chose, or `None` if it is
    /// unassigned or the space has no such choice.
    #[must_use]
    pub fn alternative(&self, choice: &Identifier) -> Option<&Part<dyn Alternative>> {
        let position = self.0.space.position(choice)?;
        let index = self.0.chosen[position]?;
        match self.0.space.decision_at(position) {
            Decision::Choice(choice) => choice.alternatives().get(index),
            Decision::Variable(_) => None,
        }
    }

    /// Return each assigned decision's name and value, in canonical order.
    pub fn entries(&self) -> impl ExactSizeIterator<Item = (&Identifier, &Value)> + '_ {
        self.0.assigned.iter().map(|&position| {
            (
                self.0.space.decision_at(position).name(),
                self.0.values[position]
                    .as_ref()
                    .expect("an assigned position holds a value"),
            )
        })
    }

    /// Return the activity of the decision `name`, or `None` if the space
    /// has no such decision.
    #[must_use]
    pub fn activity(&self, name: &Identifier) -> Option<Activity> {
        self.0
            .space
            .position(name)
            .map(|position| self.0.activities[position])
    }

    /// Return whether every decision is assigned or inactive.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.0
            .activities
            .iter()
            .zip(&self.0.values)
            .all(|(activity, value)| *activity == Activity::Inactive || value.is_some())
    }

    /// Return the configuration's key within its space.
    #[must_use]
    pub fn key(&self) -> ConfigurationKey {
        let space = &self.0.space;
        let entries = (0..space.decision_count())
            .map(|position| {
                if self.0.activities[position] == Activity::Inactive {
                    return KeyEntry::Inactive;
                }
                if let Some(index) = self.0.chosen[position] {
                    return KeyEntry::Alternative(index);
                }
                match &self.0.values[position] {
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
            match (&left.values[position], &right.values[position]) {
                (None, None) => true,
                (Some(left_value), Some(right_value)) => match left.chosen[position] {
                    Some(index) => right.chosen[position] == Some(index),
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

/// One decision's entry in a [`ConfigurationKey`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum KeyEntry {
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
enum KeyValue {
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

/// Return the configuration of `space` holding `base` (each decision's
/// value, in canonical order, or none) with `entries` given on top, checked
/// as [`Configuration`] documents.
fn check(
    space: &Space,
    base: Vec<Option<Value>>,
    entries: impl IntoIterator<Item = (Identifier, Value)>,
    context: &ParamContext<'_>,
    value_check: ValueCheck,
) -> Result<Configuration, ConfigurationErrors> {
    let count = space.decision_count();
    let mut problems = Vec::new();
    let mut values = if base.is_empty() {
        vec![None; count]
    } else {
        base
    };
    let mut is_given = vec![false; count];
    for (name, value) in entries {
        let Some(position) = space.position(&name) else {
            problems.push(ConfigurationError::UnknownDecision { name });
            continue;
        };
        if is_given[position] {
            problems.push(ConfigurationError::DuplicateEntry { name });
            continue;
        }
        is_given[position] = true;
        values[position] = Some(value);
    }

    let mut activities = vec![Activity::Pending; count];
    let mut chosen: Vec<Option<usize>> = vec![None; count];
    for &position in space.order_positions() {
        let activity = find_activity(
            space,
            position,
            &activities,
            &values,
            &chosen,
            context,
            &mut problems,
        );
        activities[position] = activity;
        let Some(value) = values[position].take() else {
            continue;
        };
        let name = space.decision_at(position).name();
        if activity != Activity::Active {
            problems.push(ConfigurationError::InactiveDecision { name: name.clone() });
            continue;
        }
        match space.decision_at(position) {
            Decision::Choice(choice) => {
                let index = match &value {
                    Value::Identifier(identifier) => choice
                        .alternatives()
                        .iter()
                        .position(|alternative| alternative.get().name() == identifier),
                    _ => None,
                };
                let Some(index) = index else {
                    problems.push(ConfigurationError::UnknownAlternative {
                        choice: name.clone(),
                        value,
                    });
                    continue;
                };
                chosen[position] = Some(index);
            }
            Decision::Variable(variable) => {
                let param = variable.get().param().clone();
                let assigned = match value_check {
                    ValueCheck::New => ParamAssignment::new(param, value.clone(), context),
                    ValueCheck::Restore => ParamAssignment::restore(param, value.clone(), context),
                };
                if let Err(error) = assigned {
                    problems.push(ConfigurationError::Assignment {
                        variable: name.clone(),
                        error,
                    });
                    continue;
                }
            }
        }
        values[position] = Some(value);
    }

    for (index, clause) in space.forbidden().iter().enumerate() {
        let references = space.forbidden_references(index);
        let Some(bindings) = bind_if_decided(space, references, &activities, &values) else {
            continue;
        };
        match evaluate(clause.when(), &bindings, context) {
            Ok(Outcome::Satisfied) => problems.push(ConfigurationError::Forbidden { index }),
            Ok(Outcome::Violated) => {}
            Ok(Outcome::Undecided) => {
                problems.push(ConfigurationError::UndecidedForbidden { index });
            }
            Err(error) => problems.push(ConfigurationError::FailedForbidden { index, error }),
        }
    }

    if !problems.is_empty() {
        return Err(ConfigurationErrors::new(problems));
    }
    let assigned = (0..count)
        .filter(|&position| values[position].is_some())
        .collect();
    Ok(Configuration(Arc::new(ConfigurationInner {
        space: space.clone(),
        values,
        activities,
        chosen,
        assigned,
    })))
}

/// Return the activity of the decision at `position`, given the activities
/// and values of the decisions before it in decision order, reporting an
/// undecided or failing condition to `problems`.
fn find_activity(
    space: &Space,
    position: usize,
    activities: &[Activity],
    values: &[Option<Value>],
    chosen: &[Option<usize>],
    context: &ParamContext<'_>,
    problems: &mut Vec<ConfigurationError>,
) -> Activity {
    let from_parent = match space.parent_at(position) {
        None => Activity::Active,
        Some((choice, alternative)) => match activities[choice] {
            Activity::Active => match chosen[choice] {
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
    let Some((when, references)) = space.condition_with_references_at(position) else {
        return from_parent;
    };
    if references
        .iter()
        .any(|&reference| activities[reference] == Activity::Inactive)
    {
        return Activity::Inactive;
    }
    let Some(bindings) = bind_if_decided(space, references, activities, values) else {
        return Activity::Pending;
    };
    let target = || space.decision_at(position).name().clone();
    let from_condition = match evaluate(when, &bindings, context) {
        Ok(Outcome::Satisfied) => Activity::Active,
        Ok(Outcome::Violated) => Activity::Inactive,
        Ok(Outcome::Undecided) => {
            problems.push(ConfigurationError::UndecidedCondition { target: target() });
            Activity::Pending
        }
        Err(error) => {
            problems.push(ConfigurationError::FailedCondition {
                target: target(),
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

/// Return the bindings of the decisions at `references` to their values,
/// or `None` when one is inactive, pending or unassigned, so that what
/// names them cannot be decided yet.
fn bind_if_decided(
    space: &Space,
    references: &[usize],
    activities: &[Activity],
    values: &[Option<Value>],
) -> Option<Bindings> {
    references
        .iter()
        .map(|&reference| {
            if activities[reference] != Activity::Active {
                return None;
            }
            let value = values[reference].clone()?;
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
) -> Result<Outcome, crate::constraint::ConstraintError> {
    system.evaluate(bindings, context.constraint_context())
}
