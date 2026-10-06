//! [`Configuration`]: a point of a space, checked against it, and its
//! [`ConfigurationKey`].
//!
//! This module owns a configuration's invariant: it is valid for its
//! space.

#![expect(
    unused_variables,
    dead_code,
    reason = "interface stub: the bodies are todo!() until the implementation"
)]

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::constraint::Value;
use crate::foreign::Part;
use crate::identifier::Identifier;
use crate::param::ParamContext;
use crate::term::{AlphaEquivalence, AlphaRenaming};

use super::alternative::Alternative;
use super::error::{ConfigurationErrors, EquivalenceError};
use super::space::Space;

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
        todo!()
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
        todo!()
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
        todo!()
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
        todo!()
    }

    /// Return the space the configuration is a point of.
    #[must_use]
    pub fn space(&self) -> &Space {
        todo!()
    }

    /// Return the value of the decision `name`, or `None` if it is
    /// unassigned or the space has no such decision.
    #[must_use]
    pub fn value(&self, name: &Identifier) -> Option<&Value> {
        todo!()
    }

    /// Return the alternative the choice `choice` chose, or `None` if it is
    /// unassigned or the space has no such choice.
    #[must_use]
    pub fn alternative(&self, choice: &Identifier) -> Option<&Part<dyn Alternative>> {
        todo!()
    }

    /// Return each assigned decision's name and value, in canonical order.
    #[expect(
        unreachable_code,
        reason = "interface stub: the empty iterator names the type until it is implemented"
    )]
    pub fn entries(&self) -> impl ExactSizeIterator<Item = (&Identifier, &Value)> + '_ {
        todo!();
        std::iter::empty()
    }

    /// Return the activity of the decision `name`, or `None` if the space
    /// has no such decision.
    #[must_use]
    pub fn activity(&self, name: &Identifier) -> Option<Activity> {
        todo!()
    }

    /// Return whether every decision is assigned or inactive.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        todo!()
    }

    /// Return the configuration's key within its space.
    #[must_use]
    pub fn key(&self) -> ConfigurationKey {
        todo!()
    }

    /// Return whether `other` has structurally equivalent spaces, as
    /// [`Space::is_structurally_equivalent`] compares them, and equal
    /// values.
    ///
    /// # Errors
    ///
    /// Returns [`EquivalenceError::Extension`] for a hook that fails.
    pub fn is_structurally_equivalent(&self, other: &Self) -> Result<bool, EquivalenceError> {
        todo!()
    }
}

impl PartialEq for Configuration {
    /// Compare the spaces, as [`Space`]'s `==` does, and the values.
    fn eq(&self, other: &Self) -> bool {
        todo!()
    }
}

impl Eq for Configuration {}

impl Hash for Configuration {
    /// Feed the space and the values.
    fn hash<H: Hasher>(&self, state: &mut H) {
        todo!()
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
        todo!()
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
        todo!()
    }
}

impl Eq for KeyValue {}

impl Hash for KeyValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        todo!()
    }
}
