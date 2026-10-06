//! [`Space`]: the decisions of a search, with the [`Condition`]s on when
//! each is active and the [`Forbidden`] combinations of their values.
//!
//! This module owns the space's invariants: the names, the scopes of the
//! conditions and forbidden clauses, and the canonical and decision
//! orders.

#![expect(
    unused_variables,
    dead_code,
    reason = "interface stub: the bodies are todo!() until the implementation"
)]

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::constraint::ConstraintSystem;
use crate::diagnostic::Note;
use crate::foreign::Part;
use crate::identifier::Identifier;
use crate::term::{AlphaEquivalence, AlphaRenaming};

use super::choice::Choice;
use super::error::{EquivalenceError, SpaceError};
use super::variable::Variable;

/// When the decision [`target`](Self::target) is active: while the
/// constraints [`when`](Self::when) holds, read under the values of the
/// decisions it names.
///
/// A condition names a choice only in a set constraint whose members are
/// the choice's alternatives' names, and names no decision under its
/// target. [`Space::new`] checks both. `==` and `Hash` compare the target
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
        todo!()
    }

    /// Return the decision the condition is on.
    #[must_use]
    pub fn target(&self) -> &Identifier {
        todo!()
    }

    /// Return the constraints that must hold.
    #[must_use]
    pub fn when(&self) -> &ConstraintSystem {
        todo!()
    }
}

/// A combination of values no configuration may take: one that satisfies
/// the constraints [`when`](Self::when) once every decision they name is
/// active and assigned.
///
/// A positive rule `p` across decisions is written as the forbidden clause
/// `not p`. A forbidden clause names a choice only in a set constraint
/// whose members are the choice's alternatives' names, and names at least
/// one decision. [`Space::new`] checks both. `==` and `Hash` compare the
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
        todo!()
    }

    /// Return the constraints a configuration must not satisfy.
    #[must_use]
    pub fn when(&self) -> &ConstraintSystem {
        todo!()
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
        todo!()
    }
}

/// A search space: top-level variables and choices, the conditions on when
/// each decision is active, and the forbidden combinations of values.
///
/// Building one checks, in order, that:
///
/// 1. every name it holds is distinct: its own, and every decision's,
///    alternative's and bound identifier's at every depth, read in
///    canonical order;
/// 2. each condition's target is a decision; every identifier the
///    condition names is a decision outside the target's subtree (the
///    target and everything under its alternatives); and none of its
///    equations names a choice;
/// 3. each forbidden clause names at least one decision, only decisions,
///    and no choice in an equation;
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
    conditions: Vec<Condition>,
    forbidden: Vec<Forbidden>,
    notes: Vec<Note>,
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
    /// [`SpaceError::EquationOverChoice`] and
    /// [`SpaceError::ConditionReferencesSubtree`]; for each forbidden
    /// clause in order, [`SpaceError::EmptyForbidden`],
    /// [`SpaceError::UnknownReference`] and
    /// [`SpaceError::EquationOverChoice`]; then
    /// [`SpaceError::CyclicDependency`]. Within one condition or clause,
    /// the names it refers to are checked in the order of their ids.
    /// [`SpaceError::Constraint`] reports a custom constraint whose scope
    /// or key fails.
    pub fn new(
        name: Identifier,
        variables: Vec<Part<dyn Variable>>,
        choices: Vec<Choice>,
        conditions: Vec<Condition>,
        forbidden: Vec<Forbidden>,
    ) -> Result<Self, SpaceError> {
        todo!()
    }

    /// Return this space with `notes` in place of its notes.
    #[must_use]
    pub fn with_notes(self, notes: Vec<Note>) -> Self {
        todo!()
    }

    /// Return the space's name.
    #[must_use]
    pub fn name(&self) -> &Identifier {
        todo!()
    }

    /// Return the top-level variables, in order.
    #[must_use]
    pub fn variables(&self) -> &[Part<dyn Variable>] {
        todo!()
    }

    /// Return the top-level choices, in order.
    #[must_use]
    pub fn choices(&self) -> &[Choice] {
        todo!()
    }

    /// Return the conditions, one per target, in canonical order of their
    /// targets.
    #[must_use]
    pub fn conditions(&self) -> &[Condition] {
        todo!()
    }

    /// Return the forbidden clauses, in the order given.
    #[must_use]
    pub fn forbidden(&self) -> &[Forbidden] {
        todo!()
    }

    /// Return the notes.
    #[must_use]
    pub fn notes(&self) -> &[Note] {
        todo!()
    }

    /// Return every decision, at every depth, in canonical order.
    #[expect(
        unreachable_code,
        reason = "interface stub: the empty iterator names the type until it is implemented"
    )]
    pub fn decisions(&self) -> impl ExactSizeIterator<Item = Decision<'_>> + '_ {
        todo!();
        std::iter::empty()
    }

    /// Return the decision named `name`, if the space has one.
    #[must_use]
    pub fn decision(&self, name: &Identifier) -> Option<Decision<'_>> {
        todo!()
    }

    /// Return every decision's name in decision order: each after its
    /// choice and after every decision its condition names, and otherwise
    /// in canonical order. Without conditions it is the canonical order.
    #[must_use]
    pub fn decision_order(&self) -> &[Identifier] {
        todo!()
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
        todo!()
    }
}

impl PartialEq for Space {
    /// Compare the names, the variables and choices in order (parts
    /// through their `eq_part`), the conditions, the forbidden clauses and
    /// the notes.
    fn eq(&self, other: &Self) -> bool {
        todo!()
    }
}

impl Eq for Space {}

impl Hash for Space {
    /// Feed what `==` compares, parts through their `hash_part`.
    fn hash<H: Hasher>(&self, state: &mut H) {
        todo!()
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
        todo!()
    }
}
