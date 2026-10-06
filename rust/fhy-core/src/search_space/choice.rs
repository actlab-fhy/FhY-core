//! [`Choice`]: a named decision among alternatives.

#![expect(
    unused_variables,
    dead_code,
    reason = "interface stub: the bodies are todo!() until the implementation"
)]

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::diagnostic::Note;
use crate::foreign::Part;
use crate::identifier::Identifier;
use crate::term::{AlphaEquivalence, AlphaRenaming};

use super::alternative::Alternative;
use super::error::{EquivalenceError, SpaceError};

/// A named decision among one or more [`Alternative`]s, in order.
///
/// A configuration gives a choice the name of the alternative it chooses.
/// Every name the choice holds is distinct: its own, its alternatives',
/// their bound identifiers, and the names of their variables and
/// sub-choices at every depth. Building one reads each alternative's
/// [`bound_identifiers`](Alternative::bound_identifiers) once.
///
/// Cloning one shares it. `==` and `Hash` compare the name, the
/// alternatives (through their [`eq_part`](Alternative::eq_part)) and the
/// notes. Its wire form is `{"identifier", "alternatives", "notes"}` (see
/// [`wire`](super::wire)).
#[derive(Debug, Clone)]
pub struct Choice(Arc<ChoiceInner>);

#[derive(Debug, Clone)]
struct ChoiceInner {
    name: Identifier,
    alternatives: Vec<Part<dyn Alternative>>,
    notes: Vec<Note>,
}

impl Choice {
    /// Return the choice named `name` among `alternatives`, in order, with
    /// no notes.
    ///
    /// # Errors
    ///
    /// In order: [`SpaceError::EmptyChoice`] for no alternative;
    /// [`SpaceError::Hook`] for an alternative whose
    /// [`bound_identifiers`](Alternative::bound_identifiers) fails; and
    /// [`SpaceError::DuplicateName`] naming the first name, in canonical
    /// order, that the choice's names repeat.
    pub fn new(
        name: Identifier,
        alternatives: Vec<Part<dyn Alternative>>,
    ) -> Result<Self, SpaceError> {
        todo!()
    }

    /// Return this choice with `notes` in place of its notes.
    #[must_use]
    pub fn with_notes(self, notes: Vec<Note>) -> Self {
        todo!()
    }

    /// Return the choice's name.
    #[must_use]
    pub fn name(&self) -> &Identifier {
        todo!()
    }

    /// Return the alternatives, in order.
    #[must_use]
    pub fn alternatives(&self) -> &[Part<dyn Alternative>] {
        todo!()
    }

    /// Return the notes.
    #[must_use]
    pub fn notes(&self) -> &[Note] {
        todo!()
    }

    /// Return whether `other` is the same choice up to identity: equal
    /// names and notes, and as many alternatives, each structurally
    /// equivalent to `other`'s at its position.
    ///
    /// # Errors
    ///
    /// Returns [`EquivalenceError::Extension`] for a hook that fails.
    pub fn is_structurally_equivalent(&self, other: &Self) -> Result<bool, EquivalenceError> {
        todo!()
    }
}

impl PartialEq for Choice {
    /// Compare the names, the alternatives in order, through their
    /// [`eq_part`](Alternative::eq_part), and the notes.
    fn eq(&self, other: &Self) -> bool {
        todo!()
    }
}

impl Eq for Choice {}

impl Hash for Choice {
    /// Feed the name, the alternatives through their
    /// [`hash_part`](Alternative::hash_part), and the notes.
    fn hash<H: Hasher>(&self, state: &mut H) {
        todo!()
    }
}

impl AlphaEquivalence for Choice {
    type Error = EquivalenceError;

    /// Compare the two choices with every name each holds paired as a
    /// binder on top of `renaming`, in canonical order: the choice's name,
    /// then each alternative's names. Under that frame: equal notes, and as
    /// many alternatives, each corresponding to `other`'s at its position.
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
