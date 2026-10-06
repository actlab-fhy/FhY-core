//! [`Choice`]: a named decision among alternatives.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::diagnostic::Note;
use crate::foreign::Part;
use crate::identifier::Identifier;
use crate::term::{AlphaEquivalence, AlphaRenaming};

use super::alternative::Alternative;
use super::equivalence::{
    alternative_labels, is_choice_alpha_equivalent, is_choice_structurally_equivalent,
};
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
    /// Every name the choice holds, in canonical order.
    labels: Vec<Identifier>,
}

/// Return the first of `labels`, in order, that an earlier one equals.
pub(super) fn first_repeat(labels: &[Identifier]) -> Option<&Identifier> {
    let mut seen = HashSet::with_capacity(labels.len());
    labels.iter().find(|label| !seen.insert(*label))
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
        if alternatives.is_empty() {
            return Err(SpaceError::EmptyChoice { choice: name });
        }
        let mut labels = vec![name.clone()];
        for alternative in &alternatives {
            let alternative = alternative.get();
            labels.extend(
                alternative_labels(alternative).map_err(|source| SpaceError::Hook {
                    alternative: alternative.name().clone(),
                    source,
                })?,
            );
        }
        if let Some(repeated) = first_repeat(&labels) {
            return Err(SpaceError::DuplicateName {
                name: repeated.clone(),
            });
        }
        Ok(Self(Arc::new(ChoiceInner {
            name,
            alternatives,
            notes: Vec::new(),
            labels,
        })))
    }

    /// Return this choice with `notes` in place of its notes.
    #[must_use]
    pub fn with_notes(self, notes: Vec<Note>) -> Self {
        let mut inner = Arc::unwrap_or_clone(self.0);
        inner.notes = notes;
        Self(Arc::new(inner))
    }

    /// Return the choice's name.
    #[must_use]
    pub fn name(&self) -> &Identifier {
        &self.0.name
    }

    /// Return the alternatives, in order.
    #[must_use]
    pub fn alternatives(&self) -> &[Part<dyn Alternative>] {
        &self.0.alternatives
    }

    /// Return the notes.
    #[must_use]
    pub fn notes(&self) -> &[Note] {
        &self.0.notes
    }

    /// Return every name the choice holds, in canonical order: its own,
    /// then each alternative's.
    pub(super) fn labels(&self) -> &[Identifier] {
        &self.0.labels
    }

    /// Return whether `other` is the same choice up to identity: equal
    /// names and notes, and as many alternatives, each structurally
    /// equivalent to `other`'s at its position.
    ///
    /// # Errors
    ///
    /// Returns [`EquivalenceError::Extension`] for a hook that fails.
    pub fn is_structurally_equivalent(&self, other: &Self) -> Result<bool, EquivalenceError> {
        is_choice_structurally_equivalent(self, other)
    }
}

impl PartialEq for Choice {
    /// Compare the names, the alternatives in order, through their
    /// [`eq_part`](Alternative::eq_part), and the notes.
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
            || (self.0.name == other.0.name
                && self.0.alternatives == other.0.alternatives
                && self.0.notes == other.0.notes)
    }
}

impl Eq for Choice {}

impl Hash for Choice {
    /// Feed the name, the alternatives through their
    /// [`hash_part`](Alternative::hash_part), and the notes.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.name.hash(state);
        self.0.alternatives.hash(state);
        self.0.notes.hash(state);
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
        is_choice_alpha_equivalent(self, other, renaming)
    }
}
