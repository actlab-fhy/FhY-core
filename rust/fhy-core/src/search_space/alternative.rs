//! [`Alternative`]: one option of a choice, the trait other crates
//! implement, and [`PlainAlternative`], this module's own implementation.

#![expect(
    unused_variables,
    reason = "interface stub: the bodies are todo!() until the implementation"
)]

use std::borrow::Cow;
use std::hash::Hasher;

use crate::diagnostic::Note;
use crate::foreign::{BoxError, ForeignPart, Part, impl_part, is_same_part};
use crate::identifier::Identifier;
use crate::term::{AlphaEquivalence, AlphaRenaming};

use super::choice::Choice;
use super::error::{EquivalenceError, SpaceError};
use super::variable::Variable;

/// One option of a [`Choice`]: a name, the variables and sub-choices that
/// exist only while it is chosen, and the identifiers it binds.
///
/// Its [`name`](Self::name) is the value a configuration gives its choice
/// when it chooses it. Its [`bound_identifiers`](Self::bound_identifiers)
/// are labels its own data introduces, such as the axes of a walk, which
/// the space binds beside its decisions' and alternatives' names. A choice
/// holds an alternative as a [`Part<dyn Alternative>`](Part), so
/// implementations of several types mix in one choice.
///
/// The core compares the name, the variables, the sub-choices, the bound
/// identifiers' positions, the kind and the notes itself, and asks the
/// `is_extension_*` hooks only about the implementation's own data, after
/// it has checked that both sides have one [`kind`](Self::kind). An
/// implementation keeps the contract in the
/// [module documentation](super#implementing-variable-and-alternative).
pub trait Alternative: ForeignPart {
    /// Return the type id the implementation is registered under: unique
    /// to the implementing type, stable, and the type id its
    /// [`to_foreign`](ForeignPart::to_foreign) writes.
    fn kind(&self) -> Cow<'_, str>;

    /// Return the alternative's name, the value of its choice when chosen.
    fn name(&self) -> &Identifier;

    /// Return the variables that exist while the alternative is chosen, in
    /// order.
    fn variables(&self) -> &[Part<dyn Variable>];

    /// Return the choices that exist while the alternative is chosen, in
    /// order. The default holds none.
    fn choices(&self) -> &[Choice] {
        &[]
    }

    /// Return the notes attached to the alternative. The default holds
    /// none.
    fn notes(&self) -> &[Note] {
        &[]
    }

    /// Return the identifiers the alternative's own data binds, beyond its
    /// variables' and sub-choices' names, in a fixed order. The default
    /// binds none.
    ///
    /// They must be distinct, the same on every call, and as many for two
    /// alternatives that should correspond.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error, which fails building the choice
    /// that holds the alternative, or comparing the alternative on its own.
    fn bound_identifiers(&self) -> Result<Vec<Identifier>, BoxError> {
        Ok(Vec::new())
    }

    /// Return whether the implementation's own data equals `other`'s, for
    /// structural equivalence. `other` has the same
    /// [`kind`](Self::kind), so an implementation downcasts it to its own
    /// type, through [`as_any`](crate::foreign::AsAny::as_any), and answers
    /// `false` if that fails.
    ///
    /// The default has no data of its own and answers `true`.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error, which fails the comparison.
    fn is_extension_structurally_equivalent(
        &self,
        other: &dyn Alternative,
    ) -> Result<bool, BoxError> {
        let _ = other;
        Ok(true)
    }

    /// Return whether the implementation's own data corresponds to
    /// `other`'s under `renaming`, which already pairs every name of the
    /// space (or of the alternative compared on its own), the
    /// [`bound_identifiers`](Self::bound_identifiers) included. Identifiers
    /// in the data are compared only through
    /// [`AlphaRenaming::is_corresponding`].
    ///
    /// The default has no data of its own and answers `true`.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error, which fails the comparison.
    fn is_extension_alpha_equivalent_under(
        &self,
        other: &dyn Alternative,
        renaming: &AlphaRenaming,
    ) -> Result<bool, BoxError> {
        let _ = (other, renaming);
        Ok(true)
    }

    /// Return whether `other` is equal, for `==` on a
    /// [`Part<dyn Alternative>`](Part), and so for `==` on the choices and
    /// spaces that hold it.
    ///
    /// It must be an equivalence relation and agree with
    /// [`hash_part`](Self::hash_part). The default is identity: the same
    /// alternative.
    fn eq_part(&self, other: &dyn Alternative) -> bool {
        is_same_part(self, other)
    }

    /// Feed the alternative's hash to `state`, consistently with
    /// [`eq_part`](Self::eq_part). The default feeds nothing.
    fn hash_part(&self, state: &mut dyn Hasher) {
        let _ = state;
    }
}

impl_part!(Alternative);

impl Part<dyn Alternative> {
    /// Return whether `other` is the same alternative up to identity: of
    /// one kind, with equal names, structurally equivalent variables and
    /// sub-choices in order, equal bound identifiers and notes, and own
    /// data the
    /// [structural hook](Alternative::is_extension_structurally_equivalent)
    /// accepts.
    ///
    /// # Errors
    ///
    /// Returns [`EquivalenceError::Extension`] for a hook that fails,
    /// [`bound_identifiers`](Alternative::bound_identifiers) included.
    pub fn is_structurally_equivalent(&self, other: &Self) -> Result<bool, EquivalenceError> {
        todo!()
    }
}

impl AlphaEquivalence for Part<dyn Alternative> {
    type Error = EquivalenceError;

    /// Compare the two alternatives with their names paired as binders on
    /// top of `renaming`, in canonical order: the alternative's name, its
    /// bound identifiers, then its variables' names and its sub-choices'
    /// names, depth first. Under that frame: of one kind, the same numbers
    /// of variables, sub-choices and bound identifiers, each variable and
    /// sub-choice corresponding in order, equal notes, and own data the
    /// [alpha hook](Alternative::is_extension_alpha_equivalent_under)
    /// accepts.
    ///
    /// # Errors
    ///
    /// Returns [`EquivalenceError::Extension`] for a hook that fails,
    /// [`bound_identifiers`](Alternative::bound_identifiers) included, and
    /// [`EquivalenceError::Constraint`] for a custom constraint that fails.
    fn is_alpha_equivalent_under(
        &self,
        other: &Self,
        renaming: &AlphaRenaming,
    ) -> Result<bool, EquivalenceError> {
        todo!()
    }
}

/// This module's own [`Alternative`]: a name, variables, sub-choices and
/// notes, and no data or bound identifier of its own.
///
/// Its kind is [`PlainAlternative::KIND`]. `==` and `Hash` compare the
/// name, the variables, the sub-choices and the notes, and so does
/// [`Alternative::eq_part`]. Its wire form is `{"identifier", "variables",
/// "choices", "notes"}` (see [`wire`](super::wire)).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PlainAlternative {
    name: Identifier,
    variables: Vec<Part<dyn Variable>>,
    choices: Vec<Choice>,
    notes: Vec<Note>,
}

impl PlainAlternative {
    /// The kind of every plain alternative.
    pub const KIND: &'static str = "search_space.alternative";

    /// Return the alternative named `name` holding `variables` and the
    /// sub-choices `choices`, in order, with no notes.
    ///
    /// # Errors
    ///
    /// Returns [`SpaceError::DuplicateName`] naming the first name, in
    /// canonical order, that the alternative's name, its variables' names
    /// and the names its sub-choices hold repeat.
    pub fn new(
        name: Identifier,
        variables: Vec<Part<dyn Variable>>,
        choices: Vec<Choice>,
    ) -> Result<Self, SpaceError> {
        todo!()
    }

    /// Return this alternative with `notes` in place of its notes.
    #[must_use]
    pub fn with_notes(self, notes: Vec<Note>) -> Self {
        todo!()
    }
}

impl ForeignPart for PlainAlternative {
    /// Return `"PlainAlternative"`.
    fn type_name(&self) -> Cow<'_, str> {
        todo!()
    }
}

impl Alternative for PlainAlternative {
    /// Return [`PlainAlternative::KIND`].
    fn kind(&self) -> Cow<'_, str> {
        todo!()
    }

    fn name(&self) -> &Identifier {
        todo!()
    }

    fn variables(&self) -> &[Part<dyn Variable>] {
        todo!()
    }

    fn choices(&self) -> &[Choice] {
        todo!()
    }

    fn notes(&self) -> &[Note] {
        todo!()
    }

    /// Compare as `==` does, with a plain alternative only.
    fn eq_part(&self, other: &dyn Alternative) -> bool {
        todo!()
    }

    /// Feed the hash `Hash` feeds.
    fn hash_part(&self, state: &mut dyn Hasher) {
        todo!()
    }
}
