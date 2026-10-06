//! [`Variable`]: a decision over a param's values, the trait other crates
//! implement, and [`PlainVariable`], this module's own implementation.

use std::borrow::Cow;
use std::hash::{Hash, Hasher};

use crate::diagnostic::Note;
use crate::foreign::{BoxError, ForeignPart, Part, impl_part, is_same_part};
use crate::identifier::Identifier;
use crate::param::Param;
use crate::term::{AlphaEquivalence, AlphaRenaming};

use super::domain::StepDomain;
use super::equivalence::{is_variable_alpha_equivalent, is_variable_structurally_equivalent};
use super::error::EquivalenceError;

/// A decision over the values of a [`Param`], named in its space.
///
/// Its [`name`](Self::name) is how conditions, forbidden clauses and
/// configurations refer to it; the param's own variable is a different
/// identifier, bound by the param, that only the param's constraints name.
/// A space holds a variable as a [`Part<dyn Variable>`](Part), so
/// implementations of several types mix in one space.
///
/// The core compares the name, the param, the kind and the notes itself,
/// and asks the `is_extension_*` hooks only about the implementation's own
/// data, after it has checked that both sides have one
/// [`kind`](Self::kind). An implementation keeps the contract in the
/// [module documentation](super#implementing-variable-and-alternative).
pub trait Variable: ForeignPart {
    /// Return the type id the implementation is registered under: unique
    /// to the implementing type, stable, and the type id its
    /// [`to_foreign`](ForeignPart::to_foreign) writes.
    fn kind(&self) -> Cow<'_, str>;

    /// Return the name the variable's space knows it by.
    fn name(&self) -> &Identifier;

    /// Return the param whose values the variable takes.
    fn param(&self) -> &Param;

    /// Return the notes attached to the variable. The default holds none.
    fn notes(&self) -> &[Note] {
        &[]
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
    fn is_extension_structurally_equivalent(&self, other: &dyn Variable) -> Result<bool, BoxError> {
        let _ = other;
        Ok(true)
    }

    /// Return whether the implementation's own data corresponds to
    /// `other`'s under `renaming`, which pairs every name of the space (or
    /// of the variable compared on its own). Identifiers in the data are
    /// compared only through
    /// [`AlphaRenaming::is_corresponding`].
    ///
    /// The default has no data of its own and answers `true`.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error, which fails the comparison.
    fn is_extension_alpha_equivalent_under(
        &self,
        other: &dyn Variable,
        renaming: &AlphaRenaming,
    ) -> Result<bool, BoxError> {
        let _ = (other, renaming);
        Ok(true)
    }

    /// Return whether `other` is equal, for `==` on a
    /// [`Part<dyn Variable>`](Part), and so for `==` on the choices and
    /// spaces that hold it.
    ///
    /// It must be an equivalence relation and agree with
    /// [`hash_part`](Self::hash_part). The default is identity: the same
    /// variable.
    fn eq_part(&self, other: &dyn Variable) -> bool {
        is_same_part(self, other)
    }

    /// Feed the variable's hash to `state`, consistently with
    /// [`eq_part`](Self::eq_part). The default feeds nothing.
    fn hash_part(&self, state: &mut dyn Hasher) {
        let _ = state;
    }

    /// Return the domain a static step over the variable offers, or `None`
    /// to derive it from the param: a categorical or ordinal param's
    /// values as a choice domain, a permutation param's as an order domain,
    /// and an integer param bounded at both ends as one strided run.
    ///
    /// An implementation whose param has a custom domain offers one here.
    /// It must hold exactly the values the param's domain admits, before
    /// its constraints, in an order that is the same for the value's life
    /// (contract clause 7). The default derives it.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error, which stops the run asking it.
    fn search_domain(&self) -> Result<Option<StepDomain>, BoxError> {
        Ok(None)
    }
}

impl_part!(Variable);

impl Part<dyn Variable> {
    /// Return whether `other` is the same variable up to identity: of one
    /// kind, with equal names, structurally equivalent params, equal notes,
    /// and own data the
    /// [structural hook](Variable::is_extension_structurally_equivalent)
    /// accepts.
    ///
    /// # Errors
    ///
    /// Returns [`EquivalenceError::Extension`] for a hook that fails.
    pub fn is_structurally_equivalent(&self, other: &Self) -> Result<bool, EquivalenceError> {
        is_variable_structurally_equivalent(self.get(), other.get())
    }
}

impl AlphaEquivalence for Part<dyn Variable> {
    type Error = EquivalenceError;

    /// Compare the two variables with their names paired as binders on top
    /// of `renaming`: of one kind, params whose domains correspond (an
    /// identifier member by [`AlphaRenaming::is_corresponding`]) and whose
    /// constraints correspond under the params' variables, equal notes,
    /// and own data the
    /// [alpha hook](Variable::is_extension_alpha_equivalent_under) accepts.
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
        is_variable_alpha_equivalent(self, other, renaming)
    }
}

/// This module's own [`Variable`]: a name, a param and notes, and no data
/// of its own.
///
/// Its kind is [`PlainVariable::KIND`]. `==` and `Hash` compare the name,
/// the param and the notes, and so does [`Variable::eq_part`]. Its wire
/// form is `{"identifier", "param", "notes"}` (see [`wire`](super::wire)).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PlainVariable {
    name: Identifier,
    param: Param,
    notes: Vec<Note>,
}

impl PlainVariable {
    /// The kind of every plain variable.
    pub const KIND: &'static str = "search_space.variable";

    /// Return the variable named `name` over `param`, with no notes.
    #[must_use]
    pub fn new(name: Identifier, param: Param) -> Self {
        Self {
            name,
            param,
            notes: Vec::new(),
        }
    }

    /// Return this variable with `notes` in place of its notes.
    #[must_use]
    pub fn with_notes(self, notes: Vec<Note>) -> Self {
        Self { notes, ..self }
    }
}

impl ForeignPart for PlainVariable {
    /// Return `"PlainVariable"`.
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed("PlainVariable")
    }
}

impl Variable for PlainVariable {
    /// Return [`PlainVariable::KIND`].
    fn kind(&self) -> Cow<'_, str> {
        Cow::Borrowed(Self::KIND)
    }

    fn name(&self) -> &Identifier {
        &self.name
    }

    fn param(&self) -> &Param {
        &self.param
    }

    fn notes(&self) -> &[Note] {
        &self.notes
    }

    /// Compare as `==` does, with a plain variable only.
    fn eq_part(&self, other: &dyn Variable) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| self == other)
    }

    /// Feed the hash `Hash` feeds.
    fn hash_part(&self, state: &mut dyn Hasher) {
        let mut state = state;
        self.hash(&mut state);
    }
}
