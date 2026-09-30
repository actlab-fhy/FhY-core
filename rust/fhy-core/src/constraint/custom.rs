//! [`CustomConstraint`]: a constraint defined outside this module, which a
//! [`ConstraintSystem`](super::ConstraintSystem) holds beside the built-in
//! kinds.

use std::borrow::Cow;
use std::collections::HashSet;
use std::hash::Hasher;

use crate::expression::Expression;
use crate::foreign::{BoxError, ForeignPart, impl_part, is_same_part};
use crate::identifier::Identifier;
use crate::term::AlphaRenaming;

use super::Outcome;
use super::binding::Bindings;
use super::context::ConstraintContext;

/// A constraint of a kind this module does not define, such as one a
/// language binding defines.
///
/// A system calls it as it needs: its key once, when the system is built,
/// and the others per question. Every hook but [`eq_part`](Self::eq_part)
/// and [`hash_part`](Self::hash_part) is fallible, and a failure is the
/// error of the operation that asked, as [`ConstraintError::Custom`](super::ConstraintError::Custom).
pub trait CustomConstraint: ForeignPart {
    /// Return the scope: every identifier the constraint refers to.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error.
    fn free_identifiers(&self) -> Result<HashSet<Identifier>, BoxError>;

    /// Decide the constraint under `bindings`, with the context a built-in
    /// constraint is evaluated with.
    ///
    /// The bindings may carry their caller's own form, which only the
    /// binding that built them reads (see [`Bindings::source`]).
    ///
    /// # Errors
    ///
    /// Returns the implementation's error.
    fn evaluate(
        &self,
        bindings: &Bindings,
        context: &ConstraintContext<'_>,
    ) -> Result<Outcome, BoxError>;

    /// Return the expression equivalent to the constraint.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error.
    fn to_expression(&self) -> Result<Expression, BoxError>;

    /// Return the canonical ordering key: equal for two constraints exactly
    /// when they are structurally equivalent ([`eq_part`](Self::eq_part)),
    /// and distinct from the built-in kinds' keys.
    ///
    /// Two inequivalent constraints with one key break this contract, as a
    /// `Hash` that disagrees with `==` does: a system still groups them by
    /// equivalence, and compares them as a multiset, but their order among
    /// themselves then follows the input.
    ///
    /// A [`ConstraintSystem`](super::ConstraintSystem) reads it once, when
    /// it is built.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error, which fails building the system.
    fn ordering_key(&self) -> Result<Cow<'_, str>, BoxError>;

    /// Return whether `other` is structurally equivalent, for `==` on a
    /// [`Part<dyn CustomConstraint>`](crate::foreign::Part) and for
    /// [`Constraint::is_structurally_equivalent`](super::Constraint::is_structurally_equivalent).
    ///
    /// It must be an equivalence relation, symmetric included, and agree
    /// with [`ordering_key`](Self::ordering_key) and
    /// [`hash_part`](Self::hash_part). The default is identity: the same
    /// constraint.
    fn eq_part(&self, other: &dyn CustomConstraint) -> bool {
        is_same_part(self, other)
    }

    /// Feed the constraint's hash to `state`, consistently with
    /// [`eq_part`](Self::eq_part). The default feeds nothing.
    fn hash_part(&self, state: &mut dyn Hasher) {
        let _ = state;
    }

    /// Return whether `other` is alpha-equivalent under `renaming`.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error.
    fn is_alpha_equivalent_under(
        &self,
        other: &dyn CustomConstraint,
        renaming: &AlphaRenaming,
    ) -> Result<bool, BoxError>;
}

impl_part!(CustomConstraint);
