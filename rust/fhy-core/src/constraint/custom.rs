//! [`CustomConstraint`]: a constraint defined outside this module, which a
//! [`ConstraintSystem`](super::ConstraintSystem) holds beside the built-in
//! kinds.

use std::any::Any;
use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt;

use crate::expression::Expression;
use crate::foreign::{BoxError, Foreign, ForeignError};
use crate::identifier::Identifier;
use crate::term::AlphaRenaming;

use super::Outcome;
use super::binding::Bindings;

/// A constraint of a kind this module does not define, such as one a
/// language binding defines.
///
/// A system calls it as it needs: its key once, when the system is built,
/// and the others per question.
pub trait CustomConstraint: Send + Sync + fmt::Debug {
    /// Return the scope: every identifier the constraint refers to. An
    /// implementation that fails reports no identifier.
    fn free_identifiers(&self) -> HashSet<Identifier>;

    /// Decide the constraint under `bindings`, which may carry their
    /// caller's own form (see [`Bindings::source`]).
    ///
    /// # Errors
    ///
    /// Returns the implementation's error.
    fn evaluate(&self, bindings: &Bindings) -> Result<Outcome, BoxError>;

    /// Return the expression equivalent to the constraint.
    ///
    /// # Errors
    ///
    /// Returns the implementation's error.
    fn to_expression(&self) -> Result<Expression, BoxError>;

    /// Return the canonical ordering key: equal for structurally
    /// equivalent constraints, and distinct from the built-in kinds' keys.
    fn ordering_key(&self) -> Cow<'_, str>;

    /// Return whether `other` is structurally equivalent.
    fn is_structurally_equivalent(&self, other: &dyn CustomConstraint) -> bool;

    /// Return whether `other` is alpha-equivalent under `renaming`.
    fn is_alpha_equivalent_under(
        &self,
        other: &dyn CustomConstraint,
        renaming: &AlphaRenaming,
    ) -> bool;

    /// Return the constraint as a [`Foreign`] part, for serialization.
    ///
    /// # Errors
    ///
    /// The default returns [`ForeignError::NoWireForm`]: the constraint
    /// cannot be serialized.
    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        Err(ForeignError::NoWireForm {
            type_name: "custom constraint".to_owned(),
        })
    }

    /// Return the constraint as [`Any`], so an implementation can recognize
    /// its own constraints.
    fn as_any(&self) -> &dyn Any;
}
