//! Constraints: predicates over identifiers, decided under bindings of
//! their values.
//!
//! - An [`EquationConstraint`] is a Boolean expression that must hold. Its
//!   scope is the expression's free identifiers. It is decided by
//!   simplifying the expression under the bindings, with the
//!   [`Solver`](crate::solver::Solver)'s simplifier.
//! - A [`SetConstraint`] decides whether one identifier's value is a member
//!   of a [`MemberSet`], or is not, by its [`Polarity`]. Membership is
//!   type-strict: a Boolean, an integer and a float never compare equal.
//! - A [`Constraint`] is either, and answers an [`Outcome`].
//!
//! Values are bound as [`Bindings`], each an expression or a [`Value`],
//! which a constraint judges only when it reads it. Why an outcome is
//! undecided is reported to the [`ConstraintContext`]'s [`Observer`] as an
//! [`Event`]. A value only its producer can compare, such as an object of
//! the Python binding, is an [`Opaque`] value.
//!
//! Each constraint has a canonical ordering key, a text equal for two
//! constraints exactly when they are structurally equivalent.
//!
//! # Examples
//!
//! ```
//! use fhy_core::constraint::{
//!     Binding, Bindings, Constraint, ConstraintContext, Member, MemberSet, Outcome, Polarity,
//!     SetConstraint, Value,
//! };
//! use fhy_core::identifier::Identifier;
//! use fhy_core::solver::Solver;
//!
//! let x = Identifier::new("x");
//! let members: MemberSet = [1, 2, 3]
//!     .into_iter()
//!     .map(|value| Member::try_from_value(Value::Int(value.into())))
//!     .collect::<Result<_, _>>()?;
//! let constraint = Constraint::from(SetConstraint::new(x.clone(), members, Polarity::In));
//!
//! let solver = Solver::new();
//! let context = ConstraintContext::new(&solver);
//! let bindings = Bindings::from_iter([(x, Binding::Value(Value::Int(2.into())))]);
//!
//! assert_eq!(constraint.evaluate(&bindings, &context)?, Outcome::Satisfied);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod binding;
mod context;
mod custom;
mod equation;
mod error;
mod key;
mod set;
mod system;
mod value;
pub mod wire;

use std::collections::HashSet;
use std::sync::Arc;

use crate::expression::Expression;
use crate::identifier::Identifier;
use crate::term::{AlphaEquivalence, AlphaRenaming, FreeIdentifiers};

pub use binding::{Binding, Bindings};
pub use context::{ConstraintContext, Event, NoObserver, Observer};
pub use custom::{CustomConstraint, CustomError};
pub use equation::EquationConstraint;
pub use error::{ConstraintError, UnusableBindingReason};
pub use set::{Polarity, SetConstraint};
pub use system::ConstraintSystem;
pub use value::{
    Member, MemberError, MemberKind, MemberSet, Opaque, OpaqueError, OpaqueValue, Value,
};

/// The answer to whether a constraint holds.
#[expect(
    clippy::exhaustive_enums,
    reason = "a constraint question has exactly these three answers"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Outcome {
    /// It provably holds.
    Satisfied,
    /// It provably fails.
    Violated,
    /// It could not be decided.
    Undecided,
}

/// A constraint of one of the kinds this module defines.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Constraint {
    /// A Boolean expression that must hold.
    Equation(EquationConstraint),
    /// Membership of one identifier's value in a set, or its absence.
    Set(SetConstraint),
    /// A constraint of a kind defined elsewhere.
    Custom(Arc<dyn CustomConstraint>),
}

impl Constraint {
    /// Return whether `other` is this very constraint, or a clone of it,
    /// rather than an equal one built apart.
    #[must_use]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        match (this, other) {
            (Self::Equation(left), Self::Equation(right)) => {
                Expression::ptr_eq(left.expression(), right.expression())
            }
            (Self::Set(left), Self::Set(right)) => SetConstraint::ptr_eq(left, right),
            (Self::Custom(left), Self::Custom(right)) => Arc::ptr_eq(left, right),
            _ => false,
        }
    }

    /// Return the scope: every identifier the constraint refers to.
    #[must_use]
    pub fn free_identifiers(&self) -> HashSet<Identifier> {
        match self {
            Self::Equation(constraint) => constraint.free_identifiers(),
            Self::Set(constraint) => constraint.free_identifiers(),
            Self::Custom(constraint) => constraint.free_identifiers(),
        }
    }

    /// Decide the constraint under `bindings`, as
    /// [`EquationConstraint::evaluate`] and [`SetConstraint::evaluate`] do.
    ///
    /// # Errors
    ///
    /// Returns what the kind's evaluation returns.
    pub fn evaluate(
        &self,
        bindings: &Bindings,
        context: &ConstraintContext<'_>,
    ) -> Result<Outcome, ConstraintError> {
        match self {
            Self::Equation(constraint) => constraint.evaluate(bindings, context),
            Self::Set(constraint) => constraint.evaluate(bindings, context),
            Self::Custom(constraint) => constraint
                .evaluate(bindings)
                .map_err(ConstraintError::Custom),
        }
    }

    /// Return the expression equivalent to the constraint: an equation's
    /// expression, or [`SetConstraint::to_expression`].
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError::UnliftableMember`] for a set constraint
    /// with a member that does not lift.
    pub fn to_expression(&self) -> Result<Expression, ConstraintError> {
        match self {
            Self::Equation(constraint) => Ok(constraint.expression().clone()),
            Self::Set(constraint) => constraint.to_expression(),
            Self::Custom(constraint) => constraint.to_expression().map_err(ConstraintError::Custom),
        }
    }

    /// Return the canonical ordering key.
    #[must_use]
    pub fn ordering_key(&self) -> String {
        match self {
            Self::Equation(constraint) => constraint.ordering_key(),
            Self::Set(constraint) => constraint.ordering_key(),
            Self::Custom(constraint) => constraint.ordering_key().into_owned(),
        }
    }

    /// Return whether `other` is of the same kind and structurally equal.
    #[must_use]
    pub fn is_structurally_equivalent(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Equation(left), Self::Equation(right)) => left.is_structurally_equivalent(right),
            (Self::Set(left), Self::Set(right)) => left.is_structurally_equivalent(right),
            (Self::Custom(left), Self::Custom(right)) => {
                left.is_structurally_equivalent(right.as_ref())
            }
            _ => false,
        }
    }
}

impl AlphaEquivalence for Constraint {
    /// Compare two constraints of the same kind: equations by their
    /// expressions, set constraints by their polarity, their members and
    /// the correspondence of their variables.
    fn is_alpha_equivalent_under(&self, other: &Self, renaming: &AlphaRenaming) -> bool {
        match (self, other) {
            (Self::Equation(left), Self::Equation(right)) => {
                left.is_alpha_equivalent_under(right, renaming)
            }
            (Self::Set(left), Self::Set(right)) => left.is_alpha_equivalent_under(right, renaming),
            (Self::Custom(left), Self::Custom(right)) => {
                left.is_alpha_equivalent_under(right.as_ref(), renaming)
            }
            _ => false,
        }
    }
}

impl FreeIdentifiers for Constraint {
    fn free_identifiers(&self) -> HashSet<Identifier> {
        Self::free_identifiers(self)
    }
}

impl From<EquationConstraint> for Constraint {
    fn from(constraint: EquationConstraint) -> Self {
        Self::Equation(constraint)
    }
}

impl From<SetConstraint> for Constraint {
    fn from(constraint: SetConstraint) -> Self {
        Self::Set(constraint)
    }
}
