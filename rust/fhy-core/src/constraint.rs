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
//! undecided is reported to the [`ConstraintContext`]'s [`ConstraintObserver`] as an
//! [`ConstraintEvent`]. A value only its producer can compare, such as an object of
//! the Python binding, is an opaque value, a [`Part<dyn OpaqueValue>`](crate::foreign::Part).
//!
//! Each constraint has a canonical ordering key, a text equal for two
//! constraints exactly when they are structurally equivalent. An
//! equation's key writes its expression's distinct nodes once each, as a
//! table, so it is linear in them however the expression shares. A
//! [`CustomConstraint`], held in a [`Part`], answers
//! through fallible hooks, except the `eq_part` and `hash_part` behind
//! `==`: its failure is a [`ConstraintError::Custom`], so the key a
//! [`ConstraintSystem`] reads once when it is built, and a constraint's
//! scope, can fail.
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
//!     .map(|value| Member::try_from(Value::Int(value.into())))
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

use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem;

use crate::expression::Expression;
use crate::foreign::Part;
use crate::identifier::Identifier;
use crate::term::{AlphaEquivalence, AlphaRenaming, FreeIdentifiers};

pub use binding::{Binding, Bindings};
pub use context::{ConstraintContext, ConstraintEvent, ConstraintObserver, NoConstraintObserver};
pub use custom::CustomConstraint;
pub use equation::EquationConstraint;
pub use error::{ConstraintError, UnusableBindingReason};
pub use set::{Polarity, SetConstraint};
pub use system::ConstraintSystem;
pub use value::{Member, MemberError, MemberKind, MemberSet, OpaqueValue, Value};

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
    Custom(Part<dyn CustomConstraint>),
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
            (Self::Custom(left), Self::Custom(right)) => Part::ptr_eq(left, right),
            _ => false,
        }
    }

    /// Return the scope: every identifier the constraint refers to.
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError::Custom`] for a custom constraint that
    /// fails; a built-in kind never fails.
    pub fn free_identifiers(&self) -> Result<HashSet<Identifier>, ConstraintError> {
        match self {
            Self::Equation(constraint) => Ok(constraint.free_identifiers()),
            Self::Set(constraint) => Ok(constraint.free_identifiers()),
            Self::Custom(constraint) => constraint
                .get()
                .free_identifiers()
                .map_err(ConstraintError::Custom),
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
                .get()
                .evaluate(bindings, context)
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
            Self::Custom(constraint) => constraint
                .get()
                .to_expression()
                .map_err(ConstraintError::Custom),
        }
    }

    /// Return the canonical ordering key.
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError::Custom`] for a custom constraint whose
    /// key fails; a built-in kind never fails.
    pub fn ordering_key(&self) -> Result<String, ConstraintError> {
        match self {
            Self::Equation(constraint) => Ok(constraint.ordering_key()),
            Self::Set(constraint) => Ok(constraint.ordering_key()),
            Self::Custom(constraint) => constraint
                .get()
                .ordering_key()
                .map(Cow::into_owned)
                .map_err(ConstraintError::Custom),
        }
    }

    /// Return whether `other` is of the same kind and structurally equal.
    #[must_use]
    pub fn is_structurally_equivalent(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Equation(left), Self::Equation(right)) => left.is_structurally_equivalent(right),
            (Self::Set(left), Self::Set(right)) => left.is_structurally_equivalent(right),
            (Self::Custom(left), Self::Custom(right)) => left == right,
            _ => false,
        }
    }
}

impl AlphaEquivalence for Constraint {
    type Error = ConstraintError;

    /// Compare two constraints of the same kind: equations by their
    /// expressions, set constraints by their polarity, their members and
    /// the correspondence of their variables, and custom constraints
    /// through their hook.
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError::Custom`] for a custom constraint's
    /// failure.
    fn is_alpha_equivalent_under(
        &self,
        other: &Self,
        renaming: &AlphaRenaming,
    ) -> Result<bool, ConstraintError> {
        Ok(match (self, other) {
            (Self::Equation(left), Self::Equation(right)) => {
                left.is_alpha_equivalent_under(right, renaming)
            }
            (Self::Set(left), Self::Set(right)) => left.is_alpha_equivalent_under(right, renaming),
            (Self::Custom(left), Self::Custom(right)) => left
                .get()
                .is_alpha_equivalent_under(right.get(), renaming)
                .map_err(ConstraintError::Custom)?,
            _ => false,
        })
    }
}

impl FreeIdentifiers for Constraint {
    type Error = ConstraintError;

    fn free_identifiers(&self) -> Result<HashSet<Identifier>, ConstraintError> {
        Self::free_identifiers(self)
    }
}

impl PartialEq for Constraint {
    /// Compare as [`is_structurally_equivalent`](Self::is_structurally_equivalent)
    /// does: of one kind, built-in kinds structurally, and custom
    /// constraints through their [`eq_part`](CustomConstraint::eq_part).
    fn eq(&self, other: &Self) -> bool {
        self.is_structurally_equivalent(other)
    }
}

impl Eq for Constraint {}

impl Hash for Constraint {
    /// Feed the kind, then the constraint: a custom one through its
    /// [`hash_part`](CustomConstraint::hash_part).
    fn hash<H: Hasher>(&self, state: &mut H) {
        mem::discriminant(self).hash(state);
        match self {
            Self::Equation(constraint) => constraint.hash(state),
            Self::Set(constraint) => constraint.hash(state),
            Self::Custom(constraint) => constraint.hash(state),
        }
    }
}

impl fmt::Display for Constraint {
    /// Write an equation's expression, a set constraint as `x in {1, 2}`,
    /// and a custom constraint as its type name in angle brackets.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Equation(constraint) => write!(f, "{constraint}"),
            Self::Set(constraint) => write!(f, "{constraint}"),
            Self::Custom(constraint) => write!(f, "<{}>", constraint.get().type_name()),
        }
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
