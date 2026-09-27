//! [`SetConstraint`]: whether an identifier's value is a member of a set.

use std::collections::HashSet;
use std::sync::Arc;

use crate::expression::{Expression, ExpressionKind};
use crate::identifier::Identifier;
use crate::term::AlphaRenaming;

use super::Outcome;
use super::binding::{Binding, Bindings};
use super::context::{ConstraintContext, ConstraintEvent};
use super::error::{ConstraintError, UnusableBindingReason};
use super::key;
use super::value::{MemberSet, Value};

/// Whether a [`SetConstraint`] is satisfied by membership or by its
/// absence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Polarity {
    /// The value must be a member.
    In,
    /// The value must not be a member.
    NotIn,
}

/// Whether the value of one identifier, its variable, is a member of a set
/// of members, or is not, by its [`Polarity`].
///
/// The scope is the variable. Membership is type-strict (see
/// [`MemberSet`]). Cloning one shares it.
#[derive(Debug, Clone)]
pub struct SetConstraint(Arc<SetInner>);

#[derive(Debug)]
struct SetInner {
    variable: Identifier,
    members: MemberSet,
    polarity: Polarity,
}

impl SetConstraint {
    /// Return the constraint that `variable`'s value is (for
    /// [`Polarity::In`]) or is not (for [`Polarity::NotIn`]) one of
    /// `members`.
    #[must_use]
    pub fn new(variable: Identifier, members: MemberSet, polarity: Polarity) -> Self {
        Self(Arc::new(SetInner {
            variable,
            members,
            polarity,
        }))
    }

    /// Return whether `other` is this very constraint, or a clone of it.
    #[must_use]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        Arc::ptr_eq(&this.0, &other.0)
    }

    /// Return the variable.
    #[must_use]
    pub fn variable(&self) -> &Identifier {
        &self.0.variable
    }

    /// Return the members.
    #[must_use]
    pub fn members(&self) -> &MemberSet {
        &self.0.members
    }

    /// Return the polarity.
    #[must_use]
    pub fn polarity(&self) -> Polarity {
        self.0.polarity
    }

    /// Return the scope: the variable.
    #[must_use]
    pub fn free_identifiers(&self) -> HashSet<Identifier> {
        HashSet::from([self.0.variable.clone()])
    }

    /// Decide the constraint under `bindings`.
    ///
    /// It reads the variable's binding only. In order:
    ///
    /// 1. an unbound variable answers [`Outcome::Undecided`], reporting
    ///    [`ConstraintEvent::Unbound`];
    /// 2. a literal expression is decided by its value, a decimal being no
    ///    member; another expression cannot be decided;
    /// 3. another value must be member-shaped and hashable, and is decided
    ///    by type-strict membership;
    /// 4. a variable that is a native constant's canonical identifier
    ///    answers [`Outcome::Undecided`], reporting
    ///    [`ConstraintEvent::BoundNativeConstants`];
    /// 5. an expression that is not a literal answers
    ///    [`Outcome::Undecided`], reporting [`ConstraintEvent::SymbolicBinding`];
    /// 6. membership satisfies an [`In`](Polarity::In) constraint and
    ///    violates a [`NotIn`](Polarity::NotIn) one.
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError::UnusableBinding`] for a value that is not
    /// member-shaped, or not hashable.
    pub fn evaluate(
        &self,
        bindings: &Bindings,
        context: &ConstraintContext<'_>,
    ) -> Result<Outcome, ConstraintError> {
        let variable = self.variable();
        let Some(binding) = bindings.get(variable) else {
            context.notify(&ConstraintEvent::Unbound { variable });
            return Ok(Outcome::Undecided);
        };
        let is_member = match binding {
            Binding::Expression(expression) => match expression.kind() {
                ExpressionKind::Literal(literal) => {
                    Some(self.members().contains_value(&Value::from(literal.clone())))
                }
                _ => None,
            },
            Binding::Value(value) => {
                if !value.is_member_shaped() {
                    return Err(ConstraintError::UnusableBinding {
                        identifier: variable.clone(),
                        reason: UnusableBindingReason::NotMemberShaped,
                    });
                }
                value
                    .check_hashable()
                    .map_err(|error| ConstraintError::UnusableBinding {
                        identifier: variable.clone(),
                        reason: UnusableBindingReason::Unhashable(error),
                    })?;
                Some(self.members().contains_value(value))
            }
        };
        if context.is_native_constant(variable) {
            context.notify(&ConstraintEvent::BoundNativeConstants {
                identifiers: std::slice::from_ref(variable),
            });
            return Ok(Outcome::Undecided);
        }
        let Some(is_member) = is_member else {
            if let Binding::Expression(expression) = binding {
                context.notify(&ConstraintEvent::SymbolicBinding {
                    variable,
                    binding: expression,
                });
            }
            return Ok(Outcome::Undecided);
        };
        let is_satisfied = match self.polarity() {
            Polarity::In => is_member,
            Polarity::NotIn => !is_member,
        };
        Ok(if is_satisfied {
            Outcome::Satisfied
        } else {
            Outcome::Violated
        })
    }

    /// Return the expression equivalent to the constraint.
    ///
    /// With no member, it is `false` for [`In`](Polarity::In) and `true`
    /// for [`NotIn`](Polarity::NotIn). Otherwise each member, in canonical
    /// order, is compared with the variable, `variable == member` or
    /// `variable != member`, and several comparisons are one disjunction or
    /// conjunction.
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError::UnliftableMember`] for the first member,
    /// in canonical order, that does not lift to a literal.
    pub fn to_expression(&self) -> Result<Expression, ConstraintError> {
        let reference = Expression::from(self.variable());
        let mut comparisons = Vec::with_capacity(self.members().len());
        for member in self.members() {
            let literal = member
                .to_literal()
                .ok_or_else(|| ConstraintError::UnliftableMember(member.clone()))?;
            comparisons.push(match self.polarity() {
                Polarity::In => reference.equals(literal),
                Polarity::NotIn => reference.not_equals(literal),
            });
        }
        Ok(match self.polarity() {
            Polarity::In => Expression::any(comparisons),
            Polarity::NotIn => Expression::all(comparisons),
        })
    }

    /// Return the canonical ordering key: `in_set|` or `not_in_set|`, the
    /// variable's id, and the members' keys.
    #[must_use]
    pub fn ordering_key(&self) -> String {
        key::set_key(self)
    }

    /// Return whether `other` has the same polarity, variable and members.
    #[must_use]
    pub fn is_structurally_equivalent(&self, other: &Self) -> bool {
        self.polarity() == other.polarity()
            && self.variable() == other.variable()
            && self.members() == other.members()
    }

    /// Return whether `other` has the same polarity and members, and a
    /// variable corresponding to this one's under `renaming`.
    #[must_use]
    pub fn is_alpha_equivalent_under(&self, other: &Self, renaming: &AlphaRenaming) -> bool {
        self.polarity() == other.polarity()
            && renaming.is_corresponding(self.variable(), other.variable())
            && self.members() == other.members()
    }
}
