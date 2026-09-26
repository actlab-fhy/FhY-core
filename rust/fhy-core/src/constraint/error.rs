//! Errors of evaluating a constraint and of converting it to an expression.

use std::error::Error;
use std::fmt;

use crate::expression::{Expression, LiteralTextError, NonBooleanLogicalOperandError};
use crate::identifier::Identifier;
use crate::solver::SolveError;

use super::value::{Member, MemberKind, OpaqueError};

/// Why a constraint cannot use the value bound to an identifier in its
/// scope.
#[derive(Debug)]
#[non_exhaustive]
pub enum UnusableBindingReason {
    /// An equation needs an expression or a literal value, and the value is
    /// neither.
    NotALiteral,
    /// An equation needs an expression or a literal value, and the string
    /// is outside the literal grammar.
    UnparsableText(LiteralTextError),
    /// A set constraint needs an expression or a value that could be a
    /// member, and the value is neither.
    NotMemberShaped,
    /// A set constraint needs to look the value up, and it cannot be.
    Unhashable(OpaqueError),
}

/// A constraint that cannot be evaluated or converted.
#[derive(Debug)]
#[non_exhaustive]
pub enum ConstraintError {
    /// A constraint cannot use the value bound to an identifier in its
    /// scope.
    UnusableBinding {
        /// The identifier.
        identifier: Identifier,
        /// Why the value is unusable.
        reason: UnusableBindingReason,
    },
    /// An equation's expression, with its bindings, cannot be a predicate.
    IllTyped(NonBooleanLogicalOperandError),
    /// An equation simplified to a literal that is not a Boolean, so its
    /// expression denotes a number.
    NonBooleanResult {
        /// The equation's expression.
        predicate: Expression,
        /// The literal it simplified to.
        result: Expression,
    },
    /// A set constraint's member does not lift to a literal expression.
    UnliftableMember(Member),
    /// The solver refused or failed the question.
    Solve(SolveError),
}

impl fmt::Display for ConstraintError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnusableBinding { identifier, reason } => match reason {
                UnusableBindingReason::NotALiteral => write!(
                    f,
                    "the binding of {identifier:?} is neither an expression nor a literal"
                ),
                UnusableBindingReason::UnparsableText(_) => write!(
                    f,
                    "the binding of {identifier:?} cannot be lifted into a literal"
                ),
                UnusableBindingReason::NotMemberShaped => write!(
                    f,
                    "the binding of {identifier:?} is neither an expression nor a value that \
                     could be a member"
                ),
                UnusableBindingReason::Unhashable(_) => write!(
                    f,
                    "the binding of {identifier:?} is unhashable, so its membership cannot be \
                     checked"
                ),
            },
            Self::IllTyped(_) => f.write_str("the predicate is ill-typed"),
            Self::NonBooleanResult { predicate, result } => write!(
                f,
                "the predicate {predicate} simplified to the literal {result}, which is not a \
                 boolean, so it denotes a number"
            ),
            Self::UnliftableMember(member) => match member.kind() {
                MemberKind::Str(text) => write!(
                    f,
                    "the string member {text:?} cannot be converted to an expression: \
                     membership is type-strict, but literal equality would canonicalize the \
                     string against numeric members"
                ),
                _ => write!(
                    f,
                    "conversion of type {} to an expression is not supported",
                    member.kind_name()
                ),
            },
            Self::Solve(_) => f.write_str("the solver refused or failed the question"),
        }
    }
}

impl Error for ConstraintError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::UnusableBinding { reason, .. } => match reason {
                UnusableBindingReason::UnparsableText(error) => Some(error),
                UnusableBindingReason::Unhashable(error) => Some(&**error),
                UnusableBindingReason::NotALiteral | UnusableBindingReason::NotMemberShaped => None,
            },
            Self::IllTyped(error) => Some(error),
            Self::Solve(error) => Some(error),
            Self::NonBooleanResult { .. } | Self::UnliftableMember(_) => None,
        }
    }
}
