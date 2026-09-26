//! Errors of building domains and params and of deciding questions about
//! them.

use std::error::Error;
use std::fmt;

use crate::constraint::{ConstraintError, CustomError};
use crate::identifier::Identifier;

use super::domain::DomainKind;

/// The set operation a [`ParamError::KindMismatch`] refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SetOperation {
    /// The union of two value sets.
    Union,
    /// The intersection of two value sets.
    Intersection,
}

impl SetOperation {
    /// Return the operation's verb: `union` or `intersect`.
    #[must_use]
    pub fn verb(self) -> &'static str {
        match self {
            Self::Union => "union",
            Self::Intersection => "intersect",
        }
    }
}

/// A domain or param that cannot be built, or a question that cannot be
/// decided.
#[derive(Debug)]
#[non_exhaustive]
pub enum ParamError {
    /// A finite domain was given no value.
    EmptyValues(DomainKind),
    /// A finite domain was given a value that is no value of its kind,
    /// such as a container, or a float for a categorical domain.
    NotALeafValue {
        /// The domain's kind.
        kind: DomainKind,
        /// The value's position among the values given.
        index: usize,
    },
    /// A finite domain was given a NaN, which is unequal to itself.
    NanValue(DomainKind),
    /// An ordinal domain was given two values that do not order against
    /// each other.
    IncomparableValues,
    /// A finite domain was given two equal values.
    DuplicateValues(DomainKind),
    /// A domain forbids the kind of a constraint.
    ForbiddenConstraintKind(DomainKind),
    /// An interval domain was given an equation that is not a bound
    /// `x <cmp> k` of an integer `k`.
    NotABound,
    /// A set operation got a domain of another kind.
    KindMismatch {
        /// The operation.
        operation: SetOperation,
        /// The kind of the domain the operation was asked of.
        own: DomainKind,
        /// The kind of the other domain.
        other: DomainKind,
    },
    /// The union of two finite value sets is empty.
    EmptyUnion(DomainKind),
    /// The intersection of two finite value sets is empty.
    EmptyIntersection(DomainKind),
    /// Two permutation domains range over different members, so no value
    /// is admissible to both.
    DifferentPermutationMembers,
    /// A set constraint rescoped from `from` to `to` constrains another
    /// variable.
    Rescope {
        /// The variable the constraint was expected to constrain.
        from: Identifier,
        /// The variable it was to be rescoped to.
        to: Identifier,
        /// The variable it constrains.
        variable: Identifier,
    },
    /// A constraint of a kind the rescoping does not know, a custom one.
    UnexpectedConstraintKind,
    /// A constraint failed to evaluate or convert.
    Constraint(ConstraintError),
    /// A [`CustomDomain`](super::CustomDomain) failed.
    Custom(CustomError),
}

impl fmt::Display for ParamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyValues(kind) => {
                write!(f, "the values of {} must be non-empty", kind.article())
            }
            Self::NotALeafValue { kind, index } => {
                write!(f, "value {index} is no value {} admits", kind.article())
            }
            Self::NanValue(kind) => write!(
                f,
                "the values of {} must not include NaN: NaN is unequal to itself, so a NaN \
                 value could never be admitted",
                kind.article()
            ),
            Self::IncomparableValues => {
                f.write_str("ordinal values must be mutually comparable for sorting")
            }
            Self::DuplicateValues(kind) => {
                write!(f, "the values of {} must be unique", kind.article())
            }
            Self::ForbiddenConstraintKind(kind) => match kind {
                DomainKind::IntervalInteger => {
                    f.write_str("interval integer parameters only support equation constraints")
                }
                _ => write!(
                    f,
                    "only in-set and not-in-set constraints are allowed for {} parameters",
                    kind.name()
                ),
            },
            Self::NotABound => f.write_str(
                "interval integer parameters only support bound expressions of the form \
                 \"x >= k\", \"x > k\", \"x <= k\", or \"x < k\" where k is an integer",
            ),
            Self::KindMismatch {
                operation,
                own,
                other,
            } => write!(
                f,
                "cannot {} {} with {}",
                operation.verb(),
                own.article(),
                other.article()
            ),
            Self::EmptyUnion(kind) => {
                write!(f, "the union of the {} value sets is empty", kind.name())
            }
            Self::EmptyIntersection(kind) => {
                write!(
                    f,
                    "the intersection of the {} value sets is empty",
                    kind.name()
                )
            }
            Self::DifferentPermutationMembers => f.write_str(
                "the intersection of permutation domains with different member sets is empty",
            ),
            Self::Rescope { from, to, variable } => write!(
                f,
                "cannot rescope a set constraint from {from:?} to {to:?}: it is scoped to \
                 {variable:?}, not {from:?}"
            ),
            Self::UnexpectedConstraintKind => {
                f.write_str("cannot rescope a constraint of an unexpected kind")
            }
            Self::Constraint(_) => f.write_str("a constraint failed"),
            Self::Custom(_) => f.write_str("a custom domain failed"),
        }
    }
}

impl Error for ParamError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Constraint(error) => Some(error),
            Self::Custom(error) => Some(&**error),
            _ => None,
        }
    }
}

impl From<ConstraintError> for ParamError {
    fn from(error: ConstraintError) -> Self {
        Self::Constraint(error)
    }
}
