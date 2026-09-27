//! Errors of building domains and params and of deciding questions about
//! them.

use std::error::Error;
use std::fmt;

use crate::constraint::{Constraint, ConstraintError};
use crate::foreign::BoxError;
use crate::identifier::Identifier;

use super::domain::DomainKind;
use super::interval::BoundSide;

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
    /// A param's variable is a native constant's canonical identifier,
    /// which names a value rather than a variable.
    NativeConstantVariable(Identifier),
    /// A constraint's scope does not hold the param's variable.
    OutOfScope {
        /// The constraint.
        constraint: Constraint,
        /// The param's variable.
        variable: Identifier,
    },
    /// The bindings of a value check bind the param's own variable, whose
    /// value is the value checked.
    BindingsBindVariable(Identifier),
    /// A value is not admissible in the param's domain.
    Inadmissible,
    /// A value provably violates the param's constraint at `member`, in
    /// canonical order.
    ViolatedConstraint {
        /// The constraint's position.
        member: usize,
    },
    /// A value could not be verified against the param's constraint at
    /// `member`, in canonical order.
    UnverifiedConstraint {
        /// The constraint's position.
        member: usize,
    },
    /// An integer bound a non-negative domain's gate refuses: a bound
    /// literal the natural numbers do not admit.
    NaturalBound {
        /// Which bound.
        side: BoundSide,
        /// Whether the domain admits zero.
        zero_included: bool,
        /// Whether the bound admits its own value.
        is_inclusive: bool,
        /// Whether the bound is negative.
        is_negative: bool,
    },
    /// A lower bound exceeds its upper bound, or equals it with an
    /// exclusive side, so the bounds enclose no value.
    UnorderedBounds,
    /// The bounds of an interval param enclose no integer.
    EmptyInterval(Identifier),
    /// A param that is no interval operand, where interval arithmetic
    /// needs one.
    NotAnIntervalOperand,
    /// An operand of interval arithmetic that is neither an integer nor a
    /// param over an integer domain.
    UnsupportedOperand,
    /// An integer param carries a constraint that is not a bound, so it
    /// cannot be recast as an interval operand.
    NonBoundOperand(Option<ConstraintError>),
    /// An interval param holds a constraint that is not a bound, which its
    /// domain never allows.
    MalformedBound,
    /// A domain of `kind` represents no union.
    UnsupportedUnion(DomainKind),
    /// The intersection of two params is provably empty.
    EmptyParamIntersection,
    /// A constraint failed to evaluate or convert.
    Constraint(ConstraintError),
    /// A [`CustomDomain`](super::CustomDomain) failed.
    Custom(BoxError),
}

impl fmt::Display for ParamError {
    #[expect(clippy::too_many_lines, reason = "one arm per variant")]
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
            Self::NativeConstantVariable(variable) => write!(
                f,
                "the variable {variable:?} is a native constant's canonical identifier, which \
                 names a value rather than a variable"
            ),
            Self::OutOfScope { variable, .. } => write!(
                f,
                "a constraint's scope must include the param's variable {variable:?}"
            ),
            Self::BindingsBindVariable(variable) => write!(
                f,
                "the bindings must not bind the param's own variable {variable:?}, whose value \
                 is the value checked"
            ),
            Self::Inadmissible => f.write_str("the value is not admissible"),
            Self::ViolatedConstraint { member } => {
                write!(f, "the value violates the param's constraint {member}")
            }
            Self::UnverifiedConstraint { member } => write!(
                f,
                "the value could not be verified against the param's constraint {member}"
            ),
            Self::NaturalBound {
                side,
                zero_included,
                is_inclusive,
                is_negative,
            } => f.write_str(natural_bound_text(
                *side,
                *zero_included,
                *is_inclusive,
                *is_negative,
            )),
            Self::UnorderedBounds => {
                f.write_str("lower bound must be less than or equal to upper bound")
            }
            Self::EmptyInterval(variable) => write!(
                f,
                "empty integer interval represented by the constraints of {variable:?}"
            ),
            Self::NotAnIntervalOperand => {
                f.write_str("arithmetic is only supported on interval-integer parameters")
            }
            Self::UnsupportedOperand => f.write_str("unsupported operand of interval arithmetic"),
            Self::NonBoundOperand(_) => f.write_str(
                "cannot coerce an integer parameter with non-bound constraints to an interval \
                 parameter",
            ),
            Self::MalformedBound => {
                f.write_str("an interval parameter holds a constraint that is not a bound")
            }
            Self::UnsupportedUnion(kind) => {
                write!(f, "union is not supported for {}", kind.article())
            }
            Self::EmptyParamIntersection => {
                f.write_str("the intersection of the parameters is empty")
            }
            Self::Constraint(_) => f.write_str("a constraint failed"),
            Self::Custom(_) => f.write_str("a custom domain failed"),
        }
    }
}

/// Return the text of a bound the natural-number gate refuses.
fn natural_bound_text(
    side: BoundSide,
    zero_included: bool,
    is_inclusive: bool,
    is_negative: bool,
) -> &'static str {
    match (side, zero_included, is_inclusive) {
        (BoundSide::Lower, true, _) if is_negative => "lower bound must be non-negative",
        (BoundSide::Lower, true, _) => {
            "lower bound must be at least 1 if zero is included and bound is exclusive"
        }
        (BoundSide::Lower, false, true) => {
            "lower bound must be at least 1 when zero is not included"
        }
        (BoundSide::Lower, false, false) => {
            "lower bound must be non-negative when zero is not included and bound is exclusive"
        }
        (BoundSide::Upper, true, true) => "upper bound must be non-negative when zero is included",
        (BoundSide::Upper, true, false) => {
            "upper bound must be at least 1 if zero is included and bound is exclusive"
        }
        (BoundSide::Upper, false, true) => {
            "upper bound must be at least 1 when zero is not included"
        }
        (BoundSide::Upper, false, false) => {
            "upper bound must be at least 2 when zero is not included and bound is exclusive"
        }
    }
}

impl Error for ParamError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Constraint(error) | Self::NonBoundOperand(Some(error)) => Some(error),
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
