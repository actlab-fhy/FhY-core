//! Errors raised while building, comparing, or screening expressions.
//!
//! [`PiecewiseError`] and [`RebuildError`] refuse a node that would break an
//! invariant, [`NonInjectiveRenamingError`] a renaming or binder frame that
//! is not injective, and [`NonBooleanLogicalOperandError`] a number that a
//! [`BooleanScreen`](super::BooleanScreen) finds at a [`BooleanPosition`].

use std::error::Error;
use std::fmt;

use crate::identifier::Identifier;

use super::node::Expression;
use super::operation::LogicalOperation;

/// A piecewise that could not be built because it would break a piecewise
/// invariant: at least one case, and no case condition that is a literal
/// other than a Boolean.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::{Expression, PiecewiseError};
///
/// let no_cases: Vec<(Expression, Expression)> = Vec::new();
/// let result = Expression::piecewise(no_cases, 0);
/// assert_eq!(result, Err(PiecewiseError::NoCases));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PiecewiseError {
    /// The piecewise was given no cases.
    ///
    /// Displays as `piecewise has no cases`.
    NoCases,
    /// A case condition is a literal that is not a Boolean.
    ///
    /// Displays as `condition of piecewise case {case_index} is a
    /// non-boolean literal`.
    NonBooleanConditionLiteral {
        /// The zero-based index of the first offending case.
        case_index: usize,
    },
}

impl fmt::Display for PiecewiseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoCases => f.write_str("piecewise has no cases"),
            Self::NonBooleanConditionLiteral { case_index } => write!(
                f,
                "condition of piecewise case {case_index} is a non-boolean literal"
            ),
        }
    }
}

impl Error for PiecewiseError {}

/// A node that could not be rebuilt from new children.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::{Expression, LiteralValue, RebuildError};
///
/// let leaf = Expression::from(LiteralValue::from(1));
/// let result = leaf.rebuild_with_children(vec![leaf.clone()]);
/// assert_eq!(result, Err(RebuildError::ChildCount { expected: 0, actual: 1 }));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RebuildError {
    /// The number of children differs from the node's own child count.
    ///
    /// Displays as `expected {expected} children, got {actual}`.
    ChildCount {
        /// The number of children the node has.
        expected: usize,
        /// The number of children given.
        actual: usize,
    },
    /// The new children make an invalid piecewise.
    ///
    /// Displays as `invalid piecewise`; the [`PiecewiseError`] is the
    /// [`source`](Error::source).
    Piecewise(PiecewiseError),
}

impl fmt::Display for RebuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChildCount { expected, actual } => {
                write!(f, "expected {expected} children, got {actual}")
            }
            Self::Piecewise(_) => f.write_str("invalid piecewise"),
        }
    }
}

impl Error for RebuildError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ChildCount { .. } => None,
            Self::Piecewise(error) => Some(error),
        }
    }
}

/// A free-identifier renaming or a binder frame that sends two identifiers
/// to one image.
///
/// Returned by [`AlphaRenaming::try_new`](super::AlphaRenaming::try_new) for
/// the free renaming and by
/// [`AlphaRenaming::enter_binder`](super::AlphaRenaming::enter_binder) for a
/// frame. Displays as `a free-identifier renaming must be injective, but
/// more than one identifier maps to {name}::{id}`, or `a binder frame must
/// be injective, but ...` for a frame, with the shared image's name hint and
/// id.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
///
/// use fhy_core::expression::{AlphaRenaming, RenamingPart};
/// use fhy_core::identifier::Identifier;
///
/// let (a, b, c) = (Identifier::new("a"), Identifier::new("b"), Identifier::new("c"));
/// let mut renaming = AlphaRenaming::default();
///
/// let error = renaming
///     .enter_binder(HashMap::from([(a, c.clone()), (b, c.clone())]))
///     .expect_err("a and b share the image c");
///
/// assert_eq!(error.image(), &c);
/// assert_eq!(error.part(), RenamingPart::BinderFrame);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct NonInjectiveRenamingError {
    image: Identifier,
    part: RenamingPart,
}

impl NonInjectiveRenamingError {
    pub(super) fn new(image: Identifier, part: RenamingPart) -> Self {
        Self { image, part }
    }

    /// Return an image that more than one identifier maps to.
    #[must_use]
    pub fn image(&self) -> &Identifier {
        &self.image
    }

    /// Return the part of the renaming that is not injective.
    #[must_use]
    pub fn part(&self) -> RenamingPart {
        self.part
    }
}

impl fmt::Display for NonInjectiveRenamingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let part = match self.part {
            RenamingPart::FreeRenaming => "a free-identifier renaming",
            RenamingPart::BinderFrame => "a binder frame",
        };
        write!(
            f,
            "{part} must be injective, but more than one identifier maps to {}::{}",
            self.image.name_hint(),
            self.image.id()
        )
    }
}

/// The part of an [`AlphaRenaming`](super::AlphaRenaming) a
/// [`NonInjectiveRenamingError`] refuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RenamingPart {
    /// The free-identifier renaming given to
    /// [`AlphaRenaming::try_new`](super::AlphaRenaming::try_new).
    FreeRenaming,
    /// A binder frame given to
    /// [`AlphaRenaming::enter_binder`](super::AlphaRenaming::enter_binder).
    BinderFrame,
}

impl Error for NonInjectiveRenamingError {}

/// Where a Boolean position sits relative to the node that imposes it.
///
/// `Display` writes the phrase naming the position: `the operand of a
/// logical not`, `operand {operand_index} of a logical {and|or}`, `the
/// condition of piecewise case {case_index}`, `the value of piecewise case
/// {case_index}`, or `the otherwise branch of a piecewise`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BooleanPosition {
    /// The operand of a logical negation.
    NegatedOperand,
    /// An operand of a conjunction or a disjunction.
    LogicalOperand {
        /// The conjunction or disjunction the operand belongs to.
        operation: LogicalOperation,
        /// The zero-based position of the operand among the node's operands.
        operand_index: usize,
    },
    /// The condition of a piecewise case.
    CaseCondition {
        /// The zero-based index of the case.
        case_index: usize,
    },
    /// The value of a piecewise case, when the piecewise itself sits in a
    /// Boolean position.
    CaseValue {
        /// The zero-based index of the case.
        case_index: usize,
    },
    /// The otherwise branch of a piecewise that itself sits in a Boolean
    /// position.
    Otherwise,
}

impl fmt::Display for BooleanPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NegatedOperand => f.write_str("the operand of a logical not"),
            Self::LogicalOperand {
                operation,
                operand_index,
            } => write!(f, "operand {operand_index} of a logical {operation}"),
            Self::CaseCondition { case_index } => {
                write!(f, "the condition of piecewise case {case_index}")
            }
            Self::CaseValue { case_index } => write!(f, "the value of piecewise case {case_index}"),
            Self::Otherwise => f.write_str("the otherwise branch of a piecewise"),
        }
    }
}

/// A Boolean position holding an operand that provably denotes a number.
///
/// Carries the offending operand and, unless it is the root of a predicate,
/// the node that puts it in a Boolean position and the position within that
/// node. `Display` is one line that writes no expression: `{position}
/// provably denotes a number but sits in a boolean position`, with the
/// position's phrase (see [`BooleanPosition`]), or `the predicate provably
/// denotes a number` for a predicate root. Write [`operand`](Self::operand)
/// and [`parent`](Self::parent) with [`Expression::display`] to show the
/// expressions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonBooleanLogicalOperandError {
    operand: Expression,
    parent: Option<(Expression, BooleanPosition)>,
}

impl NonBooleanLogicalOperandError {
    pub(super) fn new(operand: Expression, parent: Expression, position: BooleanPosition) -> Self {
        Self {
            operand,
            parent: Some((parent, position)),
        }
    }

    /// Construct the error for `operand` as the root of a predicate.
    pub(super) fn new_predicate_root(operand: Expression) -> Self {
        Self {
            operand,
            parent: None,
        }
    }

    /// Return the operand that provably denotes a number.
    #[must_use]
    pub fn operand(&self) -> &Expression {
        &self.operand
    }

    /// Return the node that puts the operand in a Boolean position, and
    /// where the operand sits in it; `None` exactly when the operand is the
    /// root of a predicate.
    #[must_use]
    pub fn parent(&self) -> Option<(&Expression, BooleanPosition)> {
        self.parent
            .as_ref()
            .map(|(parent, position)| (parent, *position))
    }
}

impl fmt::Display for NonBooleanLogicalOperandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.parent {
            Some((_, position)) => write!(
                f,
                "{position} provably denotes a number but sits in a boolean position"
            ),
            None => f.write_str("the predicate provably denotes a number"),
        }
    }
}

impl Error for NonBooleanLogicalOperandError {}
