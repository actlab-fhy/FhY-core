//! Errors raised while building, comparing, or screening expressions.
//!
//! [`PiecewiseError`] reports a piecewise that could not be built because it
//! breaks a piecewise invariant, [`RebuildError`] a node that could not be
//! rebuilt from new children, and [`FunctionNameError`] a call name that
//! could not be accepted.
//! [`NonInjectiveRenamingError`] reports a free-identifier renaming that
//! sends two identifiers to one image.
//! [`NonBooleanLogicalOperandError`] reports a Boolean position that holds
//! an operand provably denoting a number, as found by
//! [`validate_logical_operands`](super::validate_logical_operands) and
//! [`validate_predicate`](super::validate_predicate); its
//! [`BooleanPosition`] says where that operand sits.

use std::error::Error;
use std::fmt;

use crate::identifier::Identifier;

use super::display::{FormatOptions, IdentifierStyle};
use super::node::Expression;
use super::operation::LogicalOperation;

/// A piecewise that could not be built because it would break a piecewise
/// invariant: at least one case, and no case condition that is a literal
/// other than a Boolean.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::{Expression, PiecewiseError, build_piecewise};
///
/// let no_cases: Vec<(Expression, Expression)> = Vec::new();
/// let result = build_piecewise(no_cases, 0);
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
/// use fhy_core::expr::{Expression, LiteralValue, RebuildError};
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

/// A call function name that could not be accepted.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::{FunctionNameError, build_call};
///
/// let result = build_call("", [1]);
/// assert_eq!(result, Err(FunctionNameError::Empty));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FunctionNameError {
    /// The name is empty.
    ///
    /// Displays as `function name is empty`.
    Empty,
}

impl fmt::Display for FunctionNameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("function name is empty"),
        }
    }
}

impl Error for FunctionNameError {}

/// A free-identifier renaming that sends two identifiers to one image.
///
/// Returned by [`AlphaRenaming::try_new`](super::AlphaRenaming::try_new).
/// Displays as `a free-identifier renaming must be injective, but more than
/// one identifier maps to {name}::{id}`, with the shared image's name hint
/// and id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonInjectiveRenamingError {
    image: Identifier,
}

impl NonInjectiveRenamingError {
    /// Construct the error for the shared `image`.
    pub(super) fn new(image: Identifier) -> Self {
        Self { image }
    }

    /// Return an image that more than one identifier maps to.
    #[must_use]
    pub fn image(&self) -> &Identifier {
        &self.image
    }
}

impl fmt::Display for NonInjectiveRenamingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "a free-identifier renaming must be injective, but more than one identifier maps to \
             {}::{}",
            self.image.name_hint(),
            self.image.id()
        )
    }
}

impl Error for NonInjectiveRenamingError {}

/// Where a Boolean position sits relative to the node that imposes it.
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
    /// The root of an expression used as a predicate.
    PredicateRoot,
}

/// A Boolean position holding an operand that provably denotes a number.
///
/// Carries the offending operand, the node that puts it in a Boolean
/// position (none for [`BooleanPosition::PredicateRoot`]), and the position
/// itself. `Display` names the position and writes the parent and the
/// operand as [`Expression::display`] does in [`Notation::Symbolic`] with
/// [`IdentifierStyle::NameHintWithId`], so a tree of any depth displays
/// without exhausting the thread's stack:
///
/// - negated operand: `{parent} applies the Boolean connective logical_not
///   to the operand {operand}, which provably denotes a number`
/// - conjunction or disjunction operand: `{parent} applies the Boolean
///   connective {operation} to the operand {operand}, which provably
///   denotes a number`, with the operation's wire name
/// - case condition: `{parent} takes {operand} as the condition of case
///   {case_index}, which provably denotes a number`
/// - case value: `{parent} takes {operand} as the value of case
///   {case_index}, which provably denotes a number`
/// - otherwise branch: `{parent} takes {operand} as its otherwise branch,
///   which provably denotes a number`
/// - predicate root: `{operand} is used as a predicate but provably
///   denotes a number`
///
/// followed in every case by `; the expression is ill-typed and no symbolic
/// backend lowers it faithfully`.
///
/// [`Notation::Symbolic`]: super::Notation::Symbolic
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NonBooleanLogicalOperandError {
    operand: Expression,
    parent: Option<Expression>,
    position: BooleanPosition,
}

impl NonBooleanLogicalOperandError {
    /// Construct the error for `operand` at `position` under `parent`.
    pub(super) fn new(
        operand: Expression,
        parent: Option<Expression>,
        position: BooleanPosition,
    ) -> Self {
        Self {
            operand,
            parent,
            position,
        }
    }

    /// Return the operand that provably denotes a number.
    #[must_use]
    pub fn operand(&self) -> &Expression {
        &self.operand
    }

    /// Return the node that puts the operand in a Boolean position, or
    /// `None` when the operand is the root of a predicate.
    #[must_use]
    pub fn parent(&self) -> Option<&Expression> {
        self.parent.as_ref()
    }

    /// Return where the operand sits relative to its parent.
    #[must_use]
    pub fn position(&self) -> BooleanPosition {
        self.position
    }
}

impl fmt::Display for NonBooleanLogicalOperandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const NUMBER: &str = "which provably denotes a number";
        let options =
            FormatOptions::default().with_identifier_style(IdentifierStyle::NameHintWithId);
        let operand = self.operand.display(options);
        let parent = self
            .parent
            .as_ref()
            .map(|parent| parent.display(options).to_string())
            .unwrap_or_default();
        match self.position {
            BooleanPosition::NegatedOperand => write!(
                f,
                "{parent} applies the Boolean connective logical_not to the operand {operand}, \
                 {NUMBER}"
            ),
            BooleanPosition::LogicalOperand { operation, .. } => write!(
                f,
                "{parent} applies the Boolean connective {operation} to the operand {operand}, \
                 {NUMBER}"
            ),
            BooleanPosition::CaseCondition { case_index } => write!(
                f,
                "{parent} takes {operand} as the condition of case {case_index}, {NUMBER}"
            ),
            BooleanPosition::CaseValue { case_index } => write!(
                f,
                "{parent} takes {operand} as the value of case {case_index}, {NUMBER}"
            ),
            BooleanPosition::Otherwise => {
                write!(
                    f,
                    "{parent} takes {operand} as its otherwise branch, {NUMBER}"
                )
            }
            BooleanPosition::PredicateRoot => write!(
                f,
                "{operand} is used as a predicate but provably denotes a number"
            ),
        }?;
        f.write_str("; the expression is ill-typed and no symbolic backend lowers it faithfully")
    }
}

impl Error for NonBooleanLogicalOperandError {}
