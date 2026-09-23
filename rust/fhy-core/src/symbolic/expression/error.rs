//! Errors raised while building or screening expressions.
//!
//! [`ExpressionBuildError`] reports a node that could not be built because
//! its operands or its child list break a node invariant.
//! [`NonBooleanLogicalOperandError`] reports a Boolean position that holds
//! an operand provably denoting a number, as found by
//! [`validate_logical_operands`](super::validate_logical_operands) and
//! [`validate_predicate`](super::validate_predicate); its
//! [`BooleanPosition`] says where that operand sits.

use std::error::Error;
use std::fmt;

use super::node::Expression;
use super::operation::BinaryOperation;

/// A node that could not be built because it would break a node invariant.
///
/// # Examples
///
/// ```
/// use fhy_core::symbolic::expression::{ExpressionBuildError, build_piecewise, Expression};
///
/// let no_cases: Vec<(Expression, Expression)> = Vec::new();
/// let result = build_piecewise(no_cases, 0);
/// assert_eq!(result, Err(ExpressionBuildError::EmptyPiecewise));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExpressionBuildError {
    /// A piecewise expression was given no cases.
    ///
    /// Displays as `a piecewise expression needs at least one case`.
    EmptyPiecewise,
    /// A piecewise case condition is a literal that is not a Boolean.
    ///
    /// Displays as `piecewise case {case_index} condition literal must be a
    /// boolean`.
    NonBooleanConditionLiteral {
        /// The zero-based index of the offending case.
        case_index: usize,
    },
    /// A call expression was given an empty function name.
    ///
    /// Displays as `a call expression needs a non-empty function name`.
    EmptyFunctionName,
    /// A conjunction or disjunction was given fewer than two operands.
    ///
    /// Displays as `{operation} requires at least two operands, but got
    /// {count}`, with the operation's wire name.
    TooFewLogicalOperands {
        /// [`BinaryOperation::LogicalAnd`] or [`BinaryOperation::LogicalOr`].
        operation: BinaryOperation,
        /// The number of operands given.
        count: usize,
    },
    /// A node was rebuilt from a child list whose length differs from its
    /// own child count.
    ///
    /// Displays as `rebuilding the node needs {expected} children, but got
    /// {actual}`.
    ChildCountMismatch {
        /// The number of children the node has.
        expected: usize,
        /// The number of children given.
        actual: usize,
    },
}

impl fmt::Display for ExpressionBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPiecewise => f.write_str("a piecewise expression needs at least one case"),
            Self::NonBooleanConditionLiteral { case_index } => write!(
                f,
                "piecewise case {case_index} condition literal must be a boolean"
            ),
            Self::EmptyFunctionName => {
                f.write_str("a call expression needs a non-empty function name")
            }
            Self::TooFewLogicalOperands { operation, count } => write!(
                f,
                "{operation} requires at least two operands, but got {count}"
            ),
            Self::ChildCountMismatch { expected, actual } => write!(
                f,
                "rebuilding the node needs {expected} children, but got {actual}"
            ),
        }
    }
}

impl Error for ExpressionBuildError {}

/// Where a Boolean position sits relative to the node that imposes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BooleanPosition {
    /// The operand of a logical negation.
    NegatedOperand,
    /// An operand of a conjunction or a disjunction.
    LogicalOperand {
        /// [`BinaryOperation::LogicalAnd`] or [`BinaryOperation::LogicalOr`].
        operation: BinaryOperation,
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
/// itself. `Display` names the position and renders the parent and the
/// operand with their `Debug` text:
///
/// - negated operand: `the logical_not in {parent:?} takes the operand
///   {operand:?}, which provably denotes a number`
/// - conjunction or disjunction operand: `the {operation} in {parent:?}
///   takes the operand {operand:?}, which provably denotes a number`, with
///   the operation's wire name
/// - case condition: `{parent:?} takes {operand:?} as the condition of case
///   {case_index}, which provably denotes a number`
/// - case value: `{parent:?} takes {operand:?} as the value of case
///   {case_index}, which provably denotes a number`
/// - otherwise branch: `{parent:?} takes {operand:?} as its otherwise
///   branch, which provably denotes a number`
/// - predicate root: `{operand:?} is used as a predicate but provably
///   denotes a number`
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
        let operand = &self.operand;
        let Some(parent) = &self.parent else {
            return write!(
                f,
                "{operand:?} is used as a predicate but provably denotes a number"
            );
        };
        match self.position {
            BooleanPosition::NegatedOperand => write!(
                f,
                "the logical_not in {parent:?} takes the operand {operand:?}, {NUMBER}"
            ),
            BooleanPosition::LogicalOperand { operation } => write!(
                f,
                "the {operation} in {parent:?} takes the operand {operand:?}, {NUMBER}"
            ),
            BooleanPosition::CaseCondition { case_index } => write!(
                f,
                "{parent:?} takes {operand:?} as the condition of case {case_index}, {NUMBER}"
            ),
            BooleanPosition::CaseValue { case_index } => write!(
                f,
                "{parent:?} takes {operand:?} as the value of case {case_index}, {NUMBER}"
            ),
            BooleanPosition::Otherwise => write!(
                f,
                "{parent:?} takes {operand:?} as its otherwise branch, {NUMBER}"
            ),
            BooleanPosition::PredicateRoot => write!(
                f,
                "{operand:?} is used as a predicate but provably denotes a number"
            ),
        }
    }
}

impl Error for NonBooleanLogicalOperandError {}
