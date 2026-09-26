//! The errors of asking a [`Solver`](super::Solver) and of lowering an
//! expression to SMT-LIB2.

use std::error::Error;
use std::fmt;

use crate::expression::{
    Callee, Expression, ExpressionKind, NonBooleanLogicalOperandError, PiecewiseError,
};
use crate::identifier::Identifier;

use super::QueryKind;
use super::backend::BackendError;

/// Write `identifiers` as `name::id` items separated by commas.
pub(super) fn write_identifiers(
    f: &mut fmt::Formatter<'_>,
    identifiers: &[Identifier],
) -> fmt::Result {
    for (index, identifier) in identifiers.iter().enumerate() {
        if index > 0 {
            f.write_str(", ")?;
        }
        write!(f, "{identifier:?}")?;
    }
    Ok(())
}

/// A question a [`Solver`](super::Solver) refused or failed to answer.
///
/// A question the screens refuse, or that a backend gives up on, is no
/// error: it is answered [`Answer::Unknown`](super::Answer::Unknown).
///
/// # Examples
///
/// ```
/// use fhy_core::solver::{QueryKind, SolveError};
///
/// let error = SolveError::NoCapableBackend(QueryKind::Satisfiability);
///
/// assert_eq!(
///     error.to_string(),
///     "the backends of this solver cannot answer satisfiability queries"
/// );
/// ```
#[derive(Debug)]
#[non_exhaustive]
pub enum SolveError {
    /// The solver holds no backend that answers this kind of question.
    NoCapableBackend(QueryKind),
    /// The symbol types declare no value kind for these identifiers of the
    /// question, ordered by id. A native constant's identifier needs none.
    MissingSymbolTypes(Vec<Identifier>),
    /// An expression of the question holds a number in a Boolean position.
    IllTyped(NonBooleanLogicalOperandError),
    /// The environment of a simplification binds these native constants,
    /// which the expression refers to and whose values are fixed, ordered
    /// by id.
    BoundNativeConstant(Vec<Identifier>),
    /// Substituting the environment of a simplification put a literal
    /// other than a Boolean in a piecewise case condition.
    Substitution(PiecewiseError),
    /// An expression of the question has no SMT-LIB2 lowering.
    Lowering(LoweringError),
    /// A backend failed.
    Backend {
        /// The backend's [`name`](super::SmtSolver::name).
        backend: String,
        /// What it reported.
        source: BackendError,
    },
}

impl fmt::Display for SolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoCapableBackend(kind) => write!(
                f,
                "the backends of this solver cannot answer {} queries",
                kind.description()
            ),
            Self::MissingSymbolTypes(identifiers) => {
                f.write_str("symbol_types is missing entries for identifiers: ")?;
                write_identifiers(f, identifiers)
            }
            Self::IllTyped(_) => f.write_str("the expression is ill-typed"),
            Self::BoundNativeConstant(identifiers) => {
                f.write_str(
                    "the environment binds native constants the expression refers to, \
                     whose values are fixed: ",
                )?;
                write_identifiers(f, identifiers)
            }
            Self::Substitution(_) => {
                f.write_str("substituting the environment built an invalid piecewise")
            }
            Self::Lowering(_) => f.write_str("the expression has no smt-lib2 lowering"),
            Self::Backend { backend, .. } => write!(f, "the backend {backend:?} failed"),
        }
    }
}

impl Error for SolveError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::IllTyped(error) => Some(error),
            Self::Substitution(error) => Some(error),
            Self::Lowering(error) => Some(error),
            Self::Backend { source, .. } => Some(source.as_ref()),
            Self::NoCapableBackend(_)
            | Self::MissingSymbolTypes(_)
            | Self::BoundNativeConstant(_) => None,
        }
    }
}

/// An expression that has no SMT-LIB2 lowering.
///
/// [`SmtScript::lower`](super::SmtScript::lower) checks, in this order,
/// that every identifier but a native constant's has a symbol type, that
/// no Boolean position holds a number, and that no native constant is
/// referred to, and then refuses the first node it has no term for.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::{Expression, NoRegisteredSorts, SymbolType};
/// use fhy_core::identifier::Identifier;
/// use fhy_core::solver::{LoweringError, SmtScript};
///
/// let b = Identifier::new("b");
/// let symbol_types = |_: &Identifier| Some(SymbolType::Bool);
/// let error = SmtScript::lower(&(Expression::from(b) + 1), &symbol_types, &NoRegisteredSorts)
///     .expect_err("a boolean has no sum");
///
/// assert!(matches!(error, LoweringError::SortMismatch(_)));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum LoweringError {
    /// The symbol types declare no value kind for these identifiers, ordered
    /// by id.
    MissingSymbolTypes(Vec<Identifier>),
    /// A Boolean position holds a number.
    IllTyped(NonBooleanLogicalOperandError),
    /// The expression refers to these native constants, which SMT-LIB2 has
    /// no term for, ordered by id.
    NativeConstants(Vec<Identifier>),
    /// A float literal that is infinite or NaN, which has no rational value.
    NonFiniteLiteral(Expression),
    /// A call, which SMT-LIB2 has no term for; a composed or user function
    /// has to be inlined first.
    Call(Expression),
    /// A node where a Boolean meets a number: an arithmetic or ordering
    /// operand, an equality of the two, or piecewise branches of both.
    SortMismatch(Expression),
    /// A power whose exponent is not an integer literal of at least one.
    UnsupportedPower(Expression),
}

impl fmt::Display for LoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSymbolTypes(identifiers) => {
                f.write_str("symbol_types is missing entries for identifiers: ")?;
                write_identifiers(f, identifiers)
            }
            Self::IllTyped(_) => f.write_str("the expression is ill-typed"),
            Self::NativeConstants(identifiers) => {
                f.write_str("smt-lib2 has no term for the native constants ")?;
                write_identifiers(f, identifiers)
            }
            Self::NonFiniteLiteral(node) => {
                write!(f, "smt-lib2 has no term for the non-finite float {node}")
            }
            Self::Call(node) => match node.kind() {
                ExpressionKind::Call(call) => match call.callee() {
                    Callee::Builtin(function) if function.composed().is_some() => write!(
                        f,
                        "smt-lib2 has no term for a call of the built-in {:?}; inline it first \
                         with inline_functions",
                        function.name()
                    ),
                    Callee::Builtin(function) => write!(
                        f,
                        "smt-lib2 has no term for a call of the native built-in {:?}",
                        function.name()
                    ),
                    Callee::Named(name) => write!(
                        f,
                        "smt-lib2 has no term for a call of {:?}; a user function must be \
                         inlined first with inline_functions",
                        name.as_str()
                    ),
                },
                _ => write!(f, "smt-lib2 has no term for the call {node}"),
            },
            Self::SortMismatch(node) => {
                write!(f, "a boolean and a number meet in the node {node}")
            }
            Self::UnsupportedPower(node) => write!(
                f,
                "smt-lib2 has no term for the power {node}, whose exponent is not an integer \
                 literal of at least one"
            ),
        }
    }
}

impl Error for LoweringError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::IllTyped(error) => Some(error),
            Self::MissingSymbolTypes(_)
            | Self::NativeConstants(_)
            | Self::NonFiniteLiteral(_)
            | Self::Call(_)
            | Self::SortMismatch(_)
            | Self::UnsupportedPower(_) => None,
        }
    }
}
