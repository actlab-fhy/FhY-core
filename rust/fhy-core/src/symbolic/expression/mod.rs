//! The symbolic expression tree, its builders, and the analyses over it.
//!
//! Integer literals of any size hold a [`BigInt`], re-exported here from
//! `num-bigint`, so building and reading them needs no direct dependency
//! on that crate.

pub mod builtins;
pub mod pattern;

mod alpha;
mod build;
mod error;
mod literal;
mod node;
mod operation;
mod pprint;
mod screen;
mod sort;
mod wire;

pub use alpha::AlphaRenaming;
pub use build::{IntoOperand, build_call, build_logical_and, build_logical_or, build_piecewise};
pub use error::{
    BooleanPosition, ExpressionBuildError, NonBooleanLogicalOperandError, NonInjectiveRenamingError,
};
pub use literal::{LiteralKind, LiteralTextError, LiteralValue};
pub use node::{
    BinaryExpression, CallExpression, Expression, ExpressionKind, PiecewiseExpression,
    UnaryExpression,
};
pub use operation::{BinaryOperation, UnaryOperation};
pub use pprint::{FormatOptions, IdentifierStyle, Notation, format_expression};
pub use screen::{NoRegisteredSorts, SortLookup, validate_logical_operands, validate_predicate};
pub use sort::FunctionSort;

/// The arbitrary-precision signed integer an integer literal holds, as
/// [`LiteralKind::Int`] shows it and as [`LiteralValue::from`] and the
/// expression builders take it.
///
/// This is `num_bigint::BigInt` itself, re-exported so that callers need
/// no direct dependency on `num-bigint`. A caller that depends on a
/// `num-bigint` release semver-compatible with this crate's (0.5) sees the
/// same type under both paths.
///
/// # Examples
///
/// ```
/// use fhy_core::symbolic::expression::{BigInt, LiteralKind, LiteralValue};
///
/// let big: BigInt = "100000000000000000000".parse().expect("digits");
/// let literal = LiteralValue::from(big.clone());
///
/// assert_eq!(literal.kind(), LiteralKind::Int(&big));
/// ```
pub use num_bigint::BigInt;
