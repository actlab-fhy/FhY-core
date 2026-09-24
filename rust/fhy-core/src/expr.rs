//! Symbolic expressions and the vocabulary they are built from: the
//! expression tree, its builders, the analyses over it, and the compiler
//! passes over it.
//!
//! Integer literals of any size hold a [`BigInt`], re-exported here from
//! `num-bigint`, so building and reading them needs no direct dependency
//! on that crate.

pub mod builtins;
pub mod pattern;

mod alpha;
mod build;
mod display;
mod error;
mod literal;
mod node;
mod operation;
mod registration;
mod screen;
mod sort;
mod symbol_type;
mod wire;

pub use alpha::AlphaRenaming;
pub use build::IntoOperand;
pub use display::{
    ExpressionDisplay, ExpressionPrettyFormatter, FormatOptions, IdentifierStyle, Notation,
};
pub use error::{
    BooleanPosition, FunctionNameError, NonBooleanLogicalOperandError, NonInjectiveRenamingError,
    PiecewiseError, RebuildError,
};
pub use literal::{Decimal, LiteralTextError, LiteralValue};
pub use node::{
    BinaryExpression, CallExpression, Expression, ExpressionKind, LogicalExpression,
    PiecewiseExpression, UnaryExpression,
};
pub use operation::{BinaryOperation, LogicalOperation, UnaryOperation, UnknownNameError};
pub use registration::register_expression_passes;
pub use screen::{NoRegisteredSorts, SortLookup, validate_logical_operands, validate_predicate};
pub use sort::FunctionSort;
pub use symbol_type::SymbolType;

/// The arbitrary-precision signed integer an integer literal holds, as
/// [`LiteralValue::Int`] holds it and as [`LiteralValue::from`] and the
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
/// use fhy_core::expr::{BigInt, LiteralValue};
///
/// let big: BigInt = "100000000000000000000".parse().expect("digits");
/// let literal = LiteralValue::from(big.clone());
///
/// assert!(matches!(literal, LiteralValue::Int(value) if value == big));
/// ```
pub use num_bigint::BigInt;
