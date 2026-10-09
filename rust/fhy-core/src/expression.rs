//! Symbolic expressions and the vocabulary they are built from: the
//! expression tree, its builders, the analyses over it, and the compiler
//! passes over it.
//!
//! Integer literals of any size hold a [`BigInt`], re-exported here from
//! `num-bigint`, so building and reading them needs no direct dependency
//! on that crate.

pub mod builtins;
pub mod evaluate;
pub mod passes;
pub mod pattern;
pub mod registry;

mod affine;
mod build;
mod callee;
mod canonical;
mod display;
mod error;
mod literal;
mod node;
pub(crate) mod operation;
mod screen;
mod sort;
mod symbol_type;
mod wire;

pub use affine::AffineForm;
pub use callee::{Callee, FunctionName, FunctionNameError};
pub(crate) use canonical::{CanonicalTable, Equivalence};
pub(crate) use display::Bounded;
pub use display::{ExpressionDisplay, FormatOptions, IdentifierStyle, Notation};
pub use error::{BooleanPosition, NonBooleanLogicalOperandError, PiecewiseError, RebuildError};
pub use literal::exact::Rational;
pub(crate) use literal::exact::{
    ExactNumber, borrow_parts, build_integer, build_zero, compute_arithmetic, floor,
    keep_within_limits, raise_to_power, reduce, take_root,
};
pub use literal::{Decimal, DecimalPartsError, LiteralTextError, LiteralValue};
pub(crate) use literal::{float_text, integer_text, serialize_display_text, write_float};
pub use node::{
    BinaryExpression, CallExpression, Expression, ExpressionKind, LogicalExpression,
    PiecewiseExpression, UnaryExpression,
};
pub use operation::{BinaryOperation, LogicalOperation, UnaryOperation};
pub use screen::{BooleanScreen, Environment, NoRegisteredSorts, SortLookup, SymbolTypes};
pub use sort::FunctionSort;
pub use symbol_type::SymbolType;

/// The arbitrary-precision signed integer of an integer literal.
///
/// [`LiteralValue::Int`] holds it, and [`LiteralValue::from`] and the
/// expression builders take it. This is `num_bigint::BigInt` itself, so a
/// caller depending on a `num-bigint` release semver-compatible with this
/// crate's (0.5) sees the same type under both paths.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::{BigInt, LiteralValue};
///
/// let big: BigInt = "100000000000000000000".parse().expect("digits");
/// let literal = LiteralValue::from(big.clone());
///
/// assert!(matches!(literal, LiteralValue::Int(value) if value == big));
/// ```
pub use num_bigint::BigInt;
