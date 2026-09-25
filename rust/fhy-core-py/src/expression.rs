//! `PyO3` classes and functions for [`fhy_core::expression`]: the bases of
//! `fhy_core.symbolic.expression.core`'s expression classes on the Rust
//! backend (pattern P2, as a class hierarchy), and the Boolean-position
//! screen.
//!
//! On the Rust backend the Python expression API takes the Rust core's
//! semantics (decision D-S4-1 of `docs/design/python-switch.md`): `==` and
//! `hash` are structural, literals are normalized, conjunctions and
//! disjunctions are one n-ary `LogicalExpression`, built-in function names
//! are reserved, and `str`, `repr` and `pformat_expression` print the
//! core's text. Payloads keep the `__type__`/`__data__` envelope of
//! `WrappedFamilySerializable`, which the public classes inherit (D-S4-5).

mod alpha;
mod literal;
mod materialize;
mod node;
mod operation;
mod screen;
mod text;

pub(crate) use node::{
    PyBinaryExpression, PyCallExpression, PyExpression, PyIdentifierExpression,
    PyLiteralExpression, PyLogicalExpression, PyPiecewiseExpression, PyUnaryExpression,
};
pub(crate) use screen::{validate_logical_operands, validate_predicate};
