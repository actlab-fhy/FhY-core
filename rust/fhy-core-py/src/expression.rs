//! `PyO3` classes and functions for [`fhy_core::expression`]: the bases of
//! `fhy_core.symbolic.expression.core`'s expression classes (pattern P2, as a
//! class hierarchy), and the Boolean-position screen.
//!
//! The Python expression API takes the Rust core's
//! semantics (decision D-S4-1 of `docs/design/python-switch.md`): `==` and
//! `hash` are structural, literals are normalized, conjunctions and
//! disjunctions are one n-ary `LogicalExpression`, built-in function names
//! are reserved, and `str`, `repr` and `pformat_expression` print the
//! core's text. Payloads keep the `__type__`/`__data__` envelope of
//! `WrappedFamilySerializable`, which the public classes inherit (D-S4-5).
//!
//! The `pattern` submodule binds the core's patterns and rewrite rules (S5),
//! and the `registry` submodule the function registry and inlining (S7).
//! The solver binding (S8) reads expressions, the registry snapshot and the
//! materializer through the crate-visible items below.

mod literal;
mod materialize;
mod node;
mod operation;
mod pattern;
mod payload;
mod registry;
mod screen;
mod text;

pub(crate) use materialize::{materialize_expression, materialize_substituted};
pub(crate) use node::{
    PyBinaryExpression, PyCallExpression, PyExpression, PyIdentifierExpression,
    PyLiteralExpression, PyLogicalExpression, PyPiecewiseExpression, PyUnaryExpression,
};
pub(crate) use pattern::{
    PyAlternativesPattern, PyBinaryExpressionPattern, PyCallExpressionPattern, PyCapture,
    PyCapturePattern, PyFiredRule, PyIdentifierPattern, PyLiteralPattern,
    PyLogicalExpressionPattern, PyMatchBindings, PyPattern, PyPiecewiseExpressionPattern,
    PyPredicatePattern, PyRewriteRule, PyRuleBase, PyUnaryExpressionPattern, PyWildcardPattern,
    apply_rewrite_rules,
};
pub(crate) use registry::snapshot as registry_snapshot;
pub(crate) use registry::{
    PyNativeConstant, PyNativeFunction, PyRegisteredFunction, get_native_constant_identifier,
    get_registered_entries, get_registered_entry, inline_functions, is_entry_registered,
    register_function, register_native_constant, register_native_function,
    set_registry_state_for_tests, try_get_native_constant_for_identifier,
    try_get_registered_result_sort,
};
pub(crate) use screen::{validate_logical_operands, validate_predicate};

/// Return the `repr` of the Python object of `expression`, such as
/// `BinaryExpression((add x::7 1))`.
pub(crate) fn render_expression_repr(expression: &fhy_core::expression::Expression) -> String {
    text::render_kind_repr(expression)
}
