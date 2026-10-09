//! `PyO3` classes and functions for [`fhy_core::expression`]: the bases of
//! `fhy_core.symbolic.expression.core`'s expression classes, built as a
//! class hierarchy, and the Boolean-position screen.
//!
//! The Python expression API takes the Rust core's semantics: `==` and
//! `hash` are structural, literals are normalized, conjunctions and
//! disjunctions are one n-ary `LogicalExpression`, built-in function names
//! are reserved, and `str`, `repr` and `pformat_expression` print the
//! core's text. Payloads keep the `__type__`/`__data__` envelope of
//! `WrappedFamilySerializable`, which the public classes inherit.
//!
//! The `pattern` submodule binds the core's patterns and rewrite rules,
//! the `registry` submodule the function registry and inlining, and
//! the `evaluate` submodule the evaluators. The solver binding reads
//! expressions, the registry snapshot and the materializer through the
//! crate-visible items below.

mod affine;
mod evaluate;
mod literal;
mod materialize;
mod node;
mod operation;
mod pattern;
mod payload;
mod registry;
mod screen;
mod table;
mod text;

pub(crate) use affine::{PyAffineForm, affine_form};
pub(crate) use evaluate::{
    PyBuiltinNativeImplementation, coerce_literal_value, evaluate_expression_with_numpy,
    evaluation_error_to_python, fold_expression, is_decimal_text_exactly_binary,
};
pub(crate) use literal::{big_int_to_python, decimal_class, read_big_int, read_decimal};
pub(crate) use materialize::{
    materialize_expression, materialize_substituted, materialize_with_known,
};
pub(crate) use node::{
    PyBinaryExpression, PyCallExpression, PyExpression, PyIdentifierExpression,
    PyLiteralExpression, PyLogicalExpression, PyPiecewiseExpression, PyUnaryExpression,
    coerce_to_expression,
};
pub(crate) use pattern::{
    PyAlternativesPattern, PyBinaryExpressionPattern, PyCallExpressionPattern, PyCapture,
    PyCapturePattern, PyFiredRule, PyIdentifierPattern, PyLiteralPattern,
    PyLogicalExpressionPattern, PyMatchBindings, PyPattern, PyPiecewiseExpressionPattern,
    PyPredicatePattern, PyRewriteRule, PyRuleBase, PyUnaryExpressionPattern, PyWildcardPattern,
    apply_rewrite_rules,
};
pub(crate) use registry::RegistryState;
pub(crate) use registry::snapshot as registry_snapshot;
pub(crate) use registry::{
    PyNativeConstant, PyNativeFunction, PyRegisteredFunction, get_native_constant_identifier,
    get_registered_entries, get_registered_entry, inline_functions, is_entry_registered,
    read_call_target, read_sort, register_function, register_native_constant,
    register_native_function, set_registry_state, try_get_native_constant_for_identifier,
    try_get_registered_result_sort,
};
#[cfg(test)]
pub(crate) use registry::{install_core_registry, reinstall};
pub(crate) use screen::{non_boolean_operand_error, validate_logical_operands, validate_predicate};

/// Return the `repr` of the Python object of `expression`, such as
/// `BinaryExpression((add x::7 1))`.
pub(crate) fn render_expression_repr(expression: &fhy_core::expression::Expression) -> String {
    text::render_kind_repr(expression)
}
