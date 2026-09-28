//! The Python exceptions of the constraint core's errors.
//!
//! An unusable binding raises `ConstraintError` naming the identifier, the
//! value's `repr` and its type, in the Python implementation's words, since
//! it describes a Python value; a set constraint's refusal is chained to the
//! member validation's error, and a literal the equation cannot lift to the
//! `LiteralExpression` constructor's. An ill-typed predicate raises
//! `NonBooleanLogicalOperandError`, a member that does not lift
//! `ConstraintError` with the core's text, and a solver error the solver's
//! own exception.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;

use fhy_core::constraint::{ConstraintError, UnusableBindingReason};

use crate::error::IntoPyErr;
use crate::expression::{decimal_class, non_boolean_operand_error};
use crate::solver::solve_error_to_py;

use super::value::{constraint_error, read_member_value, repr_text, type_name};

/// Return `fhy_core.symbolic.expression.LiteralExpression`.
fn literal_expression_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    crate::python::cached_attr!(py, "fhy_core.symbolic.expression", "LiteralExpression" => PyType)
}

/// Return the `MissingSymbolTypeError` with `message`.
fn missing_symbol_type_error(py: Python<'_>, message: String) -> PyErr {
    crate::exceptions::MISSING_SYMBOL_TYPE_ERROR.err(py, (message,))
}

/// Return whether `value` is a `LiteralType`: a `str`, `float`, `int`,
/// `bool` or `Decimal`.
fn is_literal_type(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    use pyo3::types::{PyFloat, PyInt, PyString};
    Ok(value.is_instance_of::<PyString>()
        || value.is_instance_of::<PyFloat>()
        || value.is_instance_of::<PyInt>()
        || value.is_instance(decimal_class(value.py())?)?)
}

/// Return the exception of `error`, where `identifier` and `value` are the
/// Python objects of the binding it concerns, if any.
pub(crate) fn constraint_error_to_py(
    py: Python<'_>,
    error: ConstraintError,
    binding: Option<(&Bound<'_, PyAny>, &Bound<'_, PyAny>)>,
) -> PyErr {
    let text = error.to_string();
    match error {
        ConstraintError::UnusableBinding { reason, .. } => match binding {
            Some((identifier, value)) => unusable_binding_error(identifier, value, reason),
            None => constraint_error(py, text),
        },
        ConstraintError::IllTyped(error) => error.into_py_err(),
        ConstraintError::NonBooleanResult { .. } => non_boolean_operand_error(py, text),
        ConstraintError::Solve(error) => solve_error_to_py(py, error),
        ConstraintError::MissingSymbolTypes(_) => missing_symbol_type_error(py, text),
        ConstraintError::Custom(error) => crate::exceptions::unbox_py_err(error)
            .unwrap_or_else(|error| PyRuntimeError::new_err(format!("{text}: {error}"))),
        ConstraintError::Substitution(error) => PyValueError::new_err(format!("{text}: {error}")),
        _ => constraint_error(py, text),
    }
}

/// Return the `ConstraintError` of the unusable binding of `identifier` to
/// `value`.
fn unusable_binding_error(
    identifier: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
    reason: UnusableBindingReason,
) -> PyErr {
    let py = identifier.py();
    let identifier_repr = repr_text(identifier);
    let value_repr = repr_text(value);
    let value_type = type_name(value);
    match reason {
        UnusableBindingReason::NotMemberShaped => {
            let cause = read_member_value(value).err();
            let detail = cause.as_ref().map_or_else(String::new, |cause| {
                cause
                    .value(py)
                    .str()
                    .map_or_else(|_| String::new(), |text| format!(": {text}"))
            });
            let error = constraint_error(
                py,
                format!(
                    "Binding for identifier {identifier_repr} must be an `Expression` or a value \
                     that could be a constraint member, but got value {value_repr} of type \
                     {value_type}{detail}"
                ),
            );
            error.set_cause(py, cause);
            error
        }
        UnusableBindingReason::Unhashable(source) => {
            let error = constraint_error(
                py,
                format!(
                    "Binding for identifier {identifier_repr} is unhashable: value {value_repr} of \
                     type {value_type} cannot be checked for membership."
                ),
            );
            if let Ok(cause) = crate::exceptions::unbox_py_err(source) {
                error.set_cause(py, Some(cause));
            }
            error
        }
        UnusableBindingReason::NotALiteral if !is_literal_type(value).unwrap_or(false) => {
            constraint_error(
                py,
                format!(
                    "Binding for identifier {identifier_repr} must be an `Expression` or a literal \
                     (`str`, `float`, `int`, `bool`, `Decimal`), but got value {value_repr} of \
                     type {value_type}."
                ),
            )
        }
        _ => {
            // A literal the equation cannot lift: the constructor's refusal
            // is the cause, as the Python implementation chained it.
            let cause = literal_expression_class(py)
                .and_then(|class| class.call1((value,)))
                .err();
            let detail = cause.as_ref().map_or_else(String::new, |cause| {
                cause
                    .value(py)
                    .str()
                    .map_or_else(|_| String::new(), |text| text.to_string())
            });
            let error = constraint_error(
                py,
                format!(
                    "Binding for identifier {identifier_repr} cannot be lifted into a literal: \
                     value {value_repr} of type {value_type} is not one a `LiteralExpression` \
                     holds ({detail})"
                ),
            );
            error.set_cause(py, cause);
            error
        }
    }
}
