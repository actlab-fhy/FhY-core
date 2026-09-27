//! The Python exceptions of the param core's errors (D-S16-12).
//!
//! An error with a core counterpart raises the class the Python
//! implementation raised, with the core's text; one whose Python text names
//! Python values, such as a class, keeps Python's words.

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;

use fhy_core::param::{DomainKind, ParamError, SetOperation};

use crate::constraint::{constraint_error, constraint_error_to_py, type_name};

use super::value::value_kind_message;

/// Return `fhy_core.symbolic.param.values.ParamError`.
fn param_error_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    CLASS.import(py, "fhy_core.symbolic.param.values", "ParamError")
}

/// Return the `ParamError` with `message`.
pub(super) fn param_error(py: Python<'_>, message: impl Into<String>) -> PyErr {
    match param_error_class(py).and_then(|class| class.call1((message.into(),))) {
        Ok(error) => PyErr::from_value(error),
        Err(error) => error,
    }
}

/// Return the name of the Python class of a domain of `kind`, after its
/// indefinite article.
fn class_with_article(kind: DomainKind) -> &'static str {
    match kind {
        DomainKind::Integer => "an IntegerDomain",
        DomainKind::IntervalInteger => "an IntervalIntegerDomain",
        DomainKind::Real => "a RealDomain",
        DomainKind::Ordinal => "an OrdinalDomain",
        DomainKind::Categorical => "a CategoricalDomain",
        DomainKind::Permutation => "a PermutationDomain",
        _ => "a ParamDomain",
    }
}

/// Return the exception of `error`, where `other` is the other domain of a
/// set operation, if any.
pub(super) fn param_error_to_py(
    py: Python<'_>,
    error: ParamError,
    other: Option<&Bound<'_, PyAny>>,
) -> PyErr {
    let text = error.to_string();
    match error {
        ParamError::NotALeafValue { kind, .. } => PyTypeError::new_err(value_kind_message(kind)),
        ParamError::IncomparableValues
        | ParamError::NotAnIntervalOperand
        | ParamError::ForbiddenConstraintKind(DomainKind::IntervalInteger) => {
            PyTypeError::new_err(text)
        }
        ParamError::KindMismatch { operation, own, .. } => {
            let verb = match operation {
                SetOperation::Union => "union",
                _ => "intersect",
            };
            let other = other.map_or_else(|| "?".to_owned(), type_name);
            PyTypeError::new_err(format!(
                "Cannot {verb} {} with a domain of type {other}.",
                class_with_article(own)
            ))
        }
        ParamError::Rescope { .. } | ParamError::UnexpectedConstraintKind => {
            constraint_error(py, text)
        }
        ParamError::Constraint(error) => constraint_error_to_py(py, error, None),
        ParamError::Custom(error) => match error.downcast::<PyErr>() {
            Ok(error) => *error,
            Err(error) => PyRuntimeError::new_err(format!("{text}: {error}")),
        },
        ParamError::EmptyValues(_)
        | ParamError::NanValue(_)
        | ParamError::DuplicateValues(_)
        | ParamError::ForbiddenConstraintKind(_)
        | ParamError::NotABound
        | ParamError::EmptyUnion(_)
        | ParamError::EmptyIntersection(_)
        | ParamError::DifferentPermutationMembers
        | ParamError::EmptyInterval(_)
        | ParamError::NaturalBound { .. }
        | ParamError::UnorderedBounds
        | ParamError::EmptyParamIntersection => param_error(py, text),
        _ => PyRuntimeError::new_err(text),
    }
}

/// Return the exception of a failed ordinal construction: a raising `<`'s
/// `TypeError` chained under the core's `TypeError`, another exception it
/// raised as itself, and otherwise the error's own exception.
pub(super) fn ordinal_error_to_py(
    py: Python<'_>,
    error: ParamError,
    raised: Option<PyErr>,
) -> PyErr {
    match (error, raised) {
        (error @ ParamError::IncomparableValues, Some(raised))
            if raised.is_instance_of::<PyTypeError>(py) =>
        {
            let refused = PyTypeError::new_err(error.to_string());
            refused.set_cause(py, Some(raised));
            refused
        }
        (_, Some(raised)) => raised,
        (error, None) => param_error_to_py(py, error, None),
    }
}
