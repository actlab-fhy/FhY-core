//! The Python exceptions of evaluation and folding errors (D-S9-14): the
//! core's text under the classes the replaced Python API documents.

use pyo3::exceptions::{
    PyMemoryError, PyOverflowError, PyRuntimeError, PyTypeError, PyValueError, PyZeroDivisionError,
};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;

use fhy_core::expression::evaluate::{EvaluationError, FoldError, LaneFailure};
use fhy_core::foreign::BoxError;

use crate::error::IntoPyErr;

use super::super::registry::{arity_error, inline_error_to_python, lookup_error};

/// Return the exception `fhy_core.symbolic.expression.errors.<name>`,
/// whose class `cell` caches, carrying `message`.
fn expression_error(
    py: Python<'_>,
    cell: &'static PyOnceLock<Py<PyType>>,
    name: &str,
    message: &str,
) -> PyErr {
    match cell.import(py, "fhy_core.symbolic.expression.errors", name) {
        Ok(class) => match class.call1((message,)) {
            Ok(error) => PyErr::from_value(error),
            Err(error) => error,
        },
        Err(error) => error,
    }
}

/// Define a function returning the exception of one class of
/// `fhy_core.symbolic.expression.errors`.
macro_rules! expression_error_fn {
    ($function:ident, $name:literal) => {
        #[doc = concat!("Return the `", $name, "` carrying `message`.")]
        pub(super) fn $function(py: Python<'_>, message: &str) -> PyErr {
            static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
            expression_error(py, &CLASS, $name, message)
        }
    };
}

expression_error_fn!(native_constant_binding_error, "NativeConstantBindingError");
expression_error_fn!(native_result_sort_error, "NativeResultSortError");
expression_error_fn!(non_finite_cast_error, "NonFiniteCastError");
expression_error_fn!(
    string_literal_precision_error,
    "StringLiteralPrecisionError"
);
expression_error_fn!(unbound_variable_error, "UnboundVariableError");
expression_error_fn!(unsupported_lowering_error, "UnsupportedNumpyLoweringError");

/// Return the Python exception a callback error carries: the exception the
/// callback raised, unchanged.
pub(super) fn callback_error_to_python(error: BoxError) -> PyErr {
    match error.downcast::<PyErr>() {
        Ok(error) => *error,
        Err(error) => PyRuntimeError::new_err(error.to_string()),
    }
}

/// Return the Python exception of the evaluation error `error`.
pub(super) fn evaluation_error_to_python(py: Python<'_>, error: EvaluationError) -> PyErr {
    let message = error.to_string();
    match error {
        EvaluationError::Inline(error) => inline_error_to_python(py, &error),
        EvaluationError::IllTyped(error) => error.into_py_err(),
        EvaluationError::BoundNativeConstant(_) => native_constant_binding_error(py, &message),
        EvaluationError::Unbound { .. } => unbound_variable_error(py, &message),
        EvaluationError::InexactDecimal(_) => string_literal_precision_error(py, &message),
        EvaluationError::IntegerOutOfRange(_) => PyOverflowError::new_err(message),
        EvaluationError::Unsupported(_) => unsupported_lowering_error(py, &message),
        EvaluationError::BooleanArithmetic(_) | EvaluationError::MixedBranches(_) => {
            PyTypeError::new_err(message)
        }
        EvaluationError::Shape { .. } | EvaluationError::BroadcastTooLarge { .. } => {
            PyValueError::new_err(message)
        }
        EvaluationError::OutOfMemory { .. } => PyMemoryError::new_err(message),
        EvaluationError::Lane { failure, .. } => match failure {
            LaneFailure::IntegerOverflow | LaneFailure::OutOfRangeCast => {
                PyOverflowError::new_err(message)
            }
            LaneFailure::DivisionByZero => PyZeroDivisionError::new_err(message),
            LaneFailure::NonFiniteCast => non_finite_cast_error(py, &message),
            _ => PyValueError::new_err(message),
        },
        EvaluationError::Kernel { source, .. } => callback_error_to_python(source),
        _ => PyRuntimeError::new_err(message),
    }
}

/// Return the Python exception of the folding error `error`.
pub(super) fn fold_error_to_python(py: Python<'_>, error: FoldError) -> PyErr {
    let message = error.to_string();
    match error {
        FoldError::UnknownFunction(_) => lookup_error(py, &message),
        FoldError::NotCallable(_) | FoldError::Arity { .. } => arity_error(py, &message),
        FoldError::ArgumentSort { .. } => PyTypeError::new_err(message),
        FoldError::ResultSort { .. } => native_result_sort_error(py, &message),
        FoldError::InexactDecimal(_) => string_literal_precision_error(py, &message),
        FoldError::NonFiniteCast { .. } => non_finite_cast_error(py, &message),
        FoldError::Piecewise(source) => PyValueError::new_err(format!("{message}: {source}")),
        FoldError::Native { source, .. } => callback_error_to_python(source),
        _ => PyRuntimeError::new_err(message),
    }
}
