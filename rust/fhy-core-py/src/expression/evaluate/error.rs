//! The Python exceptions of evaluation and folding errors (D-S9-14): the
//! core's text under the classes the replaced Python API documents.

use pyo3::exceptions::{
    PyMemoryError, PyOverflowError, PyRuntimeError, PyTypeError, PyValueError, PyZeroDivisionError,
};
use pyo3::prelude::*;

use fhy_core::expression::evaluate::{EvaluationError, FoldError, LaneFailure};

use crate::error::IntoPyErr;

use super::super::registry::{arity_error, inline_error_to_python, lookup_error};

/// Define a function returning the exception of one class of
/// `fhy_core.symbolic.expression.errors`, carrying `message`.
macro_rules! expression_error_fn {
    ($function:ident, $class:ident) => {
        #[doc = concat!("Return the [`crate::exceptions::", stringify!($class), "`] exception carrying `message`.")]
        pub(super) fn $function(py: Python<'_>, message: &str) -> PyErr {
            crate::exceptions::$class.err(py, (message,))
        }
    };
}

expression_error_fn!(native_constant_binding_error, NATIVE_CONSTANT_BINDING_ERROR);
expression_error_fn!(native_result_sort_error, NATIVE_RESULT_SORT_ERROR);
expression_error_fn!(non_finite_cast_error, NON_FINITE_CAST_ERROR);
expression_error_fn!(
    string_literal_precision_error,
    STRING_LITERAL_PRECISION_ERROR
);
expression_error_fn!(unbound_variable_error, UNBOUND_VARIABLE_ERROR);
expression_error_fn!(unsupported_lowering_error, UNSUPPORTED_NUMPY_LOWERING_ERROR);

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
        EvaluationError::Kernel { source, .. } => crate::exceptions::boxed_error_to_py(source),
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
        FoldError::Native { source, .. } => crate::exceptions::boxed_error_to_py(source),
        _ => PyRuntimeError::new_err(message),
    }
}
