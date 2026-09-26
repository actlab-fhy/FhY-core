//! The Python exceptions of the type system's core errors (D-S11-14).

use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;

use fhy_core::types::{LiteralTypeError, PromotionError, TemplateWidthError, UnificationError};

use crate::error::IntoPyErr;

/// Return `fhy_core.types.core.FhYCoreTypeError`.
pub(crate) fn core_type_error_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    CLASS.import(py, "fhy_core.types.core", "FhYCoreTypeError")
}

/// Return `fhy_core.traits.verifiable.VerificationError`.
fn verification_error_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    CLASS.import(py, "fhy_core.traits.verifiable", "VerificationError")
}

/// Return the exception of `class` with `message`, or the error importing
/// the class.
fn build_error(class: PyResult<&Bound<'_, PyType>>, message: String) -> PyErr {
    match class.and_then(|class| class.call1((message,))) {
        Ok(error) => PyErr::from_value(error),
        Err(error) => error,
    }
}

/// Raises `FhYCoreTypeError` with the core's text.
impl IntoPyErr for PromotionError {
    fn into_py_err(self) -> PyErr {
        Python::attach(|py| build_error(core_type_error_class(py), self.to_string()))
    }
}

/// Raises `NotImplementedError` for a literal kind with no core data type,
/// and `FhYCoreTypeError` otherwise, with the core's text.
impl IntoPyErr for LiteralTypeError {
    fn into_py_err(self) -> PyErr {
        if self.is_unsupported() {
            return PyNotImplementedError::new_err(self.to_string());
        }
        Python::attach(|py| build_error(core_type_error_class(py), self.to_string()))
    }
}

/// Raises `ValueError` with the core's text.
impl IntoPyErr for TemplateWidthError {
    fn into_py_err(self) -> PyErr {
        PyValueError::new_err(self.to_string())
    }
}

/// Raises the exception a Python-defined type's handler raised, and
/// `VerificationError` with the core's text otherwise.
impl IntoPyErr for UnificationError {
    fn into_py_err(self) -> PyErr {
        match self {
            UnificationError::Extension(source) => match source.downcast::<PyErr>() {
                Ok(error) => *error,
                Err(other) => Python::attach(|py| {
                    build_error(verification_error_class(py), other.to_string())
                }),
            },
            other => {
                Python::attach(|py| build_error(verification_error_class(py), other.to_string()))
            }
        }
    }
}
