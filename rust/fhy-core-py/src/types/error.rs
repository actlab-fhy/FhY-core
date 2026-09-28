//! The Python exceptions of the type system's core errors (D-S11-14).

use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

use fhy_core::types::{LiteralTypeError, PromotionError, TemplateWidthError, UnificationError};

use crate::error::IntoPyErr;

/// Raises `FhYCoreTypeError` with the core's text.
impl IntoPyErr for PromotionError {
    fn into_py_err(self) -> PyErr {
        Python::attach(|py| crate::exceptions::CORE_TYPE_ERROR.err(py, (self.to_string(),)))
    }
}

/// Raises `NotImplementedError` for a literal kind with no core data type,
/// and `FhYCoreTypeError` otherwise, with the core's text.
impl IntoPyErr for LiteralTypeError {
    fn into_py_err(self) -> PyErr {
        if self.is_unsupported() {
            return PyNotImplementedError::new_err(self.to_string());
        }
        Python::attach(|py| crate::exceptions::CORE_TYPE_ERROR.err(py, (self.to_string(),)))
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
            UnificationError::Extension(source) => crate::exceptions::unbox_py_err(source)
                .unwrap_or_else(|other| {
                    Python::attach(|py| {
                        crate::exceptions::VERIFICATION_ERROR.err(py, (other.to_string(),))
                    })
                }),
            other => Python::attach(|py| {
                crate::exceptions::VERIFICATION_ERROR.err(py, (other.to_string(),))
            }),
        }
    }
}
