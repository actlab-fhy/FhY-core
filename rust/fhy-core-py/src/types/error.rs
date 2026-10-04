//! The Python exceptions of the type system's core errors.

use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

use fhy_core::types::{LiteralTypeError, PromotionError, TemplateWidthError, UnificationError};

use crate::error::IntoPyErr;

/// Raises `FhYCoreTypeError` with the core's text.
impl IntoPyErr for PromotionError {
    fn into_py_err(self) -> PyErr {
        Python::attach(|py| crate::util::exceptions::CORE_TYPE_ERROR.err(py, (self.to_string(),)))
    }
}

/// Raises `NotImplementedError` for a literal kind with no core data type,
/// and `FhYCoreTypeError` otherwise, with the core's text.
impl IntoPyErr for LiteralTypeError {
    fn into_py_err(self) -> PyErr {
        if self.is_unsupported() {
            return PyNotImplementedError::new_err(self.to_string());
        }
        Python::attach(|py| crate::util::exceptions::CORE_TYPE_ERROR.err(py, (self.to_string(),)))
    }
}

/// Raises `ValueError` with the core's text.
impl IntoPyErr for TemplateWidthError {
    fn into_py_err(self) -> PyErr {
        PyValueError::new_err(self.to_string())
    }
}

/// Raises the exception a Python-defined type's handler raised, and
/// `VerificationError` with the core's text otherwise; for a refused
/// substitution, the text ends with the expression's refusal, which is its
/// `__cause__`, as the expression raises it.
impl IntoPyErr for UnificationError {
    fn into_py_err(self) -> PyErr {
        match self {
            Self::Extension(source) => crate::util::exceptions::unbox_py_err(source)
                .unwrap_or_else(|other| {
                    Python::attach(|py| {
                        crate::util::exceptions::VERIFICATION_ERROR.err(py, (other.to_string(),))
                    })
                }),
            Self::Substitution(source) => Python::attach(|py| {
                let text = format!("{self}: {source}");
                let error = crate::util::exceptions::VERIFICATION_ERROR.err(py, (text,));
                error.set_cause(py, Some(source.into_py_err()));
                error
            }),
            other => Python::attach(|py| {
                crate::util::exceptions::VERIFICATION_ERROR.err(py, (other.to_string(),))
            }),
        }
    }
}
