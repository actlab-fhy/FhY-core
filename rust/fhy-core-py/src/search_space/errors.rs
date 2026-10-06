//! The exceptions of `fhy_core.search_space` and the conversions of the
//! core's errors into them.
//!
//! | Core error | Python exception |
//! |---|---|
//! | `SpaceError::DuplicateName` | `DuplicateNameError` |
//! | `SpaceError::Hook` | the hook's exception itself |
//! | `SpaceError::Constraint` | the constraint error, as the constraint module raises it |
//! | any other `SpaceError` | `SearchSpaceError` |
//! | `ConfigurationErrors` | the exception a Python-defined constraint raised, if one did; otherwise `ConfigurationError`, with each problem's text in `problems` |
//! | `EquivalenceError::Extension` | the hook's exception itself |
//! | `EquivalenceError::Constraint` | the constraint error |
//!
//! Each message is the core error's `Display` text.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use fhy_core::constraint::ConstraintError;
use fhy_core::search_space::{
    ConfigurationError, ConfigurationErrors, EquivalenceError, SpaceError,
};

use crate::constraint::constraint_error_to_py;
use crate::util::exceptions::{ExceptionClass, unbox_py_err};

/// The module of the exceptions.
const MODULE: &str = "fhy_core.search_space.errors";

/// `SearchSpaceError`.
static SEARCH_SPACE_ERROR: ExceptionClass = ExceptionClass::new(MODULE, "SearchSpaceError");

/// `DuplicateNameError`.
static DUPLICATE_NAME_ERROR: ExceptionClass = ExceptionClass::new(MODULE, "DuplicateNameError");

/// `ConfigurationError`.
static CONFIGURATION_ERROR: ExceptionClass = ExceptionClass::new(MODULE, "ConfigurationError");

/// Return the Python exception of the core's refusal to build a space, a
/// choice or an alternative.
pub(super) fn space_error_to_py(py: Python<'_>, error: SpaceError) -> PyErr {
    let text = error.to_string();
    match error {
        SpaceError::DuplicateName { .. } => DUPLICATE_NAME_ERROR.err(py, (text,)),
        SpaceError::Hook { source, .. } => unbox_py_err(source)
            .unwrap_or_else(|source| SEARCH_SPACE_ERROR.err(py, (format!("{text}: {source}"),))),
        SpaceError::Constraint(error) => constraint_error_to_py(py, error, None),
        _ => SEARCH_SPACE_ERROR.err(py, (text,)),
    }
}

/// Return a copy of the Python exception a Python-defined constraint raised
/// in `error`, if it holds one.
fn raised_by_constraint(py: Python<'_>, error: &ConstraintError) -> Option<PyErr> {
    match error {
        ConstraintError::Custom(source) => source
            .downcast_ref::<PyErr>()
            .map(|raised| raised.clone_ref(py)),
        _ => None,
    }
}

/// Return the exception of `errors`: the first a Python-defined constraint
/// raised, as itself, or the `ConfigurationError` carrying each problem's
/// text.
pub(super) fn configuration_errors_to_py(py: Python<'_>, errors: &ConfigurationErrors) -> PyErr {
    let raised = errors.errors().iter().find_map(|problem| match problem {
        ConfigurationError::FailedCondition { error, .. }
        | ConfigurationError::FailedForbidden { error, .. } => raised_by_constraint(py, error),
        _ => None,
    });
    if let Some(raised) = raised {
        return raised;
    }
    let problems = errors
        .errors()
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    match PyTuple::new(py, problems) {
        Ok(problems) => CONFIGURATION_ERROR.err(py, (errors.to_string(), problems)),
        Err(error) => error,
    }
}

/// Return the Python exception of a comparison that failed.
pub(super) fn equivalence_error_to_py(py: Python<'_>, error: EquivalenceError) -> PyErr {
    let text = error.to_string();
    match error {
        EquivalenceError::Extension(source) => unbox_py_err(source)
            .unwrap_or_else(|source| PyRuntimeError::new_err(format!("{text}: {source}"))),
        EquivalenceError::Constraint(error) => constraint_error_to_py(py, error, None),
        _ => PyRuntimeError::new_err(text),
    }
}
