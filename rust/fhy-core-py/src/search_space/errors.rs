//! The exceptions of `fhy_core.search_space` and the conversions of the
//! core's errors into them.
//!
//! | Core error | Python exception |
//! |---|---|
//! | `SpaceError::DuplicateName` | `DuplicateNameError` |
//! | `SpaceError::Hook` | the hook's exception itself |
//! | `SpaceError::Constraint` | the constraint error, as the constraint module raises it |
//! | any other `SpaceError` | `SearchSpaceError` |
//! | `ConfigurationErrors` | `ConfigurationError`, with each problem's text in `problems` |
//! | `EquivalenceError::Extension` | the hook's exception itself |
//! | `EquivalenceError::Constraint` | the constraint error |
//!
//! Each message is the core error's `Display` text.

use pyo3::prelude::*;

use fhy_core::search_space::{ConfigurationErrors, EquivalenceError, SpaceError};

/// The module of the exceptions.
const MODULE: &str = "fhy_core.search_space.errors";

/// `SearchSpaceError`.
pub(super) static SEARCH_SPACE_ERROR: crate::util::exceptions::ExceptionClass =
    crate::util::exceptions::ExceptionClass::new(MODULE, "SearchSpaceError");

/// `DuplicateNameError`.
pub(super) static DUPLICATE_NAME_ERROR: crate::util::exceptions::ExceptionClass =
    crate::util::exceptions::ExceptionClass::new(MODULE, "DuplicateNameError");

/// `ConfigurationError`.
pub(super) static CONFIGURATION_ERROR: crate::util::exceptions::ExceptionClass =
    crate::util::exceptions::ExceptionClass::new(MODULE, "ConfigurationError");

/// Return the Python exception of the core's refusal to build a space, a
/// choice or an alternative.
pub(super) fn space_error_to_py(py: Python<'_>, error: SpaceError) -> PyErr {
    todo!()
}

/// Return the `ConfigurationError` of `errors`, carrying each problem's
/// text.
pub(super) fn configuration_errors_to_py(py: Python<'_>, errors: &ConfigurationErrors) -> PyErr {
    todo!()
}

/// Return the Python exception of a comparison that failed.
pub(super) fn equivalence_error_to_py(py: Python<'_>, error: EquivalenceError) -> PyErr {
    todo!()
}
