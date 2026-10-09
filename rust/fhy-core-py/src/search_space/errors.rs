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
//! | `StepDomainError` | `StepDomainError` |
//! | `TraceError::CoordinateOutOfDomain`, `Inadmissible` | `InadmissibleAnswerError` |
//! | `TraceError::NotEnumerable` | `NotEnumerableError` |
//! | `TraceError::DeadEnd` | `DeadEndError` |
//! | `TraceError::Oracle`, `Hook` | the oracle's or hook's exception itself; a `ReplayError` source as `ReplayMismatchError` |
//! | `TraceError::Configuration` | as `ConfigurationErrors` |
//! | any other `TraceError` | `TraceError` |
//! | `ReplayError` | `ReplayMismatchError` |
//! | `MeasurementError` | `MeasurementError` |
//!
//! Each message is the core error's `Display` text.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use fhy_core::constraint::ConstraintError;
use fhy_core::foreign::BoxError;
use fhy_core::search_space::{
    ConfigurationError, ConfigurationErrors, EquivalenceError, MeasurementError, ReplayError,
    SpaceError, StepDomainError, TraceError,
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

/// `StepDomainError`.
static STEP_DOMAIN_ERROR: ExceptionClass = ExceptionClass::new(MODULE, "StepDomainError");

/// `TraceError`.
static TRACE_ERROR: ExceptionClass = ExceptionClass::new(MODULE, "TraceError");

/// `InadmissibleAnswerError`.
static INADMISSIBLE_ANSWER_ERROR: ExceptionClass =
    ExceptionClass::new(MODULE, "InadmissibleAnswerError");

/// `ReplayMismatchError`.
static REPLAY_MISMATCH_ERROR: ExceptionClass = ExceptionClass::new(MODULE, "ReplayMismatchError");

/// `NotEnumerableError`.
static NOT_ENUMERABLE_ERROR: ExceptionClass = ExceptionClass::new(MODULE, "NotEnumerableError");

/// `MeasurementError`.
static MEASUREMENT_ERROR: ExceptionClass = ExceptionClass::new(MODULE, "MeasurementError");

/// `DeadEndError`.
static DEAD_END_ERROR: ExceptionClass = ExceptionClass::new(MODULE, "DeadEndError");

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

/// Return the Python exception of a domain the core refuses to build:
/// `StepDomainError`.
pub(super) fn step_domain_error_to_py(py: Python<'_>, error: &StepDomainError) -> PyErr {
    STEP_DOMAIN_ERROR.err(py, (error.to_string(),))
}

/// Return the Python exception of a run the core stopped:
/// `InadmissibleAnswerError` for an answer outside its domain or not
/// admissible, `NotEnumerableError`, `DeadEndError`, `StepDomainError`, the
/// oracle's or the hook's exception as itself, `ReplayMismatchError` for a
/// replay's mismatch, the configuration's errors as
/// [`configuration_errors_to_py`] raises them, and `TraceError` otherwise.
pub(super) fn trace_error_to_py(py: Python<'_>, error: TraceError) -> PyErr {
    let text = error.to_string();
    match error {
        TraceError::Domain(error) => step_domain_error_to_py(py, &error),
        TraceError::CoordinateOutOfDomain { .. } | TraceError::Inadmissible { .. } => {
            INADMISSIBLE_ANSWER_ERROR.err(py, (text,))
        }
        TraceError::NotEnumerable { .. } => NOT_ENUMERABLE_ERROR.err(py, (text,)),
        TraceError::DeadEnd { .. } => DEAD_END_ERROR.err(py, (text,)),
        TraceError::Oracle { source, .. } | TraceError::Hook { source, .. } => {
            boxed_error_to_py(py, source, &text)
        }
        TraceError::Configuration(errors) => configuration_errors_to_py(py, &errors),
        _ => TRACE_ERROR.err(py, (text,)),
    }
}

/// Return the Python exception of the error `source` an oracle or a hook
/// returned: a Python exception as itself, a replay's mismatch as
/// [`replay_error_to_py`] raises it, a run's error as
/// [`trace_error_to_py`] raises it, and anything else as `TraceError`,
/// `text` followed by the error's own.
pub(super) fn boxed_error_to_py(py: Python<'_>, source: BoxError, text: &str) -> PyErr {
    let source = match unbox_py_err(source) {
        Ok(raised) => return raised,
        Err(source) => source,
    };
    let source = match source.downcast::<ReplayError>() {
        Ok(error) => return replay_error_to_py(py, *error),
        Err(source) => source,
    };
    match source.downcast::<TraceError>() {
        Ok(error) => trace_error_to_py(py, *error),
        Err(source) => TRACE_ERROR.err(py, (format!("{text}: {source}"),)),
    }
}

/// Return the `TraceError` of the text `text`.
pub(super) fn trace_error_text_to_py(py: Python<'_>, text: &str) -> PyErr {
    TRACE_ERROR.err(py, (text.to_owned(),))
}

/// Return the Python exception of a replay's mismatch:
/// `ReplayMismatchError`, or what [`trace_error_to_py`] and
/// [`configuration_errors_to_py`] raise for the errors it holds.
pub(super) fn replay_error_to_py(py: Python<'_>, error: ReplayError) -> PyErr {
    match error {
        ReplayError::Configuration(errors) => configuration_errors_to_py(py, &errors),
        ReplayError::Trace(error) => trace_error_to_py(py, *error),
        error => REPLAY_MISMATCH_ERROR.err(py, (error.to_string(),)),
    }
}

/// Return the Python exception of an objective or a measurement the core
/// refuses: `MeasurementError`.
pub(super) fn measurement_error_to_py(py: Python<'_>, error: &MeasurementError) -> PyErr {
    MEASUREMENT_ERROR.err(py, (error.to_string(),))
}
