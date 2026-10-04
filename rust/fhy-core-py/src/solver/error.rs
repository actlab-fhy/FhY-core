//! The Python exceptions of the solver's errors, and its warnings.
//!
//! Each core error raises the exception class the Python API documents,
//! with the core's text. A backend written in Python fails with its own
//! exception, which propagates as the same object; the SymPy backend's
//! failure raises the bridge's exception, and another Rust backend's
//! `SolverBackendError`.

use std::collections::HashMap;

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;

use fhy_core::expression::{Expression, SymbolType};
use fhy_core::identifier::Identifier;
use fhy_core::solver::{Hazard, LoweringError, SolveError};

use crate::error::IntoPyErr;
use crate::expression::render_expression_repr;

use super::sympy::SympyError;
use super::values::symbol_type_name;

/// Return `SolverCapabilityError` with `message`.
pub(super) fn capability_error(py: Python<'_>, message: String) -> PyErr {
    crate::util::exceptions::SOLVER_CAPABILITY_ERROR.err(py, (message,))
}

/// Return `UndecidableError(message, reason=reason)`.
pub(super) fn undecidable_error(py: Python<'_>, message: String, reason: &str) -> PyErr {
    let keywords = PyDict::new(py);
    if let Err(error) = keywords.set_item(intern!(py, "reason"), reason) {
        return error;
    }
    crate::util::exceptions::UNDECIDABLE_ERROR
        .build(py, (message,), Some(&keywords))
        .unwrap_or_else(|error| error)
}

/// Return the Python exception of a lowering refusal.
pub(super) fn lowering_error_to_py(py: Python<'_>, error: LoweringError) -> PyErr {
    match error {
        LoweringError::MissingSymbolTypes(_) => PyKeyError::new_err(error.to_string()),
        LoweringError::IllTyped(error) => error.into_py_err(),
        LoweringError::NativeConstants(_) => {
            crate::util::exceptions::NATIVE_CONSTANT_LOWERING_ERROR.err(py, (error.to_string(),))
        }
        other => PyTypeError::new_err(other.to_string()),
    }
}

/// Return the Python exception of a failed question.
pub(crate) fn solve_error_to_py(py: Python<'_>, error: SolveError) -> PyErr {
    let text = error.to_string();
    match error {
        SolveError::NoCapableBackend(_) => capability_error(py, text),
        SolveError::MissingSymbolTypes(_) => PyKeyError::new_err(text),
        SolveError::IllTyped(error) => error.into_py_err(),
        SolveError::BoundNativeConstant(_) => {
            crate::util::exceptions::NATIVE_CONSTANT_BINDING_ERROR.err(py, (text,))
        }
        SolveError::Substitution(source) => PyValueError::new_err(format!("{text}: {source}")),
        SolveError::Lowering(error) => lowering_error_to_py(py, error),
        SolveError::Backend { backend, source } => {
            match crate::util::exceptions::unbox_py_err(source) {
                Ok(error) => error,
                Err(source) => match source.downcast::<SympyError>() {
                    Ok(error) => super::sympy::sympy_error_to_py(py, *error, true),
                    Err(source) => crate::util::exceptions::SOLVER_BACKEND_ERROR
                        .err(py, (format!("the backend {backend:?} failed: {source}"),)),
                },
            }
        }
        _ => PyRuntimeError::new_err(text),
    }
}

/// Return whether `error` is a backend failure [`solve_error_to_py`]
/// raises as a `PassExecutionError`, such as SymPy failing to lower an
/// expression, or a Python backend raising one.
pub(crate) fn is_pass_execution_failure(py: Python<'_>, error: &SolveError) -> bool {
    let SolveError::Backend { source, .. } = error else {
        return false;
    };
    if let Some(error) = source.downcast_ref::<PyErr>() {
        return crate::util::exceptions::PASS_EXECUTION_ERROR.is_instance_of(py, error);
    }
    source
        .downcast_ref::<SympyError>()
        .is_some_and(|error| super::sympy::is_raised_as_pass_error(py, error))
}

/// Return `fhy_core.symbolic.solver`'s logger.
fn solver_logger(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    static LOGGER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    LOGGER
        .get_or_try_init(py, || {
            py.import(intern!(py, "fhy_core.logger"))?
                .call_method1(intern!(py, "get_logger"), ("fhy_core.symbolic.solver",))
                .map(Bound::unbind)
        })
        .map(|logger| logger.bind(py).clone())
}

/// Return each identifier of `node` with its sort, as `x::7: INT`, ordered
/// by id; `none` without identifiers, and `unknown` for a sort the symbol
/// types lack.
fn render_identifier_sorts(
    node: &Expression,
    symbol_types: &HashMap<Identifier, SymbolType>,
) -> String {
    let mut identifiers: Vec<Identifier> = node.free_identifiers().into_iter().collect();
    if identifiers.is_empty() {
        return "none".to_owned();
    }
    identifiers.sort_by_key(Identifier::id);
    identifiers
        .iter()
        .map(|identifier| {
            let sort = symbol_types
                .get(identifier)
                .map_or("unknown", |sort| symbol_type_name(*sort));
            format!("{identifier:?}: {sort}")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Log the refusal of `hazard` by the entry point `context` at WARNING, on
/// `fhy_core.symbolic.solver`: the core's text, the refused node's `repr`,
/// and the sorts of its identifiers.
pub(crate) fn warn_hazard(
    py: Python<'_>,
    context: &str,
    hazard: &Hazard,
    symbol_types: &HashMap<Identifier, SymbolType>,
) -> PyResult<()> {
    let detail = match hazard.node() {
        None => String::new(),
        Some(node) if matches!(hazard, Hazard::NonFiniteLiteral(_)) => {
            format!(": node {}", render_expression_repr(node))
        }
        Some(node) => format!(
            ": node {}; identifier sorts at that node: {}",
            render_expression_repr(node),
            render_identifier_sorts(node, symbol_types)
        ),
    };
    solver_logger(py)?.call_method1(
        intern!(py, "warning"),
        (
            "%s: %s%s. The expression is not handed to the solver; bounding \
             timeout_milliseconds cannot change this outcome.",
            context,
            hazard.to_string(),
            detail,
        ),
    )?;
    Ok(())
}

/// Log that the backend `backend` answered `unknown` for `reason` at
/// WARNING, on `fhy_core.symbolic.solver`.
pub(crate) fn warn_unknown(
    py: Python<'_>,
    context: &str,
    backend: &str,
    reason: &str,
) -> PyResult<()> {
    solver_logger(py)?.call_method1(
        intern!(py, "warning"),
        (
            "%s: the backend %s answered unknown (%s)",
            context,
            backend,
            reason,
        ),
    )?;
    Ok(())
}
