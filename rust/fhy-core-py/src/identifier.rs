//! `PyO3` bindings for the process-global identifier id counter in
//! [`fhy_core::identifier`].
//!
//! `fhy_core.identifier.Identifier` draws its ids from these functions when
//! the package runs on the Rust backend. The counter itself, including its
//! refusal to wrap and its cap on payload ids, lives in the pure-Rust core.
//! As the pure-Python counter does, a counter that cannot advance raises
//! `RuntimeError("identifier id space exhausted")`, and advancing past an id
//! outside `[0, 2**63)` raises `OverflowError`.

use pyo3::exceptions::{PyOverflowError, PyRuntimeError};
use pyo3::prelude::*;

use fhy_core::identifier::{self as rust_identifier, IdOutOfRange, IdSpaceExhausted};

use crate::error::{IntoPyErr, IntoPyResult};

/// Raises the `RuntimeError` the pure-Python counter raises when it cannot
/// advance.
impl IntoPyErr for IdSpaceExhausted {
    fn into_py_err(self) -> PyErr {
        PyRuntimeError::new_err(self.to_string())
    }
}

/// Raises the `OverflowError` `PyO3` raises for an id outside
/// `[0, 2**64)`, so every id the counter cannot advance past raises the same
/// class.
impl IntoPyErr for IdOutOfRange {
    fn into_py_err(self) -> PyErr {
        PyOverflowError::new_err(self.to_string())
    }
}

/// Draw the next identifier id from the process-global counter.
///
/// # Errors
///
/// Raises `RuntimeError`, leaving the counter unchanged, if the counter has
/// reached `2**64 - 1`.
#[pyfunction]
pub(crate) fn allocate_identifier_id() -> PyResult<u64> {
    rust_identifier::try_allocate_id().into_py_result()
}

/// Advance the process-global identifier counter so `identifier_id` is
/// never issued, leaving it unchanged when it is already past the id.
///
/// # Errors
///
/// Raises `OverflowError`, leaving the counter unchanged, if
/// `identifier_id` is outside `[0, 2**63)`: `PyO3` rejects an id outside
/// `[0, 2**64)` before the call, and the core rejects one at or above the
/// cap `2**63`.
#[pyfunction]
#[pyo3(signature = (identifier_id, /))]
pub(crate) fn advance_identifier_counter_past(identifier_id: u64) -> PyResult<()> {
    rust_identifier::try_advance_counter_past(identifier_id).into_py_result()
}
