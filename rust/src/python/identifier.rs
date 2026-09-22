//! `PyO3` bindings for the process-global identifier id counter in
//! [`crate::identifier`].
//!
//! `fhy_core.identifier.Identifier` draws its ids from these functions when
//! the package runs on the Rust backend. The counter itself, including its
//! refusal to wrap, lives in the pure-Rust core. A counter that cannot
//! advance raises `RuntimeError("identifier id space exhausted")`, as the
//! pure-Python counter does.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::identifier::{self as rust_identifier, IdSpaceExhausted};

impl From<IdSpaceExhausted> for PyErr {
    fn from(exhausted: IdSpaceExhausted) -> Self {
        PyRuntimeError::new_err(exhausted.to_string())
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
    Ok(rust_identifier::try_allocate_id()?)
}

/// Advance the process-global identifier counter so `identifier_id` is
/// never issued, leaving it unchanged when it is already past the id.
///
/// # Errors
///
/// Raises `RuntimeError`, leaving the counter unchanged, if `identifier_id`
/// is `2**64 - 1`, which the counter cannot advance past.
#[pyfunction]
#[pyo3(signature = (identifier_id, /))]
pub(crate) fn advance_identifier_counter_past(identifier_id: u64) -> PyResult<()> {
    Ok(rust_identifier::try_advance_counter_past(identifier_id)?)
}
