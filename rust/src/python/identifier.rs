//! `PyO3` bindings for the process-global identifier id counter in
//! [`crate::identifier`].
//!
//! `fhy_core.identifier.Identifier` draws its ids from these functions when
//! the package runs on the Rust backend. The counter itself, including its
//! refusal to wrap, lives in the pure-Rust core.

use pyo3::prelude::*;

use crate::identifier as rust_identifier;

/// Draw the next identifier id from the process-global counter.
///
/// A counter that has reached `2**64 - 1` panics, which Python sees as a
/// `pyo3_runtime.PanicException`, instead of wrapping.
#[pyfunction]
#[must_use]
pub(crate) fn allocate_identifier_id() -> u64 {
    rust_identifier::allocate_id()
}

/// Advance the process-global identifier counter so `identifier_id` is
/// never issued, leaving it unchanged when it is already past the id.
///
/// Advancing past `2**64 - 1` panics, which Python sees as a
/// `pyo3_runtime.PanicException`, instead of wrapping.
#[pyfunction]
#[pyo3(signature = (identifier_id, /))]
pub(crate) fn advance_identifier_counter_past(identifier_id: u64) {
    rust_identifier::advance_counter_past(identifier_id);
}
