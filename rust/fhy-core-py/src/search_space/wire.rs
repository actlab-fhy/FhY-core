//! The foreign parts of a search-space payload, and the depth check of a
//! payload before it reaches the core.
//!
//! [`PyResolver`] resolves a foreign variable or alternative by its type
//! id: a registered downstream Rust kind's resolver first, then the Python
//! registry, whose class decodes the data into a subclass instance that
//! becomes an adapter.
//!
//! The core decodes a space recursively, once per level of choices, so a
//! payload is refused (`RecursionError`) when its choices nest deeper than
//! Python's recursion limit, as a deep provenance is.

use pyo3::prelude::*;

use fhy_core::foreign::{Foreign, ForeignError, Part, Resolve};
use fhy_core::search_space::{Alternative, Variable};

use crate::wire::PyResolver;

impl Resolve<Part<dyn Variable>> for PyResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn Variable>, ForeignError> {
        todo!()
    }
}

impl Resolve<Part<dyn Alternative>> for PyResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn Alternative>, ForeignError> {
        todo!()
    }
}

/// Raise `RecursionError` if the choices of the V2 payload `data`, a dict
/// of the shape of `family` (`"space"`, `"choice"`, `"alternative"` or
/// `"configuration"`), nest deeper than the recursion limit.
///
/// # Errors
///
/// Raises as described, and what reading the limit raises.
pub(super) fn ensure_payload_depth(data: &Bound<'_, PyAny>, family: &str) -> PyResult<()> {
    todo!()
}

/// Raise `RecursionError` if `depth` levels of choices of a `class` are
/// deeper than the recursion limit.
///
/// # Errors
///
/// Raises as described, and what reading the limit raises.
pub(super) fn ensure_depth_within_recursion_limit(
    py: Python<'_>,
    class: &str,
    depth: usize,
) -> PyResult<()> {
    todo!()
}
