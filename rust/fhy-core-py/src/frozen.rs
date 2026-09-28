//! The frozen contract of `fhy_core.traits.frozen`, for the Rust-backed
//! classes that stand in for `FrozenMixin` subclasses.
//!
//! A Rust-backed class is registered as a virtual subclass of `FrozenMixin`
//! rather than inheriting it, so it implements the mixin's observable
//! behavior itself: its instances are always frozen, and mutating one raises
//! `FrozenMutationError` with the mixin's message.

use pyo3::prelude::*;
use pyo3::types::PyType;

/// Return `fhy_core.traits.frozen.FrozenMutationError`.
fn frozen_mutation_error_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    crate::python::cached_attr!(py, "fhy_core.traits.frozen", "FrozenMutationError" => PyType)
}

/// Return the `FrozenMutationError` for modifying (`action` "modify") or
/// deleting (`action` "delete") attribute `name` of the frozen `object`.
///
/// Matches the Python implementation: `FrozenMixin.__setattr__` and
/// `FrozenMixin.__delattr__`.
pub(crate) fn build_frozen_mutation_error(
    object: &Bound<'_, PyAny>,
    action: &str,
    name: &str,
) -> PyResult<PyErr> {
    let message = format!(
        "Cannot {action} \"{name}\" on frozen {}.",
        object.get_type().name()?
    );
    Ok(PyErr::from_value(
        frozen_mutation_error_class(object.py())?.call1((message,))?,
    ))
}
