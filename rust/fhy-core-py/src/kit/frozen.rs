//! The frozen contract of `fhy_core.traits.frozen`, for the Rust-backed
//! classes that stand in for `FrozenMixin` subclasses.
//!
//! A Rust-backed class is registered as a virtual subclass of `FrozenMixin`
//! rather than inheriting it, so it implements the mixin's observable
//! behavior itself: its instances are always frozen, and mutating one raises
//! `FrozenMutationError` with the mixin's message.

use pyo3::prelude::*;

/// Refuse to set attribute `name` of the frozen `object`: always
/// `FrozenMutationError`.
///
/// Matches the Python implementation: `FrozenMixin.__setattr__`.
pub(crate) fn refuse_attribute_assignment(object: &Bound<'_, PyAny>, name: &str) -> PyResult<()> {
    Err(build_frozen_mutation_error(object, "modify", name)?)
}

/// Refuse to delete attribute `name` of the frozen `object`: always
/// `FrozenMutationError`.
///
/// Matches the Python implementation: `FrozenMixin.__delattr__`.
pub(crate) fn refuse_attribute_deletion(object: &Bound<'_, PyAny>, name: &str) -> PyResult<()> {
    Err(build_frozen_mutation_error(object, "delete", name)?)
}

/// Return the `FrozenMutationError` for modifying (`action` "modify") or
/// deleting (`action` "delete") attribute `name` of the frozen `object`.
fn build_frozen_mutation_error(
    object: &Bound<'_, PyAny>,
    action: &str,
    name: &str,
) -> PyResult<PyErr> {
    let message = format!(
        "Cannot {action} \"{name}\" on frozen {}.",
        object.get_type().name()?
    );
    Ok(crate::kit::exceptions::FROZEN_MUTATION_ERROR.err(object.py(), (message,)))
}
