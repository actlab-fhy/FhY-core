//! `PyO3` class for [`fhy_core::op_attribute`]: `fhy_core._rs.OpAttribute`,
//! the base of `fhy_core.op_attribute.OpAttribute`.

use pyo3::prelude::*;

use fhy_core::interned::Canonical;
use fhy_core::op_attribute::OpAttribute;

use crate::described_tag::define_described_tag_class;

define_described_tag_class! {
    /// Open semantic tag attached to a compiler operation, backed by the
    /// canonical Rust [`OpAttribute`].
    class PyOpAttribute as "OpAttribute";
    seed OpAttributeSeed;
    tag OpAttribute;
    extra_methods {}
}

/// Return the canonical attribute of the Python `OpAttribute` `object`.
///
/// # Errors
///
/// Raises `TypeError` if `object` is not an `OpAttribute`.
pub(crate) fn op_attribute_from_python(
    object: &Bound<'_, PyAny>,
) -> PyResult<Canonical<OpAttribute>> {
    object
        .cast::<PyOpAttribute>()
        .map(|attribute| attribute.get().tag.clone())
        .map_err(|_not_an_attribute| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "expected an OpAttribute, got {}.",
                object
                    .get_type()
                    .name()
                    .map_or_else(|_| "?".to_owned(), |name| name.to_string())
            ))
        })
}

/// Return the single Python object of the canonical `attribute`, an
/// instance of the public `OpAttribute` class.
///
/// # Errors
///
/// Raises what importing `fhy_core.op_attribute` or building the object
/// raises.
pub(crate) fn op_attribute_to_python(
    py: Python<'_>,
    attribute: Canonical<OpAttribute>,
) -> PyResult<Bound<'_, PyAny>> {
    // The module registers the public class when it is imported.
    py.import("fhy_core.op_attribute")?;
    PyOpAttribute::to_python(py, None, attribute, None)
}
