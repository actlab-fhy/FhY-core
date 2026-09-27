//! `PyO3` bindings for [`fhy_core::identifier`]: the process-global id
//! counter, and the conversion of identifiers between the two languages.
//!
//! `fhy_core.identifier.Identifier` draws its ids from these functions. The
//! counter itself, including its refusal to wrap and its cap on payload ids,
//! lives in the pure-Rust core. A counter that cannot advance raises
//! `RuntimeError("identifier id space exhausted")`, and advancing past an id
//! outside `[0, 2**62)` that this process did not issue below `2**63`
//! raises `OverflowError`.
//!
//! The Python `Identifier` stays a Python class (pattern P1), so a Rust
//! value that holds an identifier converts it by id and name hint: into Rust
//! through [`Identifier::try_restore`], and back into Python through
//! `Identifier.deserialize_from_dict`. Neither direction issues a new id.

use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyTypeError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyString, PyType};

use fhy_core::identifier::{self as rust_identifier, IdOutOfRange, IdSpaceExhausted, Identifier};

use crate::error::{IntoPyErr, IntoPyResult};

/// Raises `RuntimeError` with the core's message when the counter cannot
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
/// reached `2**63`.
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
/// `identifier_id` is outside the payload range: `PyO3` rejects an id
/// outside `[0, 2**64)` before the call, and the core rejects one at or
/// above `2**63`, or at or above `2**62` that this process did not issue.
#[pyfunction]
#[pyo3(signature = (identifier_id, /))]
pub(crate) fn advance_identifier_counter_past(identifier_id: u64) -> PyResult<()> {
    rust_identifier::try_advance_counter_past(identifier_id).into_py_result()
}

/// Return the id the process-global counter issues next, so
/// `Identifier.deserialize_from_dict` can tell an id this process issued at
/// or above `2**62` from a foreign one before it touches the counter.
#[pyfunction]
pub(crate) fn next_identifier_id() -> u64 {
    rust_identifier::next_id()
}

/// The Python `fhy_core.identifier.Identifier` class, which stays a Python
/// class (pattern P1).
fn python_identifier_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static IDENTIFIER_CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    IDENTIFIER_CLASS.import(py, "fhy_core.identifier", "Identifier")
}

/// Return a new Python `Identifier` named `name_hint`, with a new id.
///
/// # Errors
///
/// Raises whatever the constructor raises.
pub(crate) fn new_python_identifier<'py>(
    py: Python<'py>,
    name_hint: &str,
) -> PyResult<Bound<'py, PyAny>> {
    python_identifier_class(py)?.call1((name_hint,))
}

/// Return the id of `object` if it is a Python `Identifier`, or `None` for
/// any other object.
///
/// # Errors
///
/// Raises whatever reading the identifier's `id` raises.
pub(crate) fn read_identifier_id(object: &Bound<'_, PyAny>) -> PyResult<Option<u64>> {
    if !object.is_instance(python_identifier_class(object.py())?)? {
        return Ok(None);
    }
    object
        .getattr(intern!(object.py(), "id"))?
        .extract()
        .map(Some)
}

/// Convert a Python `Identifier` to the Rust identifier with the same id and
/// name hint.
///
/// Restoring advances the Rust counter past the id, as deserialization does.
/// That never changes the counter, since every Python identifier's id was
/// issued by, or already advanced, the same counter.
///
/// # Errors
///
/// Raises `TypeError` naming `owner` and `field` if `object` is not an
/// `Identifier`, and `OverflowError` if its id is outside the payload range,
/// which no `Identifier` built by this process's counter or deserialization
/// is.
pub(crate) fn restore_identifier(
    object: &Bound<'_, PyAny>,
    owner: &str,
    field: &str,
) -> PyResult<Identifier> {
    let py = object.py();
    if object.get_type().is(python_identifier_class(py)?) {
        // An exact `Identifier` holds its id and name hint in the instance
        // attributes its properties read, so read them without the
        // properties' Python calls.
        let id = object.getattr(intern!(py, "_id"))?.extract()?;
        let name_hint = object.getattr(intern!(py, "_name_hint"))?;
        return Identifier::try_restore(id, name_hint.cast::<PyString>()?.to_str()?)
            .into_py_result();
    }
    let Some(id) = read_identifier_id(object)? else {
        return Err(PyTypeError::new_err(format!(
            "{owner} {field} must be an Identifier, got {}.",
            object.get_type().name()?
        )));
    };
    let name_hint = object.getattr(intern!(object.py(), "name_hint"))?;
    Identifier::try_restore(id, name_hint.cast::<PyString>()?.to_str()?).into_py_result()
}

/// Return the payload `Identifier.serialize_to_dict` returns for
/// `identifier`: `{"id": .., "name_hint": ..}`.
///
/// # Errors
///
/// Raises whatever building the dict raises.
pub(crate) fn serialize_identifier<'py>(
    py: Python<'py>,
    identifier: &Identifier,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item(intern!(py, "id"), identifier.id())?;
    payload.set_item(intern!(py, "name_hint"), identifier.name_hint())?;
    Ok(payload)
}

/// Build the Python `Identifier` with `identifier`'s id and name hint,
/// through its deserialization path, which never issues a new id.
///
/// # Errors
///
/// Raises whatever `Identifier.deserialize_from_dict` raises.
pub(crate) fn identifier_to_python<'py>(
    py: Python<'py>,
    identifier: &Identifier,
) -> PyResult<Bound<'py, PyAny>> {
    python_identifier_class(py)?.call_method1(
        intern!(py, "deserialize_from_dict"),
        (serialize_identifier(py, identifier)?,),
    )
}

/// Build the Python `Identifier` from a payload, through
/// `Identifier.deserialize_from_dict`, so a malformed payload raises exactly
/// the Python implementation's errors.
///
/// # Errors
///
/// Raises whatever `Identifier.deserialize_from_dict` raises.
pub(crate) fn deserialize_identifier<'py>(
    payload: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = payload.py();
    python_identifier_class(py)?.call_method1(intern!(py, "deserialize_from_dict"), (payload,))
}
