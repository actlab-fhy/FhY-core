//! What the Rust-backed classes share to stand in for the Python
//! dataclasses they replace: argument checks in one message style, the
//! `__eq__` and `__repr__` a dataclass generates, hashing, and arguments
//! that may be omitted.

use std::hash::{DefaultHasher, Hash, Hasher};

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyString, PyTuple, PyType};

/// Return the `TypeError` for an argument `field` of `owner` that is not a
/// `expected`.
pub(crate) fn build_argument_type_error(
    owner: &str,
    field: &str,
    expected: &str,
    value: &Bound<'_, PyAny>,
) -> PyResult<PyErr> {
    Ok(PyTypeError::new_err(format!(
        "{owner} {field} must be {expected}, got {}.",
        value.get_type().name()?
    )))
}

/// Return `value` as a `str`, or raise the `TypeError` naming `owner` and
/// `field`.
pub(crate) fn read_str<'a, 'py>(
    value: &'a Bound<'py, PyAny>,
    owner: &str,
    field: &str,
) -> PyResult<&'a Bound<'py, PyString>> {
    match value.cast::<PyString>() {
        Ok(value) => Ok(value),
        Err(_not_a_str) => Err(build_argument_type_error(owner, field, "a str", value)?),
    }
}

/// Return the hash of `value` from the standard hasher.
///
/// Equal values hash equally within a process, which is all Python needs;
/// the hashes differ from the ones the replaced dataclasses computed.
pub(crate) fn hash_value(value: &impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Return `NotImplemented` unless `other` is an instance of exactly the
/// class of `object`, and otherwise whether `is_equal` holds for the two.
///
/// Matches the Python implementation: the `__eq__` a dataclass generates.
pub(crate) fn compare_as_dataclass<'py, T, F>(
    object: &Bound<'py, T>,
    other: &Bound<'py, PyAny>,
    is_equal: F,
) -> PyResult<Bound<'py, PyAny>>
where
    T: pyo3::PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + Sync,
    F: FnOnce(&T, &T) -> PyResult<bool>,
{
    let py = object.py();
    if !object.as_any().get_type().is(other.get_type()) {
        return Ok(py.NotImplemented().into_bound(py));
    }
    let other = other.cast::<T>()?;
    let is_equal = is_equal(object.get(), other.get())?;
    Ok(PyBool::new(py, is_equal).to_owned().into_any())
}

/// Return the items of the iterable `values` as a tuple, `values` itself if
/// it is a tuple.
pub(crate) fn collect_tuple<'py>(values: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    if let Ok(values) = values.cast_exact::<PyTuple>() {
        return Ok(values.clone());
    }
    PyTuple::new(
        values.py(),
        values.try_iter()?.collect::<PyResult<Vec<_>>>()?,
    )
}

/// Render `class(field=value, ...)` from the reprs of `fields`, as a
/// dataclass's `__repr__` does.
pub(crate) fn format_dataclass_repr(
    class: &Bound<'_, PyType>,
    fields: &[(&str, &Bound<'_, PyAny>)],
) -> PyResult<String> {
    let mut text = class.qualname()?.to_string();
    text.push('(');
    for (index, (name, value)) in fields.iter().enumerate() {
        if index > 0 {
            text.push_str(", ");
        }
        text.push_str(name);
        text.push('=');
        text.push_str(value.repr()?.to_str()?);
    }
    text.push(')');
    Ok(text)
}

/// A constructor argument that may be omitted, so that an explicit `None`
/// is not mistaken for the omitted argument's default.
pub(crate) enum OptionalArgument<'py> {
    /// The caller did not pass the argument.
    Omitted,
    /// The caller passed this object.
    Given(Bound<'py, PyAny>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for OptionalArgument<'py> {
    type Error = PyErr;

    fn extract(object: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self::Given(object.to_owned()))
    }
}
