//! What the Rust-backed classes share to behave as Python dataclasses:
//! argument checks in one message style, the `__eq__` and `__repr__` a
//! dataclass generates, hashing, and arguments that may be omitted.

use std::hash::{DefaultHasher, Hash, Hasher};

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyString, PyTuple, PyType};

/// Return the `TypeError` for an argument `field` of `owner` that is not a
/// `expected`: `<owner> <field> must be <expected>, got <type>.`
///
/// # Errors
///
/// Raises what reading the name of `value`'s type raises.
pub fn build_argument_type_error(
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
///
/// # Errors
///
/// Raises the `TypeError` of [`build_argument_type_error`], expecting `a str`,
/// for a `value` that is not a `str`.
pub fn read_str<'a, 'py>(
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
/// Equal values hash equally within a process, which is all Python needs.
#[must_use]
pub fn hash_value(value: &impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// An answer that is either a plain value or a Python exception, so a class
/// whose equality or hash asks Python, and can raise, shares the members of
/// a class whose equality and hash cannot fail.
///
/// It is implemented for `bool`, `u64` and `PyResult<T>`, and sealed: no
/// other type implements it, so the set of answers can grow without a
/// breaking change.
pub trait Outcome<T>: sealed::Sealed {
    /// Return the value, or the exception.
    ///
    /// # Errors
    ///
    /// Returns the exception a `PyResult` holds.
    fn into_result(self) -> PyResult<T>;
}

/// The supertrait that seals [`Outcome`]: it is not nameable outside this
/// module.
mod sealed {
    #[expect(
        unnameable_types,
        reason = "the sealed-trait pattern: nothing outside names it"
    )]
    pub trait Sealed {}
}

impl sealed::Sealed for bool {}

impl sealed::Sealed for u64 {}

impl<T> sealed::Sealed for PyResult<T> {}

impl Outcome<bool> for bool {
    fn into_result(self) -> PyResult<bool> {
        Ok(self)
    }
}

impl Outcome<u64> for u64 {
    fn into_result(self) -> PyResult<u64> {
        Ok(self)
    }
}

impl<T> Outcome<T> for PyResult<T> {
    fn into_result(self) -> PyResult<T> {
        self
    }
}

/// Return `NotImplemented` unless `other` is an instance of exactly the
/// class of `object`, and otherwise whether `is_equal` holds for the two.
///
/// `is_equal` answers a `bool` or a `PyResult<bool>` (see [`Outcome`]), so a
/// class whose equality asks Python uses the same function.
///
/// Matches the Python implementation: the `__eq__` a dataclass generates.
///
/// # Errors
///
/// Raises what `is_equal` raises, and `TypeError` if `other` has the class
/// of `object` but is not a `T`, which only a class that is not its own
/// `PyO3` type can cause.
pub fn compare_as_dataclass<'py, T, F, R>(
    object: &Bound<'py, T>,
    other: &Bound<'py, PyAny>,
    is_equal: F,
) -> PyResult<Bound<'py, PyAny>>
where
    T: pyo3::PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + Sync,
    F: FnOnce(&T, &T) -> R,
    R: Outcome<bool>,
{
    let py = object.py();
    if !object.as_any().get_type().is(other.get_type()) {
        return Ok(py.NotImplemented().into_bound(py));
    }
    let other = other.cast::<T>()?;
    let is_equal = is_equal(object.get(), other.get()).into_result()?;
    Ok(PyBool::new(py, is_equal).to_owned().into_any())
}

/// Return whether `left` and `right` are one object or equal by Python's
/// `==`, as a dataclass compares a field.
///
/// # Errors
///
/// Raises what `==` raises.
pub fn is_same_or_equal(left: &Bound<'_, PyAny>, right: &Bound<'_, PyAny>) -> PyResult<bool> {
    if left.is(right) {
        Ok(true)
    } else {
        left.eq(right)
    }
}

/// Return the items of the iterable `values` as a tuple, `values` itself if
/// it is exactly a `tuple`; an instance of a subclass is copied.
///
/// # Errors
///
/// Raises `TypeError` if `values` is not iterable, and what iterating raises.
pub fn collect_tuple<'py>(values: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
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
///
/// # Errors
///
/// Raises what reading the class's qualified name or a field's `repr`
/// raises.
pub fn format_dataclass_repr(
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
///
/// A `#[pyfunction]` or `#[new]` takes it as `#[pyo3(signature = (value =
/// OptionalArgument::Omitted))]`: any object the caller passes, `None`
/// included, extracts as [`Given`](Self::Given).
#[derive(Debug)]
#[expect(
    clippy::exhaustive_enums,
    reason = "an argument is either omitted or given, and callers match both"
)]
pub enum OptionalArgument<'py> {
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

#[cfg(test)]
mod tests;
