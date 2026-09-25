//! The parts of Python's serialization framework, `fhy_core.serialization`,
//! that the Rust-backed classes use to keep today's Python payloads.
//!
//! The core crate serializes in plain serde shapes, and the
//! `__type__`/`__data__` envelope and the Python field shapes belong to the
//! binding. The Rust-backed classes build their payloads themselves and
//! validate incoming ones exactly as the classes they replace do, raising
//! the framework's own exceptions with the same messages.

use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyMapping, PyString, PyType};

const MODULE: &str = "fhy_core.serialization";

/// Return `fhy_core.serialization.DeserializationValueError`.
pub(crate) fn deserialization_value_error_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    CLASS.import(py, MODULE, "DeserializationValueError")
}

/// Return whether `value` is a payload dict: a mapping with `str` keys and
/// serializable values, as `fhy_core.serialization.is_serialized_dict`
/// decides.
pub(crate) fn is_serialized_dict(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    static FUNCTION: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    FUNCTION
        .import(value.py(), MODULE, "is_serialized_dict")?
        .call1((value,))?
        .is_truthy()
}

/// What a payload field must hold, as the derived deserialization of the
/// replaced dataclasses checks it.
#[derive(Debug, Clone, Copy)]
pub(crate) enum FieldShape {
    /// A nested payload dict, the field of a serializable value.
    Payload,
    /// A `str`.
    Str,
    /// A nested payload dict or `None`, an optional serializable value.
    OptionalPayload,
}

impl FieldShape {
    /// Return whether `value` has this shape.
    fn accepts(self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        match self {
            Self::Payload => is_serialized_dict(value),
            Self::Str => Ok(value.is_instance_of::<PyString>()),
            Self::OptionalPayload => Ok(value.is_none() || is_serialized_dict(value)?),
        }
    }

    /// Return the type the structure error names for this shape.
    fn expected_type(self, py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        let dict_type = py.get_type::<PyDict>().into_any();
        match self {
            Self::Payload => Ok(dict_type),
            Self::Str => Ok(py.get_type::<PyString>().into_any()),
            Self::OptionalPayload => dict_type.bitor(py.None()),
        }
    }
}

/// Check that `data` holds exactly the fields `fields` names, each of its
/// shape, and return its values in the order of `fields`.
///
/// Matches the Python implementation: the structure check of the derived
/// `Serializable.deserialize_from_dict`.
///
/// # Errors
///
/// Raises `DeserializationDictStructureError(cls, <expected fields>, data)`
/// if a field is missing, extra or of the wrong shape.
pub(crate) fn read_payload_fields<'py, const N: usize>(
    cls: &Bound<'py, PyType>,
    data: &Bound<'py, PyAny>,
    fields: [(&str, FieldShape); N],
) -> PyResult<[Bound<'py, PyAny>; N]> {
    static STRUCTURE_ERROR: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    if let Some(values) = read_fields_of_shape(data, &fields)? {
        return Ok(values);
    }
    let py = cls.py();
    let expected = PyDict::new(py);
    for (name, shape) in fields {
        expected.set_item(name, shape.expected_type(py)?)?;
    }
    let error = STRUCTURE_ERROR
        .import(py, MODULE, "DeserializationDictStructureError")?
        .call1((cls, expected, data))?;
    Err(PyErr::from_value(error))
}

/// Return the values of `fields` in `data`, or `None` if `data` is not a
/// mapping holding exactly those fields, each of its shape.
fn read_fields_of_shape<'py, const N: usize>(
    data: &Bound<'py, PyAny>,
    fields: &[(&str, FieldShape); N],
) -> PyResult<Option<[Bound<'py, PyAny>; N]>> {
    let Ok(mapping) = data.cast::<PyMapping>() else {
        return Ok(None);
    };
    if mapping.len()? != N {
        return Ok(None);
    }
    let mut values = Vec::with_capacity(N);
    for (name, shape) in fields {
        let name = PyString::new(data.py(), name);
        if !mapping.contains(&name)? {
            return Ok(None);
        }
        let value = mapping.get_item(&name)?;
        if !shape.accepts(&value)? {
            return Ok(None);
        }
        values.push(value);
    }
    Ok(values.try_into().ok())
}

/// Return the values of the fields `names` in the mapping `fields`, in
/// order, as the constructor call `cls(**fields)` of the replaced
/// dataclasses would receive them.
///
/// The last `optional` names may be missing, and read as `None`.
///
/// # Errors
///
/// Raises `TypeError` if `fields` is not a mapping, lacks a required name,
/// or holds a name not in `names`.
pub(crate) fn read_constructor_fields<'py, const N: usize>(
    cls: &Bound<'py, PyType>,
    fields: &Bound<'py, PyAny>,
    names: [&str; N],
    optional: usize,
) -> PyResult<[Bound<'py, PyAny>; N]> {
    let py = cls.py();
    let mapping = fields.cast::<PyMapping>()?;
    let mut values = Vec::with_capacity(N);
    let mut found = 0;
    for (index, name) in names.into_iter().enumerate() {
        match mapping.get_item(name) {
            Ok(value) => {
                found += 1;
                values.push(value);
            }
            Err(error) if error.is_instance_of::<PyKeyError>(py) && index + optional >= N => {
                values.push(py.None().into_bound(py));
            }
            Err(error) if error.is_instance_of::<PyKeyError>(py) => {
                return Err(PyTypeError::new_err(format!(
                    "{}.construct_from_fields() missing field '{name}'",
                    cls.name()?
                )));
            }
            Err(error) => return Err(error),
        }
    }
    if mapping.len()? != found {
        return Err(PyTypeError::new_err(format!(
            "{}.construct_from_fields() takes only the fields {}, got {}",
            cls.name()?,
            names.map(|name| format!("'{name}'")).join(", "),
            mapping.keys()?.repr()?,
        )));
    }
    Ok(values
        .try_into()
        .unwrap_or_else(|_values: Vec<_>| unreachable!("one value per name")))
}
