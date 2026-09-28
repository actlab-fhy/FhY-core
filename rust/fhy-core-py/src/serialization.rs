//! The parts of Python's serialization framework, `fhy_core.serialization`,
//! that the Rust-backed classes use to read and write their V1 payloads
//! (and the payload shapes of the few classes whose V1 and V2 forms agree).
//!
//! V1: removed with the V1 wire format, except the field checks of the
//! classes whose V2 form is their V1 form (`Position`, `Span`, `Note`, the
//! tags). The V2 format is `wire.rs`'s.
//!
//! The core crate serializes in plain serde shapes, and the
//! `__type__`/`__data__` envelope and the Python field shapes belong to the
//! binding. The Rust-backed classes build their payloads themselves and
//! validate incoming ones exactly as the classes they replace do, raising
//! the framework's own exceptions with the same messages.

use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyMapping, PyString, PyType};

const MODULE: &str = "fhy_core.serialization";

/// Return the payload of the serializable `value` from its own
/// `serialize_to_dict`, or `None` for `None`.
///
/// Matches the Python implementation: the derived encoding of a
/// serializable field.
pub(crate) fn serialize_nested<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if value.is_none() {
        return Ok(value.clone());
    }
    value.call_method0(intern!(value.py(), "serialize_to_dict"))
}

/// Return whether `value` is a payload dict: a mapping with `str` keys and
/// serializable values, as `fhy_core.serialization.is_serialized_dict`
/// decides.
pub(crate) fn is_serialized_dict(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    crate::python::cached_attr!(value.py(), MODULE, "is_serialized_dict" => PyAny)?
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
    /// An `int` that is not a `bool`.
    Int,
    /// A nested payload dict or `None`, an optional serializable value.
    OptionalPayload,
    /// A `str` or `None`.
    OptionalStr,
    /// An `int` that is not a `bool`, or `None`.
    OptionalInt,
    /// A list of nested payload dicts, a sequence of serializable values.
    PayloadList,
    /// A literal expression's value: a `str`, `float`, `int` or `bool`.
    Literal,
    /// A list of `int`s that are not `bool`s, or `None`.
    OptionalIntList,
}

impl FieldShape {
    /// Return whether `value` has this shape.
    fn accepts(self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        let is_int = |value: &Bound<'_, PyAny>| {
            value.is_instance_of::<PyInt>() && !value.is_instance_of::<PyBool>()
        };
        match self {
            Self::Payload => is_serialized_dict(value),
            Self::Str => Ok(value.is_instance_of::<PyString>()),
            Self::Int => Ok(is_int(value)),
            Self::OptionalPayload => Ok(value.is_none() || is_serialized_dict(value)?),
            Self::OptionalStr => Ok(value.is_none() || value.is_instance_of::<PyString>()),
            Self::OptionalInt => Ok(value.is_none() || is_int(value)),
            Self::PayloadList => {
                let Ok(items) = value.cast::<PyList>() else {
                    return Ok(false);
                };
                for item in items {
                    if !is_serialized_dict(&item)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Self::OptionalIntList => {
                if value.is_none() {
                    return Ok(true);
                }
                let Ok(items) = value.cast::<PyList>() else {
                    return Ok(false);
                };
                Ok(items.iter().all(|item| is_int(&item)))
            }
            Self::Literal => Ok(value.is_instance_of::<PyString>()
                || value.is_instance_of::<PyFloat>()
                || value.is_instance_of::<PyInt>()),
        }
    }

    /// Return the type the structure error names for this shape.
    fn expected_type(self, py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        let dict_type = py.get_type::<PyDict>().into_any();
        let str_type = py.get_type::<PyString>().into_any();
        let int_type = py.get_type::<PyInt>().into_any();
        match self {
            Self::Payload => Ok(dict_type),
            Self::Str => Ok(str_type),
            Self::Int => Ok(int_type),
            Self::OptionalPayload => dict_type.bitor(py.None()),
            Self::OptionalStr => str_type.bitor(py.None()),
            Self::OptionalInt => int_type.bitor(py.None()),
            Self::PayloadList => Ok(py.get_type::<PyList>().into_any()),
            Self::OptionalIntList => py
                .get_type::<PyList>()
                .get_item(py.get_type::<PyInt>())?
                .bitor(py.None()),
            Self::Literal => str_type
                .bitor(py.get_type::<PyFloat>())?
                .bitor(int_type)?
                .bitor(py.get_type::<PyBool>()),
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
    if let Some(values) = read_fields_of_shape(data, &fields)? {
        return Ok(values);
    }
    let py = cls.py();
    let expected = PyDict::new(py);
    for (name, shape) in fields {
        expected.set_item(name, shape.expected_type(py)?)?;
    }
    Err(crate::exceptions::DESERIALIZATION_DICT_STRUCTURE_ERROR.err(py, (cls, expected, data)))
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

/// Build an instance of `cls` from the decoded payload `fields` through
/// `cls.construct_from_fields(fields)`.
///
/// Matches the Python implementation: the construction step of the derived
/// `Serializable.deserialize_from_dict`.
///
/// # Errors
///
/// Raises what the construction raises, except that a `ValueError` or
/// `TypeError` outside the serialization hierarchy becomes a
/// `DeserializationValueError` with its message, caused by it.
pub(crate) fn construct_from_decoded_fields<'py>(
    cls: &Bound<'py, PyType>,
    fields: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = cls.py();
    match cls.call_method1(pyo3::intern!(py, "construct_from_fields"), (fields,)) {
        Ok(instance) => Ok(instance),
        Err(error)
            if !error.is_instance(py, crate::exceptions::SERIALIZATION_ERROR.class(py)?)
                && (error.is_instance_of::<PyValueError>(py)
                    || error.is_instance_of::<PyTypeError>(py)) =>
        {
            let message = error.value(py).str()?;
            let wrapped = crate::exceptions::DESERIALIZATION_VALUE_ERROR.err(py, (message,));
            wrapped.set_cause(py, Some(error));
            Err(wrapped)
        }
        Err(error) => Err(error),
    }
}
