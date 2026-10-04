//! The parts of Python's serialization framework, `fhy_core.serialization`,
//! that the Rust-backed classes use to read and write their payloads.
//!
//! The core crate serializes in plain serde shapes, and the
//! `__type__`/`__data__` envelope and the Python field shapes belong to the
//! binding. The Rust-backed classes build their payloads themselves and
//! validate incoming ones as the framework's derived deserialization does,
//! raising the framework's own exceptions with its messages.
//!
//! A reader checks the payload's structure with [`read_payload_fields`], a
//! list of fields each of a [`FieldShape`], then decodes nested values
//! ([`read_nested_value`], [`read_nested_list`]) and builds the instance
//! with [`construct_from_decoded_fields`], which raises the framework's
//! `DeserializationValueError` for a value the class refuses. A downstream
//! class takes the shapes it needs: the list of [`FieldShape`] variants is
//! `#[non_exhaustive]`.
//!
//! V1 (the deprecated envelope format) is read and written by the classes'
//! own code over these functions, except that the V2 wire format is
//! `wire.rs`'s.

use pyo3::exceptions::{PyKeyError, PyOverflowError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyMapping, PyString, PyType};

use super::exceptions::{
    DESERIALIZATION_DICT_STRUCTURE_ERROR, DESERIALIZATION_VALUE_ERROR, SERIALIZATION_ERROR,
};

const MODULE: &str = "fhy_core.serialization";

/// Return the payload of the serializable `value` from its own
/// `serialize_to_dict`, or `None` for `None`.
///
/// Matches the Python implementation: the derived encoding of a
/// serializable field.
///
/// # Errors
///
/// Raises what `value.serialize_to_dict()` raises, `AttributeError` if
/// `value` has no such method.
pub fn serialize_nested<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if value.is_none() {
        return Ok(value.clone());
    }
    value.call_method0(intern!(value.py(), "serialize_to_dict"))
}

/// Return whether `value` is a payload dict: a mapping with `str` keys and
/// serializable values, as `fhy_core.serialization.is_serialized_dict`
/// decides.
///
/// # Errors
///
/// Raises what importing `fhy_core.serialization` or the check raises.
pub fn is_serialized_dict(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    crate::cached_attr!(value.py(), MODULE, "is_serialized_dict" => PyAny)?
        .call1((value,))?
        .is_truthy()
}

/// What a payload field must hold, as the derived deserialization of a
/// dataclass checks it.
///
/// The shapes `fhy_core`'s classes need come first, and the ones that a
/// downstream class with nested readers needs follow: [`Object`](Self::Object)
/// and [`Any`](Self::Any) leave the check of a field to the reader of the
/// nested value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FieldShape {
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
    /// A `bool`.
    Bool,
    /// A list of `int`s that are not `bool`s.
    IntList,
    /// A mapping, which a nested value's own reader checks further.
    Object,
    /// A mapping or `None`.
    OptionalObject,
    /// A list, whose items a nested reader checks.
    List,
    /// A list of mappings.
    ObjectList,
    /// A mapping whose values are mappings.
    ObjectMap,
    /// A list of lists of `int`s that are not `bool`s.
    IntListList,
    /// A list of lists of mappings.
    ObjectListList,
    /// Anything: the field's own reader checks it.
    Any,
}

/// Return whether `value` is an `int` that is not a `bool`.
fn is_int(value: &Bound<'_, PyAny>) -> bool {
    value.is_instance_of::<PyInt>() && !value.is_instance_of::<PyBool>()
}

/// Return whether `value` is a mapping.
fn is_mapping(value: &Bound<'_, PyAny>) -> bool {
    value.cast::<PyMapping>().is_ok()
}

impl FieldShape {
    /// Return whether `value` has this shape.
    fn accepts(self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
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
            Self::Bool => Ok(value.is_instance_of::<PyBool>()),
            Self::IntList => Ok(value
                .cast::<PyList>()
                .is_ok_and(|items| items.iter().all(|item| is_int(&item)))),
            Self::IntListList => Ok(value.cast::<PyList>().is_ok_and(|rows| {
                rows.iter().all(|row| {
                    row.cast::<PyList>()
                        .is_ok_and(|items| items.iter().all(|item| is_int(&item)))
                })
            })),
            Self::ObjectListList => Ok(value.cast::<PyList>().is_ok_and(|rows| {
                rows.iter().all(|row| {
                    row.cast::<PyList>()
                        .is_ok_and(|items| items.iter().all(|item| is_mapping(&item)))
                })
            })),
            Self::Object => Ok(is_mapping(value)),
            Self::OptionalObject => Ok(value.is_none() || is_mapping(value)),
            Self::List => Ok(value.is_instance_of::<PyList>()),
            Self::ObjectList => Ok(value
                .cast::<PyList>()
                .is_ok_and(|items| items.iter().all(|item| is_mapping(&item)))),
            Self::ObjectMap => Ok(value.cast::<PyMapping>().is_ok_and(|entries| {
                entries.values().is_ok_and(|values| {
                    values.try_iter().is_ok_and(|mut values| {
                        values.all(|item| item.is_ok_and(|item| is_mapping(&item)))
                    })
                })
            })),
            Self::Any => Ok(true),
        }
    }

    /// Return the type the structure error names for this shape.
    fn expected_type(self, py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        let dict_type = py.get_type::<PyDict>().into_any();
        let str_type = py.get_type::<PyString>().into_any();
        let int_type = py.get_type::<PyInt>().into_any();
        let list_type = py.get_type::<PyList>().into_any();
        match self {
            Self::Payload | Self::Object | Self::ObjectMap => Ok(dict_type),
            Self::Str => Ok(str_type),
            Self::Int => Ok(int_type),
            Self::OptionalPayload | Self::OptionalObject => dict_type.bitor(py.None()),
            Self::OptionalStr => str_type.bitor(py.None()),
            Self::OptionalInt => int_type.bitor(py.None()),
            Self::PayloadList
            | Self::List
            | Self::ObjectList
            | Self::IntList
            | Self::IntListList
            | Self::ObjectListList => Ok(list_type),
            Self::OptionalIntList => list_type.get_item(int_type)?.bitor(py.None()),
            Self::Literal => str_type
                .bitor(py.get_type::<PyFloat>())?
                .bitor(int_type)?
                .bitor(py.get_type::<PyBool>()),
            Self::Bool => Ok(py.get_type::<PyBool>().into_any()),
            Self::Any => Ok(py.get_type::<PyAny>().into_any()),
        }
    }
}

/// The fields a payload must hold, each of its shape, and whether it may
/// hold others.
///
/// An array of `(name, shape)` pairs is the exact form, as the derived
/// deserialization of a dataclass checks it, and converts into this type.
/// [`PayloadFields::allowing_extra`] is the form of a hand-written reader
/// that looks up the keys it needs and ignores the rest.
#[derive(Debug, Clone, Copy)]
pub struct PayloadFields<'a, const N: usize> {
    fields: [(&'a str, FieldShape); N],
    allows_extra: bool,
}

impl<'a, const N: usize> PayloadFields<'a, N> {
    /// Return the fields of a payload that holds exactly these.
    #[must_use]
    pub const fn exact(fields: [(&'a str, FieldShape); N]) -> Self {
        Self {
            fields,
            allows_extra: false,
        }
    }

    /// Return the fields of a payload that may hold other keys too.
    #[must_use]
    pub const fn allowing_extra(fields: [(&'a str, FieldShape); N]) -> Self {
        Self {
            fields,
            allows_extra: true,
        }
    }
}

impl<'a, const N: usize> From<[(&'a str, FieldShape); N]> for PayloadFields<'a, N> {
    fn from(fields: [(&'a str, FieldShape); N]) -> Self {
        Self::exact(fields)
    }
}

/// Check that `data` holds the fields `fields` names, each of its shape (and
/// no others, unless `fields` allows them), and return its values in the
/// order of `fields`.
///
/// `fields` is an array of `(name, shape)` pairs, which refuses extra keys,
/// or a [`PayloadFields`].
///
/// Matches the Python implementation: the structure check of the derived
/// `Serializable.deserialize_from_dict`, or of a hand-written reader.
///
/// # Errors
///
/// Raises `DeserializationDictStructureError(cls, <expected fields>, data)`
/// if `data` is no mapping, or a field is missing, extra (when extra ones
/// are refused) or of the wrong shape.
pub fn read_payload_fields<'py, 'a, const N: usize>(
    cls: &Bound<'py, PyType>,
    data: &Bound<'py, PyAny>,
    fields: impl Into<PayloadFields<'a, N>>,
) -> PyResult<[Bound<'py, PyAny>; N]> {
    let fields = fields.into();
    if let Some(values) = read_fields_of_shape(data, &fields)? {
        return Ok(values);
    }
    let py = cls.py();
    let expected = PyDict::new(py);
    for (name, shape) in fields.fields {
        expected.set_item(name, shape.expected_type(py)?)?;
    }
    Err(DESERIALIZATION_DICT_STRUCTURE_ERROR.err(py, (cls, expected, data)))
}

/// Return the values of `fields` in `data`, or `None` if `data` is not a
/// mapping holding those fields, each of its shape, and no others unless
/// `fields` allows them.
fn read_fields_of_shape<'py, const N: usize>(
    data: &Bound<'py, PyAny>,
    fields: &PayloadFields<'_, N>,
) -> PyResult<Option<[Bound<'py, PyAny>; N]>> {
    let Ok(mapping) = data.cast::<PyMapping>() else {
        return Ok(None);
    };
    if !fields.allows_extra && mapping.len()? != N {
        return Ok(None);
    }
    let mut values = Vec::with_capacity(N);
    for (name, shape) in &fields.fields {
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
/// order, as the constructor call `cls(**fields)` of a dataclass would
/// receive them.
///
/// The last `optional` names may be missing, and read as `None`; pass `0`
/// when every name is required.
///
/// # Errors
///
/// Raises `TypeError` if `fields` is not a mapping, lacks a required name,
/// or holds a name not in `names`, and what reading an entry raises.
pub fn read_constructor_fields<'py, const N: usize>(
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
    values
        .try_into()
        .map_err(|_values: Vec<_>| build_arity_error())
}

/// Return the error for a read that found a different number of values than
/// it has names, which no caller can cause.
fn build_arity_error() -> PyErr {
    PyRuntimeError::new_err("a payload read returned one value per field")
}

/// Return the values of `fields` unchanged: the decoding step of a class
/// whose payload holds no nested value.
///
/// # Errors
///
/// Never fails; the signature is the one every class's decoding step has.
pub fn keep_fields<'py, const N: usize>(
    _cls: &Bound<'py, PyType>,
    values: [Bound<'py, PyAny>; N],
) -> PyResult<[Bound<'py, PyAny>; N]> {
    Ok(values)
}

/// Return the nested value of the payload `data`, read by the class
/// `nested_class`: `nested_class.deserialize_from_dict(data)`.
///
/// # Errors
///
/// Raises what the nested class's reader raises.
pub fn read_nested_value<'py>(
    nested_class: &Bound<'py, PyType>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    nested_class.call_method1(intern!(nested_class.py(), "deserialize_from_dict"), (data,))
}

/// Return the list of the nested values of the payloads in the iterable
/// `items`, each read by `nested_class`.
///
/// # Errors
///
/// Raises `TypeError` if `items` is not iterable, and what a nested class's
/// reader raises.
pub fn read_nested_list<'py>(
    nested_class: &Bound<'py, PyType>,
    items: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let values = PyList::empty(nested_class.py());
    for item in items.try_iter()? {
        values.append(read_nested_value(nested_class, &item?)?)?;
    }
    Ok(values)
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
/// `DeserializationValueError` with its message, caused by it. Use
/// [`construct_from_decoded_fields_reporting_overflow`] for a class whose
/// construction raises `OverflowError` for a number out of range.
pub fn construct_from_decoded_fields<'py>(
    cls: &Bound<'py, PyType>,
    fields: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    construct_reporting(cls, fields, false)
}

/// Build an instance of `cls` from the decoded payload `fields` through
/// `cls.construct_from_fields(fields)`, reporting an integer out of range
/// as a refused value too.
///
/// It is [`construct_from_decoded_fields`] for a class that takes machine
/// integers, whose construction raises `OverflowError` for a number it
/// cannot hold.
///
/// # Errors
///
/// Raises what the construction raises, except that a `ValueError`,
/// `TypeError` or `OverflowError` outside the serialization hierarchy
/// becomes a `DeserializationValueError` with its message, caused by it.
pub fn construct_from_decoded_fields_reporting_overflow<'py>(
    cls: &Bound<'py, PyType>,
    fields: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    construct_reporting(cls, fields, true)
}

/// Build an instance of `cls` from `fields`, wrapping the refusals of the
/// construction, `OverflowError` too when `overflow`.
fn construct_reporting<'py>(
    cls: &Bound<'py, PyType>,
    fields: &Bound<'py, PyDict>,
    overflow: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let py = cls.py();
    match cls.call_method1(intern!(py, "construct_from_fields"), (fields,)) {
        Ok(instance) => Ok(instance),
        Err(error)
            if !error.is_instance(py, SERIALIZATION_ERROR.class(py)?)
                && (error.is_instance_of::<PyValueError>(py)
                    || error.is_instance_of::<PyTypeError>(py)
                    || (overflow && error.is_instance_of::<PyOverflowError>(py))) =>
        {
            let message = error.value(py).str()?;
            let wrapped = DESERIALIZATION_VALUE_ERROR.err(py, (message,));
            wrapped.set_cause(py, Some(error));
            Err(wrapped)
        }
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests;
