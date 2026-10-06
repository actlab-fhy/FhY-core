//! The V2 wire format of the Rust-backed classes: the core's serde shapes,
//! as canonical JSON text and as the dicts `json.loads` makes of it, and the
//! version the writers write in.
//!
//! A Rust-backed class writes V2 by serializing its core value with
//! `serde_json`, so its text is byte-identical to the core's, and reads V2
//! by parsing into the core's wire form and building the value with
//! [`PyResolver`], which turns each foreign part back into the Python
//! object its registered class decodes. The Python-defined parts of a
//! value give their foreign parts through
//! [`read_foreign`](crate::util::foreign::read_foreign). A Python exception
//! raised inside either hook is kept as the pending exception
//! ([`util::pending`](crate::util::pending)) and raised as itself when serde
//! returns.
//!
//! V1, the deprecated envelope format, is read and written by the classes'
//! own V1 code; [`is_writing_v1`] and [`is_v1_payload`] choose it.

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyString, PyType};
use serde::Serialize;
use serde::de::DeserializeOwned;

use fhy_core::constraint::{Constraint, CustomConstraint, OpaqueValue, Value};
use fhy_core::foreign::{BuildError, Foreign, ForeignError, Part, Resolve};
use fhy_core::param::{CustomDomain, ParamDomain};
use fhy_core::types::{DataType, DataTypeExtension, Type, TypeExtension};

use crate::constraint::{read_constraint, read_opaque_member};
use crate::util::exceptions::{
    DESERIALIZATION_VALUE_ERROR, MALFORMED_PAYLOAD_ERROR, SERIALIZATION_ERROR,
};
use crate::util::foreign::record_foreign_failure;
use crate::util::pending::with_pending_errors;

mod families;
mod python_value;
mod values;

pub(crate) use families::{
    decode_wire_family, decode_wire_family_json, encode_wire_dict, encode_wire_json,
};
pub(crate) use values::{deserialize_wire_value, serialize_wire_value};

/// The Python module of the serialization framework.
const MODULE: &str = "fhy_core.serialization";

/// Return whether the writers write V1 in the current context.
///
/// # Errors
///
/// Raises what reading the framework's context variable raises.
pub(crate) fn is_writing_v1(py: Python<'_>) -> PyResult<bool> {
    static V1: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let version = crate::util::python::cached_attr!(py, MODULE, "_WIRE_VERSION" => PyAny)?
        .call_method0(intern!(py, "get"))?;
    let v1 = V1.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
        Ok(py
            .import(MODULE)?
            .getattr("WireVersion")?
            .getattr("V1")?
            .unbind())
    })?;
    Ok(version.is(v1.bind(py)))
}

/// Return whether `data` is a V1 family payload: a dict holding an
/// envelope key, or any payload nested in a V1 payload being read, as the
/// framework's `_is_v1_envelope` decides.
pub(crate) fn is_v1_payload(data: &Bound<'_, PyAny>) -> bool {
    let py = data.py();
    if is_reading_v1(py) {
        return true;
    }
    let Ok(data) = data.cast::<PyDict>() else {
        return false;
    };
    data.contains(intern!(py, "__type__")).unwrap_or(false)
        || data.contains(intern!(py, "__data__")).unwrap_or(false)
}

/// Return whether a V1 payload is being read in this context.
pub(crate) fn is_reading_v1(py: Python<'_>) -> bool {
    crate::util::python::cached_attr!(py, MODULE, "_READING_V1" => PyAny)
        .and_then(|flag| flag.call_method0(intern!(py, "get")))
        .and_then(|value| value.is_truthy())
        .unwrap_or(false)
}

/// Warn that a reader of `cls` met a deprecated V1 payload.
///
/// # Errors
///
/// Raises the warning when warnings are errors.
pub(crate) fn warn_v1_read(cls: &Bound<'_, PyType>) -> PyResult<()> {
    crate::util::python::cached_attr!(cls.py(), MODULE, "_warn_v1_read" => PyAny)?.call1((cls,))?;
    Ok(())
}

/// Return the `SerializationError` of a foreign part that failed.
pub(crate) fn foreign_error(py: Python<'_>, error: &ForeignError) -> PyErr {
    SERIALIZATION_ERROR.err(py, (error.to_string(),))
}

/// Return the name of the class `cls`, or `?`.
fn class_name(cls: &Bound<'_, PyType>) -> String {
    cls.name()
        .map_or_else(|_| "?".to_owned(), |name| name.to_string())
}

/// Return the canonical V2 text of `value`.
///
/// # Errors
///
/// Raises the exception a Python-defined part's hook raised, and
/// `SerializationError` with serde's text otherwise.
pub(crate) fn to_json<T: Serialize + ?Sized>(py: Python<'_>, value: &T) -> PyResult<String> {
    with_pending_errors(|| {
        serde_json::to_string(value)
            .map_err(|error| SERIALIZATION_ERROR.err(py, (error.to_string(),)))
    })
}

/// Return the V2 dict of `value`: what `json.loads` makes of its canonical
/// text, built without the text.
///
/// # Errors
///
/// Raises what [`to_json`] raises.
pub(crate) fn to_dict<'py, T: Serialize + ?Sized>(
    py: Python<'py>,
    value: &T,
) -> PyResult<Bound<'py, PyAny>> {
    with_pending_errors(|| {
        python_value::to_python(py, value).map_err(|error| SERIALIZATION_ERROR.err(py, (error.0,)))
    })
}

/// Return the text of a JSON payload given as `str`, `bytes` or
/// `bytearray`.
///
/// # Errors
///
/// Raises `TypeError` for another type, and `MalformedPayloadError` for
/// bytes that are not UTF-8.
pub(crate) fn read_text(payload: &Bound<'_, PyAny>) -> PyResult<String> {
    let py = payload.py();
    if let Ok(text) = payload.cast::<PyString>() {
        return Ok(text.to_str()?.to_owned());
    }
    let bytes: Vec<u8> = if let Ok(bytes) = payload.cast::<PyBytes>() {
        bytes.as_bytes().to_vec()
    } else if let Ok(bytes) = payload.cast::<PyByteArray>() {
        bytes.to_vec()
    } else {
        return Err(PyTypeError::new_err(format!(
            "a JSON payload must be a str, bytes or bytearray, got {}",
            payload.get_type().name()?
        )));
    };
    String::from_utf8(bytes).map_err(|_invalid| {
        MALFORMED_PAYLOAD_ERROR.err(py, ("JSON payload is not valid UTF-8.".to_owned(),))
    })
}

/// Return the wire form `D` of the V2 text `text`, a payload of `cls`.
///
/// # Errors
///
/// Raises `MalformedPayloadError` for text that is not JSON, and
/// `DeserializationValueError` with serde's text for JSON of another shape.
pub(crate) fn parse<D: DeserializeOwned>(cls: &Bound<'_, PyType>, text: &str) -> PyResult<D> {
    let py = cls.py();
    serde_json::from_str(text).map_err(|error| {
        if error.is_syntax() || error.is_eof() {
            MALFORMED_PAYLOAD_ERROR.err(py, ("JSON payload is not valid JSON.".to_owned(),))
        } else {
            DESERIALIZATION_VALUE_ERROR.err(
                py,
                (format!(
                    "Invalid V2 payload for \"{}\": {error}",
                    class_name(cls)
                ),),
            )
        }
    })
}

/// Return the wire form `D` of the V2 dict `data`, a payload of `cls`.
///
/// The dict is read into a JSON value tree, the shape `json.loads` makes,
/// without writing its text.
///
/// # Errors
///
/// Raises `DeserializationValueError` for a payload that is not JSON-shaped
/// or not of the shape `D` reads.
pub(crate) fn parse_dict<D: DeserializeOwned>(
    cls: &Bound<'_, PyType>,
    data: &Bound<'_, PyAny>,
) -> PyResult<D> {
    let py = cls.py();
    let invalid = |reason: String| {
        DESERIALIZATION_VALUE_ERROR.err(
            py,
            (format!(
                "Invalid V2 payload for \"{}\": {reason}",
                class_name(cls)
            ),),
        )
    };
    let value = read_json_value(data, 0)?.map_err(invalid)?;
    serde_json::from_value(value).map_err(|error| invalid(error.to_string()))
}

/// The deepest nesting of `dict`s and `list`s the reader of a Python payload
/// accepts, `serde_json`'s own limit for JSON text. V2 payloads are shallow:
/// an expression is a flat node table.
const MAX_PAYLOAD_DEPTH: usize = 128;

/// Return the JSON value of the Python payload `object`, `depth` levels
/// inside the payload, or the reason it has none: a `dict` with `str` keys,
/// a `list` or `tuple`, a `str`, an `int` that fits 64 bits, a finite
/// `float`, a `bool` or `None`, nested at most [`MAX_PAYLOAD_DEPTH`] levels.
///
/// The limit bounds the recursion, and so the depth of the value built,
/// whose drop recurses as deep.
fn read_json_value(
    object: &Bound<'_, PyAny>,
    depth: usize,
) -> PyResult<Result<serde_json::Value, String>> {
    use serde_json::Value as Json;
    if object.is_none() {
        return Ok(Ok(Json::Null));
    }
    if let Ok(flag) = object.cast::<pyo3::types::PyBool>() {
        return Ok(Ok(Json::Bool(flag.is_true())));
    }
    if let Ok(text) = object.cast::<PyString>() {
        return Ok(Ok(Json::String(text.to_str()?.to_owned())));
    }
    let is_container =
        object.cast::<PyDict>().is_ok() || object.cast::<pyo3::types::PyList>().is_ok();
    if is_container && depth >= MAX_PAYLOAD_DEPTH {
        return Ok(Err(format!(
            "the payload nests more than {MAX_PAYLOAD_DEPTH} levels"
        )));
    }
    if let Ok(dict) = object.cast::<PyDict>() {
        let mut map = serde_json::Map::with_capacity(dict.len());
        for (key, item) in dict.iter() {
            let Ok(key) = key.cast::<PyString>() else {
                return Ok(Err(format!("a key {} is not a str", key.repr()?)));
            };
            match read_json_value(&item, depth + 1)? {
                Ok(value) => {
                    map.insert(key.to_str()?.to_owned(), value);
                }
                Err(reason) => return Ok(Err(reason)),
            }
        }
        return Ok(Ok(Json::Object(map)));
    }
    if let Ok(list) = object.cast::<pyo3::types::PyList>() {
        let mut items = Vec::with_capacity(list.len());
        for item in list.iter() {
            match read_json_value(&item, depth + 1)? {
                Ok(value) => items.push(value),
                Err(reason) => return Ok(Err(reason)),
            }
        }
        return Ok(Ok(Json::Array(items)));
    }
    if let Ok(integer) = object.cast::<pyo3::types::PyInt>() {
        if let Ok(value) = integer.extract::<u64>() {
            return Ok(Ok(Json::from(value)));
        }
        if let Ok(value) = integer.extract::<i64>() {
            return Ok(Ok(Json::from(value)));
        }
        return Ok(Err(format!(
            "the integer {} does not fit 64 bits",
            integer.repr()?
        )));
    }
    if let Ok(float) = object.cast::<pyo3::types::PyFloat>() {
        return Ok(serde_json::Number::from_f64(float.value())
            .map(Json::Number)
            .ok_or_else(|| format!("the float {} is not finite", float.value())));
    }
    Ok(Err(format!(
        "a value of type {} is not JSON",
        object.get_type().name()?
    )))
}

/// Return the value `build` builds, raising a Python-defined part's
/// exception as itself and any other failure as `DeserializationValueError`
/// with its text, naming `cls`.
///
/// # Errors
///
/// Raises as described.
pub(crate) fn build<T>(
    cls: &Bound<'_, PyType>,
    build: impl FnOnce() -> Result<T, BuildError>,
) -> PyResult<T> {
    let py = cls.py();
    with_pending_errors(|| {
        build().map_err(|error| {
            DESERIALIZATION_VALUE_ERROR.err(
                py,
                (format!(
                    "Invalid V2 payload for \"{}\": {error}",
                    class_name(cls)
                ),),
            )
        })
    })
}

/// The resolver of the foreign parts a Python payload holds: each part's
/// type id is looked up in the framework's registry, its class decodes its
/// data, and the object becomes the part through the binding's adapter.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PyResolver;

/// Return the Python object the foreign part `foreign` encodes, a family
/// member when `family`.
fn resolve_object<'py>(
    py: Python<'py>,
    foreign: &Foreign,
    family: bool,
) -> Result<Bound<'py, PyAny>, ForeignError> {
    let result = (|| -> PyResult<Bound<'py, PyAny>> {
        let keywords = PyDict::new(py);
        keywords.set_item(intern!(py, "family"), family)?;
        crate::util::python::cached_attr!(py, MODULE, "_resolve_foreign" => PyAny)?
            .call((foreign.type_id(), foreign.data()), Some(&keywords))
    })();
    result.map_err(|error| record_foreign_failure(py, foreign.type_id(), error))
}

/// Return the error of a resolved object of the wrong kind for its place.
fn wrong_kind(py: Python<'_>, foreign: &Foreign, expected: &str) -> ForeignError {
    record_foreign_failure(
        py,
        foreign.type_id(),
        PyTypeError::new_err(format!(
            "the foreign part \"{}\" is not a {expected}",
            foreign.type_id()
        )),
    )
}

impl Resolve<Part<dyn OpaqueValue>> for PyResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn OpaqueValue>, ForeignError> {
        Python::attach(|py| {
            let object = resolve_object(py, foreign, false)?;
            match read_opaque_member(&object) {
                Ok(Value::Opaque(opaque)) => Ok(opaque),
                Ok(_) => Err(wrong_kind(py, foreign, "Serializable value")),
                Err(error) => Err(record_foreign_failure(py, foreign.type_id(), error)),
            }
        })
    }
}

impl Resolve<Part<dyn CustomConstraint>> for PyResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn CustomConstraint>, ForeignError> {
        Python::attach(|py| {
            let object = resolve_object(py, foreign, true)?;
            match read_constraint(&object) {
                Ok(Constraint::Custom(custom)) => Ok(custom),
                Ok(_) => Err(wrong_kind(py, foreign, "Python-defined Constraint")),
                Err(error) => Err(record_foreign_failure(py, foreign.type_id(), error)),
            }
        })
    }
}

impl Resolve<Part<dyn CustomDomain>> for PyResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn CustomDomain>, ForeignError> {
        Python::attach(|py| {
            let object = resolve_object(py, foreign, true)?;
            match crate::param::read_domain_object(&object) {
                Ok(ParamDomain::Custom(custom)) => Ok(custom),
                Ok(_) => Err(wrong_kind(py, foreign, "Python-defined ParamDomain")),
                Err(error) => Err(record_foreign_failure(py, foreign.type_id(), error)),
            }
        })
    }
}

impl Resolve<Part<dyn TypeExtension>> for PyResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn TypeExtension>, ForeignError> {
        Python::attach(|py| {
            let object = resolve_object(py, foreign, true)?;
            match crate::types::read_type_value(&object) {
                Some(Type::Extension(extension)) => Ok(extension),
                _ => Err(wrong_kind(py, foreign, "Python-defined Type")),
            }
        })
    }
}

impl Resolve<Part<dyn DataTypeExtension>> for PyResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn DataTypeExtension>, ForeignError> {
        Python::attach(|py| {
            let object = resolve_object(py, foreign, true)?;
            match crate::types::read_data_type_value(&object) {
                Some(DataType::Extension(extension)) => Ok(extension),
                _ => Err(wrong_kind(py, foreign, "Python-defined DataType")),
            }
        })
    }
}

/// Return the Python object of a resolved Python-defined frame, the object
/// the foreign part `foreign` encodes.
///
/// # Errors
///
/// Raises what decoding the frame raises.
pub(crate) fn resolve_frame<'py>(
    py: Python<'py>,
    foreign: &Foreign,
) -> PyResult<Bound<'py, PyAny>> {
    with_pending_errors(|| {
        resolve_object(py, foreign, true)
            .map_err(|error| SERIALIZATION_ERROR.err(py, (error.to_string(),)))
    })
}

/// Return `Serializable.<name>`, the framework's own method, to call with
/// an explicit receiver.
fn base_method<'py>(
    py: Python<'py>,
    cell: &'static PyOnceLock<Py<PyAny>>,
    name: &str,
) -> PyResult<&'py Bound<'py, PyAny>> {
    cell.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
        let method = py
            .import(MODULE)?
            .getattr("Serializable")?
            .getattr("__dict__")?
            .get_item(name)?;
        match method.getattr("__func__") {
            Ok(function) => Ok(function.unbind()),
            Err(_) => Ok(method.unbind()),
        }
    })
    .map(|method| method.bind(py))
}

/// Return the V1 envelope of the family member `object`.
///
/// # Errors
///
/// Raises what the member's V1 data hook raises.
pub(crate) fn write_v1_envelope<'py>(object: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    crate::util::python::cached_attr!(object.py(), MODULE, "_write_v1_envelope" => PyAny)?
        .call1((object,))
}

/// Return what `read` reads from a V1 payload of `cls`, warning that V1 is
/// deprecated unless the payload is nested in a V1 payload being read.
///
/// # Errors
///
/// Raises what `read` raises, and the warning when warnings are errors.
pub(crate) fn reading_v1<T>(
    cls: &Bound<'_, PyType>,
    read: impl FnOnce() -> PyResult<T>,
) -> PyResult<T> {
    let py = cls.py();
    let flag = crate::util::python::cached_attr!(py, MODULE, "_READING_V1" => PyAny)?;
    if flag.call_method0(intern!(py, "get"))?.is_truthy()? {
        return read();
    }
    warn_v1_read(cls)?;
    let token = flag.call_method1(intern!(py, "set"), (true,))?;
    let result = read();
    flag.call_method1(intern!(py, "reset"), (token,))?;
    result
}

/// Return the member of the family `cls` the V1 envelope `data` encodes,
/// warning that V1 is deprecated.
///
/// # Errors
///
/// Raises what the framework's V1 reader raises.
pub(crate) fn read_v1_envelope<'py>(
    cls: &Bound<'py, PyType>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    reading_v1(cls, || {
        cls.call_method1(intern!(cls.py(), "_deserialize_v1_envelope"), (data,))
    })
}

/// Return whether `indent` and `sort_keys` ask for the canonical text.
fn asks_for_canonical_text(
    indent: Option<&Bound<'_, PyAny>>,
    sort_keys: Option<&Bound<'_, PyAny>>,
) -> PyResult<bool> {
    let is_indented = indent.is_some_and(|indent| !indent.is_none());
    let is_sorted = match sort_keys {
        Some(sort_keys) if !sort_keys.is_none() => sort_keys.is_truthy()?,
        _ => false,
    };
    Ok(!is_indented && !is_sorted)
}

/// Return the payload dict of `object`: its V2 dict, the one `value`
/// serializes, or under V1 the one `v1` returns.
///
/// # Errors
///
/// Raises what the chosen writer raises.
pub(crate) fn write_dict<'py, T: Serialize>(
    object: &Bound<'py, PyAny>,
    v1: impl FnOnce() -> PyResult<Bound<'py, PyAny>>,
    value: impl FnOnce() -> PyResult<T>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = object.py();
    if is_writing_v1(py)? {
        return v1();
    }
    to_dict(py, &value()?)
}

/// Return the JSON text of `object`: the canonical text of the value
/// `value` serializes, or what the framework's `to_json` writes under V1
/// or for `indent` or `sort_keys`.
///
/// # Errors
///
/// Raises what the chosen writer raises.
pub(crate) fn write_json<T: Serialize>(
    object: &Bound<'_, PyAny>,
    indent: Option<&Bound<'_, PyAny>>,
    sort_keys: Option<&Bound<'_, PyAny>>,
    value: impl FnOnce() -> PyResult<T>,
) -> PyResult<String> {
    static TO_JSON: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let py = object.py();
    if !is_writing_v1(py)? && asks_for_canonical_text(indent, sort_keys)? {
        return to_json(py, &value()?);
    }
    let keywords = PyDict::new(py);
    keywords.set_item(intern!(py, "indent"), indent)?;
    keywords.set_item(intern!(py, "sort_keys"), sort_keys)?;
    base_method(py, &TO_JSON, "to_json")?
        .call((object,), Some(&keywords))?
        .extract()
}

/// Return the object of the JSON payload `payload`, a `str`, `bytes` or
/// `bytearray`, of `cls`: through `decode` for a V2 text, and through the
/// framework's `from_json` for a text that may hold a V1 envelope.
///
/// # Errors
///
/// Raises what the chosen reader raises.
pub(crate) fn read_json<'py>(
    cls: &Bound<'py, PyType>,
    payload: &Bound<'py, PyAny>,
    decode: impl FnOnce(&str) -> PyResult<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    static FROM_JSON: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let py = cls.py();
    let text = read_text(payload)?;
    if text.contains("\"__type__\"") || text.contains("\"__data__\"") {
        return base_method(py, &FROM_JSON, "from_json")?.call1((cls, payload));
    }
    decode(&text)
}

/// Raise `TypeError` unless `object`, which a payload of `cls` decoded to,
/// is an instance of `cls`.
///
/// # Errors
///
/// Raises the framework's `SerializationError` as its family check does.
pub(crate) fn check_instance<'py>(
    cls: &Bound<'py, PyType>,
    object: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if object.is_instance(cls)? {
        return Ok(object);
    }
    Err(SERIALIZATION_ERROR.err(
        cls.py(),
        (format!(
            "Wrapped type \"{}\" is not a subclass of expected family \"{}\".",
            object.get_type().str()?,
            cls.str()?
        ),),
    ))
}
