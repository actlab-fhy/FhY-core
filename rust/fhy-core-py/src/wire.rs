//! The V2 wire format of the Rust-backed classes (slice S17 of
//! `docs/design/python-switch.md`): the core's serde shapes, as canonical
//! JSON text and as the dicts `json.loads` makes of it, and the version the
//! writers write in.
//!
//! A Rust-backed class writes V2 by serializing its core value with
//! `serde_json`, so its text is byte-identical to the core's, and reads V2
//! by parsing into the core's wire form and building the value with
//! [`PyResolver`], which turns each foreign part back into the Python
//! object its registered class decodes. The Python-defined parts of a
//! value give their foreign parts through [`foreign_of`]. A Python exception
//! raised inside either hook is kept in the constraint binding's
//! pending-error slot and raised as itself when serde returns.
//!
//! V1, the deprecated envelope format, is read and written by the classes'
//! own V1 code; [`is_writing_v1`] and [`is_v1_payload`] choose it.

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyString, PyType};
use serde::Serialize;
use serde::de::DeserializeOwned;

use fhy_core::constraint::{Constraint, CustomConstraint, Opaque, Value};
use fhy_core::foreign::{BuildError, Foreign, ForeignError, Resolve};
use fhy_core::param::{CustomDomain, ParamDomain};
use fhy_core::types::{DataType, DataTypeExtension, Type, TypeExtension};

use crate::constraint::{
    read_constraint, read_opaque_member, record_pending_error, with_pending_errors,
};

mod families;
mod values;

pub(crate) use families::{
    decode_wire_family, decode_wire_family_json, encode_wire_dict, encode_wire_json,
};
pub(crate) use values::{deserialize_wire_value, serialize_wire_value};

/// The Python module of the serialization framework.
const MODULE: &str = "fhy_core.serialization";

/// Return the attribute `name` of the framework's module, imported once into
/// `cell`.
fn framework<'py>(
    py: Python<'py>,
    cell: &'static PyOnceLock<Py<PyAny>>,
    name: &str,
) -> PyResult<&'py Bound<'py, PyAny>> {
    cell.import(py, MODULE, name)
}

/// Return whether the writers write V1 in the current context.
///
/// # Errors
///
/// Raises what reading the framework's context variable raises.
pub(crate) fn is_writing_v1(py: Python<'_>) -> PyResult<bool> {
    static VERSION: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    static V1: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let version = framework(py, &VERSION, "_WIRE_VERSION")?.call_method0(intern!(py, "get"))?;
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
    static READING: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    framework(py, &READING, "_READING_V1")
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
    static WARN: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    framework(cls.py(), &WARN, "_warn_v1_read")?.call1((cls,))?;
    Ok(())
}

/// Return `fhy_core.serialization.<name>`, an exception class.
fn error_class<'py>(
    py: Python<'py>,
    cell: &'static PyOnceLock<Py<PyAny>>,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    framework(py, cell, name).cloned()
}

/// Return the `DeserializationValueError` with `message`.
fn deserialization_error(py: Python<'_>, message: String) -> PyErr {
    static CLASS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    match error_class(py, &CLASS, "DeserializationValueError")
        .and_then(|class| class.call1((message,)))
    {
        Ok(error) => PyErr::from_value(error),
        Err(error) => error,
    }
}

/// Return the `MalformedPayloadError` with `message`.
fn malformed_error(py: Python<'_>, message: String) -> PyErr {
    static CLASS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    match error_class(py, &CLASS, "MalformedPayloadError").and_then(|class| class.call1((message,)))
    {
        Ok(error) => PyErr::from_value(error),
        Err(error) => error,
    }
}

/// Return the `SerializationError` with `message`.
fn serialization_error(py: Python<'_>, message: String) -> PyErr {
    static CLASS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    match error_class(py, &CLASS, "SerializationError").and_then(|class| class.call1((message,))) {
        Ok(error) => PyErr::from_value(error),
        Err(error) => error,
    }
}

/// Return the `SerializationError` of a foreign part that failed.
pub(crate) fn foreign_error(py: Python<'_>, error: &ForeignError) -> PyErr {
    serialization_error(py, error.to_string())
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
        serde_json::to_string(value).map_err(|error| serialization_error(py, error.to_string()))
    })
}

/// Return the V2 dict of `value`: `json.loads` of its canonical text.
///
/// # Errors
///
/// Raises what [`to_json`] raises.
pub(crate) fn to_dict<'py, T: Serialize + ?Sized>(
    py: Python<'py>,
    value: &T,
) -> PyResult<Bound<'py, PyAny>> {
    let text = to_json(py, value)?;
    loads(py, &text)
}

/// Return `json.loads(text)`.
///
/// # Errors
///
/// Raises what `json.loads` raises.
pub(crate) fn loads<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyAny>> {
    static LOADS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    LOADS.import(py, "json", "loads")?.call1((text,))
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
    String::from_utf8(bytes)
        .map_err(|_invalid| malformed_error(py, "JSON payload is not valid UTF-8.".to_owned()))
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
            malformed_error(py, "JSON payload is not valid JSON.".to_owned())
        } else {
            deserialization_error(
                py,
                format!("Invalid V2 payload for \"{}\": {error}", class_name(cls)),
            )
        }
    })
}

/// Return the wire form `D` of the V2 dict `data`, a payload of `cls`.
///
/// # Errors
///
/// Raises `DeserializationValueError` for a payload that is not JSON-shaped
/// or not of the shape `D` reads.
pub(crate) fn parse_dict<D: DeserializeOwned>(
    cls: &Bound<'_, PyType>,
    data: &Bound<'_, PyAny>,
) -> PyResult<D> {
    static DUMPS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let py = cls.py();
    let text = framework(py, &DUMPS, "_dump_canonical_json")?
        .call1((data,))
        .map_err(|error| {
            let message = format!(
                "Invalid V2 payload for \"{}\": {}",
                class_name(cls),
                error.value(py)
            );
            let wrapped = deserialization_error(py, message);
            wrapped.set_cause(py, Some(error));
            wrapped
        })?;
    let text = text.cast::<PyString>()?.to_str()?;
    serde_json::from_str(text).map_err(|error| {
        deserialization_error(
            py,
            format!("Invalid V2 payload for \"{}\": {error}", class_name(cls)),
        )
    })
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
            deserialization_error(
                py,
                format!("Invalid V2 payload for \"{}\": {error}", class_name(cls)),
            )
        })
    })
}

/// A message of a Python exception, the source of a failed foreign part.
#[derive(Debug)]
struct RaisedError(String);

impl fmt::Display for RaisedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for RaisedError {}

/// Keep `error` as the pending exception, and return the foreign error of
/// the part `type_id` it stands for.
fn failed(py: Python<'_>, type_id: &str, error: PyErr) -> ForeignError {
    let message = error.value(py).to_string();
    record_pending_error(error);
    ForeignError::Failed {
        type_id: type_id.to_owned(),
        source: Box::new(RaisedError(message)),
    }
}

/// Return the foreign part of the Python-defined part `object`: its type
/// id and the canonical text of its data, as a family member when
/// `family`, and its whole payload otherwise.
///
/// # Errors
///
/// Returns [`ForeignError::Failed`], keeping the Python exception pending,
/// when the object's hooks raise.
pub(crate) fn foreign_of(object: &Py<PyAny>, family: bool) -> Result<Foreign, ForeignError> {
    static PAYLOAD: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    Python::attach(|py| {
        let object = object.bind(py);
        let result = (|| -> PyResult<(String, String)> {
            let keywords = PyDict::new(py);
            keywords.set_item(intern!(py, "family"), family)?;
            framework(py, &PAYLOAD, "_foreign_payload")?
                .call((object,), Some(&keywords))?
                .extract()
        })();
        match result {
            Ok((type_id, data)) => Ok(Foreign::new(type_id, data)),
            Err(error) => {
                let name = object
                    .get_type()
                    .name()
                    .map_or_else(|_| "?".to_owned(), |name| name.to_string());
                Err(failed(py, &name, error))
            }
        }
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
    static RESOLVE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let result = (|| -> PyResult<Bound<'py, PyAny>> {
        let keywords = PyDict::new(py);
        keywords.set_item(intern!(py, "family"), family)?;
        framework(py, &RESOLVE, "_resolve_foreign")?
            .call((foreign.type_id(), foreign.data()), Some(&keywords))
    })();
    result.map_err(|error| failed(py, foreign.type_id(), error))
}

/// Return the error of a resolved object of the wrong kind for its place.
fn wrong_kind(py: Python<'_>, foreign: &Foreign, expected: &str) -> ForeignError {
    failed(
        py,
        foreign.type_id(),
        PyTypeError::new_err(format!(
            "the foreign part \"{}\" is not a {expected}",
            foreign.type_id()
        )),
    )
}

impl Resolve<Opaque> for PyResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Opaque, ForeignError> {
        Python::attach(|py| {
            let object = resolve_object(py, foreign, false)?;
            match read_opaque_member(&object) {
                Ok(Value::Opaque(opaque)) => Ok(opaque),
                Ok(_) => Err(wrong_kind(py, foreign, "Serializable value")),
                Err(error) => Err(failed(py, foreign.type_id(), error)),
            }
        })
    }
}

impl Resolve<Arc<dyn CustomConstraint>> for PyResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Arc<dyn CustomConstraint>, ForeignError> {
        Python::attach(|py| {
            let object = resolve_object(py, foreign, true)?;
            match read_constraint(&object) {
                Ok(Constraint::Custom(custom)) => Ok(custom),
                Ok(_) => Err(wrong_kind(py, foreign, "Python-defined Constraint")),
                Err(error) => Err(failed(py, foreign.type_id(), error)),
            }
        })
    }
}

impl Resolve<Arc<dyn CustomDomain>> for PyResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Arc<dyn CustomDomain>, ForeignError> {
        Python::attach(|py| {
            let object = resolve_object(py, foreign, true)?;
            match crate::param::read_domain_object(&object) {
                Ok(ParamDomain::Custom(custom)) => Ok(custom),
                Ok(_) => Err(wrong_kind(py, foreign, "Python-defined ParamDomain")),
                Err(error) => Err(failed(py, foreign.type_id(), error)),
            }
        })
    }
}

impl Resolve<Arc<dyn TypeExtension>> for PyResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Arc<dyn TypeExtension>, ForeignError> {
        Python::attach(|py| {
            let object = resolve_object(py, foreign, true)?;
            match crate::types::read_type_value(&object) {
                Some(Type::Extension(extension)) => Ok(extension),
                _ => Err(wrong_kind(py, foreign, "Python-defined Type")),
            }
        })
    }
}

impl Resolve<Arc<dyn DataTypeExtension>> for PyResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Arc<dyn DataTypeExtension>, ForeignError> {
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
            .map_err(|error| serialization_error(py, error.to_string()))
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
    static WRITE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    framework(object.py(), &WRITE, "_write_v1_envelope")?.call1((object,))
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
    static READING: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let py = cls.py();
    let flag = framework(py, &READING, "_READING_V1")?;
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
    Err(serialization_error(
        cls.py(),
        format!(
            "Wrapped type \"{}\" is not a subclass of expected family \"{}\".",
            object.get_type().str()?,
            cls.str()?
        ),
    ))
}
