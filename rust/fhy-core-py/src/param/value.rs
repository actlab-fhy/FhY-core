//! Python values as core param values: the values of a finite domain, read
//! strictly with Python's kind predicates, and candidate values, read
//! leniently.

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    PyBool, PyByteArray, PyBytes, PyFloat, PyFrozenSet, PyInt, PyString, PyTuple, PyType,
};

use fhy_core::constraint::Value;
use fhy_core::constraint::wire::MAX_VALUE_DEPTH;
use fhy_core::param::DomainKind;

use crate::constraint::{read_bound_value, read_identifier, read_opaque_member};
use crate::expression::read_big_int;

/// Return whether `value` is a `Serializable`.
fn is_serializable(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    value.is_instance(crate::util::python::cached_attr!(value.py(), "fhy_core.serialization", "Serializable" => PyType)?)
}

/// Return whether `value` defines usable equality: through the `Equal`
/// trait, or an `__eq__` of its own class and a `__hash__`.
fn supports_equal_value_semantics(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    static OBJECT_EQ: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let py = value.py();
    if value
        .is_instance(crate::util::python::cached_attr!(py, "fhy_core.traits", "Equal" => PyType)?)?
    {
        return value.getattr(intern!(py, "supports_equality"))?.is_truthy();
    }
    let object_eq = OBJECT_EQ.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
        Ok(py
            .import(intern!(py, "builtins"))?
            .getattr(intern!(py, "object"))?
            .getattr(intern!(py, "__eq__"))?
            .unbind())
    })?;
    let class = value.get_type();
    let eq = class.getattr(intern!(py, "__eq__"))?;
    let hash = class.getattr(intern!(py, "__hash__"))?;
    Ok(!eq.is(object_eq.bind(py)) && !hash.is_none())
}

/// Return whether `value` defines a usable total order: through the
/// `Orderable` trait, or an `__lt__` defined in its class's MRO below
/// `object`.
fn supports_orderable_value_semantics(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = value.py();
    if value.is_instance(
        crate::util::python::cached_attr!(py, "fhy_core.traits", "Orderable" => PyType)?,
    )? {
        return value.getattr(intern!(py, "supports_ordering"))?.is_truthy();
    }
    let mro = value.get_type().mro();
    let count = mro.len();
    for class in mro.iter().take(count.saturating_sub(1)) {
        if class
            .getattr(intern!(py, "__dict__"))?
            .contains(intern!(py, "__lt__"))?
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Return the text of the `TypeError` a value of the wrong kind raises in a
/// finite domain of `kind`.
pub(super) const fn value_kind_message(kind: DomainKind) -> &'static str {
    match kind {
        DomainKind::Ordinal => {
            "Ordinal values must satisfy orderable semantics and be serializable, or be primitive \
             bool/int/float/str values."
        }
        DomainKind::Categorical => {
            "Categorical values must satisfy equal semantics and be serializable, or be primitive \
             bool/int/str values."
        }
        _ => {
            "Permutation members must satisfy equal semantics and be serializable, or be \
             primitive bool/int/float/str values."
        }
    }
}

/// Return the core value of one value of a finite domain of `kind`, or
/// `None` if it is no value of that kind.
///
/// A `bool`, `int` or `str` is one, and a `float` unless the kind is
/// categorical; a number of a subclass is read as the exact number it
/// denotes. An `Identifier` is one, read as the core's identifier, unless
/// the kind is ordinal, since identifiers do not order. Another
/// `Serializable` is one when its class supports the ordering
/// (ordinal) or the equality (categorical, permutation) the kind needs, and
/// is read as an opaque value Python compares. In a categorical domain a
/// `tuple` or a `frozenset` of such values is one too, nested at most as
/// deep as a value may be on the wire.
fn read_finite_value(kind: DomainKind, value: &Bound<'_, PyAny>) -> PyResult<Option<Value>> {
    read_finite_value_at(kind, value, MAX_VALUE_DEPTH)
}

/// Return the core value of one value of a finite domain of `kind`, with
/// `remaining` more tuples or frozen sets allowed around its innermost
/// value, or `None` if it is no value of that kind.
fn read_finite_value_at(
    kind: DomainKind,
    value: &Bound<'_, PyAny>,
    remaining: usize,
) -> PyResult<Option<Value>> {
    if let Ok(boolean) = value.cast::<PyBool>() {
        return Ok(Some(Value::Bool(boolean.is_true())));
    }
    if value.is_instance_of::<PyInt>() {
        return Ok(Some(Value::Int(read_big_int(value)?)));
    }
    if let Ok(float) = value.cast::<PyFloat>() {
        return Ok((kind != DomainKind::Categorical).then(|| Value::Float(float.value())));
    }
    if let Ok(text) = value.cast::<PyString>() {
        return Ok(Some(Value::Str(text.to_str()?.to_owned())));
    }
    if let Some(identifier) = read_identifier(value)? {
        return Ok((kind != DomainKind::Ordinal).then_some(Value::Identifier(identifier)));
    }
    let is_tuple = value.is_instance_of::<PyTuple>();
    if kind == DomainKind::Categorical && (is_tuple || value.is_instance_of::<PyFrozenSet>()) {
        let Some(remaining) = remaining.checked_sub(1) else {
            return Ok(None);
        };
        let mut elements = Vec::new();
        for element in value.try_iter()? {
            match read_finite_value_at(kind, &element?, remaining)? {
                Some(read) => elements.push(read),
                None => return Ok(None),
            }
        }
        return Ok(Some(if is_tuple {
            Value::Tuple(elements)
        } else {
            Value::FrozenSet(elements)
        }));
    }
    if !is_serializable(value)? {
        return Ok(None);
    }
    let is_capable = match kind {
        DomainKind::Ordinal => supports_orderable_value_semantics(value)?,
        _ => supports_equal_value_semantics(value)?,
    };
    if !is_capable {
        return Ok(None);
    }
    read_opaque_member(value).map(Some)
}

/// Return the core values of `values`, the values of a finite domain of
/// `kind`, and their Python objects, in order.
///
/// # Errors
///
/// Raises `TypeError`, with the Python implementation's text, at the first
/// value that is no value of the kind.
pub(super) fn read_finite_values(
    kind: DomainKind,
    values: &Bound<'_, PyAny>,
) -> PyResult<Vec<Value>> {
    let mut read = Vec::new();
    for value in values.try_iter()? {
        let value = value?;
        match read_finite_value(kind, &value)? {
            Some(core) => read.push(core),
            None => return Err(PyTypeError::new_err(value_kind_message(kind))),
        }
    }
    Ok(read)
}

/// Return whether `value` is a sequence that is no string, as a
/// permutation value must be.
fn is_permutation_sequence(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    if value.is_instance_of::<PyString>()
        || value.is_instance_of::<PyBytes>()
        || value.is_instance_of::<PyByteArray>()
    {
        return Ok(false);
    }
    value.is_instance(
        crate::util::python::cached_attr!(value.py(), "collections.abc", "Sequence" => PyType)?,
    )
}

/// Return the core value of a candidate value, read leniently: a sequence
/// that is no string as a tuple of its elements, since a permutation domain
/// admits any such sequence, and anything else as a bound value.
///
/// # Errors
///
/// Raises what reading an element raises.
pub(super) fn read_candidate(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    if is_permutation_sequence(value)? {
        let elements = value
            .try_iter()?
            .map(|element| read_bound_value(&element?))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(Value::Tuple(elements));
    }
    read_bound_value(value)
}

/// Return `value` normalized for a permutation domain: the tuple of a
/// sequence's elements.
///
/// # Errors
///
/// Raises what `tuple(value)` raises.
pub(super) fn to_tuple<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = value.py();
    py.import(intern!(py, "builtins"))?
        .getattr(intern!(py, "tuple"))?
        .call1((value,))
}
