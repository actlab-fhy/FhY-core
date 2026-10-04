//! Python values as core constraint values, and core members as Python
//! values.
//!
//! A member is read strictly: the value must be a `str`, `int`, `float` or
//! `bool`, a hashable `tuple` or `frozenset` of members, or a `Serializable`
//! that is also `Hashable`, and anything else raises `ConstraintError` with
//! the Python implementation's text. A number whose type subclasses `int`
//! or `float` becomes the exact number it denotes. A bound value is read
//! leniently: every value becomes a core value, and one the core cannot
//! hold, such as `None`, a list or an object, becomes an opaque value that
//! is not member-shaped, so only the constraint that reads it judges it.
//!
//! A `Serializable` member is an opaque value: [`PyOpaqueValue`]
//! compares it with Python's `==`, after `type(a) is type(b)`. A comparison
//! that raises answers `false` and keeps its exception in a per-thread
//! slot, which the entry point that started the comparison raises when the
//! core returns. Once an exception is kept,
//! no comparison calls Python again during that call, and an exception that
//! is not an `Exception`, such as `KeyboardInterrupt`, replaces a kept one
//! that is.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt;
use std::sync::OnceLock;

use pyo3::exceptions::{PyRecursionError, PyTypeError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyFrozenSet, PyInt, PyString, PyTuple, PyType};

use fhy_core::constraint::{Member, MemberKind, OpaqueValue, Value};
use fhy_core::foreign::{BoxError, ForeignPart, Part};

use crate::expression::{big_int_to_python, decimal_class, read_big_int, read_decimal};
use crate::kit::gc::Slot;
use crate::kit::pending::{has_pending_error, record_pending_error};
pub(crate) use crate::kit::python::type_name;

/// Return the `ConstraintError` with `message`.
pub(crate) fn constraint_error(py: Python<'_>, message: impl Into<String>) -> PyErr {
    crate::kit::exceptions::CONSTRAINT_ERROR.err(py, (message.into(),))
}

/// Return `fhy_core.serialization.Serializable`.
fn serializable_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    crate::kit::python::cached_attr!(py, "fhy_core.serialization", "Serializable" => PyType)
}

/// Return `collections.abc.Hashable`.
fn hashable_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    crate::kit::python::cached_attr!(py, "collections.abc", "Hashable" => PyType)
}

/// Return whether `value` is both `Serializable` and `Hashable`.
fn is_serializable_hashable(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = value.py();
    Ok(value.is_instance(serializable_class(py)?)? && value.is_instance(hashable_class(py)?)?)
}

/// Return `repr(value)`, or `?` if it raises.
pub(crate) fn repr_text(value: &Bound<'_, PyAny>) -> String {
    value
        .repr()
        .map_or_else(|_| "?".to_owned(), |text| text.to_string())
}

/// Return `str(value)`, or `?` if it raises.
fn str_text(value: &Bound<'_, PyAny>) -> String {
    value
        .str()
        .map_or_else(|_| "?".to_owned(), |text| text.to_string())
}

/// Return `str(type(value))`, such as `<class 'list'>`.
fn type_text(value: &Bound<'_, PyAny>) -> String {
    str_text(value.get_type().as_any())
}

/// A Python object as a core opaque value.
///
/// The object and its class are kept in [`Slot`]s, which the object whose construction
/// made the adapter owns and traverses.
pub(crate) struct PyOpaqueValue {
    object: Slot,
    class: Slot,
    type_name: String,
    is_member_shaped: bool,
    /// The ordering key, computed when a member is read, or on first use.
    key: OnceLock<String>,
}

impl PyOpaqueValue {
    /// Return the opaque value of `object`, member-shaped as told, with its
    /// ordering key when it is known.
    fn new(object: &Bound<'_, PyAny>, is_member_shaped: bool, key: Option<String>) -> Self {
        Self {
            object: Slot::new(object.clone().unbind()),
            class: Slot::new(object.get_type().into_any().unbind()),
            type_name: type_name(object),
            is_member_shaped,
            key: key.map_or_else(OnceLock::new, OnceLock::from),
        }
    }

    /// Return the Python object.
    pub(crate) fn object<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.object.get(py)
    }
}

impl fmt::Debug for PyOpaqueValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PyOpaqueValue")
            .field("type_name", &self.type_name)
            .finish_non_exhaustive()
    }
}

/// Return the ordering key of the member `value`: its type's qualified name
/// and the `repr` of its payload for a `Serializable` member, or of the
/// value itself otherwise; the payload is its V2 one whatever version is
/// being written, so a member's key does not depend on the context.
fn build_ordering_key(value: &Bound<'_, PyAny>) -> PyResult<String> {
    let py = value.py();
    let class = value.get_type();
    let module: String = class.getattr(intern!(py, "__module__"))?.str()?.to_string();
    let qualified_name = class.qualname()?;
    let payload = if value.is_instance(serializable_class(py)?)? {
        crate::kit::python::cached_attr!(py, "fhy_core.serialization", "_serialize_ordering_payload" => PyAny)?
            .call1((value,))?
    } else {
        value.clone()
    };
    Ok(format!("{module}.{qualified_name}:{}", payload.repr()?))
}

impl ForeignPart for PyOpaqueValue {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.type_name)
    }

    fn to_foreign(&self) -> Result<fhy_core::foreign::Foreign, fhy_core::foreign::ForeignError> {
        Python::attach(|py| crate::kit::foreign::foreign_of(&self.object.object(py), false))
    }
}

impl OpaqueValue for PyOpaqueValue {
    fn is_member_shaped(&self) -> bool {
        self.is_member_shaped
    }

    /// Compare with `type(a) is type(b) and a == b`; once an exception is
    /// pending, answer `false` without calling Python. A comparison that
    /// raises answers `false` and keeps its exception, which the entry
    /// function raises: `==` on a part cannot fail.
    fn eq_part(&self, other: &dyn OpaqueValue) -> bool {
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return false;
        };
        if has_pending_error() {
            return false;
        }
        Python::attach(|py| {
            if !self.class.get(py).is(other.class.get(py)) {
                return false;
            }
            match self.object.get(py).eq(other.object.get(py)) {
                Ok(is_equal) => is_equal,
                Err(error) => {
                    record_pending_error(error);
                    false
                }
            }
        })
    }

    fn check_hashable(&self) -> Result<(), BoxError> {
        Python::attach(|py| self.object.get(py).hash().map(|_hash| ()))
            .map_err(|error| Box::new(error) as BoxError)
    }

    /// Return the key computed when the member was read, or compute and
    /// keep it; a key that raises is the error, and is not kept.
    fn ordering_key(&self) -> Result<Cow<'_, str>, BoxError> {
        cached_key(&self.key, || {
            Python::attach(|py| build_ordering_key(&self.object.get(py)))
        })
        .map_err(|error| Box::new(error) as BoxError)
    }

    /// Order with Python's `<`, both ways: `Less` when `self < other`,
    /// `Greater` when `other < self`, and `Equal` otherwise. A comparison
    /// that raises is the error; once an exception is pending, answer
    /// `None` without calling Python.
    fn order_against(&self, other: &dyn OpaqueValue) -> Result<Option<Ordering>, BoxError> {
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return Ok(None);
        };
        if has_pending_error() {
            return Ok(None);
        }
        Python::attach(|py| {
            let (left, right) = (self.object.get(py), other.object.get(py));
            let ordering = left.lt(&right).and_then(|is_less| {
                if is_less {
                    Ok(Ordering::Less)
                } else {
                    right.lt(&left).map(|is_greater| {
                        if is_greater {
                            Ordering::Greater
                        } else {
                            Ordering::Equal
                        }
                    })
                }
            });
            ordering
                .map(Some)
                .map_err(|error| Box::new(error) as BoxError)
        })
    }
}

/// Return the key `cell` holds, or compute it with `compute` and keep it.
///
/// A key that fails to compute is the error, and is not kept, so a later
/// call computes it again.
fn cached_key(
    cell: &OnceLock<String>,
    compute: impl FnOnce() -> PyResult<String>,
) -> PyResult<Cow<'_, str>> {
    if let Some(key) = cell.get() {
        return Ok(Cow::Borrowed(key));
    }
    let key = compute()?;
    Ok(Cow::Borrowed(cell.get_or_init(|| key)))
}

/// Return the member-shaped opaque value of the `Serializable` `value`,
/// keyed as a constraint member's.
///
/// # Errors
///
/// Raises what computing the key raises.
pub(crate) fn read_opaque_member(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    let key = build_ordering_key(value)?;
    Ok(Value::Opaque(Part::new(PyOpaqueValue::new(
        value,
        true,
        Some(key),
    ))))
}

/// Return the Python value of the core `value`: a `bool`, an `int`, a
/// `float`, a `Decimal`, a `str`, a `tuple`, a `frozenset`, or the object of
/// an opaque value.
///
/// # Errors
///
/// Raises `TypeError` for an opaque value the binding did not build or a
/// value of a kind it does not know, and whatever building a value raises.
pub(crate) fn value_to_python<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    value_to_python_at(py, value, 0, &mut RecursionLimit(None))
}

/// The depth below which no value is checked against the recursion limit.
const SHALLOW_DEPTH: usize = 64;

/// Python's recursion limit, read once per call the first time a value
/// nests past [`SHALLOW_DEPTH`].
struct RecursionLimit(Option<usize>);

impl RecursionLimit {
    /// Raise `RecursionError` if `depth` levels of nesting pass Python's
    /// recursion limit.
    ///
    /// Each level of a value recurses on the Rust stack, so refusing what
    /// Python's own limit would refuse keeps a deep value from overflowing
    /// it, as the provenance binding does.
    fn check(&mut self, py: Python<'_>, depth: usize) -> PyResult<()> {
        if depth <= SHALLOW_DEPTH {
            return Ok(());
        }
        let limit = match self.0 {
            Some(limit) => limit,
            None => *self.0.insert(
                py.import(intern!(py, "sys"))?
                    .call_method0(intern!(py, "getrecursionlimit"))?
                    .extract()?,
            ),
        };
        if depth > limit {
            return Err(PyRecursionError::new_err(format!(
                "maximum recursion depth exceeded: the value is nested more than {limit} \
                 levels deep"
            )));
        }
        Ok(())
    }
}

/// Return the Python value of the core `value`, `depth` levels inside the
/// value being converted.
fn value_to_python_at<'py>(
    py: Python<'py>,
    value: &Value,
    depth: usize,
    limit: &mut RecursionLimit,
) -> PyResult<Bound<'py, PyAny>> {
    limit.check(py, depth)?;
    match value {
        Value::Bool(value) => Ok(PyBool::new(py, *value).to_owned().into_any()),
        Value::Int(value) => big_int_to_python(py, value),
        Value::Float(value) => Ok(PyFloat::new(py, *value).into_any()),
        Value::Decimal(value) => decimal_class(py)?.call1((value.to_string(),)),
        Value::Str(value) => Ok(PyString::new(py, value).into_any()),
        Value::Tuple(values) => {
            let elements = values
                .iter()
                .map(|value| value_to_python_at(py, value, depth + 1, limit))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyTuple::new(py, elements)?.into_any())
        }
        Value::FrozenSet(values) => {
            let elements = values
                .iter()
                .map(|value| value_to_python_at(py, value, depth + 1, limit))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyFrozenSet::new(py, &elements)?.into_any())
        }
        Value::Opaque(opaque) => opaque
            .get()
            .as_any()
            .downcast_ref::<PyOpaqueValue>()
            .map(|value| value.object(py))
            .ok_or_else(|| PyTypeError::new_err("an opaque value has no Python object")),
        _ => Err(PyTypeError::new_err(format!(
            "a value of an unknown kind has no Python form: {value:?}"
        ))),
    }
}

/// Return the opaque value of `object`, member-shaped as told.
fn build_opaque(object: &Bound<'_, PyAny>, is_member_shaped: bool) -> Value {
    Value::Opaque(Part::new(PyOpaqueValue::new(
        object,
        is_member_shaped,
        None,
    )))
}

/// Return the core value of a Python `bool`, `int` or `float`, or `None`
/// for another value.
fn read_number(value: &Bound<'_, PyAny>) -> PyResult<Option<Value>> {
    if let Ok(boolean) = value.cast::<PyBool>() {
        return Ok(Some(Value::Bool(boolean.is_true())));
    }
    if value.is_instance_of::<PyInt>() {
        return Ok(Some(Value::Int(read_big_int(value)?)));
    }
    if let Ok(float) = value.cast::<PyFloat>() {
        return Ok(Some(Value::Float(float.value())));
    }
    Ok(None)
}

/// Return the member of the Python value `value`.
///
/// # Errors
///
/// Raises `ConstraintError`, with the Python implementation's text, for a
/// value that cannot be a member, for a member whose hash raises, and with
/// the core's text for a NaN.
pub(crate) fn read_member(value: &Bound<'_, PyAny>) -> PyResult<Member> {
    let py = value.py();
    let read = read_member_value(value)?;
    let member = Member::try_from(read)
        .map_err(|error| constraint_error(py, format!("{error}: {}", repr_text(value))))?;
    check_member_hash(value, &member)?;
    Ok(member)
}

/// Raise the `ConstraintError` of a member whose hash raises, chained to
/// the hash's exception.
fn check_member_hash(value: &Bound<'_, PyAny>, member: &Member) -> PyResult<()> {
    let py = value.py();
    let mut pending = vec![member];
    while let Some(member) = pending.pop() {
        match member.kind() {
            MemberKind::Opaque(opaque) => {
                if let Err(error) = opaque.get().check_hashable() {
                    let refused = constraint_error(
                        py,
                        format!(
                            "Constraint member is unhashable after validation: value {} of \
                             type {}.",
                            repr_text(value),
                            type_name(value)
                        ),
                    );
                    // The opaque value is a Python object, so the error is
                    // the exception its hash raised.
                    refused.set_cause(py, Some(crate::kit::exceptions::boxed_error_to_py(error)));
                    return Err(refused);
                }
            }
            MemberKind::Tuple(members) => pending.extend(members),
            MemberKind::FrozenSet(members) => pending.extend(members.iter()),
            MemberKind::Bool(_)
            | MemberKind::Int(_)
            | MemberKind::Float(_)
            | MemberKind::Str(_) => {}
        }
    }
    Ok(())
}

/// Return the core value of the member-shaped Python value `value`,
/// validated as a member.
///
/// # Errors
///
/// Raises `ConstraintError` with the Python implementation's text for a
/// value that cannot be a member.
pub(crate) fn read_member_value(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    read_member_value_at(value, 0, &mut RecursionLimit(None))
}

/// Return the core value of the member-shaped `value`, `depth` levels inside
/// the member being read.
fn read_member_value_at(
    value: &Bound<'_, PyAny>,
    depth: usize,
    limit: &mut RecursionLimit,
) -> PyResult<Value> {
    let py = value.py();
    limit.check(py, depth)?;
    if value.is_none() {
        return Err(constraint_error(py, "Constraint members cannot be `None`."));
    }
    let is_tuple = value.is_instance_of::<PyTuple>();
    if is_tuple || value.is_instance_of::<PyFrozenSet>() {
        if !value.is_instance(hashable_class(py)?)? {
            return Err(constraint_error(
                py,
                format!(
                    "Constraint member containers must be hashable, but got value {} of type {}.",
                    str_text(value),
                    type_text(value)
                ),
            ));
        }
        let elements = value
            .try_iter()?
            .map(|element| read_member_value_at(&element?, depth + 1, limit))
            .collect::<PyResult<Vec<Value>>>()?;
        return Ok(if is_tuple {
            Value::Tuple(elements)
        } else {
            Value::FrozenSet(elements)
        });
    }
    if let Some(number) = read_number(value)? {
        return Ok(number);
    }
    if let Ok(text) = value.cast::<PyString>() {
        return Ok(Value::Str(text.to_str()?.to_owned()));
    }
    if is_serializable_hashable(value)? {
        let key = build_ordering_key(value)?;
        return Ok(Value::Opaque(Part::new(PyOpaqueValue::new(
            value,
            true,
            Some(key),
        ))));
    }
    Err(constraint_error(
        py,
        format!(
            "Constraint member must be either a primitive literal (`str`, `int`, `float`, \
             `bool`), both `Serializable` and `Hashable`, or a tuple/frozenset containing valid \
             constraint members, but got value {} of type {}.",
            str_text(value),
            type_text(value)
        ),
    ))
}

/// Return the core value of the Python value `value` bound to an
/// identifier, without judging it.
///
/// # Errors
///
/// Raises whatever reading a number raises, which an `int` or a `float`
/// never does.
pub(crate) fn read_bound_value(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    read_bound_value_at(value, 0, &mut RecursionLimit(None))
}

/// Return the core value of the bound `value`, `depth` levels inside the
/// value being read.
fn read_bound_value_at(
    value: &Bound<'_, PyAny>,
    depth: usize,
    limit: &mut RecursionLimit,
) -> PyResult<Value> {
    let py = value.py();
    limit.check(py, depth)?;
    if let Some(number) = read_number(value)? {
        return Ok(number);
    }
    if let Ok(text) = value.cast::<PyString>() {
        return Ok(Value::Str(text.to_str()?.to_owned()));
    }
    let is_tuple = value.is_instance_of::<PyTuple>();
    if is_tuple || value.is_instance_of::<PyFrozenSet>() {
        if !value.is_instance(hashable_class(py)?)? {
            return Ok(build_opaque(value, false));
        }
        let elements = value
            .try_iter()?
            .map(|element| read_bound_value_at(&element?, depth + 1, limit))
            .collect::<PyResult<Vec<Value>>>()?;
        return Ok(if is_tuple {
            Value::Tuple(elements)
        } else {
            Value::FrozenSet(elements)
        });
    }
    if value.is_instance(decimal_class(py)?)? {
        return Ok(match read_decimal(value) {
            Ok(decimal) => Value::Decimal(decimal),
            Err(_refused) => build_opaque(value, false),
        });
    }
    if value.is_none() {
        return Ok(build_opaque(value, false));
    }
    let is_member_shaped = is_serializable_hashable(value)?;
    Ok(build_opaque(value, is_member_shaped))
}

/// Return the Python value of `member`: a `bool`, an `int`, a `float`, a
/// `str`, a `tuple`, a `frozenset`, or the object of an opaque member.
///
/// # Errors
///
/// Raises `TypeError` for an opaque member the binding did not build, and
/// whatever building a value raises.
pub(crate) fn member_to_python<'py>(
    py: Python<'py>,
    member: &Member,
) -> PyResult<Bound<'py, PyAny>> {
    match member.kind() {
        MemberKind::Bool(value) => Ok(PyBool::new(py, value).to_owned().into_any()),
        MemberKind::Int(value) => big_int_to_python(py, value),
        MemberKind::Float(value) => Ok(PyFloat::new(py, value).into_any()),
        MemberKind::Str(value) => Ok(PyString::new(py, value).into_any()),
        MemberKind::Tuple(members) => {
            let elements = members
                .iter()
                .map(|member| member_to_python(py, member))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyTuple::new(py, elements)?.into_any())
        }
        MemberKind::FrozenSet(members) => {
            let elements = members
                .iter()
                .map(|member| member_to_python(py, member))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyFrozenSet::new(py, &elements)?.into_any())
        }
        MemberKind::Opaque(opaque) => opaque
            .get()
            .as_any()
            .downcast_ref::<PyOpaqueValue>()
            .map(|value| value.object(py))
            .ok_or_else(|| PyTypeError::new_err("an opaque member has no Python object")),
    }
}

#[cfg(test)]
mod tests {
    use pyo3::exceptions::PyValueError;

    use super::*;

    #[test]
    fn a_failed_key_is_not_kept() {
        Python::initialize();
        let cell = OnceLock::new();

        cached_key(&cell, || Err(PyValueError::new_err("once"))).expect_err("the key fails");
        assert!(cell.get().is_none());

        let key = cached_key(&cell, || Ok("real".to_owned())).expect("computes");
        assert_eq!(key, "real");
        assert_eq!(cell.get().map(String::as_str), Some("real"));
    }
}
