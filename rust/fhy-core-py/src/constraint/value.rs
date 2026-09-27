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
//! A `Serializable` member is an opaque value (D-S13-3): [`PyOpaqueValue`]
//! compares it with Python's `==`, after `type(a) is type(b)`. A comparison
//! that raises answers `false` and keeps its exception in a per-thread
//! slot, which the entry point that started the comparison raises when the
//! core returns (S10's deferred-error pattern).

use std::any::Any;
use std::borrow::Cow;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::fmt;
use std::sync::OnceLock;

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyFloat, PyFrozenSet, PyInt, PyString, PyTuple, PyType};

use fhy_core::constraint::{Member, MemberKind, Opaque, OpaqueError, OpaqueValue, Value};

use crate::expression::{big_int_to_python, decimal_class, read_big_int, read_decimal};

thread_local! {
    /// The first exception an opaque value's `==` raised during the current
    /// call into the core on this thread.
    static PENDING_ERROR: RefCell<Option<PyErr>> = const { RefCell::new(None) };
}

/// Keep `error` as the pending exception, unless one is kept already.
pub(crate) fn record_pending_error(error: PyErr) {
    PENDING_ERROR.with(|pending| {
        let mut pending = pending.borrow_mut();
        if pending.is_none() {
            *pending = Some(error);
        }
    });
}

/// Run `call`, and return the exception an opaque value raised during it,
/// if any, in place of its result.
pub(crate) fn with_pending_errors<T>(call: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    let outer = PENDING_ERROR.with(|pending| pending.borrow_mut().take());
    let result = call();
    let raised = PENDING_ERROR.with(|pending| {
        let mut pending = pending.borrow_mut();
        let raised = pending.take();
        *pending = outer;
        raised
    });
    match raised {
        Some(error) => Err(error),
        None => result,
    }
}

/// Run `call`, and return its result with the exception an opaque value
/// raised during it, if any.
pub(crate) fn capture_pending_errors<T>(call: impl FnOnce() -> T) -> (T, Option<PyErr>) {
    let outer = PENDING_ERROR.with(|pending| pending.borrow_mut().take());
    let result = call();
    let raised = PENDING_ERROR.with(|pending| {
        let mut pending = pending.borrow_mut();
        let raised = pending.take();
        *pending = outer;
        raised
    });
    (result, raised)
}

/// Import the class `name` of `module` once, in `cell`.
fn import_class<'py>(
    py: Python<'py>,
    cell: &'static PyOnceLock<Py<PyType>>,
    module: &str,
    name: &str,
) -> PyResult<&'py Bound<'py, PyType>> {
    cell.import(py, module, name)
}

/// Return `fhy_core.symbolic.constraint.errors.ConstraintError`.
pub(crate) fn constraint_error_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    import_class(
        py,
        &CLASS,
        "fhy_core.symbolic.constraint.errors",
        "ConstraintError",
    )
}

/// Return the `ConstraintError` with `message`.
pub(crate) fn constraint_error(py: Python<'_>, message: impl Into<String>) -> PyErr {
    match constraint_error_class(py).and_then(|class| class.call1((message.into(),))) {
        Ok(error) => PyErr::from_value(error),
        Err(error) => error,
    }
}

/// Return `fhy_core.serialization.Serializable`.
fn serializable_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    import_class(py, &CLASS, "fhy_core.serialization", "Serializable")
}

/// Return `collections.abc.Hashable`.
fn hashable_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    import_class(py, &CLASS, "collections.abc", "Hashable")
}

/// Return whether `value` is both `Serializable` and `Hashable`.
fn is_serializable_hashable(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = value.py();
    Ok(value.is_instance(serializable_class(py)?)? && value.is_instance(hashable_class(py)?)?)
}

/// Return the name of `value`'s type.
pub(crate) fn type_name(value: &Bound<'_, PyAny>) -> String {
    value
        .get_type()
        .name()
        .map_or_else(|_| "?".to_owned(), |name| name.to_string())
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
pub(crate) struct PyOpaqueValue {
    object: Py<PyAny>,
    class: Py<PyType>,
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
            object: object.clone().unbind(),
            class: object.get_type().unbind(),
            type_name: type_name(object),
            is_member_shaped,
            key: key.map_or_else(OnceLock::new, OnceLock::from),
        }
    }

    /// Return the Python object.
    pub(crate) fn object(&self) -> &Py<PyAny> {
        &self.object
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
/// and the `repr` of its payload, as the Python implementation keyed a
/// `Serializable` member.
fn build_ordering_key(value: &Bound<'_, PyAny>) -> PyResult<String> {
    let py = value.py();
    let class = value.get_type();
    let module: String = class.getattr(intern!(py, "__module__"))?.str()?.to_string();
    let qualified_name = class.qualname()?;
    let payload = if value.is_instance(serializable_class(py)?)? {
        value.call_method0(intern!(py, "serialize_to_dict"))?
    } else {
        value.clone()
    };
    Ok(format!("{module}.{qualified_name}:{}", payload.repr()?))
}

impl OpaqueValue for PyOpaqueValue {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.type_name)
    }

    fn is_member_shaped(&self) -> bool {
        self.is_member_shaped
    }

    /// Compare with `type(a) is type(b) and a == b`.
    fn is_equal(&self, other: &dyn OpaqueValue) -> bool {
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return false;
        };
        Python::attach(|py| {
            if !self.class.bind(py).is(other.class.bind(py)) {
                return false;
            }
            match self.object.bind(py).eq(other.object.bind(py)) {
                Ok(is_equal) => is_equal,
                Err(error) => {
                    record_pending_error(error);
                    false
                }
            }
        })
    }

    fn check_hashable(&self) -> Result<(), OpaqueError> {
        Python::attach(|py| self.object.bind(py).hash().map(|_hash| ()))
            .map_err(|error| Box::new(error) as OpaqueError)
    }

    fn ordering_key(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.key.get_or_init(|| {
            Python::attach(|py| {
                build_ordering_key(self.object.bind(py)).unwrap_or_else(|error| {
                    record_pending_error(error);
                    String::new()
                })
            })
        }))
    }

    /// Order with Python's `<`, both ways: `Less` when `self < other`,
    /// `Greater` when `other < self`, and `Equal` otherwise. A comparison
    /// that raises answers `None`, and its exception is kept.
    fn order_against(&self, other: &dyn OpaqueValue) -> Option<Ordering> {
        let other = other.as_any().downcast_ref::<Self>()?;
        Python::attach(|py| {
            let (left, right) = (self.object.bind(py), other.object.bind(py));
            let ordering = left.lt(right).and_then(|is_less| {
                if is_less {
                    Ok(Ordering::Less)
                } else {
                    right.lt(left).map(|is_greater| {
                        if is_greater {
                            Ordering::Greater
                        } else {
                            Ordering::Equal
                        }
                    })
                }
            });
            match ordering {
                Ok(ordering) => Some(ordering),
                Err(error) => {
                    record_pending_error(error);
                    None
                }
            }
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Return the member-shaped opaque value of the `Serializable` `value`,
/// keyed as a constraint member's.
///
/// # Errors
///
/// Raises what computing the key raises.
pub(crate) fn read_opaque_member(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    let key = build_ordering_key(value)?;
    Ok(Value::Opaque(Opaque::new(PyOpaqueValue::new(
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
/// Raises `TypeError` for an opaque value the binding did not build, and
/// whatever building a value raises.
pub(crate) fn value_to_python<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    match value {
        Value::Bool(value) => Ok(PyBool::new(py, *value).to_owned().into_any()),
        Value::Int(value) => big_int_to_python(py, value),
        Value::Float(value) => Ok(PyFloat::new(py, *value).into_any()),
        Value::Decimal(value) => decimal_class(py)?.call1((value.to_string(),)),
        Value::Str(value) => Ok(PyString::new(py, value).into_any()),
        Value::Tuple(values) => {
            let elements = values
                .iter()
                .map(|value| value_to_python(py, value))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyTuple::new(py, elements)?.into_any())
        }
        Value::FrozenSet(values) => {
            let elements = values
                .iter()
                .map(|value| value_to_python(py, value))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyFrozenSet::new(py, &elements)?.into_any())
        }
        Value::Opaque(opaque) => opaque
            .get()
            .as_any()
            .downcast_ref::<PyOpaqueValue>()
            .map(|value| value.object().bind(py).clone())
            .ok_or_else(|| PyTypeError::new_err("an opaque value has no Python object")),
    }
}

/// Return the opaque value of `object`, member-shaped as told.
fn build_opaque(object: &Bound<'_, PyAny>, is_member_shaped: bool) -> Value {
    Value::Opaque(Opaque::new(PyOpaqueValue::new(
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
    let member = Member::try_from_value(read)
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
                    if let Ok(cause) = error.downcast::<PyErr>() {
                        refused.set_cause(py, Some(*cause));
                    }
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
    let py = value.py();
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
            .map(|element| read_member_value(&element?))
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
        return Ok(Value::Opaque(Opaque::new(PyOpaqueValue::new(
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
    let py = value.py();
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
            .map(|element| read_bound_value(&element?))
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
            .map(|value| value.object().bind(py).clone())
            .ok_or_else(|| PyTypeError::new_err("an opaque member has no Python object")),
    }
}
