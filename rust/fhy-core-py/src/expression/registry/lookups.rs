//! The registry's Python functions: registration, the lookups, the test
//! seam, and inlining (D-S7-11, D-S7-12, D-S7-14, D-S7-7).
//!
//! Every lookup resolves a name through the built-ins first, then the user
//! registry (N-S7-3 (a)), and returns the entry's single Python object. A
//! registration returns the object later lookups return.

use pyo3::exceptions::{PyRecursionError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyMapping, PyString, PyTuple, PyType};

use fhy_core::expression::FunctionName;
use fhy_core::expression::builtins::{BuiltinConstant, BuiltinFunction};
use fhy_core::expression::registry::InlineError;

use crate::identifier::read_identifier_id;

use super::super::materialize::materialize_beside;
use super::super::node::read_expression;
use super::entries::{PyNativeConstant, PyNativeFunction, PyRegisteredFunction, sort_to_python};
use super::state;

/// Return `fhy_core.symbolic.expression.errors.<name>`.
fn error_class<'py>(
    py: Python<'py>,
    cell: &'static PyOnceLock<Py<PyType>>,
    name: &str,
) -> PyResult<&'py Bound<'py, PyType>> {
    cell.import(py, "fhy_core.symbolic.expression.errors", name)
}

/// Return the `EntryRegistrationError` carrying `message`.
pub(super) fn registration_error(py: Python<'_>, message: &str) -> PyErr {
    static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    match error_class(py, &CLASS, "EntryRegistrationError") {
        Ok(class) => match class.call1((message,)) {
            Ok(error) => PyErr::from_value(error),
            Err(error) => error,
        },
        Err(error) => error,
    }
}

/// Return the `EntryLookupError` carrying `message`.
pub(in crate::expression) fn lookup_error(py: Python<'_>, message: &str) -> PyErr {
    static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    match error_class(py, &CLASS, "EntryLookupError") {
        Ok(class) => match class.call1((message,)) {
            Ok(error) => PyErr::from_value(error),
            Err(error) => error,
        },
        Err(error) => error,
    }
}

/// Return the `FunctionArityError` of `fhy_core.symbolic.expression.passes
/// .inline` carrying `message`.
pub(in crate::expression) fn arity_error(py: Python<'_>, message: &str) -> PyErr {
    static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    match CLASS.import(
        py,
        "fhy_core.symbolic.expression.passes.inline",
        "FunctionArityError",
    ) {
        Ok(class) => match class.call1((message,)) {
            Ok(error) => PyErr::from_value(error),
            Err(error) => error,
        },
        Err(error) => error,
    }
}

/// Return the Python exception of the inlining error `error`: the core's
/// text under `EntryLookupError` for an unknown name, `FunctionArityError`
/// for a wrong argument count or a constant called, `RecursionError` for a
/// recursive function, and `ValueError`, followed by the refusal, for an
/// invalid piecewise.
pub(in crate::expression) fn inline_error_to_python(py: Python<'_>, error: &InlineError) -> PyErr {
    let message = error.to_string();
    match error {
        InlineError::UnknownFunction(_) => lookup_error(py, &message),
        InlineError::ArityMismatch { .. } | InlineError::NotCallable(_) => {
            arity_error(py, &message)
        }
        InlineError::Recursive(_) => PyRecursionError::new_err(message),
        InlineError::Piecewise(source) => PyValueError::new_err(format!("{message}: {source}")),
        other => PyValueError::new_err(other.to_string()),
    }
}

/// Build an entry by calling `class` with `arguments`, raising a
/// `ValueError` it raises as `EntryRegistrationError` with the same text,
/// caused by it, and register the entry.
fn build_and_register<'py>(
    class: &Bound<'py, PyType>,
    arguments: Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = class.py();
    let object = class.call1(arguments).map_err(|error| {
        if error.is_instance_of::<PyValueError>(py) {
            let wrapped = registration_error(py, &error.value(py).to_string());
            wrapped.set_cause(py, Some(error));
            wrapped
        } else {
            error
        }
    })?;
    state::register(py, &object)?;
    Ok(object)
}

/// Register the function `name` of the identifiers `parameters`, whose
/// sorts are `parameter_sorts` in order, returning `result_sort`, as the
/// expression `body`, and return its entry.
///
/// The body is not type-checked, and may call functions not registered yet.
/// Its free identifiers must be parameters, or identifiers of constants
/// registered so far or built in.
///
/// Raises `TypeError` for an argument of the wrong type, and
/// `EntryRegistrationError` with the core's text for a name that is empty,
/// a built-in's or taken, a sort count other than the parameter count, a
/// repeated parameter, or a captured identifier.
#[pyfunction]
pub(crate) fn register_function<'py>(
    name: &Bound<'py, PyAny>,
    parameters: &Bound<'py, PyAny>,
    parameter_sorts: &Bound<'py, PyAny>,
    result_sort: &Bound<'py, PyAny>,
    body: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = name.py();
    let arguments = PyTuple::new(py, [name, parameters, parameter_sorts, result_sort, body])?;
    build_and_register(PyRegisteredFunction::public_class().get(py)?, arguments)
}

/// Register the native function `name` taking arguments of
/// `parameter_sorts`, in order, returning `result_sort`, computed by the
/// callable `implementation`, and return its entry.
///
/// Raises `TypeError` for an argument of the wrong type, and
/// `EntryRegistrationError` for a name that is empty, a built-in's or
/// taken, or an implementation whose inspectable signature cannot take as
/// many positional arguments as there are sorts.
#[pyfunction]
pub(crate) fn register_native_function<'py>(
    name: &Bound<'py, PyAny>,
    parameter_sorts: &Bound<'py, PyAny>,
    result_sort: &Bound<'py, PyAny>,
    implementation: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = name.py();
    let arguments = PyTuple::new(py, [name, parameter_sorts, result_sort, implementation])?;
    build_and_register(PyNativeFunction::public_class().get(py)?, arguments)
}

/// Register the constant `name` of the sort `sort` holding `value`, minting
/// the identifier an expression refers to it by, and return its entry.
///
/// Raises `TypeError` for an argument of the wrong type, and
/// `EntryRegistrationError` for a name that is empty, a built-in's or
/// taken, or a value the sort does not accept.
#[pyfunction]
pub(crate) fn register_native_constant<'py>(
    name: &Bound<'py, PyAny>,
    sort: &Bound<'py, PyAny>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = name.py();
    let arguments = PyTuple::new(py, [name, sort, value])?;
    build_and_register(PyNativeConstant::public_class().get(py)?, arguments)
}

/// Return the entry registered, or built in, under `name`.
///
/// Raises `EntryLookupError` if no entry has the name.
#[pyfunction]
pub(crate) fn get_registered_entry<'py>(name: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = name.py();
    if let Ok(text) = name.cast::<PyString>() {
        let text = text.to_str()?;
        if let Some(object) = state::find_builtin(py, text)? {
            return Ok(object);
        }
        if let Some(object) = state::snapshot().object(py, text) {
            return Ok(object);
        }
    }
    Err(lookup_error(
        py,
        &format!("No entry is registered under the name {}.", name.repr()?),
    ))
}

/// Return the immutable mapping of every entry's name to its entry: the
/// built-ins in catalogue order (the constants, the composed functions, the
/// native functions), then the user entries in registration order.
#[pyfunction]
pub(crate) fn get_registered_entries(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    state::snapshot().entries_view(py)
}

/// Return whether an entry is registered, or built in, under `name`.
#[pyfunction]
pub(crate) fn is_entry_registered(name: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = name.py();
    let Ok(text) = name.cast::<PyString>() else {
        return Ok(false);
    };
    let text = text.to_str()?;
    Ok(state::builtins(py)?.contains(text) || state::snapshot().object(py, text).is_some())
}

/// Return the identifier an expression refers to the constant `name` by.
///
/// A built-in constant's identifier has its fixed reserved id, the same in
/// every process.
///
/// Raises `EntryLookupError` if no constant has the name.
#[pyfunction]
pub(crate) fn get_native_constant_identifier<'py>(
    name: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = name.py();
    if let Ok(text) = name.cast::<PyString>() {
        let text = text.to_str()?;
        if let Some(identifier) = state::builtins(py)?.constant_identifier(py, text) {
            return Ok(identifier);
        }
        if let Some(identifier) = state::snapshot().constant_identifier(py, text)? {
            return Ok(identifier);
        }
    }
    Err(lookup_error(
        py,
        &format!(
            "No native constant is registered under the name {}.",
            name.repr()?
        ),
    ))
}

/// Return the constant `identifier` refers to, or `None`.
///
/// Resolution is by the identifier's id: only a constant's own identifier
/// refers to it, and one merely named like it does not.
#[pyfunction]
pub(crate) fn try_get_native_constant_for_identifier<'py>(
    identifier: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = identifier.py();
    let Some(id) = read_identifier_id(identifier)? else {
        return Ok(None);
    };
    if let Some(object) = state::builtins(py)?.constant_object(py, id) {
        return Ok(Some(object));
    }
    Ok(state::snapshot().constant_object(py, id))
}

/// Return the result sort of the function registered, or built in, under
/// `function_name`, or `None` for a constant or an unknown name.
#[pyfunction]
pub(crate) fn try_get_registered_result_sort<'py>(
    function_name: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = function_name.py();
    let Ok(text) = function_name.cast::<PyString>() else {
        return Ok(None);
    };
    let text = text.to_str()?;
    if let Ok(function) = text.parse::<BuiltinFunction>() {
        return sort_to_python(py, function.result_sort()).map(Some);
    }
    if text.parse::<BuiltinConstant>().is_ok() {
        return Ok(None);
    }
    let Ok(name) = FunctionName::try_new(text) else {
        return Ok(None);
    };
    match state::snapshot().registry().result_sort(&name) {
        Some(sort) => sort_to_python(py, sort).map(Some),
        None => Ok(None),
    }
}

/// Replace the user entries with the entries of `state`, a mapping of names
/// to entry objects, for the tests' snapshot fixture.
///
/// An entry registered under its name keeps its place, and a constant its
/// identifier, when `state` holds that very object; the other entries of
/// `state` are registered anew after them, a constant with a new
/// identifier. Built-in names in `state` are ignored: the built-ins are no
/// state and cannot be removed. Production code must not call it.
///
/// Raises `TypeError` for a value that is no user entry, and
/// `EntryRegistrationError` for one that cannot be registered, leaving the
/// registry as it was.
#[pyfunction]
pub(crate) fn set_registry_state_for_tests(state: &Bound<'_, PyAny>) -> PyResult<()> {
    state::restore(state.py(), state.cast::<PyMapping>()?)
}

/// Return `expression` with every call of a composed built-in or of a
/// registered function replaced by its body over the call's arguments,
/// inlined in turn; native calls are kept, with their arity checked.
///
/// Returns `expression` itself when it calls nothing to inline, and
/// otherwise a tree sharing every subtree object without such a call. A
/// node that occurs in several places is inlined once, and a tree of any
/// depth inlines.
///
/// Raises `TypeError` for a value that is not an `Expression`,
/// `EntryLookupError` for an unknown name, `FunctionArityError` for a
/// wrong argument count or a call of a constant, `RecursionError` for a
/// function reached again inside its own body, and `ValueError` if an
/// argument lands in a piecewise condition as a numeric literal.
#[pyfunction]
pub(crate) fn inline_functions<'py>(expression: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = expression.py();
    let expression = read_expression(expression, "inline_functions", "expression")?;
    let snapshot = state::snapshot();
    let result = snapshot
        .registry()
        .inline(expression.get().expression())
        .map_err(|error| inline_error_to_python(py, &error))?;
    materialize_beside(expression, &result)
}
