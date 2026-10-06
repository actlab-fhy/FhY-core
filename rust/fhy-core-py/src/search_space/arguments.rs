//! What every class of the module shares: reading constructor arguments,
//! the depth guard, and the seed through which the binding builds an
//! instance of a public class from a core value.

use pyo3::exceptions::{PyRecursionError, PyTypeError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyType};

use fhy_core::diagnostic::Note;
use fhy_core::identifier::Identifier;
use fhy_core::search_space::Choice;

use crate::diagnostic::note_from_python;
use crate::identifier::{new_python_identifier, restore_identifier};
use crate::util::python::{Seed, read_type_name};

use super::choice::PyChoice;
use super::configuration::PyConfiguration;
use super::space::{PyCondition, PyForbidden, PySpace};

/// Return the `TypeError` of the argument `field` of `owner` that is not
/// `expected`: `<owner> <field> must be <expected>, got <type>.`.
pub(super) fn wrong_argument(
    owner: &str,
    field: &str,
    expected: &str,
    value: &Bound<'_, PyAny>,
) -> PyErr {
    PyTypeError::new_err(format!(
        "{owner} {field} must be {expected}, got {}.",
        read_type_name(value)
    ))
}

/// Return the name `name` gives, a fresh `Identifier(hint)` when it is
/// omitted or `None`, as the core identifier and its object.
///
/// # Errors
///
/// Raises `TypeError` naming `owner` for a name that is no `Identifier`.
pub(super) fn read_name<'py>(
    py: Python<'py>,
    name: Option<&Bound<'py, PyAny>>,
    owner: &str,
    hint: &str,
) -> PyResult<(Identifier, Bound<'py, PyAny>)> {
    let object = match name {
        Some(name) if !name.is_none() => name.clone(),
        _ => new_python_identifier(py, hint)?,
    };
    Ok((restore_identifier(&object, owner, "name")?, object))
}

/// Return the items of the iterable `values` as a tuple, the empty tuple
/// when omitted or `None`.
///
/// # Errors
///
/// Raises the `TypeError` of [`wrong_argument`] for a value that is not
/// iterable.
pub(super) fn read_items<'py>(
    py: Python<'py>,
    values: Option<&Bound<'py, PyAny>>,
    owner: &str,
    field: &str,
    expected: &str,
) -> PyResult<Bound<'py, PyTuple>> {
    let Some(values) = values.filter(|values| !values.is_none()) else {
        return Ok(PyTuple::empty(py));
    };
    if let Ok(tuple) = values.cast_exact::<PyTuple>() {
        return Ok(tuple.clone());
    }
    let Ok(items) = values.try_iter() else {
        return Err(wrong_argument(owner, field, expected, values));
    };
    PyTuple::new(py, items.collect::<PyResult<Vec<_>>>()?)
}

/// Return the notes `notes` gives, the core notes and their tuple.
///
/// # Errors
///
/// Raises the `TypeError` of [`wrong_argument`] for a value that is no
/// iterable of `Note`s.
pub(super) fn read_notes<'py>(
    py: Python<'py>,
    notes: Option<&Bound<'py, PyAny>>,
    owner: &str,
) -> PyResult<(Vec<Note>, Bound<'py, PyTuple>)> {
    let objects = read_items(py, notes, owner, "notes", "Notes")?;
    let notes = objects
        .iter()
        .map(|note| {
            note_from_python(&note).ok_or_else(|| wrong_argument(owner, "notes", "Notes", &note))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok((notes, objects))
}

/// Return the levels of choices `choice` nests, itself included, walking
/// it without recursion.
pub(super) fn choice_depth(choice: &Choice) -> usize {
    let mut deepest = 0;
    let mut pending = vec![(choice, 1)];
    while let Some((choice, depth)) = pending.pop() {
        deepest = deepest.max(depth);
        for alternative in choice.alternatives() {
            for sub_choice in alternative.get().choices() {
                pending.push((sub_choice, depth + 1));
            }
        }
    }
    deepest
}

/// Return the deepest of `choices`' depths, 0 for none.
pub(super) fn choices_depth<'a>(choices: impl IntoIterator<Item = &'a Choice>) -> usize {
    choices.into_iter().map(choice_depth).max().unwrap_or(0)
}

/// Return Python's recursion limit.
///
/// # Errors
///
/// Raises what reading it raises.
pub(super) fn recursion_limit(py: Python<'_>) -> PyResult<usize> {
    py.import(intern!(py, "sys"))?
        .call_method0(intern!(py, "getrecursionlimit"))?
        .extract()
}

/// Raise `RecursionError` if a `class` nesting `depth` levels of choices
/// is deeper than Python's recursion limit.
///
/// The core compares, hashes, serializes and drops a value recursively,
/// once per level of choices, so a value deeper than Python itself would
/// recurse is refused, as a deep provenance is.
///
/// # Errors
///
/// Raises as described, and what reading the limit raises.
pub(super) fn ensure_depth(py: Python<'_>, class: &str, depth: usize) -> PyResult<()> {
    if depth > recursion_limit(py)? {
        return Err(PyRecursionError::new_err(format!(
            "maximum recursion depth exceeded: the {class} is {depth} levels deep"
        )));
    }
    Ok(())
}

/// The value an instance of a public class is built from, handed to its
/// `__new__` through the private keyword `_seed`.
pub(super) enum Seeded {
    Choice(PyChoice),
    Condition(PyCondition),
    Forbidden(PyForbidden),
    Space(PySpace),
    Configuration(PyConfiguration),
}

/// The seed of an instance the binding builds from a core value.
#[pyclass(frozen, module = "fhy_core._rs", name = "_SearchSpaceSeed")]
pub(super) struct PySeed(Seed<Seeded>);

/// Return a new instance of the public class `class` built from `seeded`,
/// through its `__new__`, its `arguments` positional arguments `None`.
///
/// # Errors
///
/// Raises what calling `__new__` raises.
pub(super) fn instantiate<'py>(
    class: &Bound<'py, PyType>,
    arguments: usize,
    seeded: Seeded,
) -> PyResult<Bound<'py, PyAny>> {
    let py = class.py();
    let seed = Py::new(py, PySeed(Seed::new(seeded)))?;
    let keywords = PyDict::new(py);
    keywords.set_item(intern!(py, "_seed"), seed)?;
    let mut positional: Vec<Bound<'py, PyAny>> = vec![class.clone().into_any()];
    positional.extend((0..arguments).map(|_| py.None().into_bound(py)));
    class.call_method(
        intern!(py, "__new__"),
        PyTuple::new(py, positional)?,
        Some(&keywords),
    )
}

/// Return what the seed in the keywords `kwargs` holds, if any.
///
/// # Errors
///
/// Raises `TypeError` for any other keyword or a seed taken already.
pub(super) fn take_seed(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Option<Seeded>> {
    let Some(kwargs) = kwargs else {
        return Ok(None);
    };
    let mut seeded = None;
    for (key, value) in kwargs.iter() {
        let seed = key
            .eq("_seed")?
            .then(|| value.cast_into::<PySeed>().ok())
            .flatten()
            .ok_or_else(|| {
                PyTypeError::new_err(format!(
                    "unexpected keyword argument {}",
                    key.repr()
                        .map_or_else(|_| "?".to_owned(), |text| text.to_string())
                ))
            })?;
        seeded = Some(seed.get().0.take("a search-space")?);
    }
    Ok(seeded)
}

/// Return the `TypeError` of a seed of the wrong class.
pub(super) fn wrong_seed() -> PyErr {
    PyTypeError::new_err("the seed holds a value of another class")
}
