//! A Python-defined `Constraint` as a core [`CustomConstraint`]: a member of
//! a system that is not one of the built-in kinds.
//!
//! The adapter calls the object's own methods, once per question, as the
//! system needs them. Its ordering key is read once, when the system is
//! built. `evaluate_with_bindings` receives the Python snapshot of the
//! caller's mapping, which the core carries as the bindings' source, so the
//! member sees the objects it saw before; bindings the core built itself,
//! with no source, reach it as a dict rebuilt from them. An
//! exception a hook raises propagates as the same object: it is the hook's
//! error, except that `is_structurally_equivalent`, which backs the core's
//! equality and cannot fail, answers `false` and keeps its exception for
//! the entry function to raise. Once an exception is pending, the hooks
//! answer a fallback without calling Python.

use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt;

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use fhy_core::constraint::{Binding, Bindings, ConstraintContext, CustomConstraint, Outcome};
use fhy_core::expression::Expression;
use fhy_core::foreign::{BoxError, ForeignPart};
use fhy_core::identifier::Identifier;
use fhy_core::term::AlphaRenaming;

use crate::expression::{PyExpression, materialize_expression};
use crate::identifier::{identifier_to_python, restore_identifier};
use crate::term::PyAlphaRenaming;
use crate::util::gc::Slot;

use super::kinds::outcome_to_python;
use super::value::{read_type_name, value_to_python};
use crate::util::hook::ask;
use crate::util::pending::has_pending_error;

/// The Python form of a system's bindings: the snapshot of the caller's
/// mapping.
pub(crate) struct PythonBindings(pub(crate) Py<PyDict>);

/// A Python-defined constraint, driven through its methods.
///
/// The object is kept in a [`Slot`], which the object whose construction
/// made the adapter owns and traverses.
pub(crate) struct PyCustomConstraint {
    object: Slot,
    key: String,
}

impl PyCustomConstraint {
    /// Return the adapter of `object`, reading its ordering key.
    ///
    /// # Errors
    ///
    /// Raises what `build_ordering_key` raises, and `TypeError` for a key
    /// that is not a `str`.
    pub(crate) fn new(object: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = object.py();
        let key = object.call_method0(intern!(py, "build_ordering_key"))?;
        let key = key
            .cast::<PyString>()
            .map_err(|_not_a_string| {
                PyTypeError::new_err(format!(
                    "{}.build_ordering_key must return a str, got {}.",
                    read_type_name(object),
                    read_type_name(&key)
                ))
            })?
            .to_str()?
            .to_owned();
        Ok(Self {
            object: Slot::new(object.clone().unbind()),
            key,
        })
    }

    /// Return the Python object.
    pub(crate) fn object<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.object.get(py)
    }
}

/// Return the Python dict of the core `bindings`, for a Python-defined
/// member evaluated under bindings the core built: each identifier's Python
/// object, bound to an expression's object or a value's.
fn build_python_bindings<'py>(
    py: Python<'py>,
    bindings: &Bindings,
) -> PyResult<Bound<'py, PyDict>> {
    let mapping = PyDict::new(py);
    for (identifier, binding) in bindings.iter() {
        let value = match binding {
            Binding::Expression(expression) => materialize_expression(py, expression)?,
            Binding::Value(value) => value_to_python(py, value)?,
        };
        mapping.set_item(identifier_to_python(py, identifier)?, value)?;
    }
    Ok(mapping)
}

impl fmt::Debug for PyCustomConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PyCustomConstraint")
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

/// Return the Python `AlphaRenaming` of the core `renaming`.
fn renaming_to_python<'py>(
    py: Python<'py>,
    renaming: &AlphaRenaming,
) -> PyResult<Bound<'py, PyAny>> {
    let class = py.get_type::<PyAlphaRenaming>();
    let free = PyDict::new(py);
    for (key, value) in renaming.free_renaming().iter() {
        free.set_item(
            identifier_to_python(py, key)?,
            identifier_to_python(py, value)?,
        )?;
    }
    let mut built = class.call_method1(intern!(py, "with_free_renaming"), (free,))?;
    for frame in renaming.frames() {
        let bindings = PyDict::new(py);
        for (key, value) in frame.iter() {
            bindings.set_item(
                identifier_to_python(py, key)?,
                identifier_to_python(py, value)?,
            )?;
        }
        built = built.call_method1(intern!(py, "extend"), (bindings,))?;
    }
    Ok(built)
}

/// Return the outcome of the `ConstraintOutcome` member `value`.
pub(crate) fn read_outcome(
    object: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<Outcome> {
    let py = value.py();
    for outcome in [Outcome::Satisfied, Outcome::Violated, Outcome::Undecided] {
        if value.is(&outcome_to_python(py, outcome)?) {
            return Ok(outcome);
        }
    }
    Err(PyTypeError::new_err(format!(
        "{}.evaluate_with_bindings must return a ConstraintOutcome, got {}.",
        read_type_name(object),
        read_type_name(value)
    )))
}

impl ForeignPart for PyCustomConstraint {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Owned(Python::attach(|py| read_type_name(&self.object.get(py))))
    }

    fn to_foreign(&self) -> Result<fhy_core::foreign::Foreign, fhy_core::foreign::ForeignError> {
        Python::attach(|py| crate::util::foreign::read_foreign(&self.object.object(py), true))
    }
}

impl CustomConstraint for PyCustomConstraint {
    /// Return the object's `get_free_identifiers`; once an exception is
    /// pending, the empty scope, without calling Python.
    fn free_identifiers(&self) -> Result<HashSet<Identifier>, BoxError> {
        if has_pending_error() {
            return Ok(HashSet::new());
        }
        Python::attach(|py| -> PyResult<HashSet<Identifier>> {
            let object = &self.object.get(py);
            let identifiers = object.call_method0(intern!(py, "get_free_identifiers"))?;
            identifiers
                .try_iter()?
                .map(|identifier| {
                    restore_identifier(&identifier?, "get_free_identifiers", "identifier")
                })
                .collect()
        })
        .map_err(|error| Box::new(error) as BoxError)
    }

    fn evaluate(
        &self,
        bindings: &Bindings,
        _context: &ConstraintContext<'_>,
    ) -> Result<Outcome, BoxError> {
        Python::attach(|py| -> PyResult<Outcome> {
            let object = &self.object.get(py);
            let mapping = match bindings
                .source()
                .and_then(|source| source.downcast_ref::<PythonBindings>())
            {
                Some(PythonBindings(mapping)) => mapping.bind(py).clone(),
                None => build_python_bindings(py, bindings)?,
            };
            let outcome = object.call_method1(intern!(py, "evaluate_with_bindings"), (mapping,))?;
            read_outcome(object, &outcome)
        })
        .map_err(|error| Box::new(error) as BoxError)
    }

    fn to_expression(&self) -> Result<Expression, BoxError> {
        Python::attach(|py| -> PyResult<Expression> {
            let object = &self.object.get(py);
            let expression = object.call_method0(intern!(py, "convert_to_expression"))?;
            expression
                .cast::<PyExpression>()
                .map(|node| node.get().expression().clone())
                .map_err(|_not_an_expression| {
                    PyTypeError::new_err(format!(
                        "{}.convert_to_expression must return an Expression, got {}.",
                        read_type_name(object),
                        read_type_name(&expression)
                    ))
                })
        })
        .map_err(|error| Box::new(error) as BoxError)
    }

    /// Return the key read when the adapter was built.
    fn ordering_key(&self) -> Result<Cow<'_, str>, BoxError> {
        Ok(Cow::Borrowed(&self.key))
    }

    /// Ask the object's `is_structurally_equivalent`. It cannot fail, so an
    /// exception answers `false` and is kept for the entry function to
    /// raise; once one is pending, answer `false` without calling Python.
    fn eq_part(&self, other: &dyn CustomConstraint) -> bool {
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return false;
        };
        ask(false, |py| {
            self.object
                .get(py)
                .call_method1(
                    intern!(py, "is_structurally_equivalent"),
                    (other.object.get(py),),
                )
                .and_then(|answer| answer.is_truthy())
        })
    }

    /// Ask the object's `is_alpha_equivalent_under`; once an exception is
    /// pending, answer `false` without calling Python.
    fn is_alpha_equivalent_under(
        &self,
        other: &dyn CustomConstraint,
        renaming: &AlphaRenaming,
    ) -> Result<bool, BoxError> {
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return Ok(false);
        };
        if has_pending_error() {
            return Ok(false);
        }
        Python::attach(|py| {
            renaming_to_python(py, renaming)
                .and_then(|renaming| {
                    self.object.get(py).call_method1(
                        intern!(py, "is_alpha_equivalent_under"),
                        (other.object.get(py), renaming),
                    )
                })
                .and_then(|answer| answer.is_truthy())
        })
        .map_err(|error| Box::new(error) as BoxError)
    }
}
