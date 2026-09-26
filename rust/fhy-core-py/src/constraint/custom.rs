//! A Python-defined `Constraint` as a core [`CustomConstraint`] (P3;
//! D-S13-5): a member of a system that is not one of the built-in kinds.
//!
//! The adapter calls the object's own methods, once per question, as the
//! system needs them. Its ordering key is read once, when the system is
//! built. `evaluate_with_bindings` receives the Python snapshot of the
//! caller's mapping, which the core carries as the bindings' source, so the
//! member sees the objects it saw before. An exception a hook raises
//! propagates as the same object; a comparison that raises answers `false`
//! and its exception is raised when the core returns.

use std::any::Any;
use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt;

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use fhy_core::constraint::{Bindings, CustomConstraint, CustomError, Outcome};
use fhy_core::expression::Expression;
use fhy_core::identifier::Identifier;
use fhy_core::term::AlphaRenaming;

use crate::expression::PyExpression;
use crate::identifier::{identifier_to_python, restore_identifier};
use crate::term::PyAlphaRenaming;

use super::kinds::outcome_to_python;
use super::value::{record_pending_error, type_name};

/// The Python form of a system's bindings: the snapshot of the caller's
/// mapping.
pub(super) struct PythonBindings(pub(super) Py<PyDict>);

/// A Python-defined constraint, driven through its methods.
pub(super) struct PyCustomConstraint {
    object: Py<PyAny>,
    key: String,
}

impl PyCustomConstraint {
    /// Return the adapter of `object`, reading its ordering key.
    ///
    /// # Errors
    ///
    /// Raises what `build_ordering_key` raises, and `TypeError` for a key
    /// that is not a `str`.
    pub(super) fn new(object: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = object.py();
        let key = object.call_method0(intern!(py, "build_ordering_key"))?;
        let key = key
            .cast::<PyString>()
            .map_err(|_not_a_string| {
                PyTypeError::new_err(format!(
                    "{}.build_ordering_key must return a str, got {}.",
                    type_name(object),
                    type_name(&key)
                ))
            })?
            .to_str()?
            .to_owned();
        Ok(Self {
            object: object.clone().unbind(),
            key,
        })
    }
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
fn read_outcome(object: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<Outcome> {
    let py = value.py();
    for outcome in [Outcome::Satisfied, Outcome::Violated, Outcome::Undecided] {
        if value.is(&outcome_to_python(py, outcome)?) {
            return Ok(outcome);
        }
    }
    Err(PyTypeError::new_err(format!(
        "{}.evaluate_with_bindings must return a ConstraintOutcome, got {}.",
        type_name(object),
        type_name(value)
    )))
}

impl CustomConstraint for PyCustomConstraint {
    fn free_identifiers(&self) -> HashSet<Identifier> {
        Python::attach(|py| -> PyResult<HashSet<Identifier>> {
            let object = self.object.bind(py);
            let identifiers = object.call_method0(intern!(py, "get_free_identifiers"))?;
            identifiers
                .try_iter()?
                .map(|identifier| {
                    restore_identifier(&identifier?, "get_free_identifiers", "identifier")
                })
                .collect()
        })
        .unwrap_or_else(|error| {
            record_pending_error(error);
            HashSet::new()
        })
    }

    fn evaluate(&self, bindings: &Bindings) -> Result<Outcome, CustomError> {
        Python::attach(|py| -> PyResult<Outcome> {
            let object = self.object.bind(py);
            let mapping = match bindings
                .source()
                .and_then(|source| source.downcast_ref::<PythonBindings>())
            {
                Some(PythonBindings(mapping)) => mapping.bind(py).clone(),
                None => PyDict::new(py),
            };
            let outcome = object.call_method1(intern!(py, "evaluate_with_bindings"), (mapping,))?;
            read_outcome(object, &outcome)
        })
        .map_err(|error| Box::new(error) as CustomError)
    }

    fn to_expression(&self) -> Result<Expression, CustomError> {
        Python::attach(|py| -> PyResult<Expression> {
            let object = self.object.bind(py);
            let expression = object.call_method0(intern!(py, "convert_to_expression"))?;
            expression
                .cast::<PyExpression>()
                .map(|node| node.get().expression().clone())
                .map_err(|_not_an_expression| {
                    PyTypeError::new_err(format!(
                        "{}.convert_to_expression must return an Expression, got {}.",
                        type_name(object),
                        type_name(&expression)
                    ))
                })
        })
        .map_err(|error| Box::new(error) as CustomError)
    }

    fn ordering_key(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.key)
    }

    fn is_structurally_equivalent(&self, other: &dyn CustomConstraint) -> bool {
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return false;
        };
        Python::attach(|py| {
            self.object
                .bind(py)
                .call_method1(
                    intern!(py, "is_structurally_equivalent"),
                    (other.object.bind(py),),
                )
                .and_then(|answer| answer.is_truthy())
                .unwrap_or_else(|error| {
                    record_pending_error(error);
                    false
                })
        })
    }

    fn is_alpha_equivalent_under(
        &self,
        other: &dyn CustomConstraint,
        renaming: &AlphaRenaming,
    ) -> bool {
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return false;
        };
        Python::attach(|py| {
            renaming_to_python(py, renaming)
                .and_then(|renaming| {
                    self.object.bind(py).call_method1(
                        intern!(py, "is_alpha_equivalent_under"),
                        (other.object.bind(py), renaming),
                    )
                })
                .and_then(|answer| answer.is_truthy())
                .unwrap_or_else(|error| {
                    record_pending_error(error);
                    false
                })
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
