//! `fhy_core._rs.Choice`: a named decision among alternatives.

use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyTuple, PyType};

use fhy_core::search_space::Choice;

use crate::util::gc::Slots;
use crate::util::public_class::PublicClass;

/// A named decision among one or more alternatives, backed by the core
/// [`Choice`]; the base of the public `Choice`.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Choice")]
pub(crate) struct PyChoice {
    choice: Choice,
    /// The `Identifier` the choice is named by.
    name: Py<PyAny>,
    /// The `Alternative`s, as given.
    alternatives: Py<PyTuple>,
    /// The `Note`s.
    notes: Py<PyTuple>,
    /// The levels of choices the choice nests, itself included.
    depth: usize,
    /// The slots of the adapters of the subclass alternatives it holds.
    slots: Slots,
}

impl PyChoice {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("Choice");
        &PUBLIC_CLASS
    }

    /// Return the core choice.
    pub(super) const fn core(&self) -> &Choice {
        &self.choice
    }

    /// Return the levels of choices the choice nests, itself included.
    pub(super) const fn depth(&self) -> usize {
        self.depth
    }
}

#[pymethods]
impl PyChoice {
    /// Return the choice named `name` (a fresh `Identifier("choice")` when
    /// `None`) among `alternatives`, with `notes`.
    ///
    /// Raises `TypeError` for an argument of the wrong type,
    /// `SearchSpaceError` for no alternative, `DuplicateNameError` for
    /// names that repeat, the exception an alternative's
    /// `extension_bound_identifiers` raises, and `RecursionError` for
    /// choices nested deeper than the recursion limit.
    #[new]
    #[pyo3(signature = (alternatives, name = None, notes = None))]
    fn new(
        alternatives: &Bound<'_, PyAny>,
        name: Option<&Bound<'_, PyAny>>,
        notes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        todo!()
    }

    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.name)?;
        visit.call(&self.alternatives)?;
        visit.call(&self.notes)?;
        self.slots.traverse(&visit)
    }

    /// Register `cls` as the public `Choice` class.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }

    /// The `Identifier` the choice is named by.
    #[getter]
    fn name<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        todo!()
    }

    /// The `Alternative`s, in order.
    #[getter]
    fn alternatives<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        todo!()
    }

    /// The `Note`s attached to the choice.
    #[getter]
    fn notes<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        todo!()
    }

    /// Return whether `other` is a choice with the same name, structurally
    /// equivalent alternatives in order and equal notes.
    fn is_structurally_equivalent(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        todo!()
    }

    /// Return whether `other` is the same choice up to the renaming of the
    /// names it binds, under no renaming.
    fn is_alpha_equivalent(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        todo!()
    }

    /// Return whether `other` is the same choice up to the renaming of the
    /// names it binds, under `renaming`.
    fn is_alpha_equivalent_under(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        renaming: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        todo!()
    }

    /// Refuse to set an attribute: the choice is frozen.
    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        todo!()
    }

    /// Refuse to delete an attribute: the choice is frozen.
    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        todo!()
    }

    /// Return `Choice(name=..., alternatives=..., notes=...)`.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        todo!()
    }

    /// Pickle as a call of the class with its fields.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
    }

    /// Return the V2 payload `{"identifier", "alternatives", "notes"}`.
    ///
    /// Raises `SerializationError` inside `wire_version(WireVersion.V1)`.
    fn serialize_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return the canonical V2 text of the payload, re-formatted for
    /// `indent` or `sort_keys`.
    #[pyo3(signature = (*, indent = None, sort_keys = None))]
    fn to_json(
        slf: &Bound<'_, Self>,
        indent: Option<&Bound<'_, PyAny>>,
        sort_keys: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        todo!()
    }

    /// Return the choice of the V2 payload `data`.
    #[classmethod]
    fn deserialize_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return the choice of the JSON text `payload`.
    #[classmethod]
    fn from_json<'py>(
        cls: &Bound<'py, PyType>,
        payload: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }
}

/// Return the core choice of the `Choice` object `object`, sharing it.
///
/// # Errors
///
/// Raises `TypeError` for an object that is no `Choice`.
pub(crate) fn choice_from_python(object: &Bound<'_, PyAny>) -> PyResult<Choice> {
    todo!()
}

/// Return a new public `Choice` of `choice`, over the objects of its parts.
///
/// # Errors
///
/// Raises what building the object of a part raises.
pub(crate) fn choice_to_python<'py>(
    py: Python<'py>,
    choice: &Choice,
) -> PyResult<Bound<'py, PyAny>> {
    todo!()
}
