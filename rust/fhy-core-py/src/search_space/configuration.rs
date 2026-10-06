//! `fhy_core._rs.Configuration` and `ConfigurationKey`.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::pyclass::{CompareOp, PyTraverseError, PyVisit};
use pyo3::types::{PyTuple, PyType};

use fhy_core::identifier::Identifier;
use fhy_core::search_space::{Configuration, ConfigurationKey};

use crate::util::public_class::PublicClass;

/// A point of a space, checked against it, backed by the core
/// [`Configuration`]; the base of the public `Configuration`.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Configuration")]
pub(crate) struct PyConfiguration {
    configuration: Configuration,
    /// The `Space` the configuration is a point of.
    space: Py<PyAny>,
    /// The value objects given, by decision name.
    values: HashMap<Identifier, Py<PyAny>>,
}

impl PyConfiguration {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("Configuration");
        &PUBLIC_CLASS
    }
}

#[pymethods]
impl PyConfiguration {
    /// Return the configuration of `space` holding `entries`, a mapping or
    /// an iterable of `(name, value)` pairs, a choice's value the chosen
    /// alternative's name, checked under the default solver's context.
    ///
    /// Raises `TypeError` for an argument of the wrong type,
    /// `ConfigurationError` carrying every problem of the entries, and the
    /// exception a Python-defined constraint raised.
    #[new]
    #[pyo3(signature = (space, entries = None))]
    fn new(space: &Bound<'_, PyAny>, entries: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        todo!()
    }

    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.space)?;
        for value in self.values.values() {
            visit.call(value)?;
        }
        Ok(())
    }

    /// Register `cls` as the public `Configuration` class.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }

    /// The `Space` the configuration is a point of.
    #[getter]
    fn space<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        todo!()
    }

    /// The `(name, value)` entries, in canonical order of the decisions.
    #[getter]
    fn entries<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
    }

    /// Return the value of the decision `name`, or `None` when it has none.
    ///
    /// Raises `TypeError` if `name` is not an `Identifier`.
    fn value<'py>(
        slf: &Bound<'py, Self>,
        name: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        todo!()
    }

    /// Return the alternative the choice `choice` chose, or `None`.
    ///
    /// Raises `TypeError` if `choice` is not an `Identifier`.
    fn alternative<'py>(
        slf: &Bound<'py, Self>,
        choice: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        todo!()
    }

    /// Return the `Activity` of the decision `name`, or `None` for a name
    /// the space lacks.
    ///
    /// Raises `TypeError` if `name` is not an `Identifier`.
    fn activity<'py>(
        &self,
        py: Python<'py>,
        name: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        todo!()
    }

    /// Return whether every decision is assigned or inactive.
    fn is_complete(&self) -> bool {
        todo!()
    }

    /// Return the key that identifies the configuration within its space.
    fn key(&self) -> PyConfigurationKey {
        todo!()
    }

    /// Return the configuration with `value` for the decision `name`, in
    /// place of its value if it has one, checked as a whole.
    ///
    /// Raises as the constructor does.
    fn with_entry<'py>(
        slf: &Bound<'py, Self>,
        name: &Bound<'py, PyAny>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return the configuration with `entries` in place of or beside its
    /// entries, checked as a whole.
    ///
    /// Raises as the constructor does.
    fn with_entries<'py>(
        slf: &Bound<'py, Self>,
        entries: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return whether `other` is a configuration of a structurally
    /// equivalent space with equal values.
    fn is_structurally_equivalent(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        todo!()
    }

    /// Return whether `other` is a configuration of an alpha-equivalent
    /// space whose values correspond, under no renaming.
    fn is_alpha_equivalent(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        todo!()
    }

    /// Return whether `other` is a configuration of an alpha-equivalent
    /// space whose values correspond, under `renaming`.
    fn is_alpha_equivalent_under(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        renaming: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        todo!()
    }

    /// Refuse to set an attribute: the configuration is frozen.
    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        todo!()
    }

    /// Refuse to delete an attribute: the configuration is frozen.
    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        todo!()
    }

    /// Return `Configuration(space=<space name>, entries=...)`.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        todo!()
    }

    /// Pickle as a call of the class with its space and entries.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
    }

    /// Return the V2 payload `{"space", "entries"}`.
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

    /// Return the configuration of the V2 payload `data`, its values
    /// checked as a restored assignment is.
    #[classmethod]
    fn deserialize_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return the configuration of the JSON text `payload`.
    #[classmethod]
    fn from_json<'py>(
        cls: &Bound<'py, PyType>,
        payload: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }
}

/// The identity of a configuration within its space, backed by the core
/// [`ConfigurationKey`]: equal for configurations of alpha-equivalent
/// spaces whose values correspond.
///
/// It is hashable and compares structurally, so it keys a dict. It has no
/// constructor; it pickles through its wire form, which is self-contained:
/// each identifier the space binds is written as its position among the
/// space's names.
#[pyclass(frozen, module = "fhy_core._rs", name = "ConfigurationKey")]
pub(crate) struct PyConfigurationKey {
    key: ConfigurationKey,
}

#[pymethods]
impl PyConfigurationKey {
    /// Compare structurally with another key; another type is
    /// `NotImplemented`.
    fn __richcmp__<'py>(
        &self,
        other: &Bound<'py, PyAny>,
        op: CompareOp,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return the key's hash, consistent with `==`.
    fn __hash__(&self) -> u64 {
        todo!()
    }

    /// Return `ConfigurationKey(...)`.
    fn __repr__(&self) -> String {
        todo!()
    }

    /// Pickle as a call of `_from_wire` with the key's V2 text.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
    }

    /// Return the key of its V2 text `text`, the inverse of the text
    /// `__reduce__` writes.
    ///
    /// Raises `DeserializationValueError` for a text of another shape.
    #[staticmethod]
    fn _from_wire(text: &str) -> PyResult<Self> {
        todo!()
    }
}
