//! `fhy_core._rs.Objective` and `Measurement`: what a search measures and
//! the record of one measured configuration; the bases of the public
//! classes registered as `search_space.objective` and
//! `search_space.measurement`.
//!
//! A direction is read as the public `Direction` (a `str` enum) or its
//! value, and written as the `Direction` member; a status is written as
//! the `MeasurementStatus` member. A value is an `int` or a `float`; a
//! `bool` is refused.

#![expect(
    unused_variables,
    dead_code,
    reason = "interface stub: the bodies are todo!() until the implementation"
)]

use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::{PyDict, PyTuple, PyType};

use fhy_core::search_space::{Direction, Measurement, Objective};

use crate::util::public_class::PublicClass;

/// Return the core direction of `direction`, a `Direction` or its value.
///
/// # Errors
///
/// Raises `TypeError` for an object that is no `str`, and `ValueError` for
/// another string.
pub(super) fn read_direction(direction: &Bound<'_, PyAny>) -> PyResult<Direction> {
    todo!()
}

/// Return the public `Direction` member of `direction`.
///
/// # Errors
///
/// Raises what importing `fhy_core.search_space` raises.
pub(super) fn direction_to_python(
    py: Python<'_>,
    direction: Direction,
) -> PyResult<Bound<'_, PyAny>> {
    todo!()
}

/// A named quantity a search measures and which way it is better, backed by
/// the core [`Objective`]; the base of the public `Objective`.
///
/// `==` and `hash` are structural: the name and the direction.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Objective")]
pub(crate) struct PyObjective {
    objective: Objective,
}

impl PyObjective {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("Objective");
        &PUBLIC_CLASS
    }

    /// Return the core objective.
    pub(crate) const fn core(&self) -> &Objective {
        &self.objective
    }
}

/// Return a new public `Objective` of `objective`.
///
/// # Errors
///
/// Raises what building the object raises.
pub(super) fn objective_to_python<'py>(
    py: Python<'py>,
    objective: &Objective,
) -> PyResult<Bound<'py, PyAny>> {
    todo!()
}

#[pymethods]
impl PyObjective {
    /// Return the objective `name`, better in `direction`.
    ///
    /// Raises `TypeError` for a name that is no `str` or a direction that
    /// is no `Direction` or `str`, `ValueError` for an unknown direction,
    /// and `MeasurementError` for an empty name.
    #[new]
    #[pyo3(signature = (name = None, direction = None, **kwargs))]
    fn new(
        name: Option<&Bound<'_, PyAny>>,
        direction: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        todo!()
    }

    /// Register `cls` as the public class the binding instantiates.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }

    /// The objective's name, a `str`.
    #[getter]
    fn name(&self) -> String {
        todo!()
    }

    /// Which way the objective's values are better, a `Direction`.
    #[getter]
    fn direction<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return `1` when the value `left` is better than `right`, `-1` when
    /// it is worse, `0` when neither is, and `None` for a `REPORT`
    /// objective. A NaN is worse than every number and equal to another
    /// NaN.
    ///
    /// Raises `TypeError` for a value that is no `int` or `float` (a
    /// `bool` included).
    fn compare(&self, left: &Bound<'_, PyAny>, right: &Bound<'_, PyAny>) -> PyResult<Option<i8>> {
        todo!()
    }

    /// Compare structurally with another objective; another type is
    /// `NotImplemented`.
    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Py<PyAny>> {
        todo!()
    }

    /// Return the objective's hash, consistent with `==`.
    fn __hash__(&self) -> u64 {
        todo!()
    }

    /// Return `Objective(name=..., direction=...)`.
    fn __repr__(&self) -> String {
        todo!()
    }

    /// Refuse to set an attribute: the objective is frozen.
    fn __setattr__(slf: &Bound<'_, Self>, name: &str, _value: &Bound<'_, PyAny>) -> PyResult<()> {
        todo!()
    }

    /// Refuse to delete an attribute: the objective is frozen.
    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        todo!()
    }

    /// Return the V2 dict `{"name", "direction"}`.
    fn serialize_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return the canonical V2 text, re-formatted for `indent` or
    /// `sort_keys`.
    #[pyo3(signature = (*, indent = None, sort_keys = None))]
    fn to_json(
        slf: &Bound<'_, Self>,
        indent: Option<&Bound<'_, PyAny>>,
        sort_keys: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        todo!()
    }

    /// Return the objective of the V2 dict `data`, an instance of `cls`.
    ///
    /// Raises `DeserializationValueError` for a payload of another shape or
    /// an empty name.
    #[classmethod]
    fn deserialize_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return the objective of the V2 JSON text `payload`, an instance of
    /// `cls`.
    #[classmethod]
    fn from_json<'py>(
        cls: &Bound<'py, PyType>,
        payload: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Pickle as a call of `from_json` with the V2 text.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
    }
}

/// The record of one measured configuration, backed by the core
/// [`Measurement`]; the base of the public `Measurement`.
///
/// `==` and `hash` are identity.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Measurement")]
pub(crate) struct PyMeasurement {
    measurement: Measurement,
}

impl PyMeasurement {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("Measurement");
        &PUBLIC_CLASS
    }

    /// Return the core measurement.
    pub(crate) const fn core(&self) -> &Measurement {
        &self.measurement
    }
}

#[pymethods]
impl PyMeasurement {
    /// Refuse direct construction: use `ok`, `infeasible`, `failed` or
    /// `timeout`.
    ///
    /// Raises `TypeError` unless the binding seeds the instance.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        todo!()
    }

    /// Register `cls` as the public class the binding instantiates.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }

    /// Return the successful measurement of the configuration whose key is
    /// `key`, a value per objective: `values` a mapping of `Objective` to
    /// value, or `(Objective, value)` pairs, in order.
    ///
    /// Raises `TypeError` for a key that is no `ConfigurationKey`, an
    /// objective that is no `Objective` or a value that is no `int` or
    /// `float` (a `bool` included), and `MeasurementError` for no value, a
    /// repeated objective name or a value that is not finite.
    #[classmethod]
    fn ok<'py>(
        cls: &Bound<'py, PyType>,
        key: &Bound<'py, PyAny>,
        values: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return the measurement of a configuration that cannot be realized,
    /// for `reason`, a `str`.
    #[classmethod]
    fn infeasible<'py>(
        cls: &Bound<'py, PyType>,
        key: &Bound<'py, PyAny>,
        reason: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return the measurement of a configuration that broke, for `reason`,
    /// a `str`.
    #[classmethod]
    fn failed<'py>(
        cls: &Bound<'py, PyType>,
        key: &Bound<'py, PyAny>,
        reason: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return the measurement of a configuration that ran out of time.
    #[classmethod]
    fn timeout<'py>(
        cls: &Bound<'py, PyType>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// The key of the configuration measured, a `ConfigurationKey`.
    #[getter]
    fn key(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        todo!()
    }

    /// How the measurement went, a `MeasurementStatus`.
    #[getter]
    fn status<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Why an infeasible or failed measurement did not succeed, or `None`.
    #[getter]
    fn reason(&self) -> Option<String> {
        todo!()
    }

    /// Return whether the measurement succeeded.
    fn is_ok(&self) -> bool {
        todo!()
    }

    /// The values, a `dict` of `Objective` to `float` in the order given;
    /// empty for a measurement that did not succeed.
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        todo!()
    }

    /// Return the value of `objective`, an `Objective` (matched by name
    /// and direction) or a name, or `None`.
    ///
    /// Raises `TypeError` for another argument.
    fn value(&self, objective: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
        todo!()
    }

    /// The `Note`s attached to the measurement.
    #[getter]
    fn notes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
    }

    /// Return the measurement with the `Note`s `notes` in place of its own.
    ///
    /// Raises `TypeError` for an argument that is no iterable of `Note`s.
    fn with_notes<'py>(
        slf: &Bound<'py, Self>,
        notes: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return whether this measurement dominates `other`, a `Measurement`.
    ///
    /// Raises `TypeError` for another argument, and `MeasurementError` for
    /// a measurement that did not succeed or measurements over different
    /// objectives.
    fn dominates(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        todo!()
    }

    /// Return `Measurement(status=..., values=...)`.
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        todo!()
    }

    /// Refuse to set an attribute: the measurement is frozen.
    fn __setattr__(slf: &Bound<'_, Self>, name: &str, _value: &Bound<'_, PyAny>) -> PyResult<()> {
        todo!()
    }

    /// Refuse to delete an attribute: the measurement is frozen.
    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        todo!()
    }

    /// Return the V2 dict `{"key", "status", "values", "notes"}`.
    fn serialize_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return the canonical V2 text, re-formatted for `indent` or
    /// `sort_keys`.
    #[pyo3(signature = (*, indent = None, sort_keys = None))]
    fn to_json(
        slf: &Bound<'_, Self>,
        indent: Option<&Bound<'_, PyAny>>,
        sort_keys: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        todo!()
    }

    /// Return the measurement of the V2 dict `data`, an instance of `cls`.
    ///
    /// Raises `DeserializationValueError` for a payload of another shape or
    /// one a constructor refuses.
    #[classmethod]
    fn deserialize_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return the measurement of the V2 JSON text `payload`, an instance of
    /// `cls`.
    #[classmethod]
    fn from_json<'py>(
        cls: &Bound<'py, PyType>,
        payload: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Pickle as a call of `from_json` with the V2 text.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
    }
}
