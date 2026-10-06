//! `fhy_core._rs.Objective` and `Measurement`: what a search measures and
//! the record of one measured configuration; the bases of the public
//! classes registered as `search_space.objective` and
//! `search_space.measurement`.
//!
//! A direction is read as the public `Direction` (a `str` enum) or its
//! value, and written as the `Direction` member; a status is written as
//! the `MeasurementStatus` member. A value is an `int` or a `float`; a
//! `bool` is refused. A measurement holds only the core measurement: its
//! notes are read into core notes and written as new `Note` objects.

use std::cmp::Ordering;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyMapping, PyString, PyTuple, PyType};

use fhy_core::search_space::wire::MeasurementData;
use fhy_core::search_space::{Direction, Measurement, MeasurementStatus, Objective};

use crate::diagnostic::note_to_python;
use crate::util::dataclass::hash_value;
use crate::util::frozen::{refuse_attribute_assignment, refuse_attribute_deletion};
use crate::util::pending::with_pending_errors;
use crate::util::public_class::PublicClass;
use crate::util::python::read_type_name;
use crate::wire::{
    PyResolver, build, check_instance, parse_tree, read_text, read_text_tree, read_tree, to_json,
};

use super::arguments::{Seeded, instantiate, read_notes, take_seed, wrong_seed};
use super::configuration::PyConfigurationKey;
use super::errors::measurement_error_to_py;
use super::wire::{refuse_v1, write_part, write_part_json};

/// The module of the public classes.
const PUBLIC_MODULE: &str = "fhy_core.search_space.core";

/// Return the core direction of `direction`, a `Direction` or its value.
///
/// # Errors
///
/// Raises `TypeError` for an object that is no `str`, and `ValueError` for
/// another string.
fn read_direction(direction: &Bound<'_, PyAny>) -> PyResult<Direction> {
    let text = direction.cast::<PyString>().map_err(|_not_text| {
        PyTypeError::new_err(format!(
            "a direction must be a Direction or its value, got {}.",
            read_type_name(direction)
        ))
    })?;
    match text.to_str()? {
        "minimize" => Ok(Direction::Minimize),
        "maximize" => Ok(Direction::Maximize),
        "report" => Ok(Direction::Report),
        other => Err(PyValueError::new_err(format!(
            "{other:?} is no direction: use minimize, maximize or report"
        ))),
    }
}

/// Return the public `Direction` member of `direction`.
///
/// # Errors
///
/// Raises what importing `fhy_core.search_space` raises.
fn direction_to_python(
    py: Python<'_>,
    direction: Direction,
) -> PyResult<Bound<'_, PyAny>> {
    crate::cached_attr!(py, PUBLIC_MODULE, "Direction")?.call1((direction.as_str(),))
}

/// Return the value `value`, an `int` or a `float`.
///
/// # Errors
///
/// Raises `TypeError` for anything else, a `bool` included, and what
/// converting a large `int` raises.
fn read_number(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    if let Ok(float) = value.cast::<PyFloat>() {
        return Ok(float.value());
    }
    if value.is_instance_of::<PyInt>() && !value.is_instance_of::<PyBool>() {
        return value.extract();
    }
    Err(PyTypeError::new_err(format!(
        "a value must be an int or a float, got {}.",
        read_type_name(value)
    )))
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
    const fn core(&self) -> &Objective {
        &self.objective
    }
}

/// Return a new public `Objective` of `objective`.
///
/// # Errors
///
/// Raises what building the object raises.
fn objective_to_python<'py>(
    py: Python<'py>,
    objective: &Objective,
) -> PyResult<Bound<'py, PyAny>> {
    instantiate(
        PyObjective::public_class().get(py)?,
        0,
        Seeded::Objective(PyObjective {
            objective: objective.clone(),
        }),
    )
}

/// Return the object of `objective`, an instance of `cls`.
///
/// # Errors
///
/// Raises what building the object raises, and `SerializationError` for
/// an object that is no instance of `cls`.
fn objective_of_class<'py>(
    cls: &Bound<'py, PyType>,
    objective: Objective,
) -> PyResult<Bound<'py, PyAny>> {
    check_instance(
        cls,
        instantiate(cls, 0, Seeded::Objective(PyObjective { objective }))?,
    )
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
        if let Some(seeded) = take_seed(kwargs)? {
            return match seeded {
                Seeded::Objective(objective) => Ok(objective),
                _ => Err(wrong_seed()),
            };
        }
        let (Some(name), Some(direction)) = (name, direction) else {
            return Err(PyTypeError::new_err(
                "Objective takes a name and a direction",
            ));
        };
        let py = name.py();
        let name = name.cast::<PyString>().map_err(|_not_text| {
            PyTypeError::new_err(format!(
                "an objective's name must be a str, got {}.",
                read_type_name(name)
            ))
        })?;
        let direction = read_direction(direction)?;
        let objective = Objective::new(name.to_str()?, direction)
            .map_err(|error| measurement_error_to_py(py, &error))?;
        Ok(Self { objective })
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
        self.objective.name().to_owned()
    }

    /// Which way the objective's values are better, a `Direction`.
    #[getter]
    fn direction<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        direction_to_python(py, self.objective.direction())
    }

    /// Return `1` when the value `left` is better than `right`, `-1` when
    /// it is worse, `0` when neither is, and `None` for a `REPORT`
    /// objective. A NaN is worse than every number and equal to another
    /// NaN.
    ///
    /// Raises `TypeError` for a value that is no `int` or `float` (a
    /// `bool` included).
    fn compare(&self, left: &Bound<'_, PyAny>, right: &Bound<'_, PyAny>) -> PyResult<Option<i8>> {
        let ordering = self
            .objective
            .compare(read_number(left)?, read_number(right)?);
        Ok(ordering.map(|ordering| match ordering {
            Ordering::Greater => 1,
            Ordering::Equal => 0,
            Ordering::Less => -1,
        }))
    }

    /// Compare structurally with another objective; another type is
    /// `NotImplemented`.
    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> Py<PyAny> {
        let py = other.py();
        let Ok(other) = other.cast::<Self>() else {
            return py.NotImplemented();
        };
        let equal = self.objective == other.get().objective;
        match op {
            CompareOp::Eq => PyBool::new(py, equal).to_owned().into_any().unbind(),
            CompareOp::Ne => PyBool::new(py, !equal).to_owned().into_any().unbind(),
            _ => py.NotImplemented(),
        }
    }

    /// Return the objective's hash, consistent with `==`.
    fn __hash__(&self) -> u64 {
        hash_value(&self.objective)
    }

    /// Return `Objective(name=..., direction=...)`.
    fn __repr__(&self) -> String {
        format!(
            "Objective(name={:?}, direction={})",
            self.objective.name(),
            self.objective.direction()
        )
    }

    /// Refuse to set an attribute: the objective is frozen.
    fn __setattr__(slf: &Bound<'_, Self>, name: &str, _value: &Bound<'_, PyAny>) -> PyResult<()> {
        refuse_attribute_assignment(slf.as_any(), name)
    }

    /// Refuse to delete an attribute: the objective is frozen.
    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        refuse_attribute_deletion(slf.as_any(), name)
    }

    /// Return the V2 dict `{"name", "direction"}`.
    ///
    /// Raises `SerializationError` inside `wire_version(WireVersion.V1)`.
    fn serialize_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        refuse_v1(slf.as_any())?;
        write_part(slf.py(), || Ok(&slf.get().objective))
    }

    /// Return the canonical V2 text, re-formatted for `indent` or
    /// `sort_keys`.
    #[pyo3(signature = (*, indent = None, sort_keys = None))]
    fn to_json(
        slf: &Bound<'_, Self>,
        indent: Option<&Bound<'_, PyAny>>,
        sort_keys: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        refuse_v1(slf.as_any())?;
        write_part_json(slf.as_any(), indent, sort_keys, || Ok(&slf.get().objective))
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
        let objective: Objective = parse_tree(cls, read_tree(cls, data)?)?;
        objective_of_class(cls, objective)
    }

    /// Return the objective of the V2 JSON text `payload`, an instance of
    /// `cls`.
    #[classmethod]
    fn from_json<'py>(
        cls: &Bound<'py, PyType>,
        payload: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let objective: Objective = parse_tree(cls, read_text_tree(cls, &read_text(payload)?)?)?;
        objective_of_class(cls, objective)
    }

    /// Pickle as a call of `from_json` with the V2 text.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let text = to_json(py, &slf.get().objective)?;
        let restore = slf.get_type().getattr(intern!(py, "from_json"))?;
        PyTuple::new(py, [restore, PyTuple::new(py, [text])?.into_any()])
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
    const fn core(&self) -> &Measurement {
        &self.measurement
    }
}

/// Return the object of `measurement`, an instance of `cls`.
///
/// # Errors
///
/// Raises what building the object raises, and `SerializationError` for
/// an object that is no instance of `cls`.
fn measurement_of_class<'py>(
    cls: &Bound<'py, PyType>,
    measurement: Measurement,
) -> PyResult<Bound<'py, PyAny>> {
    check_instance(
        cls,
        instantiate(cls, 0, Seeded::Measurement(PyMeasurement { measurement }))?,
    )
}

/// Return the core key of `key`, a `ConfigurationKey`.
///
/// # Errors
///
/// Raises `TypeError` for another object.
fn read_key(key: &Bound<'_, PyAny>) -> PyResult<fhy_core::search_space::ConfigurationKey> {
    key.cast::<PyConfigurationKey>()
        .map(|key| key.get().core().clone())
        .map_err(|_not_a_key| {
            PyTypeError::new_err(format!(
                "a measurement's key must be a ConfigurationKey, got {}.",
                read_type_name(key)
            ))
        })
}

/// Return the reason `reason`, a `str`.
///
/// # Errors
///
/// Raises `TypeError` for another object.
fn read_reason(reason: &Bound<'_, PyAny>) -> PyResult<String> {
    reason
        .cast::<PyString>()
        .map_err(|_not_text| {
            PyTypeError::new_err(format!(
                "a reason must be a str, got {}.",
                read_type_name(reason)
            ))
        })?
        .to_str()
        .map(str::to_owned)
}

/// Return the values `values` gives: a mapping of `Objective` to value, or
/// `(Objective, value)` pairs, in order.
///
/// # Errors
///
/// Raises `TypeError` for values of another shape.
fn read_values(values: &Bound<'_, PyAny>) -> PyResult<Vec<(Objective, f64)>> {
    let expected = "a mapping of Objective to value, or (Objective, value) pairs";
    let wrong = |object: &Bound<'_, PyAny>| {
        PyTypeError::new_err(format!(
            "a measurement's values must be {expected}, got {}.",
            read_type_name(object)
        ))
    };
    let pairs = match values.cast::<PyMapping>() {
        Ok(mapping) => mapping.items()?.into_any(),
        Err(_not_a_mapping) => values.clone(),
    };
    pairs
        .try_iter()
        .map_err(|_not_iterable| wrong(values))?
        .map(|item| {
            let item = item?;
            let (objective, value): (Bound<'_, PyAny>, Bound<'_, PyAny>) =
                item.extract().map_err(|_not_a_pair| wrong(&item))?;
            let objective = objective
                .cast::<PyObjective>()
                .map_err(|_not_an_objective| wrong(&objective))?
                .get()
                .core()
                .clone();
            Ok((objective, read_number(&value)?))
        })
        .collect()
}

#[pymethods]
impl PyMeasurement {
    /// Refuse direct construction: use `ok`, `infeasible`, `failed` or
    /// `timeout`.
    ///
    /// Raises `TypeError` unless the binding seeds the instance.
    #[new]
    #[pyo3(signature = (*_args, **kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        match take_seed(kwargs)? {
            Some(Seeded::Measurement(measurement)) => Ok(measurement),
            Some(_) => Err(wrong_seed()),
            None => Err(PyTypeError::new_err(
                "Measurement has no constructor: use Measurement.ok, infeasible, failed or timeout",
            )),
        }
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
        let py = cls.py();
        let measurement = Measurement::ok(read_key(key)?, read_values(values)?)
            .map_err(|error| measurement_error_to_py(py, &error))?;
        measurement_of_class(cls, measurement)
    }

    /// Return the measurement of a configuration that cannot be realized,
    /// for `reason`, a `str`.
    #[classmethod]
    fn infeasible<'py>(
        cls: &Bound<'py, PyType>,
        key: &Bound<'py, PyAny>,
        reason: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let measurement = Measurement::infeasible(read_key(key)?, read_reason(reason)?);
        measurement_of_class(cls, measurement)
    }

    /// Return the measurement of a configuration that broke, for `reason`,
    /// a `str`.
    #[classmethod]
    fn failed<'py>(
        cls: &Bound<'py, PyType>,
        key: &Bound<'py, PyAny>,
        reason: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let measurement = Measurement::failed(read_key(key)?, read_reason(reason)?);
        measurement_of_class(cls, measurement)
    }

    /// Return the measurement of a configuration that ran out of time.
    #[classmethod]
    fn timeout<'py>(
        cls: &Bound<'py, PyType>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        measurement_of_class(cls, Measurement::timeout(read_key(key)?))
    }

    /// The key of the configuration measured, a `ConfigurationKey`.
    #[getter]
    fn key(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(Py::new(py, PyConfigurationKey::of(self.measurement.key().clone()))?.into_any())
    }

    /// How the measurement went, a `MeasurementStatus`.
    #[getter]
    fn status<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let text = match self.measurement.status() {
            MeasurementStatus::Infeasible { .. } => "infeasible",
            MeasurementStatus::Failed { .. } => "failed",
            MeasurementStatus::Timeout => "timeout",
            _ => "ok",
        };
        crate::cached_attr!(py, PUBLIC_MODULE, "MeasurementStatus")?.call1((text,))
    }

    /// Why an infeasible or failed measurement did not succeed, or `None`.
    #[getter]
    fn reason(&self) -> Option<String> {
        match self.measurement.status() {
            MeasurementStatus::Infeasible { reason } | MeasurementStatus::Failed { reason } => {
                Some(reason.clone())
            }
            _ => None,
        }
    }

    /// Return whether the measurement succeeded.
    fn is_ok(&self) -> bool {
        self.measurement.is_ok()
    }

    /// The values, a `dict` of `Objective` to `float` in the order given;
    /// empty for a measurement that did not succeed.
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let values = PyDict::new(py);
        for (objective, value) in self.measurement.values() {
            values.set_item(objective_to_python(py, objective)?, value)?;
        }
        Ok(values)
    }

    /// Return the value of `objective`, an `Objective` (matched by name
    /// and direction) or a name, or `None`.
    ///
    /// Raises `TypeError` for another argument.
    fn value(&self, objective: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
        if let Ok(name) = objective.cast::<PyString>() {
            return Ok(self.measurement.value(name.to_str()?));
        }
        let objective = objective
            .cast::<PyObjective>()
            .map_err(|_not_an_objective| {
                PyTypeError::new_err(format!(
                    "value takes an Objective or a name, got {}.",
                    read_type_name(objective)
                ))
            })?;
        let wanted = objective.get().core();
        Ok(self
            .measurement
            .values()
            .iter()
            .find(|(held, _)| held == wanted)
            .map(|(_, value)| *value))
    }

    /// The `Note`s attached to the measurement.
    #[getter]
    fn notes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let notes = self
            .measurement
            .notes()
            .iter()
            .map(|note| note_to_python(py, note))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, notes)
    }

    /// Return the measurement with the `Note`s `notes` in place of its own.
    ///
    /// Raises `TypeError` for an argument that is no iterable of `Note`s.
    fn with_notes<'py>(
        slf: &Bound<'py, Self>,
        notes: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (notes, _) = read_notes(slf.py(), Some(notes), "Measurement.with_notes")?;
        let measurement = slf.get().measurement.clone().with_notes(notes);
        measurement_of_class(&slf.get_type(), measurement)
    }

    /// Return whether this measurement dominates `other`, a `Measurement`.
    ///
    /// Raises `TypeError` for another argument, and `MeasurementError` for
    /// a measurement that did not succeed or measurements over different
    /// objectives.
    fn dominates(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = other.py();
        let other = other.cast::<Self>().map_err(|_not_a_measurement| {
            PyTypeError::new_err(format!(
                "dominates takes a Measurement, got {}.",
                read_type_name(other)
            ))
        })?;
        self.measurement
            .dominates(other.get().core())
            .map_err(|error| measurement_error_to_py(py, &error))
    }

    /// Return `Measurement(status=..., ...)`.
    fn __repr__(&self) -> String {
        match self.measurement.status() {
            MeasurementStatus::Infeasible { reason } => {
                format!("Measurement(status=infeasible, reason={reason:?})")
            }
            MeasurementStatus::Failed { reason } => {
                format!("Measurement(status=failed, reason={reason:?})")
            }
            MeasurementStatus::Timeout => "Measurement(status=timeout)".to_owned(),
            _ => {
                let values = self
                    .measurement
                    .values()
                    .iter()
                    .map(|(objective, value)| format!("{:?}: {value:?}", objective.name()))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("Measurement(status=ok, values={{{values}}})")
            }
        }
    }

    /// Refuse to set an attribute: the measurement is frozen.
    fn __setattr__(slf: &Bound<'_, Self>, name: &str, _value: &Bound<'_, PyAny>) -> PyResult<()> {
        refuse_attribute_assignment(slf.as_any(), name)
    }

    /// Refuse to delete an attribute: the measurement is frozen.
    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        refuse_attribute_deletion(slf.as_any(), name)
    }

    /// Return the V2 dict `{"key", "status", "values", "notes"}`.
    ///
    /// Raises `SerializationError` inside `wire_version(WireVersion.V1)`.
    fn serialize_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        refuse_v1(slf.as_any())?;
        write_part(slf.py(), || MeasurementData::of(&slf.get().measurement))
    }

    /// Return the canonical V2 text, re-formatted for `indent` or
    /// `sort_keys`.
    #[pyo3(signature = (*, indent = None, sort_keys = None))]
    fn to_json(
        slf: &Bound<'_, Self>,
        indent: Option<&Bound<'_, PyAny>>,
        sort_keys: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        refuse_v1(slf.as_any())?;
        write_part_json(slf.as_any(), indent, sort_keys, || {
            MeasurementData::of(&slf.get().measurement)
        })
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
        let data: MeasurementData = parse_tree(cls, read_tree(cls, data)?)?;
        let measurement = build(cls, || data.build(&PyResolver))?;
        measurement_of_class(cls, measurement)
    }

    /// Return the measurement of the V2 JSON text `payload`, an instance of
    /// `cls`.
    #[classmethod]
    fn from_json<'py>(
        cls: &Bound<'py, PyType>,
        payload: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let data: MeasurementData = parse_tree(cls, read_text_tree(cls, &read_text(payload)?)?)?;
        let measurement = build(cls, || data.build(&PyResolver))?;
        measurement_of_class(cls, measurement)
    }

    /// Pickle as a call of `from_json` with the V2 text.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let text = with_pending_errors(|| {
            let data = MeasurementData::of(&slf.get().measurement)
                .map_err(|error| crate::wire::foreign_error(py, &error))?;
            to_json(py, &data)
        })?;
        let restore = slf.get_type().getattr(intern!(py, "from_json"))?;
        PyTuple::new(py, [restore, PyTuple::new(py, [text])?.into_any()])
    }
}
