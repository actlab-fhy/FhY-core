//! `fhy_core._rs.TraceStep` and `Trace`: the steps of a run, recorded.
//!
//! A trace recorded in this process keeps the objects its steps were
//! answered with, and a step returns them; a trace read back holds the
//! decoded values, and none for an opaque one.

#![expect(
    unused_variables,
    dead_code,
    reason = "interface stub: the bodies are todo!() until the implementation"
)]

use pyo3::prelude::*;
use pyo3::pyclass::{CompareOp, PyTraverseError, PyVisit};
use pyo3::types::{PyDict, PyTuple, PyType};

use fhy_core::search_space::{Trace, TraceStep};

use crate::util::public_class::PublicClass;

/// One recorded step, backed by the core [`TraceStep`].
#[pyclass(frozen, module = "fhy_core._rs", name = "TraceStep")]
pub(crate) struct PyTraceStep {
    step: TraceStep,
    /// The object answered, for a step recorded in this process.
    value: Option<Py<PyAny>>,
}

#[pymethods]
impl PyTraceStep {
    /// What the step is about, a `str`.
    #[getter]
    fn kind(&self) -> String {
        todo!()
    }

    /// The step's subject, an `Identifier`: a static step's decision name.
    #[getter]
    fn subject(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        todo!()
    }

    /// A static step's canonical position in its space, or `None`.
    #[getter]
    fn decision(&self) -> Option<usize> {
        todo!()
    }

    /// The number of values the step's domain held, an `int`.
    #[getter]
    fn cardinality(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        todo!()
    }

    /// The answer: an `int`, or a tuple of `int`s for an order.
    #[getter]
    fn coordinate(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        todo!()
    }

    /// The object answered, the decoded value, or `None`.
    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        todo!()
    }

    /// The signature of the domain, as its V2 text.
    #[getter]
    fn signature(&self) -> PyResult<String> {
        todo!()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        todo!()
    }

    /// Visit the objects held, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(value) = &self.value {
            visit.call(value)?;
        }
        Ok(())
    }
}

/// The steps of a run, backed by the core [`Trace`]; the base of the public
/// `Trace`.
///
/// `==` and `hash` are structural, over the core steps.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Trace")]
pub(crate) struct PyTrace {
    trace: Trace,
    /// The `TraceStep` objects, in order.
    steps: Py<PyTuple>,
}

impl PyTrace {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("Trace");
        &PUBLIC_CLASS
    }

    /// Return the core trace.
    pub(crate) const fn core(&self) -> &Trace {
        &self.trace
    }
}

#[pymethods]
impl PyTrace {
    /// Return the trace of the `TraceStep`s `steps`, in order.
    ///
    /// Raises `TypeError` for an argument that is no iterable of
    /// `TraceStep`s.
    #[new]
    #[pyo3(signature = (steps = None, **kwargs))]
    fn new(steps: Option<&Bound<'_, PyAny>>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        todo!()
    }

    /// Register `cls` as the public class the binding instantiates.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }

    /// The steps, in the order asked.
    #[getter]
    fn steps<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        todo!()
    }

    /// The steps' coordinates, in the order asked.
    #[getter]
    fn coordinates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
    }

    /// The product of the steps' cardinalities, an `int`; 1 for no step.
    #[getter]
    fn traversed_cardinality(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        todo!()
    }

    /// Return the steps of the kind `kind`, a `str`, in the order asked.
    fn of_kind<'py>(&self, py: Python<'py>, kind: &str) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
    }

    fn __len__(&self) -> usize {
        todo!()
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        todo!()
    }

    fn __str__(&self) -> String {
        todo!()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        todo!()
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Py<PyAny>> {
        todo!()
    }

    fn __hash__(&self) -> u64 {
        todo!()
    }

    /// Return the V2 dict of the trace.
    fn serialize_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return the V2 JSON text of the trace.
    #[pyo3(signature = (*, indent = None, sort_keys = None))]
    fn to_json(&self, indent: Option<usize>, sort_keys: Option<bool>) -> PyResult<String> {
        todo!()
    }

    /// Return the trace of the V2 dict `data`, an instance of `cls`.
    ///
    /// Raises `DeserializationValueError` for a payload of another shape or
    /// a coordinate outside its signature.
    #[classmethod]
    fn deserialize_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return the trace of the V2 JSON text `payload`, an instance of
    /// `cls`.
    #[classmethod]
    fn from_json<'py>(cls: &Bound<'py, PyType>, payload: &str) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Pickle as a call of `from_json` with the V2 text.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
    }

    /// Visit the objects held, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.steps)
    }
}
