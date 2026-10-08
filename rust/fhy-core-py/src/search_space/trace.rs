//! `fhy_core._rs.TraceStep` and `Trace`: the steps of a run, recorded.
//!
//! A trace recorded in this process keeps the objects its steps were
//! asked about and answered with, and a step returns them; a trace read
//! back holds the decoded values, and none for an opaque one.

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::{CompareOp, PyTraverseError, PyVisit};
use pyo3::types::{PyBool, PyDict, PyTuple, PyType};

use fhy_core::search_space::{DecisionKind, Trace, TraceStep};

use crate::constraint::value_to_python;
use crate::identifier::identifier_to_python;
use crate::util::dataclass::hash_value;
use crate::util::frozen::{refuse_attribute_assignment, refuse_attribute_deletion};
use crate::util::gc::{Slots, collect_slots};
use crate::util::pending::with_pending_errors;
use crate::util::public_class::PublicClass;
use crate::util::python::read_type_name;
use crate::wire::{check_instance, parse_tree, read_text, read_text_tree, read_tree, to_json};

use super::arguments::{Seeded, instantiate, take_seed, wrong_seed};
use super::domain::{coordinate_to_python, holds_opaque, natural_to_python, owned_step_domain};
use super::wire::{refuse_v1, write_part, write_part_json};

/// The objects a step recorded in this process was asked about and
/// answered with.
#[derive(Default)]
pub(super) struct StepObjects {
    /// The subject, an `Identifier`.
    pub(super) subject: Option<Py<PyAny>>,
    /// The object answered.
    pub(super) value: Option<Py<PyAny>>,
    /// The domain object of a dynamic step, from which a step object reads
    /// the opaque values it keeps.
    pub(super) domain: Option<Py<PyAny>>,
}

impl StepObjects {
    /// Return new references to the same objects.
    pub(super) fn clone_ref(&self, py: Python<'_>) -> Self {
        let copy = |object: &Option<Py<PyAny>>| object.as_ref().map(|object| object.clone_ref(py));
        Self {
            subject: copy(&self.subject),
            value: copy(&self.value),
            domain: copy(&self.domain),
        }
    }

    /// Visit the objects held.
    pub(super) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for object in [&self.subject, &self.value, &self.domain]
            .into_iter()
            .flatten()
        {
            visit.call(object)?;
        }
        Ok(())
    }
}

/// One recorded step, backed by the core [`TraceStep`].
#[pyclass(frozen, module = "fhy_core._rs", name = "TraceStep")]
pub(crate) struct PyTraceStep {
    step: TraceStep,
    /// The objects of a step recorded in this process.
    objects: StepObjects,
    /// The slots of the opaque values of `step`, which the step owns.
    slots: Slots,
}

#[pymethods]
impl PyTraceStep {
    /// What the step is about, a `str`.
    #[getter]
    fn kind(&self) -> String {
        self.step.kind().as_str().to_owned()
    }

    /// The step's subject, an `Identifier`: a static step's decision name.
    #[getter]
    fn subject(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.objects.subject {
            Some(subject) => Ok(subject.clone_ref(py)),
            None => identifier_to_python(py, self.step.subject()).map(Bound::unbind),
        }
    }

    /// A static step's canonical position in its space, or `None`.
    #[getter]
    fn decision(&self) -> Option<usize> {
        self.step.decision()
    }

    /// The number of values the step's domain held, an `int`.
    #[getter]
    fn cardinality(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        natural_to_python(py, self.step.cardinality()).map(Bound::unbind)
    }

    /// The answer: an `int`, or a tuple of `int`s for an order.
    #[getter]
    fn coordinate(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        coordinate_to_python(py, self.step.coordinate()).map(Bound::unbind)
    }

    /// The object answered, the decoded value, or `None`.
    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if let Some(value) = &self.objects.value {
            return Ok(value.clone_ref(py));
        }
        match self.step.value() {
            Some(value) => value_to_python(py, value).map(Bound::unbind),
            None => Ok(py.None()),
        }
    }

    /// The signature of the domain, as its V2 text.
    #[getter]
    fn signature(&self, py: Python<'_>) -> PyResult<String> {
        to_json(py, self.step.signature())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "TraceStep(kind={:?}, subject={}, coordinate={})",
            self.step.kind().as_str(),
            self.subject(py)?.bind(py).repr()?,
            self.coordinate(py)?.bind(py).repr()?
        ))
    }

    /// Visit the objects held, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.objects.traverse(&visit)?;
        self.slots.traverse(&visit)
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

    /// Return the trace of `trace`, each step's objects the ones `objects`
    /// returns for its position.
    ///
    /// A step whose value is or holds an opaque value, and whose domain
    /// object is known, is rebuilt over that domain read anew, so the step
    /// object owns the slots of the objects its value keeps alive.
    fn assemble(
        py: Python<'_>,
        trace: &Trace,
        objects: impl Fn(usize) -> StepObjects,
    ) -> PyResult<Self> {
        let mut core = Vec::with_capacity(trace.len());
        let mut steps = Vec::with_capacity(trace.len());
        for (position, step) in trace.steps().iter().enumerate() {
            let objects = objects(position);
            let (step, slots) = match &objects.domain {
                Some(domain) if step.value().is_some_and(holds_opaque) => {
                    let (owned, slots) = collect_slots(|| owned_step(domain.bind(py), step));
                    (owned?, slots)
                }
                _ => (step.clone(), Slots::default()),
            };
            core.push(step.clone());
            steps.push(Bound::new(
                py,
                PyTraceStep {
                    step,
                    objects,
                    slots,
                },
            )?);
        }
        Ok(Self {
            trace: Trace::new(core),
            steps: PyTuple::new(py, steps)?.unbind(),
        })
    }
}

/// Return `step`, a dynamic step over the domain object `domain`, its value
/// read anew from `domain`'s objects.
///
/// # Errors
///
/// Raises what reading the domain raises.
fn owned_step(domain: &Bound<'_, PyAny>, step: &TraceStep) -> PyResult<TraceStep> {
    let py = domain.py();
    let owned = owned_step_domain(domain)?;
    TraceStep::dynamic(
        step.kind().clone(),
        step.subject().clone(),
        &owned,
        step.coordinate().clone(),
    )
    .map_err(|error| super::errors::trace_error_to_py(py, error))
}

/// Return a new public `Trace` of `trace`, each step's objects the ones
/// `objects` returns for its position.
///
/// # Errors
///
/// Raises what building the object raises.
pub(super) fn trace_to_python<'py>(
    py: Python<'py>,
    trace: &Trace,
    objects: impl Fn(usize) -> StepObjects,
) -> PyResult<Bound<'py, PyAny>> {
    let seeded = PyTrace::assemble(py, trace, objects)?;
    instantiate(PyTrace::public_class().get(py)?, 0, Seeded::Trace(seeded))
}

/// Return the core trace of the `Trace` object `object`.
///
/// # Errors
///
/// Raises `TypeError` naming `owner` for an object that is no `Trace`.
pub(super) fn read_trace(object: &Bound<'_, PyAny>, owner: &str) -> PyResult<Trace> {
    object
        .cast::<PyTrace>()
        .map(|trace| trace.get().core().clone())
        .map_err(|_not_a_trace| {
            PyTypeError::new_err(format!(
                "{owner} takes a Trace, got {}.",
                read_type_name(object)
            ))
        })
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
        if let Some(seeded) = take_seed(kwargs)? {
            return match seeded {
                Seeded::Trace(trace) => Ok(trace),
                _ => Err(wrong_seed()),
            };
        }
        let Some(steps) = steps.filter(|steps| !steps.is_none()) else {
            return Python::attach(|py| {
                Ok(Self {
                    trace: Trace::default(),
                    steps: PyTuple::empty(py).unbind(),
                })
            });
        };
        let py = steps.py();
        let wrong = |object: &Bound<'_, PyAny>| {
            PyTypeError::new_err(format!(
                "Trace takes an iterable of TraceSteps, got {}.",
                read_type_name(object)
            ))
        };
        let items = steps
            .try_iter()
            .map_err(|_not_iterable| wrong(steps))?
            .collect::<PyResult<Vec<_>>>()?;
        let core = items
            .iter()
            .map(|item| {
                item.cast::<PyTraceStep>()
                    .map(|step| step.get().step.clone())
                    .map_err(|_not_a_step| wrong(item))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            trace: Trace::new(core),
            steps: PyTuple::new(py, items)?.unbind(),
        })
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
        self.steps.bind(py).clone()
    }

    /// The steps' coordinates, in the order asked.
    #[getter]
    fn coordinates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let coordinates = self
            .trace
            .coordinates()
            .map(|coordinate| coordinate_to_python(py, coordinate))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, coordinates)
    }

    /// The product of the steps' cardinalities, an `int`; 1 for no step.
    #[getter]
    fn traversed_cardinality(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        natural_to_python(py, self.trace.traversed_cardinality()).map(Bound::unbind)
    }

    /// Return the steps of the kind `kind`, a `str`, in the order asked.
    fn of_kind<'py>(&self, py: Python<'py>, kind: &str) -> PyResult<Bound<'py, PyTuple>> {
        let steps = self.steps.bind(py);
        let Ok(kind) = DecisionKind::new(kind) else {
            return Ok(PyTuple::empty(py));
        };
        let matching = self
            .trace
            .steps()
            .iter()
            .enumerate()
            .filter(|(_, step)| *step.kind() == kind)
            .map(|(position, _)| steps.get_item(position))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, matching)
    }

    fn __len__(&self) -> usize {
        self.trace.len()
    }

    /// Refuse to set an attribute: the trace is frozen.
    fn __setattr__(slf: &Bound<'_, Self>, name: &str, _value: &Bound<'_, PyAny>) -> PyResult<()> {
        refuse_attribute_assignment(slf.as_any(), name)
    }

    /// Refuse to delete an attribute: the trace is frozen.
    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        refuse_attribute_deletion(slf.as_any(), name)
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.steps.bind(py).try_iter()?.into_any().unbind())
    }

    fn __str__(&self) -> String {
        self.trace.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Trace({})", self.trace)
    }

    /// Compare structurally with another trace; another type is
    /// `NotImplemented`.
    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let Ok(other) = other.cast::<Self>() else {
            return Ok(py.NotImplemented());
        };
        let equal = with_pending_errors(|| Ok(self.trace == other.get().trace))?;
        Ok(match op {
            CompareOp::Eq => PyBool::new(py, equal).to_owned().into_any().unbind(),
            CompareOp::Ne => PyBool::new(py, !equal).to_owned().into_any().unbind(),
            _ => py.NotImplemented(),
        })
    }

    /// Return the trace's hash, consistent with `==`.
    fn __hash__(&self) -> PyResult<u64> {
        with_pending_errors(|| Ok(hash_value(&self.trace)))
    }

    /// Return the V2 dict of the trace.
    ///
    /// Raises `SerializationError` inside `wire_version(WireVersion.V1)`.
    fn serialize_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        refuse_v1(slf.as_any())?;
        write_part(slf.py(), || Ok(&slf.get().trace))
    }

    /// Return the canonical V2 text of the trace, re-formatted for
    /// `indent` or `sort_keys`.
    #[pyo3(signature = (*, indent = None, sort_keys = None))]
    fn to_json(
        slf: &Bound<'_, Self>,
        indent: Option<&Bound<'_, PyAny>>,
        sort_keys: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        refuse_v1(slf.as_any())?;
        write_part_json(slf.as_any(), indent, sort_keys, || Ok(&slf.get().trace))
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
        let trace: Trace = parse_tree(cls, read_tree(cls, data)?)?;
        decoded(cls, &trace)
    }

    /// Return the trace of the V2 JSON text `payload`, an instance of
    /// `cls`.
    #[classmethod]
    fn from_json<'py>(
        cls: &Bound<'py, PyType>,
        payload: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let trace: Trace = parse_tree(cls, read_text_tree(cls, &read_text(payload)?)?)?;
        decoded(cls, &trace)
    }

    /// Pickle as a call of `from_json` with the V2 text.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let text = to_json(py, &slf.get().trace)?;
        let restore = slf.get_type().getattr(intern!(py, "from_json"))?;
        PyTuple::new(py, [restore, PyTuple::new(py, [text])?.into_any()])
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

/// Return a new instance of `cls` of the decoded `trace`.
///
/// # Errors
///
/// Raises what building the object raises, and `SerializationError` for
/// an object that is no instance of `cls`.
fn decoded<'py>(cls: &Bound<'py, PyType>, trace: &Trace) -> PyResult<Bound<'py, PyAny>> {
    let seeded = PyTrace::assemble(cls.py(), trace, |_| StepObjects::default())?;
    check_instance(cls, instantiate(cls, 0, Seeded::Trace(seeded))?)
}
