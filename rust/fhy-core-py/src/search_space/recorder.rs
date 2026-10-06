//! `fhy_core._rs.Recorder`: one run of a search, driven from Python.

use std::sync::{Mutex, MutexGuard, TryLockError};

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::PyTuple;

use fhy_core::constraint::Value;
use fhy_core::identifier::Identifier;
use fhy_core::search_space::{DecisionKind, Recorder};

use crate::constraint::value_to_python;
use crate::convert::param::run_attached_with_context;
use crate::identifier::restore_identifier;
use crate::util::gc::{Slots, collect_slots};
use crate::util::python::read_type_name;

use super::choice::PyChoice;
use super::configuration::{PyConfiguration, configuration_to_python};
use super::domain::{object_at, owned_step_domain};
use super::errors::trace_error_to_py;
use super::oracle::{StepFrame, with_oracle};
use super::space::PySpace;
use super::trace::{StepObjects, trace_to_python};

/// A run in progress: the core recorder and the objects of each step it
/// recorded, by position.
struct Run {
    recorder: Recorder,
    objects: Vec<StepObjects>,
    /// The slots of the opaque values the recorder's steps keep, which the
    /// run owns.
    slots: Vec<Slots>,
}

/// One run of a search over the core [`Recorder`]: its oracle, its space
/// or configuration, and the objects its steps were asked about and
/// answered with.
#[pyclass(frozen, module = "fhy_core._rs", name = "Recorder")]
pub(crate) struct PyRecorder {
    /// The oracle argument, read anew for each step.
    oracle: Py<PyAny>,
    /// The `Space` or `Configuration` given, if any.
    target: Option<Py<PyAny>>,
    /// The `Space` of the run, if it has one.
    space: Option<Py<PySpace>>,
    /// The run, until `finish` takes it.
    run: Mutex<Option<Run>>,
}

impl PyRecorder {
    /// Run `step` on the run in progress.
    ///
    /// # Errors
    ///
    /// Raises `RuntimeError` after `finish`, or when the run is asking a
    /// step already (its oracle used the recorder), and what `step`
    /// raises.
    fn with_run<T>(&self, step: impl FnOnce(&mut Run) -> PyResult<T>) -> PyResult<T> {
        let mut guard = self.lock_run()?;
        let run = guard.as_mut().ok_or_else(finished)?;
        step(run)
    }

    /// Lock the run.
    ///
    /// # Errors
    ///
    /// Raises `RuntimeError` when the run is asking a step already: its
    /// oracle used the recorder.
    fn lock_run(&self) -> PyResult<MutexGuard<'_, Option<Run>>> {
        match self.run.try_lock() {
            Ok(guard) => Ok(guard),
            Err(TryLockError::Poisoned(poisoned)) => Ok(poisoned.into_inner()),
            Err(TryLockError::WouldBlock) => Err(PyRuntimeError::new_err(
                "the Recorder is asking a step already",
            )),
        }
    }

    /// Return the objects a Python oracle's snapshots show, beyond the
    /// space's: `subject` and `domain` for a dynamic step.
    fn frame(
        &self,
        py: Python<'_>,
        subject: Option<&Bound<'_, PyAny>>,
        domain: Option<&Bound<'_, PyAny>>,
    ) -> StepFrame {
        StepFrame {
            space: self.space.as_ref().map(|space| space.clone_ref(py)),
            subject: subject.map(|subject| subject.clone().unbind()),
            domain: domain.map(|domain| domain.clone().unbind()),
        }
    }

    /// Return the object of the value `value` of the decision `name`: the
    /// chosen `Alternative` object for a choice, a new object of the value
    /// for a variable.
    fn value_object<'py>(
        &self,
        py: Python<'py>,
        name: &Identifier,
        value: &Value,
    ) -> PyResult<Bound<'py, PyAny>> {
        let chosen = match (&self.space, value) {
            (Some(space), Value::Identifier(chosen)) => {
                let space = space.bind(py).get();
                match space.position(name) {
                    Some(position) => Some((space.decision_at(py, position)?, chosen)),
                    None => None,
                }
            }
            _ => None,
        };
        if let Some((decision, chosen)) = chosen
            && let Ok(choice) = decision.cast::<PyChoice>()
        {
            let index = choice
                .get()
                .core()
                .alternatives()
                .iter()
                .position(|alternative| alternative.get().name() == chosen);
            if let Some(index) = index {
                return choice.get().alternative_objects(py).get_item(index);
            }
        }
        value_to_python(py, value)
    }
}

/// Return the `RuntimeError` of a recorder used after `finish`.
fn finished() -> PyErr {
    PyRuntimeError::new_err("the Recorder was finished already")
}

#[pymethods]
impl PyRecorder {
    /// Return the recorder of a run answered by `oracle`: over the `Space`
    /// `space`, realizing the `Configuration` `configuration`, or, with
    /// neither, of dynamic steps only. The oracle is read at each step.
    ///
    /// Raises `ValueError` for both, and `TypeError` for a space or
    /// configuration of the wrong class.
    #[new]
    #[pyo3(signature = (oracle, *, space = None, configuration = None))]
    fn new(
        oracle: &Bound<'_, PyAny>,
        space: Option<&Bound<'_, PyAny>>,
        configuration: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let py = oracle.py();
        let space = space.filter(|space| !space.is_none());
        let configuration = configuration.filter(|configuration| !configuration.is_none());
        let wrong = |expected: &str, object: &Bound<'_, PyAny>| {
            PyTypeError::new_err(format!(
                "Recorder takes {expected}, got {}.",
                read_type_name(object)
            ))
        };
        let (recorder, target, space) = match (space, configuration) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "Recorder takes a space or a configuration, not both",
                ));
            }
            (Some(space), None) => {
                let space_object = space
                    .cast::<PySpace>()
                    .map_err(|_not_a_space| wrong("a Space", space))?;
                (
                    Recorder::over(space_object.get().core()),
                    Some(space.clone().unbind()),
                    Some(space_object.clone().unbind()),
                )
            }
            (None, Some(configuration)) => {
                let read = configuration
                    .cast::<PyConfiguration>()
                    .map_err(|_not_a_configuration| wrong("a Configuration", configuration))?
                    .get();
                (
                    Recorder::realizing(read.core()),
                    Some(configuration.clone().unbind()),
                    Some(read.space_object(py).clone().unbind()),
                )
            }
            (None, None) => (Recorder::new(), None, None),
        };
        Ok(Self {
            oracle: oracle.clone().unbind(),
            target,
            space,
            run: Mutex::new(Some(Run {
                recorder,
                objects: Vec::new(),
                slots: Vec::new(),
            })),
        })
    }

    /// Ask the decision named `name` and return its value: for a choice,
    /// the chosen `Alternative` object.
    ///
    /// Raises `TraceError` (or `InadmissibleAnswerError`,
    /// `NotEnumerableError`, `DeadEndError`, `ReplayMismatchError`) for the
    /// core's refusals, the oracle's exception as itself, and
    /// `RuntimeError` after `finish`.
    fn decide(&self, py: Python<'_>, name: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let identifier = restore_identifier(name, "Recorder", "name")?;
        self.with_run(|run| {
            let value = with_oracle(self.oracle.bind(py), self.frame(py, None, None), |oracle| {
                run_attached_with_context(
                    py,
                    |context| run.recorder.decide(&identifier, oracle, context),
                    |error| trace_error_to_py(py, error),
                )
            })?;
            let object = self.value_object(py, &identifier, &value)?.unbind();
            run.objects.push(StepObjects {
                subject: Some(name.clone().unbind()),
                value: Some(object.clone_ref(py)),
                domain: None,
            });
            Ok(object)
        })
    }

    /// Ask the dynamic step of `kind`, a `str`, about the `Identifier`
    /// `subject` over the domain object `domain`, and return the domain's
    /// object at the answered coordinate.
    ///
    /// Raises `TypeError` for a subject that is no `Identifier` or a domain
    /// that is no domain object, `TraceError` for an empty kind, and as
    /// `decide` raises.
    fn decide_dynamic(
        &self,
        py: Python<'_>,
        kind: &str,
        subject: &Bound<'_, PyAny>,
        domain: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let kind = DecisionKind::new(kind)
            .map_err(|error| super::errors::trace_error_text_to_py(py, &error.to_string()))?;
        let identifier = restore_identifier(subject, "Recorder", "subject")?;
        let (core_domain, slots) = collect_slots(|| owned_step_domain(domain));
        let core_domain = core_domain?;
        self.with_run(|run| {
            let frame = self.frame(py, Some(subject), Some(domain));
            let coordinate = with_oracle(self.oracle.bind(py), frame, |oracle| {
                run_attached_with_context(
                    py,
                    |context| {
                        run.recorder.decide_dynamic(
                            &kind,
                            &identifier,
                            &core_domain,
                            oracle,
                            context,
                        )
                    },
                    |error| trace_error_to_py(py, error),
                )
            })?;
            let object = object_at(domain, &coordinate)?
                .ok_or_else(|| PyRuntimeError::new_err("the answer names no object of the domain"))?
                .unbind();
            run.objects.push(StepObjects {
                subject: Some(subject.clone().unbind()),
                value: Some(object.clone_ref(py)),
                domain: Some(domain.clone().unbind()),
            });
            run.slots.push(slots);
            Ok(object)
        })
    }

    /// The steps recorded so far, a `Trace`.
    #[getter]
    fn trace(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.with_run(|run| {
            let objects = &run.objects;
            trace_to_python(py, run.recorder.trace(), |position| {
                objects
                    .get(position)
                    .map_or_else(StepObjects::default, |objects| objects.clone_ref(py))
            })
            .map(Bound::unbind)
        })
    }

    /// The run's `Configuration` so far, or `None` for a run over no
    /// space.
    #[getter]
    fn configuration(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.with_run(|run| match (&self.space, run.recorder.configuration()) {
            (Some(space), Some(configuration)) => {
                configuration_to_python(space.bind(py).as_any(), configuration.clone())
                    .map(|configuration| Some(configuration.unbind()))
            }
            _ => Ok(None),
        })
    }

    /// Finish the run and return `(trace, configuration)`.
    ///
    /// Raises `TraceError` when a realizing run never asked a decision its
    /// configuration assigns, and `RuntimeError` when finished already.
    fn finish<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let run = self.lock_run()?.take().ok_or_else(finished)?;
        let Run {
            recorder, objects, ..
        } = run;
        let recorded = match recorder.finish() {
            Ok(recorded) => recorded,
            Err(error) => return Err(trace_error_to_py(py, error)),
        };
        let (trace, configuration) = recorded.into_parts();
        let trace = trace_to_python(py, trace, |position| {
            objects
                .get(position)
                .map_or_else(StepObjects::default, |objects| objects.clone_ref(py))
        })?;
        let configuration = match (&self.space, configuration) {
            (Some(space), Some(configuration)) => {
                configuration_to_python(space.bind(py).as_any(), configuration)?
            }
            _ => py.None().into_bound(py),
        };
        PyTuple::new(py, [trace, configuration])
    }

    /// Visit the objects held, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.oracle)?;
        if let Some(target) = &self.target {
            visit.call(target)?;
        }
        if let Some(space) = &self.space {
            visit.call(space)?;
        }
        crate::util::gc::traverse_locked(&self.run, |run| {
            let Some(run) = run else {
                return Ok(());
            };
            for objects in &run.objects {
                objects.traverse(&visit)?;
            }
            for slots in &run.slots {
                slots.traverse(&visit)?;
            }
            Ok(())
        })
    }
}
