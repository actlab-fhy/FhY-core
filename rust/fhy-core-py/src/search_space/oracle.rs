//! The oracles of `fhy_core.search_space`: `PendingStep`, the step an
//! oracle is asked; `RandomOracle`, `ReplayOracle` and `ExhaustiveOracle`,
//! the core's oracles; the adapter through which a Python oracle answers
//! the core; and the reading of an oracle argument.
//!
//! An oracle argument is read, in order, as one of the core's oracle
//! classes, as an instance of a registered Rust oracle kind (borrowed for
//! the call through its lease), or as any object with a callable `decide`
//! (through [`PythonOracle`]); anything else is `TypeError`. A core oracle
//! class's state is locked while it answers: asking it again from inside
//! its own run (from a hook it calls) is `RuntimeError`.

use std::fmt;
use std::sync::{Mutex, MutexGuard, TryLockError};

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};

use fhy_core::foreign::BoxError;
use fhy_core::identifier::Identifier;
use fhy_core::search_space::{
    Configuration, Coordinate, DecisionKind, ExhaustiveOracle, PendingStep, ReplayOracle, Rng,
    SearchOracle, StepDomain, TraceError,
};

use crate::constraint::read_bound_value;
use crate::convert::param::run_attached_with_context;
use crate::identifier::identifier_to_python;
use crate::util::pending::with_pending_errors;
use crate::util::python::read_type_name;

use super::configuration::configuration_to_python;
use super::domain::{coordinate_to_python, read_coordinate, step_domain_to_python};
use super::errors::{boxed_error_to_py, replay_error_to_py};
use super::kinds::oracle_kind_of;
use super::rng::PyRng;
use super::space::PySpace;
use super::trace::read_trace;

/// The text an oracle's failure is reported with when it is no Python
/// exception.
const ORACLE_FAILED: &str = "the oracle failed";

/// Lock `state`, the state of the core oracle class `class`.
///
/// # Errors
///
/// Raises `RuntimeError` when it is locked already: the oracle is
/// answering a step, and a hook asked it again.
fn lock_state<'a, T>(state: &'a Mutex<T>, class: &str) -> PyResult<MutexGuard<'a, T>> {
    match state.try_lock() {
        Ok(guard) => Ok(guard),
        Err(TryLockError::Poisoned(poisoned)) => Ok(poisoned.into_inner()),
        Err(TryLockError::WouldBlock) => Err(PyRuntimeError::new_err(format!(
            "the {class} is answering a step already"
        ))),
    }
}

/// The objects a Python oracle's snapshot of a step shows, which the core
/// step does not hold.
#[derive(Default)]
pub(super) struct StepFrame {
    /// The `Space` object of a run over a space.
    pub(super) space: Option<Py<PySpace>>,
    /// The subject object of a dynamic step.
    pub(super) subject: Option<Py<PyAny>>,
    /// The domain object of a dynamic step.
    pub(super) domain: Option<Py<PyAny>>,
}

/// A step as a Python oracle is asked it: an owned snapshot, so its
/// methods still work after the call.
#[pyclass(frozen, module = "fhy_core._rs", name = "PendingStep")]
pub(crate) struct PyPendingStep {
    kind: DecisionKind,
    /// The subject's core identifier.
    subject_name: Identifier,
    /// The subject, an `Identifier`.
    subject: Py<PyAny>,
    /// The domain object: the one a dynamic step was given, or one built.
    domain: Py<PyAny>,
    /// The core domain.
    core_domain: StepDomain,
    position: usize,
    /// The `Variable` or `Choice` a static step asks, or `None`.
    decision: Option<Py<PyAny>>,
    /// The run's `Configuration` so far, for a static step.
    configuration: Option<Py<PyAny>>,
    /// The core configuration so far, for a static step.
    core_configuration: Option<Configuration>,
}

impl PyPendingStep {
    /// Return the snapshot of `step`, its objects the ones `frame` holds or
    /// new ones.
    ///
    /// # Errors
    ///
    /// Raises what building an object raises.
    fn of(py: Python<'_>, step: &PendingStep<'_>, frame: &StepFrame) -> PyResult<Self> {
        let is_static = step.configuration().is_some();
        let given = |object: &Option<Py<PyAny>>| {
            object
                .as_ref()
                .filter(|_| !is_static)
                .map(|object| object.clone_ref(py))
        };
        let subject = match given(&frame.subject) {
            Some(subject) => subject,
            None => identifier_to_python(py, step.subject())?.unbind(),
        };
        let domain = match given(&frame.domain) {
            Some(domain) => domain,
            None => step_domain_to_python(py, step.domain())?.unbind(),
        };
        let space = frame
            .space
            .as_ref()
            .filter(|_| is_static)
            .map(|space| space.bind(py));
        let decision = match space {
            Some(space) => space
                .get()
                .position(step.subject())
                .map(|position| space.get().decision_at(py, position))
                .transpose()?
                .map(Bound::unbind),
            None => None,
        };
        let configuration = match (space, step.configuration()) {
            (Some(space), Some(configuration)) => {
                Some(configuration_to_python(space.as_any(), configuration.clone())?.unbind())
            }
            _ => None,
        };
        Ok(Self {
            kind: step.kind().clone(),
            subject_name: step.subject().clone(),
            subject,
            domain,
            core_domain: step.domain().clone(),
            position: step.position(),
            decision,
            configuration,
            core_configuration: step.configuration().cloned(),
        })
    }

    /// Return what `ask` answers about the core step the snapshot was taken
    /// of, under the default solver's context.
    ///
    /// # Errors
    ///
    /// Raises the error `ask` returns as the binding raises an oracle's.
    fn ask<T>(
        &self,
        py: Python<'_>,
        ask: impl FnOnce(&PendingStep<'_>) -> Result<T, BoxError>,
    ) -> PyResult<T> {
        run_attached_with_context(
            py,
            |context| {
                let step = match &self.core_configuration {
                    Some(configuration) => PendingStep::of_decision(
                        &self.kind,
                        configuration,
                        &self.subject_name,
                        &self.core_domain,
                        self.position,
                        context,
                    )
                    .ok_or_else(|| {
                        Box::new(TraceError::UnknownDecision {
                            name: self.subject_name.clone(),
                        }) as BoxError
                    })?,
                    None => PendingStep::dynamic(
                        &self.kind,
                        &self.subject_name,
                        &self.core_domain,
                        self.position,
                        context,
                    ),
                };
                ask(&step)
            },
            |error| boxed_error_to_py(py, error, ORACLE_FAILED),
        )
    }

    /// Return the coordinate `oracle` answers the step with, as a Python
    /// object.
    fn answer(&self, py: Python<'_>, oracle: &mut dyn SearchOracle) -> PyResult<Py<PyAny>> {
        let coordinate = self.ask(py, |step| oracle.decide(step))?;
        coordinate_to_python(py, &coordinate).map(Bound::unbind)
    }
}

#[pymethods]
impl PyPendingStep {
    /// What the step is about, a `str`.
    #[getter]
    fn kind(&self) -> String {
        self.kind.as_str().to_owned()
    }

    /// The step's subject, an `Identifier`.
    #[getter]
    fn subject(&self, py: Python<'_>) -> Py<PyAny> {
        self.subject.clone_ref(py)
    }

    /// The domain the step may be answered from.
    #[getter]
    fn domain(&self, py: Python<'_>) -> Py<PyAny> {
        self.domain.clone_ref(py)
    }

    /// The step's position in its run.
    #[getter]
    const fn position(&self) -> usize {
        self.position
    }

    /// The `Variable` or `Choice` a static step asks, or `None`.
    #[getter]
    fn decision(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.decision
            .as_ref()
            .map(|decision| decision.clone_ref(py))
    }

    /// The run's `Configuration` so far, for a static step, or `None`.
    #[getter]
    fn configuration(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.configuration
            .as_ref()
            .map(|configuration| configuration.clone_ref(py))
    }

    /// Return whether `coordinate` is an admissible answer, checked under
    /// the default solver's context.
    ///
    /// Raises `TypeError` for a coordinate that is no `int` or tuple of
    /// `int`s.
    fn admits(&self, py: Python<'_>, coordinate: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Some(coordinate) = read_coordinate(coordinate, "PendingStep.admits's coordinate")?
        else {
            return Ok(false);
        };
        self.ask(py, |step| Ok(step.admits(&coordinate)?))
    }

    /// Return the coordinate of `value` in the step's domain.
    ///
    /// Raises `ValueError` when the domain does not admit it.
    fn coordinate_of(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let read = read_bound_value(value)?;
        let coordinate = with_pending_errors(|| Ok(self.core_domain.coordinate_of(&read)))?
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "the step's domain does not admit a {}",
                    read_type_name(value)
                ))
            })?;
        coordinate_to_python(py, &coordinate).map(Bound::unbind)
    }

    /// Return an admissible coordinate drawn uniformly with `rng`.
    ///
    /// Raises `DeadEndError` when none is admissible.
    fn draw_uniform(&self, py: Python<'_>, rng: &Bound<'_, PyRng>) -> PyResult<Py<PyAny>> {
        self.answer(py, &mut SharedRandom { rng: rng.get() })
    }

    fn __str__(&self) -> String {
        format!(
            "{} step for {} over {} value(s)",
            self.kind,
            self.subject_name,
            self.core_domain.cardinality()
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "PendingStep(kind={:?}, subject={}, position={})",
            self.kind.as_str(),
            self.subject_name,
            self.position
        )
    }

    /// Visit the objects held, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.subject)?;
        visit.call(&self.domain)?;
        if let Some(decision) = &self.decision {
            visit.call(decision)?;
        }
        if let Some(configuration) = &self.configuration {
            visit.call(configuration)?;
        }
        Ok(())
    }
}

/// The core's random oracle over the generator of an `Rng` object: each
/// step draws from a copy of the generator, which then replaces it, so a
/// hook the draw calls may use the same `Rng`.
struct SharedRandom<'a> {
    rng: &'a PyRng,
}

impl fmt::Debug for SharedRandom<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedRandom").finish_non_exhaustive()
    }
}

impl SearchOracle for SharedRandom<'_> {
    fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError> {
        let mut rng: Rng = self.rng.snapshot();
        let drawn = step.draw_uniform(&mut rng);
        self.rng.restore(rng);
        Ok(drawn?)
    }
}

/// The core's random oracle over a shared `Rng` object.
#[pyclass(frozen, module = "fhy_core._rs", name = "RandomOracle")]
pub(crate) struct PyRandomOracle {
    /// The generator every draw comes from.
    rng: Py<PyRng>,
}

#[pymethods]
impl PyRandomOracle {
    /// Return the oracle drawing from `rng`, or from a new `Rng` seeded with
    /// `seed`, or, with neither, with a seed drawn from `os.urandom`.
    ///
    /// Raises `ValueError` for both, and what `Rng(seed)` raises.
    #[new]
    #[pyo3(signature = (seed = None, *, rng = None))]
    fn new(
        py: Python<'_>,
        seed: Option<&Bound<'_, PyAny>>,
        rng: Option<Py<PyRng>>,
    ) -> PyResult<Self> {
        let seed = seed.filter(|seed| !seed.is_none());
        let rng = match (seed, rng) {
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "RandomOracle takes a seed or an rng, not both",
                ));
            }
            (None, Some(rng)) => rng,
            (Some(seed), None) => Py::new(py, PyRng::from_seed_object(seed)?)?,
            (None, None) => {
                let bytes: Vec<u8> = py
                    .import(intern!(py, "os"))?
                    .call_method1(intern!(py, "urandom"), (8,))?
                    .extract()?;
                let mut word = [0_u8; 8];
                for (slot, byte) in word.iter_mut().zip(bytes) {
                    *slot = byte;
                }
                Py::new(py, PyRng::seeded(u64::from_le_bytes(word)))?
            }
        };
        Ok(Self { rng })
    }

    /// The seed of the generator the draws come from.
    #[getter]
    fn seed(&self, py: Python<'_>) -> u64 {
        self.rng.bind(py).get().seed()
    }

    /// The generator the draws come from.
    #[getter]
    fn rng(&self, py: Python<'_>) -> Py<PyRng> {
        self.rng.clone_ref(py)
    }

    /// Return the answer to `step`, a `PendingStep`.
    fn decide(&self, py: Python<'_>, step: &Bound<'_, PyPendingStep>) -> PyResult<Py<PyAny>> {
        step.get().answer(
            py,
            &mut SharedRandom {
                rng: self.rng.bind(py).get(),
            },
        )
    }

    /// Visit the objects held, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.rng)
    }
}

/// The core's replay oracle.
#[pyclass(frozen, module = "fhy_core._rs", name = "ReplayOracle")]
pub(crate) struct PyReplayOracle {
    /// The `Trace` replayed.
    trace: Py<PyAny>,
    /// The oracle, until `finish` succeeds.
    oracle: Mutex<Option<ReplayOracle>>,
}

impl PyReplayOracle {
    /// Run `run` with the oracle.
    ///
    /// # Errors
    ///
    /// Raises `RuntimeError` after `finish`, or while the oracle answers a
    /// step, and what `run` raises.
    fn with_oracle<T>(&self, run: impl FnOnce(&mut ReplayOracle) -> PyResult<T>) -> PyResult<T> {
        let mut guard = lock_state(&self.oracle, "ReplayOracle")?;
        let oracle = guard
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("the ReplayOracle was finished already"))?;
        run(oracle)
    }
}

#[pymethods]
impl PyReplayOracle {
    /// Return the oracle replaying the `Trace` `trace`.
    ///
    /// Raises `TypeError` for an argument that is no `Trace`.
    #[new]
    fn new(trace: &Bound<'_, PyAny>) -> PyResult<Self> {
        let core = read_trace(trace, "ReplayOracle")?;
        Ok(Self {
            trace: trace.clone().unbind(),
            oracle: Mutex::new(Some(ReplayOracle::new(core))),
        })
    }

    /// The `Trace` replayed.
    #[getter]
    fn trace(&self, py: Python<'_>) -> Py<PyAny> {
        self.trace.clone_ref(py)
    }

    /// Whether every recorded step was answered; `True` once finished.
    #[getter]
    fn is_exhausted(&self) -> PyResult<bool> {
        let guard = lock_state(&self.oracle, "ReplayOracle")?;
        Ok(guard.as_ref().is_none_or(ReplayOracle::is_exhausted))
    }

    /// Finish the replay.
    ///
    /// Raises `ReplayMismatchError` when a recorded step was never asked,
    /// the replay staying open, and `RuntimeError` when the replay was
    /// finished already.
    fn finish(&self, py: Python<'_>) -> PyResult<()> {
        let mut guard = lock_state(&self.oracle, "ReplayOracle")?;
        let oracle = guard
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("the ReplayOracle was finished already"))?;
        oracle
            .clone()
            .finish()
            .map_err(|error| replay_error_to_py(py, error))?;
        *guard = None;
        Ok(())
    }

    /// Return the answer to `step`, a `PendingStep`.
    ///
    /// Raises `ReplayMismatchError` for a step the trace does not describe.
    fn decide(&self, py: Python<'_>, step: &Bound<'_, PyPendingStep>) -> PyResult<Py<PyAny>> {
        self.with_oracle(|oracle| step.get().answer(py, oracle))
    }

    /// Visit the objects held, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.trace)
    }
}

/// The core's exhaustive oracle.
#[pyclass(frozen, module = "fhy_core._rs", name = "ExhaustiveOracle")]
pub(crate) struct PyExhaustiveOracle {
    oracle: Mutex<ExhaustiveOracle>,
}

#[pymethods]
impl PyExhaustiveOracle {
    /// Return the oracle at its first path.
    #[new]
    fn new() -> Self {
        Self {
            oracle: Mutex::new(ExhaustiveOracle::new()),
        }
    }

    /// Move to the next path; return `False` when every path was taken.
    fn advance(&self) -> PyResult<bool> {
        Ok(lock_state(&self.oracle, "ExhaustiveOracle")?.advance())
    }

    /// Return the answer to `step`, a `PendingStep`.
    fn decide(&self, py: Python<'_>, step: &Bound<'_, PyPendingStep>) -> PyResult<Py<PyAny>> {
        let mut oracle = lock_state(&self.oracle, "ExhaustiveOracle")?;
        step.get().answer(py, &mut *oracle)
    }
}

/// A Python object with a callable `decide`, as a core oracle: each call
/// receives a [`PyPendingStep`] snapshot and must return an `int` or a
/// tuple of `int`s; its exception is boxed and raised by the entry point
/// as itself.
pub(crate) struct PythonOracle {
    object: Py<PyAny>,
    /// The objects the snapshots show.
    frame: StepFrame,
}

impl PythonOracle {
    /// Return the coordinate the object answers `step` with, `None` for an
    /// integer no domain has a coordinate for.
    fn ask(&self, py: Python<'_>, step: &PendingStep<'_>) -> PyResult<Option<Coordinate>> {
        let object = self.object.bind(py);
        let snapshot = Bound::new(py, PyPendingStep::of(py, step, &self.frame)?)?;
        let answer = object.call_method1(intern!(py, "decide"), (snapshot,))?;
        read_coordinate(&answer, "decide's answer").map_err(|_wrong_type| {
            PyTypeError::new_err(format!(
                "{}.decide must return an int or a tuple of ints, got {}.",
                read_type_name(object),
                read_type_name(&answer)
            ))
        })
    }
}

impl fmt::Debug for PythonOracle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let type_name = Python::attach(|py| read_type_name(self.object.bind(py)));
        f.debug_struct("PythonOracle")
            .field("type", &type_name)
            .finish_non_exhaustive()
    }
}

impl SearchOracle for PythonOracle {
    fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError> {
        match Python::attach(|py| self.ask(py, step)) {
            Ok(Some(coordinate)) => Ok(coordinate),
            Ok(None) => Err(Box::new(TraceError::CoordinateOutOfDomain {
                position: step.position(),
            })),
            Err(error) => Err(Box::new(error)),
        }
    }
}

/// Run `run` with the core oracle of the oracle argument `object`, read as
/// the module documentation says, a Python oracle's snapshots showing the
/// objects of `frame`.
///
/// # Errors
///
/// Raises `TypeError` for an object that is no oracle, `RuntimeError` for
/// a core oracle class that is answering a step already or a finished
/// replay, what a lease raises, and what `run` raises.
pub(crate) fn with_oracle<T>(
    object: &Bound<'_, PyAny>,
    frame: StepFrame,
    run: impl FnOnce(&mut dyn SearchOracle) -> PyResult<T>,
) -> PyResult<T> {
    let py = object.py();
    if let Ok(random) = object.cast::<PyRandomOracle>() {
        let rng = random.get().rng.bind(py);
        return run(&mut SharedRandom { rng: rng.get() });
    }
    if let Ok(replay) = object.cast::<PyReplayOracle>() {
        return replay.get().with_oracle(|oracle| run(oracle));
    }
    if let Ok(exhaustive) = object.cast::<PyExhaustiveOracle>() {
        let mut oracle = lock_state(&exhaustive.get().oracle, "ExhaustiveOracle")?;
        return run(&mut *oracle);
    }
    if let Some(kind) = oracle_kind_of(object)? {
        let mut oracle = (kind.lease)(object)?;
        return run(&mut *oracle);
    }
    let decide = intern!(py, "decide");
    if object.hasattr(decide)? && object.getattr(decide)?.is_callable() {
        return run(&mut PythonOracle {
            object: object.clone().unbind(),
            frame,
        });
    }
    Err(PyTypeError::new_err(format!(
        "an oracle must have a callable decide, got {}.",
        read_type_name(object)
    )))
}
