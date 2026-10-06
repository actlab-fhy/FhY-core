//! The oracles of `fhy_core.search_space`: `PendingStep`, the step an
//! oracle is asked; `RandomOracle`, `ReplayOracle` and `ExhaustiveOracle`,
//! the core's oracles; the adapter through which a Python oracle answers
//! the core; and the reading of an oracle argument.
//!
//! An oracle argument is read, in order, as one of the core's oracle
//! classes, as an instance of a registered Rust oracle kind (borrowed for
//! the call through its lease), or as any object with a callable `decide`
//! (through [`PythonOracle`]); anything else is `TypeError`.

#![expect(
    unused_variables,
    dead_code,
    reason = "interface stub: the bodies are todo!() until the implementation"
)]

use std::fmt;
use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};

use fhy_core::foreign::BoxError;
use fhy_core::search_space::{
    Configuration, Coordinate, ExhaustiveOracle, PendingStep, ReplayOracle, SearchOracle, Space,
    StepDomain,
};

use super::rng::PyRng;

/// A step as a Python oracle is asked it: an owned snapshot, so its
/// methods still work after the call.
#[pyclass(frozen, module = "fhy_core._rs", name = "PendingStep")]
pub(crate) struct PyPendingStep {
    kind: String,
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

#[pymethods]
impl PyPendingStep {
    /// What the step is about, a `str`.
    #[getter]
    fn kind(&self) -> String {
        todo!()
    }

    /// The step's subject, an `Identifier`.
    #[getter]
    fn subject(&self, py: Python<'_>) -> Py<PyAny> {
        todo!()
    }

    /// The domain the step may be answered from.
    #[getter]
    fn domain(&self, py: Python<'_>) -> Py<PyAny> {
        todo!()
    }

    /// The step's position in its run.
    #[getter]
    fn position(&self) -> usize {
        todo!()
    }

    /// The `Variable` or `Choice` a static step asks, or `None`.
    #[getter]
    fn decision(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        todo!()
    }

    /// The run's `Configuration` so far, for a static step, or `None`.
    #[getter]
    fn configuration(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        todo!()
    }

    /// Return whether `coordinate` is an admissible answer, checked under
    /// the default solver's context.
    ///
    /// Raises `TypeError` for a coordinate that is no `int` or tuple of
    /// `int`s.
    fn admits(&self, py: Python<'_>, coordinate: &Bound<'_, PyAny>) -> PyResult<bool> {
        todo!()
    }

    /// Return the coordinate of `value` in the step's domain.
    ///
    /// Raises `ValueError` when the domain does not admit it.
    fn coordinate_of(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        todo!()
    }

    /// Return an admissible coordinate drawn uniformly with `rng`.
    ///
    /// Raises `DeadEndError` when none is admissible.
    fn draw_uniform(&self, py: Python<'_>, rng: &Bound<'_, PyRng>) -> PyResult<Py<PyAny>> {
        todo!()
    }

    fn __str__(&self) -> String {
        todo!()
    }

    fn __repr__(&self) -> String {
        todo!()
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
        todo!()
    }

    /// The seed of the generator the draws come from.
    #[getter]
    fn seed(&self, py: Python<'_>) -> u64 {
        todo!()
    }

    /// The generator the draws come from.
    #[getter]
    fn rng(&self, py: Python<'_>) -> Py<PyRng> {
        todo!()
    }

    /// Return the answer to `step`, a `PendingStep`.
    fn decide(&self, py: Python<'_>, step: &Bound<'_, PyPendingStep>) -> PyResult<Py<PyAny>> {
        todo!()
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
    /// The oracle, until `finish` takes it.
    oracle: Mutex<Option<ReplayOracle>>,
}

#[pymethods]
impl PyReplayOracle {
    /// Return the oracle replaying the `Trace` `trace`.
    ///
    /// Raises `TypeError` for an argument that is no `Trace`.
    #[new]
    fn new(trace: &Bound<'_, PyAny>) -> PyResult<Self> {
        todo!()
    }

    /// The `Trace` replayed.
    #[getter]
    fn trace(&self, py: Python<'_>) -> Py<PyAny> {
        todo!()
    }

    /// Whether every recorded step was answered.
    #[getter]
    fn is_exhausted(&self) -> bool {
        todo!()
    }

    /// Finish the replay.
    ///
    /// Raises `ReplayMismatchError` when a recorded step was never asked,
    /// and `RuntimeError` when the replay was finished already.
    fn finish(&self) -> PyResult<()> {
        todo!()
    }

    /// Return the answer to `step`, a `PendingStep`.
    ///
    /// Raises `ReplayMismatchError` for a step the trace does not describe.
    fn decide(&self, py: Python<'_>, step: &Bound<'_, PyPendingStep>) -> PyResult<Py<PyAny>> {
        todo!()
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
        todo!()
    }

    /// Move to the next path; return `False` when every path was taken.
    fn advance(&self) -> bool {
        todo!()
    }

    /// Return the answer to `step`, a `PendingStep`.
    fn decide(&self, py: Python<'_>, step: &Bound<'_, PyPendingStep>) -> PyResult<Py<PyAny>> {
        todo!()
    }
}

/// A Python object with a callable `decide`, as a core oracle: each call
/// receives a [`PyPendingStep`] snapshot and must return an `int` or a
/// tuple of `int`s; its exception is boxed and raised by the entry point
/// as itself.
pub(crate) struct PythonOracle {
    object: Py<PyAny>,
    /// The space a static step's snapshot holds, if the run has one.
    space: Option<Space>,
}

impl fmt::Debug for PythonOracle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

impl SearchOracle for PythonOracle {
    fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError> {
        todo!()
    }
}

/// Run `run` with the core oracle of the oracle argument `object`, read as
/// the module documentation says.
///
/// # Errors
///
/// Raises `TypeError` for an object that is no oracle, what a lease
/// raises, and what `run` raises.
pub(crate) fn with_oracle<T>(
    object: &Bound<'_, PyAny>,
    run: impl FnOnce(&mut dyn SearchOracle) -> PyResult<T>,
) -> PyResult<T> {
    todo!()
}
