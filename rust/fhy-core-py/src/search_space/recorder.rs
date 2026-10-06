//! `fhy_core._rs.Recorder`: one run of a search, driven from Python.

#![expect(
    unused_variables,
    dead_code,
    reason = "interface stub: the bodies are todo!() until the implementation"
)]

use std::collections::HashMap;
use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::PyTuple;

use fhy_core::search_space::Recorder;

/// One run of a search over the core [`Recorder`]: its oracle, its space
/// or configuration, and the objects its steps were answered with.
#[pyclass(frozen, module = "fhy_core._rs", name = "Recorder")]
pub(crate) struct PyRecorder {
    /// The oracle argument, read anew for each step.
    oracle: Py<PyAny>,
    /// The `Space` or `Configuration` given, if any.
    target: Option<Py<PyAny>>,
    /// The run, until `finish` takes it.
    recorder: Mutex<Option<Recorder>>,
    /// The objects answered, by step position.
    values: Mutex<HashMap<usize, Py<PyAny>>>,
}

#[pymethods]
impl PyRecorder {
    /// Return the recorder of a run answered by `oracle`: over the `Space`
    /// `space`, realizing the `Configuration` `configuration`, or, with
    /// neither, of dynamic steps only.
    ///
    /// Raises `ValueError` for both, and `TypeError` for an oracle that is
    /// none or a space or configuration of the wrong class.
    #[new]
    #[pyo3(signature = (oracle, *, space = None, configuration = None))]
    fn new(
        oracle: &Bound<'_, PyAny>,
        space: Option<&Bound<'_, PyAny>>,
        configuration: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        todo!()
    }

    /// Ask the decision named `name` and return its value: for a choice,
    /// the chosen `Alternative` object.
    ///
    /// Raises `TraceError` (or `InadmissibleAnswerError`,
    /// `NotEnumerableError`, `DeadEndError`, `ReplayMismatchError`) for the
    /// core's refusals, the oracle's exception as itself, and
    /// `RuntimeError` after `finish`.
    fn decide(&self, py: Python<'_>, name: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        todo!()
    }

    /// Ask the dynamic step of `kind`, a `str`, about the `Identifier`
    /// `subject` over the domain object `domain`, and return the domain's
    /// object at the answered coordinate.
    fn decide_dynamic(
        &self,
        py: Python<'_>,
        kind: &str,
        subject: &Bound<'_, PyAny>,
        domain: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        todo!()
    }

    /// The steps recorded so far, a `Trace`.
    #[getter]
    fn trace(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        todo!()
    }

    /// The run's `Configuration` so far, or `None` for a run over no
    /// space.
    #[getter]
    fn configuration(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        todo!()
    }

    /// Finish the run and return `(trace, configuration)`.
    ///
    /// Raises `TraceError` when a realizing run never asked a decision its
    /// configuration assigns, and `RuntimeError` when finished already.
    fn finish<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
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
        crate::util::gc::traverse_locked(&self.values, |values| {
            for value in values.values() {
                visit.call(value)?;
            }
            Ok(())
        })
    }
}
