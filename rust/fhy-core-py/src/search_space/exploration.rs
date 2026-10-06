//! What a `Space` and a `Configuration` do with the search stream on their
//! own: sample, sample uniformly, replay, enumerate, count and mutate; and
//! `fhy_core._rs.SpaceEnumeration`, the iterator `Space.enumerate` returns.

use std::convert::Infallible;
use std::num::NonZeroU32;
use std::sync::{Mutex, TryLockError};

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::PyTuple;

use fhy_core::search_space::{Cardinality, ExhaustiveOracle, Recorded, Rng, Space, TraceError};

use crate::convert::param::{run_attached_with_context, run_with_context};
use crate::identifier::identifier_to_python;
use crate::util::python::read_type_name;

use super::configuration::{PyConfiguration, configuration_to_python};
use super::domain::natural_to_python;
use super::errors::{replay_error_to_py, trace_error_to_py};
use super::oracle::{StepFrame, with_oracle};
use super::rng::PyRng;
use super::space::PySpace;
use super::trace::{StepObjects, read_trace, trace_to_python};

/// Return `(configuration, trace)` of the run `recorded` over the `Space`
/// object `space`.
///
/// # Errors
///
/// Raises what building the objects raises.
fn recorded_to_python<'py>(
    space: &Bound<'py, PySpace>,
    recorded: Recorded,
) -> PyResult<Bound<'py, PyTuple>> {
    let py = space.py();
    let (trace, configuration) = recorded.into_parts();
    let trace = trace_to_python(py, &trace, |_| StepObjects::default())?;
    let configuration = match configuration {
        Some(configuration) => configuration_to_python(space.as_any(), configuration)?,
        None => py.None().into_bound(py),
    };
    PyTuple::new(py, [configuration, trace])
}

/// Return the `Rng` object `rng`.
///
/// # Errors
///
/// Raises `TypeError` naming `owner` for an object that is no `Rng`.
fn read_rng<'a, 'py>(rng: &'a Bound<'py, PyAny>, owner: &str) -> PyResult<&'a Bound<'py, PyRng>> {
    rng.cast::<PyRng>().map_err(|_not_an_rng| {
        PyTypeError::new_err(format!(
            "{owner} takes an Rng, got {}.",
            read_type_name(rng)
        ))
    })
}

/// Return the attempts `attempts`, which must be positive.
///
/// # Errors
///
/// Raises `ValueError` for zero.
fn read_attempts(attempts: u32) -> PyResult<NonZeroU32> {
    NonZeroU32::new(attempts).ok_or_else(|| PyValueError::new_err("attempts must be at least 1"))
}

/// Run `draw` with a copy of the generator of `rng` under the default
/// solver's context, detached, the copy then replacing the generator.
///
/// # Errors
///
/// Raises the error `draw` returns as [`trace_error_to_py`] raises it.
fn draw_with<T: Send>(
    rng: &Bound<'_, PyRng>,
    draw: impl FnOnce(&mut Rng, &fhy_core::param::ParamContext<'_>) -> Result<T, TraceError> + Send,
) -> PyResult<T> {
    let py = rng.py();
    let mut generator = rng.get().snapshot();
    let result = run_with_context(
        py,
        true,
        |context| draw(&mut generator, context),
        |error| trace_error_to_py(py, error),
    );
    rng.get().restore(generator);
    result
}

/// `Space.sample`: ask `oracle` every active decision of `space`.
pub(super) fn sample<'py>(
    space: &Bound<'py, PySpace>,
    oracle: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let py = space.py();
    let frame = StepFrame {
        space: Some(space.clone().unbind()),
        ..StepFrame::default()
    };
    let core = space.get().core();
    let recorded = with_oracle(oracle, frame, |oracle| {
        run_attached_with_context(
            py,
            |context| core.sample(oracle, context),
            |error| trace_error_to_py(py, error),
        )
    })?;
    recorded_to_python(space, recorded)
}

/// `Space.sample_uniform`: draw a complete configuration of `space`
/// uniformly with `rng`.
pub(super) fn sample_uniform<'py>(
    space: &Bound<'py, PySpace>,
    rng: &Bound<'py, PyAny>,
    attempts: u32,
) -> PyResult<Bound<'py, PyTuple>> {
    let rng = read_rng(rng, "Space.sample_uniform")?;
    let attempts = read_attempts(attempts)?;
    let core = space.get().core();
    let recorded = draw_with(rng, |generator, context| {
        core.sample_uniform(generator, context, attempts)
    })?;
    recorded_to_python(space, recorded)
}

/// `Space.replay`: the configuration of `space` the `Trace` `trace`
/// describes.
pub(super) fn replay<'py>(
    space: &Bound<'py, PySpace>,
    trace: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = space.py();
    let trace = read_trace(trace, "Space.replay")?;
    let core = space.get().core();
    let configuration = run_with_context(
        py,
        true,
        |context| core.replay(&trace, context),
        |error| replay_error_to_py(py, error),
    )?;
    configuration_to_python(space.as_any(), configuration)
}

/// `Space._cardinality`: the count of complete configurations of `space`,
/// as `(kind, count, decision)`.
pub(super) fn cardinality<'py>(
    space: &Bound<'py, PySpace>,
    budget: u64,
) -> PyResult<Bound<'py, PyTuple>> {
    let py = space.py();
    let core = space.get().core();
    let count = run_with_context(
        py,
        true,
        |context| core.cardinality(context, budget),
        |error| trace_error_to_py(py, error),
    )?;
    let none = || py.None().into_bound(py);
    let (kind, count, decision) = match count {
        Cardinality::Exact(count) => ("exact", natural_to_python(py, count)?, none()),
        Cardinality::AtLeast(count) => ("at_least", natural_to_python(py, count)?, none()),
        Cardinality::Unbounded { decision } => {
            ("unbounded", none(), identifier_to_python(py, &decision)?)
        }
        Cardinality::Unknown { decision } => {
            ("unknown", none(), identifier_to_python(py, &decision)?)
        }
    };
    PyTuple::new(py, [kind.into_pyobject(py)?.into_any(), count, decision])
}

/// `Space.mutate`: `configuration` with one decision of `space` changed and
/// the rest repaired, with `rng`.
pub(super) fn mutate<'py>(
    space: &Bound<'py, PySpace>,
    configuration: &Bound<'py, PyAny>,
    rng: &Bound<'py, PyAny>,
    attempts: u32,
) -> PyResult<Bound<'py, PyTuple>> {
    let configuration = configuration
        .cast::<PyConfiguration>()
        .map_err(|_not_a_configuration| {
            PyTypeError::new_err(format!(
                "Space.mutate takes a Configuration, got {}.",
                read_type_name(configuration)
            ))
        })?
        .get()
        .core();
    let rng = read_rng(rng, "Space.mutate")?;
    let attempts = read_attempts(attempts)?;
    let core = space.get().core();
    let recorded = draw_with(rng, |generator, context| {
        core.mutate(configuration, generator, context, attempts)
    })?;
    recorded_to_python(space, recorded)
}

/// `Configuration.trace`: the trace of the assigned decisions of
/// `configuration`.
pub(super) fn configuration_trace<'py>(
    configuration: &Bound<'py, PyConfiguration>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = configuration.py();
    let core = configuration.get().core();
    let trace = run_with_context(
        py,
        true,
        |context| core.trace(context),
        |error| trace_error_to_py(py, error),
    )?;
    trace_to_python(py, &trace, |_| StepObjects::default())
}

/// Where an enumeration stands.
struct EnumerationState {
    oracle: ExhaustiveOracle,
    is_done: bool,
}

/// The complete configurations of a space, each once, in lexicographic
/// order of their coordinates in decision order: what `Space.enumerate`
/// returns.
#[pyclass(frozen, module = "fhy_core._rs", name = "SpaceEnumeration")]
pub(crate) struct PySpaceEnumeration {
    space: Py<PySpace>,
    state: Mutex<EnumerationState>,
}

impl PySpaceEnumeration {
    /// Return the enumeration of `space` from its first configuration.
    pub(super) fn of(space: &Bound<'_, PySpace>) -> Self {
        Self {
            space: space.clone().unbind(),
            state: Mutex::new(EnumerationState {
                oracle: ExhaustiveOracle::new(),
                is_done: false,
            }),
        }
    }
}

#[pymethods]
impl PySpaceEnumeration {
    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    /// Return the next complete configuration, or stop.
    ///
    /// Raises `NotEnumerableError` for a variable with no finite domain,
    /// which ends the enumeration, and `RuntimeError` when asked from
    /// inside its own step.
    fn __next__(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => {
                return Err(PyRuntimeError::new_err(
                    "the enumeration is building a configuration already",
                ));
            }
        };
        let space = self.space.bind(py);
        let core: &Space = space.get().core();
        while !state.is_done {
            let oracle = &mut state.oracle;
            let result = run_with_context(
                py,
                true,
                |context| Ok::<_, Infallible>(core.sample(oracle, context)),
                |never| match never {},
            )?;
            state.is_done = !state.oracle.advance();
            match result {
                Ok(recorded) => {
                    if let Some(configuration) = recorded.into_parts().1 {
                        return configuration_to_python(space.as_any(), configuration)
                            .map(|configuration| Some(configuration.unbind()));
                    }
                }
                Err(error) if ExhaustiveOracle::is_backtrack(&error) => {}
                Err(error) => {
                    state.is_done = true;
                    return Err(trace_error_to_py(py, error));
                }
            }
        }
        Ok(None)
    }

    /// Visit the objects held, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.space)
    }
}
