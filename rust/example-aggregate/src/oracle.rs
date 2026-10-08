//! A downstream Rust oracle, registered with the binding's kind registry:
//! the template of a product's own oracle kind.
//!
//! `CountingOracle` (kind `example.counting_oracle`) answers every step
//! with the coordinate of the domain's first value and counts the steps it
//! answered. Its lease borrows it out of its object for one call, so a run
//! asks it with no Python call per step.

use std::sync::{Mutex, PoisonError};

use pyo3::prelude::*;
use pyo3::types::PyType;

use fhy_core::foreign::BoxError;
use fhy_core::search_space::{Coordinate, PendingStep, SearchOracle, StepDomain};
use fhy_core_py::convert;

/// The kind of [`PyCountingOracle`].
const COUNTING_ORACLE: &str = "example.counting_oracle";

/// An oracle answering every step with its domain's first value, counting
/// the steps.
#[pyclass(frozen, module = "fhy_example_aggregate", name = "CountingOracle")]
struct PyCountingOracle {
    count: Mutex<usize>,
}

#[pymethods]
impl PyCountingOracle {
    /// Return the oracle, having answered no step.
    #[new]
    fn new() -> Self {
        Self {
            count: Mutex::new(0),
        }
    }

    /// The number of steps answered.
    #[getter]
    fn count(&self) -> usize {
        *self.count.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The oracle borrowed out of its object for one call.
struct Lease<'a>(&'a PyCountingOracle);

impl SearchOracle for Lease<'_> {
    fn decide(&mut self, step: &PendingStep<'_>) -> Result<Coordinate, BoxError> {
        *self.0.count.lock().unwrap_or_else(PoisonError::into_inner) += 1;
        Ok(match step.domain() {
            StepDomain::Order(domain) => {
                let positions = (0..domain.elements().len())
                    .map(u32::try_from)
                    .collect::<Result<Box<[u32]>, _>>()?;
                Coordinate::Order(positions)
            }
            _ => Coordinate::Index(0),
        })
    }
}

/// Borrow the oracle out of a `CountingOracle` object.
fn lease<'a>(object: &'a Bound<'_, PyAny>) -> PyResult<Box<dyn SearchOracle + 'a>> {
    let oracle = object.cast::<PyCountingOracle>()?;
    Ok(Box::new(Lease(oracle.get())))
}

/// Registers `CountingOracle`'s lease under other kinds and classes, for
/// the tests of the registry's refusals.
#[pyclass(frozen, module = "fhy_example_aggregate", name = "OracleRegistrar")]
struct PyOracleRegistrar;

#[pymethods]
impl PyOracleRegistrar {
    /// Register `cls` as the oracle kind `kind` of `module`, with
    /// `CountingOracle`'s lease.
    #[staticmethod]
    fn register_oracle(
        module: &Bound<'_, PyModule>,
        kind: &str,
        cls: &Bound<'_, PyType>,
    ) -> PyResult<()> {
        convert::search_space::register_oracle_kind(module, kind, cls, lease)
    }
}

/// Add the oracle's classes to `module` and register its kind.
///
/// # Errors
///
/// Raises whatever adding a class or registering the kind raises.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<PyCountingOracle>()?;
    module.add_class::<PyOracleRegistrar>()?;
    PyOracleRegistrar::register_oracle(module, COUNTING_ORACLE, &py.get_type::<PyCountingOracle>())
}
