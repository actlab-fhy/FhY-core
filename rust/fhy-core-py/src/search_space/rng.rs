//! `fhy_core._rs.Rng`: the generator every random draw of a search takes
//! its numbers from.

#![expect(
    unused_variables,
    dead_code,
    reason = "interface stub: the bodies are todo!() until the implementation"
)]

use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use fhy_core::search_space::Rng;

/// The core [`Rng`], seeded from Python, whose stream is the same from
/// Python and from Rust.
#[pyclass(frozen, module = "fhy_core._rs", name = "Rng")]
pub(crate) struct PyRng {
    /// The seed the generator started from.
    seed: u64,
    /// The generator, in its current state.
    rng: Mutex<Rng>,
}

impl PyRng {
    /// Run `draw` on the generator.
    pub(crate) fn with_rng<T>(&self, draw: impl FnOnce(&mut Rng) -> T) -> T {
        todo!()
    }
}

#[pymethods]
impl PyRng {
    /// Return the generator seeded with `seed`, an `int` in `[0, 2**64)`.
    ///
    /// Raises `TypeError` for a seed that is no `int` (a `bool` included)
    /// and `ValueError` for one outside that range.
    #[new]
    fn new(seed: &Bound<'_, PyAny>) -> PyResult<Self> {
        todo!()
    }

    /// The seed the generator started from.
    #[getter]
    fn seed(&self) -> u64 {
        todo!()
    }

    /// Return the next number of the stream.
    fn next_u64(&self) -> u64 {
        todo!()
    }

    /// Return a number drawn uniformly from `[0, bound)`, `bound` a positive
    /// `int` (`ValueError` otherwise).
    fn below(&self, bound: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        todo!()
    }

    /// Shuffle the list `items` in place, uniformly.
    fn shuffle(&self, items: &Bound<'_, PyList>) -> PyResult<()> {
        todo!()
    }

    /// Return a generator of an independent stream, seeded with this one's
    /// next number.
    fn split(&self) -> Self {
        todo!()
    }

    /// Pickle as a call of `_from_state` with the seed and the current
    /// state, so the copy continues the stream.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
    }

    /// Return the generator of seed `seed` in the state `state`, which
    /// `__reduce__` writes.
    #[staticmethod]
    fn _from_state(seed: u64, state: u64) -> Self {
        todo!()
    }

    fn __repr__(&self) -> String {
        todo!()
    }
}
