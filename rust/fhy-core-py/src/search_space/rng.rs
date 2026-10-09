//! `fhy_core._rs.Rng`: the generator every random draw of a search takes
//! its numbers from.

use std::num::NonZeroU64;
use std::sync::{Mutex, PoisonError};

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyInt, PyList, PyTuple};

use fhy_core::search_space::Rng;

use crate::expression::{big_int_to_python, read_big_int};
use crate::util::python::read_type_name;

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
    /// Return the generator seeded with `seed`.
    pub(crate) fn seeded(seed: u64) -> Self {
        Self {
            seed,
            rng: Mutex::new(Rng::new(seed)),
        }
    }

    /// Return the generator seeded with the seed object `seed`, read as
    /// `Rng(seed)` reads it.
    ///
    /// # Errors
    ///
    /// Raises what `Rng(seed)` raises.
    pub(crate) fn from_seed_object(seed: &Bound<'_, PyAny>) -> PyResult<Self> {
        read_seed(seed).map(Self::seeded)
    }

    /// Run `draw` on the generator.
    ///
    /// `draw` must not call Python: the generator is locked meanwhile.
    pub(crate) fn with_rng<T>(&self, draw: impl FnOnce(&mut Rng) -> T) -> T {
        draw(&mut self.rng.lock().unwrap_or_else(PoisonError::into_inner))
    }

    /// Return a copy of the generator in its current state.
    pub(crate) fn snapshot(&self) -> Rng {
        self.with_rng(|rng| rng.clone())
    }

    /// Move the generator to the state of `rng`, a copy that drew on.
    pub(crate) fn restore(&self, rng: Rng) {
        self.with_rng(|current| *current = rng);
    }

    /// Return the generator's state, the number its wire form records.
    fn state(&self) -> PyResult<u64> {
        let wire = serde_json::to_value(self.snapshot())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        wire.get("state")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| PyValueError::new_err("the generator's wire form holds no state"))
    }
}

/// Return the seed `seed`, an `int` in `[0, 2**64)`.
///
/// # Errors
///
/// Raises `TypeError` for an object that is no `int` (a `bool` included)
/// and `ValueError` for one outside that range.
fn read_seed(seed: &Bound<'_, PyAny>) -> PyResult<u64> {
    if seed.is_instance_of::<PyBool>() || !seed.is_instance_of::<PyInt>() {
        return Err(PyTypeError::new_err(format!(
            "the seed must be an int, got {}.",
            read_type_name(seed)
        )));
    }
    u64::try_from(&read_big_int(seed)?).map_err(|_out_of_range| {
        PyValueError::new_err(format!("the seed {seed} is outside [0, 2**64)"))
    })
}

#[pymethods]
impl PyRng {
    /// Return the generator seeded with `seed`, an `int` in `[0, 2**64)`.
    ///
    /// Raises `TypeError` for a seed that is no `int` (a `bool` included)
    /// and `ValueError` for one outside that range.
    #[new]
    fn new(seed: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_seed_object(seed)
    }

    /// The seed the generator started from.
    #[getter]
    pub(crate) const fn seed(&self) -> u64 {
        self.seed
    }

    /// Return the next number of the stream.
    fn next_u64(&self) -> u64 {
        self.with_rng(Rng::next_u64)
    }

    /// Return a number drawn uniformly from `[0, bound)`, `bound` a positive
    /// `int` (`ValueError` otherwise).
    fn below(&self, bound: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = bound.py();
        if bound.is_instance_of::<PyBool>() || !bound.is_instance_of::<PyInt>() {
            return Err(PyTypeError::new_err(format!(
                "the bound must be an int, got {}.",
                read_type_name(bound)
            )));
        }
        let read = read_big_int(bound)?;
        let not_positive = || PyValueError::new_err(format!("the bound {bound} is not positive"));
        if let Ok(word) = u64::try_from(&read) {
            let word = NonZeroU64::new(word).ok_or_else(not_positive)?;
            return Ok(self
                .with_rng(|rng| rng.below(word))
                .into_pyobject(py)?
                .into_any()
                .unbind());
        }
        let wide = read.to_biguint().ok_or_else(not_positive)?;
        let drawn = self.with_rng(|rng| rng.below_big(&wide));
        big_int_to_python(py, &drawn.into()).map(Bound::unbind)
    }

    /// Shuffle the list `items` in place, uniformly.
    fn shuffle(&self, items: &Bound<'_, PyList>) -> PyResult<()> {
        let mut order: Vec<Bound<'_, PyAny>> = items.iter().collect();
        self.with_rng(|rng| rng.shuffle(&mut order));
        for (index, item) in order.into_iter().enumerate() {
            items.set_item(index, item)?;
        }
        Ok(())
    }

    /// Return a generator of an independent stream, seeded with this one's
    /// next number.
    fn split(&self) -> Self {
        Self::seeded(self.next_u64())
    }

    /// Pickle as a call of `_from_state` with the seed and the current
    /// state, so the copy continues the stream.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.get();
        let restore = slf.get_type().getattr(intern!(py, "_from_state"))?;
        let arguments = PyTuple::new(py, [this.seed, this.state()?])?;
        PyTuple::new(py, [restore, arguments.into_any()])
    }

    /// Return the generator of seed `seed` in the state `state`, which
    /// `__reduce__` writes.
    #[staticmethod]
    fn _from_state(seed: u64, state: u64) -> Self {
        Self {
            seed,
            rng: Mutex::new(Rng::new(state)),
        }
    }

    fn __repr__(&self) -> String {
        format!("Rng(seed={})", self.seed)
    }
}
