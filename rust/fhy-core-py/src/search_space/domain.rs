//! `fhy_core._rs.ChoiceDomain`, `OrderDomain`, `StridedRun` and
//! `StridedDomain`: the domains of a search's steps, and their conversion
//! to and from the core's [`StepDomain`].
//!
//! A domain keeps the objects it was built from and returns them: an
//! object that reads as a core value is that value, any other object an
//! opaque value compared by its own `==`, so an `eq=False` object matches
//! only itself.

#![expect(
    unused_variables,
    dead_code,
    reason = "interface stub: the bodies are todo!() until the implementation"
)]

use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::PyTuple;

use fhy_core::search_space::{ChoiceDomain, OrderDomain, StepDomain, StridedDomain, StridedRun};

/// A list of distinct values, backed by the core [`ChoiceDomain`].
#[pyclass(frozen, module = "fhy_core._rs", name = "ChoiceDomain")]
pub(crate) struct PyChoiceDomain {
    domain: ChoiceDomain,
    /// The objects given, in order.
    choices: Py<PyTuple>,
}

#[pymethods]
impl PyChoiceDomain {
    /// Return the domain of the objects `choices`, in order.
    ///
    /// Raises `TypeError` for an argument that is not iterable and
    /// `StepDomainError` for no choice, a NaN or two equal choices.
    #[new]
    fn new(choices: &Bound<'_, PyAny>) -> PyResult<Self> {
        todo!()
    }

    /// The objects given, in order.
    #[getter]
    fn choices<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        todo!()
    }

    /// The number of choices.
    #[getter]
    fn cardinality(&self) -> u64 {
        todo!()
    }

    /// Return whether a choice equals `value`.
    fn admits(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        todo!()
    }

    /// Return the choice at `index`, the object given.
    ///
    /// Raises `TypeError` for an index that is no `int` and `IndexError`
    /// past the last choice.
    fn value_at(&self, py: Python<'_>, index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        todo!()
    }

    /// Return the index of the choice equal to `value`.
    ///
    /// Raises `ValueError` when no choice is.
    fn coordinate_of(&self, value: &Bound<'_, PyAny>) -> PyResult<u64> {
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
        visit.call(&self.choices)
    }
}

/// The orderings of distinct elements, backed by the core [`OrderDomain`].
#[pyclass(frozen, module = "fhy_core._rs", name = "OrderDomain")]
pub(crate) struct PyOrderDomain {
    domain: OrderDomain,
    /// The objects given, in their canonical order.
    elements: Py<PyTuple>,
}

#[pymethods]
impl PyOrderDomain {
    /// Return the domain of the orderings of the objects `elements`.
    ///
    /// Raises `TypeError` for an argument that is not iterable and
    /// `StepDomainError` for no element, a NaN or two equal elements.
    #[new]
    fn new(elements: &Bound<'_, PyAny>) -> PyResult<Self> {
        todo!()
    }

    /// The objects given, in their canonical order.
    #[getter]
    fn elements<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        todo!()
    }

    /// The number of orderings, an `int`.
    #[getter]
    fn cardinality(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        todo!()
    }

    /// Return whether `value` is a tuple holding each element once.
    fn admits(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        todo!()
    }

    /// Return the ordering `positions` names: a tuple of the objects given.
    ///
    /// Raises `TypeError` for positions that are no tuple of `int`s and
    /// `ValueError` for one that is not a permutation of the positions.
    fn value_at<'py>(
        &self,
        py: Python<'py>,
        positions: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
    }

    /// Return the positions `value` orders the elements in.
    ///
    /// Raises `ValueError` when `value` is not an ordering of them.
    fn coordinate_of<'py>(
        &self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
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
        visit.call(&self.elements)
    }
}

/// A run of integers below a stop by a stride, backed by the core
/// [`StridedRun`].
#[pyclass(frozen, module = "fhy_core._rs", name = "StridedRun")]
pub(crate) struct PyStridedRun {
    run: StridedRun,
}

#[pymethods]
impl PyStridedRun {
    /// Return the run from `start` below `stop` by `stride`.
    ///
    /// Raises `TypeError` for a bound that is no `int` (a `bool` included)
    /// and `StepDomainError` for an empty run or a stride below 1.
    #[new]
    #[pyo3(signature = (start, stop, stride = None))]
    fn new(
        start: &Bound<'_, PyAny>,
        stop: &Bound<'_, PyAny>,
        stride: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        todo!()
    }

    /// The run's first integer.
    #[getter]
    fn start(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        todo!()
    }

    /// The bound every integer of the run is below.
    #[getter]
    fn stop(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        todo!()
    }

    /// The distance between consecutive integers.
    #[getter]
    fn stride(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        todo!()
    }

    /// The number of integers the run holds.
    #[getter]
    fn width(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        todo!()
    }

    /// Return whether `value` is an integer of the run; a `bool` is not.
    fn admits(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        todo!()
    }

    fn __repr__(&self) -> String {
        todo!()
    }
}

/// A union of disjoint, ascending strided runs, backed by the core
/// [`StridedDomain`].
#[pyclass(frozen, module = "fhy_core._rs", name = "StridedDomain")]
pub(crate) struct PyStridedDomain {
    domain: StridedDomain,
    /// The `StridedRun` objects given, in order.
    runs: Py<PyTuple>,
}

#[pymethods]
impl PyStridedDomain {
    /// Return the domain of the `StridedRun`s `runs`.
    ///
    /// Raises `TypeError` for an argument that is no iterable of
    /// `StridedRun`s and `StepDomainError` for no run, runs out of order or
    /// overlapping, or more than `2**64 - 1` integers.
    #[new]
    fn new(runs: &Bound<'_, PyAny>) -> PyResult<Self> {
        todo!()
    }

    /// The runs given, in order.
    #[getter]
    fn runs<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        todo!()
    }

    /// The number of integers the runs hold.
    #[getter]
    fn cardinality(&self) -> u64 {
        todo!()
    }

    /// Return whether `value` is an integer of the runs; a `bool` is not.
    fn admits(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        todo!()
    }

    /// Return the integer at `index`, counting the runs' integers in order.
    ///
    /// Raises `TypeError` for an index that is no `int` and `IndexError`
    /// past the last.
    fn value_at(&self, py: Python<'_>, index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        todo!()
    }

    /// Return the position of the integer `value`.
    ///
    /// Raises `ValueError` when the runs do not hold it.
    fn coordinate_of(&self, value: &Bound<'_, PyAny>) -> PyResult<u64> {
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
        visit.call(&self.runs)
    }
}

/// Return the core domain of the domain object `object`.
///
/// # Errors
///
/// Raises `TypeError` for an object that is none of the three domain
/// classes.
pub(crate) fn step_domain_from_python(object: &Bound<'_, PyAny>) -> PyResult<StepDomain> {
    todo!()
}

/// Return a new domain object of the core domain `domain`, its values
/// written as Python objects.
///
/// # Errors
///
/// Raises what writing a value raises.
pub(crate) fn step_domain_to_python<'py>(
    py: Python<'py>,
    domain: &StepDomain,
) -> PyResult<Bound<'py, PyAny>> {
    todo!()
}
