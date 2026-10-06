//! `fhy_core._rs.ChoiceDomain`, `OrderDomain`, `StridedRun` and
//! `StridedDomain`: the domains of a search's steps, and their conversion
//! to and from the core's [`StepDomain`].
//!
//! A domain keeps the objects it was built from and returns them: an
//! object that reads as a core value is that value, any other object an
//! opaque value compared by its own `==`, so an `eq=False` object matches
//! only itself.

use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyBool, PyInt, PyTuple};

use fhy_core::constraint::Value;
use fhy_core::expression::BigInt;
use fhy_core::search_space::{
    ChoiceDomain, Coordinate, OrderDomain, StepDomain, StepDomainError, StridedDomain, StridedRun,
};

use crate::constraint::{read_bound_value, value_to_python};
use crate::expression::{big_int_to_python, read_big_int};
use crate::util::gc::{Slots, collect_slots};
use crate::util::pending::with_pending_errors;
use crate::util::python::read_type_name;

use super::errors::step_domain_error_to_py;

/// Return the integer `object`, `what` naming it in the `TypeError` raised
/// for an object that is no `int` or is a `bool`.
pub(super) fn read_int(object: &Bound<'_, PyAny>, what: &str) -> PyResult<BigInt> {
    if object.is_instance_of::<PyBool>() || !object.is_instance_of::<PyInt>() {
        return Err(PyTypeError::new_err(format!(
            "{what} must be an int, got {}.",
            read_type_name(object)
        )));
    }
    read_big_int(object)
}

/// Return the index `object`, an `int` as [`read_int`] reads it, or `None`
/// for one that is negative or wider than 64 bits, which no domain has.
fn read_index(object: &Bound<'_, PyAny>, what: &str) -> PyResult<Option<u64>> {
    Ok(u64::try_from(&read_int(object, what)?).ok())
}

/// Return the coordinate `object` names: an `int` is an index, a tuple of
/// `int`s an ordering's positions; `None` for an integer no domain has a
/// coordinate for.
///
/// # Errors
///
/// Raises `TypeError` naming `what` for any other object.
pub(super) fn read_coordinate(
    object: &Bound<'_, PyAny>,
    what: &str,
) -> PyResult<Option<Coordinate>> {
    let wrong = || {
        PyTypeError::new_err(format!(
            "{what} must be an int or a tuple of ints, got {}.",
            read_type_name(object)
        ))
    };
    if let Ok(positions) = object.cast::<PyTuple>() {
        let mut read = Vec::with_capacity(positions.len());
        let mut is_in_range = true;
        for position in positions.iter() {
            let position = read_int(&position, what).map_err(|_not_an_int| wrong())?;
            match u32::try_from(&position) {
                Ok(position) => read.push(position),
                Err(_out_of_range) => is_in_range = false,
            }
        }
        return Ok(is_in_range.then(|| Coordinate::Order(read.into())));
    }
    let index = read_index(object, what).map_err(|_not_an_int| wrong())?;
    Ok(index.map(Coordinate::Index))
}

/// Return the Python object of `coordinate`: an `int`, or a tuple of
/// `int`s for an ordering.
pub(super) fn coordinate_to_python<'py>(
    py: Python<'py>,
    coordinate: &Coordinate,
) -> PyResult<Bound<'py, PyAny>> {
    match coordinate {
        Coordinate::Index(index) => Ok(index.into_pyobject(py)?.into_any()),
        Coordinate::Order(positions) => Ok(PyTuple::new(py, positions.iter())?.into_any()),
    }
}

/// Return the Python `int` of `value`, a natural number.
pub(super) fn natural_to_python<'py>(
    py: Python<'py>,
    value: impl Into<BigInt>,
) -> PyResult<Bound<'py, PyAny>> {
    big_int_to_python(py, &value.into())
}

/// Return the objects of the iterable `objects` and their core values.
///
/// # Errors
///
/// Raises `TypeError` naming `owner` for an argument that is not
/// iterable.
fn read_objects<'py>(
    objects: &Bound<'py, PyAny>,
    owner: &str,
) -> PyResult<(Bound<'py, PyTuple>, Vec<Value>)> {
    let items = objects.try_iter().map_err(|_not_iterable| {
        PyTypeError::new_err(format!(
            "{owner} takes an iterable, got {}.",
            read_type_name(objects)
        ))
    })?;
    let items = items.collect::<PyResult<Vec<_>>>()?;
    let values = items
        .iter()
        .map(read_bound_value)
        .collect::<PyResult<Vec<_>>>()?;
    Ok((PyTuple::new(objects.py(), items)?, values))
}

/// Return the domain `build` makes of `values`, raising `StepDomainError`
/// for its refusal and the exception a value's `==` raised.
fn build_domain<D>(
    py: Python<'_>,
    build: impl FnOnce() -> Result<D, StepDomainError>,
) -> PyResult<D> {
    with_pending_errors(|| build().map_err(|error| step_domain_error_to_py(py, &error)))
}

/// Return `IndexError` for the index `index` past the last of `count`
/// values.
fn past_the_end(index: &Bound<'_, PyAny>, count: impl std::fmt::Display) -> PyErr {
    PyIndexError::new_err(format!(
        "the index {index} is outside a domain of {count} value(s)"
    ))
}

/// Return `ValueError` for `value`, which is not `what` of a domain.
fn not_in_domain(value: &Bound<'_, PyAny>, what: &str) -> PyErr {
    let text = value.repr().map_or_else(
        |_unprintable| read_type_name(value),
        |text| text.to_string(),
    );
    PyValueError::new_err(format!("{text} is not {what} of the domain"))
}

/// Return the object at `coordinate` of the domain object `domain`: the
/// object given for a choice, the integer for a strided domain, and the
/// tuple of the objects given for an ordering; `None` when `domain` is no
/// domain or holds no such coordinate.
pub(super) fn object_at<'py>(
    domain: &Bound<'py, PyAny>,
    coordinate: &Coordinate,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = domain.py();
    match coordinate {
        Coordinate::Index(index) => {
            let Ok(position) = usize::try_from(*index) else {
                return Ok(None);
            };
            if let Ok(choices) = domain.cast::<PyChoiceDomain>() {
                return choices.get().choices.bind(py).get_item(position).map(Some);
            }
            if let Ok(strided) = domain.cast::<PyStridedDomain>() {
                return strided
                    .get()
                    .domain
                    .value_at(*index)
                    .map(|value| big_int_to_python(py, &value))
                    .transpose();
            }
            Ok(None)
        }
        Coordinate::Order(positions) => {
            let Ok(order) = domain.cast::<PyOrderDomain>() else {
                return Ok(None);
            };
            order
                .get()
                .ordering(py, positions)
                .map(|ordering| ordering.map(Bound::into_any))
        }
    }
}

/// A list of distinct values, backed by the core [`ChoiceDomain`].
#[pyclass(frozen, module = "fhy_core._rs", name = "ChoiceDomain")]
pub(crate) struct PyChoiceDomain {
    domain: ChoiceDomain,
    /// The objects given, in order.
    choices: Py<PyTuple>,
    /// The slots of the opaque values read from the objects.
    slots: Slots,
}

#[pymethods]
impl PyChoiceDomain {
    /// Return the domain of the objects `choices`, in order.
    ///
    /// Raises `TypeError` for an argument that is not iterable and
    /// `StepDomainError` for no choice, a NaN or two equal choices.
    #[new]
    fn new(choices: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (read, slots) = collect_slots(|| read_objects(choices, "ChoiceDomain"));
        let (objects, values) = read?;
        let domain = build_domain(choices.py(), || ChoiceDomain::new(values))?;
        Ok(Self {
            domain,
            choices: objects.unbind(),
            slots,
        })
    }

    /// The objects given, in order.
    #[getter]
    fn choices<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        self.choices.bind(py).clone()
    }

    /// The number of choices.
    #[getter]
    fn cardinality(&self) -> u64 {
        self.domain.cardinality()
    }

    /// Return whether a choice equals `value`.
    fn admits(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        let value = read_bound_value(value)?;
        with_pending_errors(|| Ok(self.domain.admits(&value)))
    }

    /// Return the choice at `index`, the object given.
    ///
    /// Raises `TypeError` for an index that is no `int` and `IndexError`
    /// past the last choice.
    fn value_at(&self, py: Python<'_>, index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let choices = self.choices.bind(py);
        read_index(index, "ChoiceDomain.value_at's index")?
            .and_then(|position| usize::try_from(position).ok())
            .filter(|&position| position < choices.len())
            .map(|position| choices.get_item(position).map(Bound::unbind))
            .unwrap_or_else(|| Err(past_the_end(index, choices.len())))
    }

    /// Return the index of the choice equal to `value`.
    ///
    /// Raises `ValueError` when no choice is.
    fn coordinate_of(&self, value: &Bound<'_, PyAny>) -> PyResult<u64> {
        let read = read_bound_value(value)?;
        with_pending_errors(|| Ok(self.domain.coordinate_of(&read)))?
            .ok_or_else(|| not_in_domain(value, "a choice"))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("ChoiceDomain({})", self.choices.bind(py).repr()?))
    }

    /// Visit the objects held, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.choices)?;
        self.slots.traverse(&visit)
    }
}

/// The orderings of distinct elements, backed by the core [`OrderDomain`].
#[pyclass(frozen, module = "fhy_core._rs", name = "OrderDomain")]
pub(crate) struct PyOrderDomain {
    domain: OrderDomain,
    /// The objects given, in their canonical order.
    elements: Py<PyTuple>,
    /// The slots of the opaque values read from the objects.
    slots: Slots,
}

#[pymethods]
impl PyOrderDomain {
    /// Return the domain of the orderings of the objects `elements`.
    ///
    /// Raises `TypeError` for an argument that is not iterable and
    /// `StepDomainError` for no element, a NaN or two equal elements.
    #[new]
    fn new(elements: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (read, slots) = collect_slots(|| read_objects(elements, "OrderDomain"));
        let (objects, values) = read?;
        let domain = build_domain(elements.py(), || OrderDomain::new(values))?;
        Ok(Self {
            domain,
            elements: objects.unbind(),
            slots,
        })
    }

    /// The objects given, in their canonical order.
    #[getter]
    fn elements<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        self.elements.bind(py).clone()
    }

    /// The number of orderings, an `int`.
    #[getter]
    fn cardinality(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        natural_to_python(py, self.domain.cardinality()).map(Bound::unbind)
    }

    /// Return whether `value` is a tuple holding each element once.
    fn admits(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        if !value.is_instance_of::<PyTuple>() {
            return Ok(false);
        }
        let value = read_bound_value(value)?;
        with_pending_errors(|| Ok(self.domain.admits(&value)))
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
        let what = "OrderDomain.value_at's positions";
        if !positions.is_instance_of::<PyTuple>() {
            return Err(PyTypeError::new_err(format!(
                "{what} must be a tuple of ints, got {}.",
                read_type_name(positions)
            )));
        }
        let ordering = match read_coordinate(positions, what)? {
            Some(Coordinate::Order(read)) => self.ordering(py, &read)?,
            _ => None,
        };
        ordering.ok_or_else(|| {
            PyValueError::new_err(format!(
                "{} is not a permutation of the positions of {} element(s)",
                positions
                    .repr()
                    .map_or_else(|_| "?".to_owned(), |text| text.to_string()),
                self.domain.elements().len()
            ))
        })
    }

    /// Return the positions `value` orders the elements in.
    ///
    /// Raises `ValueError` when `value` is not an ordering of them.
    fn coordinate_of<'py>(
        &self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let read = read_bound_value(value)?;
        let positions = with_pending_errors(|| Ok(self.domain.coordinate_of(&read)))?
            .ok_or_else(|| not_in_domain(value, "an ordering"))?;
        PyTuple::new(py, positions.iter())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("OrderDomain({})", self.elements.bind(py).repr()?))
    }

    /// Visit the objects held, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.elements)?;
        self.slots.traverse(&visit)
    }
}

impl PyOrderDomain {
    /// Return the tuple of the objects given in the order `positions`
    /// names, or `None` when it is no permutation of their positions.
    fn ordering<'py>(
        &self,
        py: Python<'py>,
        positions: &[u32],
    ) -> PyResult<Option<Bound<'py, PyTuple>>> {
        if !StepDomain::Order(self.domain.clone()).contains(&Coordinate::Order(positions.into())) {
            return Ok(None);
        }
        let elements = self.elements.bind(py);
        let ordered = positions
            .iter()
            .map(|&position| elements.get_item(position as usize))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, ordered).map(Some)
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
        let py = start.py();
        let start = read_int(start, "StridedRun's start")?;
        let stop = read_int(stop, "StridedRun's stop")?;
        let stride = match stride {
            Some(stride) => read_int(stride, "StridedRun's stride")?,
            None => BigInt::from(1_u8),
        };
        let stride = stride
            .to_biguint()
            .ok_or_else(|| step_domain_error_to_py(py, &StepDomainError::ZeroStride))?;
        let run = StridedRun::new(start, stop, stride)
            .map_err(|error| step_domain_error_to_py(py, &error))?;
        Ok(Self { run })
    }

    /// The run's first integer.
    #[getter]
    fn start(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        big_int_to_python(py, self.run.start()).map(Bound::unbind)
    }

    /// The bound every integer of the run is below.
    #[getter]
    fn stop(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        big_int_to_python(py, self.run.stop()).map(Bound::unbind)
    }

    /// The distance between consecutive integers.
    #[getter]
    fn stride(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        natural_to_python(py, self.run.stride().clone()).map(Bound::unbind)
    }

    /// The number of integers the run holds.
    #[getter]
    fn width(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        natural_to_python(py, self.run.width()).map(Bound::unbind)
    }

    /// Return whether `value` is an integer of the run; a `bool` is not.
    fn admits(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        if value.is_instance_of::<PyBool>() || !value.is_instance_of::<PyInt>() {
            return Ok(false);
        }
        Ok(self.run.admits(&read_big_int(value)?))
    }

    fn __repr__(&self) -> String {
        format!(
            "StridedRun({}, {}, {})",
            self.run.start(),
            self.run.stop(),
            self.run.stride()
        )
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
        let py = runs.py();
        let wrong = |object: &Bound<'_, PyAny>| {
            PyTypeError::new_err(format!(
                "StridedDomain takes an iterable of StridedRuns, got {}.",
                read_type_name(object)
            ))
        };
        let items = runs
            .try_iter()
            .map_err(|_not_iterable| wrong(runs))?
            .collect::<PyResult<Vec<_>>>()?;
        let core = items
            .iter()
            .map(|item| {
                item.cast::<PyStridedRun>()
                    .map(|run| run.get().run.clone())
                    .map_err(|_not_a_run| wrong(item))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let domain =
            StridedDomain::new(core).map_err(|error| step_domain_error_to_py(py, &error))?;
        Ok(Self {
            domain,
            runs: PyTuple::new(py, items)?.unbind(),
        })
    }

    /// The runs given, in order.
    #[getter]
    fn runs<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        self.runs.bind(py).clone()
    }

    /// The number of integers the runs hold.
    #[getter]
    fn cardinality(&self) -> u64 {
        self.domain.cardinality()
    }

    /// Return whether `value` is an integer of the runs; a `bool` is not.
    fn admits(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        if value.is_instance_of::<PyBool>() || !value.is_instance_of::<PyInt>() {
            return Ok(false);
        }
        Ok(self.domain.coordinate_of(&read_big_int(value)?).is_some())
    }

    /// Return the integer at `index`, counting the runs' integers in order.
    ///
    /// Raises `TypeError` for an index that is no `int` and `IndexError`
    /// past the last.
    fn value_at(&self, py: Python<'_>, index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let value = read_index(index, "StridedDomain.value_at's index")?
            .and_then(|position| self.domain.value_at(position))
            .ok_or_else(|| past_the_end(index, self.domain.cardinality()))?;
        big_int_to_python(py, &value).map(Bound::unbind)
    }

    /// Return the position of the integer `value`.
    ///
    /// Raises `ValueError` when the runs do not hold it.
    fn coordinate_of(&self, value: &Bound<'_, PyAny>) -> PyResult<u64> {
        let integer = if value.is_instance_of::<PyBool>() || !value.is_instance_of::<PyInt>() {
            None
        } else {
            Some(read_big_int(value)?)
        };
        integer
            .and_then(|integer| self.domain.coordinate_of(&integer))
            .ok_or_else(|| not_in_domain(value, "an integer"))
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("StridedDomain({})", self.runs.bind(py).repr()?))
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
    if let Ok(domain) = object.cast::<PyChoiceDomain>() {
        return Ok(StepDomain::from(domain.get().domain.clone()));
    }
    if let Ok(domain) = object.cast::<PyOrderDomain>() {
        return Ok(StepDomain::from(domain.get().domain.clone()));
    }
    if let Ok(domain) = object.cast::<PyStridedDomain>() {
        return Ok(StepDomain::from(domain.get().domain.clone()));
    }
    Err(PyTypeError::new_err(format!(
        "a step domain must be a ChoiceDomain, an OrderDomain or a StridedDomain, got {}.",
        read_type_name(object)
    )))
}

/// Return whether `value` is or holds an opaque value.
pub(super) fn holds_opaque(value: &Value) -> bool {
    match value {
        Value::Opaque(_) => true,
        Value::Tuple(values) | Value::FrozenSet(values) => values.iter().any(holds_opaque),
        _ => false,
    }
}

/// Return the core domain of the domain object `object`, its opaque values
/// read anew from the objects it keeps, so their slots belong to the
/// innermost `collect_slots`: for a holder that must own what it keeps
/// alive. A domain with no opaque value is shared as it is.
///
/// # Errors
///
/// Raises `TypeError` for an object that is no domain object, and what
/// reading a value raises.
pub(super) fn owned_step_domain(object: &Bound<'_, PyAny>) -> PyResult<StepDomain> {
    let py = object.py();
    if let Ok(domain) = object.cast::<PyChoiceDomain>() {
        let domain = domain.get();
        if domain.domain.values().iter().any(holds_opaque) {
            let (_, values) = read_objects(domain.choices.bind(py).as_any(), "ChoiceDomain")?;
            return build_domain(py, || ChoiceDomain::new(values)).map(StepDomain::from);
        }
    }
    if let Ok(domain) = object.cast::<PyOrderDomain>() {
        let domain = domain.get();
        if domain.domain.elements().iter().any(holds_opaque) {
            let (_, values) = read_objects(domain.elements.bind(py).as_any(), "OrderDomain")?;
            return build_domain(py, || OrderDomain::new(values)).map(StepDomain::from);
        }
    }
    step_domain_from_python(object)
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
    let objects = |values: &[Value]| {
        let objects = values
            .iter()
            .map(|value| value_to_python(py, value))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, objects).map(Bound::unbind)
    };
    match domain {
        StepDomain::Choice(domain) => Bound::new(
            py,
            PyChoiceDomain {
                choices: objects(domain.values())?,
                domain: domain.clone(),
                slots: Slots::default(),
            },
        )
        .map(Bound::into_any),
        StepDomain::Order(domain) => Bound::new(
            py,
            PyOrderDomain {
                elements: objects(domain.elements())?,
                domain: domain.clone(),
                slots: Slots::default(),
            },
        )
        .map(Bound::into_any),
        StepDomain::Strided(domain) => {
            let runs = domain
                .runs()
                .iter()
                .map(|run| Bound::new(py, PyStridedRun { run: run.clone() }))
                .collect::<PyResult<Vec<_>>>()?;
            Bound::new(
                py,
                PyStridedDomain {
                    runs: PyTuple::new(py, runs)?.unbind(),
                    domain: domain.clone(),
                },
            )
            .map(Bound::into_any)
        }
        _ => Err(PyTypeError::new_err(
            "the step domain has a shape this binding does not know",
        )),
    }
}
