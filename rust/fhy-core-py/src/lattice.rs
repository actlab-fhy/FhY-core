//! `PyO3` classes for [`fhy_core::lattice`]: `fhy_core._rs.PartiallyOrderedSet`
//! and `Lattice`, the bases of `fhy_core.utils.poset.PartiallyOrderedSet` and
//! `fhy_core.lattice.Lattice` (pattern P2, D-S11-5).
//!
//! The elements are Python objects, so each class keeps them in a Python
//! `dict` from element to its position, which keeps Python's hashing and
//! `==`, and a list of the objects by position; the core's order runs over
//! the positions. The core never holds a Python object.

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};

use fhy_core::lattice::{Lattice, MissingBound, OrderError, PartiallyOrderedSet};

/// The source of the diagnostics `Lattice.verify` reports.
const VERIFY_SOURCE: &str = "fhy_core.lattice.Lattice.verify";

/// The element objects of an order and their positions.
struct Elements {
    /// The position of each element, keyed by the element.
    positions: Py<PyDict>,
    /// The elements, by position.
    objects: Vec<Py<PyAny>>,
}

impl Elements {
    fn new(py: Python<'_>) -> Self {
        Self {
            positions: PyDict::new(py).unbind(),
            objects: Vec::new(),
        }
    }

    /// Return whether `element` is a member; an unhashable object is none.
    fn contains(&self, element: &Bound<'_, PyAny>) -> PyResult<bool> {
        match self.positions.bind(element.py()).contains(element) {
            Ok(contains) => Ok(contains),
            Err(error) if error.is_instance_of::<PyTypeError>(element.py()) => Ok(false),
            Err(error) => Err(error),
        }
    }

    /// Return the position of `element`, or raise the `ValueError` of a
    /// non-member.
    fn position(&self, element: &Bound<'_, PyAny>) -> PyResult<usize> {
        match self.positions.bind(element.py()).get_item(element) {
            Ok(Some(position)) => position.extract(),
            Ok(None) => Err(order_error_to_python(
                element.py(),
                &OrderError::NotAMember(0),
                std::slice::from_ref(element),
            )?),
            Err(error) if error.is_instance_of::<PyTypeError>(element.py()) => {
                Err(order_error_to_python(
                    element.py(),
                    &OrderError::NotAMember(0),
                    std::slice::from_ref(element),
                )?)
            }
            Err(error) => Err(error),
        }
    }

    /// Add `element` at the next position, and return that position.
    ///
    /// Raises the `ValueError` of a member, and `TypeError` for an
    /// unhashable element.
    fn insert(&mut self, element: &Bound<'_, PyAny>) -> PyResult<usize> {
        let py = element.py();
        let positions = self.positions.bind(py);
        if positions.contains(element)? {
            return Err(order_error_to_python(
                py,
                &OrderError::AlreadyAMember(0),
                std::slice::from_ref(element),
            )?);
        }
        let position = self.objects.len();
        positions.set_item(element, position)?;
        self.objects.push(element.clone().unbind());
        Ok(position)
    }

    /// Return the object at `position`.
    fn object<'py>(&self, py: Python<'py>, position: usize) -> Bound<'py, PyAny> {
        self.objects[position].bind(py).clone()
    }

    /// Return the objects at `positions`, as a list.
    fn list<'py>(
        &self,
        py: Python<'py>,
        positions: impl IntoIterator<Item = usize>,
    ) -> PyResult<Bound<'py, PyList>> {
        PyList::new(
            py,
            positions
                .into_iter()
                .map(|position| self.objects[position].bind(py)),
        )
    }

    /// Return the positions ranked by `key` called on each element: the rank
    /// of an element is its place when the `(key, position)` pairs are
    /// sorted, so equal keys keep insertion order.
    fn ranks(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
        let pairs = PyList::empty(py);
        for (position, object) in self.objects.iter().enumerate() {
            pairs.append((key.call1((object.bind(py),))?, position))?;
        }
        pairs.sort()?;
        let mut ranks = vec![0; self.objects.len()];
        for (rank, pair) in pairs.iter().enumerate() {
            let position: usize = pair.get_item(1)?.extract()?;
            ranks[position] = rank;
        }
        Ok(ranks)
    }
}

/// Return the Python exception of `error`, whose elements are the objects
/// `elements`, in the order the error names them: `ValueError` for a member
/// or a non-member, and `RuntimeError` for a cycle, each with the core's
/// text around the elements' `str`.
fn order_error_to_python(
    py: Python<'_>,
    error: &OrderError<usize>,
    elements: &[Bound<'_, PyAny>],
) -> PyResult<PyErr> {
    let text = |index: usize| -> PyResult<String> { Ok(elements[index].str()?.to_string()) };
    let _ = py;
    Ok(match error {
        OrderError::AlreadyAMember(_) => PyValueError::new_err(format!(
            "{} is already a member of the partially ordered set",
            text(0)?
        )),
        OrderError::NotAMember(_) => PyValueError::new_err(format!(
            "{} is not a member of the partially ordered set",
            text(0)?
        )),
        OrderError::WouldCycle { .. } => PyRuntimeError::new_err(format!(
            "ordering {} below {} would close a cycle",
            text(0)?,
            text(1)?
        )),
        _ => PyRuntimeError::new_err(error.to_string()),
    })
}

/// Add the order `lower` below `upper` to `poset`, whose elements are
/// `elements`.
fn add_order(
    elements: &Elements,
    poset: &mut PartiallyOrderedSet<usize>,
    lower: &Bound<'_, PyAny>,
    upper: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let lower_position = elements.position(lower)?;
    let upper_position = elements.position(upper)?;
    poset
        .add_order(&lower_position, &upper_position)
        .map_err(|error| {
            order_error_to_python(lower.py(), &error, &[lower.clone(), upper.clone()])
                .unwrap_or_else(|conversion| conversion)
        })
}

/// Return the class itself, so `X[T]` subscripts at run time.
fn subscript<'py>(cls: &Bound<'py, PyType>) -> Bound<'py, PyType> {
    cls.clone()
}

// ---------------------------------------------------------------------------
// PartiallyOrderedSet
// ---------------------------------------------------------------------------

/// A set of Python objects and a partial order over them, backed by the
/// core's [`PartiallyOrderedSet`] over their positions.
#[pyclass(subclass, module = "fhy_core._rs", name = "PartiallyOrderedSet")]
pub(crate) struct PyPartiallyOrderedSet {
    elements: Elements,
    poset: PartiallyOrderedSet<usize>,
}

#[pymethods]
impl PyPartiallyOrderedSet {
    /// Create an empty set; arguments are accepted and ignored, so a
    /// subclass with its own `__init__` constructs.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(
        py: Python<'_>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Self {
        Self {
            elements: Elements::new(py),
            poset: PartiallyOrderedSet::new(),
        }
    }

    #[classmethod]
    fn __class_getitem__<'py>(
        cls: &Bound<'py, PyType>,
        item: &Bound<'py, PyAny>,
    ) -> Bound<'py, PyType> {
        let _ = item;
        subscript(cls)
    }

    /// Return whether `element` is a member; an unhashable object is none.
    fn __contains__(&self, element: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.elements.contains(element)
    }

    fn __len__(&self) -> usize {
        self.poset.len()
    }

    /// Iterate the elements in a topological order in which the element
    /// added first comes first among those that can come next.
    fn __iter__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let slf = slf.borrow();
        let py = slf.py();
        slf.elements
            .list(py, slf.poset.iter().copied())?
            .try_iter()
            .map(Bound::into_any)
    }

    /// Iterate the elements in a topological order in which, among those
    /// that can come next, the one with the least `key(element)` comes
    /// first, and of equal keys the one added first.
    #[pyo3(signature = (key = None))]
    fn iter_stable<'py>(
        slf: &Bound<'py, Self>,
        key: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let slf = slf.borrow();
        let py = slf.py();
        let repr = py
            .import(intern!(py, "builtins"))?
            .getattr(intern!(py, "repr"))?;
        let ranks = slf.elements.ranks(py, key.unwrap_or(&repr))?;
        slf.elements
            .list(
                py,
                slf.poset.iter_by_key(|&position| ranks[position]).copied(),
            )?
            .try_iter()
            .map(Bound::into_any)
    }

    /// Add `element`, ordered with no other element.
    ///
    /// Raises `ValueError` if it is a member, and `TypeError` if it is not
    /// hashable.
    fn add_element(&mut self, element: &Bound<'_, PyAny>) -> PyResult<()> {
        let position = self.elements.insert(element)?;
        self.poset
            .add_element(position)
            .unwrap_or_else(|_member| unreachable!("a new position is no member"));
        Ok(())
    }

    /// Order `lower` below `upper`.
    ///
    /// Raises `ValueError` if either is not a member, and `RuntimeError` if
    /// `upper` is already at most `lower`.
    fn add_order(&mut self, lower: &Bound<'_, PyAny>, upper: &Bound<'_, PyAny>) -> PyResult<()> {
        add_order(&self.elements, &mut self.poset, lower, upper)
    }

    /// Return whether `lower` is less than or equal to `upper`.
    ///
    /// Raises `ValueError` if either is not a member.
    fn is_less_than(&self, lower: &Bound<'_, PyAny>, upper: &Bound<'_, PyAny>) -> PyResult<bool> {
        let lower_position = self.elements.position(lower)?;
        let upper_position = self.elements.position(upper)?;
        Ok(self
            .poset
            .is_at_most(&lower_position, &upper_position)
            .unwrap_or_else(|_non_member| unreachable!("both are members")))
    }

    /// Return whether `lower` is greater than or equal to `upper`.
    ///
    /// Raises `ValueError` if either is not a member.
    fn is_greater_than(
        &self,
        lower: &Bound<'_, PyAny>,
        upper: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let lower_position = self.elements.position(lower)?;
        let upper_position = self.elements.position(upper)?;
        Ok(self
            .poset
            .is_at_most(&upper_position, &lower_position)
            .unwrap_or_else(|_non_member| unreachable!("both are members")))
    }
}

// ---------------------------------------------------------------------------
// Lattice
// ---------------------------------------------------------------------------

/// A lattice over Python objects, backed by the core's [`Lattice`] over
/// their positions.
#[pyclass(subclass, module = "fhy_core._rs", name = "Lattice")]
pub(crate) struct PyLattice {
    elements: Elements,
    lattice: Lattice<usize>,
}

impl PyLattice {
    /// Return the positions of `x` and `y`, raising the `ValueError` of a
    /// non-member, `x` first.
    fn positions(&self, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<(usize, usize)> {
        Ok((self.elements.position(x)?, self.elements.position(y)?))
    }
}

/// Return `fhy_core.diagnostic`'s class `name`.
fn diagnostic_class<'py>(py: Python<'py>, name: &'static str) -> PyResult<Bound<'py, PyAny>> {
    static MODULE: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    MODULE
        .get_or_try_init(py, || py.import("fhy_core.diagnostic").map(Bound::unbind))?
        .bind(py)
        .getattr(name)
}

#[pymethods]
impl PyLattice {
    /// Create an empty lattice; arguments are accepted and ignored, so a
    /// subclass with its own `__init__` constructs.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(
        py: Python<'_>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Self {
        Self {
            elements: Elements::new(py),
            lattice: Lattice::new(),
        }
    }

    #[classmethod]
    fn __class_getitem__<'py>(
        cls: &Bound<'py, PyType>,
        item: &Bound<'py, PyAny>,
    ) -> Bound<'py, PyType> {
        let _ = item;
        subscript(cls)
    }

    /// Return whether `element` is a member; an unhashable object is none.
    fn __contains__(&self, element: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.elements.contains(element)
    }

    /// Add `element`.
    ///
    /// Raises `ValueError` if it is a member, and `TypeError` if it is not
    /// hashable.
    fn add_element(&mut self, element: &Bound<'_, PyAny>) -> PyResult<()> {
        let position = self.elements.insert(element)?;
        self.lattice
            .add_element(position)
            .unwrap_or_else(|_member| unreachable!("a new position is no member"));
        Ok(())
    }

    /// Order `lower` below `upper`.
    ///
    /// Raises `ValueError` if either is not a member, and `RuntimeError` if
    /// `upper` is already at most `lower`.
    fn add_order(&mut self, lower: &Bound<'_, PyAny>, upper: &Bound<'_, PyAny>) -> PyResult<()> {
        let lower_position = self.elements.position(lower)?;
        let upper_position = self.elements.position(upper)?;
        self.lattice
            .add_order(&lower_position, &upper_position)
            .map_err(|error| {
                order_error_to_python(lower.py(), &error, &[lower.clone(), upper.clone()])
                    .unwrap_or_else(|conversion| conversion)
            })
    }

    /// Return whether every pair of elements has a meet and a join.
    fn is_lattice(&self) -> bool {
        self.lattice.is_lattice()
    }

    /// Return the report of the pairs of elements without a meet or a
    /// join, one ERROR diagnostic each, in iteration order; the report is
    /// empty for a lattice.
    fn verify<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let error_level = diagnostic_class(py, "DiagnosticLevel")?.getattr("ERROR")?;
        let note = diagnostic_class(py, "Note")?;
        let diagnostic = diagnostic_class(py, "Diagnostic")?;
        let mut diagnostics = Vec::new();
        for missing in self.lattice.missing_bounds() {
            let (bound, x, y) = match missing {
                MissingBound::Meet(x, y) => ("meet", *x, *y),
                MissingBound::Join(x, y) => ("join", *x, *y),
                _ => continue,
            };
            let message = format!(
                "lattice has no unique {bound} for elements {} and {}",
                self.elements.object(py, x).repr()?,
                self.elements.object(py, y).repr()?
            );
            let kwargs = PyDict::new(py);
            kwargs.set_item("level", &error_level)?;
            kwargs.set_item("message", note.call1((message,))?)?;
            kwargs.set_item("source", VERIFY_SOURCE)?;
            diagnostics.push(diagnostic.call((), Some(&kwargs))?);
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("diagnostics", PyTuple::new(py, diagnostics)?)?;
        diagnostic_class(py, "ValidationReport")?.call((), Some(&kwargs))
    }

    /// Return whether `x` and `y` have a meet.
    ///
    /// Raises `ValueError` if either is not a member.
    fn has_meet(&self, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.get_meet(x, y)?.is_some())
    }

    /// Return whether `x` and `y` have a join.
    ///
    /// Raises `ValueError` if either is not a member.
    fn has_join(&self, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.get_join(x, y)?.is_some())
    }

    /// Return the join of `x` and `y`.
    ///
    /// Raises `ValueError` if either is not a member, and `RuntimeError` if
    /// they have no join.
    fn get_least_upper_bound<'py>(
        &self,
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.get_join(x, y)?
            .ok_or_else(|| match (x.str(), y.str()) {
                (Ok(x), Ok(y)) => PyRuntimeError::new_err(format!(
                    "no least upper bound of {x} and {y} found for lattice"
                )),
                (Err(error), _) | (_, Err(error)) => error,
            })
    }

    /// Return the meet of `x` and `y`, or `None` if they have none.
    ///
    /// Raises `ValueError` if either is not a member.
    fn get_meet<'py>(
        &self,
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let (x_position, y_position) = self.positions(x, y)?;
        let meet = self
            .lattice
            .meet(&x_position, &y_position)
            .unwrap_or_else(|_non_member| unreachable!("both are members"));
        Ok(meet.map(|&position| self.elements.object(x.py(), position)))
    }

    /// Return the join of `x` and `y`, or `None` if they have none.
    ///
    /// Raises `ValueError` if either is not a member.
    fn get_join<'py>(
        &self,
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let (x_position, y_position) = self.positions(x, y)?;
        let join = self
            .lattice
            .join(&x_position, &y_position)
            .unwrap_or_else(|_non_member| unreachable!("both are members"));
        Ok(join.map(|&position| self.elements.object(x.py(), position)))
    }
}
