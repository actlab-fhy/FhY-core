//! `PyO3` classes for [`fhy_core::lattice`]: `fhy_core._rs.PartiallyOrderedSet`
//! and `Lattice`, the bases of `fhy_core.utils.poset.PartiallyOrderedSet` and
//! `fhy_core.lattice.Lattice`.
//!
//! The elements are Python objects, so each class keeps them in a Python
//! `dict` from element to its position, which keeps Python's hashing and
//! `==`, and a list of the objects by position; the core's order runs over
//! the positions. The core never holds a Python object.
//!
//! No Python code runs while the object is borrowed: an element's
//! `__hash__` and `__eq__`, `iter_stable`'s key and an element's `repr` run
//! on the dict or on a copy of the element list, before or after the short
//! borrow that reads or changes the core order, so they may read the object
//! again, or add to it, and see a consistent state.
//!
//! Both classes pickle, copy and deep-copy: `__reduce__` returns the
//! elements in insertion order and every order added, as element objects,
//! and `__setstate__` replays them, so the copy iterates, orders, meets and
//! joins as the original does; a subclass's `__dict__` goes with them, and
//! its `__init__` is not called.

use pyo3::PyClass;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::boolean_struct::False;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};

use fhy_core::lattice::{Lattice, MissingBound, OrderError, PartiallyOrderedSet};

/// The source of the diagnostics `Lattice.verify` reports.
const VERIFY_SOURCE: &str = "fhy_core.lattice.Lattice.verify";

/// The element objects of an order, their positions, and the orders added
/// between them.
struct Elements {
    /// The position of each element, keyed by the element.
    positions: Py<PyDict>,
    /// The elements, by position.
    objects: Vec<Py<PyAny>>,
    /// Every order added, as `(lower, upper)` positions, in the order added,
    /// which a pickle replays.
    orders: Vec<(usize, usize)>,
}

impl Elements {
    fn new(py: Python<'_>) -> Self {
        Self {
            positions: PyDict::new(py).unbind(),
            objects: Vec::new(),
            orders: Vec::new(),
        }
    }

    /// Return the pickle state of the order: the elements in insertion
    /// order, and every order added as a `(lower, upper)` pair of elements.
    fn state<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
        let orders = PyList::empty(py);
        for &(lower, upper) in &self.orders {
            orders.append((self.objects[lower].bind(py), self.objects[upper].bind(py)))?;
        }
        Ok((self.list(py, 0..self.objects.len())?, orders))
    }

    /// Append `element` at the next position, and return that position.
    fn push(&mut self, element: &Bound<'_, PyAny>) -> usize {
        self.objects.push(element.clone().unbind());
        self.objects.len() - 1
    }

    /// Add `element` to an order no Python code can reach, such as one
    /// `__setstate__` is building, and return its position.
    ///
    /// Raises the `ValueError` of a member, and `TypeError` for an
    /// unhashable element.
    fn insert_unshared(&mut self, element: &Bound<'_, PyAny>) -> PyResult<usize> {
        let py = element.py();
        let position = self.objects.len();
        if !self.positions.bind(py).set_default(element, position)? {
            return Err(already_a_member(element));
        }
        Ok(self.push(element))
    }

    /// Return the object at `position`.
    fn object<'py>(&self, py: Python<'py>, position: usize) -> Bound<'py, PyAny> {
        self.objects[position].bind(py).clone()
    }

    /// Return a new reference to every element, by position.
    fn objects(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.objects
            .iter()
            .map(|object| object.clone_ref(py))
            .collect()
    }

    /// Visit the positions `dict` and every element.
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.positions)?;
        crate::kit::gc::traverse_all(visit, &self.objects)
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
}

/// Return whether `element` is a key of `positions`; an unhashable object is
/// none.
fn is_member(positions: &Bound<'_, PyDict>, element: &Bound<'_, PyAny>) -> PyResult<bool> {
    match positions.contains(element) {
        Ok(contains) => Ok(contains),
        Err(error) if error.is_instance_of::<PyTypeError>(element.py()) => Ok(false),
        Err(error) => Err(error),
    }
}

/// Return the position `positions` holds for `element`, or raise the
/// `ValueError` of a non-member.
fn position_of(positions: &Bound<'_, PyDict>, element: &Bound<'_, PyAny>) -> PyResult<usize> {
    match positions.get_item(element) {
        Ok(Some(position)) => position.extract(),
        Ok(None) => Err(not_a_member(element)),
        Err(error) if error.is_instance_of::<PyTypeError>(element.py()) => {
            Err(not_a_member(element))
        }
        Err(error) => Err(error),
    }
}

/// The `ValueError` of `element`, a member being added again.
fn already_a_member(element: &Bound<'_, PyAny>) -> PyErr {
    order_error_to_python(
        element.py(),
        &OrderError::AlreadyAMember(0),
        std::slice::from_ref(element),
    )
    .unwrap_or_else(|conversion| conversion)
}

/// The `ValueError` of `element`, which is no member.
fn not_a_member(element: &Bound<'_, PyAny>) -> PyErr {
    order_error_to_python(
        element.py(),
        &OrderError::NotAMember(0),
        std::slice::from_ref(element),
    )
    .unwrap_or_else(|conversion| conversion)
}

/// Return the rank of each of `objects` by `key` called on it: its place
/// when the `(key, position)` pairs are sorted, so equal keys keep insertion
/// order.
fn ranks(py: Python<'_>, objects: &[Py<PyAny>], key: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let pairs = PyList::empty(py);
    for (position, object) in objects.iter().enumerate() {
        pairs.append((key.call1((object.bind(py),))?, position))?;
    }
    pairs.sort()?;
    let mut ranks = vec![0; objects.len()];
    for (rank, pair) in pairs.iter().enumerate() {
        let position: usize = pair.get_item(1)?.extract()?;
        ranks[position] = rank;
    }
    Ok(ranks)
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

/// A class that holds its elements in [`Elements`] over a core order.
trait Order: PyClass<Frozen = False> {
    fn elements(&self) -> &Elements;

    fn elements_mut(&mut self) -> &mut Elements;

    /// Add the new position `position` to the core order.
    fn add_position(&mut self, position: usize);

    /// Order the position `lower` below `upper` in the core order.
    fn order_positions(&mut self, lower: usize, upper: usize) -> Result<(), OrderError<usize>>;

    /// Return the value's elements and order, rebuilt from a pickle state's
    /// elements and orders.
    fn rebuild(
        mut self,
        elements: &[Bound<'_, PyAny>],
        orders: &[(Bound<'_, PyAny>, Bound<'_, PyAny>)],
    ) -> PyResult<Self> {
        for element in elements {
            let position = self.elements_mut().insert_unshared(element)?;
            self.add_position(position);
        }
        for (lower, upper) in orders {
            let positions = self.elements().positions.bind(lower.py()).clone();
            let (lower_position, upper_position) = (
                position_of(&positions, lower)?,
                position_of(&positions, upper)?,
            );
            self.order_positions(lower_position, upper_position)
                .map_err(|error| cycle_error(&error, lower, upper))?;
            self.elements_mut()
                .orders
                .push((lower_position, upper_position));
        }
        Ok(self)
    }
}

/// Return the dict of `slf`'s positions, under a borrow that ends at once.
fn positions_dict<'py, T: Order>(slf: &Bound<'py, T>) -> Bound<'py, PyDict> {
    slf.borrow().elements().positions.bind(slf.py()).clone()
}

/// Return the positions of `x` and `y` in `slf`, raising the `ValueError` of
/// a non-member, `x` first; their hashing runs under no borrow.
fn two_positions<T: Order>(
    slf: &Bound<'_, T>,
    x: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
) -> PyResult<(usize, usize)> {
    let positions = positions_dict(slf);
    Ok((position_of(&positions, x)?, position_of(&positions, y)?))
}

/// The Python exception of the core's refusal to order `lower` below
/// `upper`.
fn cycle_error(
    error: &OrderError<usize>,
    lower: &Bound<'_, PyAny>,
    upper: &Bound<'_, PyAny>,
) -> PyErr {
    order_error_to_python(lower.py(), error, &[lower.clone(), upper.clone()])
        .unwrap_or_else(|conversion| conversion)
}

/// Add `element` to `slf`.
///
/// The element is hashed once, into the dict, under no borrow; its position
/// is the next free one when the borrow is taken. A hash or equality that
/// adds elements meanwhile takes the predicted position, so the element's
/// entry is then set again, to the position it got.
///
/// Raises the `ValueError` of a member, and `TypeError` for an unhashable
/// element.
fn add_element<T: Order>(slf: &Bound<'_, T>, element: &Bound<'_, PyAny>) -> PyResult<()> {
    let (positions, predicted) = {
        let this = slf.borrow();
        let elements = this.elements();
        (
            elements.positions.bind(slf.py()).clone(),
            elements.objects.len(),
        )
    };
    if !positions.set_default(element, predicted)? {
        return Err(already_a_member(element));
    }
    let position = {
        let mut this = slf.borrow_mut();
        let position = this.elements_mut().push(element);
        this.add_position(position);
        position
    };
    if position != predicted {
        positions.set_item(element, position)?;
    }
    Ok(())
}

/// Order `lower` below `upper` in `slf`, and record the order.
///
/// Raises the `ValueError` of a non-member, and `RuntimeError` if `upper`
/// is already at most `lower`.
fn add_order<T: Order>(
    slf: &Bound<'_, T>,
    lower: &Bound<'_, PyAny>,
    upper: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let (lower_position, upper_position) = two_positions(slf, lower, upper)?;
    let ordered = {
        let mut this = slf.borrow_mut();
        let ordered = this.order_positions(lower_position, upper_position);
        if ordered.is_ok() {
            this.elements_mut()
                .orders
                .push((lower_position, upper_position));
        }
        ordered
    };
    ordered.map_err(|error| cycle_error(&error, lower, upper))
}

/// Return the `__reduce__` value of `slf`: `copyreg.__newobj__`, so
/// unpickling calls no `__init__`, the class, and the state with the
/// instance `__dict__` (or `None`) last.
fn reduce<'py, T: Order>(
    slf: &Bound<'py, T>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>, Bound<'py, PyTuple>)> {
    let py = slf.py();
    let (elements, orders) = slf.borrow().elements().state(py)?;
    let instance_dict = match slf.as_any().getattr(intern!(py, "__dict__")) {
        Ok(instance_dict) => instance_dict,
        Err(_no_dict) => py.None().into_bound(py),
    };
    let state = PyTuple::new(py, [elements.into_any(), orders.into_any(), instance_dict])?;
    let new_object =
        crate::kit::python::cached_attr!(py, "copyreg", "__newobj__" => PyAny)?.clone();
    Ok((
        new_object,
        PyTuple::new(py, [slf.as_any().get_type()])?,
        state,
    ))
}

/// Restore `slf` from the state `reduce` returned, replaying its elements
/// and orders on `empty` before swapping it in, so no Python code runs under
/// the borrow.
///
/// Raises `TypeError` naming `class` for a state of another shape, and
/// whatever adding an element or an order raises.
fn set_state<T: Order>(
    slf: &Bound<'_, T>,
    state: &Bound<'_, PyAny>,
    class: &str,
    empty: T,
) -> PyResult<()> {
    let py = slf.py();
    let bad_state =
        || PyTypeError::new_err(format!("{class} state must be the one __reduce__ returns."));
    let state = state
        .cast::<PyTuple>()
        .map_err(|_not_a_tuple| bad_state())?;
    if state.len() != 3 {
        return Err(bad_state());
    }
    let elements = state
        .get_item(0)?
        .try_iter()?
        .collect::<PyResult<Vec<_>>>()?;
    let mut orders = Vec::new();
    for pair in state.get_item(1)?.try_iter()? {
        let pair = pair?;
        let pair = pair.cast::<PyTuple>().map_err(|_not_a_tuple| bad_state())?;
        if pair.len() != 2 {
            return Err(bad_state());
        }
        orders.push((pair.get_item(0)?, pair.get_item(1)?));
    }
    let restored = empty.rebuild(&elements, &orders)?;
    *slf.borrow_mut() = restored;
    let instance_dict = state.get_item(2)?;
    if !instance_dict.is_none() {
        slf.as_any()
            .getattr(intern!(py, "__dict__"))?
            .call_method1(intern!(py, "update"), (instance_dict,))?;
    }
    Ok(())
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

impl PyPartiallyOrderedSet {
    fn empty(py: Python<'_>) -> Self {
        Self {
            elements: Elements::new(py),
            poset: PartiallyOrderedSet::new(),
        }
    }
}

impl Order for PyPartiallyOrderedSet {
    fn elements(&self) -> &Elements {
        &self.elements
    }

    fn elements_mut(&mut self) -> &mut Elements {
        &mut self.elements
    }

    fn add_position(&mut self, position: usize) {
        self.poset
            .add_element(position)
            .unwrap_or_else(|_member| unreachable!("a new position is no member"));
    }

    fn order_positions(&mut self, lower: usize, upper: usize) -> Result<(), OrderError<usize>> {
        self.poset.add_order(&lower, &upper)
    }
}

#[pymethods]
impl PyPartiallyOrderedSet {
    /// Visit the elements and their positions `dict`, for the cycle
    /// collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.elements.traverse(&visit)
    }

    /// Drop the elements, for the cycle collector: the object is left
    /// empty, a consistent state.
    fn __clear__(&mut self, py: Python<'_>) {
        *self = Self::empty(py);
    }

    /// Create an empty set; arguments are accepted and ignored, so a
    /// subclass with its own `__init__` constructs.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(
        py: Python<'_>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Self {
        Self::empty(py)
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
    fn __contains__(slf: &Bound<'_, Self>, element: &Bound<'_, PyAny>) -> PyResult<bool> {
        is_member(&positions_dict(slf), element)
    }

    fn __len__(&self) -> usize {
        self.poset.len()
    }

    /// Iterate the elements in a topological order in which the element
    /// added first comes first among those that can come next.
    fn __iter__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let list = {
            let this = slf.borrow();
            this.elements.list(py, this.poset.iter().copied())?
        };
        list.try_iter().map(Bound::into_any)
    }

    /// Iterate the elements in a topological order in which, among those
    /// that can come next, the one with the least `key(element)` comes
    /// first, and of equal keys the one added first.
    ///
    /// `key` runs over a copy of the element list, under no borrow; an
    /// element it adds is iterated too, after the ranked ones it may follow,
    /// in insertion order.
    #[pyo3(signature = (key = None))]
    fn iter_stable<'py>(
        slf: &Bound<'py, Self>,
        key: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let repr = py
            .import(intern!(py, "builtins"))?
            .getattr(intern!(py, "repr"))?;
        let objects = slf.borrow().elements.objects(py);
        let ranks = ranks(py, &objects, key.unwrap_or(&repr))?;
        let list = {
            let this = slf.borrow();
            this.elements.list(
                py,
                this.poset
                    .iter_by_key(|&position| ranks.get(position).copied().unwrap_or(position))
                    .copied(),
            )?
        };
        list.try_iter().map(Bound::into_any)
    }

    /// Add `element`, ordered with no other element.
    ///
    /// Raises `ValueError` if it is a member, and `TypeError` if it is not
    /// hashable.
    fn add_element(slf: &Bound<'_, Self>, element: &Bound<'_, PyAny>) -> PyResult<()> {
        add_element(slf, element)
    }

    /// Order `lower` below `upper`.
    ///
    /// Raises `ValueError` if either is not a member, and `RuntimeError` if
    /// `upper` is already at most `lower`.
    fn add_order(
        slf: &Bound<'_, Self>,
        lower: &Bound<'_, PyAny>,
        upper: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        add_order(slf, lower, upper)
    }

    /// Return the pickle form: the class, and the elements in insertion
    /// order, every order added and the instance `__dict__`.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>, Bound<'py, PyTuple>)> {
        reduce(slf)
    }

    /// Restore the set from the state `__reduce__` returned, replaying its
    /// elements and orders.
    ///
    /// Raises `TypeError` for a state of another shape, and whatever adding
    /// an element or an order raises.
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        set_state(slf, state, "PartiallyOrderedSet", Self::empty(slf.py()))
    }

    /// Return whether `lower` is less than or equal to `upper`.
    ///
    /// Raises `ValueError` if either is not a member.
    fn is_less_than(
        slf: &Bound<'_, Self>,
        lower: &Bound<'_, PyAny>,
        upper: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let (lower_position, upper_position) = two_positions(slf, lower, upper)?;
        Ok(slf
            .borrow()
            .poset
            .is_at_most(&lower_position, &upper_position)
            .unwrap_or_else(|_non_member| unreachable!("both are members")))
    }

    /// Return whether `lower` is greater than or equal to `upper`.
    ///
    /// Raises `ValueError` if either is not a member.
    fn is_greater_than(
        slf: &Bound<'_, Self>,
        lower: &Bound<'_, PyAny>,
        upper: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let (lower_position, upper_position) = two_positions(slf, lower, upper)?;
        Ok(slf
            .borrow()
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
    fn empty(py: Python<'_>) -> Self {
        Self {
            elements: Elements::new(py),
            lattice: Lattice::new(),
        }
    }

    /// Return the meet (`meet` true) or the join of `x` and `y`, or `None`
    /// if they have none.
    ///
    /// Raises `ValueError` if either is not a member.
    fn bound<'py>(
        slf: &Bound<'py, Self>,
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
        meet: bool,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let (x_position, y_position) = two_positions(slf, x, y)?;
        let this = slf.borrow();
        let bound = if meet {
            this.lattice.meet(&x_position, &y_position)
        } else {
            this.lattice.join(&x_position, &y_position)
        }
        .unwrap_or_else(|_non_member| unreachable!("both are members"));
        Ok(bound.map(|&position| this.elements.object(slf.py(), position)))
    }
}

impl Order for PyLattice {
    fn elements(&self) -> &Elements {
        &self.elements
    }

    fn elements_mut(&mut self) -> &mut Elements {
        &mut self.elements
    }

    fn add_position(&mut self, position: usize) {
        self.lattice
            .add_element(position)
            .unwrap_or_else(|_member| unreachable!("a new position is no member"));
    }

    fn order_positions(&mut self, lower: usize, upper: usize) -> Result<(), OrderError<usize>> {
        self.lattice.add_order(&lower, &upper)
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
    /// Visit the elements and their positions `dict`, for the cycle
    /// collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.elements.traverse(&visit)
    }

    /// Drop the elements, for the cycle collector: the object is left
    /// empty, a consistent state.
    fn __clear__(&mut self, py: Python<'_>) {
        *self = Self::empty(py);
    }

    /// Create an empty lattice; arguments are accepted and ignored, so a
    /// subclass with its own `__init__` constructs.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(
        py: Python<'_>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> Self {
        Self::empty(py)
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
    fn __contains__(slf: &Bound<'_, Self>, element: &Bound<'_, PyAny>) -> PyResult<bool> {
        is_member(&positions_dict(slf), element)
    }

    /// Add `element`.
    ///
    /// Raises `ValueError` if it is a member, and `TypeError` if it is not
    /// hashable.
    fn add_element(slf: &Bound<'_, Self>, element: &Bound<'_, PyAny>) -> PyResult<()> {
        add_element(slf, element)
    }

    /// Order `lower` below `upper`.
    ///
    /// Raises `ValueError` if either is not a member, and `RuntimeError` if
    /// `upper` is already at most `lower`.
    fn add_order(
        slf: &Bound<'_, Self>,
        lower: &Bound<'_, PyAny>,
        upper: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        add_order(slf, lower, upper)
    }

    /// Return the pickle form: the class, and the elements in insertion
    /// order, every order added and the instance `__dict__`.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>, Bound<'py, PyTuple>)> {
        reduce(slf)
    }

    /// Restore the lattice from the state `__reduce__` returned, replaying
    /// its elements and orders.
    ///
    /// Raises `TypeError` for a state of another shape, and whatever adding
    /// an element or an order raises.
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        set_state(slf, state, "Lattice", Self::empty(slf.py()))
    }

    /// Return whether every pair of elements has a meet and a join.
    fn is_lattice(&self) -> bool {
        self.lattice.is_lattice()
    }

    /// Return the report of the pairs of elements without a meet or a
    /// join, one ERROR diagnostic each, in iteration order; the report is
    /// empty for a lattice.
    ///
    /// The pairs are read under the borrow, and the elements' `repr`s and
    /// the diagnostics built after it.
    fn verify<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let missing: Vec<(&str, Py<PyAny>, Py<PyAny>)> = {
            let this = slf.borrow();
            this.lattice
                .missing_bounds()
                .filter_map(|missing| match missing {
                    MissingBound::Meet(x, y) => Some(("meet", *x, *y)),
                    MissingBound::Join(x, y) => Some(("join", *x, *y)),
                    _ => None,
                })
                .map(|(bound, x, y)| {
                    (
                        bound,
                        this.elements.objects[x].clone_ref(py),
                        this.elements.objects[y].clone_ref(py),
                    )
                })
                .collect()
        };
        let error_level = diagnostic_class(py, "DiagnosticLevel")?.getattr("ERROR")?;
        let note = diagnostic_class(py, "Note")?;
        let diagnostic = diagnostic_class(py, "Diagnostic")?;
        let mut diagnostics = Vec::new();
        for (bound, x, y) in missing {
            let message = format!(
                "lattice has no unique {bound} for elements {} and {}",
                x.bind(py).repr()?,
                y.bind(py).repr()?
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
    fn has_meet(
        slf: &Bound<'_, Self>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        Ok(Self::bound(slf, x, y, true)?.is_some())
    }

    /// Return whether `x` and `y` have a join.
    ///
    /// Raises `ValueError` if either is not a member.
    fn has_join(
        slf: &Bound<'_, Self>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        Ok(Self::bound(slf, x, y, false)?.is_some())
    }

    /// Return the join of `x` and `y`.
    ///
    /// Raises `ValueError` if either is not a member, and `RuntimeError` if
    /// they have no join.
    fn get_least_upper_bound<'py>(
        slf: &Bound<'py, Self>,
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::bound(slf, x, y, false)?.ok_or_else(|| match (x.str(), y.str()) {
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
        slf: &Bound<'py, Self>,
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        Self::bound(slf, x, y, true)
    }

    /// Return the join of `x` and `y`, or `None` if they have none.
    ///
    /// Raises `ValueError` if either is not a member.
    fn get_join<'py>(
        slf: &Bound<'py, Self>,
        x: &Bound<'py, PyAny>,
        y: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        Self::bound(slf, x, y, false)
    }
}
