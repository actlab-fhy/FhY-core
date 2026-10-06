//! `fhy_core._rs.Space`, `Condition` and `Forbidden`.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyTuple, PyType};

use fhy_core::identifier::Identifier;
use fhy_core::search_space::{Condition, Forbidden, Space};

use crate::util::gc::Slots;
use crate::util::public_class::PublicClass;

/// When a decision is active, backed by the core [`Condition`]; the base of
/// the public `Condition`.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Condition")]
pub(crate) struct PyCondition {
    condition: Condition,
    /// The `Identifier` of the decision the condition is on.
    target: Py<PyAny>,
    /// The `ConstraintSystem` that must hold.
    when: Py<PyAny>,
    /// The slots of the adapters of Python-defined constraints it holds.
    slots: Slots,
}

impl PyCondition {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("Condition");
        &PUBLIC_CLASS
    }
}

#[pymethods]
impl PyCondition {
    /// Return the condition that the decision `target` is active only when
    /// `when`, a `ConstraintSystem` or an iterable of `Constraint`s, holds.
    ///
    /// Raises `TypeError` for an argument of the wrong type.
    #[new]
    fn new(target: &Bound<'_, PyAny>, when: &Bound<'_, PyAny>) -> PyResult<Self> {
        todo!()
    }

    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.target)?;
        visit.call(&self.when)?;
        self.slots.traverse(&visit)
    }

    /// Register `cls` as the public `Condition` class.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }

    /// The `Identifier` of the decision the condition is on.
    #[getter]
    fn target<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        todo!()
    }

    /// The `ConstraintSystem` that must hold.
    #[getter]
    fn when<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        todo!()
    }

    /// Refuse to set an attribute: the condition is frozen.
    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        todo!()
    }

    /// Refuse to delete an attribute: the condition is frozen.
    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        todo!()
    }

    /// Return `Condition(target=..., when=...)`.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        todo!()
    }

    /// Pickle as a call of the class with its fields.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
    }
}

/// A combination of values no configuration may take, backed by the core
/// [`Forbidden`]; the base of the public `Forbidden`.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Forbidden")]
pub(crate) struct PyForbidden {
    forbidden: Forbidden,
    /// The `ConstraintSystem` no configuration may satisfy.
    when: Py<PyAny>,
    /// The slots of the adapters of Python-defined constraints it holds.
    slots: Slots,
}

impl PyForbidden {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("Forbidden");
        &PUBLIC_CLASS
    }
}

#[pymethods]
impl PyForbidden {
    /// Return the clause forbidding every configuration that satisfies
    /// `when`, a `ConstraintSystem` or an iterable of `Constraint`s.
    ///
    /// Raises `TypeError` for an argument of the wrong type.
    #[new]
    fn new(when: &Bound<'_, PyAny>) -> PyResult<Self> {
        todo!()
    }

    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.when)?;
        self.slots.traverse(&visit)
    }

    /// Register `cls` as the public `Forbidden` class.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }

    /// The `ConstraintSystem` no configuration may satisfy.
    #[getter]
    fn when<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        todo!()
    }

    /// Refuse to set an attribute: the clause is frozen.
    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        todo!()
    }

    /// Refuse to delete an attribute: the clause is frozen.
    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        todo!()
    }

    /// Return `Forbidden(when=...)`.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        todo!()
    }

    /// Pickle as a call of the class with its fields.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
    }
}

/// The whole search space, backed by the core [`Space`]; the base of the
/// public `Space`.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Space")]
pub(crate) struct PySpace {
    space: Space,
    /// The `Identifier` the space is named by.
    name: Py<PyAny>,
    /// The top-level `Variable`s, as given.
    variables: Py<PyTuple>,
    /// The top-level `Choice`s, as given.
    choices: Py<PyTuple>,
    /// The `Condition`s, one per target in canonical order of the targets:
    /// the objects given, and one built for each merged target.
    conditions: Py<PyTuple>,
    /// The `Forbidden` clauses, as given.
    forbidden: Py<PyTuple>,
    /// The `Note`s.
    notes: Py<PyTuple>,
    /// The decision objects, `Variable`s and `Choice`s, in canonical order.
    decisions: Py<PyTuple>,
    /// The position of each decision in canonical order, by name.
    positions: HashMap<Identifier, usize>,
    /// The levels of choices the space nests.
    depth: usize,
    /// The slots of the adapters of the subclass variables it holds.
    slots: Slots,
}

impl PySpace {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("Space");
        &PUBLIC_CLASS
    }

    /// Return the core space.
    pub(super) const fn core(&self) -> &Space {
        &self.space
    }
}

#[pymethods]
impl PySpace {
    /// Return the space named `name` (a fresh `Identifier("space")` when
    /// `None`) holding the top-level `variables` and `choices`, the
    /// `conditions` and the `forbidden` clauses, with `notes`.
    ///
    /// Raises `TypeError` for an argument of the wrong type,
    /// `DuplicateNameError` for names that repeat, `SearchSpaceError` for
    /// the core's other refusals, the exception a hook or a Python-defined
    /// constraint raises, and `RecursionError` for choices nested deeper
    /// than the recursion limit.
    #[new]
    #[pyo3(signature = (
        variables = None, choices = None, conditions = None, forbidden = None,
        name = None, notes = None,
    ))]
    fn new(
        variables: Option<&Bound<'_, PyAny>>,
        choices: Option<&Bound<'_, PyAny>>,
        conditions: Option<&Bound<'_, PyAny>>,
        forbidden: Option<&Bound<'_, PyAny>>,
        name: Option<&Bound<'_, PyAny>>,
        notes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        todo!()
    }

    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.name)?;
        visit.call(&self.variables)?;
        visit.call(&self.choices)?;
        visit.call(&self.conditions)?;
        visit.call(&self.forbidden)?;
        visit.call(&self.notes)?;
        visit.call(&self.decisions)?;
        self.slots.traverse(&visit)
    }

    /// Register `cls` as the public `Space` class.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }

    /// The `Identifier` the space is named by.
    #[getter]
    fn name<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        todo!()
    }

    /// The top-level `Variable`s, in order.
    #[getter]
    fn variables<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        todo!()
    }

    /// The top-level `Choice`s, in order.
    #[getter]
    fn choices<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        todo!()
    }

    /// The `Condition`s, one per target, in canonical order of the targets.
    #[getter]
    fn conditions<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        todo!()
    }

    /// The `Forbidden` clauses, in order.
    #[getter]
    fn forbidden<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        todo!()
    }

    /// The `Note`s attached to the space.
    #[getter]
    fn notes<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        todo!()
    }

    /// The decisions, `Variable`s and `Choice`s at every depth, in
    /// canonical order.
    #[getter]
    fn decisions<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        todo!()
    }

    /// The decisions' names in decision order: each after every decision
    /// it depends on, ties in canonical order.
    #[getter]
    fn decision_order<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
    }

    /// Return the decision named `name`, or `None`.
    ///
    /// Raises `TypeError` if `name` is not an `Identifier`.
    fn decision<'py>(
        &self,
        py: Python<'py>,
        name: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        todo!()
    }

    /// Return whether `other` is a space with the same names, structurally
    /// equivalent parts and equal conditions and clauses.
    fn is_structurally_equivalent(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        todo!()
    }

    /// Return whether `other` is the same space up to the renaming of the
    /// names it binds, under no renaming.
    fn is_alpha_equivalent(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        todo!()
    }

    /// Return whether `other` is the same space up to the renaming of the
    /// names it binds, under `renaming`.
    fn is_alpha_equivalent_under(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        renaming: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        todo!()
    }

    /// Refuse to set an attribute: the space is frozen.
    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        todo!()
    }

    /// Refuse to delete an attribute: the space is frozen.
    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        todo!()
    }

    /// Return `Space(name=..., variables=..., choices=..., conditions=...,
    /// forbidden=..., notes=...)`.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        todo!()
    }

    /// Pickle as a call of the class with its fields.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
    }

    /// Return the V2 payload `{"identifier", "variables", "choices",
    /// "conditions", "forbidden", "notes"}`.
    ///
    /// Raises `SerializationError` inside `wire_version(WireVersion.V1)`.
    fn serialize_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return the canonical V2 text of the payload, re-formatted for
    /// `indent` or `sort_keys`.
    #[pyo3(signature = (*, indent = None, sort_keys = None))]
    fn to_json(
        slf: &Bound<'_, Self>,
        indent: Option<&Bound<'_, PyAny>>,
        sort_keys: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        todo!()
    }

    /// Return the space of the V2 payload `data`.
    #[classmethod]
    fn deserialize_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return the space of the JSON text `payload`.
    #[classmethod]
    fn from_json<'py>(
        cls: &Bound<'py, PyType>,
        payload: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }
}
