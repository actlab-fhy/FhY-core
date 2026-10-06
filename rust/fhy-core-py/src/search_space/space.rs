//! `fhy_core._rs.Space`, `Condition` and `Forbidden`.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyDict, PyTuple, PyType};

use fhy_core::constraint::ConstraintSystem;
use fhy_core::identifier::Identifier;
use fhy_core::search_space::wire::SpaceData;
use fhy_core::search_space::{Condition, Forbidden, Space};
use fhy_core::term::AlphaEquivalence;

use crate::constraint::{PyConstraintSystem, read_native_constraint};
use crate::diagnostic::note_to_python;
use crate::identifier::{identifier_to_python, restore_identifier};
use crate::param::constraint_to_python;
use crate::term::read_renaming;
use crate::util::dataclass::format_dataclass_repr;
use crate::util::frozen::{refuse_attribute_assignment, refuse_attribute_deletion};
use crate::util::gc::{Slots, collect_slots};
use crate::util::pending::with_pending_errors;
use crate::util::public_class::PublicClass;

use super::alternative::{alternative_children, read_choices};
use super::arguments::{
    Seeded, choices_depth, ensure_depth, instantiate, read_items, read_name, read_notes, take_seed,
    wrong_argument, wrong_seed,
};
use super::choice::{PyChoice, choice_to_python};
use super::errors::{equivalence_error_to_py, space_error_to_py};
use super::variable::{read_variable, variable_to_python};
use super::wire::{Family, decode_part, refuse_v1, write_part, write_part_json};

/// The expected type of a condition's or a clause's `when`.
const SYSTEM: &str = "a ConstraintSystem or Constraints";

/// Return the core system of `when`, a `ConstraintSystem` or an iterable of
/// `Constraint`s, and the system object: the one given, or a new one of
/// the constraints given.
///
/// # Errors
///
/// Raises `TypeError` naming `owner` for another value.
fn read_system<'py>(
    when: &Bound<'py, PyAny>,
    owner: &str,
) -> PyResult<(ConstraintSystem, Bound<'py, PyAny>)> {
    let py = when.py();
    if let Ok(system) = when.cast::<PyConstraintSystem>() {
        return Ok((system.get().core().clone(), when.clone()));
    }
    let members = read_items(py, Some(when), owner, "when", SYSTEM)?;
    let constraint = crate::cached_attr!(py, "fhy_core.symbolic.constraint", "Constraint")?;
    for member in &members {
        if read_native_constraint(&member).is_none() && !member.is_instance(constraint)? {
            return Err(wrong_argument(owner, "when", SYSTEM, &member));
        }
    }
    let system = system_object(py, members)?;
    let core = system.cast::<PyConstraintSystem>()?.get().core().clone();
    Ok((core, system))
}

/// Return a new public `ConstraintSystem` of the constraint objects
/// `members`.
///
/// # Errors
///
/// Raises what building the system raises.
fn system_object<'py>(
    py: Python<'py>,
    members: Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    crate::cached_attr!(py, "fhy_core.symbolic.constraint", "ConstraintSystem")?.call1((members,))
}

/// Return a new public `ConstraintSystem` of the core `system`.
///
/// # Errors
///
/// Raises what building an object raises.
fn system_to_python<'py>(
    py: Python<'py>,
    system: &ConstraintSystem,
) -> PyResult<Bound<'py, PyAny>> {
    let members = system
        .constraints()
        .iter()
        .map(|constraint| constraint_to_python(py, constraint))
        .collect::<PyResult<Vec<_>>>()?;
    system_object(py, PyTuple::new(py, members)?)
}

/// When a decision is active, backed by the core [`Condition`]; the base of
/// the public `Condition`.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Condition")]
pub(crate) struct PyCondition {
    condition: Condition,
    /// The `Identifier` of the decision the condition is on.
    target: Py<PyAny>,
    /// The `ConstraintSystem` that must hold.
    when: Py<PyAny>,
}

impl PyCondition {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("Condition");
        &PUBLIC_CLASS
    }

    /// Return a new public `Condition` of the core `condition` and the
    /// objects of its target and system.
    fn instantiate<'py>(
        py: Python<'py>,
        condition: Condition,
        target: &Bound<'py, PyAny>,
        when: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        instantiate(
            Self::public_class().get(py)?,
            2,
            Seeded::Condition(Self {
                condition,
                target: target.clone().unbind(),
                when: when.clone().unbind(),
            }),
        )
    }
}

#[pymethods]
impl PyCondition {
    /// Return the condition that the decision `target` is active only when
    /// `when`, a `ConstraintSystem` or an iterable of `Constraint`s, holds.
    ///
    /// Raises `TypeError` for an argument of the wrong type.
    #[new]
    #[pyo3(signature = (target, when, **kwargs))]
    fn new(
        target: &Bound<'_, PyAny>,
        when: &Bound<'_, PyAny>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if let Some(seeded) = take_seed(kwargs)? {
            return match seeded {
                Seeded::Condition(condition) => Ok(condition),
                _ => Err(wrong_seed()),
            };
        }
        let core_target = restore_identifier(target, "Condition", "target")?;
        let (system, when) = read_system(when, "Condition")?;
        Ok(Self {
            condition: Condition::new(core_target, system),
            target: target.clone().unbind(),
            when: when.unbind(),
        })
    }

    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.target)?;
        visit.call(&self.when)
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
        self.target.bind(py).clone()
    }

    /// The `ConstraintSystem` that must hold.
    #[getter]
    fn when<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.when.bind(py).clone()
    }

    /// Refuse to set an attribute: the condition is frozen.
    fn __setattr__(slf: &Bound<'_, Self>, name: &str, _value: &Bound<'_, PyAny>) -> PyResult<()> {
        refuse_attribute_assignment(slf.as_any(), name)
    }

    /// Refuse to delete an attribute: the condition is frozen.
    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        refuse_attribute_deletion(slf.as_any(), name)
    }

    /// Return `Condition(target=..., when=...)`.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        format_dataclass_repr(
            &slf.get_type(),
            &[
                ("target", this.target.bind(py)),
                ("when", this.when.bind(py)),
            ],
        )
    }

    /// Pickle as a call of the class with its fields.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.get();
        let fields = PyTuple::new(py, [this.target.bind(py), this.when.bind(py)])?;
        PyTuple::new(py, [slf.get_type().into_any(), fields.into_any()])
    }
}

/// A combination of values no configuration may take, backed by the core
/// [`Forbidden`]; the base of the public `Forbidden`.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Forbidden")]
pub(crate) struct PyForbidden {
    forbidden: Forbidden,
    /// The `ConstraintSystem` no configuration may satisfy.
    when: Py<PyAny>,
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
    #[pyo3(signature = (when, **kwargs))]
    fn new(when: &Bound<'_, PyAny>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        if let Some(seeded) = take_seed(kwargs)? {
            return match seeded {
                Seeded::Forbidden(forbidden) => Ok(forbidden),
                _ => Err(wrong_seed()),
            };
        }
        let (system, when) = read_system(when, "Forbidden")?;
        Ok(Self {
            forbidden: Forbidden::new(system),
            when: when.unbind(),
        })
    }

    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.when)
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
        self.when.bind(py).clone()
    }

    /// Refuse to set an attribute: the clause is frozen.
    fn __setattr__(slf: &Bound<'_, Self>, name: &str, _value: &Bound<'_, PyAny>) -> PyResult<()> {
        refuse_attribute_assignment(slf.as_any(), name)
    }

    /// Refuse to delete an attribute: the clause is frozen.
    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        refuse_attribute_deletion(slf.as_any(), name)
    }

    /// Return `Forbidden(when=...)`.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        format_dataclass_repr(&slf.get_type(), &[("when", slf.get().when.bind(slf.py()))])
    }

    /// Pickle as a call of the class with its fields.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let fields = PyTuple::new(py, [slf.get().when.bind(py)])?;
        PyTuple::new(py, [slf.get_type().into_any(), fields.into_any()])
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
    /// The slots of the adapters of the subclass variables it holds.
    slots: Slots,
}

/// The objects a space is assembled from.
struct SpaceObjects<'py> {
    name: Bound<'py, PyAny>,
    variables: Bound<'py, PyTuple>,
    choices: Bound<'py, PyTuple>,
    conditions: Bound<'py, PyTuple>,
    forbidden: Bound<'py, PyTuple>,
    notes: Bound<'py, PyTuple>,
}

/// Append the decision objects of `variables` and `choices` to `decisions`,
/// in canonical order: the variables, then each choice followed by each of
/// its alternatives' variables and sub-choices. It recurses once per level
/// of choices, which the depth guard bounds by the recursion limit.
fn collect_decisions<'py>(
    variables: &Bound<'py, PyTuple>,
    choices: &Bound<'py, PyTuple>,
    decisions: &mut Vec<Bound<'py, PyAny>>,
) -> PyResult<()> {
    let py = variables.py();
    decisions.extend(variables.iter());
    for choice in choices {
        decisions.push(choice.clone());
        let choice = choice.cast::<PyChoice>()?.get();
        for (object, part) in choice
            .alternative_objects(py)
            .iter()
            .zip(choice.core().alternatives())
        {
            let (variables, choices) = alternative_children(&object, part)?;
            collect_decisions(&variables, &choices, decisions)?;
        }
    }
    Ok(())
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

    /// Return the decision object at the canonical position `position`.
    pub(super) fn decision_at<'py>(
        &self,
        py: Python<'py>,
        position: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.decisions.bind(py).get_item(position)
    }

    /// Return the canonical position of the decision named `name`.
    pub(super) fn position(&self, name: &Identifier) -> Option<usize> {
        self.positions.get(name).copied()
    }

    /// Return the space of the core `space`, the objects it was built from,
    /// and the slots its construction made.
    fn assemble(space: Space, objects: SpaceObjects<'_>, slots: Slots) -> PyResult<Self> {
        let mut decisions = Vec::new();
        collect_decisions(&objects.variables, &objects.choices, &mut decisions)?;
        let positions = space
            .decisions()
            .enumerate()
            .map(|(position, decision)| (decision.name().clone(), position))
            .collect();
        let py = objects.name.py();
        Ok(Self {
            space,
            name: objects.name.unbind(),
            variables: objects.variables.unbind(),
            choices: objects.choices.unbind(),
            conditions: objects.conditions.unbind(),
            forbidden: objects.forbidden.unbind(),
            notes: objects.notes.unbind(),
            decisions: PyTuple::new(py, decisions)?.unbind(),
            positions,
            slots,
        })
    }
}

/// Return the condition objects of `space`, one per target in canonical
/// order: the object given when a target has one, and a new one over the
/// constraint objects given when its conditions merged.
fn kept_conditions<'py>(
    space: &Space,
    given: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyTuple>> {
    let py = given.py();
    let mut by_target: HashMap<Identifier, Vec<Bound<'py, PyCondition>>> = HashMap::new();
    for condition in given {
        let condition = condition.cast_into::<PyCondition>()?;
        let target = condition.get().condition.target().clone();
        by_target.entry(target).or_default().push(condition);
    }
    let objects = space
        .conditions()
        .iter()
        .map(
            |condition| match by_target.get(condition.target()).map(Vec::as_slice) {
                Some([only]) => Ok(only.clone().into_any()),
                Some(merged) => {
                    let mut members = Vec::new();
                    for part in merged {
                        let when = part.get().when.bind(py);
                        members.extend(when.getattr(pyo3::intern!(py, "constraints"))?.try_iter()?);
                    }
                    let members = members.into_iter().collect::<PyResult<Vec<_>>>()?;
                    let system = system_object(py, PyTuple::new(py, members)?)?;
                    let target = merged
                        .first()
                        .map(|first| first.get().target.bind(py).clone())
                        .map_or_else(|| identifier_to_python(py, condition.target()), Ok)?;
                    PyCondition::instantiate(py, condition.clone(), &target, &system)
                }
                None => {
                    let target = identifier_to_python(py, condition.target())?;
                    let system = system_to_python(py, condition.when())?;
                    PyCondition::instantiate(py, condition.clone(), &target, &system)
                }
            },
        )
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, objects)
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
        name = None, notes = None, **kwargs,
    ))]
    #[expect(
        clippy::too_many_arguments,
        reason = "the constructor's Python signature: six fields and the seed keyword"
    )]
    fn new(
        py: Python<'_>,
        variables: Option<&Bound<'_, PyAny>>,
        choices: Option<&Bound<'_, PyAny>>,
        conditions: Option<&Bound<'_, PyAny>>,
        forbidden: Option<&Bound<'_, PyAny>>,
        name: Option<&Bound<'_, PyAny>>,
        notes: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if let Some(seeded) = take_seed(kwargs)? {
            return match seeded {
                Seeded::Space(space) => Ok(space),
                _ => Err(wrong_seed()),
            };
        }
        let (core_name, name) = read_name(py, name, "Space", "space")?;
        let variables = read_items(py, variables, "Space", "variables", "Variables")?;
        let choices = read_items(py, choices, "Space", "choices", "Choices")?;
        let core_choices = read_choices(&choices, "Space")?;
        let conditions = read_items(py, conditions, "Space", "conditions", "Conditions")?;
        let core_conditions = conditions
            .iter()
            .map(|condition| {
                condition
                    .cast::<PyCondition>()
                    .map(|condition| condition.get().condition.clone())
                    .map_err(|_not_a_condition| {
                        wrong_argument("Space", "conditions", "Conditions", &condition)
                    })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let forbidden = read_items(py, forbidden, "Space", "forbidden", "Forbidden clauses")?;
        let core_forbidden = forbidden
            .iter()
            .map(|clause| {
                clause
                    .cast::<PyForbidden>()
                    .map(|clause| clause.get().forbidden.clone())
                    .map_err(|_not_a_clause| {
                        wrong_argument("Space", "forbidden", "Forbidden clauses", &clause)
                    })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let (core_notes, notes) = read_notes(py, notes, "Space")?;
        ensure_depth(py, "space", choices_depth(&core_choices))?;
        let (space, slots) = collect_slots(|| -> PyResult<Space> {
            let core_variables = variables
                .iter()
                .map(|variable| read_variable(&variable, "Space", "variables"))
                .collect::<PyResult<Vec<_>>>()?;
            with_pending_errors(|| {
                Space::new(
                    core_name,
                    core_variables,
                    core_choices,
                    core_conditions,
                    core_forbidden,
                )
                .map_err(|error| space_error_to_py(py, error))
            })
        });
        let space = space?.with_notes(core_notes);
        let conditions = kept_conditions(&space, &conditions)?;
        Self::assemble(
            space,
            SpaceObjects {
                name,
                variables,
                choices,
                conditions,
                forbidden,
                notes,
            },
            slots,
        )
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
        self.name.bind(py).clone()
    }

    /// The top-level `Variable`s, in order.
    #[getter]
    fn variables<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        self.variables.bind(py).clone()
    }

    /// The top-level `Choice`s, in order.
    #[getter]
    fn choices<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        self.choices.bind(py).clone()
    }

    /// The `Condition`s, one per target, in canonical order of the targets.
    #[getter]
    fn conditions<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        self.conditions.bind(py).clone()
    }

    /// The `Forbidden` clauses, in order.
    #[getter]
    fn forbidden<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        self.forbidden.bind(py).clone()
    }

    /// The `Note`s attached to the space.
    #[getter]
    fn notes<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        self.notes.bind(py).clone()
    }

    /// The decisions, `Variable`s and `Choice`s at every depth, in
    /// canonical order.
    #[getter]
    fn decisions<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        self.decisions.bind(py).clone()
    }

    /// The decisions' names in decision order: each after every decision
    /// it depends on, ties in canonical order.
    #[getter]
    fn decision_order<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let names = self
            .space
            .decision_order()
            .iter()
            .map(|name| match self.position(name) {
                Some(position) => self
                    .decision_at(py, position)?
                    .getattr(pyo3::intern!(py, "name")),
                None => identifier_to_python(py, name),
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, names)
    }

    /// Return the decision named `name`, or `None`.
    ///
    /// Raises `TypeError` if `name` is not an `Identifier`.
    fn decision<'py>(
        &self,
        py: Python<'py>,
        name: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let name = restore_identifier(name, "Space", "name")?;
        self.position(&name)
            .map(|position| self.decision_at(py, position))
            .transpose()
    }

    /// Return whether `other` is a space with the same names, structurally
    /// equivalent parts and equal conditions and clauses.
    fn is_structurally_equivalent(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let py = slf.py();
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        with_pending_errors(|| {
            slf.get()
                .space
                .is_structurally_equivalent(&other.get().space)
                .map_err(|error| equivalence_error_to_py(py, error))
        })
    }

    /// Ask every active decision of `oracle`, in decision order, and return
    /// `(configuration, trace)`.
    ///
    /// Raises what `Recorder.decide` raises.
    #[expect(
        unused_variables,
        reason = "interface stub: the body is todo!() until the implementation"
    )]
    fn sample<'py>(
        slf: &Bound<'py, Self>,
        oracle: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
    }

    /// Draw a complete configuration uniformly from all of them with the
    /// `Rng` `rng`, and return `(configuration, trace)`.
    ///
    /// Raises `NotEnumerableError` for a variable with no finite domain and
    /// `TraceError` after `attempts` refused draws.
    #[expect(
        unused_variables,
        reason = "interface stub: the body is todo!() until the implementation"
    )]
    #[pyo3(signature = (rng, *, attempts = 1000))]
    fn sample_uniform<'py>(
        slf: &Bound<'py, Self>,
        rng: &Bound<'py, PyAny>,
        attempts: u32,
    ) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
    }

    /// Return the `Configuration` the static steps of the `Trace` `trace`
    /// describe.
    ///
    /// Raises `ReplayMismatchError` for a trace that does not describe a
    /// configuration of this space.
    #[expect(
        unused_variables,
        reason = "interface stub: the body is todo!() until the implementation"
    )]
    fn replay<'py>(
        slf: &Bound<'py, Self>,
        trace: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return an iterator over the complete configurations, in
    /// lexicographic order of their coordinates.
    #[expect(
        unused_variables,
        reason = "interface stub: the body is todo!() until the implementation"
    )]
    fn enumerate<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return the count of complete configurations as `(kind, count,
    /// decision)`, which the public `Space.cardinality` wraps.
    #[expect(
        unused_variables,
        reason = "interface stub: the body is todo!() until the implementation"
    )]
    #[pyo3(signature = (*, budget = 100_000))]
    fn _cardinality<'py>(slf: &Bound<'py, Self>, budget: u64) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
    }

    /// Return `configuration` with one decision changed and the rest
    /// repaired, with the `Rng` `rng`, as `(configuration, trace)`.
    ///
    /// Raises `TraceError` for a configuration that is incomplete, of
    /// another space or has nothing to change, and after `attempts` dead
    /// ends.
    #[expect(
        unused_variables,
        reason = "interface stub: the body is todo!() until the implementation"
    )]
    #[pyo3(signature = (configuration, rng, *, attempts = 16))]
    fn mutate<'py>(
        slf: &Bound<'py, Self>,
        configuration: &Bound<'py, PyAny>,
        rng: &Bound<'py, PyAny>,
        attempts: u32,
    ) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
    }

    /// Return whether `other` is the same space up to the renaming of the
    /// names it binds, under no renaming.
    fn is_alpha_equivalent(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        with_pending_errors(|| {
            slf.get()
                .space
                .is_alpha_equivalent(&other.get().space)
                .map_err(|error| equivalence_error_to_py(py, error))
        })
    }

    /// Return whether `other` is the same space up to the renaming of the
    /// names it binds, under `renaming`.
    fn is_alpha_equivalent_under(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        renaming: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let py = slf.py();
        let renaming = read_renaming(renaming)?;
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        with_pending_errors(|| {
            slf.get()
                .space
                .is_alpha_equivalent_under(&other.get().space, renaming.get().value().renaming())
                .map_err(|error| equivalence_error_to_py(py, error))
        })
    }

    /// Refuse to set an attribute: the space is frozen.
    fn __setattr__(slf: &Bound<'_, Self>, name: &str, _value: &Bound<'_, PyAny>) -> PyResult<()> {
        refuse_attribute_assignment(slf.as_any(), name)
    }

    /// Refuse to delete an attribute: the space is frozen.
    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        refuse_attribute_deletion(slf.as_any(), name)
    }

    /// Return `Space(name=..., variables=..., choices=..., conditions=...,
    /// forbidden=..., notes=...)`.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        format_dataclass_repr(
            &slf.get_type(),
            &[
                ("name", this.name.bind(py)),
                ("variables", this.variables.bind(py).as_any()),
                ("choices", this.choices.bind(py).as_any()),
                ("conditions", this.conditions.bind(py).as_any()),
                ("forbidden", this.forbidden.bind(py).as_any()),
                ("notes", this.notes.bind(py).as_any()),
            ],
        )
    }

    /// Pickle as a call of the class with its fields.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.get();
        let fields = PyTuple::new(
            py,
            [
                this.variables.bind(py).as_any(),
                this.choices.bind(py).as_any(),
                this.conditions.bind(py).as_any(),
                this.forbidden.bind(py).as_any(),
                this.name.bind(py),
                this.notes.bind(py).as_any(),
            ],
        )?;
        PyTuple::new(py, [slf.get_type().into_any(), fields.into_any()])
    }

    /// Return the V2 payload `{"identifier", "variables", "choices",
    /// "conditions", "forbidden", "notes"}`.
    ///
    /// Raises `SerializationError` inside `wire_version(WireVersion.V1)`.
    fn serialize_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        refuse_v1(slf.as_any())?;
        write_part(slf.py(), || SpaceData::of(&slf.get().space))
    }

    /// Return the canonical V2 text of the payload, re-formatted for
    /// `indent` or `sort_keys`.
    #[pyo3(signature = (*, indent = None, sort_keys = None))]
    fn to_json(
        slf: &Bound<'_, Self>,
        indent: Option<&Bound<'_, PyAny>>,
        sort_keys: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        refuse_v1(slf.as_any())?;
        write_part_json(slf.as_any(), indent, sort_keys, || {
            SpaceData::of(&slf.get().space)
        })
    }

    /// Return the space of the V2 payload `data`.
    #[classmethod]
    fn deserialize_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        decode_part(cls, Family::Space, data, false)
    }

    /// Return the space of the JSON text `payload`.
    #[classmethod]
    fn from_json<'py>(
        cls: &Bound<'py, PyType>,
        payload: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        decode_part(cls, Family::Space, payload, true)
    }
}

/// Return a new public `Space` of `space`, over new objects of its parts,
/// owning `slots`.
///
/// # Errors
///
/// Raises what building the object of a part raises.
pub(super) fn space_to_python<'py>(
    py: Python<'py>,
    space: &Space,
    slots: Slots,
) -> PyResult<Bound<'py, PyAny>> {
    let variables = space
        .variables()
        .iter()
        .map(|variable| variable_to_python(py, variable))
        .collect::<PyResult<Vec<_>>>()?;
    let choices = space
        .choices()
        .iter()
        .map(|choice| choice_to_python(py, choice, Slots::default()))
        .collect::<PyResult<Vec<_>>>()?;
    let forbidden = space
        .forbidden()
        .iter()
        .map(|clause| {
            let when = system_to_python(py, clause.when())?;
            instantiate(
                PyForbidden::public_class().get(py)?,
                1,
                Seeded::Forbidden(PyForbidden {
                    forbidden: clause.clone(),
                    when: when.unbind(),
                }),
            )
        })
        .collect::<PyResult<Vec<_>>>()?;
    let notes = space
        .notes()
        .iter()
        .map(|note| note_to_python(py, note))
        .collect::<PyResult<Vec<_>>>()?;
    let conditions = kept_conditions(space, &PyTuple::empty(py))?;
    let seeded = PySpace::assemble(
        space.clone(),
        SpaceObjects {
            name: identifier_to_python(py, space.name())?,
            variables: PyTuple::new(py, variables)?,
            choices: PyTuple::new(py, choices)?,
            conditions,
            forbidden: PyTuple::new(py, forbidden)?,
            notes: PyTuple::new(py, notes)?,
        },
        slots,
    )?;
    instantiate(PySpace::public_class().get(py)?, 0, Seeded::Space(seeded))
}
