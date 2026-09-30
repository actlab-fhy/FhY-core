//! `fhy_core._rs.ConstraintSystem`: the base of the public
//! `ConstraintSystem`, over the core's [`ConstraintSystem`].
//!
//! A system keeps its members' Python objects in canonical order beside
//! the core system, so `constraints` returns them. A member of a built-in
//! kind is the core's; any other `Constraint` is driven through
//! [`PyCustomConstraint`]. The questions ask the default solver with the
//! registry snapshot, detached from the interpreter as the solver's own
//! questions are, and log their undecided outcomes.

use std::collections::HashMap;
use std::sync::Arc;

use fhy_core::foreign::Part;

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PyTuple, PyType};

use fhy_core::constraint::{
    Constraint, ConstraintContext, ConstraintError, ConstraintEvent, ConstraintObserver,
    ConstraintSystem, Outcome,
};
use fhy_core::expression::SymbolType;
use fhy_core::identifier::Identifier;
use fhy_core::solver::{CheckLimits, QueryKind};
use fhy_core::term::AlphaRenaming;

use crate::expression::{PyExpression, registry_snapshot};
use crate::frozen::{refuse_attribute_assignment, refuse_attribute_deletion};
use crate::gc::{Slots, collect_slots};
use crate::serialization::{
    FieldShape, construct_from_decoded_fields, read_constructor_fields, read_payload_fields,
    serialize_nested,
};
use crate::solver::{
    get_default_solver, read_limits, read_symbol_types, warn_hazard, warn_unknown,
};

use super::custom::{PyCustomConstraint, PythonBindings};
use super::error::constraint_error_to_py;
use super::kinds::{
    ReadBindings, is_same_class, outcome_to_python, read_native_constraint, read_scoped_bindings,
    with_renaming,
};
use super::observer::{DEBUG, LoggingObserver, WARNING, log, native_constant_refusal};
use super::value::{
    constraint_error, record_pending_error, repr_text, type_name, with_pending_errors,
};

/// Return `fhy_core.symbolic.constraint.core.Constraint`.
fn constraint_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    crate::python::cached_attr!(py, "fhy_core.symbolic.constraint.core", "Constraint" => PyType)
}

/// Return `fhy_core.symbolic.constraint.system`'s logger.
pub(crate) fn system_logger(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    static LOGGER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    LOGGER
        .get_or_try_init(py, || {
            py.import(intern!(py, "fhy_core.logger"))?
                .call_method1(
                    intern!(py, "get_logger"),
                    ("fhy_core.symbolic.constraint.system",),
                )
                .map(Bound::unbind)
        })
        .map(|logger| logger.bind(py).clone())
}

/// Return the core constraint of the Python member `member`.
///
/// # Errors
///
/// Raises `ConstraintError` for a value that is not a `Constraint`, and
/// what reading a Python-defined member's ordering key raises.
pub(crate) fn read_member(member: &Bound<'_, PyAny>) -> PyResult<Constraint> {
    let py = member.py();
    if let Some(constraint) = read_native_constraint(member) {
        return Ok(constraint);
    }
    if !member.is_instance(constraint_class(py)?)? {
        return Err(constraint_error(
            py,
            format!(
                "ConstraintSystem members must be Constraint instances, but got value {} of \
                 type {}.",
                repr_text(member),
                type_name(member)
            ),
        ));
    }
    Ok(Constraint::Custom(Part::new(PyCustomConstraint::new(
        member,
    )?)))
}

/// Where a system's observer logs its members' and questions' events.
struct SystemObserver {
    /// The members' Python objects, in canonical order.
    members: Py<PyTuple>,
    /// The bindings of the evaluation, if any.
    bindings: Option<Py<PyAny>>,
    /// The symbol types of the question, for the hazard warning.
    symbol_types: HashMap<Identifier, SymbolType>,
    /// The name of the solver's SMT backend, for the `unknown` warning.
    backend: String,
    /// The entry point, for the bound-constant warning.
    entry_point: &'static str,
}

impl SystemObserver {
    fn log_event(&self, py: Python<'_>, event: &ConstraintEvent<'_>) -> PyResult<()> {
        let members = self.members.bind(py);
        match *event {
            ConstraintEvent::InMember { index, event } => {
                let member = members.get_item(index)?;
                let bindings = self.bindings.as_ref().map_or_else(
                    || PyDict::new(py).into_any(),
                    |bindings| bindings.bind(py).clone(),
                );
                let variable = member.getattr(intern!(py, "variable")).ok().filter(|_| {
                    matches!(
                        event,
                        ConstraintEvent::Unbound { .. } | ConstraintEvent::SymbolicBinding { .. }
                    )
                });
                let bound = match &variable {
                    Some(variable) => bindings
                        .call_method1(intern!(py, "get"), (variable,))
                        .ok()
                        .filter(|value| !value.is_none())
                        .map(Bound::unbind),
                    None => None,
                };
                LoggingObserver::new(
                    member.get_type().name()?.to_string(),
                    variable.map(Bound::unbind),
                    bindings.unbind(),
                    bound,
                )
                .log_event(py, event)
            }
            ConstraintEvent::UndecidedMember { index } => {
                let member = members.get_item(index)?;
                log(&system_logger(py)?, DEBUG, || {
                    format!(
                        "ConstraintSystem.evaluate_with_bindings: member {} is undecided under \
                         the given bindings; the conjunction reports UNDECIDED unless a later \
                         member is violated",
                        repr_text(&member)
                    )
                })
            }
            ConstraintEvent::BoundNativeConstants { identifiers } => {
                log(&system_logger(py)?, WARNING, || {
                    let names = identifiers
                        .iter()
                        .map(|identifier| format!("{identifier:?}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    native_constant_refusal(self.entry_point, &names)
                })
            }
            ConstraintEvent::Refused { kind, hazard } => {
                warn_hazard(py, question_name(kind), hazard, &self.symbol_types)
            }
            ConstraintEvent::GaveUp { kind, reason } => {
                warn_unknown(py, question_name(kind), &self.backend, reason)
            }
            _ => Ok(()),
        }
    }
}

/// Return the name of the solver function a system's question of `kind`
/// stands for, which its warnings name.
const fn question_name(kind: QueryKind) -> &'static str {
    match kind {
        QueryKind::Implication => "does_expression_imply",
        _ => "check_expression_satisfiability",
    }
}

impl ConstraintObserver for SystemObserver {
    fn notify(&self, event: &ConstraintEvent<'_>) {
        Python::attach(|py| {
            if let Err(error) = self.log_event(py, event) {
                record_pending_error(error);
            }
        });
    }
}

/// The conjunction of constraints, in canonical order, backed by the core's
/// [`ConstraintSystem`].
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "ConstraintSystem")]
pub(crate) struct PyConstraintSystem {
    core: ConstraintSystem,
    /// The members' Python objects, in canonical order.
    constraints: Py<PyTuple>,
    /// The slots of the Python-defined members' adapters, which the system
    /// owns.
    slots: Slots,
}

impl PyConstraintSystem {
    /// Return the core system.
    pub(crate) const fn core(&self) -> &ConstraintSystem {
        &self.core
    }

    /// Run `question` with the default solver, the registry snapshot and an
    /// observer of `symbol_types`, detached from the interpreter, and map
    /// its error with `bindings`.
    fn ask(
        &self,
        py: Python<'_>,
        bindings: Option<(&ReadBindings<'_>, &Py<PyAny>)>,
        symbol_types: HashMap<Identifier, SymbolType>,
        entry_point: &'static str,
        question: impl FnOnce(
            &ConstraintSystem,
            &HashMap<Identifier, SymbolType>,
            &ConstraintContext<'_>,
        ) -> Result<Outcome, ConstraintError>
        + Send,
    ) -> PyResult<Outcome> {
        let solver = get_default_solver(py)?;
        let solver = solver.bind(py).get();
        let registry = registry_snapshot();
        let observer = SystemObserver {
            members: self.constraints.clone_ref(py),
            bindings: bindings.as_ref().map(|(_, mapping)| mapping.clone_ref(py)),
            symbol_types,
            backend: solver.backend_name(),
            entry_point,
        };
        let core = &self.core;
        with_pending_errors(|| {
            let outcome = py.detach(|| {
                let context = ConstraintContext::new(solver.core())
                    .with_registry(registry.registry())
                    .with_observer(&observer);
                question(core, &observer.symbol_types, &context)
            });
            outcome.map_err(|error| {
                let binding = match (&error, &bindings) {
                    (ConstraintError::UnusableBinding { identifier, .. }, Some((read, _))) => {
                        read.objects(identifier)
                    }
                    _ => None,
                };
                constraint_error_to_py(py, error, binding)
            })
        })
    }
}

/// Return the members of `constraints`, read, sorted stably by their keys.
fn read_members<'py>(
    constraints: &Bound<'py, PyAny>,
) -> PyResult<(ConstraintSystem, Bound<'py, PyTuple>)> {
    let py = constraints.py();
    let mut members: Vec<(String, Constraint, Bound<'py, PyAny>)> = Vec::new();
    for member in constraints.try_iter()? {
        let member = member?;
        let constraint = read_member(&member)?;
        let key = constraint
            .ordering_key()
            .map_err(|error| constraint_error_to_py(py, error, None))?;
        members.push((key, constraint, member));
    }
    members.sort_by(|(left, _, _), (right, _, _)| left.cmp(right));
    let objects = PyTuple::new(py, members.iter().map(|(_, _, object)| object))?;
    let core = ConstraintSystem::new(members.into_iter().map(|(_, constraint, _)| constraint))
        .map_err(|error| constraint_error_to_py(py, error, None))?;
    Ok((core, objects))
}

#[pymethods]
impl PyConstraintSystem {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.constraints)?;
        self.slots.traverse(&visit)
    }

    /// Create the system of `constraints`, an iterable of `Constraint`s,
    /// sorted by their ordering keys, duplicates kept.
    ///
    /// Raises `ConstraintError` for a member that is not a `Constraint`.
    #[new]
    #[pyo3(signature = (constraints))]
    fn new(constraints: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (members, slots) = collect_slots(|| read_members(constraints));
        let (core, objects) = members?;
        Ok(Self {
            core,
            constraints: objects.unbind(),
            slots,
        })
    }

    /// The members, in canonical order.
    #[getter]
    fn constraints(&self, py: Python<'_>) -> Py<PyTuple> {
        self.constraints.clone_ref(py)
    }

    /// Return the union of every member's free identifiers.
    fn get_free_identifiers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        let mut identifiers: Vec<Bound<'py, PyAny>> = Vec::new();
        for member in self.constraints.bind(py) {
            for identifier in member
                .call_method0(intern!(py, "get_free_identifiers"))?
                .try_iter()?
            {
                identifiers.push(identifier?);
            }
        }
        PyFrozenSet::new(py, &identifiers)
    }

    /// Return the conjunction of the members' outcomes under `bindings`:
    /// `VIOLATED` at the first violated member, `SATISFIED` if every member
    /// is, `UNDECIDED` otherwise, each undecided member logged at DEBUG.
    ///
    /// Raises what the first member to raise raises.
    fn evaluate_with_bindings<'py>(
        &self,
        bindings: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = bindings.py();
        let outcome = self.evaluate(bindings)?;
        outcome_to_python(py, outcome)
    }

    /// Return whether the bindings provably satisfy every member.
    fn is_satisfied_with_bindings(&self, bindings: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.evaluate(bindings)
            .map(|outcome| outcome == Outcome::Satisfied)
    }

    /// Return the conjunction of the members' expressions: `True` for no
    /// member, a member's own expression for one.
    fn convert_to_expression<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let expressions = self
            .constraints
            .bind(py)
            .iter()
            .map(|member| member.call_method0(intern!(py, "convert_to_expression")))
            .collect::<PyResult<Vec<_>>>()?;
        let expression_class = PyExpression::public_class().get(py)?;
        match expressions.len() {
            0 => py
                .import(intern!(py, "fhy_core.symbolic.expression"))?
                .getattr(intern!(py, "LiteralExpression"))?
                .call1((true,)),
            1 => Ok(expressions
                .into_iter()
                .next()
                .unwrap_or_else(|| py.None().into_bound(py))),
            _ => expression_class
                .call_method1(intern!(py, "logical_and"), PyTuple::new(py, expressions)?),
        }
    }

    /// Return whether some assignment satisfies every member.
    ///
    /// Raises `ValueError` for a bad `timeout_milliseconds`,
    /// `ConstraintError` for a member that does not convert,
    /// `MissingSymbolTypeError`, and `NonBooleanLogicalOperandError` for a
    /// member that cannot be a predicate.
    #[pyo3(signature = (symbol_types, *, timeout_milliseconds = None))]
    fn check_satisfiability<'py>(
        &self,
        symbol_types: &Bound<'py, PyAny>,
        timeout_milliseconds: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = symbol_types.py();
        let limits = read_timeout(py, timeout_milliseconds)?;
        if self.core.is_empty() {
            return outcome_to_python(py, Outcome::Satisfied);
        }
        let symbol_types = read_symbol_types(Some(symbol_types))?;
        let outcome = self.ask(
            py,
            None,
            symbol_types,
            "ConstraintSystem.check_satisfiability",
            move |system, symbol_types, context| {
                system.check_satisfiability(symbol_types, limits, context)
            },
        )?;
        outcome_to_python(py, outcome)
    }

    /// Return whether the system is satisfiable given the partial
    /// assignment `bindings`: set members whose variable is bound to a
    /// literal are decided by themselves, and the rest is substituted and
    /// asked about.
    ///
    /// Raises as `check_satisfiability` does, and `ConstraintError` for a
    /// binding that is not an `Expression` or a literal.
    #[pyo3(signature = (bindings, symbol_types, *, timeout_milliseconds = None))]
    fn check_satisfiability_with_bindings<'py>(
        &self,
        bindings: &Bound<'py, PyAny>,
        symbol_types: &Bound<'py, PyAny>,
        timeout_milliseconds: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = bindings.py();
        let limits = read_timeout(py, timeout_milliseconds)?;
        if self.core.is_empty() {
            return outcome_to_python(py, Outcome::Satisfied);
        }
        let read = read_scoped_bindings(bindings, None)?;
        let core_bindings = read.core.clone();
        let symbol_types = read_symbol_types(Some(symbol_types))?;
        let outcome = self.ask(
            py,
            Some((&read, &bindings.clone().unbind())),
            symbol_types,
            "ConstraintSystem.check_satisfiability_with_bindings",
            move |system, symbol_types, context| {
                system.check_satisfiability_with_bindings(
                    &core_bindings,
                    symbol_types,
                    limits,
                    context,
                )
            },
        )?;
        outcome_to_python(py, outcome)
    }

    /// Return whether every assignment satisfying this system satisfies
    /// `other`, a `ConstraintSystem`.
    ///
    /// Raises `TypeError` for another value, and as `check_satisfiability`
    /// does, for either side.
    #[pyo3(signature = (other, symbol_types, *, timeout_milliseconds = None))]
    fn check_implication<'py>(
        &self,
        other: &Bound<'py, PyAny>,
        symbol_types: &Bound<'py, PyAny>,
        timeout_milliseconds: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = other.py();
        let limits = read_timeout(py, timeout_milliseconds)?;
        let other = other.cast::<Self>().map_err(|_not_a_system| {
            PyTypeError::new_err(format!(
                "check_implication other must be a ConstraintSystem, got {}.",
                type_name(other)
            ))
        })?;
        let consequent = other.get().core.clone();
        let symbol_types = read_symbol_types(Some(symbol_types))?;
        let outcome = self.ask(
            py,
            None,
            symbol_types,
            "ConstraintSystem.check_implication",
            move |system, symbol_types, context| {
                system.check_implication(&consequent, symbol_types, limits, context)
            },
        )?;
        outcome_to_python(py, outcome)
    }

    /// Return whether `other` is a system of the same class whose members
    /// are structurally equivalent, pairwise in canonical order.
    fn is_structurally_equivalent(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        Self::compare_members(slf, other, None)
    }

    /// Return whether `other` is a system of the same class whose members
    /// are alpha-equivalent under `renaming`, pairwise in canonical order.
    ///
    /// Raises `TypeError` if `renaming` is not an `AlphaRenaming`.
    fn is_alpha_equivalent_under(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        renaming: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        with_renaming(renaming, |_renaming| ())?;
        Self::compare_members(slf, other, Some(renaming))
    }

    /// Return whether `other` is alpha-equivalent under no renaming.
    fn is_alpha_equivalent(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        let empty = py
            .get_type::<crate::term::PyAlphaRenaming>()
            .call_method0(intern!(py, "empty"))?;
        Self::compare_members(slf, other, Some(&empty))
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let members = self
            .constraints
            .bind(py)
            .iter()
            .map(|member| repr_text(&member))
            .collect::<Vec<_>>();
        format!("ConstraintSystem({})", members.join(", "))
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        let members = self.constraints.bind(py);
        if members.is_empty() {
            return Ok("True".to_owned());
        }
        Ok(members
            .iter()
            .map(|member| member.str().map(|text| text.to_string()))
            .collect::<PyResult<Vec<_>>>()?
            .join(" and "))
    }

    /// Always true: systems are immutable.
    #[getter]
    const fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: systems are always frozen.
    const fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: systems are always frozen, and mutating one raises.
    const fn assert_frozen(_slf: &Bound<'_, Self>) {}

    fn __setattr__(slf: &Bound<'_, Self>, name: &str, _value: &Bound<'_, PyAny>) -> PyResult<()> {
        refuse_attribute_assignment(slf, name)
    }

    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        refuse_attribute_deletion(slf, name)
    }

    /// Pickle as a constructor call of the class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        Ok((
            slf.get_type(),
            PyTuple::new(py, [slf.get().constraints.bind(py)])?,
        ))
    }

    /// Return the data payload `{"constraints": [..]}`, in canonical order.
    fn serialize_data_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let members = self
            .constraints
            .bind(py)
            .iter()
            .map(|member| serialize_nested(&member))
            .collect::<PyResult<Vec<_>>>()?;
        let payload = PyDict::new(py);
        payload.set_item(intern!(py, "constraints"), PyList::new(py, members)?)?;
        Ok(payload)
    }

    /// Return the system of a data payload.
    ///
    /// Raises the serialization framework's errors for a malformed payload.
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let [payloads] =
            read_payload_fields(cls, data, [("constraints", FieldShape::PayloadList)])?;
        let class = constraint_class(py)?;
        let members = payloads
            .try_iter()?
            .map(|payload| class.call_method1(intern!(py, "deserialize_from_dict"), (payload?,)))
            .collect::<PyResult<Vec<_>>>()?;
        let fields = PyDict::new(py);
        fields.set_item(intern!(py, "constraints"), PyTuple::new(py, members)?)?;
        construct_from_decoded_fields(cls, &fields)
    }

    /// Build the system of the decoded fields `{"constraints": ..}`.
    #[classmethod]
    fn construct_from_fields<'py>(
        cls: &Bound<'py, PyType>,
        fields: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let [constraints] = read_constructor_fields(cls, fields, ["constraints"], 0)?;
        cls.call1((constraints,))
    }
}

impl PyConstraintSystem {
    /// Evaluate the system under the Python `bindings`, snapshotted.
    fn evaluate(&self, bindings: &Bound<'_, PyAny>) -> PyResult<Outcome> {
        let py = bindings.py();
        let snapshot = PyDict::new(py);
        match bindings.cast::<PyDict>() {
            Ok(dict) => snapshot.update(dict.as_mapping())?,
            Err(_not_a_dict) => {
                let items = bindings.call_method0(intern!(py, "items"))?;
                for item in items.try_iter()? {
                    let (key, value) = item?.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()?;
                    snapshot.set_item(key, value)?;
                }
            }
        }
        let read = read_scoped_bindings(snapshot.as_any(), None)?;
        let core_bindings = read
            .core
            .clone()
            .with_source(Arc::new(PythonBindings(snapshot.clone().unbind())));
        let solver = get_default_solver(py)?;
        let solver = solver.bind(py).get();
        let registry = registry_snapshot();
        let observer = SystemObserver {
            members: self.constraints.clone_ref(py),
            bindings: Some(snapshot.into_any().unbind()),
            symbol_types: HashMap::new(),
            backend: solver.backend_name(),
            entry_point: "ConstraintSystem.evaluate_with_bindings",
        };
        let context = ConstraintContext::new(solver.core())
            .with_registry(registry.registry())
            .with_observer(&observer);
        with_pending_errors(|| {
            self.core
                .evaluate(&core_bindings, &context)
                .map_err(|error| {
                    let binding = match &error {
                        ConstraintError::UnusableBinding { identifier, .. } => {
                            read.objects(identifier)
                        }
                        _ => None,
                    };
                    constraint_error_to_py(py, error, binding)
                })
        })
    }

    /// Compare the members of `slf` and `other` pairwise: structurally, or
    /// under `renaming`. Two built-in members compare in the core; a pair
    /// with a Python-defined member through the left member's method.
    fn compare_members(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        renaming: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        let py = slf.py();
        if !is_same_class(slf.as_any(), other) {
            return Ok(false);
        }
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        let (left, right) = (slf.get(), other.get());
        if left.core.constraints().len() != right.core.constraints().len() {
            return Ok(false);
        }
        let left_objects = left.constraints.bind(py);
        let right_objects = right.constraints.bind(py);
        let core_renaming = match renaming {
            Some(renaming) => with_renaming(renaming, AlphaRenaming::clone)?,
            None => AlphaRenaming::default(),
        };
        with_pending_errors(|| {
            for (index, (left_member, right_member)) in left
                .core
                .constraints()
                .iter()
                .zip(right.core.constraints())
                .enumerate()
            {
                let is_equivalent = match (left_member, right_member) {
                    (Constraint::Custom(_), _) | (_, Constraint::Custom(_)) => {
                        let left_object = left_objects.get_item(index)?;
                        let right_object = right_objects.get_item(index)?;
                        let answer = match renaming {
                            Some(renaming) => left_object.call_method1(
                                intern!(py, "is_alpha_equivalent_under"),
                                (right_object, renaming),
                            )?,
                            None => left_object.call_method1(
                                intern!(py, "is_structurally_equivalent"),
                                (right_object,),
                            )?,
                        };
                        answer.is_truthy()?
                    }
                    _ => match renaming {
                        Some(_) => fhy_core::term::AlphaEquivalence::is_alpha_equivalent_under(
                            left_member,
                            right_member,
                            &core_renaming,
                        )
                        .map_err(|error| constraint_error_to_py(py, error, None))?,
                        None => left_member.is_structurally_equivalent(right_member),
                    },
                };
                if !is_equivalent {
                    return Ok(false);
                }
            }
            Ok(true)
        })
    }
}

/// Check `timeout_milliseconds` with the solver's
/// `validate_timeout_milliseconds`, and return the limits it gives.
fn read_timeout(
    py: Python<'_>,
    timeout_milliseconds: Option<&Bound<'_, PyAny>>,
) -> PyResult<CheckLimits> {
    let none = py.None().into_bound(py);
    read_limits(timeout_milliseconds.unwrap_or(&none))
}
