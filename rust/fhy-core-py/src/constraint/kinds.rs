//! `fhy_core._rs.EquationConstraint`, `InSetConstraint` and
//! `NotInSetConstraint`: the bases of the public constraint classes (P2;
//! D-S13-9), over the core's [`EquationConstraint`] and [`SetConstraint`].
//!
//! Each object keeps the Python objects it was given: the expression, the
//! variable, and the objects of opaque members, so the attributes return
//! them. Equality and hashing are by identity, as the replaced dataclasses'
//! `eq=False` gave. Evaluation asks the default solver, with the function
//! registry's snapshot, and logs undecided outcomes as the Python
//! implementation did.

use std::collections::HashMap;

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    PyBool, PyByteArray, PyBytes, PyDict, PyFloat, PyFrozenSet, PyInt, PyList, PyMapping, PyString,
    PyTuple, PyType,
};

use fhy_core::constraint::{
    Binding, Bindings, Constraint, ConstraintContext, EquationConstraint, MemberSet, Outcome,
    Polarity, SetConstraint,
};
use fhy_core::expression::ExpressionKind;
use fhy_core::identifier::Identifier;
use fhy_core::term::AlphaRenaming;
use fhy_core::tree::{NodeHandle, NodeIdentity};

use crate::expression::{
    PyExpression, PyIdentifierExpression, decimal_class, materialize_with_known, read_decimal,
    registry_snapshot,
};
use crate::frozen::build_frozen_mutation_error;
use crate::gc::{Slots, collect_slots};
use crate::identifier::{deserialize_identifier, read_identifier_id, restore_identifier};
use crate::serialization::{
    FieldShape, construct_from_decoded_fields, deserialization_value_error_class,
    read_constructor_fields, read_payload_fields,
};
use crate::solver::get_default_solver;
use crate::term::read_renaming;

use super::error::constraint_error_to_py;
use super::observer::LoggingObserver;
use super::value::{
    constraint_error, member_to_python, read_bound_value, read_member, read_member_value,
    repr_text, type_name, with_pending_errors,
};

/// Return the `ConstraintOutcome` member of `outcome`.
pub(crate) fn outcome_to_python(py: Python<'_>, outcome: Outcome) -> PyResult<Bound<'_, PyAny>> {
    static MEMBERS: PyOnceLock<[Py<PyAny>; 3]> = PyOnceLock::new();
    let members = MEMBERS.get_or_try_init(py, || -> PyResult<[Py<PyAny>; 3]> {
        let class = py
            .import(intern!(py, "fhy_core.symbolic.constraint.core"))?
            .getattr(intern!(py, "ConstraintOutcome"))?;
        Ok([
            class.getattr(intern!(py, "SATISFIED"))?.unbind(),
            class.getattr(intern!(py, "VIOLATED"))?.unbind(),
            class.getattr(intern!(py, "UNDECIDED"))?.unbind(),
        ])
    })?;
    let index = match outcome {
        Outcome::Satisfied => 0,
        Outcome::Violated => 1,
        Outcome::Undecided => 2,
    };
    Ok(members[index].bind(py).clone())
}

/// Return whether `outcome` is `ConstraintOutcome.SATISFIED`.
fn is_satisfied(outcome: Outcome) -> bool {
    outcome == Outcome::Satisfied
}

/// Return the core binding of the Python value `value`.
pub(crate) fn read_binding(value: &Bound<'_, PyAny>) -> PyResult<Binding> {
    Ok(match value.cast::<PyExpression>() {
        Ok(expression) => Binding::Expression(expression.get().expression().clone()),
        Err(_not_an_expression) => Binding::Value(read_bound_value(value)?),
    })
}

/// The bindings of one evaluation: the core's, and the Python objects of
/// each, by identifier id, for the messages that name them.
pub(crate) struct ReadBindings<'py> {
    pub(crate) core: Bindings,
    objects: HashMap<u64, (Bound<'py, PyAny>, Bound<'py, PyAny>)>,
}

impl<'py> ReadBindings<'py> {
    /// Return the Python key and value bound to `identifier`.
    pub(crate) fn objects(
        &self,
        identifier: &Identifier,
    ) -> Option<(&Bound<'py, PyAny>, &Bound<'py, PyAny>)> {
        self.objects
            .get(&identifier.id())
            .map(|(key, value)| (key, value))
    }
}

/// Return the bindings of `mapping` whose keys are identifiers in `scope`,
/// in the mapping's order, each read once.
///
/// # Errors
///
/// Raises `TypeError` if `mapping` is not a mapping, and whatever reading
/// its items raises.
pub(crate) fn read_scoped_bindings<'py>(
    mapping: &Bound<'py, PyAny>,
    scope: Option<&std::collections::HashSet<Identifier>>,
) -> PyResult<ReadBindings<'py>> {
    let items: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)> = match mapping.cast::<PyDict>() {
        Ok(dict) => dict.iter().collect(),
        Err(_not_a_dict) => {
            let mapping = mapping.cast::<PyMapping>().map_err(|_not_a_mapping| {
                PyTypeError::new_err(format!(
                    "bindings must be a mapping, got {}.",
                    type_name(mapping)
                ))
            })?;
            mapping
                .items()?
                .iter()
                .map(|item| item.extract::<(Bound<'py, PyAny>, Bound<'py, PyAny>)>())
                .collect::<PyResult<_>>()?
        }
    };
    let mut core = Bindings::new();
    let mut objects = HashMap::new();
    for (key, value) in items {
        let Some(id) = read_identifier_id(&key)? else {
            continue;
        };
        let identifier = restore_identifier(&key, "bindings", "key")?;
        if scope.is_some_and(|scope| !scope.contains(&identifier)) {
            continue;
        }
        core.insert(identifier, read_binding(&value)?);
        objects.insert(id, (key, value));
    }
    Ok(ReadBindings { core, objects })
}

/// Run `evaluate` with the default solver, the registry snapshot and
/// `observer`, and map its error with `bindings`.
fn run_evaluation(
    py: Python<'_>,
    bindings: &ReadBindings<'_>,
    observer: &LoggingObserver,
    evaluate: impl FnOnce(
        &ConstraintContext<'_>,
    ) -> Result<Outcome, fhy_core::constraint::ConstraintError>,
) -> PyResult<Outcome> {
    let solver = get_default_solver(py)?;
    let solver = solver.bind(py).get();
    let registry = registry_snapshot();
    let context = ConstraintContext::new(solver.core())
        .with_registry(registry.registry())
        .with_observer(observer);
    with_pending_errors(|| {
        evaluate(&context).map_err(|error| {
            let binding = match &error {
                fhy_core::constraint::ConstraintError::UnusableBinding { identifier, .. } => {
                    bindings.objects(identifier)
                }
                _ => None,
            };
            constraint_error_to_py(py, error, binding)
        })
    })
}

/// Return whether `other` is of exactly `this`'s class.
pub(crate) fn is_same_class(this: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> bool {
    this.get_type().is(other.get_type())
}

/// Return the renaming of `renaming`, an `AlphaRenaming`.
pub(crate) fn with_renaming<T>(
    renaming: &Bound<'_, PyAny>,
    use_renaming: impl FnOnce(&AlphaRenaming) -> T,
) -> PyResult<T> {
    let renaming = read_renaming(renaming)?;
    Ok(use_renaming(renaming.get().value().renaming()))
}

/// Return the payload of the serializable `value`.
pub(crate) fn serialize_nested<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    value.call_method0(intern!(value.py(), "serialize_to_dict"))
}

// ---------------------------------------------------------------------------
// EquationConstraint
// ---------------------------------------------------------------------------

/// A Boolean expression that must hold, backed by the core's
/// [`EquationConstraint`].
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "EquationConstraint")]
pub(crate) struct PyEquationConstraint {
    core: EquationConstraint,
    /// The expression object the constraint was built from.
    expression: Py<PyAny>,
}

#[pymethods]
impl PyEquationConstraint {
    /// Visit the Python objects the object holds, for the cycle collector (R2-003).
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.expression)?;
        Ok(())
    }

    /// Create the constraint that `expression`, an `Expression`, holds.
    ///
    /// Raises `ConstraintError` for another value.
    #[new]
    #[pyo3(signature = (expression))]
    fn new(expression: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = expression.py();
        let Ok(node) = expression.cast::<PyExpression>() else {
            return Err(constraint_error(
                py,
                format!(
                    "EquationConstraint requires an `Expression` instance, but got value {} of \
                     type {}.",
                    repr_text(expression),
                    type_name(expression)
                ),
            ));
        };
        Ok(Self {
            core: EquationConstraint::new(node.get().expression().clone()),
            expression: expression.clone().unbind(),
        })
    }

    /// The expression.
    #[getter]
    fn expression(&self, py: Python<'_>) -> Py<PyAny> {
        self.expression.clone_ref(py)
    }

    /// Return the expression's free identifiers.
    fn get_free_identifiers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.expression
            .bind(py)
            .call_method0(intern!(py, "get_free_identifiers"))
    }

    /// Substitute the bindings in scope, simplify, and classify: `SATISFIED`
    /// for the literal `True`, `VIOLATED` for `False`, `UNDECIDED` for
    /// anything that is not a literal.
    ///
    /// Raises `ConstraintError` for an unusable binding in scope,
    /// `NonBooleanLogicalOperandError` for an ill-typed predicate or a
    /// non-Boolean result, and the simplifier's error.
    fn evaluate_with_bindings<'py>(
        slf: &Bound<'py, Self>,
        bindings: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let outcome = Self::evaluate(slf, bindings)?;
        outcome_to_python(py, outcome)
    }

    /// Return whether the bindings provably satisfy the constraint.
    fn is_satisfied_with_bindings(
        slf: &Bound<'_, Self>,
        bindings: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        Self::evaluate(slf, bindings).map(is_satisfied)
    }

    /// Return the expression.
    fn convert_to_expression(&self, py: Python<'_>) -> Py<PyAny> {
        self.expression.clone_ref(py)
    }

    /// Return the canonical ordering key.
    fn build_ordering_key(&self) -> String {
        self.core.ordering_key()
    }

    /// Return whether `other` is an equation of the same class over a
    /// structurally equal expression.
    fn is_structurally_equivalent(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> bool {
        is_same_class(slf.as_any(), other)
            && other
                .cast::<Self>()
                .is_ok_and(|other| slf.get().core.is_structurally_equivalent(&other.get().core))
    }

    /// Return whether `other` is an equation of the same class over an
    /// alpha-equivalent expression under `renaming`.
    ///
    /// Raises `TypeError` if `renaming` is not an `AlphaRenaming`.
    fn is_alpha_equivalent_under(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        renaming: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        with_renaming(renaming, |renaming| {
            is_same_class(slf.as_any(), other)
                && other.cast::<Self>().is_ok_and(|other| {
                    slf.get()
                        .core
                        .is_alpha_equivalent_under(&other.get().core, renaming)
                })
        })
    }

    /// Return whether `other` is alpha-equivalent under no renaming.
    fn is_alpha_equivalent(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> bool {
        is_same_class(slf.as_any(), other)
            && other.cast::<Self>().is_ok_and(|other| {
                slf.get()
                    .core
                    .is_alpha_equivalent_under(&other.get().core, &AlphaRenaming::default())
            })
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "EquationConstraint(expression={})",
            repr_text(self.expression.bind(py))
        )
    }

    fn __str__(&self) -> String {
        self.core.expression().to_string()
    }

    /// Always true: constraints are immutable.
    #[getter]
    fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: constraints are always frozen.
    fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: constraints are always frozen, and mutating one raises.
    fn assert_frozen(_slf: &Bound<'_, Self>) {}

    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let _ = value;
        Err(build_frozen_mutation_error(slf, "modify", name)?)
    }

    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        Err(build_frozen_mutation_error(slf, "delete", name)?)
    }

    /// Pickle as a constructor call of the class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        Ok((
            slf.get_type(),
            PyTuple::new(py, [slf.get().expression.bind(py)])?,
        ))
    }

    /// Return the data payload `{"expression": ..}`.
    fn serialize_data_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let payload = PyDict::new(py);
        payload.set_item(
            intern!(py, "expression"),
            serialize_nested(self.expression.bind(py))?,
        )?;
        Ok(payload)
    }

    /// Return the constraint of a data payload.
    ///
    /// Raises the serialization framework's errors for a malformed payload.
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let [payload] = read_payload_fields(cls, data, [("expression", FieldShape::Payload)])?;
        let expression = PyExpression::public_class()
            .get(py)?
            .call_method1(intern!(py, "deserialize_from_dict"), (payload,))?;
        let fields = PyDict::new(py);
        fields.set_item(intern!(py, "expression"), expression)?;
        construct_from_decoded_fields(cls, &fields)
    }

    /// Build the constraint of the decoded fields `{"expression": ..}`.
    #[classmethod]
    fn construct_from_fields<'py>(
        cls: &Bound<'py, PyType>,
        fields: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let [expression] = read_constructor_fields(cls, fields, ["expression"], 0)?;
        cls.call1((expression,))
    }
}

impl PyEquationConstraint {
    /// Evaluate the constraint under the Python `bindings`.
    fn evaluate(slf: &Bound<'_, Self>, bindings: &Bound<'_, PyAny>) -> PyResult<Outcome> {
        let py = slf.py();
        let this = slf.get();
        let scope = this.core.free_identifiers();
        let read = read_scoped_bindings(bindings, Some(&scope))?;
        let observer = LoggingObserver::new(
            slf.get_type().name()?.to_string(),
            None,
            bindings.clone().unbind(),
            None,
        );
        run_evaluation(py, &read, &observer, |context| {
            this.core.evaluate(&read.core, context)
        })
    }
}

// ---------------------------------------------------------------------------
// The set constraints
// ---------------------------------------------------------------------------

/// The state of a set constraint: the core constraint, the variable object
/// it was built from, and its members as Python values, in canonical order.
struct SetState {
    core: SetConstraint,
    variable: Py<PyAny>,
    values: Py<PyTuple>,
    /// The slots of the opaque members' adapters, which the constraint
    /// owns (R2-003).
    slots: Slots,
}

impl SetState {
    /// Return the state of the constraint of class `class_name` over the
    /// Python `variable` and `values` with `polarity`.
    fn new(
        class_name: &str,
        variable: &Bound<'_, PyAny>,
        values: &Bound<'_, PyAny>,
        polarity: Polarity,
    ) -> PyResult<Self> {
        let py = variable.py();
        if read_identifier_id(variable)?.is_none() {
            return Err(constraint_error(
                py,
                format!(
                    "{class_name} constrains an identifier, but got {} of type {}. Scope, \
                     canonical ordering, and evaluation all key on the identifier, so a \
                     non-identifier fails far from here.",
                    repr_text(variable),
                    type_name(variable)
                ),
            ));
        }
        let identifier = restore_identifier(variable, class_name, "variable")?;
        let (members, slots) = collect_slots(|| read_member_collection(values));
        let members = members?;
        let objects = members
            .iter()
            .map(|member| member_to_python(py, member))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            core: SetConstraint::new(identifier, members, polarity),
            variable: variable.clone().unbind(),
            values: PyTuple::new(py, objects)?.unbind(),
            slots,
        })
    }

    /// Visit the Python objects, for the cycle collector (R2-003).
    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.variable)?;
        visit.call(&self.values)?;
        self.slots.traverse(visit)
    }

    /// Decide the constraint under the Python `bindings`, as the object
    /// `this` of class `kind`.
    fn evaluate(&self, this: &Bound<'_, PyAny>, bindings: &Bound<'_, PyAny>) -> PyResult<Outcome> {
        let py = this.py();
        let variable = self.variable.bind(py);
        let bound = match bindings.cast::<PyDict>() {
            Ok(dict) => dict.get_item(variable)?,
            Err(_not_a_dict) => {
                static UNBOUND: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
                let sentinel = UNBOUND.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
                    Ok(py
                        .import(intern!(py, "builtins"))?
                        .getattr(intern!(py, "object"))?
                        .call0()?
                        .unbind())
                })?;
                let value = bindings.call_method1(intern!(py, "get"), (variable, sentinel))?;
                (!value.is(sentinel.bind(py))).then_some(value)
            }
        };
        let mut core = Bindings::new();
        let mut objects = HashMap::new();
        if let Some(value) = &bound {
            core.insert(self.core.variable().clone(), read_binding(value)?);
            objects.insert(self.core.variable().id(), (variable.clone(), value.clone()));
        }
        let read = ReadBindings { core, objects };
        let observer = LoggingObserver::new(
            this.get_type().name()?.to_string(),
            Some(self.variable.clone_ref(py)),
            bindings.clone().unbind(),
            bound.map(Bound::unbind),
        );
        run_evaluation(py, &read, &observer, |context| {
            self.core.evaluate(&read.core, context)
        })
    }

    /// Return the expression equivalent to the constraint, whose references
    /// to the variable hold the variable object.
    fn convert_to_expression<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let expression = self
            .core
            .to_expression()
            .map_err(|error| constraint_error_to_py(py, error, None))?;
        let reference = PyIdentifierExpression::public_class()
            .get(py)?
            .call1((self.variable.bind(py),))?;
        let mut known: HashMap<NodeIdentity, Bound<'py, PyAny>> = HashMap::new();
        let mut pending = vec![&expression];
        while let Some(node) = pending.pop() {
            if matches!(node.kind(), ExpressionKind::Identifier(_)) {
                known.insert(node.identity(), reference.clone());
            }
            pending.extend(node.children());
        }
        materialize_with_known(py, &expression, known)
    }

    /// Return the `repr` of the constraint of class `kind`.
    fn repr(&self, py: Python<'_>, kind: &str) -> PyResult<String> {
        let members = self
            .values
            .bind(py)
            .iter()
            .map(|value| value.repr().map(|text| text.to_string()))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(format!(
            "{kind}({}, values={{{}}})",
            self.variable.bind(py).repr()?,
            members.join(", ")
        ))
    }

    /// Return the `str` of the constraint, with `connective` between the
    /// variable and the members.
    fn str(&self, py: Python<'_>, connective: &str) -> PyResult<String> {
        let members = self
            .values
            .bind(py)
            .iter()
            .map(|value| {
                if value.is_instance_of::<PyString>() {
                    value.repr().map(|text| text.to_string())
                } else {
                    value.str().map(|text| text.to_string())
                }
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(format!(
            "{} {connective} {{{}}}",
            self.variable.bind(py).str()?,
            members.join(", ")
        ))
    }

    /// Return the data payload `{"variable": .., "values": [..]}`.
    fn serialize_data_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let serialize = crate::python::cached_attr!(py, "fhy_core.serialization", "serialize_registry_wrapped_value" => PyAny)?;
        let members = self
            .values
            .bind(py)
            .iter()
            .map(|value| serialize.call1((value,)))
            .collect::<PyResult<Vec<_>>>()?;
        let payload = PyDict::new(py);
        payload.set_item(
            intern!(py, "variable"),
            serialize_nested(self.variable.bind(py))?,
        )?;
        payload.set_item(intern!(py, "values"), PyList::new(py, members)?)?;
        Ok(payload)
    }
}

/// Return the member set of the Python collection `values`.
///
/// # Errors
///
/// Raises `ConstraintError` for a `str`, `bytes` or `bytearray`, which
/// would split into its elements, a mapping, which would keep only its
/// keys, and each member that cannot be one, in the collection's order.
fn read_member_collection(values: &Bound<'_, PyAny>) -> PyResult<MemberSet> {
    let py = values.py();
    if values.is_instance_of::<PyString>()
        || values.is_instance_of::<PyBytes>()
        || values.is_instance_of::<PyByteArray>()
    {
        let class = type_name(values);
        return Err(constraint_error(
            py,
            format!(
                "Constraint members must be given as a collection of members, not a bare \
                 {class}, which would be split into its elements. Wrap a single member in a \
                 container, e.g. {{{}}}.",
                repr_text(values)
            ),
        ));
    }
    if values.cast::<PyMapping>().is_ok() || values.is_instance_of::<PyDict>() {
        return Err(constraint_error(
            py,
            format!(
                "Constraint members must be given as a collection of members, not a {}, whose \
                 values would be silently discarded and only its keys kept as members. Pass \
                 the intended members directly, e.g. as a set or tuple.",
                type_name(values)
            ),
        ));
    }
    let members = values
        .try_iter()?
        .map(|value| read_member(&value?))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(MemberSet::new(members))
}

/// Return the members of a decoded payload's `values`, each deserialized
/// and validated.
///
/// # Errors
///
/// Raises `DeserializationValueError` naming the field for a member that
/// does not decode, and `ConstraintError` for one that cannot be a member.
fn decode_members<'py>(values: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let py = values.py();
    let deserialize = crate::python::cached_attr!(py, "fhy_core.serialization", "deserialize_registry_wrapped_value" => PyAny)?;
    let structure_error = crate::python::cached_attr!(py, "fhy_core.serialization", "DeserializationDictStructureError" => PyType)?;
    let decoded = PyList::empty(py);
    for payload in values.try_iter()? {
        let member = match deserialize.call1((payload?,)) {
            Ok(member) => member,
            Err(error)
                if error.is_instance(py, structure_error)
                    || error.is_instance(py, deserialization_value_error_class(py)?) =>
            {
                let wrapped =
                    PyErr::from_value(deserialization_value_error_class(py)?.call1((format!(
                        "Invalid serialized member in field \"values\": {}",
                        error.value(py).str()?
                    ),))?);
                wrapped.set_cause(py, Some(error));
                return Err(wrapped);
            }
            Err(error) => return Err(error),
        };
        read_member_value(&member)?;
        decoded.append(member)?;
    }
    Ok(decoded)
}

/// Return the set constraint of class `cls` of a data payload.
fn deserialize_set_payload<'py>(
    cls: &Bound<'py, PyType>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = cls.py();
    let [variable, values] = read_payload_fields(
        cls,
        data,
        [
            ("variable", FieldShape::Payload),
            ("values", FieldShape::PayloadList),
        ],
    )?;
    let fields = PyDict::new(py);
    fields.set_item(intern!(py, "variable"), deserialize_identifier(&variable)?)?;
    fields.set_item(intern!(py, "values"), decode_members(&values)?)?;
    construct_from_decoded_fields(cls, &fields)
}

/// Define the pyclass of a set constraint of one polarity.
macro_rules! set_constraint_class {
    ($class:ident, $name:literal, $polarity:expr, $connective:literal) => {
        #[doc = concat!("`", $name, "`, backed by the core's [`SetConstraint`] of polarity `", stringify!($polarity), "`.")]
        #[pyclass(subclass, frozen, module = "fhy_core._rs", name = $name)]
        pub(crate) struct $class {
            state: SetState,
        }

        #[pymethods]
        impl $class {
            /// Visit the Python objects, for the cycle collector (R2-003).
            fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
                self.state.traverse(&visit)
            }

            /// Create the constraint on `variable`, an `Identifier`, and
            /// the members `values`.
            ///
            /// Raises `ConstraintError` for a variable that is no
            /// `Identifier`, and for members that cannot be members.
            #[new]
            #[classmethod]
            #[pyo3(signature = (variable, values))]
            fn new(
                cls: &Bound<'_, PyType>,
                variable: &Bound<'_, PyAny>,
                values: &Bound<'_, PyAny>,
            ) -> PyResult<Self> {
                let name = cls.name()?.to_string();
                Ok(Self {
                    state: SetState::new(&name, variable, values, $polarity)?,
                })
            }

            /// The constrained identifier.
            #[getter]
            fn variable(&self, py: Python<'_>) -> Py<PyAny> {
                self.state.variable.clone_ref(py)
            }

            /// The members, in canonical order.
            #[getter]
            fn values(&self, py: Python<'_>) -> Py<PyTuple> {
                self.state.values.clone_ref(py)
            }

            /// The members, in canonical order.
            #[getter]
            fn members(&self, py: Python<'_>) -> Py<PyTuple> {
                self.state.values.clone_ref(py)
            }

            /// Return the scope, `frozenset({variable})`.
            fn get_free_identifiers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
                PyFrozenSet::new(py, [self.state.variable.bind(py)])
            }

            /// Decide membership of the variable's bound value.
            ///
            /// Raises `ConstraintError` for a bound value that could never
            /// be a member, or is unhashable.
            fn evaluate_with_bindings<'py>(
                slf: &Bound<'py, Self>,
                bindings: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let outcome = slf.get().state.evaluate(slf.as_any(), bindings)?;
                outcome_to_python(slf.py(), outcome)
            }

            /// Return whether the bindings provably satisfy the constraint.
            fn is_satisfied_with_bindings(
                slf: &Bound<'_, Self>,
                bindings: &Bound<'_, PyAny>,
            ) -> PyResult<bool> {
                slf.get()
                    .state
                    .evaluate(slf.as_any(), bindings)
                    .map(is_satisfied)
            }

            /// Return the expression equivalent to the constraint.
            ///
            /// Raises `ConstraintError` for a member that does not lift to a
            /// literal.
            fn convert_to_expression<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                self.state.convert_to_expression(py)
            }

            /// Return the canonical ordering key.
            fn build_ordering_key(&self) -> String {
                self.state.core.ordering_key()
            }

            /// Return whether `other` is of the same class, with the same
            /// variable and members.
            fn is_structurally_equivalent(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
                with_pending_errors(|| {
                    Ok(is_same_class(slf.as_any(), other)
                        && other.cast::<Self>().is_ok_and(|other| {
                            slf.get()
                                .state
                                .core
                                .is_structurally_equivalent(&other.get().state.core)
                        }))
                })
            }

            /// Return whether `other` is of the same class, with the same
            /// members and a variable corresponding under `renaming`.
            ///
            /// Raises `TypeError` if `renaming` is not an `AlphaRenaming`.
            fn is_alpha_equivalent_under(
                slf: &Bound<'_, Self>,
                other: &Bound<'_, PyAny>,
                renaming: &Bound<'_, PyAny>,
            ) -> PyResult<bool> {
                with_pending_errors(|| {
                    with_renaming(renaming, |renaming| {
                        is_same_class(slf.as_any(), other)
                            && other.cast::<Self>().is_ok_and(|other| {
                                slf.get()
                                    .state
                                    .core
                                    .is_alpha_equivalent_under(&other.get().state.core, renaming)
                            })
                    })
                })
            }

            /// Return whether `other` is alpha-equivalent under no renaming.
            fn is_alpha_equivalent(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
                with_pending_errors(|| {
                    Ok(is_same_class(slf.as_any(), other)
                        && other.cast::<Self>().is_ok_and(|other| {
                            slf.get().state.core.is_alpha_equivalent_under(
                                &other.get().state.core,
                                &AlphaRenaming::default(),
                            )
                        }))
                })
            }

            fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
                slf.get()
                    .state
                    .repr(slf.py(), &slf.get_type().name()?.to_string())
            }

            fn __str__(&self, py: Python<'_>) -> PyResult<String> {
                self.state.str(py, $connective)
            }

            /// Always true: constraints are immutable.
            #[getter]
            fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
                true
            }

            /// Do nothing: constraints are always frozen.
            fn freeze(_slf: &Bound<'_, Self>) {}

            /// Do nothing: constraints are always frozen, and mutating one
            /// raises.
            fn assert_frozen(_slf: &Bound<'_, Self>) {}

            fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
                let _ = value;
                Err(build_frozen_mutation_error(slf, "modify", name)?)
            }

            fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
                Err(build_frozen_mutation_error(slf, "delete", name)?)
            }

            /// Pickle as a constructor call of the class.
            fn __reduce__<'py>(
                slf: &Bound<'py, Self>,
            ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
                let py = slf.py();
                let state = &slf.get().state;
                Ok((
                    slf.get_type(),
                    PyTuple::new(
                        py,
                        [state.variable.bind(py).clone(), state.values.bind(py).clone().into_any()],
                    )?,
                ))
            }

            /// Return the data payload `{"variable": .., "values": [..]}`,
            /// the members in canonical order.
            fn serialize_data_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
                self.state.serialize_data_to_dict(py)
            }

            /// Return the constraint of a data payload.
            ///
            /// Raises the serialization framework's errors for a malformed
            /// payload.
            #[classmethod]
            fn deserialize_data_from_dict<'py>(
                cls: &Bound<'py, PyType>,
                data: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                deserialize_set_payload(cls, data)
            }

            /// Build the constraint of the decoded fields
            /// `{"variable": .., "values": ..}`.
            #[classmethod]
            fn construct_from_fields<'py>(
                cls: &Bound<'py, PyType>,
                fields: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                let [variable, values] =
                    read_constructor_fields(cls, fields, ["variable", "values"], 0)?;
                cls.call1((variable, values))
            }
        }
    };
}

set_constraint_class!(PyInSetConstraint, "InSetConstraint", Polarity::In, "in");
set_constraint_class!(
    PyNotInSetConstraint,
    "NotInSetConstraint",
    Polarity::NotIn,
    "not in"
);

/// Return the core constraint of a Python constraint of a built-in kind,
/// or `None` for any other value.
pub(crate) fn read_native_constraint(value: &Bound<'_, PyAny>) -> Option<Constraint> {
    if let Ok(equation) = value.cast::<PyEquationConstraint>() {
        return Some(Constraint::from(equation.get().core.clone()));
    }
    if let Ok(set) = value.cast::<PyInSetConstraint>() {
        return Some(Constraint::from(set.get().state.core.clone()));
    }
    if let Ok(set) = value.cast::<PyNotInSetConstraint>() {
        return Some(Constraint::from(set.get().state.core.clone()));
    }
    None
}

/// Return whether a constraint member lifts to a `LiteralExpression`: a
/// `bool`, an `int`, a `float` or a `Decimal` the literal holds, as the
/// Python implementation answered it; not a `str`, whose literal would
/// compare against numbers, nor any other value.
#[pyfunction]
pub(crate) fn does_member_lift_to_expression(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = value.py();
    if value.is_instance_of::<PyBool>()
        || value.is_instance_of::<PyInt>()
        || value.is_instance_of::<PyFloat>()
    {
        return Ok(true);
    }
    Ok(value.is_instance(decimal_class(py)?)? && read_decimal(value).is_ok())
}
