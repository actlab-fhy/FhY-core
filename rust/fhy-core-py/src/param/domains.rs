//! `fhy_core._rs.IntegerDomain`, `IntervalIntegerDomain`, `RealDomain`,
//! `OrdinalDomain`, `CategoricalDomain` and `PermutationDomain`: the bases
//! of the public domain classes, over the core's [`ParamDomain`].
//!
//! A finite domain keeps the Python objects of its values in the core's
//! order, so its attribute returns them. The questions ask the default
//! solver with the registry snapshot, detached from the interpreter as the
//! constraint system's questions are, and log their undecided outcomes.

use fhy_core::param::{Inclusivity, Sign, ZeroInclusion};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PyTuple, PyType};

use fhy_core::constraint::{Constraint, Member, Outcome, Value};
use fhy_core::identifier::Identifier;
use fhy_core::param::{
    CategoricalDomain, DomainError, DomainKind, IntegerDomain, IntervalIntegerDomain,
    OrdinalDomain, ParamContext, ParamDomain, PermutationDomain, RealDomain, Side,
};

use crate::constraint::{join_items, member_to_python, outcome_to_python};
use crate::identifier::restore_identifier;
use crate::solver::symbol_type_to_python;
use crate::util::dataclass::OptionalArgument;
use crate::util::frozen::{refuse_attribute_assignment, refuse_attribute_deletion};
use crate::util::serialization::{construct_from_decoded_fields, is_serialized_dict};

use super::error::{ParamFailure, ordinal_error_to_py, param_error_to_py};
use super::objects::{
    constraints_to_python, domain_to_python, profile_to_python, read_constraints, read_domain,
};
use super::value::{read_candidate, read_finite_values, to_tuple};
use crate::util::pending::{capture_pending_errors, with_pending_errors};

pub(super) use crate::convert::param::run_with_context;

/// Return the context of a value-set question: a solver without backends,
/// which the built-in domains never ask, and which a Python-defined domain,
/// whose hook takes no context, never sees.
pub(super) fn value_set_context() -> ParamContext<'static> {
    static SOLVER: std::sync::LazyLock<fhy_core::solver::Solver> =
        std::sync::LazyLock::new(fhy_core::solver::Solver::new);
    ParamContext::new(&SOLVER)
}

/// Run `question` as [`run_with_context`] does, naming `other` as the other
/// domain of a set operation in its errors.
pub(super) fn run_question<T: Send, E: Into<ParamFailure> + Send>(
    py: Python<'_>,
    is_detached: bool,
    other: Option<&Bound<'_, PyAny>>,
    question: impl FnOnce(&ParamContext<'_>) -> Result<T, E> + Send,
) -> PyResult<T> {
    run_with_context(py, is_detached, question, |error| {
        param_error_to_py(py, error, other)
    })
}

/// Return the variable `variable`, an `Identifier`.
fn read_variable(variable: &Bound<'_, PyAny>) -> PyResult<Identifier> {
    restore_identifier(variable, "ParamDomain", "variable")
}

/// The state of a domain object: the core domain, and the Python objects
/// of a finite domain's values, in its order.
pub(crate) struct DomainState {
    core: ParamDomain,
    values: Option<Py<PyTuple>>,
    /// The slots of the opaque values' adapters, which the domain owns.
    slots: crate::util::gc::Slots,
}

impl DomainState {
    /// Return the state of the numeric `core`.
    fn numeric(core: impl Into<ParamDomain>) -> Self {
        Self {
            core: core.into(),
            values: None,
            slots: crate::util::gc::Slots::default(),
        }
    }

    /// Return the state of the finite `core` of `members`.
    fn finite(py: Python<'_>, core: impl Into<ParamDomain>, members: &[Member]) -> PyResult<Self> {
        let objects = members
            .iter()
            .map(|member| member_to_python(py, member))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            core: core.into(),
            values: Some(PyTuple::new(py, objects)?.unbind()),
            slots: crate::util::gc::Slots::default(),
        })
    }

    /// Return the Python objects of the values, or an empty tuple.
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        self.values
            .as_ref()
            .map_or_else(|| PyTuple::empty(py), |values| values.bind(py).clone())
    }

    fn symbol_type<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.core.symbol_type() {
            Ok(Some(symbol_type)) => symbol_type_to_python(py, symbol_type),
            Ok(None) => Ok(py.None().into_bound(py)),
            Err(error) => Err(param_error_to_py(py, error, None)),
        }
    }

    fn is_value_admissible(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let candidate = read_candidate(value, matches!(self.core, ParamDomain::Permutation(_)))?;
        with_pending_errors(|| {
            self.core
                .is_value_admissible(&candidate)
                .map_err(|error| param_error_to_py(py, error, None))
        })
    }

    fn normalize_value<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        match self.core {
            ParamDomain::Permutation(_) => to_tuple(value),
            _ => Ok(value.clone()),
        }
    }

    fn validate_constraint(
        &self,
        constraint: &Bound<'_, PyAny>,
        variable: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let py = constraint.py();
        let constraint = crate::constraint::read_constraint(constraint)?;
        let variable = read_variable(variable)?;
        self.core
            .validate_constraint(&constraint, &variable)
            .map_err(|error| param_error_to_py(py, error, None))
    }

    fn implied_constraints<'py>(
        &self,
        variable: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let py = variable.py();
        let variable = read_variable(variable)?;
        let implied = self
            .core
            .implied_constraints(&variable)
            .map_err(|error| param_error_to_py(py, error, None))?;
        constraints_to_python(py, &implied)
    }

    fn interval_profile<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.core.interval_profile() {
            Ok(Some(profile)) => profile_to_python(py, profile),
            Ok(None) => Ok(py.None().into_bound(py)),
            Err(error) => Err(param_error_to_py(py, error, None)),
        }
    }

    fn is_value_set_subset(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = other.py();
        let other = read_domain(other);
        with_pending_errors(|| {
            self.core
                .is_value_set_subset(&other, &value_set_context())
                .map_err(|error| param_error_to_py(py, error, None))
        })
    }

    fn feasibility_subset<'py>(
        &self,
        own_constraints: &Bound<'py, PyAny>,
        own_variable: &Bound<'py, PyAny>,
        other: &Bound<'py, PyAny>,
        other_constraints: &Bound<'py, PyAny>,
        other_variable: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = other.py();
        let own_constraints = read_constraints(own_constraints)?;
        let own_variable = read_variable(own_variable)?;
        let other_domain = read_domain(other);
        let other_constraints = read_constraints(other_constraints)?;
        let other_variable = read_variable(other_variable)?;
        let core = &self.core;
        let outcome = run_question(py, true, Some(other), |context| {
            core.feasibility_subset(
                Side::new(&own_constraints, &own_variable),
                &other_domain,
                Side::new(&other_constraints, &other_variable),
                context,
            )
        })?;
        outcome_to_python(py, outcome)
    }

    fn has_feasible_value<'py>(
        &self,
        constraints: &Bound<'py, PyAny>,
        variable: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = constraints.py();
        let constraints = read_constraints(constraints)?;
        let variable = read_variable(variable)?;
        let core = &self.core;
        let outcome: Outcome = run_question(py, true, None, |context| {
            core.has_feasible_value(Side::new(&constraints, &variable), context)
        })?;
        outcome_to_python(py, outcome)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the six arguments of the Python method, and the object itself"
    )]
    fn set_operation<'py>(
        &self,
        this: &Bound<'py, PyAny>,
        is_union: bool,
        own_constraints: &Bound<'py, PyAny>,
        own_variable: &Bound<'py, PyAny>,
        other: &Bound<'py, PyAny>,
        other_constraints: &Bound<'py, PyAny>,
        other_variable: &Bound<'py, PyAny>,
        variable: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = other.py();
        let own_constraints = read_constraints(own_constraints)?;
        let own_variable = read_variable(own_variable)?;
        let other_domain = read_domain(other);
        let other_constraints = read_constraints(other_constraints)?;
        let other_variable = read_variable(other_variable)?;
        let result_variable = read_variable(variable)?;
        let core = &self.core;
        let result: Option<(ParamDomain, Vec<Constraint>)> =
            run_question(py, true, Some(other), |context| {
                let own = Side::new(&own_constraints, &own_variable);
                let other = Side::new(&other_constraints, &other_variable);
                if is_union {
                    core.union(own, &other_domain, other, &result_variable, context)
                } else {
                    core.intersection(own, &other_domain, other, &result_variable, context)
                        .map(Some)
                }
            })?;
        let Some((domain, constraints)) = result else {
            return Ok(py.None().into_bound(py));
        };
        let domain = match (&self.core, &domain) {
            (ParamDomain::Permutation(_), ParamDomain::Permutation(_)) => this.clone(),
            _ => domain_to_python(py, &domain)?,
        };
        Ok(PyTuple::new(
            py,
            [domain, constraints_to_python(py, &constraints)?.into_any()],
        )?
        .into_any())
    }

    fn is_structurally_equivalent(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let other = read_domain(other);
        with_pending_errors(|| Ok(self.core.is_structurally_equivalent(&other)))
    }

    fn render_set(&self, py: Python<'_>, is_repr: bool) -> PyResult<String> {
        match self.core {
            ParamDomain::Integer(_) | ParamDomain::IntervalInteger(_) => Ok(if is_repr {
                String::new()
            } else {
                "Z".to_owned()
            }),
            ParamDomain::Real(_) => Ok(if is_repr {
                String::new()
            } else {
                "R".to_owned()
            }),
            _ => {
                let values = self.values(py);
                let text = if is_repr {
                    join_items(values.as_any())?
                } else {
                    let texts = values
                        .iter()
                        .map(|value| value.str().map(|text| text.to_string()))
                        .collect::<PyResult<Vec<_>>>()?;
                    texts.join(", ")
                };
                Ok(format!("{{{text}}}"))
            }
        }
    }
}

/// Return the payload of the finite domain's `values`: each value through
/// the serialization framework's wrapped registry.
fn serialize_values<'py>(values: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyList>> {
    let py = values.py();
    let serialize = crate::util::python::cached_attr!(py, "fhy_core.serialization", "serialize_registry_wrapped_value" => PyAny)?;
    let payloads = values
        .iter()
        .map(|value| serialize.call1((value,)))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, payloads)
}

/// Return the description of a finite kind's values in deserialization
/// errors.
const fn values_description(kind: DomainKind) -> &'static str {
    match kind {
        DomainKind::Ordinal => "a list of orderable serializable values or primitive values",
        _ => "a list of equal serializable values or primitive values",
    }
}

/// Return the `DeserializationValueError(class, field, description, value)`.
fn value_error(
    class: &Bound<'_, PyAny>,
    field: &str,
    description: &str,
    value: &Bound<'_, PyAny>,
) -> PyErr {
    let py = class.py();
    crate::util::exceptions::DESERIALIZATION_VALUE_ERROR.err(py, (class, field, description, value))
}

/// Return the values of the payload `data` of a finite domain of `kind`,
/// decoded and validated.
fn decode_values<'py>(kind: DomainKind, data: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let py = data.py();
    let Ok(payloads) = data.cast::<PyList>() else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected a list of wrapped leaf values.",
        ));
    };
    let owner = crate::util::python::cached_attr!(py, "fhy_core.symbolic.param.domains", "ParamDomain" => PyType)?;
    let description = values_description(kind);
    for payload in payloads.iter() {
        if !is_serialized_dict(&payload)? {
            return Err(value_error(
                owner.as_any(),
                "possible_values",
                description,
                data,
            ));
        }
    }
    let deserialize = crate::util::python::cached_attr!(py, "fhy_core.serialization", "deserialize_registry_wrapped_value" => PyAny)?;
    let decoded = PyList::empty(py);
    for payload in payloads.iter() {
        decoded.append(deserialize.call1((payload,))?)?;
    }
    for value in decoded.iter() {
        if read_finite_values(kind, PyTuple::new(py, [&value])?.as_any()).is_err() {
            return Err(value_error(
                owner.as_any(),
                "possible_values",
                description,
                &value,
            ));
        }
    }
    Ok(decoded)
}

/// Return the domain of class `cls` of the payload `data` holding the
/// fields `fields`, each a flag or, for a finite domain of `kind`, its
/// values, as a dataclass's derived deserialization builds it.
fn deserialize_domain<'py>(
    cls: &Bound<'py, PyType>,
    data: &Bound<'py, PyAny>,
    fields: &[(&str, Option<DomainKind>)],
) -> PyResult<Bound<'py, PyAny>> {
    let py = cls.py();
    let is_well_formed = match data.cast::<pyo3::types::PyMapping>() {
        Ok(mapping) => {
            mapping.len()? == fields.len()
                && fields.iter().try_fold(
                    true,
                    |is_well_formed, (name, kind)| -> PyResult<bool> {
                        if !is_well_formed || !mapping.contains(*name)? {
                            return Ok(false);
                        }
                        Ok(kind.is_some() || mapping.get_item(*name)?.is_instance_of::<PyBool>())
                    },
                )?
        }
        Err(_not_a_mapping) => false,
    };
    if !is_well_formed {
        let expected = PyDict::new(py);
        for (name, kind) in fields {
            if kind.is_some() {
                expected.set_item(*name, py.get_type::<PyAny>())?;
            } else {
                expected.set_item(*name, py.get_type::<PyBool>())?;
            }
        }
        return Err(
            crate::util::exceptions::DESERIALIZATION_DICT_STRUCTURE_ERROR
                .err(py, (cls, expected, data)),
        );
    }
    let decoded = PyDict::new(py);
    for (name, kind) in fields {
        let value = data.get_item(*name)?;
        match kind {
            None => decoded.set_item(*name, value)?,
            Some(kind) => match decode_values(*kind, &value) {
                Ok(values) => decoded.set_item(*name, values)?,
                Err(error)
                    if error.is_instance_of::<pyo3::exceptions::PyValueError>(py)
                        || error.is_instance_of::<pyo3::exceptions::PyTypeError>(py) =>
                {
                    let wrapped = value_error(cls.as_any(), name, "a decodable value", &value);
                    wrapped.set_cause(py, Some(error));
                    return Err(wrapped);
                }
                Err(error) => return Err(error),
            },
        }
    }
    construct_from_decoded_fields(cls, &decoded)
}

/// Return an instance of `cls` of the decoded fields `fields`, as a
/// dataclass's `cls(**fields)` does.
fn construct_domain<'py>(
    cls: &Bound<'py, PyType>,
    fields: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let keywords = PyDict::new(cls.py());
    keywords.update(fields.cast::<pyo3::types::PyMapping>()?)?;
    cls.call((), Some(&keywords))
}

/// Return the payload of the fields `fields` of a domain.
fn serialize_fields<'py>(
    py: Python<'py>,
    fields: &[(&str, Bound<'py, PyAny>)],
    finite: Option<&Bound<'py, PyTuple>>,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    for (name, value) in fields {
        match finite {
            Some(values) if value.is(values) => {
                payload.set_item(*name, serialize_values(values)?)?;
            }
            _ => payload.set_item(*name, value)?,
        }
    }
    Ok(payload)
}

/// Return the truthiness of the flag `value`, or `default` when omitted.
fn read_flag(value: OptionalArgument<'_>, default: bool) -> PyResult<bool> {
    match value {
        OptionalArgument::Omitted => Ok(default),
        OptionalArgument::Given(value) => value.is_truthy(),
    }
}

/// Return the Python `bool` of `value`.
fn boolean(py: Python<'_>, value: bool) -> Bound<'_, PyAny> {
    PyBool::new(py, value).to_owned().into_any()
}

/// Define a domain pyclass: `$fields` returns the names and values of its
/// dataclass fields, `$decoding` their payload shapes, and `$extra` holds its
/// constructor and getters.
macro_rules! domain_class {
    ($class:ident, $name:literal, $fields:expr, $decoding:expr, { $($extra:tt)* }) => {
        #[doc = concat!("`", $name, "`, backed by the core's [`ParamDomain`].")]
        #[pyclass(subclass, frozen, module = "fhy_core._rs", name = $name)]
        pub(crate) struct $class {
            state: DomainState,
        }

        impl $class {
            /// Return the core domain.
            pub(crate) fn core(&self) -> ParamDomain {
                self.state.core.clone()
            }

            /// Return the names and Python values of the dataclass fields.
            fn fields<'py>(&self, py: Python<'py>) -> Vec<(&'static str, Bound<'py, PyAny>)> {
                let fields: fn(&Self, Python<'py>) -> Vec<(&'static str, Bound<'py, PyAny>)> = $fields;
                fields(self, py)
            }
        }

        #[pymethods]
        impl $class {
            $($extra)*

            /// Visit the member objects, for the cycle collector.
            fn __traverse__(
                &self,
                visit: ::pyo3::pyclass::PyVisit<'_>,
            ) -> Result<(), ::pyo3::pyclass::PyTraverseError> {
                visit.call(self.state.values.as_ref())?;
                self.state.slots.traverse(&visit)
            }

            /// The sort the solver reasons about the values in, or `None`.
            #[getter]
            fn symbol_type<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                self.state.symbol_type(py)
            }

            /// Return whether `value` lies in the domain's value set.
            fn is_value_admissible(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
                self.state.is_value_admissible(value)
            }

            /// Return the canonical form of `value`.
            fn normalize_value<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                self.state.normalize_value(value)
            }

            /// Raise if `constraint` is not permitted for this domain.
            fn validate_constraint(
                &self,
                constraint: &Bound<'_, PyAny>,
                variable: &Bound<'_, PyAny>,
            ) -> PyResult<()> {
                self.state.validate_constraint(constraint, variable)
            }

            /// Return the constraints this domain imposes on `variable`.
            fn get_implied_constraints<'py>(
                &self,
                variable: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyTuple>> {
                self.state.implied_constraints(variable)
            }

            /// Return what interval arithmetic reads from this domain, or
            /// `None`.
            fn get_interval_profile<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
                self.state.interval_profile(py)
            }

            /// Return whether this domain's value set is a subset of
            /// `other`'s.
            fn is_value_set_subset(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
                self.state.is_value_set_subset(other)
            }

            /// Decide whether this domain's constrained set is a subset of
            /// `other`'s.
            fn compute_feasibility_subset<'py>(
                &self,
                own_constraints: &Bound<'py, PyAny>,
                own_variable: &Bound<'py, PyAny>,
                other: &Bound<'py, PyAny>,
                other_constraints: &Bound<'py, PyAny>,
                other_variable: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                self.state.feasibility_subset(
                    own_constraints,
                    own_variable,
                    other,
                    other_constraints,
                    other_variable,
                )
            }

            /// Decide whether some admissible value satisfies every
            /// constraint.
            fn has_feasible_value<'py>(
                &self,
                constraints: &Bound<'py, PyAny>,
                variable: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                self.state.has_feasible_value(constraints, variable)
            }

            /// Return the domain and constraints of the union, or `None`
            /// for a kind that represents no union.
            fn compute_union<'py>(
                slf: &Bound<'py, Self>,
                own_constraints: &Bound<'py, PyAny>,
                own_variable: &Bound<'py, PyAny>,
                other: &Bound<'py, PyAny>,
                other_constraints: &Bound<'py, PyAny>,
                other_variable: &Bound<'py, PyAny>,
                variable: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                slf.get().state.set_operation(
                    slf.as_any(),
                    true,
                    own_constraints,
                    own_variable,
                    other,
                    other_constraints,
                    other_variable,
                    variable,
                )
            }

            /// Return the domain and constraints of the intersection.
            fn compute_intersection<'py>(
                slf: &Bound<'py, Self>,
                own_constraints: &Bound<'py, PyAny>,
                own_variable: &Bound<'py, PyAny>,
                other: &Bound<'py, PyAny>,
                other_constraints: &Bound<'py, PyAny>,
                other_variable: &Bound<'py, PyAny>,
                variable: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                slf.get().state.set_operation(
                    slf.as_any(),
                    false,
                    own_constraints,
                    own_variable,
                    other,
                    other_constraints,
                    other_variable,
                    variable,
                )
            }

            /// Return whether `other` is a structurally identical domain.
            fn is_structurally_equivalent(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
                self.state.is_structurally_equivalent(other)
            }

            /// Return the `str` rendering of the value set.
            fn render_set_string(&self, py: Python<'_>) -> PyResult<String> {
                self.state.render_set(py, false)
            }

            /// Return the `repr` fragment of the value set, or `""`.
            fn render_set_repr(&self, py: Python<'_>) -> PyResult<String> {
                self.state.render_set(py, true)
            }

            fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
                let py = slf.py();
                let fields = slf.get().fields(py);
                let fields: Vec<(&str, &Bound<'_, PyAny>)> =
                    fields.iter().map(|(name, value)| (*name, value)).collect();
                crate::util::dataclass::format_dataclass_repr(&slf.get_type(), &fields)
            }

            /// Always true: domains are immutable.
            #[getter]
            const fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
                true
            }

            /// Do nothing: domains are always frozen.
            const fn freeze(_slf: &Bound<'_, Self>) {}

            /// Do nothing: domains are always frozen, and mutating one raises.
            const fn assert_frozen(_slf: &Bound<'_, Self>) {}

            fn __setattr__(slf: &Bound<'_, Self>, name: &str, _value: &Bound<'_, PyAny>) -> PyResult<()> {
                refuse_attribute_assignment(slf, name)
            }

            fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
                refuse_attribute_deletion(slf, name)
            }

            /// Pickle as a constructor call of the class with the fields.
            fn __reduce__<'py>(
                slf: &Bound<'py, Self>,
            ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
                let py = slf.py();
                let values: Vec<Bound<'py, PyAny>> =
                    slf.get().fields(py).into_iter().map(|(_, value)| value).collect();
                Ok((slf.get_type(), PyTuple::new(py, values)?))
            }

            /// Return the data payload of the fields.
            fn serialize_data_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
                let values = self.state.values(py);
                serialize_fields(py, &self.fields(py), self.state.values.as_ref().map(|_| &values))
            }

            /// Return the domain of a data payload.
            ///
            /// Raises the serialization framework's errors for a malformed
            /// payload.
            #[classmethod]
            fn deserialize_data_from_dict<'py>(
                cls: &Bound<'py, PyType>,
                data: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                deserialize_domain(cls, data, $decoding)
            }

            /// Build the domain of the decoded fields.
            #[classmethod]
            fn construct_from_fields<'py>(
                cls: &Bound<'py, PyType>,
                fields: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                construct_domain(cls, fields)
            }
        }
    };
}

domain_class!(
    PyIntegerDomain,
    "IntegerDomain",
    |this, py| {
        let ParamDomain::Integer(domain) = &this.state.core else {
            return Vec::new();
        };
        vec![
            ("non_negative", boolean(py, domain.is_non_negative())),
            ("zero_included", boolean(py, domain.is_zero_included())),
        ]
    },
    &[("non_negative", None), ("zero_included", None)],
    {
        /// Create the integers, restricted to the natural numbers when
        /// `non_negative`, with zero when `zero_included`.
        #[new]
        #[pyo3(signature = (non_negative = OptionalArgument::Omitted, zero_included = OptionalArgument::Omitted))]
        fn new(
            non_negative: OptionalArgument<'_>,
            zero_included: OptionalArgument<'_>,
        ) -> PyResult<Self> {
            let domain = IntegerDomain::new(
                Sign::non_negative_if(read_flag(non_negative, false)?),
                ZeroInclusion::included_if(read_flag(zero_included, true)?),
            );
            Ok(Self {
                state: DomainState::numeric(domain),
            })
        }

        /// Whether the domain admits only non-negative values.
        #[getter]
        fn non_negative(&self) -> bool {
            matches!(&self.state.core, ParamDomain::Integer(domain) if domain.is_non_negative())
        }

        /// Whether the domain admits zero, given it is non-negative.
        #[getter]
        fn zero_included(&self) -> bool {
            matches!(&self.state.core, ParamDomain::Integer(domain) if domain.is_zero_included())
        }
    }
);

domain_class!(
    PyIntervalIntegerDomain,
    "IntervalIntegerDomain",
    |this, py| {
        let ParamDomain::IntervalInteger(domain) = &this.state.core else {
            return Vec::new();
        };
        vec![
            (
                "prefer_inclusive",
                boolean(py, domain.is_inclusive_preferred()),
            ),
            ("non_negative", boolean(py, domain.is_non_negative())),
            ("zero_included", boolean(py, domain.is_zero_included())),
        ]
    },
    &[
        ("prefer_inclusive", None),
        ("non_negative", None),
        ("zero_included", None)
    ],
    {
        /// Create the interval integers, rendering derived bounds
        /// inclusively when `prefer_inclusive`, restricted as
        /// `IntegerDomain` is.
        #[new]
        #[pyo3(signature = (prefer_inclusive = OptionalArgument::Omitted, non_negative = OptionalArgument::Omitted, zero_included = OptionalArgument::Omitted))]
        fn new(
            prefer_inclusive: OptionalArgument<'_>,
            non_negative: OptionalArgument<'_>,
            zero_included: OptionalArgument<'_>,
        ) -> PyResult<Self> {
            let domain = IntervalIntegerDomain::new(
                Inclusivity::inclusive_if(read_flag(prefer_inclusive, true)?),
                Sign::non_negative_if(read_flag(non_negative, false)?),
                ZeroInclusion::included_if(read_flag(zero_included, true)?),
            );
            Ok(Self {
                state: DomainState::numeric(domain),
            })
        }

        /// Whether derived bounds render inclusively.
        #[getter]
        fn prefer_inclusive(&self) -> bool {
            matches!(&self.state.core, ParamDomain::IntervalInteger(domain) if domain.is_inclusive_preferred())
        }

        /// Whether the domain admits only non-negative values.
        #[getter]
        fn non_negative(&self) -> bool {
            matches!(&self.state.core, ParamDomain::IntervalInteger(domain) if domain.is_non_negative())
        }

        /// Whether the domain admits zero, given it is non-negative.
        #[getter]
        fn zero_included(&self) -> bool {
            matches!(&self.state.core, ParamDomain::IntervalInteger(domain) if domain.is_zero_included())
        }
    }
);

domain_class!(PyRealDomain, "RealDomain", |_this, _py| Vec::new(), &[], {
    /// Create the reals.
    #[new]
    fn new() -> Self {
        Self {
            state: DomainState::numeric(RealDomain),
        }
    }
});

/// Return the finite domain of `kind` of the Python `values`.
fn build_finite(
    py: Python<'_>,
    kind: DomainKind,
    values: &Bound<'_, PyAny>,
) -> PyResult<DomainState> {
    // The opaque values' adapters are this domain's to traverse.
    let (state, slots) = crate::util::gc::collect_slots(|| build_finite_state(py, kind, values));
    let mut state = state?;
    state.slots = slots;
    Ok(state)
}

/// Return the finite domain state of `kind` of the Python `values`.
fn build_finite_state(
    py: Python<'_>,
    kind: DomainKind,
    values: &Bound<'_, PyAny>,
) -> PyResult<DomainState> {
    let values: Vec<Value> = read_finite_values(kind, values)?;
    let (built, raised) = capture_pending_errors(|| -> Result<ParamDomain, DomainError> {
        Ok(match kind {
            DomainKind::Ordinal => ParamDomain::from(OrdinalDomain::new(values)?),
            DomainKind::Categorical => ParamDomain::from(CategoricalDomain::new(values)?),
            _ => ParamDomain::from(PermutationDomain::new(values)?),
        })
    });
    let domain = match built {
        Ok(domain) => {
            if let Some(raised) = raised {
                return Err(raised);
            }
            domain
        }
        Err(error) => return Err(ordinal_error_to_py(py, error, raised)),
    };
    let members: Vec<Member> = match &domain {
        ParamDomain::Ordinal(domain) => domain.values().to_vec(),
        ParamDomain::Categorical(domain) => domain.values().to_vec(),
        ParamDomain::Permutation(domain) => domain.values().to_vec(),
        _ => Vec::new(),
    };
    DomainState::finite(py, domain, &members)
}

domain_class!(
    PyOrdinalDomain,
    "OrdinalDomain",
    |this, py| vec![("sorted_values", this.state.values(py).into_any())],
    &[("sorted_values", Some(DomainKind::Ordinal))],
    {
        /// Create the ordinal domain of `sorted_values`, in any order.
        ///
        /// Raises `ParamError` for no value, a NaN or equal values, and
        /// `TypeError` for a value of the wrong kind or values that do not
        /// order.
        #[new]
        #[pyo3(signature = (sorted_values))]
        fn new(sorted_values: &Bound<'_, PyAny>) -> PyResult<Self> {
            Ok(Self {
                state: build_finite(sorted_values.py(), DomainKind::Ordinal, sorted_values)?,
            })
        }

        /// The values, ascending.
        #[getter]
        fn sorted_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
            self.state.values(py)
        }
    }
);

domain_class!(
    PyCategoricalDomain,
    "CategoricalDomain",
    |this, py| vec![("categories", this.state.values(py).into_any())],
    &[("categories", Some(DomainKind::Categorical))],
    {
        /// Create the categorical domain of `categories`.
        ///
        /// Raises `ParamError` for no value or equal values, and
        /// `TypeError` for a value of the wrong kind.
        #[new]
        #[pyo3(signature = (categories))]
        fn new(categories: &Bound<'_, PyAny>) -> PyResult<Self> {
            Ok(Self {
                state: build_finite(categories.py(), DomainKind::Categorical, categories)?,
            })
        }

        /// The categories, in the members' canonical order.
        #[getter]
        fn categories<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
            self.state.values(py)
        }
    }
);

domain_class!(
    PyPermutationDomain,
    "PermutationDomain",
    |this, py| vec![("ordered_members", this.state.values(py).into_any())],
    &[("ordered_members", Some(DomainKind::Permutation))],
    {
        /// Create the permutations of `ordered_members`, in the order given.
        ///
        /// Raises `ParamError` for no value, a NaN or equal values, and
        /// `TypeError` for a value of the wrong kind.
        #[new]
        #[pyo3(signature = (ordered_members))]
        fn new(ordered_members: &Bound<'_, PyAny>) -> PyResult<Self> {
            Ok(Self {
                state: build_finite(
                    ordered_members.py(),
                    DomainKind::Permutation,
                    ordered_members,
                )?,
            })
        }

        /// The members, in the order given.
        #[getter]
        fn ordered_members<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
            self.state.values(py)
        }
    }
);
