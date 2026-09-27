//! Python objects of core constraints, domains and profiles, and core
//! values of Python ones.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyTuple, PyType};

use fhy_core::constraint::{Binding, Bindings, Constraint, Polarity};
use fhy_core::identifier::Identifier;
use fhy_core::param::{IntervalProfile, ParamDomain};

use crate::constraint::{
    PyCustomConstraint, member_to_python, read_constraint, repr_text, value_to_python,
};
use crate::expression::materialize_expression;
use crate::identifier::identifier_to_python;

use super::custom::PyCustomDomain;
use super::domains::{
    PyCategoricalDomain, PyIntegerDomain, PyIntervalIntegerDomain, PyOrdinalDomain,
    PyPermutationDomain, PyRealDomain,
};

/// The module of the public constraint classes.
const CONSTRAINTS: &str = "fhy_core.symbolic.constraint.core";
/// The module of the public domain classes.
const DOMAINS: &str = "fhy_core.symbolic.param.domains";

/// Return the public class `name` of `module`, imported once into `cell`.
fn public_class<'py>(
    py: Python<'py>,
    cell: &'static PyOnceLock<Py<PyType>>,
    module: &str,
    name: &str,
) -> PyResult<&'py Bound<'py, PyType>> {
    cell.import(py, module, name)
}

/// Return the name of the Python class of `constraint`.
pub(super) fn constraint_class_name(py: Python<'_>, constraint: &Constraint) -> String {
    match constraint {
        Constraint::Equation(_) => "EquationConstraint".to_owned(),
        Constraint::Set(set) => match set.polarity() {
            Polarity::NotIn => "NotInSetConstraint".to_owned(),
            _ => "InSetConstraint".to_owned(),
        },
        Constraint::Custom(custom) => custom
            .get()
            .as_any()
            .downcast_ref::<PyCustomConstraint>()
            .map_or_else(
                || "Constraint".to_owned(),
                |custom| crate::constraint::type_name(custom.object().bind(py)),
            ),
        _ => "Constraint".to_owned(),
    }
}

/// Return a Python object of `constraint`: a Python-defined constraint's
/// own object, and a new instance of the public class of a built-in kind.
///
/// # Errors
///
/// Raises what building the object raises.
pub(crate) fn constraint_to_python<'py>(
    py: Python<'py>,
    constraint: &Constraint,
) -> PyResult<Bound<'py, PyAny>> {
    static EQUATION: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    static IN_SET: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    static NOT_IN_SET: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    match constraint {
        Constraint::Equation(equation) => {
            public_class(py, &EQUATION, CONSTRAINTS, "EquationConstraint")?
                .call1((materialize_expression(py, equation.expression())?,))
        }
        Constraint::Set(set) => {
            let members = set
                .members()
                .iter()
                .map(|member| member_to_python(py, member))
                .collect::<PyResult<Vec<_>>>()?;
            let class = match set.polarity() {
                Polarity::NotIn => {
                    public_class(py, &NOT_IN_SET, CONSTRAINTS, "NotInSetConstraint")?
                }
                _ => public_class(py, &IN_SET, CONSTRAINTS, "InSetConstraint")?,
            };
            class.call1((
                identifier_to_python(py, set.variable())?,
                PyTuple::new(py, members)?,
            ))
        }
        Constraint::Custom(custom) => custom
            .get()
            .as_any()
            .downcast_ref::<PyCustomConstraint>()
            .map(|custom| custom.object().bind(py).clone())
            .ok_or_else(|| {
                pyo3::exceptions::PyTypeError::new_err("a custom constraint has no Python object")
            }),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "a constraint of an unknown kind",
        )),
    }
}

/// Return the tuple of the Python objects of `constraints`.
pub(super) fn constraints_to_python<'py>(
    py: Python<'py>,
    constraints: &[Constraint],
) -> PyResult<Bound<'py, PyTuple>> {
    let objects = constraints
        .iter()
        .map(|constraint| constraint_to_python(py, constraint))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, objects)
}

/// Return the `repr` of `constraint`, as its Python object renders it.
pub(super) fn constraint_repr(py: Python<'_>, constraint: &Constraint) -> String {
    constraint_to_python(py, constraint)
        .map_or_else(|_| "?".to_owned(), |object| repr_text(&object))
}

/// Return the core constraints of the Python constraints `constraints`.
///
/// # Errors
///
/// Raises what reading a constraint raises.
pub(super) fn read_constraints(constraints: &Bound<'_, PyAny>) -> PyResult<Vec<Constraint>> {
    constraints
        .try_iter()?
        .map(|constraint| read_constraint(&constraint?))
        .collect()
}

/// Return the Python dict of the core `bindings`.
///
/// # Errors
///
/// Raises what building an object raises.
pub(super) fn bindings_to_python<'py>(
    py: Python<'py>,
    bindings: &Bindings,
) -> PyResult<Bound<'py, PyDict>> {
    let mapping = PyDict::new(py);
    for (identifier, binding) in bindings.iter() {
        let value = match binding {
            Binding::Expression(expression) => materialize_expression(py, expression)?,
            Binding::Value(value) => value_to_python(py, value)?,
        };
        mapping.set_item(identifier_to_python(py, identifier)?, value)?;
    }
    Ok(mapping)
}

/// Return the identifiers of `bindings` as the Python text names them.
pub(super) fn bound_identifiers_text(bindings: &Bindings) -> String {
    let names: Vec<String> = bindings
        .iter()
        .map(|(identifier, _)| format!("{identifier:?}"))
        .collect();
    if names.is_empty() {
        "no identifiers".to_owned()
    } else {
        names.join(", ")
    }
}

/// Return the core domain of the Python domain `domain`: a built-in
/// kind's own, and a custom domain for any other object.
pub(super) fn read_domain(domain: &Bound<'_, PyAny>) -> ParamDomain {
    if let Ok(domain) = domain.cast::<PyIntegerDomain>() {
        return domain.get().core();
    }
    if let Ok(domain) = domain.cast::<PyIntervalIntegerDomain>() {
        return domain.get().core();
    }
    if let Ok(domain) = domain.cast::<PyRealDomain>() {
        return domain.get().core();
    }
    if let Ok(domain) = domain.cast::<PyOrdinalDomain>() {
        return domain.get().core();
    }
    if let Ok(domain) = domain.cast::<PyCategoricalDomain>() {
        return domain.get().core();
    }
    if let Ok(domain) = domain.cast::<PyPermutationDomain>() {
        return domain.get().core();
    }
    ParamDomain::Custom(fhy_core::foreign::Part::new(PyCustomDomain::new(domain)))
}

/// Return the core domain of the Python `ParamDomain` `object`, as
/// [`read_domain`] reads it.
///
/// # Errors
///
/// Raises `TypeError` for an object that is not a `ParamDomain`.
pub(super) fn read_domain_object(object: &Bound<'_, PyAny>) -> PyResult<ParamDomain> {
    static PARAM_DOMAIN: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    let py = object.py();
    if !object.is_instance(public_class(py, &PARAM_DOMAIN, DOMAINS, "ParamDomain")?)? {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected a ParamDomain, got {}",
            crate::constraint::type_name(object)
        )));
    }
    Ok(read_domain(object))
}

/// Return a Python object of `domain`: a custom domain's own object, and a
/// new instance of the public class of a built-in kind.
///
/// # Errors
///
/// Raises what building the object raises.
pub(crate) fn domain_to_python<'py>(
    py: Python<'py>,
    domain: &ParamDomain,
) -> PyResult<Bound<'py, PyAny>> {
    static INTEGER: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    static INTERVAL: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    static REAL: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    static ORDINAL: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    static CATEGORICAL: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    static PERMUTATION: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    let members = |values: &[fhy_core::constraint::Member]| -> PyResult<Bound<'py, PyTuple>> {
        let objects = values
            .iter()
            .map(|member| member_to_python(py, member))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, objects)
    };
    match domain {
        ParamDomain::Integer(domain) => public_class(py, &INTEGER, DOMAINS, "IntegerDomain")?
            .call1((domain.is_non_negative(), domain.is_zero_included())),
        ParamDomain::IntervalInteger(domain) => {
            public_class(py, &INTERVAL, DOMAINS, "IntervalIntegerDomain")?.call1((
                domain.is_inclusive_preferred(),
                domain.is_non_negative(),
                domain.is_zero_included(),
            ))
        }
        ParamDomain::Real(_) => public_class(py, &REAL, DOMAINS, "RealDomain")?.call0(),
        ParamDomain::Ordinal(domain) => public_class(py, &ORDINAL, DOMAINS, "OrdinalDomain")?
            .call1((members(domain.values())?,)),
        ParamDomain::Categorical(domain) => {
            public_class(py, &CATEGORICAL, DOMAINS, "CategoricalDomain")?
                .call1((members(domain.values())?,))
        }
        ParamDomain::Permutation(domain) => {
            public_class(py, &PERMUTATION, DOMAINS, "PermutationDomain")?
                .call1((members(domain.values())?,))
        }
        ParamDomain::Custom(custom) => custom
            .get()
            .as_any()
            .downcast_ref::<PyCustomDomain>()
            .map(|custom| custom.object().bind(py).clone())
            .ok_or_else(|| {
                pyo3::exceptions::PyTypeError::new_err("a custom domain has no Python object")
            }),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "a domain of an unknown kind",
        )),
    }
}

/// Return the Python `IntervalProfile` of `profile`.
///
/// # Errors
///
/// Raises what building the object raises.
pub(super) fn profile_to_python(
    py: Python<'_>,
    profile: IntervalProfile,
) -> PyResult<Bound<'_, PyAny>> {
    static PROFILE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    let keywords = PyDict::new(py);
    keywords.set_item(
        intern!(py, "admits_only_bounds"),
        profile.admits_only_bounds,
    )?;
    keywords.set_item(intern!(py, "non_negative"), profile.non_negative)?;
    keywords.set_item(intern!(py, "zero_included"), profile.zero_included)?;
    keywords.set_item(intern!(py, "prefer_inclusive"), profile.prefer_inclusive)?;
    public_class(py, &PROFILE, DOMAINS, "IntervalProfile")?.call((), Some(&keywords))
}

/// Return the core profile of the Python `IntervalProfile` `profile`, read
/// by its attributes' truthiness.
///
/// # Errors
///
/// Raises what reading an attribute raises.
pub(super) fn read_profile(profile: &Bound<'_, PyAny>) -> PyResult<IntervalProfile> {
    let py = profile.py();
    let read = |name| -> PyResult<bool> { profile.getattr(name)?.is_truthy() };
    Ok(IntervalProfile {
        admits_only_bounds: read(intern!(py, "admits_only_bounds"))?,
        non_negative: read(intern!(py, "non_negative"))?,
        zero_included: read(intern!(py, "zero_included"))?,
        prefer_inclusive: read(intern!(py, "prefer_inclusive"))?,
    })
}

/// Return the Python `Identifier` of `identifier`.
///
/// # Errors
///
/// Raises what building the object raises.
pub(super) fn identifier_object<'py>(
    py: Python<'py>,
    identifier: &Identifier,
) -> PyResult<Bound<'py, PyAny>> {
    identifier_to_python(py, identifier)
}
