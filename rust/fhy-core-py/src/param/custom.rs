//! A Python-defined `ParamDomain` as a core [`CustomDomain`] (P3;
//! D-S16-5): the core calls the object's own methods as its procedures need
//! them, with the Python objects of the constraints, values and domains it
//! passes.
//!
//! An exception a hook raises propagates as the same object; the
//! equivalence hook, which the core cannot fail, answers `false` and keeps
//! its exception in S13's pending-error slot.

use std::any::Any;
use std::fmt;

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyString, PyTuple};

use fhy_core::constraint::{Constraint, CustomError, Outcome, Value};
use fhy_core::expression::SymbolType;
use fhy_core::identifier::Identifier;
use fhy_core::param::{CustomDomain, IntervalProfile, ParamDomain, Side};

use crate::constraint::{
    read_constraint, read_outcome, record_pending_error, type_name, value_to_python,
};

use super::objects::{
    constraint_to_python, constraints_to_python, domain_to_python, identifier_object, read_domain,
    read_profile,
};

/// A Python-defined domain, driven through its methods.
pub(crate) struct PyCustomDomain {
    object: Py<PyAny>,
}

impl PyCustomDomain {
    /// Return the adapter of `object`.
    pub(super) fn new(object: &Bound<'_, PyAny>) -> Self {
        Self {
            object: object.clone().unbind(),
        }
    }

    /// Return the Python object.
    pub(super) fn object(&self) -> &Py<PyAny> {
        &self.object
    }

    /// Call `hook` under the interpreter, boxing its exception.
    fn call<T>(
        &self,
        hook: impl FnOnce(Python<'_>, &Bound<'_, PyAny>) -> PyResult<T>,
    ) -> Result<T, CustomError> {
        Python::attach(|py| hook(py, self.object.bind(py)))
            .map_err(|error| Box::new(error) as CustomError)
    }
}

impl fmt::Debug for PyCustomDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PyCustomDomain").finish_non_exhaustive()
    }
}

/// Return the Python arguments `(constraints, variable)` of `side`.
fn side_arguments<'py>(
    py: Python<'py>,
    side: Side<'_>,
) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyAny>)> {
    Ok((
        constraints_to_python(py, side.constraints())?,
        identifier_object(py, side.variable())?,
    ))
}

/// Return the core `(domain, constraints)` of a hook's Python result.
fn read_domain_and_constraints(
    object: &Bound<'_, PyAny>,
    hook: &str,
    result: &Bound<'_, PyAny>,
) -> PyResult<(ParamDomain, Vec<Constraint>)> {
    let pair = result
        .cast::<PyTuple>()
        .ok()
        .filter(|pair| pair.len() == 2)
        .ok_or_else(|| {
            PyTypeError::new_err(format!(
                "{}.{hook} must return a (domain, constraints) pair, got {}.",
                type_name(object),
                type_name(result)
            ))
        })?;
    let domain = read_domain(&pair.get_item(0)?);
    let constraints = pair
        .get_item(1)?
        .try_iter()?
        .map(|constraint| read_constraint(&constraint?))
        .collect::<PyResult<Vec<_>>>()?;
    Ok((domain, constraints))
}

/// Return the symbol type of the Python `SymbolType` member `value`, or
/// `None` for `None`.
fn read_symbol_type(
    object: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<Option<SymbolType>> {
    if value.is_none() {
        return Ok(None);
    }
    value
        .cast::<PyString>()
        .ok()
        .and_then(|text| text.to_str().ok()?.parse::<SymbolType>().ok())
        .map(Some)
        .ok_or_else(|| {
            PyTypeError::new_err(format!(
                "{}.symbol_type must be a SymbolType or None, got {}.",
                type_name(object),
                type_name(value)
            ))
        })
}

impl CustomDomain for PyCustomDomain {
    fn symbol_type(&self) -> Result<Option<SymbolType>, CustomError> {
        self.call(|py, object| {
            let value = object.getattr(intern!(py, "symbol_type"))?;
            read_symbol_type(object, &value)
        })
    }

    fn is_value_admissible(&self, value: &Value) -> Result<bool, CustomError> {
        self.call(|py, object| {
            object
                .call_method1(
                    intern!(py, "is_value_admissible"),
                    (value_to_python(py, value)?,),
                )?
                .is_truthy()
        })
    }

    fn validate_constraint(
        &self,
        constraint: &Constraint,
        variable: &Identifier,
    ) -> Result<(), CustomError> {
        self.call(|py, object| {
            object.call_method1(
                intern!(py, "validate_constraint"),
                (
                    constraint_to_python(py, constraint)?,
                    identifier_object(py, variable)?,
                ),
            )?;
            Ok(())
        })
    }

    fn implied_constraints(&self, variable: &Identifier) -> Result<Vec<Constraint>, CustomError> {
        self.call(|py, object| {
            object
                .call_method1(
                    intern!(py, "get_implied_constraints"),
                    (identifier_object(py, variable)?,),
                )?
                .try_iter()?
                .map(|constraint| read_constraint(&constraint?))
                .collect()
        })
    }

    fn interval_profile(&self) -> Result<Option<IntervalProfile>, CustomError> {
        self.call(|py, object| {
            let profile = object.call_method0(intern!(py, "get_interval_profile"))?;
            if profile.is_none() {
                Ok(None)
            } else {
                read_profile(&profile).map(Some)
            }
        })
    }

    fn is_value_set_subset(&self, other: &ParamDomain) -> Result<bool, CustomError> {
        self.call(|py, object| {
            object
                .call_method1(
                    intern!(py, "is_value_set_subset"),
                    (domain_to_python(py, other)?,),
                )?
                .is_truthy()
        })
    }

    fn feasibility_subset(
        &self,
        own: Side<'_>,
        other_domain: &ParamDomain,
        other: Side<'_>,
    ) -> Result<Outcome, CustomError> {
        self.call(|py, object| {
            let (own_constraints, own_variable) = side_arguments(py, own)?;
            let (other_constraints, other_variable) = side_arguments(py, other)?;
            let outcome = object.call_method1(
                intern!(py, "compute_feasibility_subset"),
                (
                    own_constraints,
                    own_variable,
                    domain_to_python(py, other_domain)?,
                    other_constraints,
                    other_variable,
                ),
            )?;
            read_outcome(object, &outcome)
        })
    }

    fn has_feasible_value(&self, side: Side<'_>) -> Result<Outcome, CustomError> {
        self.call(|py, object| {
            let (constraints, variable) = side_arguments(py, side)?;
            let outcome =
                object.call_method1(intern!(py, "has_feasible_value"), (constraints, variable))?;
            read_outcome(object, &outcome)
        })
    }

    fn union(
        &self,
        own: Side<'_>,
        other_domain: &ParamDomain,
        other: Side<'_>,
        variable: &Identifier,
    ) -> Result<Option<(ParamDomain, Vec<Constraint>)>, CustomError> {
        self.call(|py, object| {
            let (own_constraints, own_variable) = side_arguments(py, own)?;
            let (other_constraints, other_variable) = side_arguments(py, other)?;
            let result = object.call_method1(
                intern!(py, "compute_union"),
                (
                    own_constraints,
                    own_variable,
                    domain_to_python(py, other_domain)?,
                    other_constraints,
                    other_variable,
                    identifier_object(py, variable)?,
                ),
            )?;
            if result.is_none() {
                return Ok(None);
            }
            read_domain_and_constraints(object, "compute_union", &result).map(Some)
        })
    }

    fn intersection(
        &self,
        own: Side<'_>,
        other_domain: &ParamDomain,
        other: Side<'_>,
        variable: &Identifier,
    ) -> Result<(ParamDomain, Vec<Constraint>), CustomError> {
        self.call(|py, object| {
            let (own_constraints, own_variable) = side_arguments(py, own)?;
            let (other_constraints, other_variable) = side_arguments(py, other)?;
            let result = object.call_method1(
                intern!(py, "compute_intersection"),
                (
                    own_constraints,
                    own_variable,
                    domain_to_python(py, other_domain)?,
                    other_constraints,
                    other_variable,
                    identifier_object(py, variable)?,
                ),
            )?;
            read_domain_and_constraints(object, "compute_intersection", &result)
        })
    }

    fn is_structurally_equivalent(&self, other: &ParamDomain) -> bool {
        Python::attach(|py| {
            domain_to_python(py, other)
                .and_then(|other| {
                    self.object
                        .bind(py)
                        .call_method1(intern!(py, "is_structurally_equivalent"), (other,))
                })
                .and_then(|answer| answer.is_truthy())
                .unwrap_or_else(|error| {
                    record_pending_error(error);
                    false
                })
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
