//! A Python-defined `ParamDomain` as a core [`CustomDomain`]: the core calls
//! the object's own methods as its procedures need them, with the Python
//! objects of the constraints, values and domains it passes.
//!
//! An exception a hook raises propagates as the same object; the
//! equivalence hook, which the core cannot fail, answers `false` and keeps
//! its exception in the constraint module's pending-error slot.

use std::borrow::Cow;
use std::fmt;

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyString, PyTuple};

use fhy_core::constraint::{Constraint, Outcome, Value};
use fhy_core::expression::SymbolType;
use fhy_core::foreign::{BoxError, ForeignPart};
use fhy_core::identifier::Identifier;
use fhy_core::param::{CustomDomain, IntervalProfile, ParamContext, ParamDomain, Side};

use crate::constraint::{
    has_pending_error, read_constraint, read_outcome, record_pending_error, type_name,
    value_to_python,
};

use crate::gc::Slot;

use super::objects::{
    constraint_to_python, constraints_to_python, domain_to_python, identifier_object, read_domain,
    read_profile,
};

/// A Python-defined domain, driven through its methods.
///
/// The object is kept in a [`Slot`], which the object whose construction
/// made the adapter owns and traverses.
pub(crate) struct PyCustomDomain {
    object: Slot,
}

impl PyCustomDomain {
    /// Return the adapter of `object`.
    pub(super) fn new(object: &Bound<'_, PyAny>) -> Self {
        Self {
            object: Slot::new(object.clone().unbind()),
        }
    }

    /// Return the Python object.
    pub(super) fn object<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.object.get(py)
    }

    /// Call `hook` under the interpreter, boxing its exception.
    fn call<T>(
        &self,
        hook: impl FnOnce(Python<'_>, &Bound<'_, PyAny>) -> PyResult<T>,
    ) -> Result<T, BoxError> {
        Python::attach(|py| hook(py, &self.object.get(py)))
            .map_err(|error| Box::new(error) as BoxError)
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

impl ForeignPart for PyCustomDomain {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Owned(Python::attach(|py| type_name(&self.object.get(py))))
    }

    fn to_foreign(&self) -> Result<fhy_core::foreign::Foreign, fhy_core::foreign::ForeignError> {
        Python::attach(|py| crate::wire::foreign_of(&self.object.object(py), true))
    }
}

impl CustomDomain for PyCustomDomain {
    fn symbol_type(&self) -> Result<Option<SymbolType>, BoxError> {
        self.call(|py, object| {
            let value = object.getattr(intern!(py, "symbol_type"))?;
            read_symbol_type(object, &value)
        })
    }

    fn is_value_admissible(&self, value: &Value) -> Result<bool, BoxError> {
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
    ) -> Result<(), BoxError> {
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

    fn implied_constraints(&self, variable: &Identifier) -> Result<Vec<Constraint>, BoxError> {
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

    fn interval_profile(&self) -> Result<Option<IntervalProfile>, BoxError> {
        self.call(|py, object| {
            let profile = object.call_method0(intern!(py, "get_interval_profile"))?;
            if profile.is_none() {
                Ok(None)
            } else {
                read_profile(&profile).map(Some)
            }
        })
    }

    fn is_value_set_subset(
        &self,
        other: &ParamDomain,
        _context: &ParamContext<'_>,
    ) -> Result<bool, BoxError> {
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
        _context: &ParamContext<'_>,
    ) -> Result<Outcome, BoxError> {
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

    fn has_feasible_value(
        &self,
        side: Side<'_>,
        _context: &ParamContext<'_>,
    ) -> Result<Outcome, BoxError> {
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
        _context: &ParamContext<'_>,
    ) -> Result<Option<(ParamDomain, Vec<Constraint>)>, BoxError> {
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
        _context: &ParamContext<'_>,
    ) -> Result<(ParamDomain, Vec<Constraint>), BoxError> {
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

    /// Ask the object's `is_structurally_equivalent` about another
    /// Python-defined domain. It cannot fail, so an exception answers
    /// `false` and is kept for the entry function to raise; once one is
    /// pending, answer `false` without calling Python.
    fn eq_part(&self, other: &dyn CustomDomain) -> bool {
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return false;
        };
        if has_pending_error() {
            return false;
        }
        Python::attach(|py| {
            self.object
                .get(py)
                .call_method1(
                    intern!(py, "is_structurally_equivalent"),
                    (other.object.get(py),),
                )
                .and_then(|answer| answer.is_truthy())
                .unwrap_or_else(|error| {
                    record_pending_error(error);
                    false
                })
        })
    }
}
