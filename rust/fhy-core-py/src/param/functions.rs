//! The module functions of `fhy_core.symbolic.param.domains` over the core:
//! `compute_constraint_implication_subset`, `evaluate_system_outcome`,
//! `are_all_constraints_satisfied` and `is_bound_expression`.

use std::sync::Arc;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use fhy_core::constraint::{Bindings, ConstraintError};
use fhy_core::param::{AssignmentError, ParamError, Side};

use crate::constraint::{
    PyConstraintSystem, PythonBindings, ReadBindings, constraint_error_to_py, outcome_to_python,
    read_binding, read_scoped_bindings, read_type_name,
};
use crate::expression::PyExpression;
use crate::identifier::restore_identifier;

use super::domains::{run_question, run_with_context};
use super::error::{ParamFailure, param_error_to_py};
use super::objects::{read_constraints, read_domain};

/// Return the exception of `error`, naming the Python objects of the
/// binding an unusable-binding error concerns.
fn evaluation_error_to_py(
    py: Python<'_>,
    error: impl Into<ParamFailure>,
    read: &ReadBindings<'_>,
) -> PyErr {
    match error.into() {
        ParamFailure::Constraint(error)
        | ParamFailure::Question(ParamError::Constraint(error))
        | ParamFailure::Assignment(AssignmentError::Constraint(error)) => {
            let binding = match &error {
                ConstraintError::UnusableBinding { identifier, .. } => read.objects(identifier),
                _ => None,
            };
            constraint_error_to_py(py, error, binding)
        }
        other => param_error_to_py(py, other, None),
    }
}

/// Decide whether the constraints of the own side are a subset of the other
/// side's, reasoning about their values in `symbol_type`.
#[pyfunction]
#[pyo3(signature = (own_domain, own_constraints, own_variable, other_domain, other_constraints, other_variable, symbol_type))]
pub(crate) fn compute_constraint_implication_subset<'py>(
    own_domain: &Bound<'py, PyAny>,
    own_constraints: &Bound<'py, PyAny>,
    own_variable: &Bound<'py, PyAny>,
    other_domain: &Bound<'py, PyAny>,
    other_constraints: &Bound<'py, PyAny>,
    other_variable: &Bound<'py, PyAny>,
    symbol_type: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = own_domain.py();
    let own = read_domain(own_domain);
    let own_constraints = read_constraints(own_constraints)?;
    let own_variable = restore_identifier(own_variable, "own_variable", "value")?;
    let other = read_domain(other_domain);
    let other_constraints = read_constraints(other_constraints)?;
    let other_variable = restore_identifier(other_variable, "other_variable", "value")?;
    let symbol_type = symbol_type
        .cast::<pyo3::types::PyString>()
        .ok()
        .and_then(|text| text.to_str().ok()?.parse().ok())
        .ok_or_else(|| {
            PyTypeError::new_err(format!(
                "symbol_type must be a SymbolType, got {}.",
                read_type_name(symbol_type)
            ))
        })?;
    let outcome = run_question(py, true, None, |context| {
        fhy_core::param::compute_constraint_implication_subset(
            &own,
            Side::new(&own_constraints, &own_variable),
            &other,
            Side::new(&other_constraints, &other_variable),
            symbol_type,
            context,
        )
    })?;
    outcome_to_python(py, outcome)
}

/// Decide `system` under `bindings`, member by member: a failure Python
/// raises as `PassExecutionError` makes its member undecided, logged at
/// WARNING.
#[pyfunction]
pub(crate) fn evaluate_system_outcome<'py>(
    system: &Bound<'py, PyAny>,
    bindings: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = system.py();
    let system = system
        .cast::<PyConstraintSystem>()
        .map_err(|_not_a_system| {
            PyTypeError::new_err(format!(
                "evaluate_system_outcome system must be a ConstraintSystem, got {}.",
                read_type_name(system)
            ))
        })?;
    let snapshot = PyDict::new(py);
    match bindings.cast::<PyDict>() {
        Ok(dict) => snapshot.update(dict.as_mapping())?,
        Err(_not_a_dict) => {
            for item in bindings
                .call_method0(pyo3::intern!(py, "items"))?
                .try_iter()?
            {
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
    let constraints = system.get().core().constraints().to_vec();
    let outcome = run_with_context(
        py,
        false,
        |context| {
            fhy_core::param::evaluate_constraints(&constraints, &core_bindings, context)
                .map(|evaluation| evaluation.outcome())
        },
        |error| evaluation_error_to_py(py, error, &read),
    )?;
    outcome_to_python(py, outcome)
}

/// Return whether `value` bound to `variable` satisfies each of
/// `constraints`, each evaluated alone.
#[pyfunction]
pub(crate) fn are_all_constraints_satisfied(
    constraints: &Bound<'_, PyAny>,
    variable: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let py = constraints.py();
    let constraints = read_constraints(constraints)?;
    let identifier = restore_identifier(variable, "variable", "value")?;
    let mapping = PyDict::new(py);
    mapping.set_item(variable, value)?;
    let read = read_scoped_bindings(mapping.as_any(), None)?;
    let bindings: Bindings = Bindings::from_iter([(identifier, read_binding(value)?)])
        .with_source(Arc::new(PythonBindings(mapping.clone().unbind())));
    run_with_context(
        py,
        false,
        |context| fhy_core::param::are_all_constraints_satisfied(&constraints, &bindings, context),
        |error| evaluation_error_to_py(py, error, &read),
    )
}

/// Return whether `expression` is an integer bound `x <cmp> k`.
#[pyfunction]
pub(crate) fn is_bound_expression(expression: &Bound<'_, PyAny>) -> bool {
    expression
        .cast::<PyExpression>()
        .is_ok_and(|expression| fhy_core::param::is_bound_expression(expression.get().expression()))
}
