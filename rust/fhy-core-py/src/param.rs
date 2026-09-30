//! `PyO3` classes and functions for [`fhy_core::param`]: the bases of the
//! six domain kinds of `fhy_core.symbolic.param.domains`, the adapter of a
//! Python-defined `ParamDomain`, and the module functions over the core's
//! procedures.
//!
//! The Python param API takes the Rust core's semantics: values match
//! type-strictly, and each finite kind keeps its values in one order. A
//! `Serializable` value is an opaque value Python compares and orders. The
//! `ParamDomain` ABC, `IntervalProfile` and `ParamError` stay Python.

mod custom;
mod domains;
mod error;
mod functions;
mod objects;
mod observer;
mod parameter;
mod value;

pub(crate) use domains::{
    PyCategoricalDomain, PyIntegerDomain, PyIntervalIntegerDomain, PyOrdinalDomain,
    PyPermutationDomain, PyRealDomain,
};
pub(crate) use functions::{
    are_all_constraints_satisfied, compute_constraint_implication_subset, evaluate_system_outcome,
    is_bound_expression,
};
pub(crate) use objects::{constraint_to_python, domain_to_python};
pub(crate) use parameter::{
    PyParam, PyParamAssignment, assignment_from_python, assignment_to_python,
    check_param_bounds_are_ordered, param_from_python, param_to_python,
};

/// Return the core domain of the Python domain `object`: a built-in kind's
/// own, and a custom domain for any other `ParamDomain`.
///
/// # Errors
///
/// Raises `TypeError` for an object that is not a `ParamDomain`.
pub(crate) fn read_domain_object(
    object: &pyo3::Bound<'_, pyo3::PyAny>,
) -> pyo3::PyResult<fhy_core::param::ParamDomain> {
    objects::read_domain_object(object)
}
