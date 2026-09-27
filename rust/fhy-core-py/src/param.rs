//! `PyO3` classes and functions for [`fhy_core::param`]: the bases of the
//! six domain kinds of `fhy_core.symbolic.param.domains` (pattern P2), the
//! adapter of a Python-defined `ParamDomain` (P3), and the module functions
//! over the core's procedures.
//!
//! The Python param API takes the Rust core's semantics (D-S4-1 of
//! `docs/design/python-switch.md`, S16): values match type-strictly, and
//! each finite kind keeps its values in one order. A `Serializable` value
//! is an opaque value Python compares and orders (D-S16-3). The
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
pub(crate) use parameter::{PyParam, PyParamAssignment, check_param_bounds_are_ordered};

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
