//! `PyO3` classes and functions for [`fhy_core::constraint`]: the bases of
//! the equation and set constraints of `fhy_core.symbolic.constraint`, and
//! `does_member_lift_to_expression`.
//!
//! The Python constraint API takes the Rust core's semantics: members
//! compare type-strictly and are kept in one canonical order, and the
//! ordering keys and the errors with a core counterpart are the core's. A
//! `Serializable` member is an opaque value that Python compares. The
//! `Constraint` ABC, the outcome enum and the error classes stay Python.

mod custom;
mod error;
mod kinds;
mod observer;
mod system;
mod value;

pub(crate) use custom::{PyCustomConstraint, PythonBindings, read_outcome};
pub(crate) use error::constraint_error_to_py;
pub(crate) use kinds::{
    PyEquationConstraint, PyInSetConstraint, PyNotInSetConstraint, does_member_lift_to_expression,
};
pub(crate) use kinds::{ReadBindings, outcome_to_python, read_binding, read_scoped_bindings};
pub(crate) use observer::{DEBUG, LoggingObserver, WARNING, core_logger, join_items, log};
pub(crate) use system::{PyConstraintSystem, read_member as read_constraint, system_logger};
pub(crate) use value::{
    constraint_error, member_to_python, read_bound_value, read_opaque_member, read_type_name,
    repr_text, value_to_python,
};
