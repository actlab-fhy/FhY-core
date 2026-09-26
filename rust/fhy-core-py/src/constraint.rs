//! `PyO3` classes and functions for [`fhy_core::constraint`]: the bases of
//! the equation and set constraints of `fhy_core.symbolic.constraint`
//! (pattern P2), and `does_member_lift_to_expression`.
//!
//! The Python constraint API takes the Rust core's semantics (D-S4-1 of
//! `docs/design/python-switch.md`, S13): members compare type-strictly and
//! are kept in one canonical order, and the ordering keys and the errors
//! with a core counterpart are the core's. A `Serializable` member is an
//! opaque value that Python compares (D-S13-3). The `Constraint` ABC, the
//! outcome enum and the error classes stay Python.

mod error;
mod kinds;
mod observer;
mod value;

pub(crate) use kinds::{
    PyEquationConstraint, PyInSetConstraint, PyNotInSetConstraint, does_member_lift_to_expression,
};
