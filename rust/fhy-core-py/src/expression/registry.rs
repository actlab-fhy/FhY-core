//! The function registry (S7): the entry classes of
//! `fhy_core.symbolic.expression.registry`, the one registry the Python API
//! reads and writes, and inlining through it.
//!
//! The registry is the core's owned
//! [`FunctionRegistry`](fhy_core::expression::registry::FunctionRegistry),
//! which the binding keeps in its module state for Python's global
//! `register_function` API (N-S7-2 (a)). The built-ins are the core's
//! catalogue, never registry entries; the Python lookups see them first,
//! through entry objects built once at import (N-S7-3 (a)). The screen and
//! the inliner read a snapshot of the registry, with no call into Python.

mod entries;
mod lookups;
mod state;

pub(crate) use entries::{PyNativeConstant, PyNativeFunction, PyRegisteredFunction};
pub(crate) use lookups::{
    get_native_constant_identifier, get_registered_entries, get_registered_entry, inline_functions,
    is_entry_registered, register_function, register_native_constant, register_native_function,
    set_registry_state_for_tests, try_get_native_constant_for_identifier,
    try_get_registered_result_sort,
};
pub(super) use state::snapshot;
