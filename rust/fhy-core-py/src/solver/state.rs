//! The default solver the module functions of `fhy_core.symbolic.solver`
//! ask when no backend is named (N-S8-2 (b)).
//!
//! The binding holds one solver object in its module state, behind a
//! `Mutex`, which is locked only to read or swap the object, never across a
//! call into Python: a replaced solver is dropped after the lock is
//! released, since dropping it may run Python finalizers. It is set when
//! `fhy_core.symbolic.solver` is imported, to a solver holding the
//! z3-solver and sympy adapters, each imported on its first use, and
//! `set_default_solver` replaces it. The core crate holds no such state.

use std::sync::{Mutex, MutexGuard, PoisonError};

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;

use super::backends::type_name;
use super::facade::PySolver;

/// The default solver, once set.
static DEFAULT_SOLVER: Mutex<Option<Py<PySolver>>> = Mutex::new(None);

/// Lock the default solver. A thread that panicked while holding the lock
/// left a whole solver in place, since solvers are swapped whole.
fn lock() -> MutexGuard<'static, Option<Py<PySolver>>> {
    DEFAULT_SOLVER
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Return the solver the module functions ask when no backend is named.
///
/// Raises `RuntimeError` if none is set, which means
/// `fhy_core.symbolic.solver` was never imported.
#[pyfunction]
pub(crate) fn get_default_solver(py: Python<'_>) -> PyResult<Py<PySolver>> {
    let current = lock().as_ref().map(|solver| solver.clone_ref(py));
    current.ok_or_else(|| {
        PyRuntimeError::new_err("no default solver is set: import fhy_core.symbolic.solver first")
    })
}

/// Make `solver`, a `Solver`, the one the module functions, and so the
/// constraints and params, ask when no backend is named.
///
/// Raises `TypeError` for another value.
#[pyfunction]
pub(crate) fn set_default_solver(solver: &Bound<'_, PyAny>) -> PyResult<()> {
    let solver = solver.cast::<PySolver>().map_err(|_not_a_solver| {
        PyTypeError::new_err(format!(
            "set_default_solver solver must be a Solver, got {}.",
            type_name(solver)
        ))
    })?;
    let replaced = lock().replace(solver.clone().unbind());
    drop(replaced);
    Ok(())
}
