//! The context in which `fhy_core` asks its param questions, for a
//! downstream binding crate that asks questions of its own.
//!
//! A param question (whether a value is in a domain, whether a constraint
//! holds, what a domain contains) is a function of a [`ParamContext`]. The
//! Python API of `fhy_core` gives every question the same context, and
//! [`with_param_context`] builds it, so a downstream crate decides a
//! question as `fhy_core` does:
//!
//! - the solver is the *default solver*, the one `set_default_solver`
//!   chose, backends included;
//! - the registry is a snapshot of the user's function registry taken as
//!   the question starts, so a registered function is visible to it and a
//!   registration made meanwhile is not;
//! - the observer logs the undecided outcomes to `fhy_core`'s loggers with
//!   the text `fhy_core` uses, by the values' Python `repr`s.
//!
//! A hook the question calls (an opaque value's comparison, a custom
//! domain's method) may raise a Python exception, which cannot unwind
//! through the core: the exception is set aside and re-raised when the
//! question returns, in place of its result, so a question never returns a
//! result that a raising hook contributed to. A question may be detached
//! from the interpreter for its duration, as `fhy_core`'s own are; a hook it
//! calls then attaches for itself.

use pyo3::prelude::*;

use fhy_core::param::ParamContext;

use crate::constraint::with_pending_errors;
use crate::expression::registry_snapshot;
use crate::param::PyParamObserver;
use crate::solver::get_default_solver;

/// Run `question` with the context `fhy_core`'s own param methods use: the
/// default solver, a snapshot of the function registry and `fhy_core`'s
/// param observer.
///
/// The question is detached from the interpreter when `detach`, so other
/// Python threads run meanwhile and the question must not touch a Python
/// object except through a hook, which attaches itself; otherwise it runs
/// holding the interpreter. A Python exception raised by a hook during the
/// question is re-raised after it, in place of the result.
///
/// # Errors
///
/// Raises `RuntimeError` if no default solver is set, which means
/// `fhy_core.symbolic.solver` was never imported. Otherwise raises the
/// exception a hook raised during the question, if any: the most severe when
/// several raised, as `fhy_core` reports it.
pub fn with_param_context<T: Send>(
    py: Python<'_>,
    detach: bool,
    question: impl FnOnce(&ParamContext<'_>) -> T + Send,
) -> PyResult<T> {
    run_with_context(py, detach, |context| Ok(question(context)), |error| error)
}

/// Run `question` with the default solver, the registry snapshot and a
/// logging observer, detached from the interpreter when `is_detached`, and
/// map its error with `map_error`, still inside the scope that collects a
/// hook's exception, which takes the place of the result.
///
/// [`with_param_context`] is this over a question that cannot fail; the
/// param methods of `fhy_core` are this over the questions of the core.
pub(crate) fn run_with_context<T: Send, E: Send>(
    py: Python<'_>,
    is_detached: bool,
    question: impl FnOnce(&ParamContext<'_>) -> Result<T, E> + Send,
    map_error: impl FnOnce(E) -> PyErr,
) -> PyResult<T> {
    let solver = get_default_solver(py)?;
    let solver = solver.bind(py).get();
    let registry = registry_snapshot();
    let observer = PyParamObserver::new(solver.backend_name());
    let core = solver.core();
    let registry = registry.registry();
    with_pending_errors(|| {
        let ask = || {
            let context = ParamContext::new(core)
                .with_registry(registry)
                .with_observer(&observer);
            question(&context)
        };
        let result = if is_detached { py.detach(ask) } else { ask() };
        result.map_err(map_error)
    })
}

#[cfg(test)]
mod tests;
