//! The stories of [`with_param_context`], run in an interpreter this test
//! binary embeds; they need no package but the standard library, since
//! `fhy_core` itself is not importable there. A `Solver` is built through
//! its class, and the registry is filled through the core, not the
//! registration functions, whose entries are Python objects of `fhy_core`'s
//! public classes.

use std::sync::{Mutex, MutexGuard, PoisonError};

use pyo3::exceptions::{PyKeyboardInterrupt, PyRuntimeError, PyValueError};

use fhy_core::expression::registry::{FunctionRegistry, NativeFunction};
use fhy_core::expression::{FunctionName, FunctionSort};

use crate::constraint::record_pending_error;
use crate::expression::{install_core_registry, reinstall};
use crate::solver::{PySolver, get_default_solver, set_default_solver};

use super::*;

/// Serializes the stories, which share the default solver and the registry.
static SERIAL: Mutex<()> = Mutex::new(());

/// Return a new `Solver` object with no backends.
fn new_solver(py: Python<'_>) -> Bound<'_, PyAny> {
    py.get_type::<PySolver>().call0().expect("a solver")
}

/// Run `body` with the embedded interpreter, alone.
fn with_interpreter<R>(body: impl FnOnce(Python<'_>) -> R) -> R {
    let _serial: MutexGuard<'_, ()> = SERIAL.lock().unwrap_or_else(PoisonError::into_inner);
    Python::initialize();
    Python::attach(body)
}

/// The address of `solver`, as the question sees it.
fn address(solver: &fhy_core::solver::Solver) -> usize {
    std::ptr::from_ref(solver) as usize
}

#[test]
fn test_question_asks_the_solver_that_set_default_solver_chose() {
    with_interpreter(|py| {
        let first = new_solver(py);
        let second = new_solver(py);
        for chosen in [&first, &second, &first] {
            set_default_solver(chosen).expect("set");
            let expected = address(chosen.cast::<PySolver>().expect("solver").get().core());
            let asked = with_param_context(py, false, |context| address(context.solver()))
                .expect("the question runs");
            assert_eq!(asked, expected);
        }
        let default = get_default_solver(py).expect("default");
        assert!(default.bind(py).as_any().is(&first));
    });
}

#[test]
fn test_question_sees_a_registered_function() {
    with_interpreter(|py| {
        let solver = new_solver(py);
        set_default_solver(&solver).expect("set");
        let name = FunctionName::new("param_context_double").expect("a name");
        let mut registry = FunctionRegistry::new();
        registry
            .register_native_function(NativeFunction::new(
                name,
                [FunctionSort::Real],
                FunctionSort::Real,
            ))
            .expect("registered");
        let previous = install_core_registry(registry);
        let seen = with_param_context(py, false, |context| {
            context
                .registry()
                .map(|registry| registry.contains("param_context_double"))
        });
        reinstall(previous);
        assert_eq!(seen.expect("the question runs"), Some(true));
        let empty = with_param_context(py, false, |context| {
            context
                .registry()
                .map(|registry| registry.contains("param_context_double"))
        });
        assert_eq!(empty.expect("the question runs"), Some(false));
    });
}

#[test]
fn test_question_re_raises_the_exception_of_a_hook_after_it() {
    with_interpreter(|py| {
        set_default_solver(&new_solver(py)).expect("set");
        let mut finished = false;
        let error = with_param_context(py, false, |_context| {
            record_pending_error(PyValueError::new_err("the hook failed"));
            finished = true;
            7
        })
        .expect_err("the exception replaces the result");
        assert!(finished, "the question ran to its end");
        assert!(error.is_instance_of::<PyValueError>(py));
        assert_eq!(error.value(py).to_string(), "the hook failed");

        // The first exception is kept, unless a later one is not an
        // `Exception`.
        let error = with_param_context(py, true, |_context| {
            record_pending_error(PyValueError::new_err("first"));
            record_pending_error(PyRuntimeError::new_err("second"));
        })
        .expect_err("raised");
        assert!(error.is_instance_of::<PyValueError>(py));
        let error = with_param_context(py, false, |_context| {
            record_pending_error(PyValueError::new_err("first"));
            record_pending_error(PyKeyboardInterrupt::new_err("interrupt"));
        })
        .expect_err("raised");
        assert!(error.is_instance_of::<PyKeyboardInterrupt>(py));

        // Nothing stays pending: the next question succeeds.
        assert_eq!(with_param_context(py, false, |_context| 3).expect("ok"), 3);
    });
}

#[test]
fn test_detached_question_releases_the_interpreter() {
    with_interpreter(|py| {
        set_default_solver(&new_solver(py)).expect("set");
        // Another thread attaches while the question runs and is joined by
        // it: attaching would block for ever if the question held the
        // interpreter.
        let attached = with_param_context(py, true, |_context| {
            std::thread::scope(|scope| {
                scope
                    .spawn(|| Python::attach(|py| py.import("sys").is_ok()))
                    .join()
                    .expect("the thread finished")
            })
        })
        .expect("the question runs");
        assert!(attached);
    });
}
