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

use crate::expression::{install_core_registry, reinstall};
use crate::solver::{PySolver, get_default_solver, set_default_solver};
use crate::util::pending::record_pending_error;

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

/// The environment variable that tells a re-run of the test binary it is the
/// child of the story that needs a process in which no default solver was
/// ever set.
const CHILD_VARIABLE: &str = "FHY_CONVERT_PARAM_NO_DEFAULT_SOLVER";

#[test]
fn test_question_raises_a_runtime_error_when_no_default_solver_is_set() {
    // The default solver is process-wide state with no way to clear it, and
    // the other stories set it, so this one runs alone in a child process.
    if std::env::var_os(CHILD_VARIABLE).is_none() {
        let status = std::process::Command::new(std::env::current_exe().expect("the test binary"))
            .args([
                "--exact",
                "convert::param::tests::test_question_raises_a_runtime_error_when_no_default_solver_is_set",
                "--test-threads=1",
            ])
            .env(CHILD_VARIABLE, "1")
            .status()
            .expect("the child process runs");
        assert!(status.success(), "the child story failed");
        return;
    }
    with_interpreter(|py| {
        for detach in [false, true] {
            let mut ran = false;
            let error = with_param_context(py, detach, |_context| ran = true)
                .expect_err("no default solver is set");

            assert!(!ran, "the question never ran");
            assert!(error.is_instance_of::<PyRuntimeError>(py));
            assert_eq!(
                error.value(py).to_string(),
                "no default solver is set: import fhy_core.symbolic.solver first"
            );
        }
    });
}

#[test]
fn test_run_with_context_maps_the_error_of_the_question() {
    with_interpreter(|py| {
        set_default_solver(&new_solver(py)).expect("set");

        let mapped: PyResult<()> = run_with_context(
            py,
            false,
            |_context| Err("refused".to_owned()),
            PyValueError::new_err,
        );
        let error = mapped.expect_err("the question's error is mapped");
        assert!(error.is_instance_of::<PyValueError>(py));
        assert_eq!(error.value(py).to_string(), "refused");

        let kept = run_with_context(
            py,
            true,
            |_context| Ok::<_, String>(4),
            PyValueError::new_err,
        );
        assert_eq!(kept.expect("an answer is not mapped"), 4);
    });
}

#[test]
fn test_run_with_context_raises_the_exception_of_a_hook_in_place_of_the_mapped_error() {
    with_interpreter(|py| {
        set_default_solver(&new_solver(py)).expect("set");

        let result: PyResult<()> = run_with_context(
            py,
            false,
            |_context| {
                record_pending_error(PyRuntimeError::new_err("the hook failed"));
                Err("refused".to_owned())
            },
            PyValueError::new_err,
        );

        let error = result.expect_err("the hook's exception wins");
        assert!(error.is_instance_of::<PyRuntimeError>(py));
        let context = error.context(py).expect("the mapped error is chained");
        assert_eq!(context.value(py).to_string(), "refused");
    });
}
