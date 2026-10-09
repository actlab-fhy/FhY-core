//! The pending exception of an infallible hook.
//!
//! Some questions the core asks a Python-defined part cannot fail: an
//! opaque value's `==`, a constraint's membership test, a custom param's
//! hook. An exception such a part raises has nowhere to go, so it is kept
//! in a per-thread slot instead, and the entry point that started the core
//! call raises it when the core returns, through [`with_pending_errors`] (or
//! [`capture_pending_errors`], for an entry point that does not return a
//! `PyResult`).
//!
//! - The first exception kept stays: an infallible hook records every
//!   exception it meets, and the caller sees the first.
//! - Once an exception is kept, [`has_pending_error`] is true and no hook
//!   may call Python again during that call, so a failed call is not
//!   followed by more Python code running.
//! - An exception that is not an `Exception`, such as `KeyboardInterrupt` or
//!   `SystemExit`, replaces a kept one that is, so an interrupt is never
//!   hidden behind an ordinary error.
//! - [`with_pending_errors`] ranks the exception its `call` returns against
//!   the one a hook recorded by the same rule, and chains the loser to the
//!   winner as its `__context__`.
//! - Each call has a frame of its own, so a nested call neither reports nor
//!   clears an outer call's exception, and a panic inside a call restores
//!   the outer frame. An exception recorded outside every call stays in a
//!   base frame of the thread.

use pyo3::exceptions::PyException;
use pyo3::prelude::*;

use super::scoped::ScopedStack;

thread_local! {
    /// The first exception a hook raised during each call into the core in
    /// progress on this thread, innermost last; a base frame, pushed on
    /// first use, keeps one raised outside every call.
    static PENDING_ERROR: ScopedStack<Option<PyErr>> = const { ScopedStack::new() };
}

/// Return whether `error` should replace the kept exception `kept`: only an
/// exception that is not an `Exception`, such as `KeyboardInterrupt` or
/// `SystemExit`, outranks one that is.
fn outranks(py: Python<'_>, error: &PyErr, kept: &PyErr) -> bool {
    kept.is_instance_of::<PyException>(py) && !error.is_instance_of::<PyException>(py)
}

/// Keep `error` as the pending exception of the call in progress on this
/// thread, unless one is kept already that it does not outrank: an
/// exception that is not an `Exception` replaces a kept `Exception`, and
/// otherwise the first one is kept.
///
/// Attaches to the interpreter if the thread is not attached. A replaced
/// or discarded exception is dropped outside the stack's borrow, since a
/// finalizer may run Python.
///
/// # Panics
///
/// Panics if the Python interpreter is not initialized, as
/// [`Python::attach`] does.
pub fn record_pending_error(error: PyErr) {
    Python::attach(|py| {
        let loser = ScopedStack::with_top_or_base_mut(
            &PENDING_ERROR,
            || None,
            move |pending| {
                if pending
                    .as_ref()
                    .is_none_or(|kept| outranks(py, &error, kept))
                {
                    pending.replace(error)
                } else {
                    Some(error)
                }
            },
        );
        // Dropped outside the borrow: a finalizer may run Python.
        drop(loser);
    });
}

/// Return whether an exception is pending on this thread, so that no
/// further hook may call Python during the current call.
#[must_use]
pub fn has_pending_error() -> bool {
    ScopedStack::with_top(&PENDING_ERROR, |pending| {
        pending.is_some_and(Option::is_some)
    })
}

/// Run `call`, and return the exception a hook raised during it, if any,
/// in place of its result.
///
/// `call` runs in a frame of its own, popped when it returns or unwinds, so
/// the exception of an enclosing call is neither seen nor changed.
///
/// # Errors
///
/// Returns the exception a hook recorded during `call`, even when `call`
/// itself returned `Ok`, and otherwise what `call` returns. When `call`
/// returned an exception too, the two are ranked as [`record_pending_error`]
/// ranks them, the recorded one first: the one that wins is returned, and
/// the other is chained to it as its `__context__`.
pub fn with_pending_errors<T>(call: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    match capture_pending_errors(call) {
        (Ok(value), None) => Ok(value),
        (Ok(_), Some(recorded)) => Err(recorded),
        (Err(error), None) => Err(error),
        (Err(returned), Some(recorded)) => Python::attach(|py| {
            let (winner, loser) = if outranks(py, &returned, &recorded) {
                (returned, recorded)
            } else {
                (recorded, returned)
            };
            attach_as_context(py, &winner, loser);
            Err(winner)
        }),
    }
}

/// Return `error` and the exceptions it is chained to through `__context__`,
/// starting with `error` and stopping where the chain repeats.
fn context_chain(py: Python<'_>, error: &PyErr) -> Vec<PyErr> {
    let mut chain = vec![error.clone_ref(py)];
    while let Some(next) = chain.last().and_then(|last| last.context(py)) {
        if chain.iter().any(|seen| seen.value(py).is(next.value(py))) {
            break;
        }
        chain.push(next);
    }
    chain
}

/// Chain `loser` after the last exception in `winner`'s `__context__` chain,
/// unless the two are chained already, which a second link would turn into a
/// cycle.
fn attach_as_context(py: Python<'_>, winner: &PyErr, loser: PyErr) {
    let winner_chain = context_chain(py, winner);
    let loser_chain = context_chain(py, &loser);
    let chained = winner_chain
        .iter()
        .any(|seen| seen.value(py).is(loser.value(py)))
        || loser_chain
            .iter()
            .any(|seen| seen.value(py).is(winner.value(py)));
    if let Some(last) = winner_chain.last().filter(|_| !chained) {
        last.set_context(py, Some(loser));
    }
}

/// Run `call`, and return its result with the exception a hook raised
/// during it, if any.
///
/// The frame is the one [`with_pending_errors`] uses; this form is for a
/// `call` that does not return a `PyResult`, and leaves what to do with the
/// exception to the caller.
pub fn capture_pending_errors<T>(call: impl FnOnce() -> T) -> (T, Option<PyErr>) {
    let scope = ScopedStack::push(&PENDING_ERROR, None);
    let result = call();
    (result, scope.pop())
}

#[cfg(test)]
mod tests {
    use pyo3::exceptions::{
        PyKeyError, PyKeyboardInterrupt, PySystemExit, PyTypeError, PyValueError,
    };

    use super::*;

    /// Return the type name of the exception `raised`, or `None`.
    fn raised_name(py: Python<'_>, raised: Option<&PyErr>) -> Option<String> {
        raised.map(|error| error.get_type(py).name().unwrap().to_string())
    }

    #[test]
    fn the_first_exception_is_kept_until_an_interrupt_replaces_it() {
        Python::initialize();
        Python::attach(|py| {
            let ((), raised) = capture_pending_errors(|| {
                record_pending_error(PyValueError::new_err("first"));
                record_pending_error(PyTypeError::new_err("second"));
            });
            assert_eq!(
                raised_name(py, raised.as_ref()).as_deref(),
                Some("ValueError")
            );

            let ((), raised) = capture_pending_errors(|| {
                record_pending_error(PyValueError::new_err("first"));
                record_pending_error(PyKeyboardInterrupt::new_err(()));
                record_pending_error(PySystemExit::new_err(()));
                record_pending_error(PyTypeError::new_err("later"));
            });
            assert_eq!(
                raised_name(py, raised.as_ref()).as_deref(),
                Some("KeyboardInterrupt")
            );
        });
    }

    #[test]
    fn a_pending_exception_is_reported_and_restored_around_a_call() {
        Python::initialize();
        Python::attach(|py| {
            let ((), outer) = capture_pending_errors(|| {
                record_pending_error(PyValueError::new_err("outer"));
                let ((), inner) = capture_pending_errors(|| {
                    assert!(!has_pending_error());
                    record_pending_error(PyTypeError::new_err("inner"));
                    assert!(has_pending_error());
                });
                assert_eq!(
                    raised_name(py, inner.as_ref()).as_deref(),
                    Some("TypeError")
                );
                assert!(has_pending_error());
            });
            assert_eq!(
                raised_name(py, outer.as_ref()).as_deref(),
                Some("ValueError")
            );
            assert!(!has_pending_error());
        });
    }

    /// Test a panic inside a call restores the outer pending exception, and
    /// leaves only the base frame behind.
    #[test]
    fn a_panic_inside_a_call_restores_the_outer_pending_exception() {
        Python::initialize();
        Python::attach(|py| {
            let ((), outer) = capture_pending_errors(|| {
                record_pending_error(PyTypeError::new_err("outer"));
                let unwound = std::panic::catch_unwind(|| {
                    with_pending_errors(|| -> PyResult<()> {
                        record_pending_error(PyTypeError::new_err("inner"));
                        panic!("inside a call")
                    })
                });
                let _panic = unwound.unwrap_err();
                assert!(has_pending_error());
            });

            assert_eq!(
                outer.map(|error| error.value(py).to_string()).as_deref(),
                Some("outer")
            );
            assert!(ScopedStack::depth(&PENDING_ERROR) <= 1);
            assert!(!has_pending_error());
        });
    }

    #[test]
    fn with_pending_errors_returns_the_result_when_nothing_was_recorded() {
        Python::initialize();
        Python::attach(|_py| {
            let result = with_pending_errors(|| Ok(5));

            assert_eq!(result.expect("the call succeeds"), 5);
            assert!(!has_pending_error());
        });
    }

    #[test]
    fn with_pending_errors_raises_the_first_recorded_exception_in_place_of_the_result() {
        Python::initialize();
        Python::attach(|py| {
            let result = with_pending_errors(|| {
                record_pending_error(PyKeyError::new_err("first"));
                assert!(has_pending_error());
                record_pending_error(PyValueError::new_err("second"));
                Ok(5)
            });

            let error = result.expect_err("the recorded exception replaces the result");
            assert!(error.is_instance_of::<PyKeyError>(py));
            assert!(!has_pending_error());
        });
    }

    #[test]
    fn a_recorded_exception_wins_over_the_one_the_call_returns_and_is_chained_to_it() {
        Python::initialize();
        Python::attach(|py| {
            let result: PyResult<()> = with_pending_errors(|| {
                record_pending_error(PyKeyError::new_err("recorded"));
                Err(PyValueError::new_err("returned"))
            });

            let error = result.expect_err("the recorded exception wins");
            assert!(error.is_instance_of::<PyKeyError>(py));
            let context = error
                .context(py)
                .expect("the returned exception is chained");
            assert!(context.is_instance_of::<PyValueError>(py));
            assert_eq!(context.value(py).to_string(), "returned");
        });
    }

    #[test]
    fn the_exception_a_call_returns_is_raised_when_none_was_recorded() {
        Python::initialize();
        Python::attach(|py| {
            let result: PyResult<()> =
                with_pending_errors(|| Err(PyValueError::new_err("returned")));

            let error = result.expect_err("the returned exception is raised");
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(error.context(py).is_none());
        });
    }

    #[test]
    fn an_interrupt_the_call_returns_outranks_a_recorded_exception() {
        Python::initialize();
        Python::attach(|py| {
            let result: PyResult<()> = with_pending_errors(|| {
                record_pending_error(PyValueError::new_err("recorded"));
                Err(PyKeyboardInterrupt::new_err(()))
            });

            let error = result.expect_err("the interrupt wins");
            assert!(error.is_instance_of::<PyKeyboardInterrupt>(py));
            let context = error
                .context(py)
                .expect("the recorded exception is chained");
            assert!(context.is_instance_of::<PyValueError>(py));
        });
    }

    #[test]
    fn the_same_exception_recorded_and_returned_is_not_chained_to_itself() {
        Python::initialize();
        Python::attach(|py| {
            let result: PyResult<()> = with_pending_errors(|| {
                let error = PyValueError::new_err("both");
                record_pending_error(error.clone_ref(py));
                Err(error)
            });

            let error = result.expect_err("the exception is raised");
            assert!(error.context(py).is_none());
        });
    }

    #[test]
    fn with_pending_errors_gives_a_nested_call_its_own_frame() {
        Python::initialize();
        Python::attach(|py| {
            let outer: PyResult<()> = with_pending_errors(|| {
                let inner: PyResult<()> = with_pending_errors(|| {
                    record_pending_error(PyKeyError::new_err("inner"));
                    Ok(())
                });
                assert!(inner.is_err());
                assert!(!has_pending_error());
                record_pending_error(PyValueError::new_err("outer"));
                Ok(())
            });

            let error = outer.expect_err("the outer call kept its own exception");
            assert!(error.is_instance_of::<PyValueError>(py));
        });
    }

    #[test]
    fn an_error_recorded_outside_every_call_stays_in_the_base_frame() {
        // A thread of its own, so no other test's frames are in the way.
        std::thread::spawn(|| {
            Python::initialize();
            Python::attach(|_py| {
                record_pending_error(PyKeyError::new_err("outside"));
                assert!(has_pending_error());

                let result = with_pending_errors(|| Ok(1));

                assert_eq!(result.expect("the call has its own frame"), 1);
                assert!(has_pending_error());
            });
        })
        .join()
        .expect("the thread finishes");
    }
}
