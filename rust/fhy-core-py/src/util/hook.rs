//! Calling a Python-defined hook from a trait method that cannot fail.
//!
//! The core asks a Python-defined part questions its traits give no error
//! to answer: an opaque value's `==`, a constraint's structural
//! equivalence. [`ask`] is the one shape every such hook has. It answers
//! the documented fallback without calling Python once an exception is
//! pending, calls the hook otherwise, and keeps the exception the hook
//! raises, or an answer it refuses to read, as the pending exception
//! ([`record_pending_error`]), which the entry point that started the core
//! call raises when the core returns ([`with_pending_errors`]).
//!
//! [`with_pending_errors`]: super::pending::with_pending_errors

use pyo3::prelude::*;

use super::pending::{has_pending_error, record_pending_error};

/// Run `call`, a question to a Python-defined hook, and return its answer;
/// the `fallback` when an exception is already pending, without calling
/// Python, and when `call` fails, in which case the exception is kept as
/// the pending one.
///
/// `call` runs attached to the interpreter, and reads the hook's answer
/// itself: an answer of the wrong type is an `Err` of its own making, kept
/// as any other exception is.
///
/// Because the fallback is returned once an exception is pending, it is
/// the answer of a call that will fail anyway; it must be a value the core
/// can use without surprise, such as `false` for an equality or an empty
/// collection for a listing.
///
/// # Pending exception
///
/// Nothing is returned for a failure, since the hook cannot fail: an
/// exception `call` returns is recorded as the pending exception, which the
/// entry point raises, and the first one pending stays.
///
/// # Panics
///
/// Panics if the Python interpreter is not initialized, as
/// [`Python::attach`] does.
pub fn ask<T>(fallback: T, call: impl FnOnce(Python<'_>) -> PyResult<T>) -> T {
    if has_pending_error() {
        return fallback;
    }
    Python::attach(|py| {
        call(py).unwrap_or_else(|error| {
            record_pending_error(error);
            fallback
        })
    })
}

#[cfg(test)]
mod tests;
