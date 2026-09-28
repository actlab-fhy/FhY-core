//! Helpers every binding module shares for talking to Python (R2-033).
//!
//! - [`Seed`]: the contents a private seed class hands a class's `__new__`,
//!   taken once.

use std::sync::{Mutex, PoisonError};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// The contents a seed hands the `__new__` of the class it builds, taken
/// once.
///
/// The binding builds an instance of a public class, whose `__new__` a
/// Python subclass may run, by passing it a private seed object; the seed
/// holds the Rust value and the Python objects the instance takes over. A
/// seed is taken once, and a second take raises, so an instance never
/// shares, or silently lacks, what its seed held.
pub(crate) struct Seed<T>(Mutex<Option<T>>);

impl<T> Seed<T> {
    /// Return a seed holding `contents`.
    pub(crate) fn new(contents: T) -> Self {
        Self(Mutex::new(Some(contents)))
    }

    /// Take the contents.
    ///
    /// # Errors
    ///
    /// Raises `RuntimeError` naming `what`, the kind of seed with its
    /// article (`"an environment"`), if the seed was taken already.
    pub(crate) fn take(&self, what: &str) -> PyResult<T> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
            .ok_or_else(|| PyRuntimeError::new_err(format!("{what} seed is used once")))
    }
}
