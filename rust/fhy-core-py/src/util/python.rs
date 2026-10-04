//! Helpers every binding module shares for talking to Python.
//!
//! - [`Seed`]: the contents a private seed class hands a class's `__new__`,
//!   taken once.
//! - [`ImportedAttr`] and [`cached_attr!`](crate::cached_attr): an attribute
//!   of a Python module, imported on first use and kept.
//! - [`type_name`]: the name of an object's type, for messages.

use std::fmt;
use std::sync::{Mutex, PoisonError};

use pyo3::PyTypeCheck;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;

/// The contents a seed hands the `__new__` of the class it builds, taken
/// once.
///
/// The binding builds an instance of a public class, whose `__new__` a
/// Python subclass may run, by passing it a private seed object; the seed
/// holds the Rust value and the Python objects the instance takes over. A
/// seed is taken once, and a second take raises, so an instance never
/// shares, or silently lacks, what its seed held.
#[derive(Debug)]
pub struct Seed<T>(Mutex<Option<T>>);

impl<T> Seed<T> {
    /// Return a seed holding `contents`.
    #[must_use]
    pub const fn new(contents: T) -> Self {
        Self(Mutex::new(Some(contents)))
    }

    /// Take the contents.
    ///
    /// A poisoned lock is recovered from: the contents are only ever taken,
    /// so a panic never leaves them half-updated.
    ///
    /// # Errors
    ///
    /// Raises `RuntimeError` naming `what`, the kind of seed with its
    /// article (`"an environment"`), if the seed was taken already.
    pub fn take(&self, what: &str) -> PyResult<T> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
            .ok_or_else(|| PyRuntimeError::new_err(format!("{what} seed is used once")))
    }
}

/// An attribute of a Python module, such as a class, imported on first use
/// and kept for the life of the process.
///
/// Declared as a `static`, it replaces a `PyOnceLock` and an import helper
/// per call site: `static CLASS: ImportedAttr<PyType> =
/// ImportedAttr::new("fhy_core.types", "Type");`, then `CLASS.get(py)?`.
pub struct ImportedAttr<T = PyAny> {
    module: &'static str,
    name: &'static str,
    cell: PyOnceLock<Py<T>>,
}

impl<T> fmt::Debug for ImportedAttr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImportedAttr")
            .field("module", &self.module)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl<T> ImportedAttr<T> {
    /// Return the attribute `name` of `module`, not yet imported.
    #[must_use]
    pub const fn new(module: &'static str, name: &'static str) -> Self {
        Self {
            module,
            name,
            cell: PyOnceLock::new(),
        }
    }
}

impl<T: PyTypeCheck> ImportedAttr<T> {
    /// Return the attribute, importing it on first use.
    ///
    /// A failed import is not kept: the next call tries again.
    ///
    /// # Errors
    ///
    /// Raises what importing the module or reading the attribute raises, and
    /// `TypeError` if the attribute is not a `T`.
    pub fn get<'py>(&'static self, py: Python<'py>) -> PyResult<&'py Bound<'py, T>> {
        self.cell.import(py, self.module, self.name)
    }
}

/// Return the attribute `$name` of the module `$module`, imported once for
/// this call site: `cached_attr!(py, "fhy_core.types", "Type")`, or with a
/// type, `cached_attr!(py, "fhy_core.types", "Type" => PyType)`.
///
/// The expression is a `PyResult<&Bound<T>>` (`T` is `PyAny` without a
/// type): the result of [`ImportedAttr::get`] on a `static` private to the
/// call site, so it raises what that raises. The macro is exported at the
/// crate root, `fhy_core_py::cached_attr!`, and re-exported here.
#[macro_export]
macro_rules! cached_attr {
    ($py:expr, $module:expr, $name:expr) => {
        $crate::cached_attr!($py, $module, $name => ::pyo3::PyAny)
    };
    ($py:expr, $module:expr, $name:expr => $type:ty) => {{
        static ATTRIBUTE: $crate::util::python::ImportedAttr<$type> =
            $crate::util::python::ImportedAttr::new($module, $name);
        ATTRIBUTE.get($py)
    }};
}

pub use cached_attr;

/// Return the name of `value`'s type, or `?` if reading it raises, for
/// messages.
#[must_use]
pub fn type_name(value: &Bound<'_, PyAny>) -> String {
    value
        .get_type()
        .name()
        .map_or_else(|_| "?".to_owned(), |name| name.to_string())
}

#[cfg(test)]
mod tests;
