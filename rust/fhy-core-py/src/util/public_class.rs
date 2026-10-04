//! The public Python classes of the Rust-backed classes.
//!
//! Each public class, such as
//! `fhy_core.diagnostic.NoteKind`, is a thin Python subclass of its `PyO3`
//! class that mixes in Python protocols. A value the binding builds from
//! Rust, with no call through a class at hand, such as the kind of a note,
//! must reach Python as an instance of that public class. So each public
//! class registers itself with the binding once, at import, through its
//! `PyO3` class's `_register_public_class`, and the binding keeps it in a
//! [`PublicClass`].
//!
//! A downstream binding crate declares the slot of each of its classes with
//! [`PublicClass::in_module`], naming its own module.

use std::fmt;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;

/// The public Python class registered for one `PyO3` class.
///
/// Write-once: a public class is registered at import and never replaced,
/// so an object built from it stays an instance of the class Python code
/// sees.
pub struct PublicClass {
    /// The module of the `PyO3` class, for error messages.
    module: &'static str,
    /// The name of the `PyO3` class, for error messages.
    rust_class_name: &'static str,
    class: PyOnceLock<Py<PyType>>,
}

impl fmt::Debug for PublicClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PublicClass")
            .field("module", &self.module)
            .field("rust_class_name", &self.rust_class_name)
            .finish_non_exhaustive()
    }
}

impl PublicClass {
    /// Create the slot of the `PyO3` class named `rust_class_name` of
    /// `fhy_core._rs`, with no public class registered.
    #[must_use]
    pub const fn new(rust_class_name: &'static str) -> Self {
        Self::in_module("fhy_core._rs", rust_class_name)
    }

    /// Create the slot of the `PyO3` class named `rust_class_name` of the
    /// module `module` (`"moga._rs"`), which the error messages name, with
    /// no public class registered.
    #[must_use]
    pub const fn in_module(module: &'static str, rust_class_name: &'static str) -> Self {
        Self {
            module,
            rust_class_name,
            class: PyOnceLock::new(),
        }
    }

    /// Register `cls` as the public class.
    ///
    /// Registering the registered class again does nothing. The caller is
    /// a class method of the `PyO3` class, so `cls` subclasses it.
    ///
    /// # Errors
    ///
    /// Raises `RuntimeError` naming the registered class if another public
    /// class is registered already, and what reading its name raises.
    pub fn register(&self, cls: &Bound<'_, PyType>) -> PyResult<()> {
        let py = cls.py();
        let registered = self.class.get_or_init(py, || cls.clone().unbind()).bind(py);
        if registered.is(cls) {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err(format!(
                "the public class of {}.{} is registered already, as {}",
                self.module,
                self.rust_class_name,
                registered.qualname()?
            )))
        }
    }

    /// Return the registered public class.
    ///
    /// # Errors
    ///
    /// Raises `RuntimeError` if no public class is registered, which means
    /// the module defining it was never imported.
    pub fn get<'py>(&self, py: Python<'py>) -> PyResult<&Bound<'py, PyType>> {
        self.class
            .get(py)
            .map(|class| class.bind(py))
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "no public class is registered for {}.{}",
                    self.module, self.rust_class_name
                ))
            })
    }
}

#[cfg(test)]
mod tests;
