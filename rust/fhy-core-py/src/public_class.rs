//! The public Python classes of the Rust-backed classes.
//!
//! On the Rust backend, each public class, such as
//! `fhy_core.diagnostic.NoteKind`, is a thin Python subclass of its `PyO3`
//! class that mixes in Python protocols. A value the binding builds from
//! Rust, with no call through a class at hand, such as the kind of a note,
//! must reach Python as an instance of that public class. So each public
//! class registers itself with the binding once, at import, through its
//! `PyO3` class's `_register_public_class`, and the binding keeps it in a
//! [`PublicClass`].

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;

/// The public Python class registered for one `PyO3` class.
///
/// Write-once: a public class is registered at import and never replaced,
/// so an object built from it stays an instance of the class Python code
/// sees.
pub(crate) struct PublicClass {
    /// The name of the `PyO3` class, for error messages.
    rust_class_name: &'static str,
    class: PyOnceLock<Py<PyType>>,
}

impl PublicClass {
    /// Create the slot of the `PyO3` class named `rust_class_name`, with no
    /// public class registered.
    pub(crate) const fn new(rust_class_name: &'static str) -> Self {
        Self {
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
    /// Raises `RuntimeError` if another public class is registered already.
    pub(crate) fn register(&self, cls: &Bound<'_, PyType>) -> PyResult<()> {
        let py = cls.py();
        let registered = self.class.get_or_init(py, || cls.clone().unbind()).bind(py);
        if registered.is(cls) {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err(format!(
                "the public class of fhy_core._rs.{} is registered already, as {}",
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
    pub(crate) fn get<'py>(&self, py: Python<'py>) -> PyResult<&Bound<'py, PyType>> {
        self.class
            .get(py)
            .map(|class| class.bind(py))
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "no public class is registered for fhy_core._rs.{}",
                    self.rust_class_name
                ))
            })
    }
}
