//! The binding's one IR type: a Python object behind a node handle.

use std::fmt;

use pyo3::prelude::*;

use fhy_core::tree::{NodeHandle, NodeIdentity};

/// A Python object as the IR of a Rust pipeline.
///
/// Python has no generics at run time, so every pipeline the binding builds
/// is over this one type. Its identity is the object's address: a handle is
/// a strong reference, so while the pipeline's analysis cache holds one, no
/// other object can take the address.
pub(super) struct PyIr(Py<PyAny>);

impl PyIr {
    /// Return the handle of `object`.
    pub(super) fn new(object: &Bound<'_, PyAny>) -> Self {
        Self(object.clone().unbind())
    }

    /// Return `object` borrowed for `py`.
    pub(super) fn bind<'a, 'py>(&'a self, py: Python<'py>) -> &'a Bound<'py, PyAny> {
        self.0.bind(py)
    }

    /// Return the object, consuming the handle.
    pub(super) fn into_inner(self) -> Py<PyAny> {
        self.0
    }
}

/// Clone the reference while attached to the interpreter, so the crate
/// needs no `py-clone` feature.
impl Clone for PyIr {
    fn clone(&self) -> Self {
        Python::attach(|py| Self(self.0.clone_ref(py)))
    }
}

impl NodeHandle for PyIr {
    fn identity(&self) -> NodeIdentity {
        NodeIdentity::of_ptr(self.0.as_ptr())
    }
}

/// Render the object's address, since rendering the object would call into
/// Python.
impl fmt::Debug for PyIr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("PyIr").field(&self.0.as_ptr()).finish()
    }
}
