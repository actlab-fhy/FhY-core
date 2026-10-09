//! `fhy_core`'s own Python extension module, `fhy_core._rs`.
//!
//! The module is only [`fhy_core_py::register`] applied to an empty module:
//! the binding is the `fhy-core-py` library, which a downstream aggregate
//! extension registers into its own module instead, so that one
//! `fhy-core` serves a whole Python process. This crate is the thin
//! `cdylib` that maturin builds when `fhy_core` is used alone, and it holds
//! nothing else. The `fhy-core-py` crate documentation explains why the
//! module and the library are separate crates.

use pyo3::prelude::*;

/// `fhy_core`'s Rust implementation.
///
/// The module declares that it uses the GIL (`gil_used = true`), so a
/// free-threaded interpreter re-enables the GIL when importing it: the
/// binding's invariants were argued for the GIL build only, and no CI job
/// runs a free-threaded one. CONTRIBUTING "One extension module per process"
/// explains why. It also sets `__version__` to the crate version, which the
/// package compares with its own when it is imported.
#[pymodule(name = "_rs", gil_used = true)]
fn rs_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    fhy_core_py::register(module.py(), module)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))
}
