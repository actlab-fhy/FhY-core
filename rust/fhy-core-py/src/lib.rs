//! `PyO3` extension module exposing `fhy-core`'s Rust implementation to
//! Python as `fhy_core._rs`.
//!
//! One declarative module holds the whole extension. Each core module's
//! bindings live in a file of the same name, and `lib.rs` exports them into
//! one flat Python namespace: `PyO3` submodules are attributes, not
//! importable packages.

mod dataclass;
mod described_tag;
mod diagnostic;
mod error;
mod frozen;
mod identifier;
mod interned;
mod op_attribute;
mod public_class;
mod serialization;
mod value_domain;

/// `fhy_core`'s Rust implementation.
#[pyo3::pymodule(name = "_rs")]
mod rs_module {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::diagnostic::{PyDiagnostic, PyNote, PyNoteKind, PyValidationReport};
    #[pymodule_export]
    use super::identifier::{advance_identifier_counter_past, allocate_identifier_id};
    #[pymodule_export]
    use super::op_attribute::PyOpAttribute;
    #[pymodule_export]
    use super::value_domain::PyValueDomain;

    /// Set the extension's `__version__` to the crate version, which the
    /// package compares with its own before it selects the Rust backend.
    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add("__version__", env!("CARGO_PKG_VERSION"))
    }
}
