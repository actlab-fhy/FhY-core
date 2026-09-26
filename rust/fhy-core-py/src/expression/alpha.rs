//! The conversion of `fhy_core.term.AlphaRenaming` to the Rust
//! [`AlphaRenaming`] (D-S4-3).
//!
//! `AlphaRenaming` stays a Python value class (pattern P1): the term
//! package's binder machinery, which is not ported, builds and extends it,
//! and consults it per identifier. An expression compared under one
//! converts it once per comparison: the free renaming through
//! [`AlphaRenaming::try_new`], then each binder frame, outermost first,
//! through [`AlphaRenaming::enter_binder`]. The Rust renaming then answers
//! for every identifier of the two trees, with the capture rules the Python
//! class implements.

use std::collections::HashMap;

use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyMapping, PyType};

use fhy_core::identifier::Identifier;
use fhy_core::term::AlphaRenaming;

use crate::error::IntoPyResult;
use crate::identifier::restore_identifier;

/// Return `fhy_core.term.AlphaRenaming`.
fn alpha_renaming_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    CLASS.import(py, "fhy_core.term.alpha_equivalence", "AlphaRenaming")
}

/// Return the identifier map of the Python mapping `mapping`.
fn read_identifier_map(mapping: &Bound<'_, PyAny>) -> PyResult<HashMap<Identifier, Identifier>> {
    let mapping = mapping.cast::<PyMapping>()?;
    let mut map = HashMap::with_capacity(mapping.len()?);
    for item in mapping.items()?.iter() {
        let (key, value) = item.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()?;
        map.insert(
            restore_identifier(&key, "AlphaRenaming", "key")?,
            restore_identifier(&value, "AlphaRenaming", "value")?,
        );
    }
    Ok(map)
}

/// Return the Rust renaming of the Python `AlphaRenaming` `renaming`, with
/// its binder frames and its free renaming.
///
/// # Errors
///
/// Raises `TypeError` if `renaming` is not an `AlphaRenaming`.
pub(super) fn read_alpha_renaming(renaming: &Bound<'_, PyAny>) -> PyResult<AlphaRenaming> {
    let py = renaming.py();
    if !renaming.is_instance(alpha_renaming_class(py)?)? {
        return Err(PyTypeError::new_err(format!(
            "renaming must be an AlphaRenaming, got {}.",
            renaming.get_type().name()?
        )));
    }
    let frames = renaming.getattr(intern!(py, "_frames"))?;
    let free_renaming = renaming.getattr(intern!(py, "_free_renaming"))?;
    let mut rust_renaming = if free_renaming.len()? == 0 {
        AlphaRenaming::default()
    } else {
        AlphaRenaming::try_new(read_identifier_map(&free_renaming)?).into_py_result()?
    };
    for frame in frames.try_iter()? {
        rust_renaming
            .enter_binder(read_identifier_map(&frame?)?)
            .into_py_result()?;
    }
    Ok(rust_renaming)
}
