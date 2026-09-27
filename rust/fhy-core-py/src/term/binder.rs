//! The functions `BinderMixin`'s derived methods call (D-S10-7): the
//! core's `Binder` algorithms over a Python binder's hooks.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyFrozenSet, PyMapping};

use fhy_core::term::Binder;

use super::adapter::{Context, PyBinder, PyTerm};
use super::renaming::read_renaming;

/// Return whether the binders `binder` and `other` are alpha-equivalent
/// under `renaming`: the same class, as many bound identifiers and scoped
/// children, and each pair of children alpha-equivalent under `renaming`
/// with one more frame pairing the bound identifiers by position. A pairing
/// the core refuses, such as a list that repeats an identifier, is not
/// equivalent.
///
/// Raises `TypeError` if `renaming` is not an `AlphaRenaming` or a bound
/// identifier is not an `Identifier`, and whatever a hook raises.
#[pyfunction]
pub(crate) fn binder_is_alpha_equivalent_under(
    binder: &Bound<'_, PyAny>,
    other: &Bound<'_, PyAny>,
    renaming: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let renaming = read_renaming(renaming)?;
    if !binder.get_type().is(other.get_type()) {
        return Ok(false);
    }
    let context = Context::new(binder.py(), Some(renaming));
    let left = PyBinder::new(binder.clone(), &context);
    let right = PyBinder::new(other.clone(), &context);
    let is_equivalent =
        left.is_binder_alpha_equivalent_under(&right, renaming.get().value().renaming());
    context.finish(is_equivalent?)
}

/// Return the free identifiers of the binder `binder`'s scoped children,
/// minus its bound identifiers, as a `frozenset`.
///
/// Raises `TypeError` if an identifier is not an `Identifier`, and whatever
/// a hook raises.
#[pyfunction]
pub(crate) fn binder_get_free_identifiers<'py>(
    binder: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyFrozenSet>> {
    let py = binder.py();
    let context = Context::new(py, None);
    let free = PyBinder::new(binder.clone(), &context).binder_free_identifiers();
    let free = context.finish(free?)?;
    let objects = free
        .iter()
        .map(|identifier| context.object_of(identifier))
        .collect::<PyResult<Vec<_>>>()?;
    PyFrozenSet::new(py, objects)
}

/// Return the binder `binder` with each free occurrence of a key of
/// `replacements` in its scoped children replaced by its value, renaming a
/// bound identifier that would capture a free identifier of a value to a
/// fresh one with the same name hint first. A key the binder binds does not
/// apply; when none applies, `binder` itself is returned.
///
/// Raises `TypeError` if `replacements` is not a mapping or a key or an
/// identifier a hook returns is not an `Identifier`, and whatever a hook
/// raises.
#[pyfunction]
pub(crate) fn binder_substitute<'py>(
    binder: &Bound<'py, PyAny>,
    replacements: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = binder.py();
    let context = Context::new(py, None);
    let Ok(mapping) = replacements.cast::<PyMapping>() else {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "BinderMixin replacements must be a mapping, got {}.",
            replacements.get_type().name()?
        )));
    };
    let mut terms = HashMap::with_capacity(mapping.len()?);
    for item in mapping.items()?.iter() {
        let (key, value) = item.extract::<(Bound<'py, PyAny>, Bound<'py, PyAny>)>()?;
        let identifier =
            crate::identifier::restore_identifier(&key, "BinderMixin", "replacement key")?;
        context.remember(&identifier, &key);
        terms.insert(identifier, PyTerm::new(value, &context));
    }
    let result = PyBinder::new(binder.clone(), &context).substitute_avoiding_capture(&terms);
    let result = context.finish(result)??;
    Ok(result.object().clone())
}
