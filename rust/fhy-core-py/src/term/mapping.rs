//! `is_identifier_mapping_alpha_equivalent_under`: the core's comparison of
//! identifier-keyed maps over Python values.

use std::collections::HashMap;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyMapping;

use fhy_core::identifier::Identifier;
use fhy_core::term::is_mapping_alpha_equivalent_under;

use crate::identifier::restore_identifier;

use super::adapter::{Context, PyTerm};
use super::renaming::read_renaming;

/// Return the identifier keys of the Python mapping `mapping`, in its
/// iteration order, each with its value as a term.
fn read_entries<'py>(
    mapping: &Bound<'py, PyAny>,
    field: &str,
    context: &std::rc::Rc<Context<'py>>,
) -> PyResult<Vec<(Identifier, PyTerm<'py>)>> {
    let Ok(mapping) = mapping.cast::<PyMapping>() else {
        return Err(PyTypeError::new_err(format!(
            "is_identifier_mapping_alpha_equivalent_under {field} must be a mapping, got {}.",
            mapping.get_type().name()?
        )));
    };
    mapping
        .items()?
        .iter()
        .map(|item| {
            let (key, value) = item.extract::<(Bound<'py, PyAny>, Bound<'py, PyAny>)>()?;
            let identifier =
                restore_identifier(&key, "is_identifier_mapping_alpha_equivalent_under", "key")?;
            context.remember(&identifier, &key);
            Ok((identifier, PyTerm::new(value, context)))
        })
        .collect()
}

/// Return whether the identifier-keyed mappings `self_mapping` and
/// `other_mapping` are alpha-equivalent under `renaming`: as many entries;
/// each key of `self_mapping` resolving through `renaming` to a distinct
/// key of `other_mapping` that it corresponds to; then each pair of values
/// alpha-equivalent under `renaming`, compared in `self_mapping`'s order
/// and stopping at the first that differs. The keys are references, not
/// binders.
///
/// Raises `TypeError` for an argument of the wrong type or a key that is
/// not an `Identifier`, and whatever a value's comparison raises.
#[pyfunction]
pub(crate) fn is_identifier_mapping_alpha_equivalent_under(
    self_mapping: &Bound<'_, PyAny>,
    other_mapping: &Bound<'_, PyAny>,
    renaming: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let renaming = read_renaming(renaming)?;
    let context = Context::new(self_mapping.py(), Some(renaming));
    let left = read_entries(self_mapping, "self_mapping", &context)?;
    let right: HashMap<Identifier, PyTerm<'_>> =
        read_entries(other_mapping, "other_mapping", &context)?
            .into_iter()
            .collect();
    let is_equivalent = is_mapping_alpha_equivalent_under(
        left.iter().map(|(key, value)| (key, value)),
        &right,
        renaming.get().value().renaming(),
    );
    context.finish(is_equivalent?)
}
