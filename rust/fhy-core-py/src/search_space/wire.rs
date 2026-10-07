//! The V2 payloads of the search-space classes: writing and reading them,
//! the foreign parts they hold, and the depth check of a payload before it
//! reaches the core.
//!
//! [`PyResolver`] resolves a foreign variable or alternative by its type
//! id: a registered downstream Rust kind's resolver first, then the Python
//! registry, whose class decodes the data into a subclass instance that
//! becomes an adapter.
//!
//! The core decodes a space recursively, once per level of choices, so a
//! payload is refused (`RecursionError`) when its choices nest deeper than
//! Python's recursion limit, as a deep provenance is. The core's own
//! decoder refuses one whose choices nest past its `MAX_CHOICE_DEPTH`
//! (`DeserializationValueError`), which, under the default recursion
//! limit, is the refusal a deep payload meets. The classes have no V1
//! form: no V1 payload of them exists.

use pyo3::exceptions::PyRecursionError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use serde::Serialize;

use fhy_core::foreign::{BuildError, Foreign, ForeignError, Part, Resolve};
use fhy_core::param::ParamContext;
use fhy_core::search_space::wire::{
    AlternativeData, ChoiceData, ConfigurationData, SpaceData, VariableData,
};
use fhy_core::search_space::{Alternative, Variable};

use crate::convert::param::run_with_context;
use crate::util::exceptions::{DESERIALIZATION_VALUE_ERROR, SERIALIZATION_ERROR};
use crate::util::foreign::record_foreign_failure;
use crate::util::gc::collect_slots;
use crate::util::pending::with_pending_errors;
use crate::wire::{
    PyResolver, check_instance, class_name, foreign_error, parse_tree, read_text, read_text_tree,
    read_tree, resolve_object, wrong_kind,
};

use super::alternative::{alternative_to_python, try_read_alternative};
use super::arguments::recursion_limit;
use super::choice::choice_to_python;
use super::configuration::configuration_to_python;
use super::kinds::{alternative_kind, variable_kind};
use super::space::space_to_python;
use super::variable::{try_read_variable, variable_to_python};

impl Resolve<Part<dyn Variable>> for PyResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn Variable>, ForeignError> {
        Python::attach(|py| {
            match variable_kind(py, foreign.type_id()) {
                Ok(Some(entry)) => return (entry.resolve)(foreign),
                Ok(None) => {}
                Err(error) => return Err(record_foreign_failure(py, foreign.type_id(), error)),
            }
            let object = resolve_object(py, foreign, true)?;
            match try_read_variable(&object) {
                Ok(Some(part)) => Ok(part),
                Ok(None) => Err(wrong_kind(py, foreign, "Variable")),
                Err(error) => Err(record_foreign_failure(py, foreign.type_id(), error)),
            }
        })
    }
}

impl Resolve<Part<dyn Alternative>> for PyResolver {
    fn resolve(&self, foreign: &Foreign) -> Result<Part<dyn Alternative>, ForeignError> {
        Python::attach(|py| {
            match alternative_kind(py, foreign.type_id()) {
                Ok(Some(entry)) => return (entry.resolve)(foreign),
                Ok(None) => {}
                Err(error) => return Err(record_foreign_failure(py, foreign.type_id(), error)),
            }
            let object = resolve_object(py, foreign, true)?;
            match try_read_alternative(&object) {
                Ok(Some(part)) => Ok(part),
                Ok(None) => Err(wrong_kind(py, foreign, "Alternative")),
                Err(error) => Err(record_foreign_failure(py, foreign.type_id(), error)),
            }
        })
    }
}

/// The search-space class a payload is read for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Family {
    Variable,
    Alternative,
    Choice,
    Space,
    Configuration,
}

/// Raise `SerializationError` if a V1 payload is being written: the class
/// of `object` has no V1 form.
///
/// # Errors
///
/// Raises as described, and what reading the wire version raises.
pub(super) fn refuse_v1(object: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = object.py();
    if crate::wire::is_writing_v1(py)? {
        return Err(SERIALIZATION_ERROR.err(
            py,
            (format!(
                "{} has no V1 form: write it as V2.",
                crate::util::python::read_type_name(object)
            ),),
        ));
    }
    Ok(())
}

/// Return the V2 dict of the wire form `of` builds.
///
/// # Errors
///
/// Raises the exception a Python-defined part's hook raised, and
/// `SerializationError` for a part without a wire form.
pub(super) fn write_part<D: Serialize>(
    py: Python<'_>,
    of: impl FnOnce() -> Result<D, ForeignError>,
) -> PyResult<Bound<'_, PyAny>> {
    with_pending_errors(|| {
        let data = of().map_err(|error| foreign_error(py, &error))?;
        crate::wire::to_dict(py, &data)
    })
}

/// Return the JSON text of `object`, the wire form `of` builds, the
/// canonical text unless `indent` or `sort_keys` re-formats it.
///
/// # Errors
///
/// Raises as [`write_part`] does.
pub(super) fn write_part_json<D: Serialize>(
    object: &Bound<'_, PyAny>,
    indent: Option<&Bound<'_, PyAny>>,
    sort_keys: Option<&Bound<'_, PyAny>>,
    of: impl FnOnce() -> Result<D, ForeignError>,
) -> PyResult<String> {
    let py = object.py();
    with_pending_errors(|| {
        crate::wire::write_json(object, indent, sort_keys, || {
            of().map_err(|error| foreign_error(py, &error))
        })
    })
}

/// Return the value `build` builds under the context of `fhy_core`'s param
/// questions, raising a Python-defined part's exception as itself and any
/// other failure as `DeserializationValueError` naming `cls`.
///
/// # Errors
///
/// Raises as described.
fn build_in_context<T: Send>(
    cls: &Bound<'_, PyType>,
    build: impl FnOnce(&ParamContext<'_>) -> Result<T, BuildError> + Send,
) -> PyResult<T> {
    let py = cls.py();
    let name = class_name(cls);
    run_with_context(py, false, build, |error| {
        DESERIALIZATION_VALUE_ERROR
            .err(py, (format!("Invalid V2 payload for \"{name}\": {error}"),))
    })
}

/// Return the object of the V2 payload `payload` of `family`, a JSON text
/// when `is_text` and a dict otherwise, an instance of `cls`.
///
/// # Errors
///
/// Raises `RecursionError` for choices nested deeper than the recursion
/// limit, `MalformedPayloadError` for text that is no JSON,
/// `DeserializationValueError` for a payload of another shape, one nested
/// deeper than a reader reads or whose choices nest past the core's
/// `MAX_CHOICE_DEPTH`, or one a constructor refuses, the exception a
/// part's hook raises, and `SerializationError` for an object that is no
/// instance of `cls`.
pub(super) fn decode_part<'py>(
    cls: &Bound<'py, PyType>,
    family: Family,
    payload: &Bound<'py, PyAny>,
    is_text: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let py = cls.py();
    let tree = if is_text {
        read_text_tree(cls, &read_text(payload)?)?
    } else {
        read_tree(cls, payload)?
    };
    ensure_payload_depth(py, &tree, family)?;
    let object = match family {
        Family::Variable => {
            let data: VariableData = parse_tree(cls, tree)?;
            let part = build_in_context(cls, |context| data.build(&PyResolver, context))?;
            variable_to_python(py, &part)?
        }
        Family::Alternative => {
            let data: AlternativeData = parse_tree(cls, tree)?;
            let part = build_in_context(cls, |context| data.build(&PyResolver, context))?;
            alternative_to_python(py, &part)?
        }
        Family::Choice => {
            let data: ChoiceData = parse_tree(cls, tree)?;
            let (choice, slots) =
                collect_slots(|| build_in_context(cls, |context| data.build(&PyResolver, context)));
            choice_to_python(py, &choice?, slots)?
        }
        Family::Space => {
            let data: SpaceData = parse_tree(cls, tree)?;
            let (space, slots) =
                collect_slots(|| build_in_context(cls, |context| data.build(&PyResolver, context)));
            space_to_python(py, &space?, slots)?
        }
        Family::Configuration => {
            let data: ConfigurationData = parse_tree(cls, tree)?;
            let (configuration, slots) =
                collect_slots(|| build_in_context(cls, |context| data.build(&PyResolver, context)));
            let configuration = configuration?;
            let space = space_to_python(py, configuration.space(), slots)?;
            configuration_to_python(&space, configuration)?
        }
    };
    check_instance(cls, object)
}

/// Return the items of the array `key` of the object `value`, none for
/// another shape.
fn array_items<'a>(value: Option<&'a serde_json::Value>, key: &str) -> &'a [serde_json::Value] {
    value
        .and_then(|value| value.get(key))
        .and_then(serde_json::Value::as_array)
        .map_or(&[], Vec::as_slice)
}

/// Return the levels of choices the payload tree `tree` of `family` nests.
///
/// A tree of another shape answers what it can: the readers refuse it
/// afterwards.
fn payload_depth(tree: &serde_json::Value, family: Family) -> usize {
    let top: &[serde_json::Value] = match family {
        Family::Variable => &[],
        Family::Alternative => array_items(tree.get("plain"), "choices"),
        Family::Choice => std::slice::from_ref(tree),
        Family::Space => array_items(Some(tree), "choices"),
        Family::Configuration => array_items(tree.get("space"), "choices"),
    };
    let mut pending: Vec<(&serde_json::Value, usize)> =
        top.iter().map(|choice| (choice, 1)).collect();
    let mut deepest = 0;
    while let Some((choice, depth)) = pending.pop() {
        deepest = deepest.max(depth);
        for alternative in array_items(Some(choice), "alternatives") {
            for sub_choice in array_items(alternative.get("plain"), "choices") {
                pending.push((sub_choice, depth + 1));
            }
        }
    }
    deepest
}

/// Raise `RecursionError` if the choices of the payload tree `tree` of
/// `family` nest deeper than the recursion limit.
///
/// # Errors
///
/// Raises as described, and what reading the limit raises.
fn ensure_payload_depth(py: Python<'_>, tree: &serde_json::Value, family: Family) -> PyResult<()> {
    let depth = payload_depth(tree, family);
    if depth > recursion_limit(py)? {
        return Err(PyRecursionError::new_err(format!(
            "maximum recursion depth exceeded: the payload nests choices {depth} levels deep"
        )));
    }
    Ok(())
}
