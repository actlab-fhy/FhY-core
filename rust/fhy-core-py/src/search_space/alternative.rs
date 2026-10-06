//! `fhy_core._rs.Alternative`, the subclassable base of the public
//! `Alternative`.
//!
//! The base holds an alternative's base fields, its name, variables,
//! sub-choices and notes, set once by `_initialize`, which the public
//! class's `__init__` calls, so a Python subclass with an `__init__` of its
//! own constructs. The public class itself reaches the core as a
//! [`PlainAlternative`]; a subclass instance as a [`PythonAlternative`],
//! which calls its `extension_*` hooks; and an object of a downstream Rust
//! kind through the kind's registered functions.

use std::sync::OnceLock;

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyDict, PyTuple, PyType};

use fhy_core::foreign::Part;
use fhy_core::search_space::wire::AlternativeData;
use fhy_core::search_space::{Alternative, Choice, PlainAlternative};
use fhy_core::term::AlphaEquivalence;

use crate::diagnostic::note_to_python;
use crate::identifier::identifier_to_python;
use crate::term::read_renaming;
use crate::util::dataclass::format_dataclass_repr;
use crate::util::gc::{Slots, collect_slots};
use crate::util::pending::with_pending_errors;
use crate::util::public_class::PublicClass;
use crate::util::python::read_type_name;

use super::adapter::PythonAlternative;
use super::arguments::{
    choices_depth, ensure_depth, read_items, read_name, read_notes, wrong_argument,
};
use super::choice::{PyChoice, choice_to_python};
use super::errors::{equivalence_error_to_py, space_error_to_py};
use super::kinds::{alternative_kind, alternative_kind_of, registry};
use super::variable::{read_kind, read_variable, reduce, variable_to_python};
use super::wire::{Family, decode_part, refuse_v1, write_part, write_part_json};

/// An alternative's base fields: the core value and the objects it was
/// built from.
pub(super) struct AlternativeFields {
    /// The base fields as the core's plain alternative, its names checked.
    pub(super) plain: PlainAlternative,
    /// The `Identifier` the alternative is named by.
    pub(super) name: Py<PyAny>,
    /// The `Variable`s.
    pub(super) variables: Py<PyTuple>,
    /// The sub-`Choice`s.
    pub(super) choices: Py<PyTuple>,
    /// The `Note`s.
    pub(super) notes: Py<PyTuple>,
    /// The slots of the adapters of the subclass variables it holds.
    slots: Slots,
}

/// The base of the public `Alternative`: one option of a choice.
///
/// Its base fields are set once, by `_initialize`; a subclass adds its own
/// data in its `__init__`, and the public class freezes it afterwards.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Alternative")]
pub(crate) struct PyAlternativeBase {
    fields: OnceLock<AlternativeFields>,
}

impl PyAlternativeBase {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("Alternative");
        &PUBLIC_CLASS
    }

    /// Return the base fields.
    ///
    /// # Errors
    ///
    /// Raises `RuntimeError` for an instance whose `__init__` never called
    /// `Alternative.__init__`.
    pub(super) fn fields<'a>(slf: &'a Bound<'_, Self>) -> PyResult<&'a AlternativeFields> {
        slf.get().fields.get().ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "{} is not initialized: its __init__ must call Alternative.__init__",
                read_type_name(slf.as_any())
            ))
        })
    }

    /// Return whether `object` is an instance of the public class itself.
    fn is_plain(object: &Bound<'_, PyAny>) -> bool {
        registered_public_class(object.py()).is_some_and(|public| object.get_type().is(public))
    }
}

/// Return the registered public `Alternative` class, if
/// `fhy_core.search_space` registered it.
pub(super) fn registered_public_class(py: Python<'_>) -> Option<&Bound<'_, PyType>> {
    PyAlternativeBase::public_class().get(py).ok()
}

#[pymethods]
impl PyAlternativeBase {
    /// Accept any arguments, so a subclass with an `__init__` of its own
    /// constructs; the base fields are set by `_initialize`.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self {
            fields: OnceLock::new(),
        }
    }

    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(fields) = self.fields.get() {
            visit.call(&fields.name)?;
            visit.call(&fields.variables)?;
            visit.call(&fields.choices)?;
            visit.call(&fields.notes)?;
            fields.slots.traverse(&visit)?;
        }
        Ok(())
    }

    /// Set the base fields: the alternative named `name` (a fresh
    /// `Identifier("alternative")` when `None`) holding `variables` and the
    /// sub-choices `choices`, with `notes`.
    ///
    /// Raises `TypeError` for an argument of the wrong type,
    /// `DuplicateNameError` for names that repeat, `RecursionError` for
    /// choices nested deeper than the recursion limit, and `RuntimeError`
    /// if the fields are set already.
    #[pyo3(signature = (variables = None, choices = None, name = None, notes = None))]
    fn _initialize(
        slf: &Bound<'_, Self>,
        variables: Option<&Bound<'_, PyAny>>,
        choices: Option<&Bound<'_, PyAny>>,
        name: Option<&Bound<'_, PyAny>>,
        notes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let py = slf.py();
        let initialized = || {
            PyRuntimeError::new_err(format!(
                "{} is initialized already",
                read_type_name(slf.as_any())
            ))
        };
        if slf.get().fields.get().is_some() {
            return Err(initialized());
        }
        let (core_name, name) = read_name(py, name, "Alternative", "alternative")?;
        let variables = read_items(py, variables, "Alternative", "variables", "Variables")?;
        let (core_variables, slots) = collect_slots(|| {
            variables
                .iter()
                .map(|variable| read_variable(&variable, "Alternative", "variables"))
                .collect::<PyResult<Vec<_>>>()
        });
        let core_variables = core_variables?;
        let choices = read_items(py, choices, "Alternative", "choices", "Choices")?;
        let core_choices = read_choices(&choices, "Alternative")?;
        let (core_notes, notes) = read_notes(py, notes, "Alternative")?;
        ensure_depth(py, "alternative", choices_depth(&core_choices))?;
        let plain = PlainAlternative::new(core_name, core_variables, core_choices)
            .map_err(|error| space_error_to_py(py, error))?
            .with_notes(core_notes);
        let fields = AlternativeFields {
            plain,
            name: name.unbind(),
            variables: variables.unbind(),
            choices: choices.unbind(),
            notes: notes.unbind(),
            slots,
        };
        slf.get().fields.set(fields).map_err(|_set| initialized())
    }

    /// Register `cls` as the public `Alternative` class, and every
    /// registered downstream kind's class as its virtual subclass.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)?;
        let state = registry(cls.py())?.get().current();
        for entry in state.alternatives() {
            cls.call_method1(intern!(cls.py(), "register"), (entry.class.bind(cls.py()),))?;
        }
        Ok(())
    }

    /// The `Identifier` the alternative is named by.
    #[getter]
    fn name<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Ok(Self::fields(slf)?.name.bind(slf.py()).clone())
    }

    /// The `Variable`s that exist while the alternative is chosen.
    #[getter]
    fn variables<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        Ok(Self::fields(slf)?.variables.bind(slf.py()).clone())
    }

    /// The `Choice`s that exist while the alternative is chosen.
    #[getter]
    fn choices<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        Ok(Self::fields(slf)?.choices.bind(slf.py()).clone())
    }

    /// The `Note`s attached to the alternative.
    #[getter]
    fn notes<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        Ok(Self::fields(slf)?.notes.bind(slf.py()).clone())
    }

    /// The alternative's kind: `"search_space.alternative"`, or a
    /// subclass's registered type id.
    #[getter]
    fn kind(slf: &Bound<'_, Self>) -> PyResult<String> {
        Self::fields(slf)?;
        if Self::is_plain(slf.as_any()) {
            return Ok(PlainAlternative::KIND.to_owned());
        }
        read_kind(slf.as_any())
    }

    /// Return whether `other` is an alternative of the same kind with the
    /// same name, structurally equivalent variables and sub-choices in
    /// order, equal bound identifiers and notes, and own data the
    /// structural hook accepts.
    fn is_structurally_equivalent(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let py = slf.py();
        let this = alternative_from_python(slf.as_any())?;
        let Some(other) = try_read_alternative(other)? else {
            return Ok(false);
        };
        with_pending_errors(|| {
            this.is_structurally_equivalent(&other)
                .map_err(|error| equivalence_error_to_py(py, error))
        })
    }

    /// Return whether `other` is the same alternative up to the renaming of
    /// the names it binds, under no renaming.
    fn is_alpha_equivalent(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        let this = alternative_from_python(slf.as_any())?;
        let Some(other) = try_read_alternative(other)? else {
            return Ok(false);
        };
        with_pending_errors(|| {
            this.is_alpha_equivalent(&other)
                .map_err(|error| equivalence_error_to_py(py, error))
        })
    }

    /// Return whether `other` is the same alternative up to the renaming of
    /// the names it binds, under `renaming`.
    fn is_alpha_equivalent_under(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        renaming: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let py = slf.py();
        let renaming = read_renaming(renaming)?;
        let this = alternative_from_python(slf.as_any())?;
        let Some(other) = try_read_alternative(other)? else {
            return Ok(false);
        };
        with_pending_errors(|| {
            this.is_alpha_equivalent_under(&other, renaming.get().value().renaming())
                .map_err(|error| equivalence_error_to_py(py, error))
        })
    }

    /// Return `<Class>(name=..., variables=..., choices=..., notes=...)`.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let fields = Self::fields(slf)?;
        format_dataclass_repr(
            &slf.get_type(),
            &[
                ("name", fields.name.bind(py)),
                ("variables", fields.variables.bind(py).as_any()),
                ("choices", fields.choices.bind(py).as_any()),
                ("notes", fields.notes.bind(py).as_any()),
            ],
        )
    }

    /// Pickle the public class as a call with its fields, and a subclass
    /// through `copyreg.__newobj__` with its base fields and `__dict__`.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let fields = Self::fields(slf)?;
        let base = PyTuple::new(
            py,
            [
                fields.variables.bind(py).as_any(),
                fields.choices.bind(py).as_any(),
                fields.name.bind(py),
                fields.notes.bind(py).as_any(),
            ],
        )?;
        reduce(slf.as_any(), Self::is_plain(slf.as_any()), base)
    }

    /// Return the V2 payload: the tagged part, `{"plain": ..}` or
    /// `{"foreign": ..}`.
    ///
    /// Raises `SerializationError` inside `wire_version(WireVersion.V1)`.
    fn serialize_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        refuse_v1(slf.as_any())?;
        let part = alternative_from_python(slf.as_any())?;
        write_part(slf.py(), || AlternativeData::of(&part))
    }

    /// Return the canonical V2 text of the payload, re-formatted for
    /// `indent` or `sort_keys`.
    #[pyo3(signature = (*, indent = None, sort_keys = None))]
    fn to_json(
        slf: &Bound<'_, Self>,
        indent: Option<&Bound<'_, PyAny>>,
        sort_keys: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        refuse_v1(slf.as_any())?;
        let part = alternative_from_python(slf.as_any())?;
        write_part_json(slf.as_any(), indent, sort_keys, || {
            AlternativeData::of(&part)
        })
    }

    /// Return the alternative of the V2 payload `data`, an instance of
    /// `cls`.
    #[classmethod]
    fn deserialize_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        decode_part(cls, Family::Alternative, data, false)
    }

    /// Return the alternative of the JSON text `payload`, an instance of
    /// `cls`.
    #[classmethod]
    fn from_json<'py>(
        cls: &Bound<'py, PyType>,
        payload: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        decode_part(cls, Family::Alternative, payload, true)
    }

    /// Return the base fields' data, `{"identifier", "variables",
    /// "choices", "notes"}`, which a subclass extends with its own.
    fn serialize_data_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let fields = Self::fields(slf)?;
        crate::wire::to_dict(slf.py(), &fields.plain)
    }

    /// Return `cls(variables=, choices=, name=, notes=)` of the base fields'
    /// data `data`.
    ///
    /// Raises `DeserializationValueError` for data of another shape.
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let tagged = PyDict::new(py);
        tagged.set_item(intern!(py, "plain"), data)?;
        let plain = decode_part(cls, Family::Alternative, tagged.as_any(), false)?;
        let plain = plain.cast::<PyAlternativeBase>()?;
        let fields = Self::fields(plain)?;
        let keywords = PyDict::new(py);
        keywords.set_item(intern!(py, "variables"), fields.variables.bind(py))?;
        keywords.set_item(intern!(py, "choices"), fields.choices.bind(py))?;
        keywords.set_item(intern!(py, "name"), fields.name.bind(py))?;
        keywords.set_item(intern!(py, "notes"), fields.notes.bind(py))?;
        cls.call((), Some(&keywords))
    }
}

/// Return the core choices of the `Choice`s `choices`, items of `owner`'s
/// `choices`.
///
/// # Errors
///
/// Raises `TypeError` naming `owner` for an item that is no `Choice`.
pub(super) fn read_choices(choices: &Bound<'_, PyTuple>, owner: &str) -> PyResult<Vec<Choice>> {
    choices
        .iter()
        .map(|choice| {
            choice
                .cast::<PyChoice>()
                .map(|choice| choice.get().core().clone())
                .map_err(|_not_a_choice| wrong_argument(owner, "choices", "Choices", &choice))
        })
        .collect()
}

/// Return the core part of the `Alternative` object `object`, or `None`
/// for an object that is no `Alternative`.
///
/// # Errors
///
/// Raises `RuntimeError` for a subclass instance that was never
/// initialized, and what a registered kind's `from_python` raises.
pub(super) fn try_read_alternative(
    object: &Bound<'_, PyAny>,
) -> PyResult<Option<Part<dyn Alternative>>> {
    if let Ok(base) = object.cast::<PyAlternativeBase>() {
        let fields = PyAlternativeBase::fields(base)?;
        if PyAlternativeBase::is_plain(object) {
            return Ok(Some(Part::new(fields.plain.clone())));
        }
    }
    if let Some(entry) = alternative_kind_of(object)? {
        return (entry.from_python)(object).map(Some);
    }
    if object.cast::<PyAlternativeBase>().is_ok() {
        return Ok(Some(Part::new(PythonAlternative::new(object)?)));
    }
    Ok(None)
}

/// Return the core part of the item `object` of `owner`'s `field`.
///
/// # Errors
///
/// Raises `TypeError` naming `owner` and `field` for an object that is no
/// `Alternative`, and what [`try_read_alternative`] raises.
pub(super) fn read_alternative(
    object: &Bound<'_, PyAny>,
    owner: &str,
    field: &str,
) -> PyResult<Part<dyn Alternative>> {
    try_read_alternative(object)?
        .ok_or_else(|| wrong_argument(owner, field, "Alternatives", object))
}

/// Return the variable and choice objects of the `Alternative` object
/// `object`, whose core part is `part`: those it keeps, or new ones for a
/// downstream kind's.
///
/// # Errors
///
/// Raises what building an object raises.
pub(super) fn alternative_children<'py>(
    object: &Bound<'py, PyAny>,
    part: &Part<dyn Alternative>,
) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyTuple>)> {
    let py = object.py();
    if let Ok(base) = object.cast::<PyAlternativeBase>() {
        let fields = PyAlternativeBase::fields(base)?;
        return Ok((
            fields.variables.bind(py).clone(),
            fields.choices.bind(py).clone(),
        ));
    }
    let variables = part
        .get()
        .variables()
        .iter()
        .map(|variable| variable_to_python(py, variable))
        .collect::<PyResult<Vec<_>>>()?;
    let choices = part
        .get()
        .choices()
        .iter()
        .map(|choice| choice_to_python(py, choice, Slots::default()))
        .collect::<PyResult<Vec<_>>>()?;
    Ok((PyTuple::new(py, variables)?, PyTuple::new(py, choices)?))
}

/// Return the core part of the `Alternative` object `object`: a plain
/// alternative, a registered kind's part, or the adapter of a Python
/// subclass instance, whose slot the innermost `collect_slots` owns.
///
/// # Errors
///
/// Raises `TypeError` for an object that is no `Alternative`, and
/// `RuntimeError` for a subclass instance that was never initialized.
pub(crate) fn alternative_from_python(
    object: &Bound<'_, PyAny>,
) -> PyResult<Part<dyn Alternative>> {
    try_read_alternative(object)?.ok_or_else(|| {
        PyTypeError::new_err(format!(
            "expected an Alternative, got {}.",
            read_type_name(object)
        ))
    })
}

/// Return the Python object of `alternative`: a new public `Alternative`
/// for a plain alternative, an adapter's own object, or a registered
/// kind's object.
///
/// # Errors
///
/// Raises `TypeError` for a part of an unregistered kind, and what
/// building the object raises.
pub(crate) fn alternative_to_python<'py>(
    py: Python<'py>,
    alternative: &Part<dyn Alternative>,
) -> PyResult<Bound<'py, PyAny>> {
    let part = alternative.get();
    if let Some(adapter) = part.as_any().downcast_ref::<PythonAlternative>() {
        return Ok(adapter.object(py));
    }
    if part.as_any().downcast_ref::<PlainAlternative>().is_some() {
        let variables = part
            .variables()
            .iter()
            .map(|variable| variable_to_python(py, variable))
            .collect::<PyResult<Vec<_>>>()?;
        let choices = part
            .choices()
            .iter()
            .map(|choice| choice_to_python(py, choice, Slots::default()))
            .collect::<PyResult<Vec<_>>>()?;
        let notes = part
            .notes()
            .iter()
            .map(|note| note_to_python(py, note))
            .collect::<PyResult<Vec<_>>>()?;
        return PyAlternativeBase::public_class().get(py)?.call1((
            PyTuple::new(py, variables)?,
            PyTuple::new(py, choices)?,
            identifier_to_python(py, part.name())?,
            PyTuple::new(py, notes)?,
        ));
    }
    match alternative_kind(py, &part.kind())? {
        Some(entry) => (entry.to_python)(py, alternative),
        None => Err(PyTypeError::new_err(format!(
            "the Alternative kind {:?} is not registered",
            part.kind()
        ))),
    }
}
