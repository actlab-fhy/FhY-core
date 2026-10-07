//! `fhy_core._rs.Choice`: a named decision among alternatives.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyDict, PyTuple, PyType};

use fhy_core::search_space::Choice;
use fhy_core::search_space::wire::ChoiceData;
use fhy_core::term::AlphaEquivalence;

use crate::diagnostic::note_to_python;
use crate::identifier::identifier_to_python;
use crate::term::read_renaming;
use crate::util::dataclass::format_dataclass_repr;
use crate::util::frozen::{refuse_attribute_assignment, refuse_attribute_deletion};
use crate::util::gc::{Slots, collect_slots};
use crate::util::pending::with_pending_errors;
use crate::util::public_class::PublicClass;
use crate::util::python::read_type_name;

use super::alternative::{alternative_to_python, read_alternative};
use super::arguments::{
    Seeded, choice_depth, ensure_depth, instantiate, read_items, read_name, read_notes, take_seed,
    wrong_seed,
};
use super::errors::{equivalence_error_to_py, space_error_to_py};
use super::wire::{Family, decode_part, refuse_v1, write_part, write_part_json};

/// A named decision among one or more alternatives, backed by the core
/// [`Choice`]; the base of the public `Choice`.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Choice")]
pub(crate) struct PyChoice {
    choice: Choice,
    /// The `Identifier` the choice is named by.
    name: Py<PyAny>,
    /// The `Alternative`s, as given.
    alternatives: Py<PyTuple>,
    /// The `Note`s.
    notes: Py<PyTuple>,
    /// The slots of the adapters of the subclass alternatives it holds.
    slots: Slots,
}

impl PyChoice {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("Choice");
        &PUBLIC_CLASS
    }

    /// Return the core choice.
    pub(super) const fn core(&self) -> &Choice {
        &self.choice
    }

    /// Return the `Alternative` objects, in order.
    pub(super) fn alternative_objects<'py>(&self, py: Python<'py>) -> &Bound<'py, PyTuple> {
        self.alternatives.bind(py)
    }
}

#[pymethods]
impl PyChoice {
    /// Return the choice named `name` (a fresh `Identifier("choice")` when
    /// `None`) among `alternatives`, with `notes`.
    ///
    /// Raises `TypeError` for an argument of the wrong type,
    /// `SearchSpaceError` for no alternative or for choices nested more
    /// than the core's `MAX_CHOICE_DEPTH` levels, `DuplicateNameError` for
    /// names that repeat, the exception an alternative's
    /// `extension_bound_identifiers` raises, and `RecursionError` for
    /// choices nested deeper than the recursion limit.
    #[new]
    #[pyo3(signature = (alternatives, name = None, notes = None, **kwargs))]
    fn new(
        alternatives: &Bound<'_, PyAny>,
        name: Option<&Bound<'_, PyAny>>,
        notes: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if let Some(seeded) = take_seed(kwargs)? {
            return match seeded {
                Seeded::Choice(choice) => Ok(choice),
                _ => Err(wrong_seed()),
            };
        }
        let py = alternatives.py();
        let (core_name, name) = read_name(py, name, "Choice", "choice")?;
        let alternatives = read_items(
            py,
            Some(alternatives),
            "Choice",
            "alternatives",
            "Alternatives",
        )?;
        let (notes_core, notes) = read_notes(py, notes, "Choice")?;
        let (choice, slots) = collect_slots(|| -> PyResult<Choice> {
            let parts = alternatives
                .iter()
                .map(|alternative| read_alternative(&alternative, "Choice", "alternatives"))
                .collect::<PyResult<Vec<_>>>()?;
            Choice::new(core_name, parts).map_err(|error| space_error_to_py(py, error))
        });
        let choice = choice?.with_notes(notes_core);
        ensure_depth(py, "choice", choice_depth(&choice))?;
        Ok(Self {
            choice,
            name: name.unbind(),
            alternatives: alternatives.unbind(),
            notes: notes.unbind(),
            slots,
        })
    }

    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.name)?;
        visit.call(&self.alternatives)?;
        visit.call(&self.notes)?;
        self.slots.traverse(&visit)
    }

    /// Register `cls` as the public `Choice` class.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }

    /// The `Identifier` the choice is named by.
    #[getter]
    fn name<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.name.bind(py).clone()
    }

    /// The `Alternative`s, in order.
    #[getter]
    fn alternatives<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        self.alternatives.bind(py).clone()
    }

    /// The `Note`s attached to the choice.
    #[getter]
    fn notes<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        self.notes.bind(py).clone()
    }

    /// Return whether `other` is a choice with the same name, structurally
    /// equivalent alternatives in order and equal notes.
    fn is_structurally_equivalent(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let py = slf.py();
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        with_pending_errors(|| {
            slf.get()
                .choice
                .is_structurally_equivalent(&other.get().choice)
                .map_err(|error| equivalence_error_to_py(py, error))
        })
    }

    /// Return whether `other` is the same choice up to the renaming of the
    /// names it binds, under no renaming.
    fn is_alpha_equivalent(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        with_pending_errors(|| {
            slf.get()
                .choice
                .is_alpha_equivalent(&other.get().choice)
                .map_err(|error| equivalence_error_to_py(py, error))
        })
    }

    /// Return whether `other` is the same choice up to the renaming of the
    /// names it binds, under `renaming`.
    fn is_alpha_equivalent_under(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        renaming: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let py = slf.py();
        let renaming = read_renaming(renaming)?;
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        with_pending_errors(|| {
            slf.get()
                .choice
                .is_alpha_equivalent_under(&other.get().choice, renaming.get().value().renaming())
                .map_err(|error| equivalence_error_to_py(py, error))
        })
    }

    /// Refuse to set an attribute: the choice is frozen.
    fn __setattr__(slf: &Bound<'_, Self>, name: &str, _value: &Bound<'_, PyAny>) -> PyResult<()> {
        refuse_attribute_assignment(slf.as_any(), name)
    }

    /// Refuse to delete an attribute: the choice is frozen.
    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        refuse_attribute_deletion(slf.as_any(), name)
    }

    /// Return `Choice(name=..., alternatives=..., notes=...)`.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        format_dataclass_repr(
            &slf.get_type(),
            &[
                ("name", this.name.bind(py)),
                ("alternatives", this.alternatives.bind(py).as_any()),
                ("notes", this.notes.bind(py).as_any()),
            ],
        )
    }

    /// Pickle as a call of the class with its fields.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.get();
        let fields = PyTuple::new(
            py,
            [
                this.alternatives.bind(py).as_any(),
                this.name.bind(py),
                this.notes.bind(py).as_any(),
            ],
        )?;
        PyTuple::new(py, [slf.get_type().into_any(), fields.into_any()])
    }

    /// Return the V2 payload `{"identifier", "alternatives", "notes"}`.
    ///
    /// Raises `SerializationError` inside `wire_version(WireVersion.V1)`.
    fn serialize_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        refuse_v1(slf.as_any())?;
        write_part(slf.py(), || ChoiceData::of(&slf.get().choice))
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
        write_part_json(slf.as_any(), indent, sort_keys, || {
            ChoiceData::of(&slf.get().choice)
        })
    }

    /// Return the choice of the V2 payload `data`.
    #[classmethod]
    fn deserialize_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        decode_part(cls, Family::Choice, data, false)
    }

    /// Return the choice of the JSON text `payload`.
    #[classmethod]
    fn from_json<'py>(
        cls: &Bound<'py, PyType>,
        payload: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        decode_part(cls, Family::Choice, payload, true)
    }
}

/// Return the core choice of the `Choice` object `object`, sharing it.
///
/// # Errors
///
/// Raises `TypeError` for an object that is no `Choice`.
pub(crate) fn choice_from_python(object: &Bound<'_, PyAny>) -> PyResult<Choice> {
    object
        .cast::<PyChoice>()
        .map(|choice| choice.get().choice.clone())
        .map_err(|_not_a_choice| {
            PyTypeError::new_err(format!(
                "expected a Choice, got {}.",
                read_type_name(object)
            ))
        })
}

/// Return a new public `Choice` of `choice`, over the objects of its parts,
/// owning `slots`.
///
/// # Errors
///
/// Raises what building the object of a part raises.
pub(crate) fn choice_to_python<'py>(
    py: Python<'py>,
    choice: &Choice,
    slots: Slots,
) -> PyResult<Bound<'py, PyAny>> {
    let alternatives = choice
        .alternatives()
        .iter()
        .map(|alternative| alternative_to_python(py, alternative))
        .collect::<PyResult<Vec<_>>>()?;
    let notes = choice
        .notes()
        .iter()
        .map(|note| note_to_python(py, note))
        .collect::<PyResult<Vec<_>>>()?;
    let seeded = PyChoice {
        choice: choice.clone(),
        name: identifier_to_python(py, choice.name())?.unbind(),
        alternatives: PyTuple::new(py, alternatives)?.unbind(),
        notes: PyTuple::new(py, notes)?.unbind(),
        slots,
    };
    instantiate(PyChoice::public_class().get(py)?, 1, Seeded::Choice(seeded))
}
