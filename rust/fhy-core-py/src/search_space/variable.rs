//! `fhy_core._rs.Variable`, the subclassable base of the public `Variable`.
//!
//! The base holds a variable's base fields, its name, param and notes, set
//! once by `_initialize`, which the public class's `__init__` calls, so a
//! Python subclass with an `__init__` of its own constructs. The public
//! class itself reaches the core as a [`PlainVariable`]; a subclass
//! instance as a [`PythonVariable`], which calls its `extension_*` hooks;
//! and an object of a downstream Rust kind through the kind's registered
//! functions.

use std::sync::OnceLock;

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyDict, PyTuple, PyType};

use fhy_core::foreign::Part;
use fhy_core::search_space::wire::VariableData;
use fhy_core::search_space::{PlainVariable, Variable};
use fhy_core::term::AlphaEquivalence;

use crate::diagnostic::note_to_python;
use crate::identifier::identifier_to_python;
use crate::param::{param_from_python, param_to_python};
use crate::term::read_renaming;
use crate::util::dataclass::format_dataclass_repr;
use crate::util::pending::with_pending_errors;
use crate::util::public_class::PublicClass;
use crate::util::python::read_type_name;

use super::adapter::PythonVariable;
use super::arguments::{read_name, read_notes, wrong_argument};
use super::errors::equivalence_error_to_py;
use super::kinds::{registry, variable_kind, variable_kind_of};
use super::wire::{Family, decode_part, refuse_v1, write_part, write_part_json};

/// A variable's base fields: the core value and the objects it was built
/// from.
pub(super) struct VariableFields {
    /// The base fields as the core's plain variable.
    pub(super) plain: PlainVariable,
    /// The `Identifier` the variable is named by.
    pub(super) name: Py<PyAny>,
    /// The `Param` whose values the variable takes.
    pub(super) param: Py<PyAny>,
    /// The `Note`s.
    pub(super) notes: Py<PyTuple>,
}

/// The base of the public `Variable`: a decision over a param's values.
///
/// Its base fields are set once, by `_initialize`; a subclass adds its own
/// data in its `__init__`, and the public class freezes it afterwards.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Variable")]
pub(crate) struct PyVariableBase {
    fields: OnceLock<VariableFields>,
}

impl PyVariableBase {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("Variable");
        &PUBLIC_CLASS
    }

    /// Return the base fields.
    ///
    /// # Errors
    ///
    /// Raises `RuntimeError` for an instance whose `__init__` never called
    /// `Variable.__init__`.
    pub(super) fn fields<'a>(slf: &'a Bound<'_, Self>) -> PyResult<&'a VariableFields> {
        slf.get().fields.get().ok_or_else(|| {
            PyRuntimeError::new_err(format!(
                "{} is not initialized: its __init__ must call Variable.__init__",
                read_type_name(slf.as_any())
            ))
        })
    }

    /// Return whether `object` is an instance of the public class itself.
    fn is_plain(object: &Bound<'_, PyAny>) -> bool {
        registered_public_class(object.py()).is_some_and(|public| object.get_type().is(public))
    }
}

/// Return the registered public `Variable` class, if `fhy_core.search_space`
/// registered it.
pub(super) fn registered_public_class(py: Python<'_>) -> Option<&Bound<'_, PyType>> {
    PyVariableBase::public_class().get(py).ok()
}

#[pymethods]
impl PyVariableBase {
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
            visit.call(&fields.param)?;
            visit.call(&fields.notes)?;
        }
        Ok(())
    }

    /// Set the base fields: the variable named `name` (a fresh
    /// `Identifier("variable")` when `None`) over `param`, with `notes`.
    ///
    /// Raises `TypeError` for an argument of the wrong type and
    /// `RuntimeError` if the fields are set already.
    #[pyo3(signature = (param, name = None, notes = None))]
    fn _initialize(
        slf: &Bound<'_, Self>,
        param: &Bound<'_, PyAny>,
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
        let core_param = param_from_python(param)
            .map_err(|_not_a_param| wrong_argument("Variable", "param", "a Param", param))?;
        let (core_name, name) = read_name(py, name, "Variable", "variable")?;
        let (core_notes, notes) = read_notes(py, notes, "Variable")?;
        let fields = VariableFields {
            plain: PlainVariable::new(core_name, core_param).with_notes(core_notes),
            name: name.unbind(),
            param: param.clone().unbind(),
            notes: notes.unbind(),
        };
        slf.get().fields.set(fields).map_err(|_set| initialized())
    }

    /// Register `cls` as the public `Variable` class, and every registered
    /// downstream kind's class as its virtual subclass.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)?;
        let state = registry(cls.py())?.get().current();
        for entry in state.variables() {
            cls.call_method1(intern!(cls.py(), "register"), (entry.class.bind(cls.py()),))?;
        }
        Ok(())
    }

    /// The `Identifier` the variable is named by.
    #[getter]
    fn name<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Ok(Self::fields(slf)?.name.bind(slf.py()).clone())
    }

    /// The `Param` whose values the variable takes.
    #[getter]
    fn param<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Ok(Self::fields(slf)?.param.bind(slf.py()).clone())
    }

    /// The `Note`s attached to the variable.
    #[getter]
    fn notes<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        Ok(Self::fields(slf)?.notes.bind(slf.py()).clone())
    }

    /// The variable's kind: `"search_space.variable"`, or a subclass's
    /// registered type id.
    #[getter]
    fn kind(slf: &Bound<'_, Self>) -> PyResult<String> {
        Self::fields(slf)?;
        if Self::is_plain(slf.as_any()) {
            return Ok(PlainVariable::KIND.to_owned());
        }
        read_kind(slf.as_any())
    }

    /// Return whether `other` is a variable of the same kind with the same
    /// name, a structurally equivalent param, equal notes, and own data
    /// the structural hook accepts.
    fn is_structurally_equivalent(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let py = slf.py();
        let this = variable_from_python(slf.as_any())?;
        let Some(other) = try_read_variable(other)? else {
            return Ok(false);
        };
        with_pending_errors(|| {
            this.is_structurally_equivalent(&other)
                .map_err(|error| equivalence_error_to_py(py, error))
        })
    }

    /// Return whether `other` is the same variable up to the renaming of
    /// the names it binds, under no renaming.
    fn is_alpha_equivalent(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        let this = variable_from_python(slf.as_any())?;
        let Some(other) = try_read_variable(other)? else {
            return Ok(false);
        };
        with_pending_errors(|| {
            this.is_alpha_equivalent(&other)
                .map_err(|error| equivalence_error_to_py(py, error))
        })
    }

    /// Return whether `other` is the same variable up to the renaming of
    /// the names it binds, under `renaming`.
    fn is_alpha_equivalent_under(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        renaming: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let py = slf.py();
        let renaming = read_renaming(renaming)?;
        let this = variable_from_python(slf.as_any())?;
        let Some(other) = try_read_variable(other)? else {
            return Ok(false);
        };
        with_pending_errors(|| {
            this.is_alpha_equivalent_under(&other, renaming.get().value().renaming())
                .map_err(|error| equivalence_error_to_py(py, error))
        })
    }

    /// Return `<Class>(name=..., param=..., notes=...)`.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let fields = Self::fields(slf)?;
        format_dataclass_repr(
            &slf.get_type(),
            &[
                ("name", fields.name.bind(py)),
                ("param", fields.param.bind(py)),
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
                fields.param.bind(py),
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
        let part = variable_from_python(slf.as_any())?;
        write_part(slf.py(), || VariableData::of(&part))
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
        let part = variable_from_python(slf.as_any())?;
        write_part_json(slf.as_any(), indent, sort_keys, || VariableData::of(&part))
    }

    /// Return the variable of the V2 payload `data`, an instance of `cls`.
    #[classmethod]
    fn deserialize_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        decode_part(cls, Family::Variable, data, false)
    }

    /// Return the variable of the JSON text `payload`, an instance of `cls`.
    #[classmethod]
    fn from_json<'py>(
        cls: &Bound<'py, PyType>,
        payload: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        decode_part(cls, Family::Variable, payload, true)
    }

    /// Return the base fields' data, `{"identifier", "param", "notes"}`,
    /// which a subclass extends with its own.
    fn serialize_data_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let fields = Self::fields(slf)?;
        crate::wire::to_dict(slf.py(), &fields.plain)
    }

    /// Return `cls(param=, name=, notes=)` of the base fields' data `data`.
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
        let plain = decode_part(cls, Family::Variable, tagged.as_any(), false)?;
        let plain = plain.cast::<PyVariableBase>()?;
        let fields = Self::fields(plain)?;
        let keywords = PyDict::new(py);
        keywords.set_item(intern!(py, "param"), fields.param.bind(py))?;
        keywords.set_item(intern!(py, "name"), fields.name.bind(py))?;
        keywords.set_item(intern!(py, "notes"), fields.notes.bind(py))?;
        cls.call((), Some(&keywords))
    }
}

/// Return the kind of the subclass instance `object`: its class's
/// registered type id.
///
/// # Errors
///
/// Raises what `get_serialization_class_type_id` raises, and `TypeError`
/// for an id that is no `str`.
pub(super) fn read_kind(object: &Bound<'_, PyAny>) -> PyResult<String> {
    object
        .get_type()
        .call_method0(intern!(object.py(), "get_serialization_class_type_id"))?
        .extract()
}

/// Return the pickle of `object`: a call of its class with `base` for an
/// instance of the public class itself, and `copyreg.__newobj__` with the
/// state `(base, __dict__)` for a subclass instance.
///
/// # Errors
///
/// Raises what reading `copyreg` or the instance's `__dict__` raises.
pub(super) fn reduce<'py>(
    object: &Bound<'py, PyAny>,
    is_plain: bool,
    base: Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyTuple>> {
    let py = object.py();
    let class = object.get_type();
    if is_plain {
        return PyTuple::new(py, [class.into_any(), base.into_any()]);
    }
    let constructor = py
        .import(intern!(py, "copyreg"))?
        .getattr(intern!(py, "__newobj__"))?;
    let dict = match object.getattr(intern!(py, "__dict__")) {
        Ok(dict) => dict,
        Err(_no_dict) => py.None().into_bound(py),
    };
    PyTuple::new(
        py,
        [
            constructor,
            PyTuple::new(py, [class])?.into_any(),
            PyTuple::new(py, [base.into_any(), dict])?.into_any(),
        ],
    )
}

/// Return the core part of the `Variable` object `object`, or `None` for
/// an object that is no `Variable`.
///
/// # Errors
///
/// Raises `RuntimeError` for a subclass instance that was never
/// initialized, and what a registered kind's `from_python` raises.
pub(super) fn try_read_variable(object: &Bound<'_, PyAny>) -> PyResult<Option<Part<dyn Variable>>> {
    if let Ok(base) = object.cast::<PyVariableBase>() {
        let fields = PyVariableBase::fields(base)?;
        if PyVariableBase::is_plain(object) {
            return Ok(Some(Part::new(fields.plain.clone())));
        }
    }
    if let Some(entry) = variable_kind_of(object)? {
        return (entry.from_python)(object).map(Some);
    }
    if object.cast::<PyVariableBase>().is_ok() {
        return Ok(Some(Part::new(PythonVariable::new(object)?)));
    }
    Ok(None)
}

/// Return the core part of the item `object` of `owner`'s `field`.
///
/// # Errors
///
/// Raises `TypeError` naming `owner` and `field` for an object that is no
/// `Variable`, and what [`try_read_variable`] raises.
pub(super) fn read_variable(
    object: &Bound<'_, PyAny>,
    owner: &str,
    field: &str,
) -> PyResult<Part<dyn Variable>> {
    try_read_variable(object)?.ok_or_else(|| wrong_argument(owner, field, "Variables", object))
}

/// Return the core part of the `Variable` object `object`: a plain
/// variable, a registered kind's part, or the adapter of a Python
/// subclass instance, whose slot the innermost `collect_slots` owns.
///
/// # Errors
///
/// Raises `TypeError` for an object that is no `Variable`, and
/// `RuntimeError` for a subclass instance that was never initialized.
pub(crate) fn variable_from_python(object: &Bound<'_, PyAny>) -> PyResult<Part<dyn Variable>> {
    try_read_variable(object)?.ok_or_else(|| {
        PyTypeError::new_err(format!(
            "expected a Variable, got {}.",
            read_type_name(object)
        ))
    })
}

/// Return the Python object of `variable`: a new public `Variable` for a
/// plain variable, an adapter's own object, or a registered kind's object.
///
/// # Errors
///
/// Raises `TypeError` for a part of an unregistered kind, and what
/// building the object raises.
pub(crate) fn variable_to_python<'py>(
    py: Python<'py>,
    variable: &Part<dyn Variable>,
) -> PyResult<Bound<'py, PyAny>> {
    let part = variable.get();
    if let Some(adapter) = part.as_any().downcast_ref::<PythonVariable>() {
        return Ok(adapter.object(py));
    }
    if part.as_any().downcast_ref::<PlainVariable>().is_some() {
        let notes = part
            .notes()
            .iter()
            .map(|note| note_to_python(py, note))
            .collect::<PyResult<Vec<_>>>()?;
        return PyVariableBase::public_class().get(py)?.call1((
            param_to_python(py, part.param())?,
            identifier_to_python(py, part.name())?,
            PyTuple::new(py, notes)?,
        ));
    }
    match variable_kind(py, &part.kind())? {
        Some(entry) => (entry.to_python)(py, variable),
        None => Err(PyTypeError::new_err(format!(
            "the Variable kind {:?} is not registered",
            part.kind()
        ))),
    }
}
