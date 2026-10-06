//! `fhy_core._rs.Variable`, the subclassable base of the public `Variable`.
//!
//! The base holds a variable's base fields, its name, param and notes, set
//! once by `_initialize`, which the public class's `__init__` calls, so a
//! Python subclass with an `__init__` of its own constructs. The public
//! class itself reaches the core as a [`PlainVariable`]; a subclass
//! instance as a [`PythonVariable`](super::adapter::PythonVariable), which
//! calls its `extension_*` hooks; and an object of a downstream Rust kind
//! through the kind's registered functions.

use std::sync::OnceLock;

use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyDict, PyTuple, PyType};

use fhy_core::foreign::Part;
use fhy_core::search_space::{PlainVariable, Variable};

use crate::util::public_class::PublicClass;

use super::kinds::registry;

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
        todo!()
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
        todo!()
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
            cls.call_method1(
                pyo3::intern!(cls.py(), "register"),
                (entry.class.bind(cls.py()),),
            )?;
        }
        Ok(())
    }

    /// The `Identifier` the variable is named by.
    #[getter]
    fn name<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// The `Param` whose values the variable takes.
    #[getter]
    fn param<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// The `Note`s attached to the variable.
    #[getter]
    fn notes<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
    }

    /// The variable's kind: `"search_space.variable"`, or a subclass's
    /// registered type id.
    #[getter]
    fn kind(slf: &Bound<'_, Self>) -> PyResult<String> {
        todo!()
    }

    /// Return whether `other` is a variable of the same kind with the same
    /// name, a structurally equivalent param, equal notes, and own data
    /// the structural hook accepts.
    fn is_structurally_equivalent(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        todo!()
    }

    /// Return whether `other` is the same variable up to the renaming of
    /// the names it binds, under no renaming.
    fn is_alpha_equivalent(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        todo!()
    }

    /// Return whether `other` is the same variable up to the renaming of
    /// the names it binds, under `renaming`.
    fn is_alpha_equivalent_under(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        renaming: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        todo!()
    }

    /// Return `<Class>(name=..., param=..., notes=...)`.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        todo!()
    }

    /// Pickle the public class as a call with its fields, and a subclass
    /// through `copyreg.__newobj__` with its base fields and `__dict__`.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        todo!()
    }

    /// Return the V2 payload: the tagged part, `{"plain": ..}` or
    /// `{"foreign": ..}`.
    ///
    /// Raises `SerializationError` inside `wire_version(WireVersion.V1)`.
    fn serialize_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return the canonical V2 text of the payload, re-formatted for
    /// `indent` or `sort_keys`.
    #[pyo3(signature = (*, indent = None, sort_keys = None))]
    fn to_json(
        slf: &Bound<'_, Self>,
        indent: Option<&Bound<'_, PyAny>>,
        sort_keys: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        todo!()
    }

    /// Return the variable of the V2 payload `data`, an instance of `cls`.
    #[classmethod]
    fn deserialize_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return the variable of the JSON text `payload`, an instance of `cls`.
    #[classmethod]
    fn from_json<'py>(
        cls: &Bound<'py, PyType>,
        payload: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return the base fields' data, `{"identifier", "param", "notes"}`,
    /// which a subclass extends with its own.
    fn serialize_data_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }

    /// Return `cls(param=, name=, notes=)` of the base fields' data `data`.
    ///
    /// Raises `DeserializationValueError` for data of another shape.
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        todo!()
    }
}

/// Return the core part of the `Variable` object `object`: a plain
/// variable, a registered kind's part, or the adapter of a Python
/// subclass instance.
///
/// # Errors
///
/// Raises `TypeError` for an object that is no `Variable`, and
/// `RuntimeError` for a subclass instance that was never initialized.
pub(crate) fn variable_from_python(object: &Bound<'_, PyAny>) -> PyResult<Part<dyn Variable>> {
    todo!()
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
    todo!()
}
