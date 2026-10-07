//! The core parts of the instances of Python subclasses of `Variable` and
//! `Alternative`.
//!
//! A container builds an adapter when it reads a subclass instance, inside
//! its `collect_slots`, so it owns the slot that holds the instance. The
//! adapter copies the base fields from the instance's base class, with no
//! call into Python, and reads its kind once: the class's registered type
//! id. Each hook of the core trait calls the instance's method of the same
//! name, once per call of the core:
//!
//! - `extension_is_structurally_equivalent(other)`;
//! - `extension_is_alpha_equivalent_under(other, renaming)`, `renaming` a
//!   new Python `AlphaRenaming` of the core's;
//! - `extension_bound_identifiers()`, an alternative's;
//! - `extension_search_domain()`, a variable's: `None`, or a
//!   `ChoiceDomain`, `OrderDomain` or `StridedDomain`.
//!
//! An exception a hook raises is the hook's error, boxed, and the entry
//! point raises it as the same object; `KeyboardInterrupt` too. A result of
//! the wrong type is a `TypeError` naming the class and the hook. Once an
//! exception is pending ([`has_pending_error`]), a hook answers its
//! fallback without calling Python, as a Python-defined constraint's do:
//! `false` for an equivalence, the param's own domain for a search domain,
//! and no identifiers for the bound ones. `==` and
//! `hash` on the part are the instance's identity, and its foreign part is
//! its type id and its `serialize_data_to_dict()` text.

use std::borrow::Cow;
use std::fmt;
use std::hash::Hasher;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyString};

use fhy_core::diagnostic::Note;
use fhy_core::foreign::{BoxError, Foreign, ForeignError, ForeignPart, Part};
use fhy_core::identifier::Identifier;
use fhy_core::param::Param;
use fhy_core::search_space::{Alternative, Choice, StepDomain, Variable};
use fhy_core::term::AlphaRenaming;

use crate::identifier::{is_python_identifier, restore_identifier};
use crate::term::renaming_to_python;
use crate::util::foreign::read_foreign;
use crate::util::gc::Slot;
use crate::util::pending::has_pending_error;
use crate::util::python::read_type_name;

use super::alternative::PyAlternativeBase;
use super::domain::step_domain_from_python;
use super::variable::{PyVariableBase, read_kind};

/// Return the core error of a hook's Python exception.
fn boxed(error: PyErr) -> BoxError {
    Box::new(error)
}

/// Return the `bool` a hook of `object` named `hook` returned.
///
/// # Errors
///
/// Raises `TypeError` for a result that is no `bool`.
fn read_bool(object: &Bound<'_, PyAny>, hook: &str, result: &Bound<'_, PyAny>) -> PyResult<bool> {
    result
        .cast::<PyBool>()
        .map(pyo3::types::PyBoolMethods::is_true)
        .map_err(|_not_a_bool| {
            PyTypeError::new_err(format!(
                "{}.{hook} must return a bool, got {}.",
                read_type_name(object),
                read_type_name(result)
            ))
        })
}

/// What an adapter knows of its instance without calling Python.
struct Instance {
    object: Slot,
    /// The instance's address, its identity for `==` and `hash`.
    address: usize,
    /// The name of the instance's class.
    type_name: String,
    /// The instance's kind, its class's registered type id.
    kind: String,
}

impl Instance {
    /// Return what an adapter of `object` knows of it.
    fn of(object: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            object: Slot::new(object.clone().unbind()),
            address: object.as_ptr().addr(),
            type_name: read_type_name(object),
            kind: read_kind(object)?,
        })
    }

    /// Return the foreign part of the instance.
    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        Python::attach(|py| read_foreign(&self.object.object(py), true))
    }

    /// Return what `other`'s instance's `hook` answers about `self`'s,
    /// called with `arguments` after `other`'s object; once an exception is
    /// pending, `false`, without calling Python.
    fn ask_bool(
        &self,
        other: &Self,
        hook: &str,
        renaming: Option<&AlphaRenaming>,
    ) -> Result<bool, BoxError> {
        if has_pending_error() {
            return Ok(false);
        }
        Python::attach(|py| -> PyResult<bool> {
            let object = self.object.get(py);
            let other = other.object.get(py);
            let hook_name = PyString::new(py, hook);
            let result = match renaming {
                Some(renaming) => {
                    object.call_method1(hook_name, (other, renaming_to_python(py, renaming)?))?
                }
                None => object.call_method1(hook_name, (other,))?,
            };
            read_bool(&object, hook, &result)
        })
        .map_err(boxed)
    }
}

/// The core variable of an instance of a Python subclass of `Variable`.
pub(crate) struct PythonVariable {
    instance: Instance,
    name: Identifier,
    param: Param,
    notes: Vec<Note>,
}

impl PythonVariable {
    /// Return the adapter of the subclass instance `object`, its slot owned
    /// by the innermost `collect_slots`.
    ///
    /// # Errors
    ///
    /// Raises `RuntimeError` for an instance that was never initialized,
    /// and what reading the class's type id raises.
    pub(crate) fn new(object: &Bound<'_, PyAny>) -> PyResult<Self> {
        let fields = PyVariableBase::fields(object.cast::<PyVariableBase>()?)?;
        let plain = &fields.plain;
        Ok(Self {
            name: plain.name().clone(),
            param: plain.param().clone(),
            notes: plain.notes().to_vec(),
            instance: Instance::of(object)?,
        })
    }

    /// Return the instance.
    pub(crate) fn object<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.instance.object.get(py)
    }
}

impl fmt::Debug for PythonVariable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PythonVariable")
            .field("kind", &self.instance.kind)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl ForeignPart for PythonVariable {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.instance.type_name)
    }

    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        self.instance.to_foreign()
    }
}

impl Variable for PythonVariable {
    fn kind(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.instance.kind)
    }

    fn name(&self) -> &Identifier {
        &self.name
    }

    fn param(&self) -> &Param {
        &self.param
    }

    fn notes(&self) -> &[Note] {
        &self.notes
    }

    fn is_extension_structurally_equivalent(&self, other: &dyn Variable) -> Result<bool, BoxError> {
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return Ok(false);
        };
        self.instance.ask_bool(
            &other.instance,
            "extension_is_structurally_equivalent",
            None,
        )
    }

    fn is_extension_alpha_equivalent_under(
        &self,
        other: &dyn Variable,
        renaming: &AlphaRenaming,
    ) -> Result<bool, BoxError> {
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return Ok(false);
        };
        self.instance.ask_bool(
            &other.instance,
            "extension_is_alpha_equivalent_under",
            Some(renaming),
        )
    }

    fn eq_part(&self, other: &dyn Variable) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.instance.address == self.instance.address)
    }

    fn hash_part(&self, state: &mut dyn Hasher) {
        state.write_usize(self.instance.address);
    }

    /// Call `extension_search_domain` once: `None` derives the domain from
    /// the param, a domain object is read as its core domain, and anything
    /// else is `TypeError`. Once an exception is pending, answer `None`
    /// without calling Python.
    fn search_domain(&self) -> Result<Option<StepDomain>, BoxError> {
        if has_pending_error() {
            return Ok(None);
        }
        Python::attach(|py| -> PyResult<Option<StepDomain>> {
            let object = self.instance.object.get(py);
            let hook = "extension_search_domain";
            let result = object.call_method0(PyString::new(py, hook))?;
            if result.is_none() {
                return Ok(None);
            }
            step_domain_from_python(&result)
                .map(Some)
                .map_err(|_not_a_domain| {
                    PyTypeError::new_err(format!(
                        "{}.{hook} must return a ChoiceDomain, an OrderDomain, a StridedDomain or \
                     None, got {}.",
                        read_type_name(&object),
                        read_type_name(&result)
                    ))
                })
        })
        .map_err(boxed)
    }
}

/// The core alternative of an instance of a Python subclass of
/// `Alternative`.
pub(crate) struct PythonAlternative {
    instance: Instance,
    name: Identifier,
    variables: Vec<Part<dyn Variable>>,
    choices: Vec<Choice>,
    notes: Vec<Note>,
}

impl PythonAlternative {
    /// Return the adapter of the subclass instance `object`, its slot owned
    /// by the innermost `collect_slots`.
    ///
    /// # Errors
    ///
    /// Raises `RuntimeError` for an instance that was never initialized,
    /// and what reading the class's type id raises.
    pub(crate) fn new(object: &Bound<'_, PyAny>) -> PyResult<Self> {
        let fields = PyAlternativeBase::fields(object.cast::<PyAlternativeBase>()?)?;
        let plain = &fields.plain;
        Ok(Self {
            name: plain.name().clone(),
            variables: plain.variables().to_vec(),
            choices: plain.choices().to_vec(),
            notes: plain.notes().to_vec(),
            instance: Instance::of(object)?,
        })
    }

    /// Return the instance.
    pub(crate) fn object<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.instance.object.get(py)
    }
}

impl fmt::Debug for PythonAlternative {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PythonAlternative")
            .field("kind", &self.instance.kind)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl ForeignPart for PythonAlternative {
    fn type_name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.instance.type_name)
    }

    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        self.instance.to_foreign()
    }
}

impl Alternative for PythonAlternative {
    fn kind(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.instance.kind)
    }

    fn name(&self) -> &Identifier {
        &self.name
    }

    fn variables(&self) -> &[Part<dyn Variable>] {
        &self.variables
    }

    fn choices(&self) -> &[Choice] {
        &self.choices
    }

    fn notes(&self) -> &[Note] {
        &self.notes
    }

    /// Call `extension_bound_identifiers`; once an exception is pending,
    /// answer none without calling Python.
    fn bound_identifiers(&self) -> Result<Vec<Identifier>, BoxError> {
        const HOOK: &str = "extension_bound_identifiers";
        if has_pending_error() {
            return Ok(Vec::new());
        }
        Python::attach(|py| -> PyResult<Vec<Identifier>> {
            let object = self.instance.object.get(py);
            let identifiers = object.call_method0(HOOK)?;
            identifiers
                .try_iter()?
                .map(|identifier| {
                    let identifier = identifier?;
                    if !is_python_identifier(&identifier)? {
                        return Err(PyTypeError::new_err(format!(
                            "{}.{HOOK} must return Identifiers, got {}.",
                            self.instance.type_name,
                            read_type_name(&identifier)
                        )));
                    }
                    restore_identifier(&identifier, &self.instance.type_name, HOOK)
                })
                .collect()
        })
        .map_err(boxed)
    }

    fn is_extension_structurally_equivalent(
        &self,
        other: &dyn Alternative,
    ) -> Result<bool, BoxError> {
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return Ok(false);
        };
        self.instance.ask_bool(
            &other.instance,
            "extension_is_structurally_equivalent",
            None,
        )
    }

    fn is_extension_alpha_equivalent_under(
        &self,
        other: &dyn Alternative,
        renaming: &AlphaRenaming,
    ) -> Result<bool, BoxError> {
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return Ok(false);
        };
        self.instance.ask_bool(
            &other.instance,
            "extension_is_alpha_equivalent_under",
            Some(renaming),
        )
    }

    fn eq_part(&self, other: &dyn Alternative) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.instance.address == self.instance.address)
    }

    fn hash_part(&self, state: &mut dyn Hasher) {
        state.write_usize(self.instance.address);
    }
}
