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
//! - `extension_bound_identifiers()`, an alternative's.
//!
//! An exception a hook raises is the hook's error, boxed, and the entry
//! point raises it as the same object; `KeyboardInterrupt` too. A result of
//! the wrong type is a `TypeError` naming the class and the hook. `==` and
//! `hash` on the part are the instance's identity, and its foreign part is
//! its type id and its `serialize_data_to_dict()` text.

use std::borrow::Cow;
use std::fmt;
use std::hash::Hasher;

use pyo3::prelude::*;

use fhy_core::diagnostic::Note;
use fhy_core::foreign::{BoxError, Foreign, ForeignError, ForeignPart, Part};
use fhy_core::identifier::Identifier;
use fhy_core::param::Param;
use fhy_core::search_space::{Alternative, Choice, Variable};
use fhy_core::term::AlphaRenaming;

use crate::util::gc::Slot;

/// The core variable of an instance of a Python subclass of `Variable`.
pub(crate) struct PythonVariable {
    object: Slot,
    kind: String,
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
        todo!()
    }

    /// Return the instance.
    pub(crate) fn object<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.object.get(py)
    }
}

impl fmt::Debug for PythonVariable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PythonVariable")
            .field("kind", &self.kind)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl ForeignPart for PythonVariable {
    fn type_name(&self) -> Cow<'_, str> {
        todo!()
    }

    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        todo!()
    }
}

impl Variable for PythonVariable {
    fn kind(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.kind)
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
        todo!()
    }

    fn is_extension_alpha_equivalent_under(
        &self,
        other: &dyn Variable,
        renaming: &AlphaRenaming,
    ) -> Result<bool, BoxError> {
        todo!()
    }

    fn eq_part(&self, other: &dyn Variable) -> bool {
        todo!()
    }

    fn hash_part(&self, state: &mut dyn Hasher) {
        todo!()
    }
}

/// The core alternative of an instance of a Python subclass of
/// `Alternative`.
pub(crate) struct PythonAlternative {
    object: Slot,
    kind: String,
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
        todo!()
    }

    /// Return the instance.
    pub(crate) fn object<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.object.get(py)
    }
}

impl fmt::Debug for PythonAlternative {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PythonAlternative")
            .field("kind", &self.kind)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl ForeignPart for PythonAlternative {
    fn type_name(&self) -> Cow<'_, str> {
        todo!()
    }

    fn to_foreign(&self) -> Result<Foreign, ForeignError> {
        todo!()
    }
}

impl Alternative for PythonAlternative {
    fn kind(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.kind)
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

    fn bound_identifiers(&self) -> Result<Vec<Identifier>, BoxError> {
        todo!()
    }

    fn is_extension_structurally_equivalent(
        &self,
        other: &dyn Alternative,
    ) -> Result<bool, BoxError> {
        todo!()
    }

    fn is_extension_alpha_equivalent_under(
        &self,
        other: &dyn Alternative,
        renaming: &AlphaRenaming,
    ) -> Result<bool, BoxError> {
        todo!()
    }

    fn eq_part(&self, other: &dyn Alternative) -> bool {
        todo!()
    }

    fn hash_part(&self, state: &mut dyn Hasher) {
        todo!()
    }
}
