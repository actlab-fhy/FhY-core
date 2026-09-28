//! Python terms and binders as implementations of the core's term traits.
//!
//! A [`PyTerm`] is any Python object with the `Term` protocol's methods,
//! and a [`PyBinder`] a `BinderMixin` instance. A comparison, a scope or a
//! substitution whose hook raises returns the exception as its error. The
//! binder's bound identifiers and scoped children are the exception: the
//! core reads them as slices, which cannot fail, so every adapter of one
//! call shares a [`Context`] that keeps the first exception those reads
//! raise, and the next fallible hook, or the entry function when the core
//! returns, raises it; no hook runs after it.
//!
//! The context also keeps the Python object of every identifier the hooks
//! returned, so a renaming the core extends can be handed back to Python
//! with its objects, and a Rust identifier the core reports can become the
//! object it came from.

use std::cell::{OnceCell, RefCell};
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasher;
use std::rc::Rc;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use fhy_core::identifier::Identifier;
use fhy_core::term::{AlphaEquivalence, AlphaRenaming, Binder, FreeIdentifiers, Term};

use crate::expression::PyExpression;
use crate::identifier::{identifier_to_python, restore_identifier};

use super::renaming::{ObjectTable, PyAlphaRenaming, RenamingValue};

/// What the adapters of one call share: the first exception a hook raised,
/// and the identifier objects seen so far.
pub(super) struct Context<'py> {
    py: Python<'py>,
    error: RefCell<Option<PyErr>>,
    objects: RefCell<(ObjectTable, HashMap<u64, Py<PyAny>>)>,
    /// The renaming the call was given, and its object, handed back to
    /// Python when the core asks under that very renaming.
    given: Option<(AlphaRenaming, Py<PyAlphaRenaming>)>,
}

impl<'py> Context<'py> {
    /// Return a context over the identifier objects of `renaming`, the
    /// renaming the call was given.
    pub(super) fn new(py: Python<'py>, renaming: Option<&Bound<'py, PyAlphaRenaming>>) -> Rc<Self> {
        let (objects, given) = match renaming {
            Some(renaming) => {
                let value = renaming.get().value();
                (
                    value.objects().clone(),
                    Some((value.renaming().clone(), renaming.clone().unbind())),
                )
            }
            None => (ObjectTable::default(), None),
        };
        Rc::new(Self {
            py,
            error: RefCell::new(None),
            objects: RefCell::new((objects, HashMap::new())),
            given,
        })
    }

    /// Return whether a hook has raised.
    pub(super) fn has_failed(&self) -> bool {
        self.error.borrow().is_some()
    }

    /// Keep `error` if it is the first.
    fn fail(&self, error: PyErr) {
        let mut kept = self.error.borrow_mut();
        if kept.is_none() {
            *kept = Some(error);
        }
    }

    /// Return the kept error, if a binder's bound identifiers or children
    /// failed to read, so no hook runs after it.
    pub(super) fn check(&self) -> PyResult<()> {
        match self.error.borrow_mut().take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// Return the value of `result`, or keep its error and return `fallback`.
    fn record<T>(&self, result: PyResult<T>, fallback: T) -> T {
        result.unwrap_or_else(|error| {
            self.fail(error);
            fallback
        })
    }

    /// Return the kept error, if any, and `result` otherwise.
    pub(super) fn finish<T>(&self, result: T) -> PyResult<T> {
        match self.error.borrow_mut().take() {
            Some(error) => Err(error),
            None => Ok(result),
        }
    }

    /// Remember the object of `identifier`.
    pub(super) fn remember(&self, identifier: &Identifier, object: &Bound<'py, PyAny>) {
        self.objects
            .borrow_mut()
            .1
            .insert(identifier.id(), object.clone().unbind());
    }

    /// Return the identifier objects seen so far.
    fn objects(&self) -> ObjectTable {
        let mut objects = self.objects.borrow_mut();
        if !objects.1.is_empty() {
            let added = std::mem::take(&mut objects.1);
            objects.0 = objects.0.with(added);
        }
        objects.0.clone()
    }

    /// Return the Python object of `identifier`.
    pub(super) fn object_of(&self, identifier: &Identifier) -> PyResult<Bound<'py, PyAny>> {
        if let Some(object) = self.objects.borrow().1.get(&identifier.id()) {
            return Ok(object.bind(self.py).clone());
        }
        match self.objects.borrow().0.get(self.py, identifier.id()) {
            Some(object) => Ok(object),
            None => identifier_to_python(self.py, identifier),
        }
    }

    /// Return the Python object of `renaming`: the given one, if it is that
    /// renaming, or a new one with the identifier objects seen so far.
    fn renaming_object(&self, renaming: &AlphaRenaming) -> PyResult<Bound<'py, PyAny>> {
        if let Some((given, object)) = &self.given {
            if given == renaming {
                return Ok(object.bind(self.py).clone().into_any());
            }
        }
        Ok(RenamingValue::new(renaming.clone(), self.objects())
            .into_python(self.py)?
            .into_any())
    }

    /// Return the Rust identifier of the Python `object`, remembering its
    /// object.
    fn read_identifier(
        &self,
        object: &Bound<'py, PyAny>,
        owner: &str,
        field: &str,
    ) -> PyResult<Identifier> {
        let identifier = restore_identifier(object, owner, field)?;
        self.remember(&identifier, object);
        Ok(identifier)
    }

    /// Return the identifiers of the Python iterable `values`, remembering
    /// their objects.
    pub(super) fn read_identifiers(
        &self,
        values: &Bound<'py, PyAny>,
        owner: &str,
        field: &str,
    ) -> PyResult<HashSet<Identifier>> {
        values
            .try_iter()?
            .map(|value| self.read_identifier(&value?, owner, field))
            .collect()
    }
}

/// Return the Rust expression of `object` when it is an expression whose
/// class does not override the method `method` of `_rs.Expression`, so the
/// core answers for it without calling Python.
pub(super) fn find_native_expression<'a>(
    object: &'a Bound<'_, PyAny>,
    method: &Bound<'_, pyo3::types::PyString>,
) -> Option<&'a PyExpression> {
    let expression = object.cast::<PyExpression>().ok()?;
    let py = object.py();
    let base = py.get_type::<PyExpression>().getattr(method).ok()?;
    let own = object.get_type().getattr(method).ok()?;
    own.is(&base).then(|| expression.get())
}

// ---------------------------------------------------------------------------
// Terms
// ---------------------------------------------------------------------------

/// A Python term: any object with `is_alpha_equivalent_under`,
/// `get_free_identifiers` and `substitute`.
#[derive(Clone)]
pub(super) struct PyTerm<'py> {
    object: Bound<'py, PyAny>,
    context: Rc<Context<'py>>,
}

impl<'py> PyTerm<'py> {
    pub(super) fn new(object: Bound<'py, PyAny>, context: &Rc<Context<'py>>) -> Self {
        Self {
            object,
            context: Rc::clone(context),
        }
    }

    pub(super) fn object(&self) -> &Bound<'py, PyAny> {
        &self.object
    }

    fn compare(&self, other: &Self, renaming: &AlphaRenaming) -> PyResult<bool> {
        let py = self.object.py();
        let method = intern!(py, "is_alpha_equivalent_under");
        if let Some(expression) = find_native_expression(&self.object, method) {
            return Ok(other.object.cast::<PyExpression>().is_ok_and(|other| {
                expression
                    .expression()
                    .is_alpha_equivalent_under(other.get().expression(), renaming)
            }));
        }
        let renaming = self.context.renaming_object(renaming)?;
        self.object
            .call_method1(method, (&other.object, renaming))?
            .is_truthy()
    }

    fn read_free_identifiers(&self) -> PyResult<HashSet<Identifier>> {
        let py = self.object.py();
        let free = self
            .object
            .call_method0(intern!(py, "get_free_identifiers"))?;
        self.context
            .read_identifiers(&free, "Term", "free identifier")
    }
}

impl AlphaEquivalence for PyTerm<'_> {
    type Error = PyErr;

    /// Ask the object's `is_alpha_equivalent_under`, or the core for an
    /// expression that does not override it; its exception is the error.
    fn is_alpha_equivalent_under(&self, other: &Self, renaming: &AlphaRenaming) -> PyResult<bool> {
        self.context.check()?;
        self.compare(other, renaming)
    }
}

impl FreeIdentifiers for PyTerm<'_> {
    type Error = PyErr;

    /// Return the object's `get_free_identifiers`; its exception is the
    /// error.
    fn free_identifiers(&self) -> PyResult<HashSet<Identifier>> {
        self.context.check()?;
        self.read_free_identifiers()
    }
}

impl Term for PyTerm<'_> {
    type SubstituteError = PyErr;

    fn substitute<S: BuildHasher>(
        &self,
        replacements: &HashMap<Identifier, Self, S>,
    ) -> PyResult<Self> {
        self.context.check()?;
        let py = self.object.py();
        let mapping = PyDict::new(py);
        for (identifier, term) in replacements {
            mapping.set_item(self.context.object_of(identifier)?, &term.object)?;
        }
        let result = self
            .object
            .call_method1(intern!(py, "substitute"), (mapping,))?;
        Ok(Self::new(result, &self.context))
    }
}

// ---------------------------------------------------------------------------
// Binders
// ---------------------------------------------------------------------------

/// A `BinderMixin` instance. Its bound identifiers and scoped children are
/// read through its hooks on first use.
#[derive(Clone)]
pub(super) struct PyBinder<'py> {
    object: Bound<'py, PyAny>,
    context: Rc<Context<'py>>,
    bound: OnceCell<Vec<Identifier>>,
    children: OnceCell<Vec<PyTerm<'py>>>,
}

impl<'py> PyBinder<'py> {
    pub(super) fn new(object: Bound<'py, PyAny>, context: &Rc<Context<'py>>) -> Self {
        Self {
            object,
            context: Rc::clone(context),
            bound: OnceCell::new(),
            children: OnceCell::new(),
        }
    }

    pub(super) fn object(&self) -> &Bound<'py, PyAny> {
        &self.object
    }

    fn read_bound(&self) -> PyResult<Vec<Identifier>> {
        let py = self.object.py();
        let bound = self
            .object
            .call_method0(intern!(py, "get_bound_identifiers"))?;
        bound
            .try_iter()?
            .map(|value| {
                self.context
                    .read_identifier(&value?, "BinderMixin", "bound identifier")
            })
            .collect()
    }

    fn read_children(&self) -> PyResult<Vec<PyTerm<'py>>> {
        let py = self.object.py();
        let children = self
            .object
            .call_method0(intern!(py, "get_scoped_children"))?;
        children
            .try_iter()?
            .map(|child| Ok(PyTerm::new(child?, &self.context)))
            .collect()
    }
}

impl<'py> Binder for PyBinder<'py> {
    type Child = PyTerm<'py>;
    type RebuildError = PyErr;

    fn bound_identifiers(&self) -> &[Identifier] {
        self.bound.get_or_init(|| {
            if self.context.has_failed() {
                return Vec::new();
            }
            self.context.record(self.read_bound(), Vec::new())
        })
    }

    fn scoped_children(&self) -> &[PyTerm<'py>] {
        self.children.get_or_init(|| {
            if self.context.has_failed() {
                return Vec::new();
            }
            self.context.record(self.read_children(), Vec::new())
        })
    }

    fn rename_bound_identifier(&self, old: &Identifier, new: Identifier) -> PyResult<Self> {
        self.context.check()?;
        let py = self.object.py();
        let new_object = identifier_to_python(py, &new)?;
        self.context.remember(&new, &new_object);
        let result = self.object.call_method1(
            intern!(py, "rename_bound_identifier"),
            (self.context.object_of(old)?, new_object),
        )?;
        Ok(Self::new(result, &self.context))
    }

    fn rebuild_with_scoped_children(&self, children: Vec<PyTerm<'py>>) -> PyResult<Self> {
        self.context.check()?;
        let py = self.object.py();
        let children = PyList::new(py, children.iter().map(PyTerm::object))?;
        let result = self
            .object
            .call_method1(intern!(py, "rebuild_with_scoped_children"), (children,))?;
        Ok(Self::new(result, &self.context))
    }
}
