//! Python-defined types as extensions of the core (D-S11-9), and the
//! context each call into the core runs in.
//!
//! A `Type` or `DataType` subclass that Python defines reaches the core as a
//! [`PyTypeAdapter`] or [`PyDataTypeAdapter`] over the object. For each hook
//! the adapter asks the dispatcher of `fhy_core.types.dispatch` which handler
//! serves the object's class: when it is the dispatcher's default, no
//! handler was registered, and the adapter answers `None`, so the core
//! applies its default rule without calling Python; otherwise it calls the
//! handler with the Python objects and converts its result back.
//!
//! The core's equality, hashing and structural equivalence are infallible,
//! while a Python handler or `==` can raise. So every entry function that
//! runs the core over such values runs in a [`Context`]: the first exception
//! an infallible hook meets is kept there, the hook answers `false`, and the
//! entry raises it when the core returns. The context also carries the
//! Python objects the call was given, so the values the core returns are
//! handed back as those objects where they are unchanged, and the class of
//! the environment the call was given, which every environment it builds
//! keeps (D-S11-13). A thread-local stack holds the contexts of the calls in
//! progress, so a nested call, from a handler, gets its own; it is empty
//! whenever no call runs.

use std::any::Any;
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hasher;
use std::rc::Rc;

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyTuple, PyType};

use fhy_core::expression::pattern::CallbackError;
use fhy_core::tree::NodeIdentity;
use fhy_core::types::{
    DataType, DataTypeExtension, Type, TypeExtension, TypeUnificationEnvironment, UnificationError,
};

use super::convert;
use super::environment::PyTypeUnificationEnvironment;

/// The Python objects of the values a call has seen, by the values' Rust
/// identity.
#[derive(Default)]
pub(super) struct Known {
    /// Types and their objects.
    pub(super) types: Vec<(Type, Py<PyAny>)>,
    /// Data types and their objects.
    pub(super) data_types: Vec<(DataType, Py<PyAny>)>,
    /// Expressions and their objects, by node identity.
    pub(super) expressions: HashMap<NodeIdentity, Py<PyAny>>,
    /// Identifier objects, by id.
    pub(super) identifiers: HashMap<u64, Py<PyAny>>,
}

/// What one call into the core shares with the adapters it drives.
pub(crate) struct Context {
    /// The objects of the values the call has seen.
    pub(super) known: RefCell<Known>,
    /// The environment the call was given, whose class and extra
    /// attributes every environment it builds keeps.
    pub(super) template: Option<Py<PyTypeUnificationEnvironment>>,
    /// The first exception an infallible hook met.
    pending: RefCell<Option<PyErr>>,
}

impl Context {
    /// Keep `error` if it is the first.
    fn fail(&self, error: PyErr) {
        let mut pending = self.pending.borrow_mut();
        if pending.is_none() {
            *pending = Some(error);
        }
    }

    /// Return whether an infallible hook has met an exception.
    fn has_failed(&self) -> bool {
        self.pending.borrow().is_some()
    }
}

thread_local! {
    /// The contexts of the calls in progress, innermost last.
    static CONTEXTS: RefCell<Vec<Rc<Context>>> = const { RefCell::new(Vec::new()) };
}

/// Run `body` in a new context over the environment `template`, and raise
/// the first exception an infallible hook met, if any, in place of its
/// result.
pub(crate) fn run_in_context<R>(
    py: Python<'_>,
    template: Option<&Bound<'_, PyTypeUnificationEnvironment>>,
    body: impl FnOnce(&Rc<Context>) -> PyResult<R>,
) -> PyResult<R> {
    let context = Rc::new(Context {
        known: RefCell::new(Known::default()),
        template: template.map(|template| template.clone().unbind()),
        pending: RefCell::new(None),
    });
    let _ = py;
    CONTEXTS.with(|contexts| contexts.borrow_mut().push(Rc::clone(&context)));
    let result = body(&context);
    CONTEXTS.with(|contexts| contexts.borrow_mut().pop());
    match context.pending.borrow_mut().take() {
        Some(error) => Err(error),
        None => result,
    }
}

/// Return the context of the innermost call in progress, or a new one when
/// none runs.
fn current_context() -> Rc<Context> {
    CONTEXTS
        .with(|contexts| contexts.borrow().last().cloned())
        .unwrap_or_else(|| {
            Rc::new(Context {
                known: RefCell::new(Known::default()),
                template: None,
                pending: RefCell::new(None),
            })
        })
}

// ---------------------------------------------------------------------------
// The dispatchers
// ---------------------------------------------------------------------------

/// The six dispatchers of `fhy_core.types.dispatch`.
struct Dispatchers {
    is_structurally_equivalent: Py<PyAny>,
    bind_template: Py<PyAny>,
    substitute_template: Py<PyAny>,
    unify: Py<PyAny>,
    bind_data_template: Py<PyAny>,
    substitute_data_template: Py<PyAny>,
}

/// Which dispatcher a hook asks.
#[derive(Clone, Copy)]
enum Hook {
    IsStructurallyEquivalent,
    BindTemplate,
    SubstituteTemplate,
    Unify,
    BindDataTemplate,
    SubstituteDataTemplate,
}

fn dispatchers(py: Python<'_>) -> PyResult<&Dispatchers> {
    static DISPATCHERS: PyOnceLock<Dispatchers> = PyOnceLock::new();
    DISPATCHERS.get_or_try_init(py, || {
        let module = py.import("fhy_core.types.dispatch")?;
        let get = |name: &str| module.getattr(name).map(Bound::unbind);
        Ok(Dispatchers {
            is_structurally_equivalent: get("is_structurally_equivalent")?,
            bind_template: get("bind_template")?,
            substitute_template: get("substitute_template")?,
            unify: get("unify")?,
            bind_data_template: get("bind_data_template")?,
            substitute_data_template: get("substitute_data_template")?,
        })
    })
}

/// Return the handler a user registered on the dispatcher of `hook` for the
/// class of `object`, or `None` when the dispatcher would run its default.
fn user_handler<'py>(
    object: &Bound<'py, PyAny>,
    hook: Hook,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = object.py();
    let dispatchers = dispatchers(py)?;
    let dispatcher = match hook {
        Hook::IsStructurallyEquivalent => &dispatchers.is_structurally_equivalent,
        Hook::BindTemplate => &dispatchers.bind_template,
        Hook::SubstituteTemplate => &dispatchers.substitute_template,
        Hook::Unify => &dispatchers.unify,
        Hook::BindDataTemplate => &dispatchers.bind_data_template,
        Hook::SubstituteDataTemplate => &dispatchers.substitute_data_template,
    }
    .bind(py);
    let handler = dispatcher.call_method1("dispatch", (object.get_type(),))?;
    let default = dispatcher
        .getattr("registry")?
        .get_item(py.get_type::<PyAny>())?;
    Ok((!handler.is(&default)).then_some(handler))
}

/// Box the exception of a handler as the core's extension error.
fn extension_error(error: PyErr) -> UnificationError {
    let boxed: CallbackError = Box::new(error);
    UnificationError::Extension(boxed)
}

/// Return the `TypeError` of a handler that returned `result`, which is not
/// `expected`.
fn build_result_type_error(
    object: &Bound<'_, PyAny>,
    hook: &str,
    expected: &str,
    result: &Bound<'_, PyAny>,
) -> PyErr {
    let class = object
        .get_type()
        .name()
        .map_or_else(|_| "?".to_owned(), |name| name.to_string());
    let got = result
        .get_type()
        .name()
        .map_or_else(|_| "?".to_owned(), |name| name.to_string());
    pyo3::exceptions::PyTypeError::new_err(format!(
        "{hook} handler for {class} must return {expected}, got {got}."
    ))
}

/// Return `str(object)`, or the class name if that raises.
fn write_object(f: &mut fmt::Formatter<'_>, object: &Py<PyAny>) -> fmt::Result {
    let text = Python::attach(|py| {
        let object = object.bind(py);
        object
            .str()
            .map(|text| text.to_string())
            .or_else(|_| object.get_type().name().map(|name| name.to_string()))
            .unwrap_or_else(|_| "?".to_owned())
    });
    f.write_str(&text)
}

/// Return `type(object).__name__`.
fn class_name(object: &Py<PyAny>) -> Cow<'static, str> {
    Python::attach(|py| {
        object
            .bind(py)
            .get_type()
            .name()
            .map_or_else(|_| Cow::Borrowed("?"), |name| Cow::Owned(name.to_string()))
    })
}

/// Return whether the objects `left` and `right` are `==`, keeping an
/// exception in the context.
fn objects_equal(left: &Py<PyAny>, right: &Py<PyAny>) -> bool {
    Python::attach(|py| {
        let (left, right) = (left.bind(py), right.bind(py));
        if left.is(right) {
            return true;
        }
        let context = current_context();
        if context.has_failed() {
            return false;
        }
        left.eq(right).unwrap_or_else(|error| {
            context.fail(error);
            false
        })
    })
}

/// Feed `hash(object)` to `state`, keeping an exception in the context.
fn hash_object(object: &Py<PyAny>, state: &mut dyn Hasher) {
    Python::attach(|py| {
        let context = current_context();
        match object.bind(py).hash() {
            Ok(hash) => state.write_isize(hash),
            Err(error) => context.fail(error),
        }
    });
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A Python-defined `Type` as a core type extension.
#[derive(Debug)]
pub(crate) struct PyTypeAdapter {
    object: Py<PyAny>,
}

impl PyTypeAdapter {
    /// Return the adapter over `object`.
    pub(crate) fn new(object: Py<PyAny>) -> Self {
        Self { object }
    }

    /// Return the Python object.
    pub(crate) fn object(&self) -> &Py<PyAny> {
        &self.object
    }
}

impl fmt::Display for PyTypeAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_object(f, &self.object)
    }
}

impl TypeExtension for PyTypeAdapter {
    fn type_name(&self) -> Cow<'_, str> {
        class_name(&self.object)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn is_structurally_equivalent(&self, other: &Type) -> bool {
        Python::attach(|py| {
            let context = current_context();
            if context.has_failed() {
                return false;
            }
            let result = (|| -> PyResult<bool> {
                let object = self.object.bind(py);
                let Some(handler) = user_handler(object, Hook::IsStructurallyEquivalent)? else {
                    return Ok(false);
                };
                let other = convert::type_to_python(py, &context, other)?;
                handler.call1((object, other))?.is_truthy()
            })();
            result.unwrap_or_else(|error| {
                context.fail(error);
                false
            })
        })
    }

    fn eq_extension(&self, other: &dyn TypeExtension) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| objects_equal(&self.object, &other.object))
    }

    fn hash_extension(&self, state: &mut dyn Hasher) {
        hash_object(&self.object, state);
    }

    fn bind_template(
        &self,
        actual: &Type,
        environment: &TypeUnificationEnvironment,
    ) -> Option<Result<TypeUnificationEnvironment, UnificationError>> {
        Python::attach(|py| {
            let context = current_context();
            let object = self.object.bind(py);
            let handler = match user_handler(object, Hook::BindTemplate) {
                Ok(Some(handler)) => handler,
                Ok(None) => return None,
                Err(error) => return Some(Err(extension_error(error))),
            };
            let result = (|| -> PyResult<TypeUnificationEnvironment> {
                let actual = convert::type_to_python(py, &context, actual)?;
                let environment = convert::environment_to_python(py, &context, environment)?;
                let result = handler.call1((object, actual, environment))?;
                convert::read_environment(&context, &result).ok_or_else(|| {
                    build_result_type_error(
                        object,
                        "bind_template",
                        "a TypeUnificationEnvironment",
                        &result,
                    )
                })
            })();
            Some(result.map_err(extension_error))
        })
    }

    fn substitute_template(
        &self,
        environment: &TypeUnificationEnvironment,
    ) -> Option<Result<Type, UnificationError>> {
        Python::attach(|py| {
            let context = current_context();
            let object = self.object.bind(py);
            let handler = match user_handler(object, Hook::SubstituteTemplate) {
                Ok(Some(handler)) => handler,
                Ok(None) => return None,
                Err(error) => return Some(Err(extension_error(error))),
            };
            let result = (|| -> PyResult<Type> {
                let environment = convert::environment_to_python(py, &context, environment)?;
                let result = handler.call1((object, environment))?;
                convert::read_type(&context, &result)?.ok_or_else(|| {
                    build_result_type_error(object, "substitute_template", "a Type", &result)
                })
            })();
            Some(result.map_err(extension_error))
        })
    }

    fn unify(
        &self,
        actual: &Type,
        environment: &TypeUnificationEnvironment,
    ) -> Option<Result<(Type, TypeUnificationEnvironment), UnificationError>> {
        Python::attach(|py| {
            let context = current_context();
            let object = self.object.bind(py);
            let handler = match user_handler(object, Hook::Unify) {
                Ok(Some(handler)) => handler,
                Ok(None) => return None,
                Err(error) => return Some(Err(extension_error(error))),
            };
            let result = (|| -> PyResult<(Type, TypeUnificationEnvironment)> {
                let actual = convert::type_to_python(py, &context, actual)?;
                let environment = convert::environment_to_python(py, &context, environment)?;
                let result = handler.call1((object, actual, environment))?;
                let wrong = || {
                    build_result_type_error(
                        object,
                        "unify",
                        "a (Type, TypeUnificationEnvironment) tuple",
                        &result,
                    )
                };
                let pair = result.cast::<PyTuple>().map_err(|_not_a_tuple| wrong())?;
                if pair.len() != 2 {
                    return Err(wrong());
                }
                let unified =
                    convert::read_type(&context, &pair.get_item(0)?)?.ok_or_else(wrong)?;
                let environment =
                    convert::read_environment(&context, &pair.get_item(1)?).ok_or_else(wrong)?;
                Ok((unified, environment))
            })();
            Some(result.map_err(extension_error))
        })
    }
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A Python-defined `DataType` as a core data-type extension.
#[derive(Debug)]
pub(crate) struct PyDataTypeAdapter {
    object: Py<PyAny>,
}

impl PyDataTypeAdapter {
    /// Return the adapter over `object`.
    pub(crate) fn new(object: Py<PyAny>) -> Self {
        Self { object }
    }

    /// Return the Python object.
    pub(crate) fn object(&self) -> &Py<PyAny> {
        &self.object
    }
}

impl fmt::Display for PyDataTypeAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_object(f, &self.object)
    }
}

impl DataTypeExtension for PyDataTypeAdapter {
    fn type_name(&self) -> Cow<'_, str> {
        class_name(&self.object)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn is_structurally_equivalent(&self, other: &DataType) -> bool {
        Python::attach(|py| {
            let context = current_context();
            if context.has_failed() {
                return false;
            }
            let result = (|| -> PyResult<bool> {
                let object = self.object.bind(py);
                let Some(handler) = user_handler(object, Hook::IsStructurallyEquivalent)? else {
                    return Ok(false);
                };
                let other = convert::data_type_to_python(py, &context, other)?;
                handler.call1((object, other))?.is_truthy()
            })();
            result.unwrap_or_else(|error| {
                context.fail(error);
                false
            })
        })
    }

    fn eq_extension(&self, other: &dyn DataTypeExtension) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| objects_equal(&self.object, &other.object))
    }

    fn hash_extension(&self, state: &mut dyn Hasher) {
        hash_object(&self.object, state);
    }

    fn bind_template(
        &self,
        actual: &DataType,
        environment: &TypeUnificationEnvironment,
    ) -> Option<Result<TypeUnificationEnvironment, UnificationError>> {
        Python::attach(|py| {
            let context = current_context();
            let object = self.object.bind(py);
            let handler = match user_handler(object, Hook::BindDataTemplate) {
                Ok(Some(handler)) => handler,
                Ok(None) => return None,
                Err(error) => return Some(Err(extension_error(error))),
            };
            let result = (|| -> PyResult<TypeUnificationEnvironment> {
                let actual = convert::data_type_to_python(py, &context, actual)?;
                let environment = convert::environment_to_python(py, &context, environment)?;
                let result = handler.call1((object, actual, environment))?;
                convert::read_environment(&context, &result).ok_or_else(|| {
                    build_result_type_error(
                        object,
                        "bind_data_template",
                        "a TypeUnificationEnvironment",
                        &result,
                    )
                })
            })();
            Some(result.map_err(extension_error))
        })
    }

    fn substitute_template(
        &self,
        environment: &TypeUnificationEnvironment,
    ) -> Option<Result<DataType, UnificationError>> {
        Python::attach(|py| {
            let context = current_context();
            let object = self.object.bind(py);
            let handler = match user_handler(object, Hook::SubstituteDataTemplate) {
                Ok(Some(handler)) => handler,
                Ok(None) => return None,
                Err(error) => return Some(Err(extension_error(error))),
            };
            let result = (|| -> PyResult<DataType> {
                let environment = convert::environment_to_python(py, &context, environment)?;
                let result = handler.call1((object, environment))?;
                convert::read_data_type(&context, &result)?.ok_or_else(|| {
                    build_result_type_error(
                        object,
                        "substitute_data_template",
                        "a DataType",
                        &result,
                    )
                })
            })();
            Some(result.map_err(extension_error))
        })
    }
}

/// Return the class environments built in `context` take: the class of its
/// template, or the public `TypeUnificationEnvironment`.
pub(super) fn environment_class<'py>(
    py: Python<'py>,
    context: &Context,
) -> PyResult<Bound<'py, PyType>> {
    match &context.template {
        Some(template) => Ok(template.bind(py).get_type()),
        None => Ok(PyTypeUnificationEnvironment::public_class()
            .get(py)?
            .clone()),
    }
}
