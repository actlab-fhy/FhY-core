//! The registry the Python API reads and writes (N-S7-2 (a)), and the
//! built-in entries it resolves first (N-S7-3 (a)).
//!
//! The user registry is one [`FunctionRegistry`] with the Python object of
//! each entry, held as an `Arc` behind a `Mutex` in the extension's module
//! state (D-S7-13). A lookup locks only to read the current state; a
//! registration builds its entry with no lock held, then, under the lock,
//! clones the state, registers into the clone, and swaps it in. So the
//! screen and the inliner run on one consistent snapshot with no lock held,
//! whatever other threads register meanwhile, and no lock is ever held
//! across a call into Python: the old state is dropped after the lock is
//! released, since dropping it may run Python finalizers. The state is not
//! append-only: `set_registry_state_for_tests` replaces it.
//!
//! The built-in entries are built once, when `builtins.py` installs them,
//! and never change.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyMapping, PyType};

use fhy_core::expression::builtins::{BuiltinConstant, BuiltinFunction};
use fhy_core::expression::registry::{FunctionRegistry, RegistryEntry};

use crate::identifier::identifier_to_python;

use super::super::evaluate::builtin_implementation;
use super::entries::{PyNativeConstant, PyNativeFunction, PyRegisteredFunction};

/// One state of the user registry: the core registry and the Python object
/// of each entry. A state never changes once it is shared.
pub(crate) struct RegistryState {
    /// The core registry.
    registry: FunctionRegistry,
    /// Each entry's Python object, by name.
    objects: HashMap<String, Py<PyAny>>,
    /// Each constant's Python object, by the id of its identifier.
    constants_by_id: HashMap<u64, Py<PyAny>>,
    /// Each constant's Python identifier, by the constant's name, built on
    /// the first request. The cell is shared by every later state that
    /// keeps the constant, so the identifier is one object.
    identifiers: HashMap<String, Arc<PyOnceLock<Py<PyAny>>>>,
    /// The mapping `get_registered_entries` returns, built on the first
    /// request for this state.
    entries_view: PyOnceLock<Py<PyAny>>,
}

impl RegistryState {
    /// Return the empty state.
    fn new() -> Self {
        Self {
            registry: FunctionRegistry::new(),
            objects: HashMap::new(),
            constants_by_id: HashMap::new(),
            identifiers: HashMap::new(),
            entries_view: PyOnceLock::new(),
        }
    }

    /// Return a copy of this state to register into.
    fn draft(&self, py: Python<'_>) -> Self {
        Self {
            registry: self.registry.clone(),
            objects: self
                .objects
                .iter()
                .map(|(name, object)| (name.clone(), object.clone_ref(py)))
                .collect(),
            constants_by_id: self
                .constants_by_id
                .iter()
                .map(|(id, object)| (*id, object.clone_ref(py)))
                .collect(),
            identifiers: self.identifiers.clone(),
            entries_view: PyOnceLock::new(),
        }
    }

    /// Return the core registry.
    pub(crate) fn registry(&self) -> &FunctionRegistry {
        &self.registry
    }

    /// Return the Python object of the entry named `name`.
    pub(super) fn object<'py>(&self, py: Python<'py>, name: &str) -> Option<Bound<'py, PyAny>> {
        self.objects.get(name).map(|object| object.bind(py).clone())
    }

    /// Return the Python callable of the native user function named
    /// `name`, or `None` if no native user function has the name.
    pub(in crate::expression) fn native_implementation<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> Option<Bound<'py, PyAny>> {
        let object = self.objects.get(name)?.bind(py);
        let function = object.cast::<PyNativeFunction>().ok()?;
        Some(function.get().implementation(py))
    }

    /// Return the Python object of the constant whose identifier has `id`.
    pub(super) fn constant_object<'py>(
        &self,
        py: Python<'py>,
        id: u64,
    ) -> Option<Bound<'py, PyAny>> {
        self.constants_by_id
            .get(&id)
            .map(|object| object.bind(py).clone())
    }

    /// Return the Python identifier of the constant named `name`, or `None`
    /// if no constant has the name.
    pub(super) fn constant_identifier<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let (Some(cell), Some(identifier)) = (
            self.identifiers.get(name),
            self.registry.constant_identifier(name),
        ) else {
            return Ok(None);
        };
        cell.get_or_try_init(py, || {
            identifier_to_python(py, identifier).map(Bound::unbind)
        })
        .map(|object| Some(object.bind(py).clone()))
    }

    /// Register the entry `object` of the class `PyRegisteredFunction`,
    /// `PyNativeFunction` or `PyNativeConstant` into this draft.
    ///
    /// # Errors
    ///
    /// Raises `TypeError` for any other object or a built-in's entry, and
    /// `EntryRegistrationError` with the core's text for a refused one.
    fn register(&mut self, py: Python<'_>, object: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(function) = object.cast::<PyRegisteredFunction>() {
            if let Some(definition) = function.get().definition() {
                let name = definition.name().as_str().to_owned();
                self.registry
                    .register_function(definition.clone())
                    .map_err(|error| super::lookups::registration_error(py, &error.to_string()))?;
                self.objects.insert(name, object.clone().unbind());
                return Ok(());
            }
        } else if let Ok(function) = object.cast::<PyNativeFunction>() {
            if let Some(declaration) = function.get().declaration() {
                let name = declaration.name().as_str().to_owned();
                self.registry
                    .register_native_function(declaration.clone())
                    .map_err(|error| super::lookups::registration_error(py, &error.to_string()))?;
                self.objects.insert(name, object.clone().unbind());
                return Ok(());
            }
        } else if let Ok(constant) = object.cast::<PyNativeConstant>() {
            if let Some(declaration) = constant.get().declaration() {
                let name = declaration.name().as_str().to_owned();
                let identifier = self
                    .registry
                    .register_constant(declaration.clone())
                    .map_err(|error| super::lookups::registration_error(py, &error.to_string()))?;
                self.objects.insert(name.clone(), object.clone().unbind());
                self.constants_by_id
                    .insert(identifier.id(), object.clone().unbind());
                self.identifiers.insert(name, Arc::new(PyOnceLock::new()));
                return Ok(());
            }
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "only a user RegisteredFunction, NativeFunction or NativeConstant registers, got {}.",
            object.repr()?
        )))
    }

    /// Keep only the entries whose name `keep` accepts, each constant with
    /// its identifier.
    fn retain(&mut self, py: Python<'_>, keep: impl Fn(&str) -> bool) {
        self.registry.retain(|entry| keep(entry.name().as_str()));
        let registry = &self.registry;
        self.objects.retain(|name, _| registry.contains(name));
        self.identifiers.retain(|name, _| registry.contains(name));
        let objects = &self.objects;
        self.constants_by_id = registry
            .iter()
            .filter_map(|entry| match entry {
                RegistryEntry::Constant(constant, identifier) => objects
                    .get(constant.name().as_str())
                    .map(|object| (identifier.id(), object.clone_ref(py))),
                _ => None,
            })
            .collect();
    }

    /// Return the mapping of every entry's name to its object: the
    /// built-ins in catalogue order, the constants, then the composed
    /// functions, then the native ones (D-S7-16), and then the user
    /// entries in registration order.
    pub(super) fn entries_view<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.entries_view
            .get_or_try_init(py, || {
                let entries = PyDict::new(py);
                for (name, object) in &builtins(py)?.in_order {
                    entries.set_item(name, object.bind(py))?;
                }
                for entry in self.registry.iter() {
                    let name = entry.name().as_str();
                    if let Some(object) = self.objects.get(name) {
                        entries.set_item(name, object.bind(py))?;
                    }
                }
                let class =
                    crate::python::cached_attr!(py, "immutabledict", "immutabledict" => PyType)?;
                class.call1((entries,)).map(Bound::unbind)
            })
            .map(|view| view.bind(py).clone())
    }
}

/// The current state of the user registry.
static STATE: LazyLock<Mutex<Arc<RegistryState>>> =
    LazyLock::new(|| Mutex::new(Arc::new(RegistryState::new())));

/// Lock the current state. A thread that panicked while holding the lock
/// left a whole state in place, since states are swapped whole.
fn lock() -> MutexGuard<'static, Arc<RegistryState>> {
    STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Return the current state, a snapshot no registration changes.
pub(crate) fn snapshot() -> Arc<RegistryState> {
    Arc::clone(&lock())
}

/// Register the entry `object` into the current state.
///
/// # Errors
///
/// Raises what [`RegistryState::register`] raises, leaving the state as it
/// was.
pub(super) fn register(py: Python<'_>, object: &Bound<'_, PyAny>) -> PyResult<()> {
    let mut guard = lock();
    let mut draft = guard.draft(py);
    let result = draft.register(py, object);
    let replaced = result
        .is_ok()
        .then(|| std::mem::replace(&mut *guard, Arc::new(draft)));
    drop(guard);
    drop(replaced);
    result
}

/// Replace the current state by the entries of `state`, a mapping of names
/// to entry objects, as `set_registry_state_for_tests` documents.
///
/// # Errors
///
/// Raises what registering an entry raises, leaving the state as it was.
pub(super) fn restore(py: Python<'_>, state: &Bound<'_, PyMapping>) -> PyResult<()> {
    let builtin_entries = builtins(py)?;
    let mut entries: Vec<(String, Bound<'_, PyAny>)> = Vec::new();
    for item in state.items()?.iter() {
        let (name, object) = item.extract::<(String, Bound<'_, PyAny>)>()?;
        if !builtin_entries.by_name.contains_key(name.as_str()) {
            entries.push((name, object));
        }
    }
    let current = snapshot();
    let kept: std::collections::HashSet<&str> = entries
        .iter()
        .filter(|(name, object)| {
            current
                .objects
                .get(name)
                .is_some_and(|registered| registered.bind(py).is(object))
        })
        .map(|(name, _)| name.as_str())
        .collect();
    let mut draft = current.draft(py);
    draft.retain(py, |name| kept.contains(name));
    for (name, object) in &entries {
        if !kept.contains(name.as_str()) {
            draft.register(py, object)?;
        }
    }
    let replaced = std::mem::replace(&mut *lock(), Arc::new(draft));
    drop(replaced);
    drop(current);
    Ok(())
}

// ---------------------------------------------------------------------------
// Built-ins
// ---------------------------------------------------------------------------

/// The entry objects of the built-ins, built once.
pub(super) struct BuiltinEntries {
    /// Each built-in's entry, by name.
    by_name: HashMap<&'static str, Py<PyAny>>,
    /// The entries in catalogue order: the constants, the composed
    /// functions, then the native ones.
    in_order: Vec<(&'static str, Py<PyAny>)>,
    /// The constants' Python identifiers, in catalogue order.
    constant_identifiers: Vec<(BuiltinConstant, Py<PyAny>, Py<PyAny>)>,
}

impl BuiltinEntries {
    /// Return the entry of the built-in named `name`.
    pub(super) fn object<'py>(&self, py: Python<'py>, name: &str) -> Option<Bound<'py, PyAny>> {
        self.by_name.get(name).map(|object| object.bind(py).clone())
    }

    /// Return whether a built-in is named `name`.
    pub(super) fn contains(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    /// Return the Python identifier of the built-in constant named `name`.
    pub(super) fn constant_identifier<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> Option<Bound<'py, PyAny>> {
        self.constant_identifiers
            .iter()
            .find(|(constant, _, _)| constant.name() == name)
            .map(|(_, identifier, _)| identifier.bind(py).clone())
    }

    /// Return the entry of the built-in constant whose identifier has `id`.
    pub(super) fn constant_object<'py>(
        &self,
        py: Python<'py>,
        id: u64,
    ) -> Option<Bound<'py, PyAny>> {
        self.constant_identifiers
            .iter()
            .find(|(constant, _, _)| constant.identifier().id() == id)
            .map(|(_, _, object)| object.bind(py).clone())
    }
}

/// The built-in entries, once installed.
static BUILTINS: PyOnceLock<BuiltinEntries> = PyOnceLock::new();

/// Return the built-in entries.
///
/// # Errors
///
/// Raises `RuntimeError` if `builtins.py` has not installed them yet.
pub(super) fn builtins(py: Python<'_>) -> PyResult<&BuiltinEntries> {
    BUILTINS.get(py).ok_or_else(|| {
        PyRuntimeError::new_err(
            "the built-in function entries are not installed: import \
             fhy_core.symbolic.expression first",
        )
    })
}

/// Return the entry of the built-in named `name`, or `None`.
///
/// # Errors
///
/// Raises `RuntimeError` if the built-ins are not installed.
pub(super) fn find_builtin<'py>(
    py: Python<'py>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    Ok(builtins(py)?.object(py, name))
}

/// Build the built-in entries, unless they are built already, each native
/// built-in computed by its `BuiltinNativeImplementation`.
///
/// # Errors
///
/// Raises whatever building an entry raises.
pub(super) fn install_builtins(py: Python<'_>) -> PyResult<()> {
    BUILTINS
        .get_or_try_init(py, || {
            let mut in_order: Vec<(&'static str, Py<PyAny>)> = Vec::new();
            let mut constant_identifiers = Vec::new();
            for constant in BuiltinConstant::iter() {
                let object = PyNativeConstant::build_builtin(py, constant)?;
                let identifier = identifier_to_python(py, constant.identifier())?;
                constant_identifiers.push((constant, identifier.unbind(), object.clone().unbind()));
                in_order.push((constant.name(), object.unbind()));
            }
            for function in BuiltinFunction::iter() {
                let object = if let Some(composed) = function.composed() {
                    PyRegisteredFunction::build_builtin(py, composed)?
                } else {
                    let implementation = builtin_implementation(py, function)?.into_any();
                    PyNativeFunction::build_builtin(py, function, &implementation)?
                };
                in_order.push((function.name(), object.unbind()));
            }
            let by_name = in_order
                .iter()
                .map(|(name, object)| (*name, object.clone_ref(py)))
                .collect();
            Ok::<_, PyErr>(BuiltinEntries {
                by_name,
                in_order,
                constant_identifiers,
            })
        })
        .map(|_| ())
}
