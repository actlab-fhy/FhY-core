//! The registry of the `Variable`, `Alternative` and oracle kinds a
//! downstream Rust crate defines: append-only module state of the
//! extension.
//!
//! A downstream binding crate registers each of its kinds once, from its
//! aggregate's `#[pymodule]`, with the functions of
//! [`convert::search_space`](crate::convert::search_space): the kind (the
//! type id its parts write), its `#[pyclass]`, and the three functions that
//! read an object of the class, build the object of a part, and resolve a
//! foreign part of the kind.
//!
//! The registry is the attribute [`KIND_REGISTRY_ATTRIBUTE`] of the module
//! `register` built, a [`PyKindRegistry`] holding one version of the state
//! behind a lock. A registration swaps in a new version; a lookup takes the
//! current one and releases the lock before it calls Python. Entries are
//! never replaced or removed. No Rust `static` holds a kind: the binding
//! reaches the registry through a write-once import cache of the module
//! attribute, as it does the verification registry.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;

use fhy_core::search_space::{PlainAlternative, PlainVariable};

use crate::convert::search_space::{
    AlternativeFromPython, AlternativeResolver, AlternativeToPython, OracleLease,
    VariableFromPython, VariableResolver, VariableToPython,
};

/// The attribute of `fhy_core._rs` that holds the kind registry.
pub(crate) const KIND_REGISTRY_ATTRIBUTE: &str = "_search_space_kinds";

/// One registered kind of a family: its class and its three functions.
pub(super) struct KindEntry<FromPython, ToPython, Resolver> {
    pub(super) class: Py<PyType>,
    pub(super) from_python: FromPython,
    pub(super) to_python: ToPython,
    pub(super) resolve: Resolver,
}

/// A registered `Variable` kind.
pub(super) type VariableKind = KindEntry<VariableFromPython, VariableToPython, VariableResolver>;

/// A registered `Alternative` kind.
pub(super) type AlternativeKind =
    KindEntry<AlternativeFromPython, AlternativeToPython, AlternativeResolver>;

/// A registered oracle kind: its class and its lease.
pub(super) struct OracleKind {
    pub(super) class: Py<PyType>,
    pub(super) lease: OracleLease,
}

/// One version of the registry.
#[derive(Default)]
pub(super) struct KindRegistryState {
    variables: HashMap<String, Arc<VariableKind>>,
    alternatives: HashMap<String, Arc<AlternativeKind>>,
    oracles: HashMap<String, Arc<OracleKind>>,
}

impl KindRegistryState {
    /// Return the entry of the `Variable` kind `kind`, if registered.
    pub(super) fn variable(&self, kind: &str) -> Option<&Arc<VariableKind>> {
        self.variables.get(kind)
    }

    /// Return the entry of the `Alternative` kind `kind`, if registered.
    pub(super) fn alternative(&self, kind: &str) -> Option<&Arc<AlternativeKind>> {
        self.alternatives.get(kind)
    }

    /// Return the registered `Variable` kinds' entries.
    pub(super) fn variables(&self) -> impl Iterator<Item = &Arc<VariableKind>> {
        self.variables.values()
    }

    /// Return the registered `Alternative` kinds' entries.
    pub(super) fn alternatives(&self) -> impl Iterator<Item = &Arc<AlternativeKind>> {
        self.alternatives.values()
    }

    /// Return the registered oracle kinds' entries.
    pub(super) fn oracles(&self) -> impl Iterator<Item = &Arc<OracleKind>> {
        self.oracles.values()
    }

    /// Return whether `class` is registered for a kind of any family.
    fn holds_class(&self, class: &Bound<'_, PyType>) -> bool {
        self.variables
            .values()
            .any(|entry| entry.class.bind(class.py()).is(class))
            || self
                .alternatives
                .values()
                .any(|entry| entry.class.bind(class.py()).is(class))
            || self
                .oracles
                .values()
                .any(|entry| entry.class.bind(class.py()).is(class))
    }
}

/// The kind registry of `fhy_core.search_space`, the module state the
/// downstream kinds are registered in.
#[pyclass(frozen, module = "fhy_core._rs", name = "_SearchSpaceKindRegistry")]
pub(crate) struct PyKindRegistry {
    state: Mutex<Arc<KindRegistryState>>,
}

impl PyKindRegistry {
    /// Return an empty registry.
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(Arc::new(KindRegistryState::default())),
        }
    }

    /// Return the current version of the registry.
    pub(super) fn current(&self) -> Arc<KindRegistryState> {
        Arc::clone(&self.state.lock().unwrap_or_else(PoisonError::into_inner))
    }

    /// Replace the current version with what `change` makes of a copy of
    /// it, unless `change` refuses.
    ///
    /// `change` runs under the lock, so it calls no Python: it returns the
    /// [`Refusal`], which the caller words once the lock is released.
    fn update(
        &self,
        change: impl FnOnce(&KindRegistryState) -> Result<KindRegistryState, Refusal>,
    ) -> Result<(), Refusal> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let changed = change(&state)?;
        *state = Arc::new(changed);
        Ok(())
    }
}

/// Why a registration was refused, decided under the registry's lock and
/// worded after it is released.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Refusal {
    /// The kind is the family's built-in plain kind.
    BuiltIn,
    /// The kind is registered already.
    KindRegistered,
    /// The class is registered for a kind already.
    ClassRegistered,
}

impl Refusal {
    /// Return the error of the refusal to register the kind `kind` of
    /// `family` for `class`. Reading the class's name may call Python, so
    /// this is never called under the lock.
    fn into_error(self, family: &str, kind: &str, class: &Bound<'_, PyType>) -> PyErr {
        match self {
            Self::BuiltIn => refuse(
                family,
                kind,
                &format!("it is the plain {}'s kind", family.to_lowercase()),
            ),
            Self::KindRegistered => refuse(family, kind, "it is registered already"),
            Self::ClassRegistered => refuse_class(family, kind, class),
        }
    }
}

#[pymethods]
impl PyKindRegistry {
    /// Visit the classes the registry holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        crate::util::gc::traverse_locked(&self.state, |state| {
            for entry in state.variables.values() {
                visit.call(&entry.class)?;
            }
            for entry in state.alternatives.values() {
                visit.call(&entry.class)?;
            }
            for entry in state.oracles.values() {
                visit.call(&entry.class)?;
            }
            Ok(())
        })
    }
}

/// Return the registered `Variable` kind whose class `object` is an
/// instance of, if any.
///
/// # Errors
///
/// Raises what importing the registry or an `isinstance` check raises.
pub(super) fn variable_kind_of(object: &Bound<'_, PyAny>) -> PyResult<Option<Arc<VariableKind>>> {
    let state = registry(object.py())?.get().current();
    for entry in state.variables() {
        if object.is_instance(entry.class.bind(object.py()))? {
            return Ok(Some(Arc::clone(entry)));
        }
    }
    Ok(None)
}

/// Return the registered `Alternative` kind whose class `object` is an
/// instance of, if any.
///
/// # Errors
///
/// Raises what importing the registry or an `isinstance` check raises.
pub(super) fn alternative_kind_of(
    object: &Bound<'_, PyAny>,
) -> PyResult<Option<Arc<AlternativeKind>>> {
    let state = registry(object.py())?.get().current();
    for entry in state.alternatives() {
        if object.is_instance(entry.class.bind(object.py()))? {
            return Ok(Some(Arc::clone(entry)));
        }
    }
    Ok(None)
}

/// Return the registered `Variable` kind `kind`, if any.
///
/// # Errors
///
/// Raises what importing the registry raises.
pub(super) fn variable_kind(py: Python<'_>, kind: &str) -> PyResult<Option<Arc<VariableKind>>> {
    Ok(registry(py)?.get().current().variable(kind).cloned())
}

/// Return the registered `Alternative` kind `kind`, if any.
///
/// # Errors
///
/// Raises what importing the registry raises.
pub(super) fn alternative_kind(
    py: Python<'_>,
    kind: &str,
) -> PyResult<Option<Arc<AlternativeKind>>> {
    Ok(registry(py)?.get().current().alternative(kind).cloned())
}

/// Return the kind registry of `fhy_core._rs`.
///
/// # Errors
///
/// Raises what importing the module attribute raises.
pub(super) fn registry(py: Python<'_>) -> PyResult<&Bound<'_, PyKindRegistry>> {
    static REGISTRY: PyOnceLock<Py<PyKindRegistry>> = PyOnceLock::new();
    REGISTRY.import(py, "fhy_core._rs", KIND_REGISTRY_ATTRIBUTE)
}

/// Return the registry `module` holds.
///
/// # Errors
///
/// Raises `RuntimeError` if `module` holds no registry, which means
/// `fhy_core`'s binding was never registered into it.
fn registry_of<'py>(module: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyKindRegistry>> {
    let missing = || {
        PyRuntimeError::new_err(format!(
            "the module {} holds no fhy_core binding: register fhy_core's classes into it first",
            module
                .name()
                .map_or_else(|_| "?".to_owned(), |name| name.to_string())
        ))
    };
    let attribute = module
        .getattr(KIND_REGISTRY_ATTRIBUTE)
        .map_err(|_missing_attribute| missing())?;
    attribute
        .cast_into::<PyKindRegistry>()
        .map_err(|_not_a_registry| missing())
}

/// Return the error of a kind that cannot be registered.
fn refuse(family: &str, kind: &str, reason: &str) -> PyErr {
    PyValueError::new_err(format!(
        "the {family} kind {kind:?} cannot be registered: {reason}"
    ))
}

/// Return the error of a class registered for a kind already.
fn refuse_class(family: &str, kind: &str, class: &Bound<'_, PyType>) -> PyErr {
    let name = class
        .qualname()
        .map_or_else(|_| "?".to_owned(), |name| name.to_string());
    refuse(
        family,
        kind,
        &format!("the class {name} is registered for a kind already"),
    )
}

/// Register `class` as a virtual subclass of the public class `public`,
/// if `public` is registered.
fn register_virtual_subclass(
    public: Option<&Bound<'_, PyType>>,
    class: &Bound<'_, PyType>,
) -> PyResult<()> {
    if let Some(public) = public {
        public.call_method1(pyo3::intern!(class.py(), "register"), (class,))?;
    }
    Ok(())
}

/// Register the `Variable` kind `kind` of the downstream class `class` in
/// the registry of `module`.
///
/// # Errors
///
/// Raises `ValueError` for a built-in kind, a kind registered already or a
/// class registered already, `RuntimeError` for a module without
/// `fhy_core`'s binding, and what registering the virtual subclass raises.
pub(crate) fn register_variable_kind(
    module: &Bound<'_, PyModule>,
    kind: &str,
    class: &Bound<'_, PyType>,
    from_python: VariableFromPython,
    to_python: VariableToPython,
    resolve: VariableResolver,
) -> PyResult<()> {
    let registry = registry_of(module)?;
    let updated = registry.get().update(|state| {
        if kind == PlainVariable::KIND {
            return Err(Refusal::BuiltIn);
        }
        if state.variables.contains_key(kind) {
            return Err(Refusal::KindRegistered);
        }
        if state.holds_class(class) {
            return Err(Refusal::ClassRegistered);
        }
        let mut variables = state.variables.clone();
        variables.insert(
            kind.to_owned(),
            Arc::new(KindEntry {
                class: class.clone().unbind(),
                from_python,
                to_python,
                resolve,
            }),
        );
        Ok(KindRegistryState {
            variables,
            alternatives: state.alternatives.clone(),
            oracles: state.oracles.clone(),
        })
    });
    updated.map_err(|refusal| refusal.into_error("Variable", kind, class))?;
    register_virtual_subclass(super::variable::registered_public_class(module.py()), class)
}

/// Register the `Alternative` kind `kind` of the downstream class `class`
/// in the registry of `module`.
///
/// # Errors
///
/// As [`register_variable_kind`].
pub(crate) fn register_alternative_kind(
    module: &Bound<'_, PyModule>,
    kind: &str,
    class: &Bound<'_, PyType>,
    from_python: AlternativeFromPython,
    to_python: AlternativeToPython,
    resolve: AlternativeResolver,
) -> PyResult<()> {
    let registry = registry_of(module)?;
    let updated = registry.get().update(|state| {
        if kind == PlainAlternative::KIND {
            return Err(Refusal::BuiltIn);
        }
        if state.alternatives.contains_key(kind) {
            return Err(Refusal::KindRegistered);
        }
        if state.holds_class(class) {
            return Err(Refusal::ClassRegistered);
        }
        let mut alternatives = state.alternatives.clone();
        alternatives.insert(
            kind.to_owned(),
            Arc::new(KindEntry {
                class: class.clone().unbind(),
                from_python,
                to_python,
                resolve,
            }),
        );
        Ok(KindRegistryState {
            variables: state.variables.clone(),
            alternatives,
            oracles: state.oracles.clone(),
        })
    });
    updated.map_err(|refusal| refusal.into_error("Alternative", kind, class))?;
    register_virtual_subclass(
        super::alternative::registered_public_class(module.py()),
        class,
    )
}

/// Return the registered oracle kind whose class `object` is an instance
/// of, if any.
///
/// # Errors
///
/// Raises what importing the registry or an `isinstance` check raises.
pub(super) fn oracle_kind_of(object: &Bound<'_, PyAny>) -> PyResult<Option<Arc<OracleKind>>> {
    let state = registry(object.py())?.get().current();
    for entry in state.oracles() {
        if object.is_instance(entry.class.bind(object.py()))? {
            return Ok(Some(Arc::clone(entry)));
        }
    }
    Ok(None)
}

/// Register the oracle kind `kind` of the downstream class `class`, whose
/// instances `lease` borrows the oracle out of, in the registry of
/// `module`.
///
/// # Errors
///
/// Raises `ValueError` for a kind registered already or a class registered
/// already, and `RuntimeError` for a module without `fhy_core`'s binding.
pub(crate) fn register_oracle_kind(
    module: &Bound<'_, PyModule>,
    kind: &str,
    class: &Bound<'_, PyType>,
    lease: OracleLease,
) -> PyResult<()> {
    let registry = registry_of(module)?;
    let updated = registry.get().update(|state| {
        if state.oracles.contains_key(kind) {
            return Err(Refusal::KindRegistered);
        }
        if state.holds_class(class) {
            return Err(Refusal::ClassRegistered);
        }
        let mut oracles = state.oracles.clone();
        oracles.insert(
            kind.to_owned(),
            Arc::new(OracleKind {
                class: class.clone().unbind(),
                lease,
            }),
        );
        Ok(KindRegistryState {
            variables: state.variables.clone(),
            alternatives: state.alternatives.clone(),
            oracles,
        })
    });
    updated.map_err(|refusal| refusal.into_error("oracle", kind, class))
}
