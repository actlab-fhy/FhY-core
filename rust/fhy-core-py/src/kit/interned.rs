//! Python identity for the canonical instances of [`fhy_core::interned`],
//! and the parts of the Python `InternedMixin` contract the Rust-backed
//! interned classes share.
//!
//! A canonical Rust value reaches Python as exactly one Python object, so
//! `is` holds for equal keys as it does for values interned in Python. Each
//! Rust-backed interned class keeps an [`IdentityCache`] from its canonical
//! keys to those objects; the core crate never holds Python objects.
//!
//! The rest matches the Python implementation, `fhy_core.traits.interned`,
//! which is defined in both languages: the `KeyError` of `require_interned`,
//! the warning logged when a payload's description is ignored, and the
//! `DeserializationValueError` of a payload that conflicts with the
//! canonical instance.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex, PoisonError};

use pyo3::exceptions::{PyKeyError, PyNotImplementedError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyString, PyType};

/// Map from the id of each canonical key to the single Python object of its
/// canonical instance.
///
/// Keyed by identifier id, because every interned class the binding exposes
/// is keyed by an identifier, and identifiers compare by id. The Rust
/// registries are append-only, so an entry never goes stale, and the cache
/// keeps each object alive for the rest of the process, as the registry
/// keeps its instance.
///
/// The lock is held only to read or insert an entry, never across a call
/// into Python.
pub(crate) struct IdentityCache {
    objects: LazyLock<Mutex<HashMap<u64, Py<PyAny>>>>,
}

impl IdentityCache {
    /// Create an empty cache.
    pub(crate) const fn new() -> Self {
        Self {
            objects: LazyLock::new(|| Mutex::new(HashMap::new())),
        }
    }

    /// Return the Python object cached for the key with id `id`, if any.
    pub(crate) fn get<'py>(&self, py: Python<'py>, id: u64) -> Option<Bound<'py, PyAny>> {
        self.lock().get(&id).map(|object| object.bind(py).clone())
    }

    /// Cache `object` for the key with id `id`, unless an object is cached
    /// for it already, and return the cached object.
    ///
    /// Another thread may cache an object for the same key between a miss
    /// and this call; the first object cached stays the canonical one.
    pub(crate) fn insert<'py>(&self, id: u64, object: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        let py = object.py();
        self.lock()
            .entry(id)
            .or_insert_with(|| object.unbind())
            .bind(py)
            .clone()
    }

    /// Recovers from poisoning: the map is only ever read or extended by one
    /// entry, so a panic never leaves it half-updated.
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<u64, Py<PyAny>>> {
        self.objects.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Raise the `NotImplementedError` of an `InternedMixin` registry operation
/// that the append-only Rust registries cannot support.
pub(crate) fn raise_registry_append_only(cls: &Bound<'_, PyType>, method: &str) -> PyResult<()> {
    Err(PyNotImplementedError::new_err(format!(
        "{}.{method} is not supported: the Rust intern registries are \
         append-only, so a canonical instance is never removed or replaced.",
        cls.name()?
    )))
}

/// Return the `KeyError` `InternedMixin.require_interned` raises when no
/// instance is registered under `key`.
///
/// Matches the Python implementation: `InternedMixin.require_interned`.
pub(crate) fn build_not_interned_error(
    cls: &Bound<'_, PyType>,
    key: &Bound<'_, PyAny>,
) -> PyResult<PyErr> {
    Ok(PyKeyError::new_err(format!(
        "No registered \"{}\" instance for key {}.",
        cls.name()?,
        key.repr()?
    )))
}

/// Log that a payload's description differs from the canonical instance's
/// and is ignored, unless the two are equal.
///
/// Matches the Python implementation: the warning
/// `fhy_core.traits.interned` logs through its own logger when
/// `InternedMixin.construct_from_fields` keeps the canonical instance of an
/// equality-excluded field.
pub(crate) fn warn_if_description_ignored(
    cls: &Bound<'_, PyType>,
    key: &Bound<'_, PyAny>,
    canonical_description: &Bound<'_, PyString>,
    payload_description: &Bound<'_, PyString>,
) -> PyResult<()> {
    if canonical_description.as_any().eq(payload_description)? {
        return Ok(());
    }
    let py = cls.py();
    let logger = crate::kit::python::cached_attr!(py, "logging", "getLogger" => PyAny)?
        .call1((intern!(py, "fhy_core.traits.interned"),))?;
    logger.call_method1(
        intern!(py, "warning"),
        (
            intern!(
                py,
                "%s %r already canonical; keeping %s=%r and ignoring payload %r."
            ),
            cls.name()?,
            key,
            intern!(py, "description"),
            canonical_description,
            payload_description,
        ),
    )?;
    Ok(())
}

/// Return the `DeserializationValueError` for a payload whose compared field
/// `field` differs from the canonical instance's.
///
/// Matches the Python implementation: the error
/// `InternedMixin.construct_from_fields` raises for a conflicting payload.
pub(crate) fn build_conflict_error(
    cls: &Bound<'_, PyType>,
    key: &Bound<'_, PyAny>,
    field: &str,
    canonical_value: &Bound<'_, PyAny>,
    payload_value: &Bound<'_, PyAny>,
) -> PyResult<PyErr> {
    let message = format!(
        "Payload for \"{}\" key {} conflicts with the canonical instance on \
         {field} (canonical {}, payload {}).",
        cls.name()?,
        key.repr()?,
        canonical_value.repr()?,
        payload_value.repr()?,
    );
    Ok(crate::kit::exceptions::DESERIALIZATION_VALUE_ERROR.err(cls.py(), (message,)))
}
