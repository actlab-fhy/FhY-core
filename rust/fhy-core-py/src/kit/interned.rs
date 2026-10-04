//! Python identity for the canonical instances of [`fhy_core::interned`],
//! and the parts of the Python `InternedMixin` contract the Rust-backed
//! interned classes share.
//!
//! A canonical Rust value reaches Python as exactly one Python object, so
//! `is` holds for equal keys as it does for values interned in Python. Each
//! Rust-backed interned class keeps an [`IdentityCache`] from its canonical
//! keys, of whatever type the class is keyed by, to those objects; the core
//! crate never holds Python objects.
//!
//! The rest matches the Python implementation, `fhy_core.traits.interned`,
//! which is defined in both languages: the `KeyError` of `require_interned`,
//! the warning logged when a payload's description is ignored, and the
//! `DeserializationValueError` of a payload that conflicts with the
//! canonical instance.

use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::sync::{LazyLock, Mutex, MutexGuard, PoisonError};

use pyo3::exceptions::{PyKeyError, PyNotImplementedError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyString, PyType};

use super::exceptions::DESERIALIZATION_VALUE_ERROR;

/// Map from each canonical key to the single Python object of its canonical
/// instance.
///
/// Generic over the key `K`, which is the identifier's id, `u64`, for every
/// interned class `fhy_core` exposes (identifiers compare by id), and
/// whatever identifies a canonical instance for a downstream class, such as
/// a `String`. Look an entry up by anything `K` borrows as, so a
/// `String`-keyed cache is read by `&str`. The Rust registries are
/// append-only, so an entry never goes stale, and the cache keeps each
/// object alive for the rest of the process, as the registry keeps its
/// instance.
///
/// The lock is held only to read or insert an entry, never across a call
/// into Python.
pub struct IdentityCache<K = u64> {
    objects: LazyLock<Mutex<HashMap<K, Py<PyAny>>>>,
}

impl<K> fmt::Debug for IdentityCache<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IdentityCache").finish_non_exhaustive()
    }
}

impl<K: Eq + Hash> IdentityCache<K> {
    /// Create an empty cache, for a `static`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            objects: LazyLock::new(|| Mutex::new(HashMap::new())),
        }
    }

    /// Return the Python object cached for `key`, if any.
    #[must_use]
    pub fn get<'py, Q>(&self, py: Python<'py>, key: &Q) -> Option<Bound<'py, PyAny>>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.lock().get(key).map(|object| object.bind(py).clone())
    }

    /// Cache `object` for `key`, unless an object is cached for it already,
    /// and return the cached object.
    ///
    /// Another thread may cache an object for the same key between a miss
    /// and this call; the first object cached stays the canonical one, so
    /// the caller returns the result and drops its own object.
    pub fn insert<'py>(&self, key: K, object: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        let py = object.py();
        self.lock()
            .entry(key)
            .or_insert_with(|| object.unbind())
            .bind(py)
            .clone()
    }

    /// Recovers from poisoning: the map is only ever read or extended by one
    /// entry, so a panic never leaves it half-updated.
    fn lock(&self) -> MutexGuard<'_, HashMap<K, Py<PyAny>>> {
        self.objects.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl<K: Eq + Hash> Default for IdentityCache<K> {
    fn default() -> Self {
        Self::new()
    }
}

/// Raise the `NotImplementedError` of an `InternedMixin` registry operation
/// that the append-only Rust registries cannot support.
///
/// # Errors
///
/// Always returns the `NotImplementedError`, naming the class and `method`;
/// returns what reading the class's name raises instead if that fails.
pub fn raise_registry_append_only(cls: &Bound<'_, PyType>, method: &str) -> PyResult<()> {
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
///
/// # Errors
///
/// Raises what reading the class's name or the key's `repr` raises.
pub fn build_not_interned_error(
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
///
/// # Errors
///
/// Raises what comparing the descriptions, reading the class's name or
/// logging raises.
pub fn warn_if_description_ignored(
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
///
/// # Errors
///
/// Raises what reading the class's name or a value's `repr` raises; the
/// returned error is otherwise the `DeserializationValueError` itself, or
/// the exception that importing or building it raised.
pub fn build_conflict_error(
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
    Ok(DESERIALIZATION_VALUE_ERROR.err(cls.py(), (message,)))
}

#[cfg(test)]
mod tests;
