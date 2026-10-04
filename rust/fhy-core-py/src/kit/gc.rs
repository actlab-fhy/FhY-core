//! Cyclic garbage collection of the binding's classes.
//!
//! A class that holds Python objects implements `__traverse__`, which
//! visits each Python object it holds a strong reference to, so the
//! interpreter's cycle collector can free a cycle that runs through it. The collector
//! subtracts one visit per reference, so each reference must be visited at
//! most once, by the object that holds it: visiting it twice could free an
//! object that is still referenced from outside the cycle, while not
//! visiting it only keeps the cycle alive. Every rule here follows from
//! that.
//!
//! - **Fields.** A `Py` field is visited by its object. One behind a
//!   `Mutex` is visited under `try_lock`, and skipped when the lock is
//!   held: a traversal must never block.
//! - **Slots.** A Python object the binding keeps where no traversal can
//!   see it, inside a Rust closure or a core trait object (a rewrite rule's
//!   callbacks, a solver's Python backends, the adapters that hold a
//!   Python-defined part in a core [`Part`](fhy_core::foreign::Part)), is
//!   held in a [`Slot`] instead. The closure or the adapter reads it from
//!   the slot. A core value can be shared by several objects, so the slot is
//!   visited only by the one object that owns it: the object whose
//!   construction made it, which collects it with [`collect_slots`] and
//!   keeps it in a [`Slots`] field. A slot made outside any construction has
//!   no owner and is not visited; it only keeps its object alive, as any
//!   Rust-held reference does.
//! - **Clearing.** `__clear__` empties what only its object holds (a list of
//!   passes, a lattice's elements). It never empties a slot, which a core
//!   value another object still uses may share, so a slot needs no lock. A
//!   cycle through a slot always also runs through a Python object, such as
//!   a callback function or an instance with a `__dict__`, whose own
//!   clearing breaks it.

use std::sync::{Arc, Mutex, TryLockError};

use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};

use super::scoped::ScopedStack;

/// A Python object held where no traversal can see it, visible to the one
/// object that owns it (see the [module docs](self)).
///
/// A slot made outside any collection has no owner, so it holds the object
/// directly: no allocation, and nothing to visit. A slot is never emptied,
/// so it needs no lock.
pub struct Slot(SlotKind);

enum SlotKind {
    /// Shared with the [`Slots`] of the object that owns it.
    Owned(Arc<Py<PyAny>>),
    /// Made outside any collection.
    Unowned(Py<PyAny>),
}

impl std::fmt::Debug for Slot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Slot")
    }
}

impl Slot {
    /// Return the slot of `object`, registered with the innermost
    /// [`collect_slots`] of this thread, if any.
    #[must_use]
    pub fn new(object: Py<PyAny>) -> Self {
        ScopedStack::with_top_mut(&COLLECTORS, |innermost| match innermost {
            Some(innermost) => {
                let object = Arc::new(object);
                innermost.push(Arc::clone(&object));
                Self(SlotKind::Owned(object))
            }
            None => Self(SlotKind::Unowned(object)),
        })
    }

    /// Return a slot for `object` that no collection owns, wherever it is
    /// made: the temporary of one call, which the collector needs not visit.
    #[must_use]
    pub const fn unowned(object: Py<PyAny>) -> Self {
        Self(SlotKind::Unowned(object))
    }

    /// Return the object.
    #[must_use]
    pub fn get<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.py_object().bind(py).clone()
    }

    /// Return a new reference to the object.
    #[must_use]
    pub fn object(&self, py: Python<'_>) -> Py<PyAny> {
        self.py_object().clone_ref(py)
    }

    fn py_object(&self) -> &Py<PyAny> {
        match &self.0 {
            SlotKind::Owned(object) => object,
            SlotKind::Unowned(object) => object,
        }
    }
}

/// The slots one object owns: the ones its construction made.
///
/// The object keeps them in a field and visits them from its
/// `__traverse__` with [`traverse`](Self::traverse).
#[derive(Debug, Default, Clone)]
pub struct Slots(Vec<Arc<Py<PyAny>>>);

impl Slots {
    /// Visit the object of each slot.
    ///
    /// # Errors
    ///
    /// Returns the error the visit returns, which stops the traversal.
    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.0.iter().try_for_each(|object| visit.call(&**object))
    }

    /// Return the number of slots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Return whether the object owns no slot.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

thread_local! {
    /// The slots each [`collect_slots`] in progress on this thread has
    /// collected, innermost last.
    static COLLECTORS: ScopedStack<Vec<Arc<Py<PyAny>>>> = const { ScopedStack::new() };
}

/// Run `build`, returning its result and the slots made while it ran,
/// which the object it builds owns.
///
/// A nested collection keeps its own slots, so each slot has at most one
/// owner. The collection is popped when `build` returns or unwinds.
///
/// # Panics
///
/// Panics if the thread's local storage is being destroyed.
pub fn collect_slots<T>(build: impl FnOnce() -> T) -> (T, Slots) {
    let scope = ScopedStack::push(&COLLECTORS, Vec::new());
    let value = build();
    (value, Slots(scope.pop()))
}

/// Visit the objects `mutex` holds with `traverse`, unless another thread
/// holds the lock.
///
/// A poisoned lock is traversed as it is.
///
/// # Errors
///
/// Returns the error `traverse` returns, which stops the traversal.
pub fn traverse_locked<T>(
    mutex: &Mutex<T>,
    traverse: impl FnOnce(&T) -> Result<(), PyTraverseError>,
) -> Result<(), PyTraverseError> {
    let guard = match mutex.try_lock() {
        Ok(guard) => guard,
        Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(TryLockError::WouldBlock) => return Ok(()),
    };
    traverse(&guard)
}

/// Replace what `mutex` holds with its default, dropping the old value after
/// the lock is released, since dropping a Python object can run Python
/// code that takes the lock again.
pub fn clear_locked<T: Default>(mutex: &Mutex<T>) {
    let old = std::mem::take(
        &mut *mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    );
    drop(old);
}

/// Visit every object of `objects`.
///
/// # Errors
///
/// Returns the error a visit returns, which stops the traversal.
pub fn traverse_all<'a, T: 'a>(
    visit: &PyVisit<'_>,
    objects: impl IntoIterator<Item = &'a Py<T>>,
) -> Result<(), PyTraverseError> {
    objects
        .into_iter()
        .try_for_each(|object| visit.call(object))
}

#[cfg(test)]
mod tests;
