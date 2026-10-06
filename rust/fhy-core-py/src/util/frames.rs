//! Per-call context for the hooks behind a core trait method that cannot
//! fail.
//!
//! A class that adapts a Python-defined part to a core trait often needs
//! context the trait's method does not carry: the namespace objects, the
//! shapes or the addressings of the entry point that started the call. The
//! entry point pushes that context as a frame, in a guard that pops it when
//! the entry point returns, on unwind included, and the hook reads the
//! innermost frame.
//!
//! [`Frames`] is that pattern over a [`ScopedStack`], with one rule that a
//! bare stack does not enforce: **a read that finds no frame is loud**. A
//! hook asked outside an entry point that pushes a frame has no context to
//! answer from, and answering a default would be a wrong answer that nothing
//! flags. A [`Frames`] read records an exception as the pending exception of
//! the call ([`record_pending_error`]) and returns the caller's fallback, so
//! the entry point raises it when the core returns.
//!
//! ```no_run
//! use fhy_core_py::util::frames::Frames;
//! use fhy_core_py::util::scoped::ScopedStack;
//!
//! thread_local! {
//!     static STACK: ScopedStack<String> = const { ScopedStack::new() };
//! }
//! static NAMES: Frames<String> = Frames::new(&STACK, "the namespace name");
//!
//! let _guard = NAMES.push("memory".to_owned());
//! assert_eq!(NAMES.read_or(0, |name| name.len()), 6);
//! ```

use std::thread::LocalKey;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use super::pending::record_pending_error;
use super::scoped::{ScopedGuard, ScopedStack};

/// A thread-local stack of context frames whose reads are loud on a miss.
///
/// Declare the stack in a `thread_local!` and the `Frames` in a `static`
/// over it: `static NAMES: Frames<Name> = Frames::new(&STACK, "the name");`.
#[derive(Debug)]
pub struct Frames<T: 'static> {
    key: &'static LocalKey<ScopedStack<T>>,
    what: &'static str,
}

impl<T: 'static> Frames<T> {
    /// Return the frames kept in the stack `key`, which `what` names in the
    /// error a read that finds none records, as in "the shapes".
    #[must_use]
    pub const fn new(key: &'static LocalKey<ScopedStack<T>>, what: &'static str) -> Self {
        Self { key, what }
    }

    /// Push `frame`, and return the guard that pops it when dropped, on
    /// unwind included.
    ///
    /// # Panics
    ///
    /// Panics as [`ScopedStack::push`] does: if called from a closure that
    /// is reading this stack.
    pub fn push(&self, frame: T) -> ScopedGuard<T> {
        ScopedStack::push(self.key, frame)
    }

    /// Return whether a frame is pushed, without recording anything.
    #[must_use]
    pub fn is_pushed(&self) -> bool {
        ScopedStack::depth(self.key) > 0
    }

    /// Return `read` of the innermost frame.
    ///
    /// When no frame is pushed, `read` is not called: the miss is recorded
    /// as the pending exception of the call, a `RuntimeError` naming what
    /// was asked for, and the result is `None`. A caller maps `None` to
    /// the fallback its trait documents; the entry point raises the
    /// exception when the core returns.
    ///
    /// `read` runs while the stack is borrowed, so it must not push onto or
    /// pop from it; clone out what it needs.
    ///
    /// # Pending exception
    ///
    /// Nothing is returned for a miss, since the hook cannot fail: it records
    /// a `RuntimeError` as the pending exception, which the entry point
    /// raises.
    ///
    /// # Panics
    ///
    /// Panics if `read` pushes onto or pops from this stack, and if the
    /// Python interpreter is not initialized (recording the miss attaches).
    pub fn read<R>(&self, read: impl FnOnce(&T) -> R) -> Option<R> {
        let found = ScopedStack::with_top(self.key, |frame| frame.map(read));
        if found.is_none() {
            record_pending_error(self.build_miss_error());
        }
        found
    }

    /// Return `read` of the innermost frame, or `fallback` when none is
    /// pushed, which is recorded as [`read`](Self::read) does.
    ///
    /// # Panics
    ///
    /// Panics as [`read`](Self::read) does.
    pub fn read_or<R>(&self, fallback: R, read: impl FnOnce(&T) -> R) -> R {
        self.read(read).unwrap_or(fallback)
    }

    /// Return a clone of the innermost frame, recording a miss as
    /// [`read`](Self::read) does.
    ///
    /// # Panics
    ///
    /// Panics as [`read`](Self::read) does.
    #[must_use]
    pub fn cloned(&self) -> Option<T>
    where
        T: Clone,
    {
        self.read(T::clone)
    }

    /// Return the error a hook that finds no frame records.
    fn build_miss_error(&self) -> PyErr {
        PyRuntimeError::new_err(format!(
            "a hook asked for {} outside an entry point that provides it",
            self.what
        ))
    }
}

#[cfg(test)]
mod tests;
