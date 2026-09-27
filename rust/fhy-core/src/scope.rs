//! Lexical scopes: a stack of frames with shadowing lookup.
//!
//! A [`Scope`] starts with one frame, the root, which cannot be popped.
//! [`push`](Scope::push) and [`pop`](Scope::pop) nest and un-nest frames
//! last in, first out, or [`with_frame`](Scope::with_frame) does both
//! around a closure. [`define`](Scope::define) binds a key in the
//! innermost frame, and [`lookup`](Scope::lookup) searches from the
//! innermost frame outward, so an inner binding shadows an outer one with
//! the same key.
//!
//! # Examples
//!
//! ```
//! use fhy_core::scope::Scope;
//!
//! let mut scope = Scope::new();
//! scope.define("x", 1);
//!
//! let inner = scope.with_frame(|scope| {
//!     scope.define("x", 2);
//!     scope.lookup("x").copied()
//! });
//!
//! assert_eq!(inner, Some(2));
//! assert_eq!(scope.lookup("x"), Some(&1));
//! assert_eq!(scope.depth(), 1);
//! assert!(scope.pop().is_err());
//! ```

use std::borrow::Borrow;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::hash::Hash;

/// A stack of lexical frames mapping `K` to `V`, with shadowing lookup.
///
/// The root frame exists for the scope's whole life, so
/// [`depth`](Self::depth) is at least 1.
#[derive(Debug, Clone)]
pub struct Scope<K, V> {
    root: HashMap<K, V>,
    inner: Vec<HashMap<K, V>>,
}

impl<K, V> Scope<K, V> {
    /// Return a scope holding only an empty root frame.
    #[must_use]
    pub fn new() -> Self {
        Self {
            root: HashMap::new(),
            inner: Vec::new(),
        }
    }

    /// Push a new, empty innermost frame.
    pub fn push(&mut self) {
        self.inner.push(HashMap::new());
    }

    /// Pop the innermost frame, discarding its bindings.
    ///
    /// # Errors
    ///
    /// [`RootFramePopError`] if only the root frame is left; the scope is
    /// then unchanged.
    pub fn pop(&mut self) -> Result<(), RootFramePopError> {
        self.inner.pop().map(drop).ok_or(RootFramePopError)
    }

    /// Return the number of frames, the root included; at least 1.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.inner.len() + 1
    }

    /// Push a frame, run `body` on this scope, and restore the depth found
    /// on entry; return `body`'s result.
    ///
    /// The depth is restored when `body` returns and when it panics, so
    /// the frame and its bindings go in both cases, as do any frames
    /// `body` pushed and left. If `body` pops below the entry depth, the
    /// scope is left as `body` left it.
    pub fn with_frame<R>(&mut self, body: impl FnOnce(&mut Self) -> R) -> R {
        let entry_inner_frames = self.inner.len();
        self.push();
        let guard = RestoreDepth {
            scope: self,
            inner_frames: entry_inner_frames,
        };
        body(&mut *guard.scope)
    }

    /// Return the innermost frame.
    fn innermost(&self) -> &HashMap<K, V> {
        self.inner.last().unwrap_or(&self.root)
    }
}

impl<K: Hash + Eq, V> Scope<K, V> {
    /// Bind `key` to `value` in the innermost frame.
    ///
    /// Replaces a binding of `key` in the innermost frame, and shadows any
    /// binding of `key` in an outer frame.
    pub fn define(&mut self, key: K, value: V) {
        let innermost = self.inner.last_mut().unwrap_or(&mut self.root);
        innermost.insert(key, value);
    }

    /// Return the value bound to `key` in the innermost frame that binds
    /// it, or `None` if no frame does.
    #[must_use]
    pub fn lookup<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner
            .iter()
            .rev()
            .chain(std::iter::once(&self.root))
            .find_map(|frame| frame.get(key))
    }

    /// Return the value bound to `key` in the innermost frame only, or
    /// `None` if that frame does not bind it.
    #[must_use]
    pub fn lookup_local<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.innermost().get(key)
    }

    /// Return whether any frame binds `key`.
    #[must_use]
    pub fn is_defined<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner
            .iter()
            .chain(std::iter::once(&self.root))
            .any(|frame| frame.contains_key(key))
    }

    /// Return whether the innermost frame binds `key`.
    #[must_use]
    pub fn is_defined_local<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.innermost().contains_key(key)
    }
}

/// Truncates a scope back to a depth when dropped, on a normal return and
/// on unwinding alike.
struct RestoreDepth<'a, K, V> {
    scope: &'a mut Scope<K, V>,
    inner_frames: usize,
}

impl<K, V> Drop for RestoreDepth<'_, K, V> {
    fn drop(&mut self) {
        self.scope.inner.truncate(self.inner_frames);
    }
}

impl<K: Hash + Eq, V: PartialEq> PartialEq for Scope<K, V> {
    /// Return whether both scopes have the same frames, each binding the
    /// same keys to equal values.
    fn eq(&self, other: &Self) -> bool {
        self.root == other.root && self.inner == other.inner
    }
}

impl<K: Hash + Eq, V: Eq> Eq for Scope<K, V> {}

impl<K, V> Default for Scope<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

/// A [`Scope::pop`] with only the root frame left.
///
/// Displays as `cannot pop the root frame`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct RootFramePopError;

impl fmt::Display for RootFramePopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("cannot pop the root frame")
    }
}

impl Error for RootFramePopError {}
