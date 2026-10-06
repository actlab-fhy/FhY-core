//! The building blocks of a Rust-backed Python class, for the binding crates
//! of the packages that build on `fhy_core`.
//!
//! `fhy_core`'s own classes are written over these, and a downstream
//! `-py` crate writes its classes the same way, so that its classes behave as
//! `fhy_core`'s do: they are frozen, hash and compare as dataclasses,
//! serialize and deserialize with the framework's exceptions, keep one
//! Python object per canonical value, and take part in the interpreter's
//! cycle collector. Nothing here holds a class or a registry of its own
//! beyond what the caller declares in a `static`, so each downstream crate's
//! items are its own even though one aggregate extension links one copy of
//! this crate (see the [crate documentation](crate)).
//!
//! | Module | What it holds |
//! | --- | --- |
//! | [`python`] | [`Seed`](python::Seed), [`ImportedAttr`](python::ImportedAttr), [`cached_attr!`](crate::cached_attr), [`read_type_name`](python::read_type_name) |
//! | [`exceptions`] | [`ExceptionClass`](exceptions::ExceptionClass) and the framework's exception classes |
//! | [`interned`] | [`IdentityCache`](interned::IdentityCache) and the `InternedMixin` contract |
//! | [`public_class`] | [`PublicClass`](public_class::PublicClass), the public class a binding class is registered under |
//! | [`frozen`] | the `FrozenMixin` refusals |
//! | [`dataclass`] | the equality, repr, hash and optional-argument helpers of a dataclass |
//! | [`serialization`] | the payload readers and field shapes of `fhy_core.serialization` |
//! | [`scoped`] | [`ScopedStack`](scoped::ScopedStack), a thread-local stack that unwinds safely |
//! | [`pending`] | the pending exception of an infallible hook |
//! | [`frames`] | [`Frames`](frames::Frames), the per-call context a hook reads, whose miss is loud |
//! | [`hook`] | [`ask`](hook::ask), a Python hook behind an infallible trait method |
//! | [`integers`] | [`read_unsigned`](integers::read_unsigned), a Python int as a Rust unsigned integer with exact errors, and [`build_too_large_error`](integers::build_too_large_error), the `OverflowError` for a number above the target's maximum |
//! | `testing` | (behind the `testing` feature; test-only, not stable API) the embedded interpreter and `fhy_core` stand-ins for a downstream crate's tests |
//! | [`gc`] | [`Slot`](gc::Slot), [`collect_slots`](gc::collect_slots) and the traverse helpers |
//! | [`foreign`] | [`read_foreign`](foreign::read_foreign), a Python-defined part as a core `Foreign` |
//!
//! Every function that takes a `Python` token or a Python object needs the
//! GIL (or an attached thread), and none holds a lock of its own across a
//! call into Python.

pub mod dataclass;
pub mod exceptions;
pub mod foreign;
pub mod frames;
pub mod frozen;
pub mod gc;
pub mod hook;
pub mod integers;
pub mod interned;
pub mod pending;
pub mod public_class;
pub mod python;
pub mod scoped;
pub mod serialization;

#[cfg(any(test, feature = "testing"))]
#[doc(hidden)]
pub mod testing;
