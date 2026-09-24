//! Core utilities for the `FhY` compiler infrastructure.
//!
//! The identifier id counter and each [`Interned`](interned::Interned)
//! type's [`InternRegistry`](interned::InternRegistry) are process-global
//! `static`s, so a process must hold exactly one compiled copy of this
//! crate. Link it into one Python extension module, and compile Rust code
//! from other `FhY` packages into that same module rather than into a second
//! one. A second copy would issue ids that collide with the first copy's,
//! and keep registries whose canonical instances never equal the first
//! copy's.
//!
//! # Serialization
//!
//! Every public type that implements `Serialize` also implements
//! `Deserialize`, with a plain serde shape that this crate defines and
//! documents on the type. The impls are format-agnostic: they work with
//! self-describing formats such as JSON and with non-self-describing formats
//! such as postcard, and the tests round-trip every such type through both.
//! Numbers a format may not hold exactly, the integers and floats of an
//! [`Expression`](expr::Expression)'s literals, serialize as strings.
//! Deserializing an [`Identifier`](identifier::Identifier) advances the
//! process-global id counter past its id, and deserializing a
//! [`Canonical<T>`](interned::Canonical) interns the value. A decode that
//! fails partway may leave both effects behind for the parts it already
//! decoded. Both effects only ever add ids and canonical values, so they
//! never invalidate an existing identifier or canonical value. The
//! `__type__`/`__data__` envelope that Python's serialization framework uses
//! belongs to the Python binding, not to these shapes. This crate does not
//! depend on `serde_json`, so depending on it changes nothing about how
//! another crate's JSON numbers parse or compare.

pub mod described_tag;
pub mod diagnostic;
pub mod expr;
pub mod identifier;
pub mod interned;
pub mod op_attribute;
pub mod pass;
pub mod provenance;
pub mod tree;
pub mod value_domain;

#[cfg(test)]
mod test_support;
