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
//! documents on the type. Apart from [`Expression`](expr::Expression), whose
//! wire form is read through a JSON value, the impls are format-agnostic:
//! they work with self-describing formats such as JSON and with
//! non-self-describing formats such as postcard, and the tests round-trip
//! each such type through both. Deserializing an
//! [`Identifier`](identifier::Identifier) advances the process-global id
//! counter past its id, and deserializing a
//! [`Canonical<T>`](interned::Canonical) interns the value. A decode that
//! fails partway may leave both effects behind for the parts it already
//! decoded. Both effects only ever add ids and canonical values, so they
//! never invalidate an existing identifier or canonical value. The
//! `__type__`/`__data__` envelope that Python's serialization framework uses
//! belongs to the Python binding, not to these shapes; only the expression
//! wire form still carries it.
//!
//! This crate turns on `serde_json`'s `arbitrary_precision` feature, so an
//! integer literal of any size serializes as a JSON integer with all its
//! digits, as Python writes it. Cargo unifies features across the build
//! graph, so depending on this crate turns the feature on for every crate in
//! the build that uses `serde_json`. There a `serde_json::Number` keeps the
//! text it was parsed from and compares by that text, so `1.0` and `1.00`
//! parse to unequal values, and a number that serde buffers for an untagged
//! enum or a `#[serde(flatten)]` field reaches it as a map, which a derived
//! `Deserialize` refuses for a numeric field.

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

mod decode;

#[cfg(test)]
mod test_support;
