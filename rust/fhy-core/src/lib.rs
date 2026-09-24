//! Core utilities for the `FhY` compiler infrastructure.
//!
//! The identifier id counter, each [`Interned`](interned::Interned) type's
//! [`InternRegistry`](interned::InternRegistry), and the pass registry and
//! run counters of [`pass`] are process-global `static`s, so a process must
//! hold exactly one compiled copy of this crate. Link it into one Python
//! extension module, and compile Rust code from other `FhY` packages into
//! that same module rather than into a second one. A second copy would
//! issue ids that collide with the first copy's, keep registries whose
//! canonical instances never equal the first copy's, and keep a separate
//! pass registry and separate run counters.
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

pub mod diagnostic;
pub mod expr;
pub mod identifier;
pub mod interned;
pub mod op_attribute;
pub mod pass;
pub mod provenance;
pub mod value_domain;

mod decode;
mod described_tag;
mod python_text;
mod shipped;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

#[cfg(test)]
mod test_support;
