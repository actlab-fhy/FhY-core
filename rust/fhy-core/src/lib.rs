//! Core utilities for the `FhY` compiler infrastructure.
//!
//! The identifier id counter and every
//! [`InternRegistry`](interned::InternRegistry) are process-global
//! `static`s, so a process must hold exactly one compiled copy of this
//! crate. Link it into one Python extension module, and compile Rust code
//! from other `FhY` packages into that same module rather than into a second
//! one. A second copy would issue ids that collide with the first copy's and
//! keep registries whose canonical instances never equal the first copy's.

pub mod identifier;
pub mod interned;
pub mod op_attribute;
pub mod value_domain;

mod decode;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

#[cfg(test)]
mod test_support;
