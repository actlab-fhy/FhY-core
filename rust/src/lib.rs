//! Core utilities for the `FhY` compiler infrastructure.

pub mod identifier;
pub mod interned;
pub mod op_attribute;
pub mod value_domain;

mod buffered_payload;

#[cfg(feature = "python")]
mod python;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

#[cfg(test)]
mod test_support;
