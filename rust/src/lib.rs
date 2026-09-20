//! Core utilities for the `FhY` compiler infrastructure.

pub mod identifier;
pub mod interned;
pub mod op_attribute;
pub mod value_domain;

#[cfg(feature = "python")]
mod python;

#[cfg(test)]
mod test_support;

/// Returns `true`, confirming the compiled Rust extension is loaded.
#[must_use]
pub fn rust_available() -> bool {
    true
}
