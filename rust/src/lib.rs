//! Core utilities for the `FhY` compiler infrastructure.

pub mod identifier;

#[cfg(feature = "python")]
mod python;

/// Returns `true`, confirming the compiled Rust extension is loaded.
#[must_use]
pub fn rust_available() -> bool {
    true
}
