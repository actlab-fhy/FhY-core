//! The hash the equality-agrees-with-hash tests compare.
//!
//! Included by the test targets that need it with
//! `#[path = "common/hashing.rs"] pub mod hashing_support;`.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Return the hash of `value` under the standard hasher.
#[must_use]
pub fn hash_of<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}
