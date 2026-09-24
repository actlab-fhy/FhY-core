//! The hash the equality-agrees-with-hash tests compare.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Return the hash of `value` under the standard hasher.
#[must_use]
pub(crate) fn hash_of<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}
