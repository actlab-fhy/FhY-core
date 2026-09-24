//! Helpers shared by the crate's unit tests.
//!
//! Compiled only under `cfg(test)`, so nothing here reaches the shipped
//! library.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Compile-time check that `T` can be shared and sent across threads, for use
/// in a `const _: () = { ... };` item.
pub(crate) const fn assert_send_sync<T: Send + Sync>() {}

/// Return `value`'s hash under the default hasher.
pub(crate) fn compute_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}
