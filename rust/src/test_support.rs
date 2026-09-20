//! Helpers shared by the crate's unit tests.
//!
//! Compiled only under `cfg(test)`, so nothing here reaches the shipped
//! library.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::identifier::Identifier;

/// Distance above a freshly allocated id at which an id is unreachable by the
/// identifiers the rest of the suite allocates while one test runs.
const ID_HEADROOM: u64 = 1_000_000;

/// Return an id no other test pins or allocates while this test runs.
///
/// A test that pins an id inside a payload cannot simply pick a number: the
/// suite runs in parallel against one process-global counter. Anchoring on a
/// freshly allocated id and adding [`ID_HEADROOM`] keeps the pinned id clear
/// of every other test's, because no two anchors are equal and the suite
/// allocates nothing like a million ids while one test runs.
pub(crate) fn reserve_pinned_id(anchor_hint: &str) -> u64 {
    Identifier::new(anchor_hint).id() + ID_HEADROOM
}

/// Return `value`'s hash under the default hasher.
///
/// Call it on the value itself, not on a handle to it: a
/// [`Canonical`](crate::interned::Canonical) hashes by identity, so
/// `compute_hash(&*handle)` and `compute_hash(&handle)` answer different
/// questions.
pub(crate) fn compute_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}
