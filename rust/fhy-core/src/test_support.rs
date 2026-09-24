//! Helpers shared by the crate's unit tests.
//!
//! Compiled only under `cfg(test)`, so nothing here reaches the shipped
//! library.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, MutexGuard, PoisonError};

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

/// Distance above a freshly allocated id at which a counter-observing test
/// pins its first id, far beyond anything [`reserve_pinned_id`] hands out.
const FAR_AHEAD_HEADROOM: u64 = 1_000_000_000_000;

/// Distance between consecutive ids one counter-observing test pins, far
/// beyond [`ID_HEADROOM`], so neither the ids other tests pin nor the ids
/// they allocate after the counter passes one pinned id reach the next.
const FAR_AHEAD_SPACING: u64 = 1_000_000_000;

/// Serializes the tests that observe whether the id counter has passed an
/// id, since each moves the counter far ahead and would otherwise pass the
/// ids another such test pinned.
static ID_COUNTER_GUARD: Mutex<()> = Mutex::new(());

/// Hold the id counter against every other counter-observing test.
pub(crate) fn hold_id_counter() -> MutexGuard<'static, ()> {
    ID_COUNTER_GUARD
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Return `N` increasing ids far ahead of the id counter, for a test that
/// holds [`hold_id_counter`] and checks which of them a decode restores.
pub(crate) fn reserve_far_ahead_ids<const N: usize>(anchor_hint: &str) -> [u64; N] {
    let first = Identifier::new(anchor_hint).id() + FAR_AHEAD_HEADROOM;
    std::array::from_fn(|index| first + FAR_AHEAD_SPACING * index as u64)
}

/// Return whether the id counter has moved past `id`, drawing one id to
/// find out.
pub(crate) fn has_counter_passed(id: u64) -> bool {
    Identifier::new("counter-probe").id() > id
}

/// Compile-time check that `T` can be shared and sent across threads, for use
/// in a `const _: () = { ... };` item.
pub(crate) const fn assert_send_sync<T: Send + Sync>() {}

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
