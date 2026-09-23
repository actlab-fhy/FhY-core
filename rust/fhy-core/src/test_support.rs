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

/// Environment variable that marks a child process running one isolated
/// test.
const ISOLATED_TEST_VARIABLE: &str = "FHY_CORE_ISOLATED_TEST";

/// Return whether this process is the child [`assert_isolated_test_passes`]
/// started.
///
/// An isolated test is `#[ignore]`d and returns early unless this holds, so
/// a plain `--ignored` run skips its effect on process-global state.
pub(crate) fn is_isolated_run() -> bool {
    std::env::var_os(ISOLATED_TEST_VARIABLE).is_some()
}

/// Run the ignored test at `test_path` alone in a child process of the test
/// binary and check that it passed, so its effect on process-global state
/// (an exhausted id counter, a registry's first use) stays in that child.
///
/// `test_path` is the test's full path within the crate, such as
/// `"identifier::tests::some_test"`.
pub(crate) fn assert_isolated_test_passes(test_path: &str) {
    let test_binary = std::env::current_exe().expect("the test binary has a path");
    let output = std::process::Command::new(test_binary)
        .args([test_path, "--exact", "--ignored", "--test-threads=1"])
        .env(ISOLATED_TEST_VARIABLE, "1")
        .output()
        .expect("the test binary runs");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("1 passed"),
        "isolated test {test_path} failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
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
