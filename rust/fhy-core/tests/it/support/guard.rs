//! The draws of the non-vacuity guards: a guard checks that a property's
//! strategy reaches the shapes the property claims to cover, over cases
//! drawn by a runner with a fixed seed, so its counts are the same on
//! every run.

use proptest::strategy::{Strategy, ValueTree};
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};

/// The cases each guard draws.
pub(crate) const GUARD_CASES: usize = 256;

/// Return `GUARD_CASES` values of `strategy`, drawn by a runner with a
/// fixed seed.
///
/// # Panics
///
/// Panics if the strategy fails to draw a value.
pub(crate) fn draw_guard_cases<S: Strategy>(strategy: &S) -> Vec<S::Value> {
    let mut runner = TestRunner::new_with_rng(
        Config::default(),
        TestRng::deterministic_rng(RngAlgorithm::ChaCha),
    );
    (0..GUARD_CASES)
        .map(|_| {
            strategy
                .new_tree(&mut runner)
                .expect("the strategy draws")
                .current()
        })
        .collect()
}
