//! Threads with a chosen stack size, for the tests of operations over deeply
//! nested values.
//!
//! Included by the test targets that need it with
//! `#[path = "common/stack.rs"] pub mod stack_support;`, so an item one
//! target does not use is not reported as dead code there.

use std::thread;

/// Stack size of the thread the operations documented as iterative run on
/// over a deeply nested value.
pub const SMALL_STACK_BYTES: usize = 128 << 10;

/// Nesting depth of the values the operations documented as iterative run
/// over on a [`SMALL_STACK_BYTES`] stack: under two bytes of stack per
/// level, less than any stack frame, so an operation recursing once per
/// level overflows the stack in an unoptimized and an optimized build alike.
pub const SMALL_STACK_DEPTH: usize = 100_000;

/// Run `body` on a new thread with a stack of `stack_bytes` bytes and return
/// its result, re-raising its panic if it panics.
///
/// # Panics
///
/// Panics if the thread cannot be spawned, and with `body`'s panic if
/// `body` panics.
pub fn run_on_stack<T, F>(stack_bytes: usize, body: F) -> T
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let handle = thread::Builder::new()
        .stack_size(stack_bytes)
        .spawn(body)
        .expect("the test thread spawns");
    match handle.join() {
        Ok(result) => result,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

/// Run `body` on a new thread with a [`SMALL_STACK_BYTES`] stack and return
/// its result, re-raising its panic if it panics.
///
/// `body` should build, use, and drop its deep values on that thread and
/// return nothing deep. A failing assertion there must not format a deep
/// value with `Debug`, which recurses once per level.
///
/// # Panics
///
/// Panics if the thread cannot be spawned, and with `body`'s panic if
/// `body` panics.
pub fn run_on_small_stack<T, F>(body: F) -> T
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    run_on_stack(SMALL_STACK_BYTES, body)
}
