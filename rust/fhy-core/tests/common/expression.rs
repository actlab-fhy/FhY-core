//! Shared builders for the expression tests.
//!
//! Included by the `expression_*` test targets with
//! `#[path = "common/expression.rs"] pub mod expression_support;`, so an
//! item one target does not use is not reported as dead code there.

use std::thread;

use fhy_core::identifier::Identifier;
use fhy_core::symbolic::expression::{Expression, LiteralValue, build_logical_and};

/// Depth of the deep trees the depth tests build.
pub const DEEP_TREE_DEPTH: usize = 4000;

/// Stack size for the substitution and screen walks over a deep tree.
pub const WALK_STACK_BYTES: usize = 16 << 20;

/// Stack size for a serialization round trip of a deep tree through
/// `serde_json` values, whose own recursion needs the most room.
pub const SERIALIZATION_STACK_BYTES: usize = 64 << 20;

/// Mint an identifier named `name` and return it with a reference to it.
#[must_use]
pub fn build_identifier(name: &str) -> (Identifier, Expression) {
    let identifier = Identifier::new(name);
    let reference = Expression::from(identifier.clone());
    (identifier, reference)
}

/// Return a literal expression holding `value`.
#[must_use]
pub fn build_literal(value: impl Into<LiteralValue>) -> Expression {
    Expression::from(value.into())
}

/// Return a literal expression holding the numeric text `text`.
///
/// # Panics
///
/// Panics if `text` is outside the literal grammar.
#[must_use]
pub fn build_text_literal(text: &str) -> Expression {
    Expression::from(LiteralValue::parse_text(text).expect("the text is a literal text"))
}

/// Return `((leaf + 1) + 1) + ...`, `depth` additions deep.
#[must_use]
pub fn build_deep_sum(leaf: &Expression, depth: usize) -> Expression {
    let mut tree = leaf.clone();
    for _ in 0..depth {
        tree = tree + 1;
    }
    tree
}

/// Return `leaf && (true && (true && ...))`, `depth` conjunctions deep, with
/// `leaf` at the bottom of the right spine.
///
/// # Panics
///
/// Panics if a conjunction of two operands is refused.
#[must_use]
pub fn build_deep_conjunction(leaf: &Expression, depth: usize) -> Expression {
    let mut tree = leaf.clone();
    for _ in 0..depth {
        tree = build_logical_and([build_literal(true), tree])
            .expect("two operands make a conjunction");
    }
    tree
}

/// Run `body` on a new thread with a stack of `stack_bytes` bytes and return
/// its result, re-raising its panic if it panics.
///
/// # Panics
///
/// Panics if the thread cannot be spawned, and with `body`'s panic if
/// `body` panics.
pub fn run_on_large_stack<T, F>(stack_bytes: usize, body: F) -> T
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
