//! Structural patterns over expressions and a bottom-up rewrite walk.
//!
//! A [`Pattern`] describes a shape of [`Expression`](super::Expression);
//! [`Pattern::matches`] tests it at the root of an expression and returns
//! the [`MatchBindings`] its [`Capture`]s recorded. A [`Rule`] is a rewrite
//! tried at the root of an expression; a [`RewriteRule`] is the rule pairing
//! a pattern with a rewrite of what it matches. [`apply_rewrite_rules`]
//! applies a list of rules to every node of a tree, bottom-up, in one pass,
//! and [`RewriteRuleApplier`](super::passes::RewriteRuleApplier) does the
//! same as a compiler pass. Callbacks supplied by the caller (predicates,
//! guards, rewrites and native rules) are fallible; their
//! [`CallbackError`]s end a match or a walk.

mod matching;
mod rewrite;

pub use self::matching::{CallbackError, Capture, MatchBindings, Pattern};
pub use self::rewrite::{
    FiredRule, RewriteError, RewriteOutcome, RewriteRule, Rule, apply_rewrite_rules,
};
pub(in crate::expr) use self::rewrite::{RuleRun, run_rewrite_rules};
