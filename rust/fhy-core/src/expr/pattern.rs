//! Structural patterns over expressions and a bottom-up rewrite walk.
//!
//! A [`Pattern`] describes a shape of [`Expression`](super::Expression);
//! [`match_pattern`] tests it at the root of an expression and returns the
//! [`MatchBindings`] its captures recorded. A [`RewriteRule`] pairs a
//! pattern with a rewrite of what it matches, and [`apply_rewrite_rules`]
//! applies a list of rules to every node of a tree, bottom-up, in one pass;
//! [`RewriteRuleApplier`](super::passes::RewriteRuleApplier) does the same
//! as a compiler pass. Callbacks
//! supplied by the caller (predicates, guards, rewrites) are fallible; their
//! [`CallbackError`]s end a match or a walk.

mod matching;
mod rewrite;

pub use self::matching::{
    CallbackError, MatchBindings, Pattern, PatternError, does_pattern_match, match_pattern,
};
pub use self::rewrite::{
    FiredRule, RewriteError, RewriteOutcome, RewriteRule, apply_rewrite_rule, apply_rewrite_rules,
};
pub(in crate::expr) use self::rewrite::{RuleRun, run_rewrite_rules};
