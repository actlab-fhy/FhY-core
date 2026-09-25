//! `PyO3` classes and functions for [`fhy_core::expression::pattern`]: the
//! bases of `fhy_core.symbolic.expression.pattern`'s classes (pattern P2
//! for captures, patterns, bindings and rules; P3 for the Python callbacks
//! and the `Rule` ABC), and the rewrite walk.
//!
//! The Python pattern API takes the Rust core's semantics (the S5 decisions
//! of `docs/design/python-switch.md`): captures are identity handles,
//! bindings are keyed by them, the pattern kinds are closed, degenerate
//! patterns build and match nothing, a rule returning the node it matched
//! declines, and the walk rewrites a shared node once, walks trees of any
//! depth, and raises the core's errors.
//!
//! Callbacks, predicates, guards, rewrites and Python rules, are called
//! from the core with Rust nodes; they receive the Python node objects of
//! the tree being matched or rewritten, which a per-call table finds or
//! builds (`objects`).

mod bindings;
mod capture;
mod kinds;
mod objects;
mod rules;
mod walk;

pub(crate) use bindings::PyMatchBindings;
pub(crate) use capture::PyCapture;
pub(crate) use kinds::{
    PyAlternativesPattern, PyBinaryExpressionPattern, PyCallExpressionPattern, PyCapturePattern,
    PyIdentifierPattern, PyLiteralPattern, PyLogicalExpressionPattern, PyPattern,
    PyPiecewiseExpressionPattern, PyPredicatePattern, PyUnaryExpressionPattern, PyWildcardPattern,
};
pub(crate) use rules::{PyFiredRule, PyRewriteRule, PyRuleBase};
pub(crate) use walk::apply_rewrite_rules;
