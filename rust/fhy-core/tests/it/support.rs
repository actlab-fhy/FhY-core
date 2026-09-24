//! Helpers shared by the test modules: builders, strategies and toy IRs.
//!
//! Every helper is `pub(crate)`, so one that no test uses is reported as
//! dead code.

pub(crate) mod expression;
pub(crate) mod hashing;
pub(crate) mod pass_ir;
pub(crate) mod pattern;
pub(crate) mod provenance;
pub(crate) mod stack;
pub(crate) mod tree_ir;
