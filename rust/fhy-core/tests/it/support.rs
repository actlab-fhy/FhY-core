//! Helpers shared by the test modules: builders, strategies and toy IRs.
//!
//! Every helper is `pub(crate)`, so one that no test uses is reported as
//! dead code.

pub(crate) mod constraint;
pub(crate) mod error_text;
pub(crate) mod expression;
pub(crate) mod foreign;
pub(crate) mod hashing;
pub(crate) mod lambda;
pub(crate) mod measurement;
pub(crate) mod param;
pub(crate) mod pass_ir;
pub(crate) mod pattern;
pub(crate) mod provenance;
pub(crate) mod search;
pub(crate) mod search_space;
pub(crate) mod serde;
pub(crate) mod solver;
pub(crate) mod stack;
pub(crate) mod tree_ir;
pub(crate) mod types;
