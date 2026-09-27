//! Integration tests for `fhy-core`, through its public API only.
//!
//! One binary, so the crate and the dev-dependencies link once and a helper
//! no test uses is reported as dead code. Every test here shares one
//! process with every other: none clears a process-global registry, and
//! none moves the identifier counter further than the ids it allocates.

mod constraint;
mod diagnostic_stories;
mod expression;
mod foreign_stories;
mod identifier_stories;
mod interned;
mod lattice;
mod param;
mod pass;
mod provenance_diagnostic_properties;
mod provenance_stories;
mod scope_stack_properties;
mod scope_stories;
mod serde_format_stories;
mod solver;
mod stack_stories;
mod support;
mod symbol_table;
mod tag_type_stories;
mod term;
mod tree;
mod types;
