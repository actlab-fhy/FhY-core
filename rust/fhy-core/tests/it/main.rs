//! Integration tests for `fhy-core`, through its public API only.
//!
//! One binary, so the crate and the dev-dependencies link once and a helper
//! no test uses is reported as dead code. Every test here shares one
//! process with every other: none clears a process-global registry, and
//! none moves the identifier counter further than the ids it allocates.

mod diagnostic_stories;
mod expr;
mod identifier_stories;
mod interned;
mod pass;
mod payload_form_stories;
mod provenance_diagnostic_properties;
mod provenance_stories;
mod support;
mod tag_type_stories;
mod tree;
