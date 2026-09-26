//! Tests for `fhy_core::solver`: the hazard screen, the SMT-LIB2 lowering,
//! the facade and its encodings, and the process backend.

#[cfg(unix)]
mod process_stories;
mod screen_stories;
mod smt_lowering_stories;
mod solver_properties;
mod solver_stories;
