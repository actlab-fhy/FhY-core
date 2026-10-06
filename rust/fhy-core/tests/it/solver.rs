//! Tests for `fhy_core::solver`: the hazard screen, the SMT-LIB2 lowering,
//! the facade and its encodings, the process backend, and, under the `z3`
//! feature, the z3 backend. The SymPy backend lives in the binding, and so
//! do its stories.

mod error_stories;
mod ground_properties;
mod ground_stories;
mod ground_strategy_stories;
#[cfg(unix)]
mod process_stories;
mod screen_stories;
mod smt_lowering_stories;
mod solver_properties;
mod solver_stories;
#[cfg(feature = "z3")]
mod z3_stories;
