//! Tests for `fhy_core::expression`: expressions, their builders, analyses, wire
//! form, text forms, built-ins, the function registry and inlining,
//! evaluation and folding, passes, and the patterns over them.

mod affine_properties;
mod affine_stories;
mod builders_stories;
mod builtins_stories;
mod error_text_stories;
#[cfg(feature = "ndarray")]
mod evaluate_array_stories;
mod evaluate_properties;
mod evaluate_stories;
mod fold_stories;
mod inline_stories;
mod literal_stories;
mod node_stories;
mod pass_stories;
mod pattern;
mod pprint_properties;
mod pprint_stories;
mod properties;
mod registry_properties;
mod registry_stories;
mod screen_stories;
mod tree_stories;
mod vocabulary_stories;
mod wire_stories;
