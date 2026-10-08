//! Tests for `fhy_core::search_space`: variables, alternatives and choices,
//! spaces with their conditions and forbidden clauses, configurations with
//! their activity, validation and keys, the implementor contract,
//! equivalence, serialization, and their properties.

mod activity_stories;
mod alternative_stories;
mod choice_stories;
mod configuration_stories;
mod domain_stories;
mod equivalence_stories;
mod error_text_stories;
mod exploration_stories;
mod forbidden_stories;
mod implementor_stories;
mod integer_bound_stories;
mod measurement_properties;
mod measurement_serde_stories;
mod measurement_stories;
mod objective_stories;
mod oracle_stories;
mod properties;
mod recorder_stories;
mod rng_stories;
mod search_domain_stories;
mod serde_stories;
mod space_stories;
mod stream_error_text_stories;
mod trace_properties;
mod trace_serde_stories;
mod trace_stories;
mod tuple_category_stories;
mod variable_stories;
