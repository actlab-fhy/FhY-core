//! `PyO3` bindings for [`fhy_core::search_space`]: the classes of
//! `fhy_core.search_space`.
//!
//! - `variable.rs` and `alternative.rs`: `Variable` and `Alternative`,
//!   subclassable bases whose public classes Python code extends with data
//!   of its own and the `extension_*` hooks.
//! - `choice.rs`, `space.rs` and `configuration.rs`: `Choice`, `Condition`,
//!   `Forbidden`, `Space`, `Configuration` and `ConfigurationKey`.
//! - `adapter.rs`: the core parts of a Python subclass instance, which call
//!   its hooks.
//! - `kinds.rs`: the registry of the `Variable` and `Alternative` kinds a
//!   downstream Rust crate defines, append-only module state of the
//!   extension.
//! - `errors.rs`: the exceptions the classes raise for the core's errors.
//! - `wire.rs`: the foreign parts of a payload, and the depth check of a
//!   payload before it reaches the core.
//!
//! Every container keeps the Python objects it was built from and returns
//! them. A Python subclass instance reaches the core through an adapter
//! that the container builds and whose slot the container owns, so the
//! cycle collector sees through it. `==` and `hash` are identity, except
//! `ConfigurationKey`'s, which are structural.
#![expect(
    dead_code,
    unused_variables,
    clippy::todo,
    clippy::needless_pass_by_value,
    reason = "interface stub; bodies are todo!() until implementation"
)]

mod adapter;
mod alternative;
mod choice;
mod configuration;
mod errors;
mod kinds;
mod space;
mod variable;
mod wire;

pub(crate) use alternative::{PyAlternativeBase, alternative_from_python, alternative_to_python};
pub(crate) use choice::{PyChoice, choice_from_python, choice_to_python};
pub(crate) use configuration::{PyConfiguration, PyConfigurationKey};
pub(crate) use kinds::{
    KIND_REGISTRY_ATTRIBUTE, PyKindRegistry, register_alternative_kind, register_variable_kind,
};
pub(crate) use space::{PyCondition, PyForbidden, PySpace};
pub(crate) use variable::{PyVariableBase, variable_from_python, variable_to_python};
