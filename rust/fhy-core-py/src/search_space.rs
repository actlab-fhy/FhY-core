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
//! - `wire.rs`: the V2 payloads, the foreign parts they hold, and the
//!   depth check of a payload before it reaches the core.
//! - `rng.rs`: `Rng`, the generator of a search's draws.
//! - `domain.rs`: `ChoiceDomain`, `OrderDomain`, `StridedRun` and
//!   `StridedDomain`, the domains of a search's steps.
//! - `trace.rs`: `TraceStep` and `Trace`, the steps of a run.
//! - `oracle.rs`: `PendingStep`, the core's oracles, the adapter of a
//!   Python oracle, and the reading of an oracle argument.
//! - `recorder.rs`: `Recorder`, one run driven from Python.
//! - `arguments.rs`: the constructors' argument readers, the depth guard of
//!   a value built from Python, and the seed through which the binding
//!   builds a public instance from a core value without checking it again.
//!
//! Every container keeps the Python objects it was built from and returns
//! them; one built from a core value builds its objects once and keeps
//! them. A Python subclass instance reaches the core through an adapter
//! that the container builds and whose slot the container owns, so the
//! cycle collector sees through it. `==` and `hash` are identity, except
//! `ConfigurationKey`'s, which are structural.
mod adapter;
mod alternative;
mod arguments;
mod choice;
mod configuration;
mod domain;
mod errors;
mod exploration;
mod kinds;
mod measurement;
mod oracle;
mod recorder;
mod rng;
mod space;
mod trace;
mod variable;
mod wire;

pub(crate) use alternative::{PyAlternativeBase, alternative_from_python, alternative_to_python};
pub(crate) use choice::{PyChoice, choice_from_python, choice_to_python};
pub(crate) use configuration::{PyConfiguration, PyConfigurationKey};
pub(crate) use domain::{
    PyChoiceDomain, PyOrderDomain, PyStridedDomain, PyStridedRun, step_domain_from_python,
    step_domain_to_python,
};
pub(crate) use kinds::{
    KIND_REGISTRY_ATTRIBUTE, PyKindRegistry, register_alternative_kind, register_oracle_kind,
    register_variable_kind,
};
pub(crate) use measurement::{PyMeasurement, PyObjective};
pub(crate) use oracle::{PyExhaustiveOracle, PyPendingStep, PyRandomOracle, PyReplayOracle};
pub(crate) use recorder::PyRecorder;
pub(crate) use rng::PyRng;
pub(crate) use space::{PyCondition, PyForbidden, PySpace};
pub(crate) use trace::{PyTrace, PyTraceStep};
pub(crate) use variable::{PyVariableBase, variable_from_python, variable_to_python};
