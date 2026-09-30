//! `PyO3` classes for [`fhy_core::pass`]: the bases of the pass
//! infrastructure's public classes in `fhy_core.pass_infrastructure`.
//!
//! `CompilerPass`, `Analysis` and `Validator` are Rust traits exposed as
//! Python abstract classes: their `PyO3` bases are
//! `CompilerPassBase`, `AnalysisBase` and `ValidatorBase`, and a Python
//! subclass is driven from Rust through an adapter that implements the
//! core's trait by calling the Python hooks. The hook names stay Python;
//! the adapter maps `should_run` and `get_noop_output` onto the core's
//! `skip`, `run_pass` onto `run`, and `get_preserved_analyses` onto
//! `preserved_analyses`, and runs the defaults of the hooks a class does
//! not override in Rust.
//!
//! The managers, `PreservedAnalyses`, `PassResult` and the records are
//! Rust-backed classes. A pipeline run builds the core's
//! `PassManager` over [`PyIr`](ir::PyIr), the one type-erased IR whose
//! identity is the Python object's, runs it, and converts what comes back:
//! each Rust diagnostic a Python hook reported returns as the object it
//! reported, and a failure becomes `PassValidationError` or
//! `PassExecutionError` with the core's message, naming the Python hook.
//!
//! A Python hook takes no context argument. While it runs, the adapter
//! keeps a frame on a thread-local stack that `report`, `get_analysis` and
//! `get_analysis_manager` find by the pass object, so borrowed Rust state
//! never reaches Python: the frame holds the diagnostics the hook reports
//! and a detached handle to the run's analysis cache, and expires when the
//! hook returns.
//!
//! The verification registry of the Python API lives in the extension's
//! module state, and a pipeline's default verifier looks each IR up in it.

mod analysis;
mod compiler_pass;
mod context;
mod convert;
mod error;
mod ir;
mod manager;
mod records;
mod scope;
mod validation;
mod verification;

pub(crate) use analysis::{PyAnalysisBase, PyPreservedAnalyses};
pub(crate) use compiler_pass::{PyCompilerPassBase, refuse_unused_arguments};
pub(crate) use context::PyAnalysisManager;
pub(crate) use manager::{PyFixpointPassGroup, PyPassManager};
pub(crate) use records::{
    PyFixpointGroupRecord, PyFixpointIterationRecord, PyPassManagerResult, PyPassResult,
    PyPassRunRecord, PyValidatorRecord,
};
pub(crate) use validation::{PyValidationManager, PyValidatorBase};
pub(crate) use verification::{
    PyVerificationRegistry, REGISTRY_ATTRIBUTE, get_verification_passes_for,
    register_verification_pass, run_verification,
};
