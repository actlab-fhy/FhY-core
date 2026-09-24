//! Compiler passes, the pipelines that run them, and the analyses they share.
//!
//! A [`CompilerPass`] transforms IR through a guarded lifecycle of hooks
//! (validation, a skip decision, the run itself, change detection, and the
//! analyses the run preserves). Each hook receives a [`PassContext`] that
//! collects the run's diagnostics and serves [`Analysis`] results. A pass
//! runs on its own with [`ExecutePass::execute`], or in a [`PassManager`]
//! pipeline, possibly repeated to a fixpoint in a [`FixpointPassGroup`]. A
//! pipeline caches analysis results per node for the length of one run,
//! keyed by the [`NodeIdentity`](crate::tree::NodeIdentity) a
//! [`NodeHandle`](crate::tree::NodeHandle) reports, and can verify
//! the IR between passes with a [`ValidationManager`], which runs
//! [`Validator`]s collect-all into one report. [`PassValidator`] runs a pass
//! as a validator.
//!
//! Hook errors become a [`PassError`] naming the pass and the hook. A pass
//! is named by [`CompilerPass::name`], by default
//! [`short_type_name`] of its type. A [`PassRegistry`] is an owned value
//! that builds passes by name, and a pipeline run reports its own run
//! statistics in its [`PassManagerResult`]; the module keeps no global
//! state.
//!
//! [`WalkPass`] and [`RewritePass`] turn the traversals of
//! [`crate::tree`] into passes.

mod adapters;
mod analysis;
mod compiler_pass;
mod context;
mod error;
mod manager;
mod preserved;
mod registry;
mod validation;

pub use adapters::{RewritePass, WalkPass};
pub use analysis::Analysis;
pub use compiler_pass::{CompilerPass, ExecutePass, PassFailure, PassOutcome, short_type_name};
pub use context::PassContext;
pub use error::{PassError, PassHook};
pub use manager::{
    FixpointGroupRecord, FixpointIterationRecord, FixpointPassGroup, PassManager,
    PassManagerResult, PassRunRecord, PipelineRecord,
};
pub use preserved::{AnalysisId, PreservedAnalyses};
pub use registry::{CreatePassError, PassInfo, PassRegistrationError, PassRegistry};
pub use validation::{PassValidator, ValidationManager, Validator, ValidatorRecord};
