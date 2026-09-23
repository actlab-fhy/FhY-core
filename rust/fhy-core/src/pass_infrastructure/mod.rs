//! Compiler passes, the pipelines that run them, and the analyses they share.
//!
//! A [`CompilerPass`] transforms IR through a guarded lifecycle of hooks
//! (validation, a skip decision, the run itself, change detection, and the
//! analyses the run preserves). Each hook receives a [`PassContext`] that
//! collects the run's diagnostics and serves [`Analysis`] results. A pass
//! runs on its own with [`ExecutePass::execute`], or in a [`PassManager`]
//! pipeline, possibly repeated to a fixpoint in a [`FixpointPassGroup`]. A
//! pipeline caches analysis results per node for the length of one run,
//! keyed by the [`NodeIdentity`] a [`NodeHandle`] reports, and can verify
//! the IR between passes with a [`ValidationManager`], which runs validation
//! passes collect-all into one report.
//!
//! Hook errors become a [`PassError`] naming the pass and the hook. The
//! process-wide registry ([`register_pass`], [`create_pass`],
//! [`registered_passes`]) names pass types, and the run counters
//! ([`run_count`], [`run_count_of`], [`total_run_count`]) count every pass
//! run that was not skipped.
//!
//! Tree-shaped IR ([`Tree`]) adds two traversals: [`walk_tree`] drives a
//! [`TreeVisitor`], and [`rewrite_tree`] drives a [`Rewriter`];
//! [`WalkPass`] and [`RewritePass`] turn them into passes.

mod analysis;
mod context;
mod error;
mod manager;
mod pass;
mod preserved;
mod registry;
mod tree;
mod validation;

pub use analysis::{Analysis, NodeHandle, NodeIdentity};
pub use context::PassContext;
pub use error::{PassError, PassHook, PassRegistrationError};
pub use manager::{
    FixpointGroupRecord, FixpointIterationRecord, FixpointPassGroup, PassManager,
    PassManagerResult, PassRunRecord, PipelineRecord,
};
pub use pass::{CompilerPass, ExecutePass, PassFailure, PassOutcome};
pub use preserved::{AnalysisId, PreservedAnalyses};
pub use registry::{
    PassInfo, create_pass, register_pass, registered_passes, run_count, run_count_of,
    total_run_count,
};
pub use tree::{
    RewritePass, RewriteTreeError, Rewriter, TraversalOrder, Tree, TreeVisitor, WalkPass,
    rewrite_tree, walk_tree,
};
pub use validation::ValidationManager;
