//! Pass pipelines, fixpoint groups, and the records of a pipeline run.

use std::borrow::Cow;
use std::fmt;
use std::num::NonZeroUsize;

use super::analysis::AnalysisCache;
use super::compiler_pass::{CompilerPass, run_lifecycle};
use super::context::PassContext;
use super::error::PassError;
use super::preserved::{AnalysisId, PreservedAnalyses};
use super::validation::ValidationManager;
use crate::diagnostic::{Diagnostic, Note};
use crate::identifier::{HasIdentifier, Identifier};
use crate::tree::NodeHandle;

/// The iteration budget of a new [`FixpointPassGroup`], as in the Python
/// pass infrastructure.
const DEFAULT_MAX_ITERATIONS: NonZeroUsize = NonZeroUsize::new(10).expect("ten is non-zero");

/// The record of one pass run in a pipeline or a validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassRunRecord {
    pass_name: Cow<'static, str>,
    changed: bool,
    skipped: bool,
    diagnostics: Vec<Diagnostic>,
    preserved: PreservedAnalyses,
}

impl PassRunRecord {
    /// Create the record of a run of the pass `pass_name`.
    pub(super) fn new(
        pass_name: Cow<'static, str>,
        changed: bool,
        skipped: bool,
        diagnostics: Vec<Diagnostic>,
        preserved: PreservedAnalyses,
    ) -> Self {
        Self {
            pass_name,
            changed,
            skipped,
            diagnostics,
            preserved,
        }
    }

    /// Return the name of the pass that ran.
    #[must_use]
    pub fn pass_name(&self) -> &str {
        &self.pass_name
    }

    /// Return whether the run changed the IR.
    #[must_use]
    pub fn is_changed(&self) -> bool {
        self.changed
    }

    /// Return whether the pass skipped the run, so
    /// [`CompilerPass::run`] was not called.
    #[must_use]
    pub fn is_skipped(&self) -> bool {
        self.skipped
    }

    /// Return the diagnostics the run emitted, in emission order.
    #[must_use]
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Return the analyses the run left valid for its output.
    #[must_use]
    pub fn preserved_analyses(&self) -> &PreservedAnalyses {
        &self.preserved
    }
}

/// The record of one iteration of a fixpoint group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixpointIterationRecord {
    iteration: usize,
    changed: bool,
    pass_runs: Vec<PassRunRecord>,
}

impl FixpointIterationRecord {
    /// Return the iteration's 1-based number.
    #[must_use]
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Return whether any pass changed the IR in this iteration.
    #[must_use]
    pub fn is_changed(&self) -> bool {
        self.changed
    }

    /// Return the records of the iteration's pass runs, in run order.
    #[must_use]
    pub fn pass_runs(&self) -> &[PassRunRecord] {
        &self.pass_runs
    }
}

/// The record of a fixpoint group's run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixpointGroupRecord {
    group_name: Identifier,
    iteration_records: Vec<FixpointIterationRecord>,
    converged: bool,
}

impl FixpointGroupRecord {
    /// Return the group's name.
    #[must_use]
    pub fn group_name(&self) -> &Identifier {
        &self.group_name
    }

    /// Return the records of the iterations run, in order.
    #[must_use]
    pub fn iteration_records(&self) -> &[FixpointIterationRecord] {
        &self.iteration_records
    }

    /// Return whether the group reached an iteration in which no pass
    /// changed the IR.
    #[must_use]
    pub fn is_converged(&self) -> bool {
        self.converged
    }

    /// Return the number of iterations run.
    #[must_use]
    pub fn iterations(&self) -> usize {
        self.iteration_records.len()
    }
}

/// The record of one item of a pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineRecord {
    /// A single pass ran.
    Pass(PassRunRecord),
    /// A fixpoint group ran.
    FixpointGroup(FixpointGroupRecord),
}

/// The result of a pipeline run: the final IR, one record per item, and the
/// run's statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassManagerResult<I> {
    output: I,
    records: Vec<PipelineRecord>,
}

impl<I> PassManagerResult<I> {
    /// Return the final IR.
    #[must_use]
    pub fn output(&self) -> &I {
        &self.output
    }

    /// Return the final IR, consuming the result.
    #[must_use]
    pub fn into_output(self) -> I {
        self.output
    }

    /// Return the records of the pipeline's items, in pipeline order.
    #[must_use]
    pub fn records(&self) -> &[PipelineRecord] {
        &self.records
    }

    /// Return the record of every pass run, in run order, with the runs of
    /// each fixpoint group's iterations in place of the group.
    pub fn pass_runs(&self) -> impl Iterator<Item = &PassRunRecord> + '_ {
        self.records.iter().flat_map(|record| {
            let (pass, group) = match record {
                PipelineRecord::Pass(pass) => (Some(pass), None),
                PipelineRecord::FixpointGroup(group) => (None, Some(group)),
            };
            let group_runs = group
                .into_iter()
                .flat_map(|group| &group.iteration_records)
                .flat_map(|iteration| &iteration.pass_runs);
            pass.into_iter().chain(group_runs)
        })
    }

    /// Return the number of pass runs the pipeline made, not counting the
    /// runs a pass skipped.
    ///
    /// The count covers this run only; nothing is counted across runs.
    #[must_use]
    pub fn run_count(&self) -> usize {
        self.pass_runs().filter(|run| !run.skipped).count()
    }
}

/// A pass sequence a pipeline repeats until no pass changes the IR.
///
/// Each iteration runs every pass in order. The group converges at the first
/// iteration in which no pass reports a change, and gives up after
/// [`max_iterations`](Self::max_iterations) iterations.
pub struct FixpointPassGroup<'p, I> {
    name: Identifier,
    max_iterations: NonZeroUsize,
    fail_on_non_convergence: bool,
    passes: Vec<Box<dyn CompilerPass<I> + 'p>>,
}

impl<'p, I> FixpointPassGroup<'p, I> {
    /// Create the empty group `name` with a budget of 10 iterations that
    /// fails when it does not converge.
    #[must_use]
    pub fn new(name: Identifier) -> Self {
        Self {
            name,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            fail_on_non_convergence: true,
            passes: Vec::new(),
        }
    }

    /// Return the group with an iteration budget of `max_iterations`.
    #[must_use]
    pub fn with_max_iterations(self, max_iterations: NonZeroUsize) -> Self {
        Self {
            max_iterations,
            ..self
        }
    }

    /// Return the group that fails a pipeline when it does not converge if
    /// `fail_on_non_convergence` holds, and otherwise hands on the IR of its
    /// last iteration.
    #[must_use]
    pub fn with_fail_on_non_convergence(self, fail_on_non_convergence: bool) -> Self {
        Self {
            fail_on_non_convergence,
            ..self
        }
    }

    /// Append `pass` to the group.
    pub fn add_pass(&mut self, pass: impl CompilerPass<I> + 'p) {
        self.passes.push(Box::new(pass));
    }

    /// Return the group's name.
    #[must_use]
    pub fn name(&self) -> &Identifier {
        &self.name
    }

    /// Return the iteration budget.
    #[must_use]
    pub fn max_iterations(&self) -> NonZeroUsize {
        self.max_iterations
    }

    /// Return whether the group fails a pipeline when it does not converge.
    #[must_use]
    pub fn fails_on_non_convergence(&self) -> bool {
        self.fail_on_non_convergence
    }
}

impl<I> HasIdentifier for FixpointPassGroup<'_, I> {
    fn identifier(&self) -> &Identifier {
        &self.name
    }
}

/// Render the group's configuration and the names of its passes.
impl<I> fmt::Debug for FixpointPassGroup<'_, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pass_names: Vec<Cow<'static, str>> =
            self.passes.iter().map(CompilerPass::name).collect();
        f.debug_struct("FixpointPassGroup")
            .field("name", &self.name)
            .field("max_iterations", &self.max_iterations)
            .field("fail_on_non_convergence", &self.fail_on_non_convergence)
            .field("passes", &pass_names)
            .finish()
    }
}

/// One item of a pipeline.
enum PipelineItem<'p, I> {
    Pass(Box<dyn CompilerPass<I> + 'p>),
    FixpointGroup(FixpointPassGroup<'p, I>),
}

/// Render a pass by its name and a group by its [`fmt::Debug`] form.
impl<I> fmt::Debug for PipelineItem<'_, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineItem::Pass(pass) => f.debug_tuple("Pass").field(&pass.name()).finish(),
            PipelineItem::FixpointGroup(group) => fmt::Debug::fmt(group, f),
        }
    }
}

/// The key the verification report of a node is cached under.
struct VerificationReportKey;

/// Where verification looks at IR.
#[derive(Debug, Clone, Copy)]
enum VerificationPoint {
    /// The pipeline's input, before its first pass.
    Input,
    /// A pass's changed output.
    Output,
}

impl VerificationPoint {
    /// Return the phrase that reports a rejection at this point.
    fn describe_rejection(self) -> &'static str {
        match self {
            VerificationPoint::Input => "rejected input IR",
            VerificationPoint::Output => "produced invalid output IR",
        }
    }
}

/// Return the name of the first pass `items` will run, if any.
fn find_first_pass_name<I>(items: &[PipelineItem<'_, I>]) -> Option<Cow<'static, str>> {
    items.iter().find_map(|item| match item {
        PipelineItem::Pass(pass) => Some(pass.name()),
        PipelineItem::FixpointGroup(group) => group.passes.first().map(CompilerPass::name),
    })
}

/// The state of one pipeline run: its analysis cache and its verifier.
struct PipelineRun<'r, 'p, I> {
    cache: AnalysisCache,
    verifier: Option<&'r mut ValidationManager<'p, I>>,
}

impl<I: NodeHandle> PipelineRun<'_, '_, I> {
    /// Verify `ir` at `point`, blaming the pass `pass_name`, whose run
    /// emitted `diagnostics`.
    ///
    /// The report is cached for the node, so a node is verified at most once
    /// per run while its results are cached.
    fn verify(
        &mut self,
        ir: &I,
        point: VerificationPoint,
        pass_name: Cow<'static, str>,
        diagnostics: &[Diagnostic],
    ) -> Result<(), PassError> {
        let Some(verifier) = self.verifier.as_deref_mut() else {
            return Ok(());
        };
        let report =
            self.cache
                .get_or_insert_with(ir, AnalysisId::of::<VerificationReportKey>(), || {
                    verifier.validate(ir)
                });
        if !report.has_errors() {
            return Ok(());
        }
        let message = format!(
            "Pass \"{pass_name}\" {}: verification reported {} error(s).",
            point.describe_rejection(),
            report.errors().count()
        );
        let mut failure_diagnostics = diagnostics.to_vec();
        failure_diagnostics.push(
            Diagnostic::error(Note::with_other_kind(message.clone()), pass_name.clone())
                .with_detail(report.to_string()),
        );
        Err(PassError::new_verification_failure(
            pass_name,
            message,
            (*report).clone(),
            failure_diagnostics,
        ))
    }

    /// Run `pass` over `input`, verify its output if it changed, and carry
    /// the preserved analyses over to the output.
    fn run_pass(
        &mut self,
        pass: &mut (dyn CompilerPass<I> + '_),
        input: &I,
    ) -> Result<(I, PassRunRecord), PassError> {
        let mut cx = PassContext::new(pass.name(), Some(&mut self.cache));
        let result = run_lifecycle(pass, input, &mut cx)?;
        let (pass_name, diagnostics) = cx.into_parts();
        if result.changed {
            self.verify(
                &result.output,
                VerificationPoint::Output,
                pass_name.clone(),
                &diagnostics,
            )?;
        }
        self.cache
            .transfer(input, &result.output, &result.preserved);
        let record = PassRunRecord::new(
            pass_name,
            result.changed,
            result.skipped,
            diagnostics,
            result.preserved,
        );
        Ok((result.output, record))
    }

    /// Run `group` from `input` until an iteration changes nothing or its
    /// budget runs out.
    fn run_fixpoint_group(
        &mut self,
        group: &mut FixpointPassGroup<'_, I>,
        input: I,
    ) -> Result<(I, FixpointGroupRecord), PassError> {
        let mut current = input;
        let mut iteration_records = Vec::new();
        let mut converged = false;
        for iteration in 1..=group.max_iterations.get() {
            let mut changed = false;
            let mut pass_runs = Vec::with_capacity(group.passes.len());
            for pass in &mut group.passes {
                let (output, record) = self.run_pass(pass.as_mut(), &current)?;
                changed |= record.changed;
                current = output;
                pass_runs.push(record);
            }
            iteration_records.push(FixpointIterationRecord {
                iteration,
                changed,
                pass_runs,
            });
            if !changed {
                converged = true;
                break;
            }
        }
        if !converged && group.fail_on_non_convergence {
            return Err(PassError::new_non_convergence(format!(
                "Fixpoint group \"{}\" did not converge in {} iterations.",
                group.name, group.max_iterations
            )));
        }
        let record = FixpointGroupRecord {
            group_name: group.name.clone(),
            iteration_records,
            converged,
        };
        Ok((current, record))
    }
}

/// An ordered pipeline of passes and fixpoint groups over IR of type `I`.
///
/// A run feeds each item the IR the previous item produced and records every
/// pass run. Analysis results are cached per node for the run: a pass's
/// output inherits the cached results its
/// [`CompilerPass::preserved_analyses`] preserves, and the cache is dropped
/// when the run ends. With a verifier set, the run also verifies its input
/// and every output a pass reports as changed.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
///
/// use fhy_core::identifier::Identifier;
/// use fhy_core::pass::{CompilerPass, PassContext, PassFailure, PassManager};
/// use fhy_core::tree::{NodeHandle, NodeIdentity};
///
/// #[derive(Clone, Debug)]
/// struct Value(Arc<i64>);
///
/// impl NodeHandle for Value {
///     fn identity(&self) -> NodeIdentity {
///         NodeIdentity::of_arc(&self.0)
///     }
/// }
///
/// struct Double;
///
/// impl CompilerPass<Value> for Double {
///     fn run(&mut self, ir: &Value, _cx: &mut PassContext<'_>) -> Result<Value, PassFailure> {
///         Ok(Value(Arc::new(*ir.0 * 2)))
///     }
///
///     fn did_change(&mut self, input: &Value, output: &Value) -> Result<bool, PassFailure> {
///         Ok(input.0 != output.0)
///     }
/// }
///
/// let mut manager = PassManager::new(Identifier::new("pipeline"));
/// manager.add_pass(Double);
/// manager.add_pass(Double);
///
/// let result = manager.run(&Value(Arc::new(3)))?;
///
/// assert_eq!(*result.output().0, 12);
/// assert_eq!(result.records().len(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct PassManager<'p, I> {
    name: Identifier,
    items: Vec<PipelineItem<'p, I>>,
    verifier: Option<ValidationManager<'p, I>>,
}

impl<'p, I: NodeHandle> PassManager<'p, I> {
    /// Create the empty pipeline `name` without a verifier.
    #[must_use]
    pub fn new(name: Identifier) -> Self {
        Self {
            name,
            items: Vec::new(),
            verifier: None,
        }
    }

    /// Return the pipeline's name.
    #[must_use]
    pub fn name(&self) -> &Identifier {
        &self.name
    }

    /// Append `pass` to the pipeline.
    ///
    /// Pass `&mut pass` to keep the pass and read its state after a run.
    pub fn add_pass(&mut self, pass: impl CompilerPass<I> + 'p) {
        self.items.push(PipelineItem::Pass(Box::new(pass)));
    }

    /// Append `group` to the pipeline.
    pub fn add_fixpoint_group(&mut self, group: FixpointPassGroup<'p, I>) {
        self.items.push(PipelineItem::FixpointGroup(group));
    }

    /// Verify the IR of every run with `verifier`, replacing any verifier set
    /// before.
    ///
    /// A run then validates its input once before the first pass, blaming
    /// that pass, and validates the output of every pass that reports a
    /// change, blaming the pass that produced it. Unchanged IR is not
    /// validated again within a run.
    pub fn set_verifier(&mut self, verifier: ValidationManager<'p, I>) {
        self.verifier = Some(verifier);
    }

    /// Run the pipeline over `ir`.
    ///
    /// # Errors
    ///
    /// Returns the [`PassError`] of the first pass that fails, a validation
    /// failure carrying the report if verification rejects IR, and an
    /// execution failure if a fixpoint group that fails on non-convergence
    /// does not converge. The failure message then reads
    /// `Fixpoint group "<name>" did not converge in <n> iterations.`.
    pub fn run(&mut self, ir: &I) -> Result<PassManagerResult<I>, PassError> {
        let mut run = PipelineRun {
            cache: AnalysisCache::new(),
            verifier: self.verifier.as_mut(),
        };
        if let Some(first_pass_name) = find_first_pass_name(&self.items) {
            run.verify(ir, VerificationPoint::Input, first_pass_name, &[])?;
        }
        let mut current = ir.clone();
        let mut records = Vec::with_capacity(self.items.len());
        for item in &mut self.items {
            let record = match item {
                PipelineItem::Pass(pass) => {
                    let (output, record) = run.run_pass(pass.as_mut(), &current)?;
                    current = output;
                    PipelineRecord::Pass(record)
                }
                PipelineItem::FixpointGroup(group) => {
                    let (output, record) = run.run_fixpoint_group(group, current)?;
                    current = output;
                    PipelineRecord::FixpointGroup(record)
                }
            };
            records.push(record);
        }
        Ok(PassManagerResult {
            output: current,
            records,
        })
    }
}

impl<I: NodeHandle> Default for PassManager<'_, I> {
    /// Create the empty pipeline `pipeline` without a verifier.
    fn default() -> Self {
        Self::new(Identifier::new("pipeline"))
    }
}

impl<I> HasIdentifier for PassManager<'_, I> {
    fn identifier(&self) -> &Identifier {
        &self.name
    }
}

/// Render the pipeline's name, its items, and its verifier.
impl<I> fmt::Debug for PassManager<'_, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PassManager")
            .field("name", &self.name)
            .field("items", &self.items)
            .field("verifier", &self.verifier)
            .finish()
    }
}
