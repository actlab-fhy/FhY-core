//! Pass pipelines, fixpoint groups, and the records of a pipeline run.

use std::borrow::Cow;
use std::fmt;
use std::num::NonZeroUsize;

use super::analysis::AnalysisCache;
use super::compiler_pass::{CompilerPass, run_lifecycle};
use super::context::PassContext;
use super::error::{PassError, VerificationPoint};
use super::preserved::PreservedAnalyses;
use super::validation::{ValidationManager, ValidatorRecord};
use crate::diagnostic::{Diagnostic, Note, ValidationReport};
use crate::identifier::{HasIdentifier, Identifier};
use crate::tree::NodeHandle;

/// The iteration budget of a new [`FixpointPassGroup`].
const DEFAULT_MAX_ITERATIONS: NonZeroUsize = NonZeroUsize::new(10).expect("ten is non-zero");

/// The record of one pass run in a pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassRunRecord {
    pass_name: Cow<'static, str>,
    changed: bool,
    skipped: bool,
    diagnostics: Vec<Diagnostic>,
    preserved: PreservedAnalyses,
}

impl PassRunRecord {
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
///
/// More kinds of items may be added, so a `match` on a record needs a
/// wildcard arm.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
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
    passes: Vec<Box<dyn CompilerPass<I> + Send + 'p>>,
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
    ///
    /// The pass is `Send`, so the group and its pipeline can move to another
    /// thread.
    pub fn add_pass(&mut self, pass: impl CompilerPass<I> + Send + 'p) {
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
    Pass(Box<dyn CompilerPass<I> + Send + 'p>),
    FixpointGroup(FixpointPassGroup<'p, I>),
}

/// Render a pass by its name and a group by its [`fmt::Debug`] form.
impl<I> fmt::Debug for PipelineItem<'_, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pass(pass) => f.debug_tuple("Pass").field(&pass.name()).finish(),
            Self::FixpointGroup(group) => fmt::Debug::fmt(group, f),
        }
    }
}

/// Return the error for verification rejecting the IR at `point` with
/// `report`, blaming the pass `pass_name`, whose run emitted `diagnostics`:
/// they end with an error diagnostic whose message is the error's.
fn build_verification_failure(
    pass_name: Cow<'static, str>,
    point: VerificationPoint,
    report: ValidationReport<ValidatorRecord>,
    mut diagnostics: Vec<Diagnostic>,
) -> PassError {
    let error = PassError::new_verification_failure(pass_name.clone(), point, report);
    diagnostics.push(Diagnostic::error(
        Note::with_other_kind(error.to_string()),
        pass_name,
    ));
    error.with_diagnostics(diagnostics)
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
    /// Verify `ir`, returning the verifier's report if it rejects `ir`, or
    /// `None` without a verifier or when the report has no error.
    ///
    /// The verifier's validators read analyses through the run's cache.
    fn find_rejection(&mut self, ir: &I) -> Option<ValidationReport<ValidatorRecord>> {
        let verifier = self.verifier.as_deref_mut()?;
        let report = verifier.validate_in(ir, &mut self.cache);
        report.has_errors().then_some(report)
    }

    /// Run `pass` over `input`, verify its output if it changed, and carry
    /// the preserved analyses over to the output.
    fn run_pass(
        &mut self,
        pass: &mut (dyn CompilerPass<I> + '_),
        input: &I,
    ) -> Result<(I, PassRunRecord), PassError> {
        let mut cx = PassContext::new(pass.name(), Some(&mut self.cache));
        let result = run_lifecycle(pass, input, &mut cx);
        let (pass_name, diagnostics) = cx.into_parts();
        let result = match result {
            Ok(result) => result,
            Err(error) => return Err(error.with_diagnostics(diagnostics)),
        };
        if result.changed {
            if let Some(report) = self.find_rejection(&result.output) {
                return Err(build_verification_failure(
                    pass_name,
                    VerificationPoint::Output,
                    report,
                    diagnostics,
                ));
            }
        }
        self.cache
            .transfer(input, &result.output, &result.preserved);
        let record = PassRunRecord {
            pass_name,
            changed: result.changed,
            skipped: result.skipped,
            diagnostics,
            preserved: result.preserved,
        };
        Ok((result.output, record))
    }

    /// Run `group` from `input` until an iteration changes nothing or its
    /// budget runs out, returning the group's output or failure together
    /// with its record.
    ///
    /// After a failing pass the record holds the iterations begun, the last
    /// listing the runs that completed; after the budget runs out it holds
    /// every iteration.
    fn run_fixpoint_group(
        &mut self,
        group: &mut FixpointPassGroup<'_, I>,
        input: I,
    ) -> (Result<I, PassError>, FixpointGroupRecord) {
        let mut current = input;
        let mut iteration_records = Vec::new();
        let mut converged = false;
        let mut failure = None;
        for iteration in 1..=group.max_iterations.get() {
            let mut changed = false;
            let mut pass_runs = Vec::with_capacity(group.passes.len());
            let iteration_result = group.passes.iter_mut().try_for_each(|pass| {
                let (output, record) = self.run_pass(pass.as_mut(), &current)?;
                changed |= record.changed;
                current = output;
                pass_runs.push(record);
                Ok(())
            });
            iteration_records.push(FixpointIterationRecord {
                iteration,
                changed,
                pass_runs,
            });
            if let Err(error) = iteration_result {
                failure = Some(error);
                break;
            }
            if !changed {
                converged = true;
                break;
            }
        }
        let result = match failure {
            Some(error) => Err(error),
            None if !converged && group.fail_on_non_convergence => Err(
                PassError::new_non_convergence(group.name.clone(), group.max_iterations),
            ),
            None => Ok(current),
        };
        let record = FixpointGroupRecord {
            group_name: group.name.clone(),
            iteration_records,
            converged,
        };
        (result, record)
    }
}

/// An ordered pipeline of passes and fixpoint groups over IR of type `I`.
///
/// A run feeds each item the IR the previous item produced and records every
/// pass run. Analysis results are cached per node for the run: a pass's
/// output gains the cached results its [`CompilerPass::preserved_analyses`]
/// preserves from its input, unless it has a result of its own for that
/// analysis. No result is dropped before the run ends, when the cache and
/// the node handles it holds are dropped. With a
/// [verifier](Self::set_verifier) set, the run also verifies its input and
/// every output a pass reports as changed.
///
/// A pipeline stores its passes and its verifier's validators as `Send`, so
/// it is `Send` whatever its IR type.
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
    /// Pass `&mut pass` to keep the pass and read its state after a run. The
    /// pass is `Send`, so the pipeline can move to another thread.
    pub fn add_pass(&mut self, pass: impl CompilerPass<I> + Send + 'p) {
        self.items.push(PipelineItem::Pass(Box::new(pass)));
    }

    /// Append `group` to the pipeline.
    pub fn add_fixpoint_group(&mut self, group: FixpointPassGroup<'p, I>) {
        self.items.push(PipelineItem::FixpointGroup(group));
    }

    /// Verify the IR of every run with `verifier`, replacing any verifier set
    /// before.
    ///
    /// A run then validates its input before the first pass, blaming that
    /// pass, and the output of every pass that reports a change, blaming the
    /// pass that produced it, even when it validated the same node before.
    /// The verifier's validators share the run's analysis cache with the
    /// passes.
    pub fn set_verifier(&mut self, verifier: ValidationManager<'p, I>) {
        self.verifier = Some(verifier);
    }

    /// Run the pipeline over `ir`.
    ///
    /// # Errors
    ///
    /// Returns the [`PassError`] of the first pass that fails; a
    /// [`Verification`](super::PassErrorKind::Verification) failure holding
    /// the report if the verifier rejects IR; and a
    /// [`NonConvergence`](super::PassErrorKind::NonConvergence) failure if a
    /// fixpoint group that fails on non-convergence does not converge. The
    /// error's [`records`](PassError::records) are those of the work
    /// completed before the failure.
    pub fn run(&mut self, ir: &I) -> Result<PassManagerResult<I>, PassError> {
        let mut run = PipelineRun {
            cache: AnalysisCache::new(),
            verifier: self.verifier.as_mut(),
        };
        if let Some(first_pass_name) = find_first_pass_name(&self.items) {
            if let Some(report) = run.find_rejection(ir) {
                return Err(build_verification_failure(
                    first_pass_name,
                    VerificationPoint::Input,
                    report,
                    Vec::new(),
                ));
            }
        }
        let mut current = ir.clone();
        let mut records = Vec::with_capacity(self.items.len());
        for item in &mut self.items {
            match item {
                PipelineItem::Pass(pass) => match run.run_pass(pass.as_mut(), &current) {
                    Ok((output, record)) => {
                        current = output;
                        records.push(PipelineRecord::Pass(record));
                    }
                    Err(error) => return Err(error.with_records(records)),
                },
                PipelineItem::FixpointGroup(group) => {
                    let (result, record) = run.run_fixpoint_group(group, current);
                    records.push(PipelineRecord::FixpointGroup(record));
                    match result {
                        Ok(output) => current = output,
                        Err(error) => return Err(error.with_records(records)),
                    }
                }
            }
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

/// Pipelines, fixpoint groups and verifiers are `Send` for every IR type,
/// even one that is not `Send` itself.
const _: () = {
    const fn assert_send<T: Send>() {}
    const fn assert_pipelines_are_send<I>() {
        assert_send::<PassManager<'static, I>>();
        assert_send::<FixpointPassGroup<'static, I>>();
        assert_send::<ValidationManager<'static, I>>();
    }
    assert_pipelines_are_send::<std::rc::Rc<()>>();
};
