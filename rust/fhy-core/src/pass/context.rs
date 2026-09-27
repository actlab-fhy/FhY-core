//! The context a pass run hands to every hook: the diagnostics sink and
//! access to analyses.

use std::borrow::Cow;
use std::sync::Arc;

use super::analysis::{Analysis, AnalysisCache};
use super::detached::{DetachedAnalyses, Detachment};
use super::preserved::AnalysisId;
use crate::diagnostic::{Diagnostic, DiagnosticLevel, Note};
use crate::tree::NodeHandle;

/// The diagnostics sink and analysis access of one pass run.
///
/// A pass receives the context in each lifecycle hook, and a validator in
/// its check. Diagnostics it reports end up in the run's
/// [`PassOutcome`](super::PassOutcome), or in the
/// [`PassError`](super::PassError) if the run fails. Under a
/// [`PassManager`](super::PassManager), analysis results are cached for the
/// whole pipeline run, its verifier included; a pass executed on its own
/// computes them afresh on every request.
#[derive(Debug)]
pub struct PassContext<'a> {
    pass_name: Cow<'static, str>,
    diagnostics: Vec<Diagnostic>,
    analyses: Option<&'a mut AnalysisCache>,
}

impl<'a> PassContext<'a> {
    /// Create the context for a run of the pass `pass_name`, caching analyses
    /// in `analyses` when one is given.
    pub(super) fn new(
        pass_name: Cow<'static, str>,
        analyses: Option<&'a mut AnalysisCache>,
    ) -> Self {
        Self {
            pass_name,
            diagnostics: Vec::new(),
            analyses,
        }
    }

    pub(super) fn into_parts(self) -> (Cow<'static, str>, Vec<Diagnostic>) {
        (self.pass_name, self.diagnostics)
    }

    /// Return the name of the running pass as the diagnostics' source holds
    /// it, without copying a borrowed name.
    pub(super) fn shared_pass_name(&self) -> Cow<'static, str> {
        self.pass_name.clone()
    }

    /// Record `diagnostic` as given, its source included.
    ///
    /// [`report_text`](Self::report_text) records a text attributed to the
    /// running pass instead.
    pub fn report(&mut self, diagnostic: Diagnostic) {
        self.diagnostics.push(diagnostic);
    }

    /// Record the diagnostic with text `message` at `level`, as a note of
    /// the uncategorized kind attributed to the running pass, with optional
    /// `detail`.
    pub fn report_text(
        &mut self,
        level: DiagnosticLevel,
        message: impl Into<String>,
        detail: Option<String>,
    ) {
        let diagnostic = Diagnostic::new(
            level,
            Note::with_other_kind(message),
            self.pass_name.clone(),
        );
        self.report(match detail {
            Some(detail) => diagnostic.with_detail(detail),
            None => diagnostic,
        });
    }

    /// Return the result of the analysis `A` for `ir`.
    ///
    /// Under a [`PassManager`](super::PassManager) the result is cached per
    /// node for the pipeline run, and a pass's output inherits it from the
    /// pass's input when the pass preserves `A`. Outside one, `A` runs on
    /// every call. `ir` may be the pass's input, its output, or any other
    /// node.
    pub fn analysis<A>(&mut self, ir: &A::Ir) -> Arc<A::Output>
    where
        A: Analysis + Default,
        A::Ir: NodeHandle,
    {
        match self.analyses.as_deref_mut() {
            Some(cache) => cache.get::<A>(ir),
            None => Arc::new(A::default().run(ir)),
        }
    }

    /// Return the result cached for `ir` under `id`, computing it with
    /// `compute` and caching it on a miss.
    ///
    /// This is [`analysis`](Self::analysis) for an analysis no Rust type
    /// names, such as one a language binding defines at run time: `id`
    /// names the analysis, usually through
    /// [`AnalysisId::of_identifier`], and `compute` performs it. Results
    /// are cached, preserved and carried to a pass's output as the results
    /// of analysis types are, and outside a
    /// [`PassManager`](super::PassManager) `compute` runs on every call.
    ///
    /// The caller keeps one computation, with one result type, per id. The
    /// id of an analysis type reaches that type's cached result when `V` is
    /// its output. A result cached under `id` with another type is
    /// recomputed and replaced.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use fhy_core::identifier::Identifier;
    /// use fhy_core::pass::{AnalysisId, CompilerPass, ExecutePass, PassContext};
    /// use fhy_core::foreign::BoxError;
    /// use fhy_core::tree::{NodeHandle, NodeIdentity};
    ///
    /// #[derive(Clone)]
    /// struct Value(Arc<i64>);
    ///
    /// impl NodeHandle for Value {
    ///     fn identity(&self) -> NodeIdentity {
    ///         NodeIdentity::of_arc(&self.0)
    ///     }
    /// }
    ///
    /// struct Negate(AnalysisId);
    ///
    /// impl CompilerPass<Value> for Negate {
    ///     fn run(&mut self, ir: &Value, cx: &mut PassContext<'_>) -> Result<Value, BoxError> {
    ///         Ok(Value(cx.analysis_by_id(ir, &self.0, |ir| -*ir.0)))
    ///     }
    ///
    ///     fn did_change(&mut self, input: &Value, output: &Value) -> Result<bool, BoxError> {
    ///         Ok(input.0 != output.0)
    ///     }
    /// }
    ///
    /// let mut pass = Negate(AnalysisId::of_identifier(&Identifier::new("negation")));
    /// assert_eq!(*pass.execute(&Value(Arc::new(3)))?.output().0, -3);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn analysis_by_id<T, V>(
        &mut self,
        ir: &T,
        id: &AnalysisId,
        compute: impl FnOnce(&T) -> V,
    ) -> Arc<V>
    where
        T: NodeHandle,
        V: Send + Sync + 'static,
    {
        match self.analyses.as_deref_mut() {
            Some(cache) => cache.get_or_insert_with(ir, id, || compute(ir)),
            None => Arc::new(compute(ir)),
        }
    }

    /// Call `callback` with an owned handle to the run's analyses, and
    /// return what it returns.
    ///
    /// For code that must hold the analyses by value rather than borrow
    /// this context. Under a [`PassManager`](super::PassManager), the run's
    /// cache moves into the handle for the length of the call and back into
    /// this context when `callback` returns or unwinds; outside one, the
    /// handle computes every request afresh. A clone of the handle kept
    /// after the call is expired. See [`DetachedAnalyses`].
    pub fn with_detached_analyses<R>(
        &mut self,
        callback: impl FnOnce(&DetachedAnalyses) -> R,
    ) -> R {
        let detachment = Detachment::new(self.analyses.as_deref_mut());
        callback(detachment.handle())
    }

    /// Return the diagnostics recorded so far, in emission order.
    #[must_use]
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Return the name of the running pass.
    #[must_use]
    pub fn pass_name(&self) -> &str {
        &self.pass_name
    }
}
