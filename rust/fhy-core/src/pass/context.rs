//! The context a pass run hands to every hook: the diagnostics sink and
//! access to analyses.

use std::borrow::Cow;
use std::sync::Arc;

use super::analysis::{Analysis, AnalysisCache};
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
    /// Build it with [`Diagnostic::error`], [`Diagnostic::warning`] or
    /// [`Diagnostic::info`], naming its source, and add a detail with
    /// [`Diagnostic::with_detail`]; [`report_text`](Self::report_text)
    /// records a text attributed to the running pass.
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
