//! The context a pass run hands to every hook: the diagnostics sink and
//! access to analyses.

use std::sync::Arc;

use super::analysis::{Analysis, AnalysisCache, NodeHandle};
use crate::diagnostic::{Diagnostic, DiagnosticLevel, Note};

/// The diagnostics sink and analysis access of one pass run.
///
/// A pass receives the context in each lifecycle hook. Diagnostics it
/// reports carry the pass's name as their source and end up in the run's
/// [`PassOutcome`](super::PassOutcome), or in the
/// [`PassError`](super::PassError) if the run fails. Under a
/// [`PassManager`](super::PassManager), analysis results are cached for the
/// whole pipeline run; a pass executed on its own computes them afresh on
/// every request.
#[derive(Debug)]
pub struct PassContext<'a> {
    pass_name: String,
    diagnostics: Vec<Diagnostic>,
    analyses: Option<&'a mut AnalysisCache>,
}

impl<'a> PassContext<'a> {
    /// Create the context for a run of the pass `pass_name`, caching analyses
    /// in `analyses` when one is given.
    pub(super) fn new(pass_name: String, analyses: Option<&'a mut AnalysisCache>) -> Self {
        Self {
            pass_name,
            diagnostics: Vec::new(),
            analyses,
        }
    }

    /// Create the context for a run of the pass `pass_name` outside a
    /// pipeline, which computes every analysis afresh.
    pub(crate) fn new_standalone(pass_name: String) -> Self {
        Self::new(pass_name, None)
    }

    /// Return the pass name and the diagnostics, consuming the context.
    pub(super) fn into_parts(self) -> (String, Vec<Diagnostic>) {
        (self.pass_name, self.diagnostics)
    }

    /// Record the diagnostic `message` at `level`, with optional `detail`.
    pub fn report(&mut self, level: DiagnosticLevel, message: Note, detail: Option<String>) {
        let diagnostic = Diagnostic::new(level, message, self.pass_name.clone(), detail);
        self.diagnostics.push(diagnostic);
    }

    /// Record the diagnostic with text `message` at `level`, with optional
    /// `detail`, as a note of the uncategorized kind.
    pub fn report_text(
        &mut self,
        level: DiagnosticLevel,
        message: impl Into<String>,
        detail: Option<String>,
    ) {
        self.report(level, Note::with_other_kind(message), detail);
    }

    /// Return the result of the analysis `A` for `ir`.
    ///
    /// Under a [`PassManager`](super::PassManager) the result is cached per
    /// node for the pipeline run and survives passes that preserve `A`.
    /// Outside one, `A` runs on every call. `ir` may be the pass's input or
    /// any other node.
    pub fn analysis<A, T>(&mut self, ir: &T) -> Arc<A::Output>
    where
        A: Analysis<T> + Default,
        T: NodeHandle,
    {
        match self.analyses.as_deref_mut() {
            Some(cache) => cache.get::<A, T>(ir),
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
