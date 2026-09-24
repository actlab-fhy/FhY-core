//! Collect-all validation pipelines.

use std::fmt;

use super::compiler_pass::{CompilerPass, run_lifecycle};
use super::context::PassContext;
use super::error::PassError;
use super::manager::PassRunRecord;
use super::preserved::PreservedAnalyses;
use crate::diagnostic::{Diagnostic, DiagnosticLevel, Note, ValidationReport};
use crate::identifier::{HasIdentifier, Identifier};

/// Return the error diagnostic for the validator `validator_name` that failed
/// with `error` without reporting an error itself.
fn synthesize_silent_failure(validator_name: &str, error: &PassError) -> Diagnostic {
    let kind = if error.is_validation_failure() {
        "validation failure"
    } else {
        "execution failure"
    };
    let message = format!(
        "Validator \"{validator_name}\" raised \"{kind}\" without reporting a diagnostic: {error}"
    );
    Diagnostic::error(Note::with_other_kind(message), validator_name.to_owned())
}

/// A sequence of validation passes whose diagnostics aggregate into one
/// report.
///
/// Unlike a [`PassManager`](super::PassManager), validation never stops
/// early: every validator runs, even after earlier ones reported errors or
/// failed. Validators run on their own, so each computes the analyses it
/// requests afresh.
pub struct ValidationManager<'p, I> {
    name: Identifier,
    validators: Vec<Box<dyn CompilerPass<I, ()> + 'p>>,
}

impl<'p, I> ValidationManager<'p, I> {
    /// Create the empty validation pipeline `name`.
    #[must_use]
    pub fn new(name: Identifier) -> Self {
        Self {
            name,
            validators: Vec::new(),
        }
    }

    /// Return the pipeline's name.
    #[must_use]
    pub fn name(&self) -> &Identifier {
        &self.name
    }

    /// Append `validator` to the pipeline.
    pub fn add(&mut self, validator: impl CompilerPass<I, ()> + 'p) {
        self.validators.push(Box::new(validator));
    }

    /// Return the validators' names, in pipeline order.
    #[must_use]
    pub fn validator_names(&self) -> Vec<String> {
        self.validators.iter().map(CompilerPass::name).collect()
    }

    /// Run every validator over `ir` and return the aggregated report.
    ///
    /// The report's diagnostics are every validator's diagnostics in
    /// pipeline order, and its records hold one unchanged, all-preserving
    /// record per validator. A validator that fails keeps the diagnostics it
    /// emitted; when none of them is an error, the report gains the error
    /// `Validator "<name>" raised "<kind>" without reporting a diagnostic:
    /// <message>`, where `<kind>` is `validation failure` or
    /// `execution failure`.
    #[must_use]
    pub fn validate(&mut self, ir: &I) -> ValidationReport<PassRunRecord> {
        let mut diagnostics = Vec::new();
        let mut records = Vec::with_capacity(self.validators.len());
        for validator in &mut self.validators {
            let mut cx = PassContext::new(validator.name(), None);
            let result = run_lifecycle(validator.as_mut(), ir, &mut cx);
            let (validator_name, mut captured) = cx.into_parts();
            if let Err(error) = result {
                let reported_error = captured
                    .iter()
                    .any(|diagnostic| diagnostic.level() == DiagnosticLevel::Error);
                if !reported_error {
                    captured.push(synthesize_silent_failure(&validator_name, &error));
                }
            }
            diagnostics.extend(captured.iter().cloned());
            records.push(PassRunRecord::new(
                validator_name,
                false,
                captured,
                PreservedAnalyses::all(),
            ));
        }
        ValidationReport::new(diagnostics, records)
    }
}

impl<I> Default for ValidationManager<'_, I> {
    /// Create the empty validation pipeline `validation-pipeline`.
    fn default() -> Self {
        Self::new(Identifier::new("validation-pipeline"))
    }
}

impl<I> HasIdentifier for ValidationManager<'_, I> {
    fn identifier(&self) -> &Identifier {
        &self.name
    }
}

/// Render the pipeline's name and its validators' names.
impl<I> fmt::Debug for ValidationManager<'_, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ValidationManager")
            .field("name", &self.name)
            .field("validators", &self.validator_names())
            .finish()
    }
}
