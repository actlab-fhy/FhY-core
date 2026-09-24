//! The errors a pass run returns.

use std::borrow::Cow;
use std::error::Error;
use std::fmt;

use super::compiler_pass::PassFailure;
use super::validation::ValidatorRecord;
use crate::diagnostic::{Diagnostic, ValidationReport};

/// Return the message of `error` followed by the message of each of its
/// sources, in order, joined by `: `.
pub(super) fn render_chain(error: &(dyn Error + 'static)) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        text.push_str(": ");
        text.push_str(&cause.to_string());
        source = cause.source();
    }
    text
}

/// A lifecycle hook of a [`CompilerPass`](super::CompilerPass).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PassHook {
    /// [`CompilerPass::validate_input`](super::CompilerPass::validate_input).
    ValidateInput,
    /// [`CompilerPass::skip`](super::CompilerPass::skip).
    Skip,
    /// [`CompilerPass::run`](super::CompilerPass::run).
    Run,
    /// [`CompilerPass::validate_output`](super::CompilerPass::validate_output).
    ValidateOutput,
    /// [`CompilerPass::did_change`](super::CompilerPass::did_change).
    DidChange,
    /// [`CompilerPass::preserved_analyses`](super::CompilerPass::preserved_analyses).
    PreservedAnalyses,
}

impl PassHook {
    /// Return the hook's method name, for example `validate_input`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            PassHook::ValidateInput => "validate_input",
            PassHook::Skip => "skip",
            PassHook::Run => "run",
            PassHook::ValidateOutput => "validate_output",
            PassHook::DidChange => "did_change",
            PassHook::PreservedAnalyses => "preserved_analyses",
        }
    }
}

/// Render [`PassHook::as_str`].
impl fmt::Display for PassHook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The class of a pass failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PassErrorClass {
    /// A validation hook failed, or verification rejected IR.
    Validation,
    /// An execution hook failed, or a fixpoint group did not converge.
    Execution,
}

/// What failed.
#[derive(Debug)]
enum PassErrorKind {
    /// A hook returned an error.
    Hook {
        class: PassErrorClass,
        pass_name: Cow<'static, str>,
        hook: PassHook,
        source: PassFailure,
    },
    /// Verification rejected the IR a pass received or produced.
    Verification {
        pass_name: Cow<'static, str>,
        report: Box<ValidationReport<ValidatorRecord>>,
    },
    /// A fixpoint group used its iteration budget without converging.
    NonConvergence,
}

/// A pass run that failed.
///
/// A failure is a validation failure (a validation hook failed, or
/// verification rejected IR), or an execution failure (an execution hook
/// failed, or a fixpoint group did not converge). The message names the pass
/// and the hook, and [`Error::source`] is the error the hook returned.
#[derive(Debug)]
pub struct PassError {
    kind: PassErrorKind,
    message: String,
    diagnostics: Vec<Diagnostic>,
}

impl PassError {
    /// Create the error for `hook` of the pass `pass_name` returning
    /// `source`, reported as `message` after the pass emitted `diagnostics`.
    pub(super) fn new_hook_failure(
        class: PassErrorClass,
        pass_name: Cow<'static, str>,
        hook: PassHook,
        message: String,
        source: PassFailure,
        diagnostics: Vec<Diagnostic>,
    ) -> Self {
        Self {
            kind: PassErrorKind::Hook {
                class,
                pass_name,
                hook,
                source,
            },
            message,
            diagnostics,
        }
    }

    /// Create the error for verification rejecting IR around the pass
    /// `pass_name`, reported as `message` with the pass's `diagnostics`.
    pub(super) fn new_verification_failure(
        pass_name: Cow<'static, str>,
        message: String,
        report: ValidationReport<ValidatorRecord>,
        diagnostics: Vec<Diagnostic>,
    ) -> Self {
        Self {
            kind: PassErrorKind::Verification {
                pass_name,
                report: Box::new(report),
            },
            message,
            diagnostics,
        }
    }

    /// Create the error for a fixpoint group that did not converge.
    pub(super) fn new_non_convergence(message: String) -> Self {
        Self {
            kind: PassErrorKind::NonConvergence,
            message,
            diagnostics: Vec::new(),
        }
    }

    /// Return the class of the failure.
    pub(super) fn class(&self) -> PassErrorClass {
        match &self.kind {
            PassErrorKind::Hook { class, .. } => *class,
            PassErrorKind::Verification { .. } => PassErrorClass::Validation,
            PassErrorKind::NonConvergence => PassErrorClass::Execution,
        }
    }

    /// Return the name of the pass that failed, or `None` for a fixpoint
    /// group that did not converge.
    #[must_use]
    pub fn pass_name(&self) -> Option<&str> {
        match &self.kind {
            PassErrorKind::Hook { pass_name, .. }
            | PassErrorKind::Verification { pass_name, .. } => Some(pass_name),
            PassErrorKind::NonConvergence => None,
        }
    }

    /// Return whether a validation hook failed or verification rejected IR.
    #[must_use]
    pub fn is_validation_failure(&self) -> bool {
        self.class() == PassErrorClass::Validation
    }

    /// Return whether an execution hook failed or a fixpoint group did not
    /// converge.
    #[must_use]
    pub fn is_execution_failure(&self) -> bool {
        self.class() == PassErrorClass::Execution
    }

    /// Return whether a fixpoint group used its iteration budget without
    /// converging.
    #[must_use]
    pub fn is_non_convergence(&self) -> bool {
        matches!(self.kind, PassErrorKind::NonConvergence)
    }

    /// Return the hook whose error this is, or `None` when verification or
    /// a fixpoint group failed.
    #[must_use]
    pub fn failed_hook(&self) -> Option<PassHook> {
        match &self.kind {
            PassErrorKind::Hook { hook, .. } => Some(*hook),
            PassErrorKind::Verification { .. } | PassErrorKind::NonConvergence => None,
        }
    }

    /// Return the verification report that rejected the IR, if verification
    /// failed.
    #[must_use]
    pub fn verification_report(&self) -> Option<&ValidationReport<ValidatorRecord>> {
        match &self.kind {
            PassErrorKind::Verification { report, .. } => Some(report),
            PassErrorKind::Hook { .. } | PassErrorKind::NonConvergence => None,
        }
    }

    /// Return the diagnostics the failing pass emitted, in emission order,
    /// ending with the error diagnostic that records the failure.
    ///
    /// A fixpoint group that did not converge carries no diagnostics.
    #[must_use]
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }
}

/// Render the failure message, for example
/// `Pass "fold" failed run with division by zero`.
impl fmt::Display for PassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for PassError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.kind {
            PassErrorKind::Hook { source, .. } => Some(source.as_ref()),
            PassErrorKind::Verification { .. } | PassErrorKind::NonConvergence => None,
        }
    }
}
