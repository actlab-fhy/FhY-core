//! The errors a pass run returns.

use std::borrow::Cow;
use std::error::Error;
use std::fmt;
use std::num::NonZeroUsize;

use super::compiler_pass::PassFailure;
use super::manager::PipelineRecord;
use super::validation::ValidatorRecord;
use crate::diagnostic::{Diagnostic, ValidationReport};
use crate::identifier::Identifier;

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
///
/// More hooks may be added, so a `match` on one needs a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
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

    /// Return the class of a failure of this hook: validation for the two
    /// validation hooks, execution for the others.
    fn class(self) -> FailureClass {
        match self {
            PassHook::ValidateInput | PassHook::ValidateOutput => FailureClass::Validation,
            PassHook::Skip | PassHook::Run | PassHook::DidChange | PassHook::PreservedAnalyses => {
                FailureClass::Execution
            }
        }
    }
}

/// Render [`PassHook::as_str`].
impl fmt::Display for PassHook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The class of a pass failure: whether IR was found invalid or a pass
/// could not do its work.
///
/// More classes may be added, so a `match` on one needs a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FailureClass {
    /// A validation hook failed, or verification rejected IR.
    Validation,
    /// Another hook failed, or a fixpoint group did not converge.
    Execution,
}

/// Where a pipeline's verifier looked at the IR it rejected.
///
/// More points may be added, so a `match` on one needs a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum VerificationPoint {
    /// The pipeline's input, before its first pass.
    Input,
    /// A pass's changed output.
    Output,
}

/// What failed.
#[derive(Debug)]
enum Failure {
    /// A hook returned an error that is not a [`PassError`].
    Hook {
        pass_name: Cow<'static, str>,
        hook: PassHook,
        source: PassFailure,
    },
    /// A hook returned a [`PassError`].
    Nested {
        pass_name: Cow<'static, str>,
        hook: PassHook,
        inner: PassError,
    },
    /// Verification rejected the IR a pass received or produced.
    Verification {
        pass_name: Cow<'static, str>,
        point: VerificationPoint,
        report: ValidationReport<ValidatorRecord>,
    },
    /// A fixpoint group used its iteration budget without converging.
    NonConvergence {
        group_name: Identifier,
        max_iterations: NonZeroUsize,
    },
}

/// The parts of a [`PassError`], boxed so the error is one pointer wide.
#[derive(Debug)]
struct Inner {
    failure: Failure,
    diagnostics: Vec<Diagnostic>,
    records: Vec<PipelineRecord>,
}

/// A pass run that failed.
///
/// [`kind`](Self::kind) tells what failed: a hook, directly or with a pass
/// error of its own, the pipeline's verifier, or a fixpoint group's
/// convergence. [`class`](Self::class) tells whether IR was found invalid
/// or a pass could not do its work. The message is one line naming what
/// failed, and never repeats its [`source`](Error::source), so printing
/// the chain of sources writes each cause once.
///
/// # Examples
///
/// ```
/// use fhy_core::pass::{
///     CompilerPass, ExecutePass, FailureClass, PassContext, PassErrorKind, PassFailure, PassHook,
/// };
///
/// struct Refuse;
///
/// impl CompilerPass<i64> for Refuse {
///     fn run(&mut self, _ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
///         Err("division by zero".into())
///     }
///
///     fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
///         Ok(input != output)
///     }
/// }
///
/// let error = Refuse.execute(&1).expect_err("the run fails");
///
/// assert_eq!(error.to_string(), "pass \"Refuse\" failed in run");
/// assert_eq!(error.class(), FailureClass::Execution);
/// assert!(matches!(error.kind(), PassErrorKind::Hook { hook: PassHook::Run, .. }));
/// assert_eq!(
///     error.diagnostics()[0].message_text(),
///     "pass \"Refuse\" failed in run: division by zero"
/// );
/// ```
#[derive(Debug)]
pub struct PassError {
    inner: Box<Inner>,
}

impl PassError {
    /// Create the error of `failure` in `inner`, without diagnostics or
    /// records.
    fn from_failure(failure: Failure) -> Self {
        Self {
            inner: Box::new(Inner {
                failure,
                diagnostics: Vec::new(),
                records: Vec::new(),
            }),
        }
    }

    /// Create the error for `hook` of the pass `pass_name` returning
    /// `failure`: nested when `failure` is a [`PassError`].
    pub(super) fn from_hook_failure(
        pass_name: Cow<'static, str>,
        hook: PassHook,
        failure: PassFailure,
    ) -> Self {
        let failure = match failure.downcast::<PassError>() {
            Ok(inner) => Failure::Nested {
                pass_name,
                hook,
                inner: *inner,
            },
            Err(source) => Failure::Hook {
                pass_name,
                hook,
                source,
            },
        };
        Self::from_failure(failure)
    }

    /// Create the error for verification rejecting the IR at `point` with
    /// `report`, blaming the pass `pass_name`.
    pub(super) fn new_verification_failure(
        pass_name: Cow<'static, str>,
        point: VerificationPoint,
        report: ValidationReport<ValidatorRecord>,
    ) -> Self {
        Self::from_failure(Failure::Verification {
            pass_name,
            point,
            report,
        })
    }

    /// Create the error for the fixpoint group `group_name` that did not
    /// converge within `max_iterations`.
    pub(super) fn new_non_convergence(
        group_name: Identifier,
        max_iterations: NonZeroUsize,
    ) -> Self {
        Self::from_failure(Failure::NonConvergence {
            group_name,
            max_iterations,
        })
    }

    /// Return the error with `diagnostics` as its diagnostics.
    pub(super) fn with_diagnostics(mut self, diagnostics: Vec<Diagnostic>) -> Self {
        self.inner.diagnostics = diagnostics;
        self
    }

    /// Return the error with `records` as the records of the pipeline work
    /// completed before it.
    pub(super) fn with_records(mut self, records: Vec<PipelineRecord>) -> Self {
        self.inner.records = records;
        self
    }

    /// Return a borrowed, matchable view of what failed.
    #[must_use]
    pub fn kind(&self) -> PassErrorKind<'_> {
        match &self.inner.failure {
            Failure::Hook {
                pass_name,
                hook,
                source,
            } => PassErrorKind::Hook {
                pass_name,
                hook: *hook,
                source: source.as_ref(),
            },
            Failure::Nested {
                pass_name,
                hook,
                inner,
            } => PassErrorKind::Nested {
                pass_name,
                hook: *hook,
                inner,
            },
            Failure::Verification {
                pass_name,
                point,
                report,
            } => PassErrorKind::Verification {
                pass_name,
                point: *point,
                report,
            },
            Failure::NonConvergence {
                group_name,
                max_iterations,
            } => PassErrorKind::NonConvergence {
                group_name,
                max_iterations: *max_iterations,
            },
        }
    }

    /// Return the class of the failure.
    ///
    /// A hook's failure has the hook's class: validation for
    /// [`validate_input`](super::CompilerPass::validate_input) and
    /// [`validate_output`](super::CompilerPass::validate_output), execution
    /// for the others. A nested failure has the hook's class too, except
    /// under [`run`](super::CompilerPass::run), where it has the inner
    /// error's class. A verification failure is a validation failure, and a
    /// fixpoint group that did not converge an execution failure.
    #[must_use]
    pub fn class(&self) -> FailureClass {
        match &self.inner.failure {
            Failure::Hook { hook, .. } => hook.class(),
            Failure::Nested { hook, inner, .. } => match hook {
                PassHook::Run => inner.class(),
                _ => hook.class(),
            },
            Failure::Verification { .. } => FailureClass::Validation,
            Failure::NonConvergence { .. } => FailureClass::Execution,
        }
    }

    /// Return the name of the pass that failed, or the pass verification
    /// blames, or `None` for a fixpoint group that did not converge.
    #[must_use]
    pub fn pass_name(&self) -> Option<&str> {
        match &self.inner.failure {
            Failure::Hook { pass_name, .. }
            | Failure::Nested { pass_name, .. }
            | Failure::Verification { pass_name, .. } => Some(pass_name),
            Failure::NonConvergence { .. } => None,
        }
    }

    /// Return the diagnostics of the failing pass run, in emission order,
    /// ending with the error diagnostic that records the failure.
    ///
    /// Verification blaming a pipeline's input has only that error, and a
    /// fixpoint group that did not converge has no diagnostics.
    #[must_use]
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.inner.diagnostics
    }

    /// Return the records of the pipeline work completed before the failure,
    /// in pipeline order.
    ///
    /// A failure inside a fixpoint group ends with the group's record, which
    /// holds the iterations begun, the last listing only the pass runs that
    /// completed; the failing pass run has no record. A standalone
    /// [`execute`](super::ExecutePass::execute) has no records.
    #[must_use]
    pub fn records(&self) -> &[PipelineRecord] {
        &self.inner.records
    }
}

/// Render what failed in one line, without the source's message, for
/// example `pass "fold" failed in run` or `fixpoint group "simplify" did not
/// converge (max iterations: 10)`.
impl fmt::Display for PassError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner.failure {
            Failure::Hook {
                pass_name, hook, ..
            }
            | Failure::Nested {
                pass_name, hook, ..
            } => write!(f, "pass {pass_name:?} failed in {hook}"),
            Failure::Verification {
                pass_name,
                point,
                report,
            } => {
                let point = match point {
                    VerificationPoint::Input => "input",
                    VerificationPoint::Output => "output",
                };
                write!(
                    f,
                    "verification rejected the {point} of pass {pass_name:?} (errors: {})",
                    report.errors().count()
                )
            }
            Failure::NonConvergence {
                group_name,
                max_iterations,
            } => write!(
                f,
                "fixpoint group {:?} did not converge (max iterations: {max_iterations})",
                group_name.name_hint()
            ),
        }
    }
}

/// The source is the hook's error for a hook failure and the inner
/// [`PassError`] for a nested failure; the other failures have none.
impl Error for PassError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.inner.failure {
            Failure::Hook { source, .. } => Some(source.as_ref()),
            Failure::Nested { inner, .. } => Some(inner),
            Failure::Verification { .. } | Failure::NonConvergence { .. } => None,
        }
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync + 'static>() {}
    assert_send_sync::<PassError>();
};

/// A borrowed view of what a [`PassError`] reports.
///
/// More kinds, and more fields of each, may be added, so a `match` on one
/// needs a wildcard arm and `..` in each pattern.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum PassErrorKind<'a> {
    /// A hook returned an error that is not a [`PassError`].
    #[non_exhaustive]
    Hook {
        /// The name of the failing pass.
        pass_name: &'a str,
        /// The hook that failed.
        hook: PassHook,
        /// The error the hook returned.
        source: &'a (dyn Error + Send + Sync + 'static),
    },
    /// A hook returned a [`PassError`], for example from running a nested
    /// pass or pipeline.
    #[non_exhaustive]
    Nested {
        /// The name of the failing pass.
        pass_name: &'a str,
        /// The hook that failed.
        hook: PassHook,
        /// The error the hook returned.
        inner: &'a PassError,
    },
    /// The pipeline's verifier rejected IR at `point`, blaming `pass_name`.
    #[non_exhaustive]
    Verification {
        /// The name of the pass blamed: the first pass for the pipeline's
        /// input, the producing pass for an output.
        pass_name: &'a str,
        /// Where the verifier looked.
        point: VerificationPoint,
        /// The verifier's report.
        report: &'a ValidationReport<ValidatorRecord>,
    },
    /// A fixpoint group that fails on non-convergence used its budget.
    #[non_exhaustive]
    NonConvergence {
        /// The group's name.
        group_name: &'a Identifier,
        /// The group's iteration budget.
        max_iterations: NonZeroUsize,
    },
}
