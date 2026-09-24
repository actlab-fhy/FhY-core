//! Tests for standalone pass runs in `fhy_core::pass`: the guarded lifecycle,
//! hook-error wrapping and pass-through, the pass context, preserved analyses,
//! analysis ids, node identities, pass names, and the pass registry.
//!
//! Public API only. Every registry is a local value, so no test shares
//! state with another.

use crate::support::pass_ir;

use std::any::{TypeId, type_name};
use std::borrow::Cow;
use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fhy_core::diagnostic::{Diagnostic, DiagnosticLevel, Note, NoteKind};
use fhy_core::pass::{
    Analysis, AnalysisId, CompilerPass, CreatePassError, ExecutePass, FailureClass, PassContext,
    PassError, PassErrorKind, PassFailure, PassHook, PassInfo, PassRegistrationError, PassRegistry,
    PreservedAnalyses,
};
use fhy_core::tree::{NodeHandle, NodeIdentity};
use pass_ir::{BoxIr, ClosurePass, DoubleAnalysis, ParityAnalysis};
use rstest::rstest;

// =============================================================================
// Helpers
// =============================================================================

/// The error the test passes' hooks return.
#[derive(Debug)]
struct HookFailure(String);

impl fmt::Display for HookFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for HookFailure {}

/// Build the hook error reported as `message`.
fn fail(message: &str) -> PassFailure {
    Box::new(HookFailure(message.to_owned()))
}

/// Adds one to an integer.
struct Increment;

impl CompilerPass<i64> for Increment {
    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        Ok(ir + 1)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// Returns an integer unchanged.
struct KeepInteger;

impl CompilerPass<i64> for KeepInteger {
    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        Ok(*ir)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// Records which hooks ran, optionally skipping the run or failing in one
/// hook.
#[derive(Default)]
struct RecordingPass {
    calls: Vec<String>,
    skips: bool,
    fails_in: Option<PassHook>,
}

impl RecordingPass {
    /// Record a call of `hook`, failing it if it is the failing hook.
    fn record(&mut self, hook: PassHook) -> Result<(), PassFailure> {
        self.calls.push(hook.as_str().to_owned());
        if self.fails_in == Some(hook) {
            return Err(fail("recorded failure"));
        }
        Ok(())
    }
}

impl CompilerPass<i64> for RecordingPass {
    fn validate_input(&mut self, _ir: &i64, _cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        self.record(PassHook::ValidateInput)
    }

    fn skip(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<Option<i64>, PassFailure> {
        self.record(PassHook::Skip)?;
        Ok(self.skips.then_some(*ir))
    }

    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        self.record(PassHook::Run)?;
        Ok(ir + 1)
    }

    fn validate_output(
        &mut self,
        _input: &i64,
        _output: &i64,
        _cx: &mut PassContext<'_>,
    ) -> Result<(), PassFailure> {
        self.record(PassHook::ValidateOutput)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        self.record(PassHook::DidChange)?;
        Ok(input != output)
    }

    fn preserved_analyses(
        &mut self,
        _input: &i64,
        _output: &i64,
        changed: bool,
    ) -> Result<PreservedAnalyses, PassFailure> {
        self.calls
            .push(format!("preserved_analyses(changed={changed})"));
        Ok(PreservedAnalyses::none())
    }
}

/// Fails in one hook with the error `<hook>-broken`.
struct FailingHookPass {
    hook: PassHook,
}

impl FailingHookPass {
    /// Fail if `hook` is the failing hook.
    fn check(&self, hook: PassHook) -> Result<(), PassFailure> {
        if self.hook == hook {
            return Err(fail(&format!("{hook}-broken")));
        }
        Ok(())
    }
}

impl CompilerPass<i64> for FailingHookPass {
    fn validate_input(&mut self, _ir: &i64, _cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        self.check(PassHook::ValidateInput)
    }

    fn skip(&mut self, _ir: &i64, _cx: &mut PassContext<'_>) -> Result<Option<i64>, PassFailure> {
        self.check(PassHook::Skip)?;
        Ok(None)
    }

    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        self.check(PassHook::Run)?;
        Ok(ir + 1)
    }

    fn validate_output(
        &mut self,
        _input: &i64,
        _output: &i64,
        _cx: &mut PassContext<'_>,
    ) -> Result<(), PassFailure> {
        self.check(PassHook::ValidateOutput)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        self.check(PassHook::DidChange)?;
        Ok(input != output)
    }

    fn preserved_analyses(
        &mut self,
        _input: &i64,
        _output: &i64,
        _changed: bool,
    ) -> Result<PreservedAnalyses, PassFailure> {
        self.check(PassHook::PreservedAnalyses)?;
        Ok(PreservedAnalyses::none())
    }
}

/// Rejects every input with the error `rejected`.
struct RejectInput;

impl CompilerPass<i64> for RejectInput {
    fn validate_input(&mut self, _ir: &i64, _cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        Err(fail("rejected"))
    }

    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        Ok(*ir)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// Fails every run with the error `crashed`.
struct CrashInRun;

impl CompilerPass<i64> for CrashInRun {
    fn run(&mut self, _ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        Err(fail("crashed"))
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// Which pass failure [`HandOverPass`] hands over.
#[derive(Debug, Clone, Copy)]
enum InnerFailure {
    /// `RejectInput`'s validation failure.
    Validation,
    /// `CrashInRun`'s execution failure.
    Execution,
}

/// The class of the failure `inner` produces.
fn classify_inner_failure(inner: InnerFailure) -> FailureClass {
    match inner {
        InnerFailure::Validation => FailureClass::Validation,
        InnerFailure::Execution => FailureClass::Execution,
    }
}

/// Produce the failure `inner` by running the pass that fails that way.
fn produce_inner_failure(inner: InnerFailure) -> PassError {
    let result = match inner {
        InnerFailure::Validation => RejectInput.execute(&0),
        InnerFailure::Execution => CrashInRun.execute(&0),
    };
    result.expect_err("the inner pass fails")
}

/// Returns a previously produced [`PassError`] from one hook, as a pass that
/// runs another pass and propagates its failure does, after warning
/// `handing over` from its run.
struct HandOverPass {
    name: &'static str,
    hook: PassHook,
    failure: Option<PassError>,
}

impl HandOverPass {
    /// Build the pass that hands over the failure `inner` from `hook`.
    fn new(hook: PassHook, inner: InnerFailure) -> Self {
        Self::handing("HandOverPass", hook, produce_inner_failure(inner))
    }

    /// Build the pass `name` that hands over `failure` from `hook`.
    fn handing(name: &'static str, hook: PassHook, failure: PassError) -> Self {
        Self {
            name,
            hook,
            failure: Some(failure),
        }
    }

    /// Hand over the failure if `hook` is the handing hook.
    fn check(&mut self, hook: PassHook) -> Result<(), PassFailure> {
        if self.hook == hook {
            if let Some(failure) = self.failure.take() {
                return Err(Box::new(failure));
            }
        }
        Ok(())
    }
}

impl CompilerPass<i64> for HandOverPass {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed(self.name)
    }

    fn validate_input(&mut self, _ir: &i64, _cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        self.check(PassHook::ValidateInput)
    }

    fn skip(&mut self, _ir: &i64, _cx: &mut PassContext<'_>) -> Result<Option<i64>, PassFailure> {
        self.check(PassHook::Skip)?;
        Ok(None)
    }

    fn run(&mut self, ir: &i64, cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        cx.report_text(DiagnosticLevel::Warning, "handing over", None);
        self.check(PassHook::Run)?;
        Ok(ir + 1)
    }

    fn validate_output(
        &mut self,
        _input: &i64,
        _output: &i64,
        _cx: &mut PassContext<'_>,
    ) -> Result<(), PassFailure> {
        self.check(PassHook::ValidateOutput)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        self.check(PassHook::DidChange)?;
        Ok(input != output)
    }

    fn preserved_analyses(
        &mut self,
        _input: &i64,
        _output: &i64,
        _changed: bool,
    ) -> Result<PreservedAnalyses, PassFailure> {
        self.check(PassHook::PreservedAnalyses)?;
        Ok(PreservedAnalyses::none())
    }
}

/// Reports one diagnostic at a configured level from its run.
struct LevelReportPass {
    level: DiagnosticLevel,
}

impl CompilerPass<i64> for LevelReportPass {
    fn run(&mut self, ir: &i64, cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        cx.report_text(self.level, "level-test-message", None);
        Ok(*ir)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

// =============================================================================
// Standalone execution
// =============================================================================

/// Test a run returns its output, reports the change, emits nothing, and
/// preserves no analysis.
#[test]
fn execute_returns_the_output_of_a_changing_run() {
    let outcome = Increment.execute(&1).expect("the run succeeds");

    assert_eq!(*outcome.output(), 2);
    assert!(outcome.is_changed());
    assert!(!outcome.is_skipped());
    assert!(outcome.diagnostics().is_empty());
    assert_eq!(outcome.preserved_analyses(), &PreservedAnalyses::none());
}

/// Test a run that leaves the IR unchanged preserves every analysis by
/// default.
#[test]
fn execute_of_an_unchanged_run_preserves_every_analysis() {
    let outcome = KeepInteger.execute(&7).expect("the run succeeds");

    assert_eq!(*outcome.output(), 7);
    assert!(!outcome.is_changed());
    assert_eq!(outcome.preserved_analyses(), &PreservedAnalyses::all());
}

/// Test `into_output` hands back the output.
#[test]
fn pass_outcome_into_output_returns_the_output() {
    let outcome = Increment.execute(&41).expect("the run succeeds");

    assert_eq!(outcome.into_output(), 42);
}

/// Test a run calls the hooks in lifecycle order.
#[test]
fn execute_calls_the_hooks_in_lifecycle_order() {
    let mut pass = RecordingPass::default();

    pass.execute(&0).expect("the run succeeds");

    assert_eq!(
        pass.calls,
        [
            "validate_input",
            "skip",
            "run",
            "validate_output",
            "did_change",
            "preserved_analyses(changed=true)",
        ]
    );
}

/// Test a skipped run calls no hook after `skip` but `preserved_analyses`,
/// asking for the analyses of an unchanged run.
#[test]
fn execute_skipped_run_calls_skip_instead_of_run() {
    let mut pass = RecordingPass {
        skips: true,
        ..RecordingPass::default()
    };

    pass.execute(&0).expect("the run succeeds");

    assert_eq!(
        pass.calls,
        [
            "validate_input",
            "skip",
            "preserved_analyses(changed=false)",
        ]
    );
}

/// Test a skipped run outputs the output `skip` supplied, unchanged and
/// skipped, preserves every analysis, and keeps the diagnostics `skip`
/// reported.
#[test]
fn execute_skipped_run_outputs_the_skip_output() {
    struct SkippedPass;

    impl CompilerPass<i64> for SkippedPass {
        fn skip(&mut self, ir: &i64, cx: &mut PassContext<'_>) -> Result<Option<i64>, PassFailure> {
            cx.report_text(DiagnosticLevel::Info, "skip requested", None);
            Ok(Some(ir + 100))
        }

        fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
            Ok(ir + 1)
        }

        fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
            Ok(input != output)
        }
    }

    let outcome = SkippedPass.execute(&2).expect("the skipped run succeeds");

    assert_eq!(*outcome.output(), 102);
    assert!(!outcome.is_changed());
    assert!(outcome.is_skipped());
    assert_eq!(outcome.preserved_analyses(), &PreservedAnalyses::all());
    assert_eq!(outcome.diagnostics().len(), 1);
    assert_eq!(outcome.diagnostics()[0].level(), DiagnosticLevel::Info);
    assert_eq!(outcome.diagnostics()[0].message_text(), "skip requested");
}

/// Test every run starts with no diagnostics, even on a reused pass.
#[test]
fn execute_starts_every_run_without_diagnostics() {
    let mut pass = LevelReportPass {
        level: DiagnosticLevel::Warning,
    };

    pass.execute(&0).expect("the first run succeeds");
    let outcome = pass.execute(&0).expect("the second run succeeds");

    assert_eq!(outcome.diagnostics().len(), 1);
}

// =============================================================================
// Hook failures
// =============================================================================

/// Test `PassHook::as_str` and `Display` render the hook's method name.
#[rstest]
#[case::validate_input(PassHook::ValidateInput, "validate_input")]
#[case::skip(PassHook::Skip, "skip")]
#[case::run(PassHook::Run, "run")]
#[case::validate_output(PassHook::ValidateOutput, "validate_output")]
#[case::did_change(PassHook::DidChange, "did_change")]
#[case::preserved_analyses(PassHook::PreservedAnalyses, "preserved_analyses")]
fn pass_hook_renders_the_method_name(#[case] hook: PassHook, #[case] expected: &str) {
    assert_eq!(hook.as_str(), expected);
    assert_eq!(hook.to_string(), expected);
}

/// Test a hook error becomes a pass error of the hook's class that names the
/// pass and the hook, keeps the hook's error as its source, has no pipeline
/// records, and ends the diagnostics with an error recording the failure
/// and its cause.
#[rstest]
#[case::validate_input(PassHook::ValidateInput, FailureClass::Validation)]
#[case::skip(PassHook::Skip, FailureClass::Execution)]
#[case::run(PassHook::Run, FailureClass::Execution)]
#[case::validate_output(PassHook::ValidateOutput, FailureClass::Validation)]
#[case::did_change(PassHook::DidChange, FailureClass::Execution)]
#[case::preserved_analyses(PassHook::PreservedAnalyses, FailureClass::Execution)]
fn execute_wraps_a_hook_error_naming_the_pass_and_hook(
    #[case] hook: PassHook,
    #[case] class: FailureClass,
) {
    let error = FailingHookPass { hook }
        .execute(&1)
        .expect_err("the hook fails");

    let PassErrorKind::Hook {
        pass_name,
        hook: failed_hook,
        source,
        ..
    } = error.kind()
    else {
        panic!("expected a hook failure, got {error:?}");
    };
    assert_eq!(pass_name, "FailingHookPass");
    assert_eq!(failed_hook, hook);
    assert_eq!(
        source
            .downcast_ref::<HookFailure>()
            .map(ToString::to_string),
        Some(format!("{hook}-broken"))
    );
    assert_eq!(error.class(), class);
    assert_eq!(error.pass_name(), Some("FailingHookPass"));
    assert!(error.records().is_empty());
    let chained = error.source().expect("the hook's error is the source");
    assert!(
        chained.downcast_ref::<HookFailure>().is_some(),
        "{chained:?}"
    );
    let last = error.diagnostics().last().expect("an error diagnostic");
    assert_eq!(last.level(), DiagnosticLevel::Error);
    assert_eq!(
        last.message_text(),
        format!("pass \"FailingHookPass\" failed in {hook}: {hook}-broken")
    );
    assert_eq!(last.source(), "FailingHookPass");
    assert_eq!(last.detail(), None);
}

/// Test a failing hook ends the run: no later hook is called.
#[rstest]
#[case::validate_input(PassHook::ValidateInput, &["validate_input"])]
#[case::skip(PassHook::Skip, &["validate_input", "skip"])]
#[case::run(PassHook::Run, &["validate_input", "skip", "run"])]
#[case::validate_output(
    PassHook::ValidateOutput,
    &["validate_input", "skip", "run", "validate_output"]
)]
fn execute_stops_at_the_first_failing_hook(#[case] hook: PassHook, #[case] expected: &[&str]) {
    let mut pass = RecordingPass {
        fails_in: Some(hook),
        ..RecordingPass::default()
    };

    pass.execute(&0).expect_err("the hook fails");

    assert_eq!(pass.calls, expected);
}

/// Test a failed run keeps the diagnostics the pass emitted before failing.
#[test]
fn execute_failure_keeps_the_diagnostics_emitted_before_it() {
    struct WarnThenCrash;

    impl CompilerPass<i64> for WarnThenCrash {
        fn run(&mut self, _ir: &i64, cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
            cx.report_text(DiagnosticLevel::Warning, "heads-up", None);
            Err(fail("boom"))
        }

        fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
            Ok(input != output)
        }
    }

    let error = WarnThenCrash.execute(&0).expect_err("the run fails");

    let messages: Vec<_> = error
        .diagnostics()
        .iter()
        .map(|diagnostic| (diagnostic.level(), diagnostic.message_text()))
        .collect();
    assert_eq!(
        messages,
        [
            (DiagnosticLevel::Warning, "heads-up"),
            (
                DiagnosticLevel::Error,
                "pass \"WarnThenCrash\" failed in run: boom"
            ),
        ]
    );
}

/// Test a pass error a hook returns is nested, not handed through: the
/// outer error names the outer pass and hook, keeps the outer pass's
/// diagnostics ending with the error recording the failure, and has the
/// inner error as its source.
#[test]
fn execute_nests_a_pass_error_keeping_the_outer_diagnostics() {
    let error = HandOverPass::new(PassHook::Run, InnerFailure::Execution)
        .execute(&0)
        .expect_err("the run hands over a failure");

    let PassErrorKind::Nested {
        pass_name,
        hook,
        inner,
        ..
    } = error.kind()
    else {
        panic!("expected a nested failure, got {error:?}");
    };
    assert_eq!((pass_name, hook), ("HandOverPass", PassHook::Run));
    assert_eq!(inner.pass_name(), Some("CrashInRun"));
    assert_eq!(error.pass_name(), Some("HandOverPass"));
    let diagnostics: Vec<_> = error
        .diagnostics()
        .iter()
        .map(|diagnostic| {
            (
                diagnostic.level(),
                diagnostic.source(),
                diagnostic.message_text(),
            )
        })
        .collect();
    assert_eq!(
        diagnostics,
        [
            (DiagnosticLevel::Warning, "HandOverPass", "handing over"),
            (
                DiagnosticLevel::Error,
                "HandOverPass",
                "pass \"HandOverPass\" failed in run: pass \"CrashInRun\" failed in run: crashed"
            ),
        ]
    );
    let source = error.source().expect("the inner error is the source");
    assert!(
        std::ptr::eq(
            source.downcast_ref::<PassError>().expect("a pass error"),
            inner
        ),
        "{source:?}"
    );
}

/// Test a nested error's class is its hook's class, except under `run`,
/// where it is the inner error's class.
#[rstest]
#[case::validate_input_of_execution(PassHook::ValidateInput, InnerFailure::Execution)]
#[case::validate_input_of_validation(PassHook::ValidateInput, InnerFailure::Validation)]
#[case::skip_of_validation(PassHook::Skip, InnerFailure::Validation)]
#[case::run_of_validation(PassHook::Run, InnerFailure::Validation)]
#[case::run_of_execution(PassHook::Run, InnerFailure::Execution)]
#[case::validate_output_of_execution(PassHook::ValidateOutput, InnerFailure::Execution)]
#[case::did_change_of_validation(PassHook::DidChange, InnerFailure::Validation)]
#[case::preserved_analyses_of_validation(PassHook::PreservedAnalyses, InnerFailure::Validation)]
fn nested_pass_error_class_is_the_hooks_except_for_run(
    #[case] hook: PassHook,
    #[case] inner: InnerFailure,
) {
    let expected = match hook {
        PassHook::Run => classify_inner_failure(inner),
        PassHook::ValidateInput | PassHook::ValidateOutput => FailureClass::Validation,
        _ => FailureClass::Execution,
    };

    let error = HandOverPass::new(hook, inner)
        .execute(&0)
        .expect_err("the hook hands over a failure");

    assert!(
        matches!(error.kind(), PassErrorKind::Nested { hook: nested_hook, .. } if nested_hook == hook),
        "{error:?}"
    );
    assert_eq!(error.class(), expected);
}

/// Test walking the source chain of a three-deep nested error writes each
/// cause exactly once: no error repeats its source's text.
#[test]
fn pass_error_chain_prints_each_cause_once() {
    let innermost = produce_inner_failure(InnerFailure::Execution);
    let middle = HandOverPass::handing("middle", PassHook::Run, innermost)
        .execute(&0)
        .expect_err("the middle pass hands over the failure");
    let outer = HandOverPass::handing("outer", PassHook::Run, middle)
        .execute(&0)
        .expect_err("the outer pass hands over the failure");

    let mut messages = vec![outer.to_string()];
    let mut source = outer.source();
    while let Some(cause) = source {
        messages.push(cause.to_string());
        source = cause.source();
    }

    assert_eq!(
        messages,
        [
            "pass \"outer\" failed in run",
            "pass \"middle\" failed in run",
            "pass \"CrashInRun\" failed in run",
            "crashed",
        ]
    );
}

/// Test a pass error is one pointer wide, so a `Result` carrying one stays
/// small.
#[test]
fn pass_error_is_one_pointer_wide() {
    assert_eq!(size_of::<PassError>(), size_of::<usize>());
}

// =============================================================================
// The pass context
// =============================================================================

/// Test `report_text` records a diagnostic at the given level, attributed to
/// the pass, with the uncategorized note kind and no detail.
#[rstest]
#[case::error(DiagnosticLevel::Error)]
#[case::warning(DiagnosticLevel::Warning)]
#[case::info(DiagnosticLevel::Info)]
fn report_text_records_a_diagnostic_at_the_given_level(#[case] level: DiagnosticLevel) {
    let outcome = LevelReportPass { level }
        .execute(&0)
        .expect("the run succeeds");

    assert_eq!(outcome.diagnostics().len(), 1);
    let diagnostic = &outcome.diagnostics()[0];
    assert_eq!(diagnostic.level(), level);
    assert_eq!(
        diagnostic.message(),
        &Note::with_other_kind("level-test-message")
    );
    assert_eq!(diagnostic.source(), "LevelReportPass");
    assert_eq!(diagnostic.detail(), None);
}

/// Test a detail is stored beside the message, not inside it.
#[test]
fn report_text_keeps_the_detail_separate_from_the_message() {
    let mut pass = ClosurePass::new("detail", |ir, cx| {
        cx.report_text(
            DiagnosticLevel::Warning,
            "primary-message",
            Some("extra-context".to_owned()),
        );
        Ok(ir.clone())
    });

    let outcome = pass.execute(&BoxIr::new(0)).expect("the run succeeds");

    let diagnostic = &outcome.diagnostics()[0];
    assert_eq!(diagnostic.message_text(), "primary-message");
    assert_eq!(diagnostic.detail(), Some("extra-context"));
}

/// Test `report` records a diagnostic exactly as given: its structured
/// note, its source, and its detail.
#[test]
fn report_keeps_a_diagnostic_as_given() {
    let diagnostic = Diagnostic::warning(
        Note::new("structured-message", NoteKind::rationale().clone()),
        "note.check",
    )
    .with_detail("extra-context");
    let reported = diagnostic.clone();
    let mut pass = ClosurePass::new("note", move |ir, cx| {
        cx.report(reported.clone());
        Ok(ir.clone())
    });

    let outcome = pass.execute(&BoxIr::new(0)).expect("the run succeeds");

    assert_eq!(outcome.diagnostics(), [diagnostic]);
}

/// Test the context names the running pass and lists the diagnostics
/// reported so far.
#[test]
fn pass_context_exposes_the_pass_name_and_diagnostics_so_far() {
    let mut observed = Vec::new();
    let mut pass = ClosurePass::new("context_view", |ir, cx| {
        observed.push((cx.pass_name().to_owned(), cx.diagnostics().len()));
        cx.report_text(DiagnosticLevel::Info, "first", None);
        observed.push((cx.pass_name().to_owned(), cx.diagnostics().len()));
        Ok(ir.clone())
    });

    pass.execute(&BoxIr::new(0)).expect("the run succeeds");
    drop(pass);

    assert_eq!(
        observed,
        [
            ("context_view".to_owned(), 0),
            ("context_view".to_owned(), 1),
        ]
    );
}

/// Test a pass run on its own computes an analysis afresh on every request.
#[test]
fn pass_context_analysis_recomputes_on_every_call_outside_a_manager() {
    let mut observed = Vec::new();
    let mut pass = ClosurePass::new("standalone_analysis", |ir, cx| {
        observed.push(*cx.analysis::<DoubleAnalysis>(ir));
        observed.push(*cx.analysis::<DoubleAnalysis>(ir));
        Ok(ir.clone())
    });
    let ir = BoxIr::new(5);

    pass.execute(&ir).expect("the run succeeds");
    drop(pass);

    assert_eq!(observed, [10, 10]);
    assert_eq!(ir.double_runs(), 2);
}

// =============================================================================
// Preserved analyses and analysis ids
// =============================================================================

/// An analysis of integers named to sort first.
struct Alpha;

impl Analysis for Alpha {
    type Ir = i64;
    type Output = i64;

    fn run(&self, ir: &i64) -> i64 {
        *ir
    }
}

/// An analysis of integers named to sort second.
struct Beta;

impl Analysis for Beta {
    type Ir = i64;
    type Output = i64;

    fn run(&self, ir: &i64) -> i64 {
        -ir
    }
}

/// Test an analysis runs over its own IR type.
#[test]
fn analysis_runs_over_its_ir_type() {
    assert_eq!((Alpha.run(&3), Beta.run(&3)), (3, -3));
}

/// Test the all-preserving set preserves every analysis and lists no ids.
#[test]
fn preserved_analyses_all_preserves_every_analysis() {
    let all = PreservedAnalyses::all();

    assert!(all.preserves_all());
    assert!(all.is_preserved::<DoubleAnalysis>());
    assert!(all.is_preserved::<ParityAnalysis>());
    assert!(all.is_id_preserved(AnalysisId::of::<Alpha>()));
    assert_eq!(all.preserved_ids().count(), 0);
}

/// Test the empty set preserves nothing.
#[test]
fn preserved_analyses_none_preserves_nothing() {
    let none = PreservedAnalyses::none();

    assert!(!none.preserves_all());
    assert!(!none.is_preserved::<DoubleAnalysis>());
    assert!(!none.is_id_preserved(AnalysisId::of::<ParityAnalysis>()));
    assert_eq!(none.preserved_ids().count(), 0);
    assert_ne!(none, PreservedAnalyses::all());
}

/// Test `preserve` adds one analysis and leaves the set it was called on
/// unchanged.
#[test]
fn preserve_adds_one_analysis_and_leaves_the_original_unchanged() {
    let original = PreservedAnalyses::none();

    let updated = original.clone().preserve::<DoubleAnalysis>();

    assert!(!original.is_preserved::<DoubleAnalysis>());
    assert!(updated.is_preserved::<DoubleAnalysis>());
    assert!(!updated.is_preserved::<ParityAnalysis>());
    assert!(!updated.preserves_all());
}

/// Test preserving on the all-preserving set keeps it all-preserving.
#[test]
fn preserve_on_all_keeps_all() {
    let preserved = PreservedAnalyses::all()
        .preserve::<DoubleAnalysis>()
        .preserve_id(AnalysisId::of::<ParityAnalysis>());

    assert_eq!(preserved, PreservedAnalyses::all());
}

/// Test preserving an analysis twice equals preserving it once.
#[test]
fn preserve_twice_equals_preserve_once() {
    let once = PreservedAnalyses::none().preserve::<DoubleAnalysis>();

    let twice = once.clone().preserve::<DoubleAnalysis>();

    assert_eq!(twice, once);
}

/// Test preserving by id equals preserving by type.
#[test]
fn preserve_id_equals_preserve_by_type() {
    let by_id = PreservedAnalyses::none().preserve_id(AnalysisId::of::<DoubleAnalysis>());

    assert_eq!(
        by_id,
        PreservedAnalyses::none().preserve::<DoubleAnalysis>()
    );
    assert!(by_id.is_id_preserved(AnalysisId::of::<DoubleAnalysis>()));
}

/// Test the listed ids come out ordered by type name, whatever the order
/// they were added in.
#[test]
fn preserved_ids_lists_the_ids_by_type_name() {
    let preserved = PreservedAnalyses::none()
        .preserve::<ParityAnalysis>()
        .preserve::<DoubleAnalysis>();

    let ids: Vec<_> = preserved.preserved_ids().collect();

    assert_eq!(
        ids,
        [
            AnalysisId::of::<DoubleAnalysis>(),
            AnalysisId::of::<ParityAnalysis>()
        ]
    );
}

/// Test analysis ids are equal exactly for the same type.
#[test]
fn analysis_id_distinguishes_analysis_types() {
    let ids: HashSet<_> = [
        AnalysisId::of::<DoubleAnalysis>(),
        AnalysisId::of::<ParityAnalysis>(),
        AnalysisId::of::<DoubleAnalysis>(),
    ]
    .into_iter()
    .collect();

    assert_eq!(
        AnalysisId::of::<DoubleAnalysis>(),
        AnalysisId::of::<DoubleAnalysis>()
    );
    assert_ne!(
        AnalysisId::of::<DoubleAnalysis>(),
        AnalysisId::of::<ParityAnalysis>()
    );
    assert_eq!(ids.len(), 2);
}

/// Test an analysis id renders the analysis type's full name.
#[test]
fn analysis_id_display_is_the_type_name() {
    assert_eq!(
        AnalysisId::of::<DoubleAnalysis>().to_string(),
        type_name::<DoubleAnalysis>()
    );
}

/// Test analysis ids order by type name.
#[test]
fn analysis_id_orders_by_type_name() {
    assert!(AnalysisId::of::<Alpha>() < AnalysisId::of::<Beta>());
    assert!(AnalysisId::of::<DoubleAnalysis>() < AnalysisId::of::<ParityAnalysis>());
}

// =============================================================================
// Node identities
// =============================================================================

/// Test clones of one `Arc` share an identity.
#[test]
fn node_identity_of_arc_is_shared_by_clones() {
    let node = Arc::new(3_i64);
    let alias = Arc::clone(&node);

    assert_eq!(NodeIdentity::of_arc(&node), NodeIdentity::of_arc(&alias));
}

/// Test two live allocations have different identities, even with equal
/// contents.
#[test]
fn node_identity_of_arc_differs_between_live_allocations() {
    let first = Arc::new(3_i64);
    let second = Arc::new(3_i64);

    assert_ne!(NodeIdentity::of_arc(&first), NodeIdentity::of_arc(&second));
}

/// Test a node handle's clone reports the same identity and a derived node
/// a different one.
#[test]
fn node_handle_identity_follows_the_node() {
    let ir = BoxIr::new(1);

    assert_eq!(ir.identity(), ir.clone().identity());
    assert_ne!(ir.identity(), ir.derive(1).identity());
}

// =============================================================================
// Pass names and descriptions
// =============================================================================

/// A generic pass whose default name must drop the generic arguments.
struct GenericNamedPass<T>(PhantomData<T>);

impl<T> CompilerPass<i64> for GenericNamedPass<T> {
    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        Ok(*ir)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// A pass over integers named and described explicitly.
#[derive(Debug, Clone, Copy)]
struct NamedPass {
    name: &'static str,
    description: &'static str,
}

impl NamedPass {
    /// Build the pass `name` described as `description`.
    fn new(name: &'static str, description: &'static str) -> Self {
        Self { name, description }
    }
}

impl CompilerPass<i64> for NamedPass {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed(self.name)
    }

    fn description(&self) -> Cow<'static, str> {
        Cow::Borrowed(self.description)
    }

    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        Ok(*ir)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// Test a pass that does not override its name is named after its type
/// without module path or generic arguments, and described by its name.
#[test]
fn name_defaults_to_the_type_name_without_path_or_generics() {
    let generic = GenericNamedPass::<Vec<String>>(PhantomData);

    assert_eq!(generic.name(), "GenericNamedPass");
    assert_eq!(generic.description(), "GenericNamedPass");
    assert_eq!(Increment.name(), "Increment");
    assert!(matches!(Increment.name(), Cow::Borrowed(_)));
}

/// Test the default name borrows from the type name, for a generic pass
/// too, so naming a pass allocates nothing.
#[test]
fn default_pass_name_is_borrowed() {
    let generic = GenericNamedPass::<Vec<String>>(PhantomData);

    assert!(matches!(generic.name(), Cow::Borrowed("GenericNamedPass")));
    assert!(matches!(
        generic.description(),
        Cow::Borrowed("GenericNamedPass")
    ));
}

/// Test a borrowed pass, a boxed pass, and a trait object forward the
/// pass's own name and description.
#[test]
fn borrowed_and_boxed_passes_forward_their_name_and_description() {
    let mut pass = NamedPass::new("tests.named", "A named pass.");
    let boxed: Box<dyn CompilerPass<i64>> = Box::new(pass);

    let borrowed = &mut pass;

    assert_eq!(CompilerPass::<i64>::name(&borrowed), "tests.named");
    assert_eq!(CompilerPass::<i64>::description(&borrowed), "A named pass.");
    assert_eq!(boxed.name(), "tests.named");
    assert_eq!(boxed.description(), "A named pass.");
}

/// Test a borrowed pass runs through the borrow and keeps its state for the
/// caller.
#[test]
fn borrowed_pass_forwards_every_hook_and_keeps_its_state() {
    let mut pass = RecordingPass::default();

    let outcome = pass.execute(&0).expect("the run succeeds");

    assert_eq!(*outcome.output(), 1);
    assert_eq!(pass.calls.len(), 6);
}

/// Test a boxed pass object runs through the box.
#[test]
fn boxed_pass_forwards_every_hook() {
    let mut pass: Box<dyn CompilerPass<i64>> = Box::new(RecordingPass {
        skips: true,
        ..RecordingPass::default()
    });

    let outcome = pass.execute(&4).expect("the skipped run succeeds");

    assert_eq!(*outcome.output(), 4);
    assert!(!outcome.is_changed());
    assert!(outcome.is_skipped());
    assert_eq!(pass.name(), "RecordingPass");
}

// =============================================================================
// Registry
// =============================================================================

/// Adds the number of its own runs to an integer.
#[derive(Debug, Default)]
struct StatefulIncrement {
    runs: i64,
}

impl CompilerPass<i64> for StatefulIncrement {
    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        self.runs += 1;
        Ok(ir + self.runs)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// A pass over two IR types.
struct TwoIrPass;

impl CompilerPass<i64> for TwoIrPass {
    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        Ok(*ir)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

impl CompilerPass<String> for TwoIrPass {
    fn run(&mut self, ir: &String, _cx: &mut PassContext<'_>) -> Result<String, PassFailure> {
        Ok(ir.clone())
    }

    fn did_change(&mut self, input: &String, output: &String) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// Two pass types that share a name, from two modules.
mod first {
    use super::{CompilerPass, PassContext, PassFailure};

    /// Adds one to an integer.
    pub(super) struct Fold;

    impl CompilerPass<i64> for Fold {
        fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
            Ok(ir + 1)
        }

        fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
            Ok(input != output)
        }
    }
}

/// The second of the two same-named pass types.
mod second {
    use super::{CompilerPass, PassContext, PassFailure};

    /// Adds two to an integer.
    pub(super) struct Fold;

    impl CompilerPass<i64> for Fold {
        fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
            Ok(ir + 2)
        }

        fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
            Ok(input != output)
        }
    }
}

/// Return a registry holding `Increment` and `KeepInteger`.
fn build_registry() -> PassRegistry {
    let mut registry = PassRegistry::new();
    registry
        .register::<Increment, i64, i64>(|| Increment)
        .expect("the name is free");
    registry
        .register::<KeepInteger, i64, i64>(|| KeepInteger)
        .expect("the name is free");
    registry
}

/// Create the pass registered as `name` over integers and run it on `ir`.
fn create_and_run(registry: &PassRegistry, name: &str, ir: i64) -> i64 {
    let mut pass = registry
        .create::<i64, i64>(name)
        .expect("the pass is registered");
    pass.execute(&ir).expect("the run succeeds").into_output()
}

/// Test a new registry holds nothing.
#[test]
fn registry_new_is_empty() {
    let registry = PassRegistry::new();

    assert!(registry.is_empty());
    assert_eq!(registry.len(), 0);
    assert_eq!(registry.iter().count(), 0);
    assert!(registry.info("Increment").is_none());
}

/// Test a pass is registered under its own name and described by its own
/// description, read from one instance its factory builds.
#[test]
fn pass_registry_keys_a_pass_by_its_own_name() {
    let mut registry = PassRegistry::new();

    registry
        .register::<Increment, i64, i64>(|| Increment)
        .expect("the name is free");
    registry
        .register::<NamedPass, i64, i64>(|| NamedPass::new("tests.named", "A named pass."))
        .expect("the name is free");

    let names: Vec<_> = registry.iter().map(PassInfo::name).collect();
    assert_eq!(names, ["Increment", "tests.named"]);
    assert_eq!(
        registry.info("tests.named").map(PassInfo::description),
        Some("A named pass.")
    );
    let created = registry
        .create::<i64, i64>("tests.named")
        .expect("the pass is registered");
    assert_eq!(created.name(), "tests.named");
    assert_eq!(Increment.name(), "Increment");
}

/// Test `create` builds a fresh instance of the registered pass on every
/// call.
#[test]
fn registry_create_builds_a_new_instance_each_call() {
    let mut registry = PassRegistry::new();
    registry
        .register::<StatefulIncrement, i64, i64>(StatefulIncrement::default)
        .expect("the name is free");

    let mut first = registry
        .create::<i64, i64>("StatefulIncrement")
        .expect("the pass is registered");
    let first_output = first.execute(&2).expect("the run succeeds").into_output();
    let second_run = first.execute(&2).expect("the run succeeds").into_output();
    let fresh_output = create_and_run(&registry, "StatefulIncrement", 2);

    assert_eq!((first_output, second_run, fresh_output), (3, 4, 3));
    assert_eq!(first.name(), "StatefulIncrement");
}

/// Test registering builds exactly one instance, to read its name and
/// description, and `create` builds one more on every call.
#[test]
fn registry_register_builds_one_instance_to_read_its_name() {
    let builds = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&builds);
    let mut registry = PassRegistry::new();

    registry
        .register::<KeepInteger, i64, i64>(move || {
            counted.fetch_add(1, Ordering::SeqCst);
            KeepInteger
        })
        .expect("the name is free");
    let registered_builds = builds.load(Ordering::SeqCst);
    let _pass = registry
        .create::<i64, i64>("KeepInteger")
        .expect("the pass is registered");

    assert_eq!(registered_builds, 1);
    assert_eq!(builds.load(Ordering::SeqCst), 2);
}

/// Test `create` refuses a name nothing is registered under.
#[test]
fn registry_create_rejects_an_unknown_name() {
    let registry = build_registry();

    let error = registry
        .create::<i64, i64>("Fold")
        .err()
        .expect("nothing is registered as the name");

    assert!(
        matches!(&error, CreatePassError::UnknownPass { name, .. } if name == "Fold"),
        "{error:?}"
    );
}

/// Test `create` refuses a pass registered for other IR types, naming both
/// the registered and the requested types.
#[test]
fn registry_create_rejects_other_ir_types() {
    let registry = build_registry();

    let error = registry
        .create::<String, String>("KeepInteger")
        .err()
        .expect("the pass takes integers");

    let CreatePassError::IrTypeMismatch {
        name,
        registered_input,
        registered_output,
        requested_input,
        requested_output,
        ..
    } = &error
    else {
        panic!("expected an IR type mismatch, got {error:?}");
    };
    assert_eq!(name, "KeepInteger");
    assert_eq!(
        (*registered_input, *registered_output),
        (type_name::<i64>(), type_name::<i64>())
    );
    assert_eq!(
        (*requested_input, *requested_output),
        (type_name::<String>(), type_name::<String>())
    );
}

/// Test the registrations are listed in name order with their metadata and
/// the pass and IR types.
#[test]
fn registry_iter_lists_registrations_by_name() {
    let registry = build_registry();

    let names: Vec<_> = registry.iter().map(PassInfo::name).collect();
    let info = registry.info("KeepInteger").expect("registered");

    assert_eq!(names, ["Increment", "KeepInteger"]);
    assert_eq!(registry.len(), 2);
    assert!(!registry.is_empty());
    assert_eq!(info.name(), "KeepInteger");
    assert_eq!(info.description(), "KeepInteger");
    assert_eq!(info.pass_type_id(), TypeId::of::<KeepInteger>());
    assert_eq!(info.input_type_id(), TypeId::of::<i64>());
    assert_eq!(info.output_type_id(), TypeId::of::<i64>());
    assert_eq!(info.pass_type_name(), type_name::<KeepInteger>());
}

/// Test a blank name is refused, registering nothing.
#[rstest]
#[case::empty("")]
#[case::spaces("   ")]
#[case::tab_and_newline("\t\n")]
#[case::no_break_space("\u{a0}")]
fn registry_register_rejects_a_blank_name(#[case] name: &'static str) {
    let mut registry = PassRegistry::new();

    let error = registry
        .register::<NamedPass, i64, i64>(move || NamedPass::new(name, "A description."))
        .expect_err("the name is blank");

    assert_eq!(error, PassRegistrationError::EmptyName);
    assert!(registry.is_empty());
}

/// Test a name of characters Rust does not count as whitespace is not
/// blank, though Python's `str.isspace` counts the information separators.
#[test]
fn registry_register_accepts_a_name_rust_does_not_call_whitespace() {
    let mut registry = PassRegistry::new();

    registry
        .register::<NamedPass, i64, i64>(|| NamedPass::new("\u{1c}", "A description."))
        .expect("the name is not blank");

    assert!(registry.info("\u{1c}").is_some());
}

/// Test a blank description is refused, registering nothing.
#[rstest]
#[case::empty("")]
#[case::spaces("  ")]
fn registry_register_rejects_a_blank_description(#[case] description: &'static str) {
    let mut registry = PassRegistry::new();

    let error = registry
        .register::<NamedPass, i64, i64>(move || NamedPass::new("tests.named", description))
        .expect_err("the description is blank");

    assert!(
        matches!(&error, PassRegistrationError::EmptyDescription { name, .. } if name == "tests.named"),
        "{error:?}"
    );
    assert!(registry.is_empty());
}

/// Test a name registered to one pass type is refused to another pass type
/// of the same name from another module, leaving the first registered.
#[test]
fn same_named_pass_types_in_two_modules_take_one_name() {
    let mut registry = PassRegistry::new();
    registry
        .register::<first::Fold, i64, i64>(|| first::Fold)
        .expect("the name is free");

    let error = registry
        .register::<second::Fold, i64, i64>(|| second::Fold)
        .expect_err("the name is taken");

    assert!(
        matches!(
            &error,
            PassRegistrationError::NameTaken { name, registered_pass_type_name, .. }
                if name == "Fold" && *registered_pass_type_name == type_name::<first::Fold>()
        ),
        "{error:?}"
    );
    assert_eq!(create_and_run(&registry, "Fold", 0), 1);
    assert_eq!(registry.len(), 1);
}

/// Test a name registered to a pass over some IR types is refused to the
/// same pass over other IR types.
#[test]
fn registry_register_rejects_a_name_taken_by_the_same_pass_over_other_ir_types() {
    let mut registry = PassRegistry::new();
    registry
        .register::<TwoIrPass, i64, i64>(|| TwoIrPass)
        .expect("the name is free");

    let error = registry
        .register::<TwoIrPass, String, String>(|| TwoIrPass)
        .expect_err("the name is taken for other IR types");

    assert!(
        matches!(&error, PassRegistrationError::NameTaken { name, .. } if name == "TwoIrPass"),
        "{error:?}"
    );
    assert_eq!(
        registry.info("TwoIrPass").map(PassInfo::input_type_id),
        Some(TypeId::of::<i64>())
    );
}

/// Test registering a pass again with a new description is refused and
/// keeps the original description.
#[test]
fn registry_register_refuses_a_new_description_for_a_registered_pass() {
    let mut registry = PassRegistry::new();
    registry
        .register::<NamedPass, i64, i64>(|| NamedPass::new("tests.named", "Original."))
        .expect("the name is free");

    let error = registry
        .register::<NamedPass, i64, i64>(|| NamedPass::new("tests.named", "Different."))
        .expect_err("the description differs");

    assert!(
        matches!(
            &error,
            PassRegistrationError::DescriptionConflict { name, registered, requested, .. }
                if name == "tests.named" && registered == "Original." && requested == "Different."
        ),
        "{error:?}"
    );
    assert_eq!(
        registry.info("tests.named").map(PassInfo::description),
        Some("Original.")
    );
}

/// Test registering the same pass, name, and description again succeeds and
/// changes nothing.
#[test]
fn registry_register_is_idempotent_for_the_same_pass_and_description() {
    let mut registry = build_registry();
    let before = registry.info("Increment").expect("registered").clone();

    registry
        .register::<Increment, i64, i64>(|| Increment)
        .expect("the same registration is accepted again");

    assert_eq!(registry.info("Increment"), Some(&before));
    assert_eq!(registry.len(), 2);
}

/// Test two registries are independent values: one name maps to different
/// passes in each, and registering in one leaves the other unchanged.
#[test]
fn pass_registries_are_independent_values() {
    let mut first_registry = PassRegistry::new();
    let mut second_registry = PassRegistry::new();

    first_registry
        .register::<first::Fold, i64, i64>(|| first::Fold)
        .expect("the name is free");
    second_registry
        .register::<second::Fold, i64, i64>(|| second::Fold)
        .expect("the name is free in the other registry");

    assert_eq!(create_and_run(&first_registry, "Fold", 0), 1);
    assert_eq!(create_and_run(&second_registry, "Fold", 0), 2);
    assert_eq!(first_registry.len(), 1);
}

/// The registry error one registry operation produces.
#[derive(Debug, Clone, Copy)]
enum RegistrationFailure {
    BlankName,
    BlankDescription,
    NameTaken,
    DescriptionConflict,
}

/// Produce the registration error `failure` names.
fn produce_registration_error(failure: RegistrationFailure) -> PassRegistrationError {
    let mut registry = PassRegistry::new();
    registry
        .register::<NamedPass, i64, i64>(|| NamedPass::new("fold", "Folds."))
        .expect("the name is free");
    let result = match failure {
        RegistrationFailure::BlankName => {
            registry.register::<NamedPass, i64, i64>(|| NamedPass::new(" ", "Folds."))
        }
        RegistrationFailure::BlankDescription => {
            registry.register::<NamedPass, i64, i64>(|| NamedPass::new("fold", ""))
        }
        RegistrationFailure::NameTaken => registry
            .register::<GenericNamedPass<()>, i64, i64>(|| GenericNamedPass(PhantomData))
            .and_then(|()| {
                registry
                    .register::<GenericNamedPass<u8>, i64, i64>(|| GenericNamedPass(PhantomData))
            }),
        RegistrationFailure::DescriptionConflict => {
            registry.register::<NamedPass, i64, i64>(|| NamedPass::new("fold", "Unfolds."))
        }
    };
    result.expect_err("the registration is refused")
}

/// Test each registration error renders its one-line message.
#[rstest]
#[case::blank_name(RegistrationFailure::BlankName, "pass name is blank")]
#[case::blank_description(
    RegistrationFailure::BlankDescription,
    "pass \"fold\" has a blank description"
)]
#[case::name_taken(
    RegistrationFailure::NameTaken,
    "pass name \"GenericNamedPass\" is already registered to another pass"
)]
#[case::description_conflict(
    RegistrationFailure::DescriptionConflict,
    "pass \"fold\" is already registered with a different description"
)]
fn pass_registration_error_display_table(
    #[case] failure: RegistrationFailure,
    #[case] expected: &str,
) {
    let error = produce_registration_error(failure);

    assert_eq!(error.to_string(), expected);
    assert!(error.source().is_none());
}

/// Test each creation error renders its one-line message.
#[rstest]
#[case::unknown_pass("fold", "no pass is registered as \"fold\"")]
#[case::ir_type_mismatch(
    "Increment",
    "pass \"Increment\" takes i64 to i64, not alloc::string::String to alloc::string::String"
)]
fn create_pass_error_display_table(#[case] name: &str, #[case] expected: &str) {
    let registry = build_registry();

    let error = registry
        .create::<String, String>(name)
        .err()
        .expect("the pass cannot be created");

    assert_eq!(error.to_string(), expected);
    assert!(error.source().is_none());
}
