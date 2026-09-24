//! Tests for standalone pass runs in `fhy_core::pass`: the guarded lifecycle,
//! hook-error wrapping and pass-through, the pass context, preserved analyses,
//! analysis ids, node identities, pass names, the registry, and the per-pass
//! run counters.
//!
//! Public API only. The registry and the run counters are process-wide and
//! the tests run in parallel, so every test that registers a pass or reads a
//! counter uses a pass type and names no other test uses. The process-wide
//! total is tested in `pass_infrastructure_run_count_stories`, alone in its
//! binary.

use crate::support::pass_ir;

use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use fhy_core::diagnostic::{DiagnosticLevel, Note, get_rationale_note_kind};
use fhy_core::pass::{
    AnalysisId, CompilerPass, ExecutePass, NodeHandle, NodeIdentity, PassContext, PassError,
    PassFailure, PassHook, PreservedAnalyses, create_pass, register_pass, registered_passes,
    run_count, run_count_of,
};
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

    fn should_run(&mut self, _ir: &i64, _cx: &mut PassContext<'_>) -> Result<bool, PassFailure> {
        self.record(PassHook::ShouldRun)?;
        Ok(!self.skips)
    }

    fn noop_output(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        self.record(PassHook::NoopOutput)?;
        Ok(*ir)
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

/// Fails in one hook with the error `<hook>-broken`; skips the run exactly
/// when the failing hook is `noop_output`.
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

    fn should_run(&mut self, _ir: &i64, _cx: &mut PassContext<'_>) -> Result<bool, PassFailure> {
        self.check(PassHook::ShouldRun)?;
        Ok(self.hook != PassHook::NoopOutput)
    }

    fn noop_output(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        self.check(PassHook::NoopOutput)?;
        Ok(*ir)
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

/// The message of the failure `inner` produces.
fn describe_inner_failure(inner: InnerFailure) -> &'static str {
    match inner {
        InnerFailure::Validation => "Pass \"RejectInput\" failed validate_input with rejected",
        InnerFailure::Execution => "Pass \"CrashInRun\" failed run with crashed",
    }
}

/// The name of the pass that produces the failure `inner`.
fn name_inner_pass(inner: InnerFailure) -> &'static str {
    match inner {
        InnerFailure::Validation => "RejectInput",
        InnerFailure::Execution => "CrashInRun",
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
/// runs another pass and propagates its failure does.
struct HandOverPass {
    hook: PassHook,
    failure: Option<PassError>,
}

impl HandOverPass {
    /// Build the pass that hands over the failure `inner` from `hook`.
    fn new(hook: PassHook, inner: InnerFailure) -> Self {
        Self {
            hook,
            failure: Some(produce_inner_failure(inner)),
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
    fn validate_input(&mut self, _ir: &i64, _cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        self.check(PassHook::ValidateInput)
    }

    fn should_run(&mut self, _ir: &i64, _cx: &mut PassContext<'_>) -> Result<bool, PassFailure> {
        self.check(PassHook::ShouldRun)?;
        Ok(self.hook != PassHook::NoopOutput)
    }

    fn noop_output(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        self.check(PassHook::NoopOutput)?;
        Ok(*ir)
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
            "should_run",
            "run",
            "validate_output",
            "did_change",
            "preserved_analyses(changed=true)",
        ]
    );
}

/// Test a skipped run calls `noop_output` instead of `run` and asks for the
/// analyses of an unchanged run.
#[test]
fn execute_skipped_run_calls_noop_output_instead_of_run() {
    let mut pass = RecordingPass {
        skips: true,
        ..RecordingPass::default()
    };

    pass.execute(&0).expect("the run succeeds");

    assert_eq!(
        pass.calls,
        [
            "validate_input",
            "should_run",
            "noop_output",
            "preserved_analyses(changed=false)",
        ]
    );
}

/// Test a skipped run outputs the no-op output unchanged, preserves every
/// analysis, and keeps the diagnostics reported before the skip.
#[test]
fn execute_skipped_run_outputs_the_noop_output() {
    struct SkippedPass;

    impl CompilerPass<i64> for SkippedPass {
        fn should_run(&mut self, _ir: &i64, cx: &mut PassContext<'_>) -> Result<bool, PassFailure> {
            cx.report_text(DiagnosticLevel::Info, "skip requested", None);
            Ok(false)
        }

        fn noop_output(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
            Ok(ir + 100)
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
    assert_eq!(outcome.preserved_analyses(), &PreservedAnalyses::all());
    assert_eq!(outcome.diagnostics().len(), 1);
    assert_eq!(outcome.diagnostics()[0].level(), DiagnosticLevel::Info);
    assert_eq!(outcome.diagnostics()[0].message_text(), "skip requested");
}

/// Test a pass without a no-op output fails a skipped run.
#[test]
fn execute_skipped_run_fails_without_a_noop_output() {
    struct NoNoopPass;

    impl CompilerPass<i64> for NoNoopPass {
        fn should_run(
            &mut self,
            _ir: &i64,
            _cx: &mut PassContext<'_>,
        ) -> Result<bool, PassFailure> {
            Ok(false)
        }

        fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
            Ok(*ir)
        }

        fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
            Ok(input != output)
        }
    }

    let error = NoNoopPass
        .execute(&0)
        .expect_err("a skipped run needs a no-op output");

    assert_eq!(
        error.to_string(),
        "Pass \"NoNoopPass\" failed noop_output with the pass has no no-op output"
    );
    assert!(error.is_execution_failure());
    assert_eq!(error.failed_hook(), Some(PassHook::NoopOutput));
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
#[case::should_run(PassHook::ShouldRun, "should_run")]
#[case::noop_output(PassHook::NoopOutput, "noop_output")]
#[case::run(PassHook::Run, "run")]
#[case::validate_output(PassHook::ValidateOutput, "validate_output")]
#[case::did_change(PassHook::DidChange, "did_change")]
#[case::preserved_analyses(PassHook::PreservedAnalyses, "preserved_analyses")]
fn pass_hook_renders_the_method_name(#[case] hook: PassHook, #[case] expected: &str) {
    assert_eq!(hook.as_str(), expected);
    assert_eq!(hook.to_string(), expected);
}

/// Test a hook error becomes a pass error of the hook's class that names the
/// pass and the hook, keeps the hook's error as its source, and ends the
/// diagnostics with an error recording the failure.
#[rstest]
#[case::validate_input(PassHook::ValidateInput, true)]
#[case::should_run(PassHook::ShouldRun, false)]
#[case::noop_output(PassHook::NoopOutput, false)]
#[case::run(PassHook::Run, false)]
#[case::validate_output(PassHook::ValidateOutput, true)]
#[case::did_change(PassHook::DidChange, false)]
#[case::preserved_analyses(PassHook::PreservedAnalyses, false)]
fn execute_wraps_a_hook_error_naming_the_pass_and_hook(
    #[case] hook: PassHook,
    #[case] is_validation: bool,
) {
    let expected_message = format!("Pass \"FailingHookPass\" failed {hook} with {hook}-broken");

    let error = FailingHookPass { hook }
        .execute(&1)
        .expect_err("the hook fails");

    assert_eq!(error.to_string(), expected_message);
    assert_eq!(error.pass_name(), Some("FailingHookPass"));
    assert_eq!(error.failed_hook(), Some(hook));
    assert_eq!(error.is_validation_failure(), is_validation);
    assert_eq!(error.is_execution_failure(), !is_validation);
    assert!(!error.is_non_convergence());
    assert!(error.verification_report().is_none());
    let source = error.source().expect("the hook's error is the source");
    assert_eq!(source.to_string(), format!("{hook}-broken"));
    assert!(source.downcast_ref::<HookFailure>().is_some(), "{source:?}");
    let last = error.diagnostics().last().expect("an error diagnostic");
    assert_eq!(last.level(), DiagnosticLevel::Error);
    assert_eq!(last.message_text(), expected_message);
    assert_eq!(last.source(), "FailingHookPass");
    assert_eq!(last.detail(), None);
}

/// Test a failing hook ends the run: no later hook is called.
#[rstest]
#[case::validate_input(PassHook::ValidateInput, &["validate_input"])]
#[case::should_run(PassHook::ShouldRun, &["validate_input", "should_run"])]
#[case::run(PassHook::Run, &["validate_input", "should_run", "run"])]
#[case::validate_output(
    PassHook::ValidateOutput,
    &["validate_input", "should_run", "run", "validate_output"]
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
                "Pass \"WarnThenCrash\" failed run with boom"
            ),
        ]
    );
}

/// Test a pass error of the class a hook may hand through leaves the run
/// unchanged; `run` hands through both classes.
#[rstest]
#[case::validate_input(PassHook::ValidateInput, InnerFailure::Validation)]
#[case::should_run(PassHook::ShouldRun, InnerFailure::Execution)]
#[case::noop_output(PassHook::NoopOutput, InnerFailure::Execution)]
#[case::run_validation(PassHook::Run, InnerFailure::Validation)]
#[case::run_execution(PassHook::Run, InnerFailure::Execution)]
#[case::validate_output(PassHook::ValidateOutput, InnerFailure::Validation)]
#[case::did_change(PassHook::DidChange, InnerFailure::Execution)]
#[case::preserved_analyses(PassHook::PreservedAnalyses, InnerFailure::Execution)]
fn execute_hands_a_matching_pass_error_through_unchanged(
    #[case] hook: PassHook,
    #[case] inner: InnerFailure,
) {
    let error = HandOverPass::new(hook, inner)
        .execute(&0)
        .expect_err("the hook hands over a failure");

    assert_eq!(error.to_string(), describe_inner_failure(inner));
    assert_eq!(
        error.is_validation_failure(),
        matches!(inner, InnerFailure::Validation)
    );
    assert_eq!(error.pass_name(), Some(name_inner_pass(inner)));
    assert_eq!(error.diagnostics().len(), 1);
    assert_eq!(
        error.diagnostics()[0].message_text(),
        describe_inner_failure(inner)
    );
}

/// Test a pass error of the other class is wrapped like any hook error.
#[rstest]
#[case::validate_input(PassHook::ValidateInput, InnerFailure::Execution, true)]
#[case::validate_output(PassHook::ValidateOutput, InnerFailure::Execution, true)]
#[case::should_run(PassHook::ShouldRun, InnerFailure::Validation, false)]
#[case::did_change(PassHook::DidChange, InnerFailure::Validation, false)]
fn execute_wraps_a_pass_error_of_the_other_class(
    #[case] hook: PassHook,
    #[case] inner: InnerFailure,
    #[case] is_validation: bool,
) {
    let expected_message = format!(
        "Pass \"HandOverPass\" failed {hook} with {}",
        describe_inner_failure(inner)
    );

    let error = HandOverPass::new(hook, inner)
        .execute(&0)
        .expect_err("the hook hands over a failure");

    assert_eq!(error.to_string(), expected_message);
    assert_eq!(error.is_validation_failure(), is_validation);
    assert_eq!(error.pass_name(), Some("HandOverPass"));
    assert_eq!(error.failed_hook(), Some(hook));
    let source = error.source().expect("the handed-over error is the source");
    let inner_error = source
        .downcast_ref::<PassError>()
        .expect("the source is the inner pass error");
    assert_eq!(inner_error.to_string(), describe_inner_failure(inner));
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
    let mut pass = ClosurePass::new("tests.core.detail", |ir, cx| {
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

/// Test `report` keeps a structured note as given.
#[test]
fn report_keeps_a_structured_note() {
    let note = Note::new("structured-message", get_rationale_note_kind().clone());
    let reported = note.clone();
    let mut pass = ClosurePass::new("tests.core.note", move |ir, cx| {
        cx.report(DiagnosticLevel::Error, reported.clone(), None);
        Ok(ir.clone())
    });

    let outcome = pass.execute(&BoxIr::new(0)).expect("the run succeeds");

    assert_eq!(outcome.diagnostics()[0].message(), &note);
    assert_eq!(outcome.diagnostics()[0].source(), "tests.core.note");
}

/// Test the context names the running pass and lists the diagnostics
/// reported so far.
#[test]
fn pass_context_exposes_the_pass_name_and_diagnostics_so_far() {
    let mut observed = Vec::new();
    let mut pass = ClosurePass::new("tests.core.context_view", |ir, cx| {
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
            ("tests.core.context_view".to_owned(), 0),
            ("tests.core.context_view".to_owned(), 1),
        ]
    );
}

/// Test a pass run on its own computes an analysis afresh on every request.
#[test]
fn pass_context_analysis_recomputes_on_every_call_outside_a_manager() {
    let mut observed = Vec::new();
    let mut pass = ClosurePass::new("tests.core.standalone_analysis", |ir, cx| {
        observed.push(*cx.analysis::<DoubleAnalysis, _>(ir));
        observed.push(*cx.analysis::<DoubleAnalysis, _>(ir));
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

/// Test the all-preserving set preserves every analysis and lists no ids.
#[test]
fn preserved_analyses_all_preserves_every_analysis() {
    let all = PreservedAnalyses::all();

    assert!(all.preserves_all());
    assert!(all.is_preserved::<DoubleAnalysis>());
    assert!(all.is_preserved::<ParityAnalysis>());
    assert!(all.is_id_preserved(AnalysisId::of::<String>()));
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
        "it::support::pass_ir::DoubleAnalysis"
    );
}

/// Test analysis ids order by type name.
#[test]
fn analysis_id_orders_by_type_name() {
    struct Alpha;
    struct Beta;

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

/// Test an unregistered pass is named after its type without module path or
/// generic arguments, and described by its name.
#[test]
fn name_defaults_to_the_type_name_without_path_or_generics() {
    let generic = GenericNamedPass::<Vec<String>>(PhantomData);

    assert_eq!(generic.name(), "GenericNamedPass");
    assert_eq!(generic.description(), "GenericNamedPass");
    assert_eq!(Increment.name(), "Increment");
}

/// Registered in `name_of_a_registered_pass_is_its_registered_name`.
struct RegisteredNamePass;

impl CompilerPass<i64> for RegisteredNamePass {
    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        Ok(*ir)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// Test a registered pass takes its registered name and description, also
/// through a borrow, a box, and a trait object.
#[test]
fn name_of_a_registered_pass_is_its_registered_name() {
    register_pass::<RegisteredNamePass, i64, i64>(
        "tests.core.registered_name",
        "A pass with a registered name.",
        || RegisteredNamePass,
    )
    .expect("the name is free");
    let mut pass = RegisteredNamePass;
    let boxed: Box<dyn CompilerPass<i64>> = Box::new(RegisteredNamePass);

    assert_eq!(pass.name(), "tests.core.registered_name");
    assert_eq!(pass.description(), "A pass with a registered name.");
    let borrowed = &mut pass;
    assert_eq!(
        CompilerPass::<i64>::name(&borrowed),
        "tests.core.registered_name"
    );
    assert_eq!(
        CompilerPass::<i64>::description(&borrowed),
        "A pass with a registered name."
    );
    assert_eq!(boxed.name(), "tests.core.registered_name");
    assert_eq!(boxed.description(), "A pass with a registered name.");
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
    assert_eq!(pass.name(), "RecordingPass");
}

// =============================================================================
// Registry
// =============================================================================

/// Registered in `create_pass_builds_a_new_instance_of_the_registered_pass`.
#[derive(Default)]
struct CreatablePass {
    runs: i64,
}

impl CompilerPass<i64> for CreatablePass {
    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        self.runs += 1;
        Ok(ir + self.runs)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// Test `create_pass` builds a fresh instance of the registered pass on
/// every call.
#[test]
fn create_pass_builds_a_new_instance_of_the_registered_pass() {
    register_pass::<CreatablePass, i64, i64>(
        "tests.core.creatable",
        "Add the number of runs to an integer.",
        CreatablePass::default,
    )
    .expect("the name is free");

    let mut first = create_pass::<i64, i64>("tests.core.creatable").expect("registered");
    let first_output = first.execute(&2).expect("the run succeeds").into_output();
    let second_run = first.execute(&2).expect("the run succeeds").into_output();
    let mut second = create_pass::<i64, i64>("tests.core.creatable").expect("registered");
    let fresh_output = second.execute(&2).expect("the run succeeds").into_output();

    assert_eq!((first_output, second_run, fresh_output), (3, 4, 3));
    assert_eq!(first.name(), "tests.core.creatable");
}

/// Test `create_pass` refuses a name nothing is registered under.
#[test]
fn create_pass_rejects_an_unknown_name() {
    let error = create_pass::<i64, i64>("tests.core.never_registered")
        .err()
        .expect("nothing is registered under the name");

    assert_eq!(
        error.to_string(),
        "Unknown pass \"tests.core.never_registered\"."
    );
}

/// Registered in `create_pass_rejects_other_ir_types`.
struct IntegerOnlyPass;

impl CompilerPass<i64> for IntegerOnlyPass {
    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        Ok(*ir)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// Test `create_pass` refuses a pass registered for other IR types.
#[test]
fn create_pass_rejects_other_ir_types() {
    register_pass::<IntegerOnlyPass, i64, i64>(
        "tests.core.integer_only",
        "Pass over integers only.",
        || IntegerOnlyPass,
    )
    .expect("the name is free");

    let error = create_pass::<String, String>("tests.core.integer_only")
        .err()
        .expect("the pass takes integers");

    assert_eq!(
        error.to_string(),
        "Pass \"tests.core.integer_only\" takes i64 to i64, not \
         alloc::string::String to alloc::string::String."
    );
}

/// Registered in `registered_passes_lists_a_registration`.
struct ListedPass;

impl CompilerPass<i64> for ListedPass {
    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        Ok(*ir)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// Test a registration shows up in the listing with its metadata.
#[test]
fn registered_passes_lists_a_registration() {
    register_pass::<ListedPass, i64, i64>("tests.core.listed", "A listed pass.", || ListedPass)
        .expect("the name is free");

    let passes = registered_passes();

    let info = passes.get("tests.core.listed").expect("listed");
    assert_eq!(info.name(), "tests.core.listed");
    assert_eq!(info.description(), "A listed pass.");
    assert_eq!(info.type_name(), "it::pass::core_stories::ListedPass");
}

/// Registered in `register_pass_does_not_call_the_factory`.
struct LazilyBuiltPass;

impl CompilerPass<i64> for LazilyBuiltPass {
    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        Ok(*ir)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// Test registering a pass builds nothing; `create_pass` calls the factory.
#[test]
fn register_pass_does_not_call_the_factory() {
    let builds = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&builds);

    register_pass::<LazilyBuiltPass, i64, i64>("tests.core.lazy", "Built lazily.", move || {
        counted.fetch_add(1, Ordering::SeqCst);
        LazilyBuiltPass
    })
    .expect("the name is free");
    let registered_builds = builds.load(Ordering::SeqCst);
    let _pass = create_pass::<i64, i64>("tests.core.lazy").expect("registered");

    assert_eq!(registered_builds, 0);
    assert_eq!(builds.load(Ordering::SeqCst), 1);
}

/// Test an empty or blank name is refused.
#[rstest]
#[case::empty("")]
#[case::spaces("   ")]
#[case::tab_and_newline("\t\n")]
#[case::information_separator("\u{1c}")]
fn register_pass_rejects_an_empty_name(#[case] name: &str) {
    let error = register_pass::<Increment, i64, i64>(name, "non-empty description", || Increment)
        .expect_err("the name is blank");

    assert_eq!(error.to_string(), "Pass name cannot be empty.");
}

/// Test an empty or blank description is refused, registering nothing.
#[rstest]
#[case::empty("tests.core.empty_description.empty", "")]
#[case::spaces("tests.core.empty_description.spaces", "  ")]
fn register_pass_rejects_an_empty_description(#[case] name: &str, #[case] description: &str) {
    let error = register_pass::<Increment, i64, i64>(name, description, || Increment)
        .expect_err("the description is blank");

    assert_eq!(error.to_string(), "Pass description cannot be empty.");
    assert!(!registered_passes().contains_key(name));
}

/// Owns `tests.core.taken` in
/// `register_pass_rejects_a_name_taken_by_another_pass_type`.
struct OwnerPass;

impl CompilerPass<i64> for OwnerPass {
    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        Ok(*ir)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// Tries to take `tests.core.taken` from [`OwnerPass`].
struct IntruderPass;

impl CompilerPass<i64> for IntruderPass {
    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        Ok(*ir)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// Test a name registered to one pass type is refused to another, leaving
/// both types as they were.
#[test]
fn register_pass_rejects_a_name_taken_by_another_pass_type() {
    register_pass::<OwnerPass, i64, i64>("tests.core.taken", "Owns the name.", || OwnerPass)
        .expect("the name is free");

    let error =
        register_pass::<IntruderPass, i64, i64>("tests.core.taken", "Wants the name.", || {
            IntruderPass
        })
        .expect_err("the name is taken");

    assert_eq!(
        error.to_string(),
        "Pass name \"tests.core.taken\" is already registered by \
         it::pass::core_stories::OwnerPass with description \"Owns the name.\"."
    );
    assert_eq!(
        registered_passes()["tests.core.taken"].type_name(),
        "it::pass::core_stories::OwnerPass"
    );
    assert_eq!(IntruderPass.name(), "IntruderPass");
}

/// A pass over two IR types, registered for one of them in
/// `register_pass_rejects_a_name_taken_by_the_same_pass_over_other_ir_types`.
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

/// Test a name registered to a pass over some IR types is refused to the
/// same pass over other IR types.
#[test]
fn register_pass_rejects_a_name_taken_by_the_same_pass_over_other_ir_types() {
    register_pass::<TwoIrPass, i64, i64>("tests.core.two_ir", "Over integers.", || TwoIrPass)
        .expect("the name is free");

    let error =
        register_pass::<TwoIrPass, String, String>("tests.core.two_ir", "Over integers.", || {
            TwoIrPass
        })
        .expect_err("the name is taken for other IR types");

    assert_eq!(
        error.to_string(),
        "Pass name \"tests.core.two_ir\" is already registered by \
         it::pass::core_stories::TwoIrPass with description \"Over integers.\"."
    );
    assert_eq!(
        create_pass::<i64, i64>("tests.core.two_ir").map(|pass| pass.name()),
        Ok("tests.core.two_ir".to_owned())
    );
}

/// Registered twice in
/// `register_pass_refuses_a_new_description_for_a_registered_pass`.
struct MismatchPass;

impl CompilerPass<i64> for MismatchPass {
    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        Ok(*ir)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// Test registering a pass again with a new description is refused and
/// keeps the original description.
#[test]
fn register_pass_refuses_a_new_description_for_a_registered_pass() {
    register_pass::<MismatchPass, i64, i64>("tests.core.mismatch", "Original description.", || {
        MismatchPass
    })
    .expect("the name is free");

    let error = register_pass::<MismatchPass, i64, i64>(
        "tests.core.mismatch",
        "A different description.",
        || MismatchPass,
    )
    .expect_err("the description differs");

    assert_eq!(
        error.to_string(),
        "Pass name \"tests.core.mismatch\" is already registered by \
         it::pass::core_stories::MismatchPass with description \
         \"Original description.\"; refusing to overwrite with new description \
         \"A different description.\"."
    );
    assert_eq!(
        registered_passes()["tests.core.mismatch"].description(),
        "Original description."
    );
    assert_eq!(MismatchPass.description(), "Original description.");
}

/// Registered twice in
/// `register_pass_is_idempotent_for_the_same_pass_and_description`.
struct IdempotentPass;

impl CompilerPass<i64> for IdempotentPass {
    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        Ok(*ir)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// Test registering the same pass, name, and description again succeeds and
/// changes nothing.
#[test]
fn register_pass_is_idempotent_for_the_same_pass_and_description() {
    register_pass::<IdempotentPass, i64, i64>(
        "tests.core.idempotent",
        "Idempotent registration.",
        || IdempotentPass,
    )
    .expect("the name is free");
    let before = registered_passes()["tests.core.idempotent"].clone();

    register_pass::<IdempotentPass, i64, i64>(
        "tests.core.idempotent",
        "Idempotent registration.",
        || IdempotentPass,
    )
    .expect("the same registration is accepted again");

    assert_eq!(registered_passes()["tests.core.idempotent"], before);
    assert_eq!(before.type_name(), "it::pass::core_stories::IdempotentPass");
}

/// Registered under two names in
/// `register_pass_under_a_second_name_makes_it_the_default_name`.
struct TwoNamePass;

impl CompilerPass<i64> for TwoNamePass {
    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        Ok(*ir)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// Test a pass registered under a second name takes that name and
/// description, and both registrations stay listed.
#[test]
fn register_pass_under_a_second_name_makes_it_the_default_name() {
    register_pass::<TwoNamePass, i64, i64>("tests.core.first_name", "First.", || TwoNamePass)
        .expect("the name is free");
    register_pass::<TwoNamePass, i64, i64>("tests.core.second_name", "Second.", || TwoNamePass)
        .expect("the name is free");

    let passes = registered_passes();

    assert_eq!(TwoNamePass.name(), "tests.core.second_name");
    assert_eq!(TwoNamePass.description(), "Second.");
    assert!(passes.contains_key("tests.core.first_name"));
    assert!(passes.contains_key("tests.core.second_name"));
}

// =============================================================================
// Run counters
// =============================================================================

/// Counted in `run_count_counts_each_executed_run`.
struct RunCountedPass;

impl CompilerPass<i64> for RunCountedPass {
    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        Ok(ir + 1)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// Test every executed run counts under the pass's default name.
#[test]
fn run_count_counts_each_executed_run() {
    let before = run_count::<RunCountedPass>();

    RunCountedPass.execute(&0).expect("the run succeeds");
    RunCountedPass.execute(&1).expect("the run succeeds");

    assert_eq!(run_count::<RunCountedPass>(), before + 2);
    assert_eq!(run_count_of("RunCountedPass"), before + 2);
}

/// Runs only on positive input; counted in `run_count_ignores_skipped_runs`.
struct GatedPass;

impl CompilerPass<i64> for GatedPass {
    fn should_run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<bool, PassFailure> {
        Ok(*ir > 0)
    }

    fn noop_output(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        Ok(*ir)
    }

    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        Ok(ir + 1)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// Test skipped runs do not count and executed runs do.
#[test]
fn run_count_ignores_skipped_runs() {
    let before = run_count::<GatedPass>();

    GatedPass.execute(&0).expect("the skipped run succeeds");
    GatedPass.execute(&0).expect("the skipped run succeeds");
    let after_skips = run_count::<GatedPass>();
    GatedPass.execute(&1).expect("the run succeeds");
    GatedPass.execute(&2).expect("the run succeeds");

    assert_eq!(after_skips, before);
    assert_eq!(run_count::<GatedPass>(), before + 2);
}

/// Fails in one hook under a name unique to that hook.
struct CountedFailingPass {
    inner: FailingHookPass,
}

impl CountedFailingPass {
    /// Return the name the pass counts under.
    fn counter_name(hook: PassHook) -> String {
        format!("tests.core.counted_failure.{hook}")
    }
}

impl CompilerPass<i64> for CountedFailingPass {
    fn name(&self) -> String {
        Self::counter_name(self.inner.hook)
    }

    fn validate_input(&mut self, ir: &i64, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        self.inner.validate_input(ir, cx)
    }

    fn should_run(&mut self, ir: &i64, cx: &mut PassContext<'_>) -> Result<bool, PassFailure> {
        self.inner.should_run(ir, cx)
    }

    fn noop_output(&mut self, ir: &i64, cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        self.inner.noop_output(ir, cx)
    }

    fn run(&mut self, ir: &i64, cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        self.inner.run(ir, cx)
    }

    fn validate_output(
        &mut self,
        input: &i64,
        output: &i64,
        cx: &mut PassContext<'_>,
    ) -> Result<(), PassFailure> {
        self.inner.validate_output(input, output, cx)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        self.inner.did_change(input, output)
    }

    fn preserved_analyses(
        &mut self,
        input: &i64,
        output: &i64,
        changed: bool,
    ) -> Result<PreservedAnalyses, PassFailure> {
        self.inner.preserved_analyses(input, output, changed)
    }
}

/// Test a run counts once `run` is reached, even when it or a later hook
/// fails, and not when an earlier hook fails.
#[rstest]
#[case::validate_input(PassHook::ValidateInput, 0)]
#[case::should_run(PassHook::ShouldRun, 0)]
#[case::noop_output(PassHook::NoopOutput, 0)]
#[case::run(PassHook::Run, 2)]
#[case::validate_output(PassHook::ValidateOutput, 2)]
#[case::did_change(PassHook::DidChange, 2)]
#[case::preserved_analyses(PassHook::PreservedAnalyses, 2)]
fn run_count_counts_a_run_that_fails_after_it_started(
    #[case] hook: PassHook,
    #[case] expected: u64,
) {
    let mut pass = CountedFailingPass {
        inner: FailingHookPass { hook },
    };

    pass.execute(&1).expect_err("the hook fails");
    pass.execute(&2).expect_err("the hook fails");

    assert_eq!(
        run_count_of(&CountedFailingPass::counter_name(hook)),
        expected
    );
}

/// Registered in `run_count_counts_a_registered_pass_under_its_registered_name`.
struct CountedRegisteredPass;

impl CompilerPass<i64> for CountedRegisteredPass {
    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        Ok(*ir)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// Test a registered pass counts under its registered name.
#[test]
fn run_count_counts_a_registered_pass_under_its_registered_name() {
    register_pass::<CountedRegisteredPass, i64, i64>(
        "tests.core.counted_registered",
        "Counted under its registered name.",
        || CountedRegisteredPass,
    )
    .expect("the name is free");

    CountedRegisteredPass.execute(&0).expect("the run succeeds");

    assert_eq!(run_count_of("tests.core.counted_registered"), 1);
    assert_eq!(run_count::<CountedRegisteredPass>(), 1);
    assert_eq!(run_count_of("CountedRegisteredPass"), 0);
}

/// Overrides its name; counted in
/// `run_count_counts_an_overridden_name_under_that_name`.
struct RenamedPass;

impl CompilerPass<i64> for RenamedPass {
    fn name(&self) -> String {
        "tests.core.renamed".to_owned()
    }

    fn run(&mut self, ir: &i64, _cx: &mut PassContext<'_>) -> Result<i64, PassFailure> {
        Ok(*ir)
    }

    fn did_change(&mut self, input: &i64, output: &i64) -> Result<bool, PassFailure> {
        Ok(input != output)
    }
}

/// Test a pass that overrides its name counts under that name, not under its
/// type's default name.
#[test]
fn run_count_counts_an_overridden_name_under_that_name() {
    RenamedPass.execute(&0).expect("the run succeeds");

    assert_eq!(run_count_of("tests.core.renamed"), 1);
    assert_eq!(run_count::<RenamedPass>(), 0);
}

/// Test a name nothing ran under counts zero.
#[test]
fn run_count_of_an_unused_name_is_zero() {
    assert_eq!(run_count_of("tests.core.never_ran"), 0);
}
