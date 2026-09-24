//! Tests for `fhy_core::pass::ValidationManager`: collect-all
//! validation, failing validators, and the aggregated report.
//!
//! Public API only; nothing here reads process-global state.

#[path = "common/pass_ir.rs"]
pub mod pass_ir;

use fhy_core::diagnostic::{DiagnosticLevel, Note, get_suggestion_note_kind};
use fhy_core::identifier::{HasIdentifier, Identifier};
use fhy_core::pass::{
    CompilerPass, ExecutePass, PassContext, PassError, PassFailure, PreservedAnalyses,
    ValidationManager,
};
use pass_ir::{BoxIr, DoubleAnalysis};
use rstest::rstest;

// =============================================================================
// Helpers
// =============================================================================

/// What a [`ScriptedValidator`] does in its run.
enum Step {
    /// Report a diagnostic.
    Report {
        level: DiagnosticLevel,
        message: &'static str,
        detail: Option<&'static str>,
    },
    /// Fail with an error reported as the message.
    Fail(&'static str),
    /// Fail with a pass error produced beforehand.
    HandOver(PassError),
}

/// A validator named explicitly that performs a script in its run.
struct ScriptedValidator {
    name: &'static str,
    steps: Vec<Step>,
}

impl ScriptedValidator {
    /// Build the validator `name` performing `steps`.
    fn new(name: &'static str, steps: Vec<Step>) -> Self {
        Self { name, steps }
    }

    /// Build the validator `name` that reports `message` at `level`.
    fn reporting(name: &'static str, level: DiagnosticLevel, message: &'static str) -> Self {
        Self::new(name, vec![report(level, message)])
    }

    /// Build the validator `name` that reports nothing.
    fn clean(name: &'static str) -> Self {
        Self::new(name, Vec::new())
    }
}

impl CompilerPass<BoxIr, ()> for ScriptedValidator {
    fn name(&self) -> String {
        self.name.to_owned()
    }

    fn run(&mut self, _ir: &BoxIr, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        for step in self.steps.drain(..) {
            match step {
                Step::Report {
                    level,
                    message,
                    detail,
                } => cx.report_text(level, message, detail.map(str::to_owned)),
                Step::Fail(message) => return Err(message.into()),
                Step::HandOver(error) => return Err(Box::new(error)),
            }
        }
        Ok(())
    }

    fn did_change(&mut self, _input: &BoxIr, _output: &()) -> Result<bool, PassFailure> {
        Ok(false)
    }
}

/// Build the step that reports `message` at `level` without detail.
fn report(level: DiagnosticLevel, message: &'static str) -> Step {
    Step::Report {
        level,
        message,
        detail: None,
    }
}

/// Rejects every input with the error `rejected`.
struct RejectInput;

impl CompilerPass<BoxIr, ()> for RejectInput {
    fn validate_input(
        &mut self,
        _ir: &BoxIr,
        _cx: &mut PassContext<'_>,
    ) -> Result<(), PassFailure> {
        Err("rejected".into())
    }

    fn run(&mut self, _ir: &BoxIr, _cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        Ok(())
    }

    fn did_change(&mut self, _input: &BoxIr, _output: &()) -> Result<bool, PassFailure> {
        Ok(false)
    }
}

/// Fails every run with the error `crashed`.
struct CrashInRun;

impl CompilerPass<BoxIr, ()> for CrashInRun {
    fn run(&mut self, _ir: &BoxIr, _cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        Err("crashed".into())
    }

    fn did_change(&mut self, _input: &BoxIr, _output: &()) -> Result<bool, PassFailure> {
        Ok(false)
    }
}

/// Produce `RejectInput`'s validation failure.
fn produce_validation_failure() -> PassError {
    RejectInput
        .execute(&BoxIr::new(0))
        .expect_err("the input is rejected")
}

/// Produce `CrashInRun`'s execution failure.
fn produce_execution_failure() -> PassError {
    CrashInRun
        .execute(&BoxIr::new(0))
        .expect_err("the run fails")
}

/// Return the level and text of every diagnostic.
fn collect_levels_and_messages(
    diagnostics: &[fhy_core::diagnostic::Diagnostic],
) -> Vec<(DiagnosticLevel, &str)> {
    diagnostics
        .iter()
        .map(|diagnostic| (diagnostic.level(), diagnostic.message_text()))
        .collect()
}

/// Build the validation pipeline holding `validators`, in order.
fn build_manager<'p>(
    validators: impl IntoIterator<Item = ScriptedValidator>,
) -> ValidationManager<'p, BoxIr> {
    let mut manager = ValidationManager::new(Identifier::new("validation"));
    for validator in validators {
        manager.add(validator);
    }
    manager
}

// =============================================================================
// Collect-all validation
// =============================================================================

/// Test every validator runs even after earlier ones reported errors.
#[test]
fn validation_manager_runs_every_validator_after_errors() {
    let mut manager = build_manager([
        ScriptedValidator::reporting(
            "tests.vm.first_error",
            DiagnosticLevel::Error,
            "first-error-msg",
        ),
        ScriptedValidator::reporting(
            "tests.vm.second_error",
            DiagnosticLevel::Error,
            "second-error-msg",
        ),
        ScriptedValidator::clean("tests.vm.third_clean"),
    ]);

    let report = manager.validate(&BoxIr::new(0));

    let names: Vec<_> = report
        .records()
        .iter()
        .map(fhy_core::pass::PassRunRecord::pass_name)
        .collect();
    assert_eq!(
        names,
        [
            "tests.vm.first_error",
            "tests.vm.second_error",
            "tests.vm.third_clean"
        ]
    );
    let errors: Vec<_> = report
        .errors()
        .map(fhy_core::diagnostic::Diagnostic::message_text)
        .collect();
    assert_eq!(errors, ["first-error-msg", "second-error-msg"]);
}

/// Test the report lists diagnostics in pipeline order, attributed to their
/// validators.
#[test]
fn validation_manager_aggregates_diagnostics_in_pipeline_order() {
    let mut manager = build_manager([
        ScriptedValidator::reporting("tests.vm.warn", DiagnosticLevel::Warning, "warn-msg"),
        ScriptedValidator::reporting("tests.vm.err", DiagnosticLevel::Error, "err-msg"),
    ]);

    let report = manager.validate(&BoxIr::new(42));

    let diagnostics: Vec<_> = report
        .diagnostics()
        .iter()
        .map(|d| (d.source(), d.level(), d.message_text()))
        .collect();
    assert_eq!(
        diagnostics,
        [
            ("tests.vm.warn", DiagnosticLevel::Warning, "warn-msg"),
            ("tests.vm.err", DiagnosticLevel::Error, "err-msg"),
        ]
    );
}

/// Test info, warning, and error diagnostics all reach the report and its
/// per-level views.
#[test]
fn validation_manager_aggregates_mixed_severity_levels() {
    let mut manager = build_manager([
        ScriptedValidator::reporting("tests.vm.mixed.info", DiagnosticLevel::Info, "note-me"),
        ScriptedValidator::reporting("tests.vm.mixed.warn", DiagnosticLevel::Warning, "watch-me"),
        ScriptedValidator::reporting("tests.vm.mixed.err", DiagnosticLevel::Error, "fix-me"),
    ]);

    let report = manager.validate(&BoxIr::new(0));

    let infos: Vec<_> = report
        .infos()
        .map(fhy_core::diagnostic::Diagnostic::message_text)
        .collect();
    let warnings: Vec<_> = report
        .warnings()
        .map(fhy_core::diagnostic::Diagnostic::message_text)
        .collect();
    let errors: Vec<_> = report
        .errors()
        .map(fhy_core::diagnostic::Diagnostic::message_text)
        .collect();
    assert_eq!(
        (infos, warnings, errors),
        (vec!["note-me"], vec!["watch-me"], vec!["fix-me"])
    );
}

/// Test a clean pipeline yields a report without diagnostics but with a
/// record per validator.
#[test]
fn validation_manager_returns_a_clean_report_when_every_validator_is_clean() {
    let mut manager = build_manager([
        ScriptedValidator::clean("tests.vm.clean_a"),
        ScriptedValidator::clean("tests.vm.clean_b"),
    ]);

    let report = manager.validate(&BoxIr::new(0));

    assert!(!report.has_errors());
    assert!(report.diagnostics().is_empty());
    let names: Vec<_> = report
        .records()
        .iter()
        .map(fhy_core::pass::PassRunRecord::pass_name)
        .collect();
    assert_eq!(names, ["tests.vm.clean_a", "tests.vm.clean_b"]);
}

/// Test a pipeline without validators yields an empty report.
#[test]
fn validation_manager_without_validators_returns_an_empty_report() {
    let mut manager = build_manager([]);

    let report = manager.validate(&BoxIr::new(0));

    assert!(report.diagnostics().is_empty());
    assert!(report.records().is_empty());
}

/// Test every diagnostic of one validator is kept, in emission order, in the
/// report and in the validator's record.
#[test]
fn validation_manager_keeps_every_diagnostic_of_one_validator_in_order() {
    let mut manager = build_manager([ScriptedValidator::new(
        "tests.vm.multi_emit",
        vec![
            report(DiagnosticLevel::Info, "info-1"),
            report(DiagnosticLevel::Warning, "warn-1"),
            report(DiagnosticLevel::Error, "err-1"),
            report(DiagnosticLevel::Warning, "warn-2"),
        ],
    )]);

    let report = manager.validate(&BoxIr::new(0));

    let expected = [
        (DiagnosticLevel::Info, "info-1"),
        (DiagnosticLevel::Warning, "warn-1"),
        (DiagnosticLevel::Error, "err-1"),
        (DiagnosticLevel::Warning, "warn-2"),
    ];
    assert_eq!(collect_levels_and_messages(report.diagnostics()), expected);
    assert_eq!(report.records().len(), 1);
    assert_eq!(
        collect_levels_and_messages(report.records()[0].diagnostics()),
        expected
    );
}

/// Test each record reports an unchanged run that preserved every analysis.
#[test]
fn validation_manager_records_each_validator_as_unchanged_and_preserving_all() {
    let mut manager = build_manager([ScriptedValidator::reporting(
        "tests.vm.record_warn",
        DiagnosticLevel::Warning,
        "msg",
    )]);

    let report = manager.validate(&BoxIr::new(0));

    let record = &report.records()[0];
    assert!(!record.is_changed());
    assert_eq!(record.preserved_analyses(), &PreservedAnalyses::all());
    assert_eq!(record.diagnostics().len(), 1);
}

/// Test a structured note reaches the report unchanged.
#[test]
fn validation_manager_keeps_a_structured_note() {
    struct NoteValidator;

    impl CompilerPass<BoxIr, ()> for NoteValidator {
        fn run(&mut self, _ir: &BoxIr, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
            let note = Note::new("structured-message", get_suggestion_note_kind().clone());
            cx.report(DiagnosticLevel::Error, note, None);
            Ok(())
        }

        fn did_change(&mut self, _input: &BoxIr, _output: &()) -> Result<bool, PassFailure> {
            Ok(false)
        }
    }
    let mut manager = ValidationManager::new(Identifier::new("validation"));
    manager.add(NoteValidator);

    let report = manager.validate(&BoxIr::new(0));

    assert_eq!(
        report
            .errors()
            .next()
            .map(fhy_core::diagnostic::Diagnostic::message),
        Some(&Note::new(
            "structured-message",
            get_suggestion_note_kind().clone()
        ))
    );
}

// =============================================================================
// Failing validators
// =============================================================================

/// Test a validator whose run fails leaves the error diagnostic that records
/// the failure, and the validators after it still run.
#[test]
fn validation_manager_records_a_failing_validator_and_runs_the_rest() {
    let mut manager = build_manager([
        ScriptedValidator::new("tests.vm.crasher", vec![Step::Fail("internal boom")]),
        ScriptedValidator::reporting(
            "tests.vm.after_crasher",
            DiagnosticLevel::Error,
            "still-runs",
        ),
    ]);

    let report = manager.validate(&BoxIr::new(0));

    let errors: Vec<_> = report
        .errors()
        .map(|d| (d.source(), d.message_text()))
        .collect();
    assert_eq!(
        errors,
        [
            (
                "tests.vm.crasher",
                "Pass \"tests.vm.crasher\" failed run with internal boom"
            ),
            ("tests.vm.after_crasher", "still-runs"),
        ]
    );
    assert_eq!(report.records().len(), 2);
}

/// Test the diagnostics a validator emitted before failing are kept.
#[test]
fn validation_manager_keeps_the_diagnostics_a_validator_emitted_before_failing() {
    let mut manager = build_manager([ScriptedValidator::new(
        "tests.vm.report_then_crash",
        vec![
            report(DiagnosticLevel::Warning, "heads-up"),
            Step::Fail("kaboom"),
        ],
    )]);

    let report = manager.validate(&BoxIr::new(0));

    assert_eq!(
        collect_levels_and_messages(report.diagnostics()),
        [
            (DiagnosticLevel::Warning, "heads-up"),
            (
                DiagnosticLevel::Error,
                "Pass \"tests.vm.report_then_crash\" failed run with kaboom"
            ),
        ]
    );
    assert_eq!(report.records()[0].diagnostics().len(), 2);
}

/// Test a validator that reported an error and then handed over a pass
/// error keeps just its own diagnostics, and the pipeline continues.
#[rstest]
#[case::execution(produce_execution_failure())]
#[case::validation(produce_validation_failure())]
fn validation_manager_adds_nothing_when_a_failing_validator_reported_an_error(
    #[case] failure: PassError,
) {
    let mut manager = build_manager([
        ScriptedValidator::new(
            "tests.vm.reported_then_failed",
            vec![
                report(DiagnosticLevel::Error, "real-problem"),
                Step::HandOver(failure),
            ],
        ),
        ScriptedValidator::reporting(
            "tests.vm.after_reported",
            DiagnosticLevel::Warning,
            "still-here",
        ),
    ]);

    let report = manager.validate(&BoxIr::new(0));

    assert_eq!(
        collect_levels_and_messages(report.diagnostics()),
        [
            (DiagnosticLevel::Error, "real-problem"),
            (DiagnosticLevel::Warning, "still-here"),
        ]
    );
    let names: Vec<_> = report
        .records()
        .iter()
        .map(fhy_core::pass::PassRunRecord::pass_name)
        .collect();
    assert_eq!(
        names,
        ["tests.vm.reported_then_failed", "tests.vm.after_reported"]
    );
}

/// Test a validator that fails without reporting an error gains one naming
/// the failure's class and message.
#[rstest]
#[case::validation(
    produce_validation_failure(),
    "Validator \"tests.vm.silent\" raised \"validation failure\" without reporting a \
     diagnostic: Pass \"RejectInput\" failed validate_input with rejected"
)]
#[case::execution(
    produce_execution_failure(),
    "Validator \"tests.vm.silent\" raised \"execution failure\" without reporting a \
     diagnostic: Pass \"CrashInRun\" failed run with crashed"
)]
fn validation_manager_adds_an_error_for_a_validator_that_fails_silently(
    #[case] failure: PassError,
    #[case] expected: &str,
) {
    let mut manager = build_manager([ScriptedValidator::new(
        "tests.vm.silent",
        vec![
            report(DiagnosticLevel::Warning, "only-a-warning"),
            Step::HandOver(failure),
        ],
    )]);

    let report = manager.validate(&BoxIr::new(0));

    let diagnostics: Vec<_> = report
        .diagnostics()
        .iter()
        .map(|d| (d.level(), d.source(), d.message_text(), d.detail()))
        .collect();
    assert_eq!(
        diagnostics,
        [
            (
                DiagnosticLevel::Warning,
                "tests.vm.silent",
                "only-a-warning",
                None
            ),
            (DiagnosticLevel::Error, "tests.vm.silent", expected, None),
        ]
    );
    assert_eq!(report.records()[0].diagnostics().len(), 2);
}

/// Test a validator whose own input check fails records that failure.
#[test]
fn validation_manager_records_a_validator_that_rejects_its_input() {
    let mut manager = ValidationManager::new(Identifier::new("validation"));
    manager.add(RejectInput);

    let report = manager.validate(&BoxIr::new(0));

    let errors: Vec<_> = report
        .errors()
        .map(fhy_core::diagnostic::Diagnostic::message_text)
        .collect();
    assert_eq!(
        errors,
        ["Pass \"RejectInput\" failed validate_input with rejected"]
    );
}

// =============================================================================
// The pipeline itself
// =============================================================================

/// Test the report of a failing pipeline renders for a human and escalates
/// into an error holding both errors.
#[test]
fn validation_manager_report_formats_and_escalates_failures() {
    let mut manager = build_manager([
        ScriptedValidator::new(
            "tests.vm.e2e.missing_return",
            vec![Step::Report {
                level: DiagnosticLevel::Error,
                message: "function foo() has no return",
                detail: Some("foo @ line 17"),
            }],
        ),
        ScriptedValidator::reporting(
            "tests.vm.e2e.unused",
            DiagnosticLevel::Warning,
            "x is unused",
        ),
        ScriptedValidator::reporting(
            "tests.vm.e2e.shape",
            DiagnosticLevel::Error,
            "shape mismatch",
        ),
    ]);

    let report = manager.validate(&BoxIr::new(0));
    let rendered = report.format();
    let failure = report.into_result().expect_err("the report has errors");

    assert_eq!(
        rendered,
        "[ERROR] tests.vm.e2e.missing_return: function foo() has no return\n    \
         detail: foo @ line 17\n\
         [WARNING] tests.vm.e2e.unused: x is unused\n\
         [ERROR] tests.vm.e2e.shape: shape mismatch"
    );
    assert_eq!(failure.report().errors().count(), 2);
}

/// Test the validator names come out in pipeline order.
#[test]
fn validation_manager_validator_names_lists_validators_in_order() {
    let manager = build_manager([
        ScriptedValidator::clean("tests.vm.add_first"),
        ScriptedValidator::clean("tests.vm.add_second"),
    ]);

    assert_eq!(
        manager.validator_names(),
        ["tests.vm.add_first", "tests.vm.add_second"]
    );
}

/// Test the pipeline's name is its identifier.
#[test]
fn validation_manager_name_is_its_identifier() {
    let name = Identifier::new("tests.vm.named");

    let manager: ValidationManager<'_, BoxIr> = ValidationManager::new(name.clone());

    assert_eq!(manager.name(), &name);
    assert_eq!(manager.identifier(), &name);
}

/// Test a default validation pipeline is named `validation-pipeline` and
/// holds no validators.
#[test]
fn validation_manager_default_is_an_empty_pipeline_named_validation_pipeline() {
    let manager: ValidationManager<'_, BoxIr> = ValidationManager::default();

    assert_eq!(manager.name().name_hint(), "validation-pipeline");
    assert!(manager.validator_names().is_empty());
}

/// Test validators compute analyses afresh on every request.
#[test]
fn validation_manager_runs_validators_without_an_analysis_cache() {
    struct TwiceReading;

    impl CompilerPass<BoxIr, ()> for TwiceReading {
        fn run(&mut self, ir: &BoxIr, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
            cx.analysis::<DoubleAnalysis, _>(ir);
            cx.analysis::<DoubleAnalysis, _>(ir);
            Ok(())
        }

        fn did_change(&mut self, _input: &BoxIr, _output: &()) -> Result<bool, PassFailure> {
            Ok(false)
        }
    }
    let mut manager = ValidationManager::new(Identifier::new("validation"));
    manager.add(TwiceReading);
    let ir = BoxIr::new(3);

    let _report = manager.validate(&ir);

    assert_eq!(ir.double_runs(), 2);
}

/// Test each validation starts afresh: a borrowed validator runs on every
/// call and reports from earlier calls do not accumulate.
#[test]
fn validation_manager_validates_afresh_on_every_call() {
    struct CountingWarner {
        runs: usize,
    }

    impl CompilerPass<BoxIr, ()> for CountingWarner {
        fn run(&mut self, _ir: &BoxIr, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
            self.runs += 1;
            cx.report_text(DiagnosticLevel::Warning, "again", None);
            Ok(())
        }

        fn did_change(&mut self, _input: &BoxIr, _output: &()) -> Result<bool, PassFailure> {
            Ok(false)
        }
    }
    let mut warner = CountingWarner { runs: 0 };
    let mut manager = ValidationManager::new(Identifier::new("validation"));
    manager.add(&mut warner);

    let first = manager.validate(&BoxIr::new(0));
    let second = manager.validate(&BoxIr::new(0));
    drop(manager);

    assert_eq!(first.diagnostics().len(), 1);
    assert_eq!(second.diagnostics().len(), 1);
    assert_eq!(warner.runs, 2);
}
