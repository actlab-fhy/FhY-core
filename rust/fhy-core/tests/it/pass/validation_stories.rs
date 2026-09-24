//! Tests for `fhy_core::pass::ValidationManager`: collect-all validation,
//! failing validators, passes run as validators, and the aggregated report
//! and its per-validator records.
//!
//! Nothing here reads process-global state.

use crate::support::pass_ir;

use std::borrow::Cow;

use fhy_core::diagnostic::{Diagnostic, DiagnosticLevel, Note, NoteKind, ValidationReport};
use fhy_core::identifier::{HasIdentifier, Identifier};
use fhy_core::pass::{
    CompilerPass, ExecutePass, PassContext, PassError, PassFailure, PassValidator,
    ValidationManager, Validator, ValidatorRecord,
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

/// A validator named explicitly that performs a script.
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

impl Validator<BoxIr> for ScriptedValidator {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed(self.name)
    }

    fn validate(&mut self, _ir: &BoxIr, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
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
}

/// The script as a pass, so a [`PassValidator`] can run it: the run performs
/// the script.
impl CompilerPass<BoxIr, ()> for ScriptedValidator {
    fn name(&self) -> Cow<'static, str> {
        Cow::Borrowed(self.name)
    }

    fn run(&mut self, ir: &BoxIr, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        self.validate(ir, cx)
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
fn collect_levels_and_messages(diagnostics: &[Diagnostic]) -> Vec<(DiagnosticLevel, &str)> {
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

/// Return the validator name of every record of `report`.
fn collect_validator_names(report: &ValidationReport<ValidatorRecord>) -> Vec<&str> {
    report
        .records()
        .iter()
        .map(ValidatorRecord::validator_name)
        .collect()
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

    assert_eq!(
        collect_validator_names(&report),
        [
            "tests.vm.first_error",
            "tests.vm.second_error",
            "tests.vm.third_clean"
        ]
    );
    let errors: Vec<_> = report.errors().map(Diagnostic::message_text).collect();
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

    let infos: Vec<_> = report.infos().map(Diagnostic::message_text).collect();
    let warnings: Vec<_> = report.warnings().map(Diagnostic::message_text).collect();
    let errors: Vec<_> = report.errors().map(Diagnostic::message_text).collect();
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
    assert_eq!(
        collect_validator_names(&report),
        ["tests.vm.clean_a", "tests.vm.clean_b"]
    );
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
        collect_levels_and_messages(report.records()[0].diagnostics_in(&report)),
        expected
    );
}

/// Test each record names its validator, says it did not fail, and holds
/// its diagnostics.
#[test]
fn validation_manager_records_each_validator_with_its_diagnostics() {
    let mut manager = build_manager([
        ScriptedValidator::reporting("tests.vm.record_warn", DiagnosticLevel::Warning, "msg"),
        ScriptedValidator::clean("tests.vm.record_clean"),
    ]);

    let report = manager.validate(&BoxIr::new(0));

    let records = report.records();
    assert_eq!(records[0].validator_name(), "tests.vm.record_warn");
    assert!(!records[0].is_failed());
    assert_eq!(
        collect_levels_and_messages(records[0].diagnostics_in(&report)),
        [(DiagnosticLevel::Warning, "msg")]
    );
    assert!(!records[1].is_failed());
    assert!(records[1].diagnostics_in(&report).is_empty());
}

/// Test the report holds every diagnostic once, and the records' slices of
/// it partition it in order.
#[test]
fn validation_report_stores_each_diagnostic_once() {
    let mut manager = build_manager([
        ScriptedValidator::new(
            "tests.vm.two",
            vec![
                report(DiagnosticLevel::Info, "one"),
                report(DiagnosticLevel::Error, "two"),
            ],
        ),
        ScriptedValidator::clean("tests.vm.none"),
        ScriptedValidator::new("tests.vm.failing", vec![Step::Fail("boom")]),
        ScriptedValidator::reporting("tests.vm.last", DiagnosticLevel::Warning, "three"),
    ]);

    let report = manager.validate(&BoxIr::new(0));

    let concatenated: Vec<&Diagnostic> = report
        .records()
        .iter()
        .flat_map(|record| record.diagnostics_in(&report))
        .collect();
    let all: Vec<&Diagnostic> = report.diagnostics().iter().collect();
    assert_eq!(concatenated.len(), all.len());
    assert!(
        concatenated
            .iter()
            .zip(&all)
            .all(|(left, right)| std::ptr::eq(*left, *right))
    );
    let sizes: Vec<_> = report
        .records()
        .iter()
        .map(|record| record.diagnostics_in(&report).len())
        .collect();
    assert_eq!(sizes, [2, 0, 1, 1]);
}

/// Test a structured note reaches the report unchanged.
#[test]
fn validation_manager_keeps_a_structured_note() {
    struct NoteValidator;

    impl Validator<BoxIr> for NoteValidator {
        fn validate(&mut self, _ir: &BoxIr, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
            let note = Note::new("structured-message", NoteKind::suggestion().clone());
            cx.report(Diagnostic::error(note, "NoteValidator"));
            Ok(())
        }
    }
    let mut manager = ValidationManager::new(Identifier::new("validation"));
    manager.add(NoteValidator);

    let report = manager.validate(&BoxIr::new(0));

    assert_eq!(
        report.errors().next().map(Diagnostic::message),
        Some(&Note::new(
            "structured-message",
            NoteKind::suggestion().clone()
        ))
    );
}

// =============================================================================
// Failing validators
// =============================================================================

/// Test a pass run as a validator whose run fails leaves the error
/// diagnostic that records the failure, is recorded as failed, and the
/// validators after it still run.
#[test]
fn validation_manager_records_a_failing_validator_and_runs_the_rest() {
    let mut manager = ValidationManager::new(Identifier::new("validation"));
    manager.add(PassValidator::new(ScriptedValidator::new(
        "tests.vm.crasher",
        vec![Step::Fail("internal boom")],
    )));
    manager.add(ScriptedValidator::reporting(
        "tests.vm.after_crasher",
        DiagnosticLevel::Error,
        "still-runs",
    ));

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
                "pass \"tests.vm.crasher\" failed in run: internal boom"
            ),
            ("tests.vm.after_crasher", "still-runs"),
        ]
    );
    let failed: Vec<_> = report
        .records()
        .iter()
        .map(ValidatorRecord::is_failed)
        .collect();
    assert_eq!(failed, [true, false]);
}

/// Test the diagnostics a pass run as a validator emitted before failing
/// are kept, followed by the error recording the failure.
#[test]
fn validation_manager_keeps_the_diagnostics_a_validator_emitted_before_failing() {
    let mut manager = ValidationManager::new(Identifier::new("validation"));
    manager.add(PassValidator::new(ScriptedValidator::new(
        "tests.vm.report_then_crash",
        vec![
            report(DiagnosticLevel::Warning, "heads-up"),
            Step::Fail("kaboom"),
        ],
    )));

    let report = manager.validate(&BoxIr::new(0));

    assert_eq!(
        collect_levels_and_messages(report.diagnostics()),
        [
            (DiagnosticLevel::Warning, "heads-up"),
            (
                DiagnosticLevel::Error,
                "pass \"tests.vm.report_then_crash\" failed in run: kaboom"
            ),
        ]
    );
    assert_eq!(report.records()[0].diagnostics_in(&report).len(), 2);
    assert!(report.records()[0].is_failed());
}

/// Test a validator that reported an error and then failed with a pass
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
    assert_eq!(
        collect_validator_names(&report),
        ["tests.vm.reported_then_failed", "tests.vm.after_reported"]
    );
}

/// Test a validator that fails without reporting an error gains one naming
/// the validator and the failure's cause chain.
#[test]
fn validation_manager_adds_an_error_for_a_validator_that_fails_silently() {
    let mut manager = build_manager([ScriptedValidator::new(
        "tests.vm.silent",
        vec![
            report(DiagnosticLevel::Warning, "only-a-warning"),
            Step::Fail("internal boom"),
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
            (
                DiagnosticLevel::Error,
                "tests.vm.silent",
                "validator \"tests.vm.silent\" failed without reporting an error: internal boom",
                None
            ),
        ]
    );
    assert_eq!(report.records()[0].diagnostics_in(&report).len(), 2);
    assert!(report.records()[0].is_failed());
}

/// Test the silent-failure error walks the failure's cause chain, writing
/// each cause once.
#[test]
fn validation_manager_writes_the_cause_chain_of_a_silent_failure() {
    let mut manager = build_manager([ScriptedValidator::new(
        "tests.vm.silent_chain",
        vec![Step::HandOver(produce_execution_failure())],
    )]);

    let report = manager.validate(&BoxIr::new(0));

    let messages: Vec<_> = report
        .diagnostics()
        .iter()
        .map(Diagnostic::message_text)
        .collect();
    assert_eq!(
        messages,
        [
            "validator \"tests.vm.silent_chain\" failed without reporting an error: \
             pass \"CrashInRun\" failed in run: crashed"
        ]
    );
}

/// Test a pass run as a validator whose input check fails records that
/// failure.
#[test]
fn validation_manager_records_a_validator_that_rejects_its_input() {
    let mut manager = ValidationManager::new(Identifier::new("validation"));
    manager.add(PassValidator::new(RejectInput));

    let report = manager.validate(&BoxIr::new(0));

    let errors: Vec<_> = report.errors().map(Diagnostic::message_text).collect();
    assert_eq!(
        errors,
        ["pass \"RejectInput\" failed in validate_input: rejected"]
    );
    assert!(report.records()[0].is_failed());
}

/// Checks a value through the pass hooks a validation runs, recording them.
#[derive(Default)]
struct HookRecordingCheck {
    calls: Vec<&'static str>,
    skips: bool,
}

impl CompilerPass<BoxIr, ()> for HookRecordingCheck {
    fn validate_input(
        &mut self,
        _ir: &BoxIr,
        _cx: &mut PassContext<'_>,
    ) -> Result<(), PassFailure> {
        self.calls.push("validate_input");
        Ok(())
    }

    fn skip(&mut self, _ir: &BoxIr, _cx: &mut PassContext<'_>) -> Result<Option<()>, PassFailure> {
        self.calls.push("skip");
        Ok(self.skips.then_some(()))
    }

    fn run(&mut self, ir: &BoxIr, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        self.calls.push("run");
        if ir.value() < 0 {
            cx.report_text(DiagnosticLevel::Error, "negative", None);
        }
        Ok(())
    }

    fn validate_output(
        &mut self,
        _input: &BoxIr,
        _output: &(),
        _cx: &mut PassContext<'_>,
    ) -> Result<(), PassFailure> {
        self.calls.push("validate_output");
        Ok(())
    }

    fn did_change(&mut self, _input: &BoxIr, _output: &()) -> Result<bool, PassFailure> {
        self.calls.push("did_change");
        Ok(false)
    }
}

/// Test a pass runs as a validator: named after the pass, its hooks up to
/// `validate_output` run, and its diagnostics reach the report.
#[test]
fn pass_validator_runs_a_pass_as_a_validator() {
    let mut check = HookRecordingCheck::default();
    let mut manager = ValidationManager::new(Identifier::new("validation"));
    manager.add(PassValidator::new(&mut check));

    let names: Vec<_> = manager.validator_names().collect();
    let report = manager.validate(&BoxIr::new(-1));
    drop(manager);

    assert_eq!(names, ["HookRecordingCheck"]);
    assert_eq!(
        check.calls,
        ["validate_input", "skip", "run", "validate_output"]
    );
    assert_eq!(
        collect_levels_and_messages(report.diagnostics()),
        [(DiagnosticLevel::Error, "negative")]
    );
    assert_eq!(report.records()[0].validator_name(), "HookRecordingCheck");
    assert!(!report.records()[0].is_failed());
}

/// Test a pass run as a validator that skips ends the check there.
#[test]
fn pass_validator_ends_the_check_at_a_skip() {
    let mut validator = PassValidator::new(HookRecordingCheck {
        skips: true,
        ..HookRecordingCheck::default()
    });
    let mut manager = ValidationManager::new(Identifier::new("validation"));
    manager.add(&mut validator);

    let report = manager.validate(&BoxIr::new(-1));
    drop(manager);

    assert!(report.diagnostics().is_empty());
    assert_eq!(validator.pass().calls, ["validate_input", "skip"]);
    assert_eq!(validator.pass_mut().calls.len(), 2);
    assert_eq!(validator.into_pass().calls.len(), 2);
}

/// Panics in its check.
struct PanickingValidator;

impl Validator<BoxIr> for PanickingValidator {
    fn validate(&mut self, _ir: &BoxIr, _cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
        panic!("the validator panics");
    }
}

/// Test a panic in a validator propagates: validation does not catch it.
#[test]
#[should_panic(expected = "the validator panics")]
fn validation_manager_propagates_a_validator_panic() {
    let mut manager = ValidationManager::new(Identifier::new("validation"));
    manager.add(PanickingValidator);

    let _report = manager.validate(&BoxIr::new(0));
}

/// Test a validator reached through a borrow or a box keeps its name and
/// its check.
#[test]
fn borrowed_and_boxed_validators_forward_their_name_and_check() {
    let mut scripted =
        ScriptedValidator::reporting("tests.vm.borrowed", DiagnosticLevel::Info, "i");
    let boxed: Box<dyn Validator<BoxIr> + Send> =
        Box::new(ScriptedValidator::clean("tests.vm.boxed"));
    let mut manager = ValidationManager::new(Identifier::new("validation"));
    manager.add(&mut scripted);
    manager.add(boxed);

    let report = manager.validate(&BoxIr::new(0));

    assert_eq!(
        collect_validator_names(&report),
        ["tests.vm.borrowed", "tests.vm.boxed"]
    );
    assert_eq!(report.diagnostics().len(), 1);
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
    let rendered = report.to_string();
    let failure = report.into_result().expect_err("the report has errors");

    assert_eq!(
        rendered,
        "error[tests.vm.e2e.missing_return]: function foo() has no return\n    \
         detail: foo @ line 17\n\
         warning[tests.vm.e2e.unused]: x is unused\n\
         error[tests.vm.e2e.shape]: shape mismatch"
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
        manager.validator_names().collect::<Vec<_>>(),
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
    assert_eq!(manager.validator_names().count(), 0);
}

/// Test validators compute analyses afresh on every request.
#[test]
fn validation_manager_runs_validators_without_an_analysis_cache() {
    struct TwiceReading;

    impl Validator<BoxIr> for TwiceReading {
        fn validate(&mut self, ir: &BoxIr, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
            cx.analysis::<DoubleAnalysis>(ir);
            cx.analysis::<DoubleAnalysis>(ir);
            Ok(())
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

    impl Validator<BoxIr> for CountingWarner {
        fn validate(&mut self, _ir: &BoxIr, cx: &mut PassContext<'_>) -> Result<(), PassFailure> {
            self.runs += 1;
            cx.report_text(DiagnosticLevel::Warning, "again", None);
            Ok(())
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
