//! Tests for `fhy_core::diagnostic`: note kinds, notes, diagnostic levels,
//! diagnostics, validation reports and the error a failed report escalates
//! into.
//!
//! Public API only. These tests share the process-wide `NoteKind` registry
//! and run in parallel, so none of them clears it, and each test that
//! registers a kind does so under a fresh identifier.

#[path = "common/hashing.rs"]
pub mod hashing_support;

use std::error::Error;

use fhy_core::diagnostic::{
    Diagnostic, DiagnosticLevel, Note, NoteKind, ValidationFailedError, ValidationReport,
    get_other_note_kind, get_rationale_note_kind, get_remark_note_kind, get_suggestion_note_kind,
};
use fhy_core::identifier::{HasIdentifier, Identifier};
use fhy_core::interned::{Canonical, InternOutcome, Interned};
use hashing_support::hash_of;
use rstest::rstest;
use serde_json::{Value, json};

// =============================================================================
// Helpers
// =============================================================================

/// A function returning one of the shipped note kinds.
type DefaultKind = fn() -> &'static Canonical<NoteKind>;

/// Build a diagnostic at `level` from `source` with an uncategorized note.
fn build_diagnostic(level: DiagnosticLevel, message: &str, source: &str) -> Diagnostic {
    Diagnostic::new(level, Note::with_other_kind(message), source, None)
}

/// Build a diagnostic at `level` from `source` carrying `detail`.
fn build_detailed_diagnostic(
    level: DiagnosticLevel,
    message: &str,
    source: &str,
    detail: &str,
) -> Diagnostic {
    Diagnostic::new(
        level,
        Note::with_other_kind(message),
        source,
        Some(detail.to_owned()),
    )
}

/// Build a report holding `diagnostics` and no records.
fn build_report(diagnostics: Vec<Diagnostic>) -> ValidationReport {
    ValidationReport::new(diagnostics, Vec::new())
}

/// Return the sources of `diagnostics`, in order.
fn collect_sources<'a>(diagnostics: impl Iterator<Item = &'a Diagnostic>) -> Vec<&'a str> {
    diagnostics.map(Diagnostic::source).collect()
}

/// Return the JSON payload of a note whose kind is `kind`.
fn encode_note_payload(message: &str, kind: &NoteKind) -> Value {
    json!({
        "message": message,
        "kind": {
            "name": {"id": kind.name().id(), "name_hint": kind.name().name_hint()},
            "description": kind.description(),
        },
    })
}

// =============================================================================
// NoteKind
// =============================================================================

/// Test each shipped kind keeps its documented name hint.
#[rstest]
#[case::rationale(get_rationale_note_kind, "rationale")]
#[case::suggestion(get_suggestion_note_kind, "suggestion")]
#[case::remark(get_remark_note_kind, "remark")]
#[case::other(get_other_note_kind, "other")]
fn shipped_note_kind_has_its_documented_name(
    #[case] get_kind: DefaultKind,
    #[case] name_hint: &str,
) {
    let kind = get_kind();

    assert_eq!(kind.name().name_hint(), name_hint);
    assert_eq!(kind.to_string(), name_hint);
}

/// Test each shipped kind is the canonical entry registered for its name.
#[rstest]
#[case::rationale(get_rationale_note_kind)]
#[case::suggestion(get_suggestion_note_kind)]
#[case::remark(get_remark_note_kind)]
#[case::other(get_other_note_kind)]
fn shipped_note_kind_is_registered_under_its_name(#[case] get_kind: DefaultKind) {
    let kind = get_kind();

    assert_eq!(
        NoteKind::intern_registry().get(kind.name()),
        Some(kind.clone())
    );
    assert!(!kind.description().trim().is_empty());
}

/// Test the four shipped kinds are pairwise distinct.
#[rstest]
#[case::rationale_suggestion(get_rationale_note_kind, get_suggestion_note_kind)]
#[case::rationale_remark(get_rationale_note_kind, get_remark_note_kind)]
#[case::rationale_other(get_rationale_note_kind, get_other_note_kind)]
#[case::suggestion_remark(get_suggestion_note_kind, get_remark_note_kind)]
#[case::suggestion_other(get_suggestion_note_kind, get_other_note_kind)]
#[case::remark_other(get_remark_note_kind, get_other_note_kind)]
fn shipped_note_kinds_are_distinct(#[case] get_kind: DefaultKind, #[case] get_other: DefaultKind) {
    let kind = get_kind();
    let other = get_other();

    assert_ne!(kind, other);
    assert_ne!(kind.name(), other.name());
}

/// Test a caller registers a new kind without changing this crate.
#[test]
fn note_kind_new_registers_a_new_kind() {
    let name = Identifier::new("performance");

    let outcome = NoteKind::new(name.clone(), "An optimization remark.");

    assert!(outcome.is_registered());
    let kind = outcome.into_canonical();
    assert_eq!(kind.name(), &name);
    assert_eq!(kind.identifier(), &name);
    assert_eq!(kind.intern_key(), &name);
    assert_eq!(kind.description(), "An optimization remark.");
    assert_eq!(NoteKind::intern_registry().get(&name), Some(kind));
}

/// Test a second kind under a taken name loses the registration, compares
/// equal to the canonical kind whatever its description, and hashes equally.
#[test]
fn note_kind_equality_ignores_description() {
    let name = Identifier::new("shared");
    let first = NoteKind::new(name.clone(), "first description").into_canonical();

    let outcome = NoteKind::new(name, "second description");

    let InternOutcome::AlreadyCanonical {
        canonical,
        discarded,
    } = outcome
    else {
        panic!("expected the second kind to lose the registration");
    };
    assert_eq!(canonical, first);
    assert_eq!(canonical.description(), "first description");
    assert_eq!(discarded.description(), "second description");
    assert_eq!(discarded, *first);
    assert_eq!(hash_of(&discarded), hash_of(&*first));
}

/// Test kinds under distinct identifiers that share a name hint are
/// distinct.
#[test]
fn note_kinds_sharing_a_name_hint_are_distinct() {
    let first = NoteKind::new(Identifier::new("twin"), "a").into_canonical();
    let second = NoteKind::new(Identifier::new("twin"), "b").into_canonical();

    assert_ne!(first, second);
    assert_ne!(*first, *second);
}

/// Test a kind encodes as its name and description.
#[test]
fn note_kind_encodes_as_name_and_description() {
    let name = Identifier::new("encoded-kind");
    let kind = NoteKind::new(name.clone(), "a description").into_canonical();

    let encoded = serde_json::to_value(&*kind).expect("kinds encode");

    assert_eq!(
        encoded,
        json!({
            "name": {"id": name.id(), "name_hint": "encoded-kind"},
            "description": "a description",
        })
    );
}

/// Test decoding a shipped kind's payload yields the shipped handle.
#[test]
fn note_kind_decode_returns_the_shipped_handle() {
    let json = serde_json::to_string(get_remark_note_kind()).expect("kinds encode");

    let restored: Canonical<NoteKind> = serde_json::from_str(&json).expect("valid payload");

    assert_eq!(&restored, get_remark_note_kind());
}

// =============================================================================
// Note
// =============================================================================

/// Test a note built without a kind carries the uncategorized kind.
#[test]
fn note_with_other_kind_uses_the_other_note_kind() {
    let note = Note::with_other_kind("lowered from ast");

    assert_eq!(note.message(), "lowered from ast");
    assert_eq!(note.kind(), get_other_note_kind());
}

/// Test a note carries the kind it was built with.
#[test]
fn note_new_carries_the_explicit_kind() {
    let note = Note::new(
        "tiled the loop for cache locality",
        get_rationale_note_kind().clone(),
    );

    assert_eq!(note.message(), "tiled the loop for cache locality");
    assert_eq!(note.kind(), get_rationale_note_kind());
}

/// Test a note renders as `kind: message`.
#[rstest]
#[case::other(get_other_note_kind, "lowered from ast", "other: lowered from ast")]
#[case::suggestion(
    get_suggestion_note_kind,
    "use a smaller tile",
    "suggestion: use a smaller tile"
)]
#[case::empty_message(get_remark_note_kind, "", "remark: ")]
#[case::multi_line_message(get_other_note_kind, "two\nlines", "other: two\nlines")]
fn note_display_renders_kind_and_message(
    #[case] get_kind: DefaultKind,
    #[case] message: &str,
    #[case] expected: &str,
) {
    let note = Note::new(message, get_kind().clone());

    assert_eq!(note.to_string(), expected);
}

/// Test notes are equal exactly when message and kind both match.
#[test]
fn note_equality_compares_message_and_kind() {
    let note = Note::with_other_kind("hello");

    assert_eq!(note, Note::new("hello", get_other_note_kind().clone()));
    assert_eq!(hash_of(&note), hash_of(&Note::with_other_kind("hello")));
    assert_ne!(note, Note::with_other_kind("goodbye"));
    assert_ne!(note, Note::new("hello", get_remark_note_kind().clone()));
}

/// Test a note encodes as its message and its kind's payload.
#[test]
fn note_encodes_as_message_and_kind() {
    let note = Note::with_other_kind("hi");

    let encoded = serde_json::to_value(&note).expect("notes encode");

    assert_eq!(encoded, encode_note_payload("hi", get_other_note_kind()));
}

/// Test a note of each shipped kind round-trips and decodes to the shipped
/// handle.
#[rstest]
#[case::rationale(get_rationale_note_kind)]
#[case::suggestion(get_suggestion_note_kind)]
#[case::remark(get_remark_note_kind)]
#[case::other(get_other_note_kind)]
fn note_round_trips_each_shipped_kind(#[case] get_kind: DefaultKind) {
    let note = Note::new("a message", get_kind().clone());
    let json = serde_json::to_string(&note).expect("notes encode");

    let restored: Note = serde_json::from_str(&json).expect("encoded notes decode");

    assert_eq!(restored, note);
    assert_eq!(restored.kind(), get_kind());
}

/// Test a note tagged with a caller's kind round-trips.
#[test]
fn note_with_custom_kind_round_trips() {
    let custom = NoteKind::new(Identifier::new("deprecation"), "A deprecated-usage remark.")
        .into_canonical();
    let note = Note::new("uses a deprecated builtin", custom.clone());
    let json = serde_json::to_string(&note).expect("notes encode");

    let restored: Note = serde_json::from_str(&json).expect("encoded notes decode");

    assert_eq!(restored, note);
    assert_eq!(restored.kind(), &custom);
}

/// Test decoding a note whose kind description differs keeps the canonical
/// kind and its description.
#[test]
fn note_decode_keeps_the_canonical_kind_description() {
    let mut payload = encode_note_payload("m", get_other_note_kind());
    payload["kind"]["description"] = json!("changed");

    let restored: Note = serde_json::from_value(payload).expect("a divergent description decodes");

    assert_eq!(restored.kind(), get_other_note_kind());
    assert_eq!(
        restored.kind().description(),
        get_other_note_kind().description()
    );
}

/// Test decoding a note whose kind has an unregistered name registers that
/// kind with the payload's description.
#[test]
fn note_decode_registers_an_unknown_kind() {
    let name = Identifier::new("decoded-note-kind");
    let payload = json!({
        "message": "new kind",
        "kind": {"name": {"id": name.id(), "name_hint": "decoded-note-kind"}, "description": "from decode"},
    });

    let restored: Note = serde_json::from_value(payload).expect("valid payload");

    assert_eq!(restored.kind().description(), "from decode");
    assert_eq!(
        NoteKind::intern_registry().get(&name).as_ref(),
        Some(restored.kind())
    );
}

/// Test malformed note payloads are rejected.
#[rstest]
#[case::missing_kind(json!({"message": "x"}))]
#[case::extra_key(json!({"message": "x", "kind": {"name": {"id": 0, "name_hint": "other"}, "description": ""}, "extra": 1}))]
#[case::message_not_a_string(json!({"message": 5, "kind": {"name": {"id": 0, "name_hint": "other"}, "description": ""}}))]
#[case::kind_missing_description(json!({"message": "x", "kind": {"name": {"id": 0, "name_hint": "other"}}}))]
#[case::negative_id(json!({"message": "x", "kind": {"name": {"id": -1, "name_hint": "other"}, "description": ""}}))]
#[case::null_kind(json!({"message": "x", "kind": null}))]
fn note_decode_rejects_malformed_payloads(#[case] payload: Value) {
    let rendered = payload.to_string();

    let error = serde_json::from_value::<Note>(payload)
        .expect_err(&format!("{rendered} is malformed and must be rejected"));

    assert!(
        !error.to_string().is_empty(),
        "the error for {rendered} has a message"
    );
}

// =============================================================================
// DiagnosticLevel and Diagnostic
// =============================================================================

/// Test each level's name and rendering.
#[rstest]
#[case::error(DiagnosticLevel::Error, "error")]
#[case::warning(DiagnosticLevel::Warning, "warning")]
#[case::info(DiagnosticLevel::Info, "info")]
fn diagnostic_level_as_str_is_the_lowercase_name(
    #[case] level: DiagnosticLevel,
    #[case] expected: &str,
) {
    assert_eq!(level.as_str(), expected);
    assert_eq!(level.to_string(), expected);
}

/// Test a diagnostic stores every field it was built with.
#[test]
fn diagnostic_new_stores_every_field() {
    let note = Note::new("missing return", get_rationale_note_kind().clone());

    let diagnostic = Diagnostic::new(
        DiagnosticLevel::Warning,
        note.clone(),
        "shape.check",
        Some("function foo()".to_owned()),
    );

    assert_eq!(diagnostic.level(), DiagnosticLevel::Warning);
    assert_eq!(diagnostic.message(), &note);
    assert_eq!(diagnostic.source(), "shape.check");
    assert_eq!(diagnostic.detail(), Some("function foo()"));
}

/// Test the message text is the note's message without its kind.
#[test]
fn diagnostic_message_text_omits_the_note_kind() {
    let diagnostic = Diagnostic::new(
        DiagnosticLevel::Info,
        Note::new("tiled", get_rationale_note_kind().clone()),
        "tiler",
        None,
    );

    assert_eq!(diagnostic.message_text(), "tiled");
    assert_eq!(diagnostic.detail(), None);
}

/// Test diagnostics compare by value and equal ones hash equally.
#[test]
fn diagnostic_equality_is_by_value() {
    let diagnostic = build_diagnostic(DiagnosticLevel::Error, "bad", "v1");

    assert_eq!(
        diagnostic,
        build_diagnostic(DiagnosticLevel::Error, "bad", "v1")
    );
    assert_eq!(
        hash_of(&diagnostic),
        hash_of(&build_diagnostic(DiagnosticLevel::Error, "bad", "v1"))
    );
    assert_ne!(
        diagnostic,
        build_diagnostic(DiagnosticLevel::Warning, "bad", "v1")
    );
    assert_ne!(
        diagnostic,
        build_detailed_diagnostic(DiagnosticLevel::Error, "bad", "v1", "")
    );
}

// =============================================================================
// ValidationReport
// =============================================================================

/// Test an empty report has no errors, no diagnostics of any level, and
/// renders the placeholder text.
#[test]
fn empty_report_has_no_errors_and_formats_placeholder_text() {
    let report = build_report(Vec::new());

    assert!(report.diagnostics().is_empty());
    assert!(report.records().is_empty());
    assert!(!report.has_errors());
    assert_eq!(report.errors().count(), 0);
    assert_eq!(report.warnings().count(), 0);
    assert_eq!(report.infos().count(), 0);
    assert_eq!(report.format(), "No validation diagnostics.");
}

/// Test the level filters partition the diagnostics in emission order.
#[test]
fn report_filters_diagnostics_by_level() {
    let report = build_report(vec![
        build_diagnostic(DiagnosticLevel::Error, "bad", "v1"),
        build_diagnostic(DiagnosticLevel::Warning, "meh", "v2"),
        build_diagnostic(DiagnosticLevel::Info, "fyi", "v3"),
        build_diagnostic(DiagnosticLevel::Error, "worse", "v4"),
    ]);

    assert!(report.has_errors());
    assert_eq!(collect_sources(report.errors()), ["v1", "v4"]);
    assert_eq!(collect_sources(report.warnings()), ["v2"]);
    assert_eq!(collect_sources(report.infos()), ["v3"]);
}

/// Test a report keeps its diagnostics and records in the order given.
#[test]
fn report_keeps_diagnostics_and_records_in_order() {
    let diagnostics = vec![
        build_diagnostic(DiagnosticLevel::Info, "first", "a"),
        build_diagnostic(DiagnosticLevel::Info, "second", "b"),
    ];

    let report = ValidationReport::new(diagnostics.clone(), vec!["pass-a", "pass-b", "pass-c"]);

    assert_eq!(report.diagnostics(), diagnostics.as_slice());
    assert_eq!(report.records(), &["pass-a", "pass-b", "pass-c"]);
}

/// Test the full rendering: level, source and message on one line, an
/// indented detail line, and no trailing newline.
#[test]
fn report_format_renders_level_source_message_and_detail() {
    let report = build_report(vec![
        build_detailed_diagnostic(
            DiagnosticLevel::Error,
            "missing return",
            "shape.check",
            "function foo() has no return statement",
        ),
        build_diagnostic(DiagnosticLevel::Warning, "unused", "scope.check"),
        build_diagnostic(DiagnosticLevel::Info, "fyi", "v3"),
    ]);

    let rendered = report.format();

    assert_eq!(
        rendered,
        "[ERROR] shape.check: missing return\n\
         \x20   detail: function foo() has no return statement\n\
         [WARNING] scope.check: unused\n\
         [INFO] v3: fyi"
    );
}

/// Test an empty detail is omitted, like an absent one.
#[test]
fn report_format_omits_an_empty_detail() {
    let report = build_report(vec![build_detailed_diagnostic(
        DiagnosticLevel::Error,
        "m",
        "s",
        "",
    )]);

    assert_eq!(report.format(), "[ERROR] s: m");
}

/// Test a detail of only whitespace is not empty and is rendered.
#[test]
fn report_format_keeps_a_whitespace_detail() {
    let report = build_report(vec![build_detailed_diagnostic(
        DiagnosticLevel::Warning,
        "",
        "",
        " ",
    )]);

    assert_eq!(report.format(), "[WARNING] : \n    detail:  ");
}

/// Test newlines inside messages and details are emitted as they are.
#[test]
fn report_format_keeps_embedded_newlines() {
    let report = build_report(vec![
        build_diagnostic(DiagnosticLevel::Error, "m", "s"),
        build_detailed_diagnostic(DiagnosticLevel::Info, "i\nj", "t", "x\ny"),
    ]);

    assert_eq!(
        report.format(),
        "[ERROR] s: m\n[INFO] t: i\nj\n    detail: x\ny"
    );
}

/// Test the note kind never appears in the rendering.
#[test]
fn report_format_omits_the_note_kind() {
    let report = build_report(vec![Diagnostic::new(
        DiagnosticLevel::Info,
        Note::new("kinds are hidden", get_rationale_note_kind().clone()),
        "s",
        None,
    )]);

    assert_eq!(report.format(), "[INFO] s: kinds are hidden");
}

/// Test braces in the text are rendered literally.
#[test]
fn report_format_renders_braces_literally() {
    let report = build_report(vec![build_detailed_diagnostic(
        DiagnosticLevel::Error,
        "dict {x} missing",
        "s",
        "{y}",
    )]);

    assert_eq!(
        report.format(),
        "[ERROR] s: dict {x} missing\n    detail: {y}"
    );
}

/// Test a report without errors passes through `into_result` unchanged.
#[rstest]
#[case::empty(vec![])]
#[case::only_warnings(vec![build_diagnostic(DiagnosticLevel::Warning, "ok-ish", "v")])]
#[case::only_infos(vec![build_diagnostic(DiagnosticLevel::Info, "fyi", "v")])]
#[case::warnings_and_infos(vec![
    build_diagnostic(DiagnosticLevel::Warning, "w", "a"),
    build_diagnostic(DiagnosticLevel::Info, "i", "b"),
])]
fn report_into_result_returns_a_report_without_errors(#[case] diagnostics: Vec<Diagnostic>) {
    let report = build_report(diagnostics);
    let expected = report.clone();

    let result = report.into_result();

    assert_eq!(result, Ok(expected));
}

/// Test a report with an error escalates into an error that owns that very
/// report and renders its text.
#[test]
fn report_into_result_escalates_errors_with_the_report() {
    let report = build_report(vec![
        build_diagnostic(DiagnosticLevel::Warning, "careful", "v.warn"),
        build_diagnostic(DiagnosticLevel::Error, "boom", "v.explode"),
    ]);
    let expected = report.clone();
    let storage = report.diagnostics().as_ptr();

    let error = report
        .into_result()
        .expect_err("a report with an error fails");

    assert_eq!(error.report(), &expected);
    assert_eq!(error.report().diagnostics().as_ptr(), storage);
    assert_eq!(
        error.to_string(),
        "[WARNING] v.warn: careful\n[ERROR] v.explode: boom"
    );
    assert_eq!(error.into_report(), expected);
}

/// Test the escalated error is a standard error with no underlying cause.
#[test]
fn validation_failed_error_is_a_standard_error() {
    let error = build_report(vec![build_diagnostic(DiagnosticLevel::Error, "boom", "v")])
        .into_result()
        .expect_err("a report with an error fails");

    let as_error: &dyn Error = &error;

    assert!(as_error.source().is_none());
    assert_eq!(as_error.to_string(), "[ERROR] v: boom");
}

/// Test an escalated error keeps the records of the report it owns.
#[test]
fn validation_failed_error_keeps_the_report_records() {
    let report = ValidationReport::new(
        vec![build_diagnostic(DiagnosticLevel::Error, "boom", "v")],
        vec![7_u32, 8],
    );

    let error: ValidationFailedError<u32> = report
        .into_result()
        .expect_err("a report with an error fails");

    assert_eq!(error.report().records(), &[7, 8]);
}

/// Test a verifier story: two checks report into one run, the run fails on
/// the error, and the caller renders the failure for the user.
#[test]
fn a_failed_validation_run_is_reported_to_the_user() {
    let diagnostics = vec![
        Diagnostic::new(
            DiagnosticLevel::Warning,
            Note::new("prefer a smaller tile", get_suggestion_note_kind().clone()),
            "tiling.check",
            None,
        ),
        Diagnostic::new(
            DiagnosticLevel::Error,
            Note::with_other_kind("loop bound is negative"),
            "bounds.check",
            Some("bound -1 in loop i".to_owned()),
        ),
    ];
    let report: ValidationReport<&str> =
        ValidationReport::new(diagnostics, vec!["tiling.check", "bounds.check"]);

    let failure = report.into_result().expect_err("the run has an error");

    assert_eq!(
        failure.to_string(),
        "[WARNING] tiling.check: prefer a smaller tile\n\
         [ERROR] bounds.check: loop bound is negative\n\
         \x20   detail: bound -1 in loop i"
    );
    assert_eq!(failure.report().errors().count(), 1);
    assert_eq!(
        failure.report().records(),
        &["tiling.check", "bounds.check"]
    );
}

/// Compile-time check that the diagnostic types can cross threads.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<NoteKind>();
    assert_send_sync::<Note>();
    assert_send_sync::<DiagnosticLevel>();
    assert_send_sync::<Diagnostic>();
    assert_send_sync::<ValidationReport>();
    assert_send_sync::<ValidationFailedError>();
};
