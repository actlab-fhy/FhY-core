//! Diagnostics, notes, and validation reports shared by every verifier.
//!
//! A [`Note`] is a self-contained, human-readable explanation. It holds no
//! reference into the IR, so no transformation has to maintain it. Its
//! [`NoteKind`] names the role of the explanation (a rationale, a
//! suggestion, a neutral remark) so tooling can filter and group notes
//! wherever they are attached. The kinds form an open, registry-backed
//! vocabulary like [`crate::value_domain::ValueDomain`]: a layer registers
//! the kinds it needs without changing this crate, and
//! [`NoteKind::rationale`], [`NoteKind::suggestion`], [`NoteKind::remark`]
//! and [`NoteKind::other`] return the four kinds shipped here. Call them
//! rather than building a fresh kind with the same name hint. Identifiers
//! compare by id, and a second `Identifier::new("other")` is a different
//! key.
//!
//! A [`Diagnostic`] is a note emitted by a named source at a
//! [`DiagnosticLevel`], with optional detail. Build one with
//! [`Diagnostic::error`], [`Diagnostic::warning`] or [`Diagnostic::info`],
//! and attach a detail with [`Diagnostic::with_detail`]. A
//! [`ValidationReport`] collects
//! the diagnostics of a validation run, in emission order, together with
//! optional per-source records, displays them one diagnostic per line,
//! and escalates a report holding an error into a [`ValidationFailedError`]
//! with [`ValidationReport::into_result`].

use std::borrow::Cow;
use std::fmt;
use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use crate::described_tag::{DescribedTag, TagKind, require_shipped, sealed};
use crate::identifier::reserved::{self, ReservedIdentifier};
use crate::interned::{Canonical, InternRegistry};

/// The vocabulary of [`NoteKind`]s.
#[derive(Debug)]
pub enum NoteKindVocabulary {}

impl TagKind for NoteKindVocabulary {}

impl sealed::Sealed for NoteKindVocabulary {
    const TYPE_NAME: &'static str = "NoteKind";

    fn registry() -> &'static InternRegistry<NoteKind> {
        static REGISTRY: InternRegistry<NoteKind> =
            InternRegistry::with_defaults(create_default_note_kinds);
        &REGISTRY
    }
}

/// Open classification of the role a [`Note`] plays.
///
/// Two kinds are equal when they carry the same
/// [`Identifier`](crate::identifier::Identifier), whatever their
/// descriptions say.
///
/// A kind encodes as `{"name": {"id": .., "name_hint": ..}, "description":
/// ..}`. Only a [`Canonical<NoteKind>`] decodes, registering the kind unless
/// its name is registered already.
pub type NoteKind = DescribedTag<NoteKindVocabulary>;

/// The shipped kinds, in registration order, with their descriptions.
const SHIPPED_NOTE_KINDS: [(ReservedIdentifier, &str); 4] = [
    (
        reserved::RATIONALE_NOTE_KIND,
        "Explains why a decision, transformation, or result occurred.",
    ),
    (
        reserved::SUGGESTION_NOTE_KIND,
        "A suggested fix or course of action.",
    ),
    (
        reserved::REMARK_NOTE_KIND,
        "A neutral informational observation.",
    ),
    (reserved::OTHER_NOTE_KIND, "Uncategorized note."),
];

/// Build the kinds this module ships, in registration order.
///
/// The registry calls this once, on its first use, and keeps the instances
/// it builds, so the shipped kinds stay canonical for the life of the
/// process.
fn create_default_note_kinds() -> Vec<NoteKind> {
    SHIPPED_NOTE_KINDS
        .iter()
        .map(|&(entry, description)| NoteKind::create_shipped(entry, description))
        .collect()
}

static RATIONALE: LazyLock<Canonical<NoteKind>> =
    LazyLock::new(|| require_shipped(reserved::RATIONALE_NOTE_KIND));

static SUGGESTION: LazyLock<Canonical<NoteKind>> =
    LazyLock::new(|| require_shipped(reserved::SUGGESTION_NOTE_KIND));

static REMARK: LazyLock<Canonical<NoteKind>> =
    LazyLock::new(|| require_shipped(reserved::REMARK_NOTE_KIND));

static OTHER: LazyLock<Canonical<NoteKind>> =
    LazyLock::new(|| require_shipped(reserved::OTHER_NOTE_KIND));

impl DescribedTag<NoteKindVocabulary> {
    /// Return the kind for notes that explain why a decision,
    /// transformation, or result occurred.
    #[must_use]
    pub fn rationale() -> &'static Canonical<NoteKind> {
        &RATIONALE
    }

    /// Return the kind for notes that suggest a fix or course of action.
    #[must_use]
    pub fn suggestion() -> &'static Canonical<NoteKind> {
        &SUGGESTION
    }

    /// Return the kind for neutral informational notes.
    #[must_use]
    pub fn remark() -> &'static Canonical<NoteKind> {
        &REMARK
    }

    /// Return the kind for uncategorized notes.
    #[must_use]
    pub fn other() -> &'static Canonical<NoteKind> {
        &OTHER
    }
}

/// A human-readable message tagged with the role it plays.
///
/// A note encodes as `{"message": .., "kind": <note kind>}`. Decoding one
/// registers its kind, so it yields the canonical kind for that name; a
/// decode that fails after reading the kind leaves the kind registered.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Note {
    message: String,
    kind: Canonical<NoteKind>,
}

impl Note {
    /// Create the note carrying `message` in the role `kind`.
    #[must_use]
    pub fn new(message: impl Into<String>, kind: Canonical<NoteKind>) -> Self {
        Self {
            message: message.into(),
            kind,
        }
    }

    /// Create the note carrying `message` with the uncategorized kind
    /// returned by [`NoteKind::other`].
    #[must_use]
    pub fn with_other_kind(message: impl Into<String>) -> Self {
        Self::new(message, NoteKind::other().clone())
    }

    /// Return the message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Return the note's kind.
    #[must_use]
    pub fn kind(&self) -> &Canonical<NoteKind> {
        &self.kind
    }
}

/// Render `kind: message`, for example `other: lowered from ast`.
impl fmt::Display for Note {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

/// Severity of a [`Diagnostic`].
///
/// Levels have no order; compare them for equality. More levels may be
/// added, so a `match` on a level needs a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DiagnosticLevel {
    /// A problem that fails validation.
    Error,
    /// A suspicious condition that does not fail validation.
    Warning,
    /// Neutral information.
    Info,
}

impl DiagnosticLevel {
    /// Return the level's lowercase name: `error`, `warning`, or `info`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            DiagnosticLevel::Error => "error",
            DiagnosticLevel::Warning => "warning",
            DiagnosticLevel::Info => "info",
        }
    }
}

/// Renders [`DiagnosticLevel::as_str`].
impl fmt::Display for DiagnosticLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A note emitted at a level by a named source, with optional detail.
///
/// # Examples
///
/// ```
/// use fhy_core::diagnostic::{Diagnostic, DiagnosticLevel, Note};
///
/// let note = Note::with_other_kind("loop bound is negative");
///
/// let diagnostic = Diagnostic::error(note, "bounds.check").with_detail("bound -1 in loop i");
///
/// assert_eq!(diagnostic.level(), DiagnosticLevel::Error);
/// assert_eq!(diagnostic.source(), "bounds.check");
/// assert_eq!(diagnostic.detail(), Some("bound -1 in loop i"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Diagnostic {
    level: DiagnosticLevel,
    message: Note,
    source: Cow<'static, str>,
    detail: Option<String>,
}

impl Diagnostic {
    /// Create the diagnostic `message` emitted by `source` at `level`, with
    /// no detail.
    ///
    /// `source` identifies the emitter, typically a pass name. A `&'static
    /// str` is stored without copying.
    #[must_use]
    pub fn new(
        level: DiagnosticLevel,
        message: Note,
        source: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self {
            level,
            message,
            source: source.into(),
            detail: None,
        }
    }

    /// Create the diagnostic `message` emitted by `source` at
    /// [`DiagnosticLevel::Error`], with no detail.
    #[must_use]
    pub fn error(message: Note, source: impl Into<Cow<'static, str>>) -> Self {
        Self::new(DiagnosticLevel::Error, message, source)
    }

    /// Create the diagnostic `message` emitted by `source` at
    /// [`DiagnosticLevel::Warning`], with no detail.
    #[must_use]
    pub fn warning(message: Note, source: impl Into<Cow<'static, str>>) -> Self {
        Self::new(DiagnosticLevel::Warning, message, source)
    }

    /// Create the diagnostic `message` emitted by `source` at
    /// [`DiagnosticLevel::Info`], with no detail.
    #[must_use]
    pub fn info(message: Note, source: impl Into<Cow<'static, str>>) -> Self {
        Self::new(DiagnosticLevel::Info, message, source)
    }

    /// Return the diagnostic with its detail replaced by `detail`, stored as
    /// given, the empty string included.
    #[must_use]
    pub fn with_detail(self, detail: impl Into<String>) -> Self {
        Self {
            detail: Some(detail.into()),
            ..self
        }
    }

    /// Return the severity.
    #[must_use]
    pub fn level(&self) -> DiagnosticLevel {
        self.level
    }

    /// Return the message as a note.
    #[must_use]
    pub fn message(&self) -> &Note {
        &self.message
    }

    /// Return the message text, without the note's kind.
    #[must_use]
    pub fn message_text(&self) -> &str {
        self.message.message()
    }

    /// Return the name of whatever emitted the diagnostic.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Return the supplementary detail, if any.
    #[must_use]
    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }
}

/// Render `level[source]: message` with the message text alone, without
/// the note kind, followed, when the detail is present and non-empty, by a
/// second line `    detail: <detail>` indented by four spaces, as in
/// `error[shape.check]: missing return`. Newlines inside the message or the
/// detail are written as they are, and there is no trailing newline.
impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}[{}]: {}",
            self.level,
            self.source,
            self.message_text()
        )?;
        match self.detail() {
            Some(detail) if !detail.is_empty() => write!(f, "\n    detail: {detail}"),
            _ => Ok(()),
        }
    }
}

/// The diagnostics of a validation run, in emission order, with optional
/// per-source records of type `R`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ValidationReport<R = ()> {
    diagnostics: Vec<Diagnostic>,
    records: Vec<R>,
}

impl<R> ValidationReport<R> {
    /// Create the report holding `diagnostics` and `records`, in order.
    #[must_use]
    pub fn new(diagnostics: Vec<Diagnostic>, records: Vec<R>) -> Self {
        Self {
            diagnostics,
            records,
        }
    }

    /// Return every diagnostic, in emission order.
    #[must_use]
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Return the records, in the order given.
    #[must_use]
    pub fn records(&self) -> &[R] {
        &self.records
    }

    /// Return the [`DiagnosticLevel::Error`] diagnostics, in emission order.
    pub fn errors(&self) -> impl Iterator<Item = &Diagnostic> + '_ {
        self.filter_level(DiagnosticLevel::Error)
    }

    /// Return the [`DiagnosticLevel::Warning`] diagnostics, in emission
    /// order.
    pub fn warnings(&self) -> impl Iterator<Item = &Diagnostic> + '_ {
        self.filter_level(DiagnosticLevel::Warning)
    }

    /// Return the [`DiagnosticLevel::Info`] diagnostics, in emission order.
    pub fn infos(&self) -> impl Iterator<Item = &Diagnostic> + '_ {
        self.filter_level(DiagnosticLevel::Info)
    }

    /// Return the diagnostics at `level`, in emission order.
    fn filter_level(&self, level: DiagnosticLevel) -> impl Iterator<Item = &Diagnostic> + '_ {
        self.diagnostics
            .iter()
            .filter(move |diagnostic| diagnostic.level == level)
    }

    /// Return whether any diagnostic is at [`DiagnosticLevel::Error`].
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.errors().next().is_some()
    }

    /// Return the report, or escalate it into an error if it has errors.
    ///
    /// # Errors
    ///
    /// Returns a [`ValidationFailedError`] owning this report if
    /// [`has_errors`](Self::has_errors) holds.
    pub fn into_result(self) -> Result<Self, ValidationFailedError<R>> {
        if self.has_errors() {
            Err(ValidationFailedError { report: self })
        } else {
            Ok(self)
        }
    }
}

/// Render each diagnostic's [`Display`](fmt::Display) in emission order,
/// joined by `\n`, with no trailing newline. A report without diagnostics
/// renders as the empty string, and the records are not rendered.
///
/// # Examples
///
/// ```
/// use fhy_core::diagnostic::{Diagnostic, Note, ValidationReport};
///
/// let report: ValidationReport = ValidationReport::new(
///     vec![
///         Diagnostic::error(Note::with_other_kind("missing return"), "shape.check")
///             .with_detail("function foo() has no return statement"),
///         Diagnostic::info(Note::with_other_kind("fyi"), "v3"),
///     ],
///     Vec::new(),
/// );
///
/// assert_eq!(
///     report.to_string(),
///     "error[shape.check]: missing return\n    \
///      detail: function foo() has no return statement\n\
///      info[v3]: fyi"
/// );
/// ```
impl<R> fmt::Display for ValidationReport<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, diagnostic) in self.diagnostics.iter().enumerate() {
            if index > 0 {
                f.write_str("\n")?;
            }
            write!(f, "{diagnostic}")?;
        }
        Ok(())
    }
}

/// A [`ValidationReport`] that holds at least one error, escalated into an
/// error.
///
/// Its message is a one-line summary counting the errors; display
/// [`report`](Self::report) for the diagnostics themselves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationFailedError<R = ()> {
    report: ValidationReport<R>,
}

impl<R> ValidationFailedError<R> {
    /// Return the report that failed validation.
    #[must_use]
    pub fn report(&self) -> &ValidationReport<R> {
        &self.report
    }

    /// Return the report that failed validation, consuming the error.
    #[must_use]
    pub fn into_report(self) -> ValidationReport<R> {
        self.report
    }
}

/// Render `validation failed with 1 error`, or `validation failed with <n>
/// errors` for any other count of error diagnostics in the report.
impl<R> fmt::Display for ValidationFailedError<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.report.errors().count() {
            1 => f.write_str("validation failed with 1 error"),
            count => write!(f, "validation failed with {count} errors"),
        }
    }
}

/// The report is the error's data, not its cause, so
/// [`source`](std::error::Error::source) is `None`.
impl<R: fmt::Debug> std::error::Error for ValidationFailedError<R> {}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::identifier::Identifier;
    use crate::interned::Interned;

    /// Return the JSON payload of a note whose kind is named by the id `id`,
    /// with `kind_trailing` appended inside the kind and `trailing` appended
    /// after the kind.
    fn encode_note_payload(id: u64, kind_trailing: &str, trailing: &str) -> String {
        format!(
            "{{\"message\":\"m\",\"kind\":{{\"name\":{{\"id\":{id},\"name_hint\":\"k{id}\"}},\
             \"description\":\"d{id}\"{kind_trailing}}}{trailing}}}"
        )
    }

    /// Test each shipped kind holds its fixed reserved id and name hint.
    #[rstest]
    #[case::rationale(NoteKind::rationale, 0, "rationale")]
    #[case::suggestion(NoteKind::suggestion, 1, "suggestion")]
    #[case::remark(NoteKind::remark, 2, "remark")]
    #[case::other(NoteKind::other, 3, "other")]
    fn a_shipped_kind_holds_its_reserved_id(
        #[case] get_shipped: fn() -> &'static Canonical<NoteKind>,
        #[case] id: u64,
        #[case] name_hint: &str,
    ) {
        let name = get_shipped().name();

        assert_eq!((name.id(), name.name_hint()), (id, name_hint));
    }

    #[test]
    fn a_valid_note_registers_its_kind() {
        let id = Identifier::new("valid-note-kind").id();

        let note: Note = serde_json::from_str(&encode_note_payload(id, "", "")).unwrap();

        assert_eq!(note.kind().name().id(), id);
        let registered = NoteKind::intern_registry().get(note.kind().name());
        assert!(registered.is_some_and(|registered| Canonical::ptr_eq(&registered, note.kind())));
    }

    #[test]
    fn a_note_with_a_trailing_unknown_field_is_rejected() {
        let id = Identifier::new("trailing-note-field").id();

        let error =
            serde_json::from_str::<Note>(&encode_note_payload(id, "", ",\"zzz\":1")).unwrap_err();

        assert!(error.to_string().contains("zzz"), "{error}");
    }

    #[test]
    fn a_note_whose_kind_precedes_a_malformed_message_is_rejected() {
        let id = Identifier::new("kind-first-note").id();
        let json = format!(
            "{{\"kind\":{{\"name\":{{\"id\":{id},\"name_hint\":\"k\"}},\"description\":\"d\"}},\
             \"message\":5}}"
        );

        let error = serde_json::from_str::<Note>(&json).unwrap_err();

        assert!(error.to_string().contains("invalid type"), "{error}");
    }

    #[test]
    fn a_note_whose_kind_has_an_unknown_field_is_rejected() {
        let id = Identifier::new("kind-unknown-field").id();

        let error =
            serde_json::from_str::<Note>(&encode_note_payload(id, ",\"zzz\":1", "")).unwrap_err();

        assert!(error.to_string().contains("zzz"), "{error}");
    }

    #[test]
    fn a_note_whose_kind_lacks_a_description_is_rejected() {
        let id = Identifier::new("kind-no-description").id();
        let json = format!(
            "{{\"message\":\"m\",\"kind\":{{\"name\":{{\"id\":{id},\"name_hint\":\"k\"}}}}}}"
        );

        let error = serde_json::from_str::<Note>(&json).unwrap_err();

        assert!(
            error.to_string().contains("missing field `description`"),
            "{error}"
        );
    }

    #[test]
    fn a_note_kind_with_an_unknown_field_is_rejected() {
        let id = Identifier::new("bare-kind").id();
        let json = format!(
            "{{\"name\":{{\"id\":{id},\"name_hint\":\"k\"}},\"description\":\"d\",\"zzz\":1}}"
        );

        let error = serde_json::from_str::<Canonical<NoteKind>>(&json).unwrap_err();

        assert!(error.to_string().contains("zzz"), "{error}");
    }

    /// Test a note and each shipped kind round-trip through postcard, a
    /// format that is not self-describing.
    #[rstest]
    #[case::rationale(NoteKind::rationale)]
    #[case::other(NoteKind::other)]
    fn a_note_round_trips_through_postcard(
        #[case] get_shipped: fn() -> &'static Canonical<NoteKind>,
    ) {
        let note = Note::new("a message", get_shipped().clone());

        let bytes = postcard::to_allocvec(&note).expect("the note encodes");
        let restored: Note = postcard::from_bytes(&bytes).expect("the note decodes");

        assert_eq!(restored.message(), "a message");
        assert!(Canonical::ptr_eq(restored.kind(), get_shipped()));
    }
}
