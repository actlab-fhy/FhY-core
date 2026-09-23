//! Diagnostics, notes, and validation reports shared by every verifier.
//!
//! A [`Note`] is a self-contained, human-readable explanation. It holds no
//! reference into the IR, so no transformation has to maintain it. Its
//! [`NoteKind`] names the role of the explanation (a rationale, a
//! suggestion, a neutral remark) so tooling can filter and group notes
//! wherever they are attached. The kinds form an open, registry-backed
//! vocabulary like [`crate::value_domain::ValueDomain`]: a layer registers
//! the kinds it needs without changing this crate, and
//! [`get_rationale_note_kind`], [`get_suggestion_note_kind`],
//! [`get_remark_note_kind`] and [`get_other_note_kind`] return the four kinds
//! shipped here. Call them rather than building a fresh kind with the same
//! name hint. Identifiers compare by id, and a second
//! `Identifier::new("other")` is a different key.
//!
//! A [`Diagnostic`] is a note emitted by a named source at a
//! [`DiagnosticLevel`], with optional detail. A [`ValidationReport`] collects
//! the diagnostics of a validation run, in emission order, together with
//! optional per-source records, renders them with
//! [`ValidationReport::format`], and escalates a report holding an error
//! into a [`ValidationFailedError`] with [`ValidationReport::into_result`].

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::LazyLock;

use serde::{Deserialize, Deserializer, Serialize, de};

use crate::decode::{self, Decode, DeferredPayload};
use crate::identifier::{HasIdentifier, Identifier, IdentifierPayload};
use crate::interned::{
    Canonical, InternOutcome, InternRegistry, Interned, intern_decoded, require_default,
};

/// Open classification of the role a [`Note`] plays.
///
/// Two kinds are equal when they carry the same [`Identifier`], whatever
/// their descriptions say. The description is human-readable metadata: the
/// first kind registered under an identifier stays canonical, and a later
/// one is handed back to its caller in [`InternOutcome::AlreadyCanonical`]
/// instead of replacing it.
///
/// A kind encodes as `{"name": {"id": .., "name_hint": ..}, "description":
/// ..}`. Decoding a kind canonicalizes it only through the handle, so
/// deserialize a [`Canonical<NoteKind>`]. Deserializing a bare `NoteKind`
/// yields a value that no registry knows about. Decoding checks every field
/// before it restores the name, so a rejected payload leaves the id counter
/// untouched.
#[derive(Debug, Serialize)]
pub struct NoteKind {
    name: Identifier,
    description: String,
}

impl NoteKind {
    /// Builds the kind named `name` and registers it as the canonical one
    /// for that name.
    ///
    /// The outcome carries the canonical handle either way. When `name` is
    /// already taken the earlier kind stays canonical, and the one built here
    /// comes back as the outcome's `discarded` value.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::diagnostic::NoteKind;
    /// use fhy_core::identifier::Identifier;
    /// use fhy_core::interned::Interned;
    ///
    /// let name = Identifier::new("performance");
    /// let kind = NoteKind::new(name.clone(), "An optimization remark.").into_canonical();
    ///
    /// assert_eq!(kind.to_string(), "performance");
    /// assert_eq!(NoteKind::intern_registry().get(&name), Some(kind));
    /// ```
    pub fn new(name: Identifier, description: impl Into<String>) -> InternOutcome<Self> {
        Self::intern_registry().intern(Self::create(name, description))
    }

    /// Build the kind without registering it.
    fn create(name: Identifier, description: impl Into<String>) -> Self {
        Self {
            name,
            description: description.into(),
        }
    }

    /// Returns the kind's name.
    #[must_use]
    pub fn name(&self) -> &Identifier {
        &self.name
    }

    /// Returns the kind's human-readable description.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Renders the name's hint, for example `other`.
impl fmt::Display for NoteKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.name, f)
    }
}

impl HasIdentifier for NoteKind {
    fn identifier(&self) -> &Identifier {
        &self.name
    }
}

impl Interned for NoteKind {
    type Key = Identifier;

    fn intern_key(&self) -> &Identifier {
        &self.name
    }

    fn intern_registry() -> &'static InternRegistry<Self> {
        static REGISTRY: InternRegistry<NoteKind> =
            InternRegistry::with_defaults(create_default_note_kinds);
        &REGISTRY
    }
}

impl PartialEq for NoteKind {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for NoteKind {}

impl Hash for NoteKind {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl Decode for NoteKind {
    type Payload = NoteKindPayload;

    fn build_from_payload<E: de::Error>(payload: Self::Payload) -> Result<Self, E> {
        Self::intern_registry().initialize();
        Ok(Self::create(payload.name.restore(), payload.description))
    }
}

/// Decoding registers the shipped kinds first if this is the registry's
/// first use, so their names draw ids before the payload's name can exhaust
/// the counter.
impl<'de> Deserialize<'de> for NoteKind {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        decode::deserialize_via_payload(deserializer)
    }
}

/// A note-kind payload, checked but with its name not yet restored.
#[derive(Deserialize)]
#[serde(rename = "NoteKind", deny_unknown_fields)]
pub(crate) struct NoteKindPayload {
    name: IdentifierPayload,
    description: String,
}

/// Name of the kind returned by [`get_rationale_note_kind`].
static RATIONALE_NAME: LazyLock<Identifier> =
    LazyLock::new(|| Identifier::new_unscoped("rationale"));

/// Name of the kind returned by [`get_suggestion_note_kind`].
static SUGGESTION_NAME: LazyLock<Identifier> =
    LazyLock::new(|| Identifier::new_unscoped("suggestion"));

/// Name of the kind returned by [`get_remark_note_kind`].
static REMARK_NAME: LazyLock<Identifier> = LazyLock::new(|| Identifier::new_unscoped("remark"));

/// Name of the kind returned by [`get_other_note_kind`].
static OTHER_NAME: LazyLock<Identifier> = LazyLock::new(|| Identifier::new_unscoped("other"));

/// Build the kinds this module ships, in registration order.
///
/// The registry calls this once, on its first use, and keeps the instances it
/// builds. A clear registers those same instances again rather than building
/// new ones, so the shipped kinds stay canonical for the life of the process.
fn create_default_note_kinds() -> Vec<NoteKind> {
    vec![
        NoteKind::create(
            RATIONALE_NAME.clone(),
            "Explains why a decision, transformation, or result occurred.",
        ),
        NoteKind::create(
            SUGGESTION_NAME.clone(),
            "A suggested fix or course of action.",
        ),
        NoteKind::create(REMARK_NAME.clone(), "A neutral informational observation."),
        NoteKind::create(OTHER_NAME.clone(), "Uncategorized note."),
    ]
}

static RATIONALE: LazyLock<Canonical<NoteKind>> =
    LazyLock::new(|| require_default(&*RATIONALE_NAME));

static SUGGESTION: LazyLock<Canonical<NoteKind>> =
    LazyLock::new(|| require_default(&*SUGGESTION_NAME));

static REMARK: LazyLock<Canonical<NoteKind>> = LazyLock::new(|| require_default(&*REMARK_NAME));

static OTHER: LazyLock<Canonical<NoteKind>> = LazyLock::new(|| require_default(&*OTHER_NAME));

/// Returns the kind for notes that explain why a decision, transformation,
/// or result occurred.
#[must_use]
pub fn get_rationale_note_kind() -> &'static Canonical<NoteKind> {
    &RATIONALE
}

/// Returns the kind for notes that suggest a fix or course of action.
#[must_use]
pub fn get_suggestion_note_kind() -> &'static Canonical<NoteKind> {
    &SUGGESTION
}

/// Returns the kind for neutral informational notes.
#[must_use]
pub fn get_remark_note_kind() -> &'static Canonical<NoteKind> {
    &REMARK
}

/// Returns the kind for uncategorized notes.
#[must_use]
pub fn get_other_note_kind() -> &'static Canonical<NoteKind> {
    &OTHER
}

/// A human-readable message tagged with the role it plays.
///
/// A note encodes as `{"message": .., "kind": <note kind>}`. Decoding checks
/// the whole payload before it restores the kind's name, and yields the
/// canonical kind for that name.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct Note {
    message: String,
    kind: Canonical<NoteKind>,
}

impl Note {
    /// Creates the note carrying `message` in the role `kind`.
    #[must_use]
    pub fn new(message: impl Into<String>, kind: Canonical<NoteKind>) -> Self {
        Self {
            message: message.into(),
            kind,
        }
    }

    /// Creates the note carrying `message` with the uncategorized kind
    /// returned by [`get_other_note_kind`].
    #[must_use]
    pub fn with_other_kind(message: impl Into<String>) -> Self {
        Self::new(message, get_other_note_kind().clone())
    }

    /// Returns the message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns the note's kind.
    #[must_use]
    pub fn kind(&self) -> &Canonical<NoteKind> {
        &self.kind
    }
}

/// Renders `kind: message`, for example `other: lowered from ast`.
impl fmt::Display for Note {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

impl Decode for Note {
    type Payload = NotePayload;

    fn build_from_payload<E: de::Error>(payload: Self::Payload) -> Result<Self, E> {
        let kind = intern_decoded(payload.kind.decode("kind")?)?;
        Ok(Self::new(payload.message, kind))
    }
}

/// Decoding rejects a missing or unknown key at either level before it
/// restores the kind's name.
impl<'de> Deserialize<'de> for Note {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        decode::deserialize_via_payload(deserializer)
    }
}

/// A note payload, checked but with its kind held unread.
#[derive(Deserialize)]
#[serde(rename = "Note", deny_unknown_fields)]
pub(crate) struct NotePayload {
    message: String,
    kind: DeferredPayload<NoteKind>,
}

/// Severity of a [`Diagnostic`].
///
/// Levels have no order; compare them for equality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiagnosticLevel {
    /// A problem that fails validation.
    Error,
    /// A suspicious condition that does not fail validation.
    Warning,
    /// Neutral information.
    Info,
}

impl DiagnosticLevel {
    /// Returns the level's lowercase name: `error`, `warning`, or `info`.
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Diagnostic {
    level: DiagnosticLevel,
    message: Note,
    source: String,
    detail: Option<String>,
}

impl Diagnostic {
    /// Creates the diagnostic `message` emitted by `source` at `level`, with
    /// supplementary `detail`.
    ///
    /// `source` identifies the emitter, typically a pass name.
    #[must_use]
    pub fn new(
        level: DiagnosticLevel,
        message: Note,
        source: impl Into<String>,
        detail: Option<String>,
    ) -> Self {
        Self {
            level,
            message,
            source: source.into(),
            detail,
        }
    }

    /// Returns the severity.
    #[must_use]
    pub fn level(&self) -> DiagnosticLevel {
        self.level
    }

    /// Returns the message as a note.
    #[must_use]
    pub fn message(&self) -> &Note {
        &self.message
    }

    /// Returns the message text, without the note's kind.
    #[must_use]
    pub fn message_text(&self) -> &str {
        self.message.message()
    }

    /// Returns the identifier of whatever emitted the diagnostic.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Returns the supplementary detail, if any.
    #[must_use]
    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }
}

/// Text [`ValidationReport::format`] renders for a report without
/// diagnostics.
const EMPTY_REPORT_TEXT: &str = "No validation diagnostics.";

/// The diagnostics of a validation run, in emission order, with optional
/// per-source records of type `R`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ValidationReport<R = ()> {
    diagnostics: Vec<Diagnostic>,
    records: Vec<R>,
}

impl<R> ValidationReport<R> {
    /// Creates the report holding `diagnostics` and `records`, in order.
    #[must_use]
    pub fn new(diagnostics: Vec<Diagnostic>, records: Vec<R>) -> Self {
        Self {
            diagnostics,
            records,
        }
    }

    /// Returns every diagnostic, in emission order.
    #[must_use]
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Returns the records, in the order given.
    #[must_use]
    pub fn records(&self) -> &[R] {
        &self.records
    }

    /// Returns the [`DiagnosticLevel::Error`] diagnostics, in emission order.
    pub fn errors(&self) -> impl Iterator<Item = &Diagnostic> + '_ {
        self.filter_level(DiagnosticLevel::Error)
    }

    /// Returns the [`DiagnosticLevel::Warning`] diagnostics, in emission
    /// order.
    pub fn warnings(&self) -> impl Iterator<Item = &Diagnostic> + '_ {
        self.filter_level(DiagnosticLevel::Warning)
    }

    /// Returns the [`DiagnosticLevel::Info`] diagnostics, in emission order.
    pub fn infos(&self) -> impl Iterator<Item = &Diagnostic> + '_ {
        self.filter_level(DiagnosticLevel::Info)
    }

    /// Return the diagnostics at `level`, in emission order.
    fn filter_level(&self, level: DiagnosticLevel) -> impl Iterator<Item = &Diagnostic> + '_ {
        self.diagnostics
            .iter()
            .filter(move |diagnostic| diagnostic.level == level)
    }

    /// Returns whether any diagnostic is at [`DiagnosticLevel::Error`].
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.errors().next().is_some()
    }

    /// Renders every diagnostic for a human reader.
    ///
    /// A report without diagnostics renders as `No validation diagnostics.`.
    /// Otherwise each diagnostic renders as `[LEVEL] source: message` with
    /// the level in upper case and the message text without its note kind,
    /// followed, when the detail is present and non-empty, by a line
    /// `    detail: <detail>` indented by four spaces. Lines are joined by
    /// `\n` with no trailing newline, and newlines inside a message or
    /// detail are kept as they are.
    #[must_use]
    pub fn format(&self) -> String {
        if self.diagnostics.is_empty() {
            return EMPTY_REPORT_TEXT.to_owned();
        }
        let mut lines = Vec::with_capacity(self.diagnostics.len());
        for diagnostic in &self.diagnostics {
            lines.push(format!(
                "[{}] {}: {}",
                diagnostic.level.as_str().to_ascii_uppercase(),
                diagnostic.source,
                diagnostic.message_text()
            ));
            if let Some(detail) = diagnostic.detail().filter(|detail| !detail.is_empty()) {
                lines.push(format!("    detail: {detail}"));
            }
        }
        lines.join("\n")
    }

    /// Returns the report, or escalates it into an error if it has errors.
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

/// A [`ValidationReport`] that holds at least one error, escalated into an
/// error.
///
/// Its message is the report's [`ValidationReport::format`] text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationFailedError<R = ()> {
    report: ValidationReport<R>,
}

impl<R> ValidationFailedError<R> {
    /// Returns the report that failed validation.
    #[must_use]
    pub fn report(&self) -> &ValidationReport<R> {
        &self.report
    }

    /// Returns the report that failed validation, consuming the error.
    #[must_use]
    pub fn into_report(self) -> ValidationReport<R> {
        self.report
    }
}

/// Renders the report's [`ValidationReport::format`] text.
impl<R> fmt::Display for ValidationFailedError<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.report.format())
    }
}

impl<R: fmt::Debug> std::error::Error for ValidationFailedError<R> {}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::identifier::{IdSpaceExhausted, try_allocate_id};
    use crate::test_support::{
        RegistryGuard, assert_isolated_test_passes, has_counter_passed, hold_id_counter,
        is_isolated_run, reserve_far_ahead_ids,
    };

    /// Serializes the test that clears the process-wide registry against the
    /// tests that need their own entries to survive.
    static REGISTRY_GUARD: RegistryGuard = RegistryGuard::new();

    /// Return the JSON payload of a note whose kind is named by the id `id`,
    /// with `kind_trailing` appended inside the kind and `trailing` appended
    /// after the kind.
    fn encode_note_payload(id: u64, kind_trailing: &str, trailing: &str) -> String {
        format!(
            "{{\"message\":\"m\",\"kind\":{{\"name\":{{\"id\":{id},\"name_hint\":\"k{id}\"}},\
             \"description\":\"d{id}\"{kind_trailing}}}{trailing}}}"
        )
    }

    #[test]
    fn clearing_the_registry_keeps_the_shipped_kinds_canonical() {
        let _guard = REGISTRY_GUARD.hold_exclusively();
        let name = Identifier::new("dropped-by-clear");
        let dropped = NoteKind::new(name.clone(), "dropped").into_canonical();
        assert_eq!(NoteKind::intern_registry().get(&name), Some(dropped));

        NoteKind::intern_registry().clear();

        assert_eq!(NoteKind::intern_registry().get(&name), None);
        for shipped in [
            get_rationale_note_kind(),
            get_suggestion_note_kind(),
            get_remark_note_kind(),
            get_other_note_kind(),
        ] {
            assert_eq!(
                NoteKind::intern_registry().get(shipped.name()),
                Some(shipped.clone())
            );
        }
    }

    #[test]
    fn a_valid_note_restores_and_registers_its_kind() {
        let _guard = REGISTRY_GUARD.hold();
        let _counter = hold_id_counter();
        let [id] = reserve_far_ahead_ids("valid-note-anchor");

        let note: Note = serde_json::from_str(&encode_note_payload(id, "", "")).unwrap();

        assert!(has_counter_passed(id));
        assert_eq!(note.kind().name().id(), id);
        assert_eq!(
            NoteKind::intern_registry().get(note.kind().name()).as_ref(),
            Some(note.kind())
        );
    }

    #[test]
    fn a_note_rejected_for_a_trailing_unknown_field_restores_nothing() {
        let _guard = REGISTRY_GUARD.hold();
        let _counter = hold_id_counter();
        let [id] = reserve_far_ahead_ids("trailing-note-field-anchor");

        let error =
            serde_json::from_str::<Note>(&encode_note_payload(id, "", ",\"zzz\":1")).unwrap_err();

        assert!(error.to_string().contains("zzz"), "{error}");
        assert!(!has_counter_passed(id));
    }

    #[test]
    fn a_note_whose_kind_precedes_a_malformed_message_restores_nothing() {
        let _guard = REGISTRY_GUARD.hold();
        let _counter = hold_id_counter();
        let [id] = reserve_far_ahead_ids("kind-first-note-anchor");
        let json = format!(
            "{{\"kind\":{{\"name\":{{\"id\":{id},\"name_hint\":\"k\"}},\"description\":\"d\"}},\
             \"message\":5}}"
        );

        let error = serde_json::from_str::<Note>(&json).unwrap_err();

        assert!(error.to_string().contains("invalid type"), "{error}");
        assert!(!has_counter_passed(id));
    }

    #[test]
    fn a_note_whose_kind_has_an_unknown_field_restores_nothing() {
        let _guard = REGISTRY_GUARD.hold();
        let _counter = hold_id_counter();
        let [id] = reserve_far_ahead_ids("kind-unknown-field-anchor");

        let error =
            serde_json::from_str::<Note>(&encode_note_payload(id, ",\"zzz\":1", "")).unwrap_err();

        assert!(error.to_string().contains("zzz"), "{error}");
        assert!(!has_counter_passed(id));
    }

    #[test]
    fn a_note_whose_kind_lacks_a_description_restores_nothing() {
        let _guard = REGISTRY_GUARD.hold();
        let _counter = hold_id_counter();
        let [id] = reserve_far_ahead_ids("kind-no-description-anchor");
        let json = format!(
            "{{\"message\":\"m\",\"kind\":{{\"name\":{{\"id\":{id},\"name_hint\":\"k\"}}}}}}"
        );

        let error = serde_json::from_str::<Note>(&json).unwrap_err();

        assert!(
            error.to_string().contains("missing field `description`"),
            "{error}"
        );
        assert!(!has_counter_passed(id));
    }

    #[test]
    fn a_bare_note_kind_rejected_for_an_unknown_field_restores_nothing() {
        let _guard = REGISTRY_GUARD.hold();
        let _counter = hold_id_counter();
        let [id] = reserve_far_ahead_ids("bare-kind-anchor");
        let json = format!(
            "{{\"name\":{{\"id\":{id},\"name_hint\":\"k\"}},\"description\":\"d\",\"zzz\":1}}"
        );

        let error = serde_json::from_str::<Canonical<NoteKind>>(&json).unwrap_err();

        assert!(error.to_string().contains("zzz"), "{error}");
        assert!(!has_counter_passed(id));
    }

    /// Check that the shipped kinds survive a decode that exhausts the id
    /// counter before the registry's first use. Only meaningful in the child
    /// process that
    /// [`decoding_the_largest_id_as_the_first_use_keeps_the_shipped_kinds`]
    /// starts.
    #[test]
    #[ignore = "exhausts the process-global counter; run through assert_isolated_test_passes"]
    fn decoding_the_largest_id_as_the_first_use_keeps_the_shipped_kinds_in_isolation() {
        if !is_isolated_run() {
            return;
        }
        let json = format!(
            "{{\"name\":{{\"id\":{},\"name_hint\":\"last\"}},\"description\":\"d\"}}",
            u64::MAX - 1
        );

        let restored: Canonical<NoteKind> =
            serde_json::from_str(&json).expect("u64::MAX - 1 is valid");

        assert_eq!(restored.name().id(), u64::MAX - 1);
        assert_eq!(try_allocate_id(), Err(IdSpaceExhausted));
        assert_eq!(get_other_note_kind().name().name_hint(), "other");
        assert_eq!(get_rationale_note_kind().name().name_hint(), "rationale");
    }

    #[test]
    fn decoding_the_largest_id_as_the_first_use_keeps_the_shipped_kinds() {
        assert_isolated_test_passes(
            "diagnostic::tests::\
             decoding_the_largest_id_as_the_first_use_keeps_the_shipped_kinds_in_isolation",
        );
    }
}
