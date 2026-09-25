//! `PyO3` classes for [`fhy_core::diagnostic`]: `fhy_core._rs.NoteKind`,
//! `Note`, `Diagnostic` and `ValidationReport`, the bases of the classes of
//! the same names in `fhy_core.diagnostic` on the Rust backend (pattern P2).
//!
//! Each class wraps the Rust value. `DiagnosticLevel` stays a Python enum
//! (pattern P1) and converts by value at the boundary. A report holds
//! arbitrary Python objects as its records, so its class wraps a
//! `ValidationReport<Py<PyAny>>`.
//!
//! The Python API is the one the pure-Python dataclasses have, with their
//! reprs, their `format()` text, their payloads and their exceptions: the
//! Rust core's `Display` text never reaches Python. The public classes
//! register themselves with the binding at import, so a value the binding
//! builds from Rust, such as a note's kind, is an instance of the public
//! class.

use std::hash::{DefaultHasher, Hash, Hasher};

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyDict, PyString, PyTuple, PyType};

use fhy_core::diagnostic::{Diagnostic, DiagnosticLevel, Note, NoteKind, ValidationReport};
use fhy_core::interned::Canonical;

use crate::described_tag::define_described_tag_class;
use crate::frozen::build_frozen_mutation_error;
use crate::public_class::PublicClass;
use crate::serialization::{FieldShape, read_payload_fields};

/// The Python module that defines the public classes.
const MODULE: &str = "fhy_core.diagnostic";

/// The text `format()` renders for a report without diagnostics.
const NO_DIAGNOSTICS_TEXT: &str = "No validation diagnostics.";

define_described_tag_class! {
    /// Open, registry-backed classification of an explanatory note's role,
    /// backed by the canonical Rust [`NoteKind`].
    class PyNoteKind as "NoteKind";
    seed NoteKindSeed;
    tag NoteKind;
    extra_methods {
        /// Return the name hint of the kind's name, as `Note` renders it.
        fn __str__(&self) -> String {
            self.tag.name().name_hint().to_owned()
        }
    }
}

/// A constructor argument that may be omitted, so that an explicit `None`
/// is not mistaken for the omitted argument's default.
enum OptionalArgument<'py> {
    /// The caller did not pass the argument.
    Omitted,
    /// The caller passed this object.
    Given(Bound<'py, PyAny>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for OptionalArgument<'py> {
    type Error = PyErr;

    fn extract(object: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self::Given(object.to_owned()))
    }
}

/// Return the `TypeError` for an argument `field` of `owner` that is not a
/// `expected`.
fn build_argument_type_error(
    owner: &str,
    field: &str,
    expected: &str,
    value: &Bound<'_, PyAny>,
) -> PyResult<PyErr> {
    Ok(PyTypeError::new_err(format!(
        "{owner} {field} must be {expected}, got {}.",
        value.get_type().name()?
    )))
}

/// Return `value` as a `str`, or raise the `TypeError` naming `owner` and
/// `field`.
fn read_str<'a, 'py>(
    value: &'a Bound<'py, PyAny>,
    owner: &str,
    field: &str,
) -> PyResult<&'a Bound<'py, PyString>> {
    match value.cast::<PyString>() {
        Ok(value) => Ok(value),
        Err(_not_a_str) => Err(build_argument_type_error(owner, field, "a str", value)?),
    }
}

/// Return the hash of `value` from the standard hasher.
///
/// Equal values hash equally within a process, which is all Python needs;
/// the hashes differ from the pure-Python classes' ones.
fn hash_value(value: &impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// Return `NotImplemented` unless `other` is an instance of exactly the
/// class of `object`, and otherwise whether `is_equal` holds for the two.
///
/// Matches the Python implementation: the `__eq__` a dataclass generates.
fn compare_as_dataclass<'py, T, F>(
    object: &Bound<'py, T>,
    other: &Bound<'py, PyAny>,
    is_equal: F,
) -> PyResult<Bound<'py, PyAny>>
where
    T: pyo3::PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + Sync,
    F: FnOnce(&T, &T) -> PyResult<bool>,
{
    let py = object.py();
    if !object.as_any().get_type().is(other.get_type()) {
        return Ok(py.NotImplemented().into_bound(py));
    }
    let other = other.cast::<T>()?;
    let is_equal = is_equal(object.get(), other.get())?;
    Ok(PyBool::new(py, is_equal).to_owned().into_any())
}

// ---------------------------------------------------------------------------
// DiagnosticLevel (pattern P1)
// ---------------------------------------------------------------------------

/// The Python `DiagnosticLevel` enum, with its member for each Rust level.
struct LevelTable {
    class: Py<PyType>,
    members: Vec<(DiagnosticLevel, Py<PyAny>)>,
}

/// The levels a Python `DiagnosticLevel` has.
const LEVELS: [DiagnosticLevel; 3] = [
    DiagnosticLevel::Error,
    DiagnosticLevel::Warning,
    DiagnosticLevel::Info,
];

/// Return the Python `DiagnosticLevel` and its members.
///
/// Each level's member is the one whose value is the level's lowercase
/// name, which both implementations share.
fn level_table(py: Python<'_>) -> PyResult<&LevelTable> {
    static TABLE: PyOnceLock<LevelTable> = PyOnceLock::new();
    TABLE.get_or_try_init(py, || {
        let class = py
            .import(MODULE)?
            .getattr(intern!(py, "DiagnosticLevel"))?
            .cast_into::<PyType>()?;
        let members = LEVELS
            .iter()
            .map(|&level| Ok((level, class.call1((level.as_str(),))?.unbind())))
            .collect::<PyResult<_>>()?;
        Ok(LevelTable {
            class: class.unbind(),
            members,
        })
    })
}

/// Convert a Python `DiagnosticLevel`, or any value the enum accepts, such
/// as `"error"`, to the Rust level.
///
/// # Errors
///
/// Raises the enum's `ValueError` for a value that names no level.
fn level_from_python(value: &Bound<'_, PyAny>) -> PyResult<DiagnosticLevel> {
    let py = value.py();
    let table = level_table(py)?;
    let find_member = |member: &Bound<'_, PyAny>| {
        table
            .members
            .iter()
            .find(|(_, candidate)| candidate.bind(py).is(member))
            .map(|&(level, _)| level)
    };
    if let Some(level) = find_member(value) {
        return Ok(level);
    }
    let member = table.class.bind(py).call1((value,))?;
    find_member(&member).ok_or_else(|| {
        PyValueError::new_err(format!(
            "{} has no Rust diagnostic level",
            member
                .repr()
                .map_or_else(|_| "?".into(), |repr| repr.to_string())
        ))
    })
}

/// Return the Python `DiagnosticLevel` member of `level`.
///
/// # Errors
///
/// Raises `ValueError` for a Rust level the Python enum lacks.
fn level_to_python(py: Python<'_>, level: DiagnosticLevel) -> PyResult<Bound<'_, PyAny>> {
    level_table(py)?
        .members
        .iter()
        .find(|&&(candidate, _)| candidate == level)
        .map(|(_, member)| member.bind(py).clone())
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "diagnostic level {level} has no Python DiagnosticLevel"
            ))
        })
}

// ---------------------------------------------------------------------------
// Note
// ---------------------------------------------------------------------------

/// A structured diagnostic message with a kind tag, backed by the Rust
/// [`Note`].
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Note")]
pub(crate) struct PyNote {
    note: Note,
    /// The message, the `str` the note was built from.
    #[pyo3(get)]
    message: Py<PyString>,
    /// The single Python object of the note's canonical kind.
    #[pyo3(get)]
    kind: Py<PyAny>,
}

impl PyNote {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("Note");
        &PUBLIC_CLASS
    }

    /// Return the canonical kind of the Python `kind`, a `NoteKind`.
    ///
    /// Every Python `NoteKind` is the single object of its canonical kind,
    /// so the note can hold `kind` itself as its kind's object.
    fn read_kind(kind: &Bound<'_, PyAny>) -> PyResult<Canonical<NoteKind>> {
        match kind.cast::<PyNoteKind>() {
            Ok(kind) => Ok(kind.get().tag.clone()),
            Err(_not_a_kind) => Err(build_argument_type_error(
                "Note",
                "kind",
                "a NoteKind",
                kind,
            )?),
        }
    }
}

#[pymethods]
impl PyNote {
    /// Create the note carrying `message` in the role `kind`, by default
    /// the uncategorized kind.
    ///
    /// Raises `TypeError` if `message` is not a `str` or `kind` is not a
    /// `NoteKind`.
    #[new]
    #[pyo3(signature = (message, kind = OptionalArgument::Omitted))]
    fn new(message: &Bound<'_, PyAny>, kind: OptionalArgument<'_>) -> PyResult<Self> {
        let py = message.py();
        let message = read_str(message, "Note", "message")?;
        let text = message.to_str()?;
        let (note, kind) = match kind {
            OptionalArgument::Omitted => {
                let note = Note::with_other_kind(text);
                let kind = PyNoteKind::to_python(py, None, note.kind().clone(), None)?;
                (note, kind)
            }
            OptionalArgument::Given(kind) => (Note::new(text, Self::read_kind(&kind)?), kind),
        };
        Ok(Self {
            note,
            message: message.clone().unbind(),
            kind: kind.unbind(),
        })
    }

    /// Always true: notes are immutable.
    #[getter]
    fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: notes are always frozen.
    fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: notes are always frozen, and mutating one raises.
    fn assert_frozen(_slf: &Bound<'_, Self>) {}

    fn __eq__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        compare_as_dataclass(slf, other, |this, other| Ok(this.note == other.note))
    }

    fn __hash__(&self) -> u64 {
        hash_value(&self.note)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        Ok(format!(
            "{}(message={}, kind={})",
            slf.get_type().qualname()?,
            this.message.bind(py).repr()?,
            this.kind.bind(py).repr()?,
        ))
    }

    /// Render `kind: message`, with the kind's name hint.
    fn __str__(&self) -> String {
        format!(
            "{}: {}",
            self.note.kind().name().name_hint(),
            self.note.message()
        )
    }

    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let _ = value;
        Err(build_frozen_mutation_error(slf, "modify", name)?)
    }

    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        Err(build_frozen_mutation_error(slf, "delete", name)?)
    }

    /// Pickle as a constructor call of the note's class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let arguments = PyTuple::new(py, [this.message.bind(py).as_any(), this.kind.bind(py)])?;
        Ok((slf.get_type(), arguments))
    }

    /// Return the payload `{"message": .., "kind": <note kind payload>}`.
    fn serialize_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let payload = PyDict::new(py);
        payload.set_item(intern!(py, "message"), self.note.message())?;
        payload.set_item(
            intern!(py, "kind"),
            PyNoteKind::serialize_tag(py, self.note.kind())?,
        )?;
        Ok(payload)
    }

    /// Return the note of a payload, registering its kind unless it is
    /// registered.
    ///
    /// Raises the Python implementation's errors for a malformed payload.
    #[classmethod]
    fn deserialize_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let [message, kind] = read_payload_fields(
            cls,
            data,
            [("message", FieldShape::Str), ("kind", FieldShape::Payload)],
        )?;
        let kind = PyNoteKind::deserialize_from_dict(PyNoteKind::public_class().get(py)?, &kind)?;
        cls.call1((message, kind))
    }

    /// Register `cls` as the public class.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }
}

// ---------------------------------------------------------------------------
// Diagnostic
// ---------------------------------------------------------------------------

/// A structured diagnostic emitted by a named source, backed by the Rust
/// [`Diagnostic`].
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Diagnostic")]
pub(crate) struct PyDiagnostic {
    diagnostic: Diagnostic,
    /// The severity's `DiagnosticLevel` member.
    #[pyo3(get)]
    level: Py<PyAny>,
    /// The Python object of the message, the `Note` it was built from.
    #[pyo3(get)]
    message: Py<PyAny>,
    /// The source, the `str` the diagnostic was built from.
    #[pyo3(get)]
    source: Py<PyString>,
    /// The detail, the `str` the diagnostic was built from, or `None`.
    #[pyo3(get)]
    detail: Py<PyAny>,
}

impl PyDiagnostic {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("Diagnostic");
        &PUBLIC_CLASS
    }
}

#[pymethods]
impl PyDiagnostic {
    /// Create the diagnostic `message` emitted by `source` at `level`, with
    /// optional `detail`.
    ///
    /// Raises `ValueError` for a `level` that names no `DiagnosticLevel`,
    /// and `TypeError` if `message` is not a `Note`, `source` is not a
    /// `str`, or `detail` is neither a `str` nor `None`.
    #[new]
    #[pyo3(signature = (level, message, source, detail = None))]
    fn new(
        level: &Bound<'_, PyAny>,
        message: &Bound<'_, PyAny>,
        source: &Bound<'_, PyAny>,
        detail: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let py = level.py();
        let level = level_from_python(level)?;
        let note = match message.cast::<PyNote>() {
            Ok(note) => note.get().note.clone(),
            Err(_not_a_note) => {
                return Err(build_argument_type_error(
                    "Diagnostic",
                    "message",
                    "a Note",
                    message,
                )?);
            }
        };
        let source = read_str(source, "Diagnostic", "source")?;
        let mut diagnostic = Diagnostic::new(level, note, source.to_str()?.to_owned());
        let detail = match detail.filter(|detail| !detail.is_none()) {
            Some(detail) => {
                let detail = read_str(detail, "Diagnostic", "detail")?;
                diagnostic = diagnostic.with_detail(detail.to_str()?);
                detail.as_any().clone()
            }
            None => py.None().into_bound(py),
        };
        Ok(Self {
            diagnostic,
            level: level_to_python(py, level)?.unbind(),
            message: message.clone().unbind(),
            source: source.clone().unbind(),
            detail: detail.unbind(),
        })
    }

    /// The underlying message text, without the kind prefix.
    #[getter]
    fn message_text<'py>(&self, py: Python<'py>) -> Bound<'py, PyString> {
        PyString::new(py, self.diagnostic.message_text())
    }

    /// Always true: diagnostics are immutable.
    #[getter]
    fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: diagnostics are always frozen.
    fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: diagnostics are always frozen, and mutating one raises.
    fn assert_frozen(_slf: &Bound<'_, Self>) {}

    fn __eq__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        compare_as_dataclass(slf, other, |this, other| {
            Ok(this.diagnostic == other.diagnostic)
        })
    }

    fn __hash__(&self) -> u64 {
        hash_value(&self.diagnostic)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        Ok(format!(
            "{}(level={}, message={}, source={}, detail={})",
            slf.get_type().qualname()?,
            this.level.bind(py).repr()?,
            this.message.bind(py).repr()?,
            this.source.bind(py).repr()?,
            this.detail.bind(py).repr()?,
        ))
    }

    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let _ = value;
        Err(build_frozen_mutation_error(slf, "modify", name)?)
    }

    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        Err(build_frozen_mutation_error(slf, "delete", name)?)
    }

    /// Pickle as a constructor call of the diagnostic's class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let arguments = PyTuple::new(
            py,
            [
                this.level.bind(py),
                this.message.bind(py),
                this.source.bind(py).as_any(),
                this.detail.bind(py),
            ],
        )?;
        Ok((slf.get_type(), arguments))
    }

    /// Register `cls` as the public class.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }
}

// ---------------------------------------------------------------------------
// ValidationReport
// ---------------------------------------------------------------------------

/// Return the Python `ValidationFailedError`, which stays a Python class.
fn validation_failed_error_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    CLASS.import(py, MODULE, "ValidationFailedError")
}

/// Return the items of the iterable `values` as a tuple, `values` itself if
/// it is a tuple.
fn collect_tuple<'py>(values: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    if let Ok(values) = values.cast_exact::<PyTuple>() {
        return Ok(values.clone());
    }
    PyTuple::new(
        values.py(),
        values.try_iter()?.collect::<PyResult<Vec<_>>>()?,
    )
}

/// Render the diagnostics as `format()` does: one `[LEVEL] source:
/// message` line each, followed by an indented `detail:` line when the
/// detail is non-empty, or the placeholder for no diagnostics.
///
/// Matches the Python implementation: `ValidationReport.format`.
fn format_diagnostics(diagnostics: &[Diagnostic]) -> String {
    if diagnostics.is_empty() {
        return NO_DIAGNOSTICS_TEXT.to_owned();
    }
    let mut text = String::new();
    for (index, diagnostic) in diagnostics.iter().enumerate() {
        if index > 0 {
            text.push('\n');
        }
        text.push('[');
        text.extend(
            diagnostic
                .level()
                .as_str()
                .chars()
                .map(|character| character.to_ascii_uppercase()),
        );
        text.push_str("] ");
        text.push_str(diagnostic.source());
        text.push_str(": ");
        text.push_str(diagnostic.message_text());
        if let Some(detail) = diagnostic.detail().filter(|detail| !detail.is_empty()) {
            text.push_str("\n    detail: ");
            text.push_str(detail);
        }
    }
    text
}

/// Aggregated diagnostics plus per-source execution records, backed by a
/// Rust [`ValidationReport`] over Python records.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "ValidationReport")]
pub(crate) struct PyValidationReport {
    report: ValidationReport<Py<PyAny>>,
    /// The Python objects of the diagnostics, in the report's order: the
    /// `Diagnostic`s the report was built from.
    #[pyo3(get)]
    diagnostics: Py<PyTuple>,
    /// The records, as given.
    #[pyo3(get)]
    records: Py<PyTuple>,
}

impl PyValidationReport {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("ValidationReport");
        &PUBLIC_CLASS
    }

    /// Return the Python objects of the diagnostics at `level`, in order.
    fn filter_level<'py>(
        &self,
        py: Python<'py>,
        level: DiagnosticLevel,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let diagnostics = self.diagnostics.bind(py);
        let selected = self
            .report
            .diagnostics()
            .iter()
            .enumerate()
            .filter(|(_, diagnostic)| diagnostic.level() == level)
            .map(|(index, _)| diagnostics.get_item(index))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, selected)
    }
}

#[pymethods]
impl PyValidationReport {
    /// Create the report of `diagnostics` and `records`, each an iterable,
    /// by default empty.
    ///
    /// Raises `TypeError` if a diagnostic is not a `Diagnostic`, or either
    /// argument is not iterable.
    #[new]
    #[pyo3(signature = (
        diagnostics = OptionalArgument::Omitted,
        records = OptionalArgument::Omitted,
    ))]
    fn new(
        py: Python<'_>,
        diagnostics: OptionalArgument<'_>,
        records: OptionalArgument<'_>,
    ) -> PyResult<Self> {
        let read_tuple = |argument| match argument {
            OptionalArgument::Omitted => Ok(PyTuple::empty(py)),
            OptionalArgument::Given(values) => collect_tuple(&values),
        };
        let diagnostics = read_tuple(diagnostics)?;
        let records = read_tuple(records)?;
        let rust_diagnostics = diagnostics
            .iter()
            .map(|diagnostic| match diagnostic.cast::<PyDiagnostic>() {
                Ok(diagnostic) => Ok(diagnostic.get().diagnostic.clone()),
                Err(_not_a_diagnostic) => Err(build_argument_type_error(
                    "ValidationReport",
                    "diagnostics",
                    "Diagnostic instances",
                    &diagnostic,
                )?),
            })
            .collect::<PyResult<_>>()?;
        let rust_records = records.iter().map(Bound::unbind).collect();
        Ok(Self {
            report: ValidationReport::new(rust_diagnostics, rust_records),
            diagnostics: diagnostics.unbind(),
            records: records.unbind(),
        })
    }

    /// Return only the ERROR-level diagnostics.
    fn errors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        self.filter_level(py, DiagnosticLevel::Error)
    }

    /// Return only the WARNING-level diagnostics.
    fn warnings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        self.filter_level(py, DiagnosticLevel::Warning)
    }

    /// Return only the INFO-level diagnostics.
    fn infos<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        self.filter_level(py, DiagnosticLevel::Info)
    }

    /// Return whether at least one ERROR-level diagnostic is present.
    fn has_errors(&self) -> bool {
        self.report.has_errors()
    }

    /// Return a human-readable rendering of every diagnostic.
    fn format(&self) -> String {
        format_diagnostics(self.report.diagnostics())
    }

    /// Raise `ValidationFailedError`, carrying this report, if any ERROR
    /// diagnostics exist.
    fn raise_if_failed(slf: &Bound<'_, Self>) -> PyResult<()> {
        if !slf.get().report.has_errors() {
            return Ok(());
        }
        let error = validation_failed_error_class(slf.py())?.call1((slf,))?;
        Err(PyErr::from_value(error))
    }

    /// Always true: reports are immutable.
    #[getter]
    fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: reports are always frozen.
    fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: reports are always frozen, and mutating one raises.
    fn assert_frozen(_slf: &Bound<'_, Self>) {}

    fn __eq__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        compare_as_dataclass(slf, other, |this, other| {
            Ok(this.report.diagnostics() == other.report.diagnostics()
                && this.records.bind(py).eq(other.records.bind(py))?)
        })
    }

    /// Hash the diagnostics and the records; raises `TypeError` if a record
    /// is unhashable.
    fn __hash__(&self, py: Python<'_>) -> PyResult<u64> {
        let mut hasher = DefaultHasher::new();
        self.report.diagnostics().hash(&mut hasher);
        self.records.bind(py).hash()?.hash(&mut hasher);
        Ok(hasher.finish())
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        Ok(format!(
            "{}(diagnostics={}, records={})",
            slf.get_type().qualname()?,
            this.diagnostics.bind(py).repr()?,
            this.records.bind(py).repr()?,
        ))
    }

    /// Raise `FrozenMutationError`, except for `__orig_class__`, which
    /// `typing` sets on an instance built through a subscripted class such
    /// as `ValidationReport[int](...)`.
    ///
    /// Matches the Python implementation: `FrozenMixin.__setattr__`.
    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if name == "__orig_class__" {
            // The public class's Python mixins give its instances a
            // `__dict__`; `object.__setattr__` would reject this class's
            // own `__setattr__`.
            if let Ok(instance_dict) = slf.getattr(intern!(slf.py(), "__dict__")) {
                return instance_dict.set_item(name, value);
            }
        }
        Err(build_frozen_mutation_error(slf, "modify", name)?)
    }

    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        Err(build_frozen_mutation_error(slf, "delete", name)?)
    }

    /// Pickle as a constructor call of the report's class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let arguments = PyTuple::new(py, [this.diagnostics.bind(py), this.records.bind(py)])?;
        Ok((slf.get_type(), arguments))
    }

    /// Register `cls` as the public class.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }
}
