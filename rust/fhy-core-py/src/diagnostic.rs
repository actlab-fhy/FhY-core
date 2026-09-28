//! `PyO3` classes for [`fhy_core::diagnostic`]: `fhy_core._rs.NoteKind`,
//! `Note`, `Diagnostic` and `ValidationReport`, the bases of the classes of
//! the same names in `fhy_core.diagnostic`.
//!
//! Each class wraps the Rust value. `DiagnosticLevel` stays a Python enum
//! and converts by value at the boundary. A report keeps the
//! tuples of `Diagnostic` objects and records it was built from, and runs
//! its operations over the Rust diagnostics borrowed from those objects,
//! as the core's `ValidationReport` does over its own.
//!
//! The Python API is the one the retired pure-Python dataclasses had, with
//! their reprs, their `format()` text, their payloads and their exceptions:
//! the Rust core's `Display` text never reaches Python. The public classes
//! register themselves with the binding at import, so a value the binding
//! builds from Rust, such as a note's kind, is an instance of the public
//! class.

use std::hash::{DefaultHasher, Hash, Hasher};

use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyString, PyTuple, PyType};

use fhy_core::diagnostic::{Diagnostic, DiagnosticLevel, Note, NoteKind};
use fhy_core::interned::Canonical;

use crate::dataclass::{
    OptionalArgument, build_argument_type_error, collect_tuple, compare_as_dataclass, hash_value,
    read_str,
};
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

// ---------------------------------------------------------------------------
// DiagnosticLevel
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
pub(crate) fn level_to_python(
    py: Python<'_>,
    level: DiagnosticLevel,
) -> PyResult<Bound<'_, PyAny>> {
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
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.message)?;
        visit.call(&self.kind)?;
        Ok(())
    }

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
    const fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: notes are always frozen.
    const fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: notes are always frozen, and mutating one raises.
    const fn assert_frozen(_slf: &Bound<'_, Self>) {}

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
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.level)?;
        visit.call(&self.message)?;
        visit.call(&self.source)?;
        visit.call(&self.detail)?;
        Ok(())
    }

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
    const fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: diagnostics are always frozen.
    const fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: diagnostics are always frozen, and mutating one raises.
    const fn assert_frozen(_slf: &Bound<'_, Self>) {}

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

/// Return the Rust diagnostic of `object`, borrowed from it, or `None` if
/// `object` is not a `Diagnostic`.
pub(crate) fn borrow_python_diagnostic<'a>(object: &'a Bound<'_, PyAny>) -> Option<&'a Diagnostic> {
    object
        .cast::<PyDiagnostic>()
        .ok()
        .map(|diagnostic| &diagnostic.get().diagnostic)
}

/// Return a new object of the public `Diagnostic` class holding
/// `diagnostic`, with a new public `Note` as its message.
///
/// For a diagnostic that reaches Python from Rust; the note's kind is the
/// single Python object of its canonical kind.
pub(crate) fn diagnostic_to_python<'py>(
    py: Python<'py>,
    diagnostic: &Diagnostic,
) -> PyResult<Bound<'py, PyAny>> {
    let note = diagnostic.message();
    let kind = PyNoteKind::to_python(py, None, note.kind().clone(), None)?;
    let note = PyNote::public_class()
        .get(py)?
        .call1((note.message(), kind))?;
    PyDiagnostic::public_class().get(py)?.call1((
        level_to_python(py, diagnostic.level())?,
        note,
        diagnostic.source(),
        diagnostic.detail(),
    ))
}

// ---------------------------------------------------------------------------
// ValidationReport
// ---------------------------------------------------------------------------

/// Append `diagnostic` to `text` as `format()` renders it: a `[LEVEL]
/// source: message` line, followed by an indented `detail:` line when the
/// detail is non-empty.
///
/// Matches the Python implementation: `ValidationReport.format`.
fn format_diagnostic(text: &mut String, diagnostic: &Diagnostic) {
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

/// Return the Rust diagnostic of `diagnostic`, a `Diagnostic` object,
/// borrowed from it.
fn borrow_diagnostic<'a>(diagnostic: Borrowed<'a, '_, PyAny>) -> PyResult<&'a Diagnostic> {
    Ok(&diagnostic.cast::<PyDiagnostic>()?.get().diagnostic)
}

/// Aggregated diagnostics plus per-source execution records.
///
/// The report keeps the tuples it was built from. Its operations run over
/// the Rust [`Diagnostic`] each diagnostic object holds, borrowed rather
/// than copied, so building a report costs a type check per diagnostic.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "ValidationReport")]
pub(crate) struct PyValidationReport {
    /// The diagnostics, in the report's order: the `Diagnostic`s the report
    /// was built from.
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

    /// Return the diagnostic objects at `level`, in order.
    fn filter_level<'py>(
        &self,
        py: Python<'py>,
        level: DiagnosticLevel,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let mut selected = Vec::new();
        for diagnostic in self.diagnostics.bind(py).iter_borrowed() {
            if borrow_diagnostic(diagnostic)?.level() == level {
                selected.push(diagnostic);
            }
        }
        PyTuple::new(py, selected)
    }

    /// Return whether at least one diagnostic is an error.
    fn contains_errors(&self, py: Python<'_>) -> PyResult<bool> {
        for diagnostic in self.diagnostics.bind(py).iter_borrowed() {
            if borrow_diagnostic(diagnostic)?.level() == DiagnosticLevel::Error {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Return whether both reports hold equal diagnostics, in order.
    fn has_equal_diagnostics(&self, py: Python<'_>, other: &Self) -> PyResult<bool> {
        let diagnostics = self.diagnostics.bind(py);
        let other_diagnostics = other.diagnostics.bind(py);
        if diagnostics.len() != other_diagnostics.len() {
            return Ok(false);
        }
        for (diagnostic, other_diagnostic) in diagnostics
            .iter_borrowed()
            .zip(other_diagnostics.iter_borrowed())
        {
            if borrow_diagnostic(diagnostic)? != borrow_diagnostic(other_diagnostic)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// Return a new object of the public `ValidationReport` class of
/// `diagnostics`, which must hold `Diagnostic`s, and `records`.
pub(crate) fn report_to_python<'py>(
    py: Python<'py>,
    diagnostics: Bound<'py, PyTuple>,
    records: Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    PyValidationReport::public_class()
        .get(py)?
        .call1((diagnostics, records))
}

#[pymethods]
impl PyValidationReport {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.diagnostics)?;
        visit.call(&self.records)?;
        Ok(())
    }

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
        for diagnostic in diagnostics.iter_borrowed() {
            if !diagnostic.is_instance_of::<PyDiagnostic>() {
                return Err(build_argument_type_error(
                    "ValidationReport",
                    "diagnostics",
                    "Diagnostic instances",
                    &diagnostic,
                )?);
            }
        }
        Ok(Self {
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
    fn has_errors(&self, py: Python<'_>) -> PyResult<bool> {
        self.contains_errors(py)
    }

    /// Return a human-readable rendering of every diagnostic: one
    /// `[LEVEL] source: message` line each, followed by an indented
    /// `detail:` line when the detail is non-empty, or a placeholder for
    /// no diagnostics.
    ///
    /// Matches the Python implementation: `ValidationReport.format`.
    fn format(&self, py: Python<'_>) -> PyResult<String> {
        let diagnostics = self.diagnostics.bind(py);
        if diagnostics.is_empty() {
            return Ok(NO_DIAGNOSTICS_TEXT.to_owned());
        }
        let mut text = String::new();
        for (index, diagnostic) in diagnostics.iter_borrowed().enumerate() {
            if index > 0 {
                text.push('\n');
            }
            format_diagnostic(&mut text, borrow_diagnostic(diagnostic)?);
        }
        Ok(text)
    }

    /// Raise `ValidationFailedError`, carrying this report, if any ERROR
    /// diagnostics exist.
    fn raise_if_failed(slf: &Bound<'_, Self>) -> PyResult<()> {
        if !slf.get().contains_errors(slf.py())? {
            return Ok(());
        }
        Err(crate::exceptions::VALIDATION_FAILED_ERROR.err(slf.py(), (slf,)))
    }

    /// Always true: reports are immutable.
    #[getter]
    const fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: reports are always frozen.
    const fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: reports are always frozen, and mutating one raises.
    const fn assert_frozen(_slf: &Bound<'_, Self>) {}

    fn __eq__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        compare_as_dataclass(slf, other, |this, other| {
            Ok(this.has_equal_diagnostics(py, other)?
                && this.records.bind(py).eq(other.records.bind(py))?)
        })
    }

    /// Hash the diagnostics and the records; raises `TypeError` if a record
    /// is unhashable.
    fn __hash__(&self, py: Python<'_>) -> PyResult<u64> {
        let mut hasher = DefaultHasher::new();
        let diagnostics = self.diagnostics.bind(py);
        hasher.write_usize(diagnostics.len());
        for diagnostic in diagnostics.iter_borrowed() {
            borrow_diagnostic(diagnostic)?.hash(&mut hasher);
        }
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
