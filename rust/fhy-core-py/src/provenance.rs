//! `PyO3` classes for [`fhy_core::provenance`]: `fhy_core._rs.Position`,
//! `Span`, `Provenance` and its five variant classes, the bases of the
//! classes of the same names in `fhy_core.provenance`, as a class hierarchy.
//!
//! `Provenance` is a `#[pyclass(subclass)]` base that holds the Rust
//! [`Provenance`], and each variant class extends it with the Python objects
//! its fields return: the objects it was built from, so reading a field
//! costs what a dataclass field read costs. `Position` and `Span` hold the
//! Rust value next to their field objects in the same way. Equality, hashing
//! and `str` run on the Rust values.
//!
//! The Python API is the one the retired pure-Python dataclasses had, with
//! their reprs, their validation errors and their payloads, including the
//! `WrappedFamilySerializable` envelope that the public classes inherit. The public classes register
//! themselves with the binding at import, so a provenance the binding builds
//! in Rust, such as the result of `Provenance.fuse`, is an instance of the
//! public class.

use pyo3::exceptions::{PyOverflowError, PyRecursionError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyDict, PyInt, PyList, PyString, PyTuple, PyType};

use fhy_core::provenance::{
    CallSiteProvenance, FileProvenance, FusedProvenance, NamedProvenance, NamedProvenanceError,
    Position, PositionError, Provenance, Span, SpanError,
};

use crate::dataclass::{
    build_argument_type_error, collect_tuple, compare_as_dataclass, format_dataclass_repr,
    hash_value, read_str,
};
use crate::error::{IntoPyErr, IntoPyResult};
use crate::frozen::build_frozen_mutation_error;
use crate::public_class::PublicClass;
use crate::serialization::{FieldShape, construct_from_decoded_fields, read_payload_fields};

/// The Python module that defines the public classes.
const MODULE: &str = "fhy_core.provenance";

/// The deepest provenance tree that is compared, hashed or rendered without
/// consulting Python's recursion limit.
const SHALLOW_DEPTH: usize = 64;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Matches the Python implementation: `Position.__post_init__`.
impl IntoPyErr for PositionError {
    fn into_py_err(self) -> PyErr {
        match self {
            Self::ZeroLine => PyValueError::new_err("\"line\" must be >= 1, got 0"),
            Self::ZeroColumn => PyValueError::new_err("\"column\" must be >= 1, got 0"),
            _ => PyValueError::new_err(self.to_string()),
        }
    }
}

/// Matches the Python implementation: `Span.__post_init__`.
impl IntoPyErr for SpanError {
    fn into_py_err(self) -> PyErr {
        match self {
            Self::EndOffsetBeforeStart { start, end } => PyValueError::new_err(format!(
                "\"end_offset\" must be >= \"start_offset\", got {end} < {start}"
            )),
            Self::EndPositionBeforeStart { start, end } => PyValueError::new_err(format!(
                "\"end_position\" must be >= \"start_position\", got {end} < {start}"
            )),
            _ => PyValueError::new_err(self.to_string()),
        }
    }
}

/// Matches the Python implementation: `NamedProvenance.__post_init__`.
impl IntoPyErr for NamedProvenanceError {
    fn into_py_err(self) -> PyErr {
        match self {
            Self::EmptyName => PyValueError::new_err("\"name\" must be non-empty"),
            _ => PyValueError::new_err(self.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

/// Return whether `value` is an `int` and not a `bool`.
///
/// Matches the Python implementation: `is_strict_int`.
fn is_strict_int(value: &Bound<'_, PyAny>) -> bool {
    value.is_instance_of::<PyInt>() && !value.is_instance_of::<PyBool>()
}

/// Return `value` rendered as an f-string renders it.
fn format_value(value: &Bound<'_, PyAny>) -> PyResult<String> {
    value
        .call_method1(intern!(value.py(), "__format__"), ("",))?
        .extract()
}

/// Return the `u64` of the strict int `value`, the argument `name`.
///
/// # Errors
///
/// Raises `ValueError` if `value` is below `minimum`, with the Python
/// implementation's message, and `OverflowError` if it exceeds `u64::MAX`,
/// which the Python implementation accepts.
fn read_unsigned(value: &Bound<'_, PyAny>, name: &str, minimum: u64) -> PyResult<u64> {
    match value.extract::<u64>() {
        Ok(unsigned) if unsigned >= minimum => Ok(unsigned),
        Ok(_) => Err(build_below_minimum_error(value, name, minimum)?),
        Err(_) if value.lt(0)? => Err(build_below_minimum_error(value, name, minimum)?),
        Err(_) => Err(PyOverflowError::new_err(format!(
            "\"{name}\" must be at most {}, got {}",
            u64::MAX,
            format_value(value)?
        ))),
    }
}

/// Return the `ValueError` for the argument `name` below `minimum`.
///
/// Matches the Python implementation: `Position.__post_init__` and
/// `Span.__post_init__`.
fn build_below_minimum_error(
    value: &Bound<'_, PyAny>,
    name: &str,
    minimum: u64,
) -> PyResult<PyErr> {
    Ok(PyValueError::new_err(format!(
        "\"{name}\" must be >= {minimum}, got {}",
        format_value(value)?
    )))
}

/// Return `value` as a Python provenance, or raise the `TypeError` naming
/// `owner`, `field` and the `expected` phrase.
fn read_provenance<'a, 'py>(
    value: &'a Bound<'py, PyAny>,
    owner: &str,
    field: &str,
    expected: &str,
) -> PyResult<&'a Bound<'py, PyProvenance>> {
    match value.cast::<PyProvenance>() {
        Ok(provenance) => Ok(provenance),
        Err(_not_a_provenance) => Err(build_argument_type_error(owner, field, expected, value)?),
    }
}

/// Return `value`, an optional argument `field` of `owner`, as a `str`, or
/// `None` for `None`.
///
/// # Errors
///
/// Raises `TypeError` if `value` is neither a `str` nor `None`.
fn read_optional_str<'py>(
    value: Option<&Bound<'py, PyAny>>,
    owner: &str,
    field: &str,
) -> PyResult<Option<Bound<'py, PyString>>> {
    match value {
        None => Ok(None),
        Some(value) => match value.cast::<PyString>() {
            Ok(value) => Ok(Some(value.clone())),
            Err(_not_a_str) => Err(build_argument_type_error(
                owner,
                field,
                "a str or None",
                value,
            )?),
        },
    }
}

/// Return `value`, or Python's `None` for `None`.
fn object_or_none<'py>(py: Python<'py>, value: Option<&Bound<'py, PyAny>>) -> Bound<'py, PyAny> {
    value.map_or_else(|| py.None().into_bound(py), Clone::clone)
}

/// Return the payload of the serializable `value` from its own
/// `serialize_to_dict`, or `None` for `None`.
///
/// Matches the Python implementation: the derived encoding of a
/// serializable field.
fn serialize_nested<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if value.is_none() {
        return Ok(value.clone());
    }
    value.call_method0(intern!(value.py(), "serialize_to_dict"))
}

/// Return the value of `payload` decoded by `class.deserialize_from_dict`,
/// or `None` for `None`.
///
/// Matches the Python implementation: the derived decoding of a
/// serializable field.
fn deserialize_nested<'py>(
    class: &Bound<'py, PyType>,
    payload: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if payload.is_none() {
        return Ok(payload.clone());
    }
    class.call_method1(intern!(class.py(), "deserialize_from_dict"), (payload,))
}

/// Return the dict of the decoded payload fields `fields`, in order.
fn build_fields<'py>(
    py: Python<'py>,
    fields: &[(&str, &Bound<'py, PyAny>)],
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (name, value) in fields {
        dict.set_item(name, value)?;
    }
    Ok(dict)
}

/// Return `pathlib.PurePath`.
fn pure_path_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    crate::python::cached_attr!(py, "pathlib", "PurePath" => PyType)
}

/// Return `pathlib.Path`.
fn path_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    crate::python::cached_attr!(py, "pathlib", "Path" => PyType)
}

// ---------------------------------------------------------------------------
// Position
// ---------------------------------------------------------------------------

/// A 1-indexed line/column position in a source text, backed by the Rust
/// [`Position`].
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Position")]
pub(crate) struct PyPosition {
    position: Position,
    /// The line, the `int` the position was built from.
    #[pyo3(get)]
    line: Py<PyAny>,
    /// The column, the `int` the position was built from.
    #[pyo3(get)]
    column: Py<PyAny>,
}

impl PyPosition {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("Position");
        &PUBLIC_CLASS
    }

    /// Return `value`, the argument `field` of `owner`, as a Rust position,
    /// or `None` for `None`.
    ///
    /// # Errors
    ///
    /// Raises `TypeError` if `value` is neither a `Position` nor `None`.
    fn read_optional(
        value: Option<&Bound<'_, PyAny>>,
        owner: &str,
        field: &str,
    ) -> PyResult<Option<Position>> {
        match value {
            None => Ok(None),
            Some(value) => match value.cast::<Self>() {
                Ok(position) => Ok(Some(position.get().position)),
                Err(_not_a_position) => Err(build_argument_type_error(
                    owner,
                    field,
                    "a Position or None",
                    value,
                )?),
            },
        }
    }

    /// Order `slf` and `other` for the comparison `op`, or return
    /// `NotImplemented` unless both have exactly the same class.
    ///
    /// Matches the Python implementation: the ordering methods of a
    /// dataclass with `order=True`.
    fn compare<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
        op: CompareOp,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        if !slf.as_any().get_type().is(other.get_type()) {
            return Ok(py.NotImplemented().into_bound(py));
        }
        let ordering = slf
            .get()
            .position
            .cmp(&other.cast::<Self>()?.get().position);
        Ok(PyBool::new(py, op.matches(ordering)).to_owned().into_any())
    }
}

#[pymethods]
impl PyPosition {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.line)?;
        visit.call(&self.column)?;
        Ok(())
    }

    /// Create the position at `line` and `column`.
    ///
    /// Raises `TypeError` if either is not a strict `int`, `ValueError` if
    /// either is below 1, and `OverflowError` if either exceeds `2**64 - 1`.
    #[new]
    fn new(line: &Bound<'_, PyAny>, column: &Bound<'_, PyAny>) -> PyResult<Self> {
        for (name, value) in [("line", line), ("column", column)] {
            if !is_strict_int(value) {
                return Err(PyTypeError::new_err(format!(
                    "\"{name}\" must be a strict int, got {}",
                    value.get_type().name()?
                )));
            }
        }
        let line_number = read_unsigned(line, "line", 1)?;
        let column_number = read_unsigned(column, "column", 1)?;
        Ok(Self {
            position: Position::new(line_number, column_number).into_py_result()?,
            line: line.clone().unbind(),
            column: column.clone().unbind(),
        })
    }

    /// Always true: positions order totally.
    #[getter]
    const fn supports_ordering(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Always true: positions order totally, so partially too.
    #[getter]
    const fn supports_partial_ordering(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Always true: positions are immutable.
    #[getter]
    const fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: positions are always frozen.
    const fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: positions are always frozen, and mutating one raises.
    const fn assert_frozen(_slf: &Bound<'_, Self>) {}

    fn __eq__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        compare_as_dataclass(
            slf,
            other,
            |this, other| Ok(this.position == other.position),
        )
    }

    fn __lt__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::compare(slf, other, CompareOp::Lt)
    }

    fn __le__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::compare(slf, other, CompareOp::Le)
    }

    fn __gt__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::compare(slf, other, CompareOp::Gt)
    }

    fn __ge__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Self::compare(slf, other, CompareOp::Ge)
    }

    fn __hash__(&self) -> u64 {
        hash_value(&self.position)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        format_dataclass_repr(
            &slf.get_type(),
            &[
                ("line", this.line.bind(py)),
                ("column", this.column.bind(py)),
            ],
        )
    }

    /// Render `line:column`.
    fn __str__(&self) -> String {
        self.position.to_string()
    }

    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let _ = value;
        Err(build_frozen_mutation_error(slf, "modify", name)?)
    }

    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        Err(build_frozen_mutation_error(slf, "delete", name)?)
    }

    /// Pickle as a constructor call of the position's class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let arguments = PyTuple::new(py, [this.line.bind(py), this.column.bind(py)])?;
        Ok((slf.get_type(), arguments))
    }

    /// Return the payload `{"line": .., "column": ..}`.
    fn serialize_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        build_fields(
            py,
            &[
                ("line", self.line.bind(py)),
                ("column", self.column.bind(py)),
            ],
        )
    }

    /// Return the position of a payload.
    ///
    /// Raises the Python implementation's errors for a malformed payload.
    #[classmethod]
    fn deserialize_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let [line, column] = read_payload_fields(
            cls,
            data,
            [("line", FieldShape::Int), ("column", FieldShape::Int)],
        )?;
        let fields = build_fields(cls.py(), &[("line", &line), ("column", &column)])?;
        construct_from_decoded_fields(cls, &fields)
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
// Span
// ---------------------------------------------------------------------------

/// A file-agnostic byte/position range, backed by the Rust [`Span`].
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Span")]
pub(crate) struct PySpan {
    span: Span,
    /// The start offset, the `int` the span was built from, or `None`.
    #[pyo3(get)]
    start_offset: Py<PyAny>,
    /// The end offset, the `int` the span was built from, or `None`.
    #[pyo3(get)]
    end_offset: Py<PyAny>,
    /// The start position, the `Position` the span was built from, or
    /// `None`.
    #[pyo3(get)]
    start_position: Py<PyAny>,
    /// The end position, the `Position` the span was built from, or `None`.
    #[pyo3(get)]
    end_position: Py<PyAny>,
}

impl PySpan {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("Span");
        &PUBLIC_CLASS
    }

    /// Return `value`, the argument `field` of `owner`, as a Rust span, or
    /// `None` for `None`.
    ///
    /// # Errors
    ///
    /// Raises `TypeError` if `value` is neither a `Span` nor `None`.
    fn read_optional(
        value: Option<&Bound<'_, PyAny>>,
        owner: &str,
        field: &str,
    ) -> PyResult<Option<Span>> {
        match value {
            None => Ok(None),
            Some(value) => match value.cast::<Self>() {
                Ok(span) => Ok(Some(span.get().span)),
                Err(_not_a_span) => Err(build_argument_type_error(
                    owner,
                    field,
                    "a Span or None",
                    value,
                )?),
            },
        }
    }
}

#[pymethods]
impl PySpan {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.start_offset)?;
        visit.call(&self.end_offset)?;
        visit.call(&self.start_position)?;
        visit.call(&self.end_position)?;
        Ok(())
    }

    /// Create the span with the given bounds, each optional.
    ///
    /// Checks the arguments in the Python implementation's order, raising
    /// its `TypeError` for an offset that is not a strict `int` and its
    /// `ValueError` for a negative offset or a pair of bounds out of order.
    /// Raises `OverflowError` for an offset above `2**64 - 1`, and
    /// `TypeError` for a position that is not a `Position`.
    #[new]
    #[pyo3(signature = (
        start_offset = None,
        end_offset = None,
        start_position = None,
        end_position = None,
    ))]
    fn new(
        py: Python<'_>,
        start_offset: Option<&Bound<'_, PyAny>>,
        end_offset: Option<&Bound<'_, PyAny>>,
        start_position: Option<&Bound<'_, PyAny>>,
        end_position: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let offsets = [("start_offset", start_offset), ("end_offset", end_offset)];
        for (name, value) in offsets {
            if let Some(value) = value.filter(|value| !is_strict_int(value)) {
                return Err(PyTypeError::new_err(format!(
                    "\"{name}\" must be a strict int or None, got {}",
                    value.get_type().name()?
                )));
            }
        }
        let [start, end] = offsets
            .map(|(name, value)| value.map(|value| read_unsigned(value, name, 0)).transpose());
        let (start, end) = (start?, end?);
        let mut span = Span::unknown();
        if let Some(start) = start {
            span = span.with_start_offset(start).into_py_result()?;
        }
        if let Some(end) = end {
            span = span.with_end_offset(end).into_py_result()?;
        }
        let start_bound = PyPosition::read_optional(start_position, "Span", "start_position")?;
        let end_bound = PyPosition::read_optional(end_position, "Span", "end_position")?;
        if let Some(start_bound) = start_bound {
            span = span.with_start_position(start_bound).into_py_result()?;
        }
        if let Some(end_bound) = end_bound {
            span = span.with_end_position(end_bound).into_py_result()?;
        }
        Ok(Self {
            span,
            start_offset: object_or_none(py, start_offset).unbind(),
            end_offset: object_or_none(py, end_offset).unbind(),
            start_position: object_or_none(py, start_position).unbind(),
            end_position: object_or_none(py, end_position).unbind(),
        })
    }

    /// Return whether the span carries no offset or position information.
    fn is_unknown(&self) -> bool {
        self.span.is_unknown()
    }

    /// Always true: spans are immutable.
    #[getter]
    const fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: spans are always frozen.
    const fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: spans are always frozen, and mutating one raises.
    const fn assert_frozen(_slf: &Bound<'_, Self>) {}

    fn __eq__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        compare_as_dataclass(slf, other, |this, other| Ok(this.span == other.span))
    }

    fn __hash__(&self) -> u64 {
        hash_value(&self.span)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        format_dataclass_repr(
            &slf.get_type(),
            &[
                ("start_offset", this.start_offset.bind(py)),
                ("end_offset", this.end_offset.bind(py)),
                ("start_position", this.start_position.bind(py)),
                ("end_position", this.end_position.bind(py)),
            ],
        )
    }

    /// Render `<unknown>`, `start-end` from the positions when either is
    /// set, or `@start-end` from the offsets, with `?` for a missing bound.
    fn __str__(&self) -> String {
        self.span.to_string()
    }

    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let _ = value;
        Err(build_frozen_mutation_error(slf, "modify", name)?)
    }

    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        Err(build_frozen_mutation_error(slf, "delete", name)?)
    }

    /// Pickle as a constructor call of the span's class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let arguments = PyTuple::new(
            py,
            [
                this.start_offset.bind(py),
                this.end_offset.bind(py),
                this.start_position.bind(py),
                this.end_position.bind(py),
            ],
        )?;
        Ok((slf.get_type(), arguments))
    }

    /// Return the payload `{"start_offset": .., "end_offset": ..,
    /// "start_position": <position payload>, "end_position": <position
    /// payload>}`, with `None` for an absent bound.
    fn serialize_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        build_fields(
            py,
            &[
                ("start_offset", self.start_offset.bind(py)),
                ("end_offset", self.end_offset.bind(py)),
                (
                    "start_position",
                    &serialize_nested(self.start_position.bind(py))?,
                ),
                (
                    "end_position",
                    &serialize_nested(self.end_position.bind(py))?,
                ),
            ],
        )
    }

    /// Return the span of a payload.
    ///
    /// Raises the Python implementation's errors for a malformed payload.
    #[classmethod]
    fn deserialize_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let [start_offset, end_offset, start_position, end_position] = read_payload_fields(
            cls,
            data,
            [
                ("start_offset", FieldShape::OptionalInt),
                ("end_offset", FieldShape::OptionalInt),
                ("start_position", FieldShape::OptionalPayload),
                ("end_position", FieldShape::OptionalPayload),
            ],
        )?;
        let position_class = PyPosition::public_class().get(py)?;
        let fields = build_fields(
            py,
            &[
                ("start_offset", &start_offset),
                ("end_offset", &end_offset),
                (
                    "start_position",
                    &deserialize_nested(position_class, &start_position)?,
                ),
                (
                    "end_position",
                    &deserialize_nested(position_class, &end_position)?,
                ),
            ],
        )?;
        construct_from_decoded_fields(cls, &fields)
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
// Provenance
// ---------------------------------------------------------------------------

/// Return `fhy_core.provenance._LOGGER`, the module's logger.
fn provenance_logger(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    static LOGGER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    LOGGER
        .get_or_try_init(py, || {
            Ok::<_, PyErr>(py.import(MODULE)?.getattr("_LOGGER")?.unbind())
        })
        .map(|logger| logger.bind(py))
}

/// Origin information for a compiler object, backed by the Rust
/// [`Provenance`]; the base of the variant classes.
///
/// The class itself has no constructor: every provenance is an instance of
/// a variant class, which sets the Rust value.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Provenance")]
pub(crate) struct PyProvenance {
    provenance: Provenance,
    /// The depth of the provenance tree: 1 for a leaf, and otherwise one
    /// more than its deepest child's depth.
    depth: usize,
}

impl PyProvenance {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("Provenance");
        &PUBLIC_CLASS
    }

    /// Return the initializer of a variant instance holding `provenance`,
    /// a tree of depth `depth`.
    fn initializer(provenance: Provenance, depth: usize) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Self { provenance, depth })
    }

    /// Raise `RecursionError` if a tree of depth `depth` is deeper than
    /// Python's recursion limit.
    ///
    /// Equality, hashing and rendering recurse through the tree on the
    /// Rust stack. Refusing the trees Python's own recursion limit would
    /// refuse keeps a deep tree from overflowing that stack, as the Python
    /// implementation raises `RecursionError` for it.
    fn ensure_depth_within_recursion_limit(py: Python<'_>, depth: usize) -> PyResult<()> {
        if depth <= SHALLOW_DEPTH {
            return Ok(());
        }
        let limit: usize = py
            .import(intern!(py, "sys"))?
            .call_method0(intern!(py, "getrecursionlimit"))?
            .extract()?;
        if depth > limit {
            return Err(PyRecursionError::new_err(format!(
                "maximum recursion depth exceeded: the provenance is {depth} levels deep"
            )));
        }
        Ok(())
    }

    /// Render the provenance as the Python implementation's `__str__` does.
    fn render(slf: &Bound<'_, Self>) -> PyResult<String> {
        let this = slf.get();
        Self::ensure_depth_within_recursion_limit(slf.py(), this.depth)?;
        Ok(this.provenance.to_string())
    }

    /// Record the reductions `fuse` made in the module's debug log.
    ///
    /// Matches the Python implementation: `Provenance.fuse`.
    fn log_fuse_reductions(
        py: Python<'_>,
        input_count: usize,
        unknowns_dropped: usize,
        fused_collapsed: usize,
        final_sources: usize,
    ) -> PyResult<()> {
        provenance_logger(py)?.call_method1(
            intern!(py, "debug"),
            (
                "reduced input (inputs=%d, unknowns_dropped=%d, fused_collapsed=%d, \
                 final_sources=%d)",
                input_count,
                unknowns_dropped,
                fused_collapsed,
                final_sources,
            ),
        )?;
        Ok(())
    }
}

#[pymethods]
impl PyProvenance {
    /// Return a new unknown provenance.
    #[staticmethod]
    fn unknown(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        PyUnknownProvenance::public_class().get(py)?.call0()
    }

    /// Return the provenance formed by fusing `provenances`, labelled with
    /// `metadata` unless it is `None`.
    ///
    /// Drops every unknown provenance and splices in the sources of every
    /// unlabelled fused provenance, at any depth, keeping the order. The
    /// result is a new unknown provenance when nothing survives, the single
    /// survivor itself when exactly one survives and `metadata` is `None`,
    /// and otherwise a new `FusedProvenance` of the survivors. Logs the
    /// reductions at debug level as the Python implementation does.
    ///
    /// Raises `TypeError` for an argument that is not a `Provenance`, or a
    /// `metadata` that is neither a `str` nor `None`.
    #[staticmethod]
    #[pyo3(signature = (*provenances, metadata = None))]
    fn fuse<'py>(
        provenances: &Bound<'py, PyTuple>,
        metadata: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = provenances.py();
        let metadata = read_optional_str(metadata, "Provenance.fuse", "metadata")?;
        let mut pending = Vec::with_capacity(provenances.len());
        for provenance in provenances.iter().rev() {
            read_provenance(
                &provenance,
                "Provenance.fuse",
                "provenances",
                "Provenance instances",
            )?;
            pending.push(provenance);
        }
        let mut flat = Vec::with_capacity(pending.len());
        let (mut unknowns_dropped, mut fused_collapsed) = (0, 0);
        while let Some(provenance) = pending.pop() {
            match &provenance.cast::<Self>()?.get().provenance {
                Provenance::Unknown => unknowns_dropped += 1,
                Provenance::Fused(fused) if fused.label().is_none() => {
                    fused_collapsed += 1;
                    let fused = provenance.cast::<PyFusedProvenance>()?;
                    pending.extend(fused.get().sources.bind(py).iter().rev());
                }
                _ => flat.push(provenance),
            }
        }
        if unknowns_dropped > 0 || fused_collapsed > 0 {
            Self::log_fuse_reductions(
                py,
                provenances.len(),
                unknowns_dropped,
                fused_collapsed,
                flat.len(),
            )?;
        }
        if flat.is_empty() {
            return PyUnknownProvenance::public_class().get(py)?.call0();
        }
        if metadata.is_none() && flat.len() == 1 {
            if let Some(survivor) = flat.pop() {
                return Ok(survivor);
            }
        }
        PyFusedProvenance::public_class()
            .get(py)?
            .call1((PyTuple::new(py, flat)?, metadata))
    }

    /// Always true: provenances are immutable.
    #[getter]
    const fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: provenances are always frozen.
    const fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: provenances are always frozen, and mutating one raises.
    const fn assert_frozen(_slf: &Bound<'_, Self>) {}

    /// Compare as a dataclass does: equal when `other` has exactly the same
    /// class and equal fields, compared recursively.
    ///
    /// Raises `RecursionError` for trees deeper than the recursion limit.
    fn __eq__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        compare_as_dataclass(slf, other, |this, other| {
            Self::ensure_depth_within_recursion_limit(py, this.depth.min(other.depth))?;
            Ok(this.provenance == other.provenance)
        })
    }

    /// Hash the provenance tree.
    ///
    /// Raises `RecursionError` for a tree deeper than the recursion limit.
    fn __hash__(&self, py: Python<'_>) -> PyResult<u64> {
        Self::ensure_depth_within_recursion_limit(py, self.depth)?;
        Ok(hash_value(&self.provenance))
    }

    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let _ = value;
        Err(build_frozen_mutation_error(slf, "modify", name)?)
    }

    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        Err(build_frozen_mutation_error(slf, "delete", name)?)
    }

    /// Return the V2 payload, the core's: `{"unknown": {}}`, `{"file":
    /// {"file_path", "span"}}`, `{"named": {"name", "child"}}`,
    /// `{"call_site": {"callee", "caller"}}` or `{"fused": {"sources",
    /// "label"}}`; or the V1 envelope inside `wire_version(WireVersion.V1)`.
    fn serialize_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::wire::write_dict(
            slf.as_any(),
            || crate::wire::write_v1_envelope(slf.as_any()),
            || Ok(slf.get().provenance.clone()),
        )
    }

    /// Return the JSON text of the payload: the canonical V2 text unless
    /// `indent` or `sort_keys` re-formats it or V1 is written.
    #[pyo3(signature = (*, indent = None, sort_keys = None))]
    fn to_json(
        slf: &Bound<'_, Self>,
        indent: Option<&Bound<'_, PyAny>>,
        sort_keys: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        crate::wire::write_json(slf.as_any(), indent, sort_keys, || {
            Ok(slf.get().provenance.clone())
        })
    }

    /// Return the provenance of the payload `data`, an instance of `cls`:
    /// a V2 payload, or a V1 envelope, which warns.
    #[classmethod]
    fn deserialize_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if crate::wire::is_v1_payload(data) {
            return crate::wire::read_v1_envelope(cls, data);
        }
        let provenance: Provenance = crate::wire::parse_dict(cls, data)?;
        crate::wire::check_instance(cls, provenance_to_python(cls.py(), &provenance)?)
    }

    /// Return the provenance of the JSON text `payload`, an instance of
    /// `cls`.
    #[classmethod]
    fn from_json<'py>(
        cls: &Bound<'py, PyType>,
        payload: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::wire::read_json(cls, payload, |text| {
            let provenance: Provenance = crate::wire::parse(cls, text)?;
            crate::wire::check_instance(cls, provenance_to_python(cls.py(), &provenance)?)
        })
    }

    /// Register `cls` as the public class.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }
}

/// Return the Python object of the position `position`.
fn position_to_python(py: Python<'_>, position: Option<Position>) -> PyResult<Bound<'_, PyAny>> {
    match position {
        None => Ok(py.None().into_bound(py)),
        Some(position) => PyPosition::public_class()
            .get(py)?
            .call1((position.line().get(), position.column().get())),
    }
}

/// Return the Python object of the span `span`.
fn span_to_python<'py>(py: Python<'py>, span: Option<&Span>) -> PyResult<Bound<'py, PyAny>> {
    let Some(span) = span else {
        return Ok(py.None().into_bound(py));
    };
    PySpan::public_class().get(py)?.call1((
        span.start_offset(),
        span.end_offset(),
        position_to_python(py, span.start_position())?,
        position_to_python(py, span.end_position())?,
    ))
}

/// Return a new Python object of the core `provenance`, built through the
/// public class of each variant, as a V2 payload decodes.
///
/// # Errors
///
/// Raises what building an object raises.
fn provenance_to_python<'py>(
    py: Python<'py>,
    provenance: &Provenance,
) -> PyResult<Bound<'py, PyAny>> {
    match provenance {
        Provenance::Unknown => PyUnknownProvenance::public_class().get(py)?.call0(),
        Provenance::File(file) => PyFileProvenance::public_class().get(py)?.call1((
            path_class(py)?.call1((file.file_path(),))?,
            span_to_python(py, file.span())?,
        )),
        Provenance::Named(named) => PyNamedProvenance::public_class()
            .get(py)?
            .call1((named.name(), provenance_to_python(py, named.child())?)),
        Provenance::CallSite(call_site) => PyCallSiteProvenance::public_class().get(py)?.call1((
            provenance_to_python(py, call_site.callee())?,
            provenance_to_python(py, call_site.caller())?,
        )),
        Provenance::Fused(fused) => {
            let sources = fused
                .sources()
                .iter()
                .map(|source| provenance_to_python(py, source))
                .collect::<PyResult<Vec<_>>>()?;
            PyFusedProvenance::public_class()
                .get(py)?
                .call1((PyTuple::new(py, sources)?, fused.label()))
        }
    }
}

// ---------------------------------------------------------------------------
// UnknownProvenance
// ---------------------------------------------------------------------------

/// Provenance with no source information, backed by
/// [`Provenance::Unknown`].
#[pyclass(
    extends = PyProvenance,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "UnknownProvenance"
)]
pub(crate) struct PyUnknownProvenance;

impl PyUnknownProvenance {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("UnknownProvenance");
        &PUBLIC_CLASS
    }
}

#[pymethods]
impl PyUnknownProvenance {
    /// Create an unknown provenance.
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyProvenance::initializer(Provenance::Unknown, 1).add_subclass(Self)
    }

    /// Render `<unknown>`.
    fn __str__<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyString> {
        intern!(slf.py(), "<unknown>").clone()
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        format_dataclass_repr(&slf.get_type(), &[])
    }

    /// Pickle as a constructor call of the provenance's class.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> (Bound<'py, PyType>, Bound<'py, PyTuple>) {
        (slf.get_type(), PyTuple::empty(slf.py()))
    }

    /// Return the empty data payload, the envelope's `__data__`.
    fn serialize_data_to_dict<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyDict> {
        PyDict::new(slf.py())
    }

    /// Return the unknown provenance of an empty data payload.
    ///
    /// Raises the Python implementation's errors for a malformed payload.
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let [] = read_payload_fields::<0>(cls, data, [])?;
        construct_from_decoded_fields(cls, &PyDict::new(cls.py()))
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
// FileProvenance
// ---------------------------------------------------------------------------

/// Provenance pointing to a region within a source code file, backed by
/// [`Provenance::File`].
#[pyclass(
    extends = PyProvenance,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "FileProvenance"
)]
pub(crate) struct PyFileProvenance {
    /// The path: the `PurePath` the provenance was built from when it is
    /// in normal form, and otherwise a `pathlib.Path` of the normalized
    /// path.
    #[pyo3(get)]
    file_path: Py<PyAny>,
    /// The span, the `Span` the provenance was built from, or `None`.
    #[pyo3(get)]
    span: Py<PyAny>,
}

impl PyFileProvenance {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("FileProvenance");
        &PUBLIC_CLASS
    }

    /// Return the text of the path `file_path`, and whether it is a
    /// `PurePath`.
    ///
    /// # Errors
    ///
    /// Raises `TypeError` unless `file_path` is a `str`, or an
    /// `os.PathLike` whose `__fspath__` returns a `str`.
    fn read_path_text<'py>(
        file_path: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyString>, bool)> {
        let py = file_path.py();
        if let Ok(text) = file_path.cast::<PyString>() {
            return Ok((text.clone(), false));
        }
        let is_pure_path = file_path.is_instance(pure_path_class(py)?)?;
        let text = if is_pure_path {
            file_path.str()?
        } else if file_path.get_type().hasattr(intern!(py, "__fspath__"))? {
            let text = file_path.call_method0(intern!(py, "__fspath__"))?;
            match text.cast_into::<PyString>() {
                Ok(text) => text,
                Err(error) => {
                    return Err(build_argument_type_error(
                        "FileProvenance",
                        "file_path",
                        "a str or os.PathLike of a str",
                        &error.into_inner(),
                    )?);
                }
            }
        } else {
            return Err(build_argument_type_error(
                "FileProvenance",
                "file_path",
                "a str or os.PathLike",
                file_path,
            )?);
        };
        Ok((text, is_pure_path))
    }
}

#[pymethods]
impl PyFileProvenance {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.file_path)?;
        visit.call(&self.span)?;
        Ok(())
    }

    /// Create the provenance for `span`, if given, in the file at
    /// `file_path`, normalizing the path as `pathlib.PurePosixPath` does.
    ///
    /// Raises `TypeError` if `file_path` is neither a `str` nor an
    /// `os.PathLike` of a `str`, or `span` is neither a `Span` nor `None`.
    #[new]
    #[pyo3(signature = (file_path, span = None))]
    fn new(
        file_path: &Bound<'_, PyAny>,
        span: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let py = file_path.py();
        let (text, is_pure_path) = Self::read_path_text(file_path)?;
        let rust_span = PySpan::read_optional(span, "FileProvenance", "span")?;
        let provenance = FileProvenance::new(text.to_str()?, rust_span);
        let file_path = if is_pure_path && text.to_str()? == provenance.file_path() {
            file_path.clone()
        } else {
            path_class(py)?.call1((provenance.file_path(),))?
        };
        Ok(
            PyProvenance::initializer(Provenance::File(provenance), 1).add_subclass(Self {
                file_path: file_path.unbind(),
                span: object_or_none(py, span).unbind(),
            }),
        )
    }

    /// Render the path, followed by `:` and the span unless the span is
    /// absent or unknown.
    fn __str__(slf: &Bound<'_, Self>) -> PyResult<String> {
        PyProvenance::render(slf.as_super())
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        format_dataclass_repr(
            &slf.get_type(),
            &[
                ("file_path", this.file_path.bind(py)),
                ("span", this.span.bind(py)),
            ],
        )
    }

    /// Pickle as a constructor call of the provenance's class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let arguments = PyTuple::new(py, [this.file_path.bind(py), this.span.bind(py)])?;
        Ok((slf.get_type(), arguments))
    }

    /// Return the data payload `{"file_path": <POSIX path>, "span": <span
    /// payload or None>}`, the envelope's `__data__`.
    fn serialize_data_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let Provenance::File(provenance) = &slf.as_super().get().provenance else {
            unreachable!("a FileProvenance holds a file provenance");
        };
        build_fields(
            py,
            &[
                ("file_path", &PyString::new(py, provenance.file_path())),
                ("span", &serialize_nested(slf.get().span.bind(py))?),
            ],
        )
    }

    /// Return the file provenance of a data payload.
    ///
    /// Raises the Python implementation's errors for a malformed payload.
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let [file_path, span] = read_payload_fields(
            cls,
            data,
            [
                ("file_path", FieldShape::Str),
                ("span", FieldShape::OptionalPayload),
            ],
        )?;
        let fields = build_fields(
            py,
            &[
                ("file_path", &path_class(py)?.call1((file_path,))?),
                (
                    "span",
                    &deserialize_nested(PySpan::public_class().get(py)?, &span)?,
                ),
            ],
        )?;
        construct_from_decoded_fields(cls, &fields)
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
// NamedProvenance
// ---------------------------------------------------------------------------

/// Wraps a child provenance with a human-readable label, backed by
/// [`Provenance::Named`].
#[pyclass(
    extends = PyProvenance,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "NamedProvenance"
)]
pub(crate) struct PyNamedProvenance {
    /// The name, the `str` the provenance was built from.
    #[pyo3(get)]
    name: Py<PyString>,
    /// The child, the `Provenance` the provenance was built from.
    #[pyo3(get)]
    child: Py<PyAny>,
}

impl PyNamedProvenance {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("NamedProvenance");
        &PUBLIC_CLASS
    }
}

#[pymethods]
impl PyNamedProvenance {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.name)?;
        visit.call(&self.child)?;
        Ok(())
    }

    /// Create the provenance naming `child` as `name`.
    ///
    /// Raises `TypeError` if `name` is not a `str` or `child` is not a
    /// `Provenance`, and the Python implementation's `ValueError` if
    /// `name` is empty.
    #[new]
    fn new(
        name: &Bound<'_, PyAny>,
        child: &Bound<'_, PyAny>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let name = read_str(name, "NamedProvenance", "name")?;
        let child_base = read_provenance(child, "NamedProvenance", "child", "a Provenance")?.get();
        let provenance =
            NamedProvenance::new(name.to_str()?, child_base.provenance.clone()).into_py_result()?;
        Ok(PyProvenance::initializer(
            Provenance::Named(provenance),
            child_base.depth.saturating_add(1),
        )
        .add_subclass(Self {
            name: name.clone().unbind(),
            child: child.clone().unbind(),
        }))
    }

    /// Render the name, followed by the child in parentheses unless the
    /// child is unknown.
    fn __str__(slf: &Bound<'_, Self>) -> PyResult<String> {
        PyProvenance::render(slf.as_super())
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        format_dataclass_repr(
            &slf.get_type(),
            &[
                ("name", this.name.bind(py).as_any()),
                ("child", this.child.bind(py)),
            ],
        )
    }

    /// Pickle as a constructor call of the provenance's class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let arguments = PyTuple::new(py, [this.name.bind(py).as_any(), this.child.bind(py)])?;
        Ok((slf.get_type(), arguments))
    }

    /// Return the data payload `{"name": .., "child": <provenance
    /// payload>}`, the envelope's `__data__`.
    fn serialize_data_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        build_fields(
            py,
            &[
                ("name", self.name.bind(py).as_any()),
                ("child", &serialize_nested(self.child.bind(py))?),
            ],
        )
    }

    /// Return the named provenance of a data payload.
    ///
    /// Raises the Python implementation's errors for a malformed payload.
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let [name, child] = read_payload_fields(
            cls,
            data,
            [("name", FieldShape::Str), ("child", FieldShape::Payload)],
        )?;
        let provenance_class = PyProvenance::public_class().get(py)?;
        let fields = build_fields(
            py,
            &[
                ("name", &name),
                ("child", &deserialize_nested(provenance_class, &child)?),
            ],
        )?;
        construct_from_decoded_fields(cls, &fields)
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
// CallSiteProvenance
// ---------------------------------------------------------------------------

/// Provenance for a value created at a call site, backed by
/// [`Provenance::CallSite`].
#[pyclass(
    extends = PyProvenance,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "CallSiteProvenance"
)]
pub(crate) struct PyCallSiteProvenance {
    /// The callee, the `Provenance` the provenance was built from.
    #[pyo3(get)]
    callee: Py<PyAny>,
    /// The caller, the `Provenance` the provenance was built from.
    #[pyo3(get)]
    caller: Py<PyAny>,
}

impl PyCallSiteProvenance {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("CallSiteProvenance");
        &PUBLIC_CLASS
    }
}

#[pymethods]
impl PyCallSiteProvenance {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.callee)?;
        visit.call(&self.caller)?;
        Ok(())
    }

    /// Create the provenance of a value from `callee` created at `caller`.
    ///
    /// Raises `TypeError` if either is not a `Provenance`.
    #[expect(
        clippy::similar_names,
        reason = "callee and caller are the standard names for the two ends of a call"
    )]
    #[new]
    fn new(
        callee: &Bound<'_, PyAny>,
        caller: &Bound<'_, PyAny>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let callee_base =
            read_provenance(callee, "CallSiteProvenance", "callee", "a Provenance")?.get();
        let caller_base =
            read_provenance(caller, "CallSiteProvenance", "caller", "a Provenance")?.get();
        let provenance = CallSiteProvenance::new(
            callee_base.provenance.clone(),
            caller_base.provenance.clone(),
        );
        Ok(PyProvenance::initializer(
            Provenance::CallSite(provenance),
            callee_base.depth.max(caller_base.depth).saturating_add(1),
        )
        .add_subclass(Self {
            callee: callee.clone().unbind(),
            caller: caller.clone().unbind(),
        }))
    }

    /// Render `callee at caller`.
    fn __str__(slf: &Bound<'_, Self>) -> PyResult<String> {
        PyProvenance::render(slf.as_super())
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        format_dataclass_repr(
            &slf.get_type(),
            &[
                ("callee", this.callee.bind(py)),
                ("caller", this.caller.bind(py)),
            ],
        )
    }

    /// Pickle as a constructor call of the provenance's class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let arguments = PyTuple::new(py, [this.callee.bind(py), this.caller.bind(py)])?;
        Ok((slf.get_type(), arguments))
    }

    /// Return the data payload `{"callee": <provenance payload>, "caller":
    /// <provenance payload>}`, the envelope's `__data__`.
    fn serialize_data_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        build_fields(
            py,
            &[
                ("callee", &serialize_nested(self.callee.bind(py))?),
                ("caller", &serialize_nested(self.caller.bind(py))?),
            ],
        )
    }

    /// Return the call-site provenance of a data payload.
    ///
    /// Raises the Python implementation's errors for a malformed payload.
    #[expect(
        clippy::similar_names,
        reason = "callee and caller are the standard names for the two ends of a call"
    )]
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let [callee, caller] = read_payload_fields(
            cls,
            data,
            [
                ("callee", FieldShape::Payload),
                ("caller", FieldShape::Payload),
            ],
        )?;
        let provenance_class = PyProvenance::public_class().get(py)?;
        let fields = build_fields(
            py,
            &[
                ("callee", &deserialize_nested(provenance_class, &callee)?),
                ("caller", &deserialize_nested(provenance_class, &caller)?),
            ],
        )?;
        construct_from_decoded_fields(cls, &fields)
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
// FusedProvenance
// ---------------------------------------------------------------------------

/// N provenances combined by a transformation, backed by
/// [`Provenance::Fused`].
#[pyclass(
    extends = PyProvenance,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "FusedProvenance"
)]
pub(crate) struct PyFusedProvenance {
    /// The sources, as a tuple of the `Provenance`s the provenance was
    /// built from.
    #[pyo3(get)]
    sources: Py<PyTuple>,
    /// The label, the `str` the provenance was built from, or `None`.
    #[pyo3(get)]
    metadata: Py<PyAny>,
}

impl PyFusedProvenance {
    /// Return the public Python class registered for this class.
    fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("FusedProvenance");
        &PUBLIC_CLASS
    }
}

#[pymethods]
impl PyFusedProvenance {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.sources)?;
        visit.call(&self.metadata)?;
        Ok(())
    }

    /// Create the fusion of `sources`, any iterable of provenances kept in
    /// order as given, labelled with `metadata` unless it is `None`.
    ///
    /// Raises `TypeError` if a source is not a `Provenance`, `sources` is
    /// not iterable, or `metadata` is neither a `str` nor `None`.
    #[new]
    #[pyo3(signature = (sources, metadata = None))]
    fn new(
        sources: &Bound<'_, PyAny>,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let py = sources.py();
        let sources = collect_tuple(sources)?;
        let mut rust_sources = Vec::with_capacity(sources.len());
        let mut depth = 0;
        for source in &sources {
            let source = read_provenance(
                &source,
                "FusedProvenance",
                "sources",
                "Provenance instances",
            )?
            .get();
            rust_sources.push(source.provenance.clone());
            depth = depth.max(source.depth);
        }
        let label = read_optional_str(metadata, "FusedProvenance", "metadata")?;
        let provenance = match &label {
            None => FusedProvenance::new(rust_sources),
            Some(label) => FusedProvenance::labelled(rust_sources, label.to_str()?),
        };
        Ok(
            PyProvenance::initializer(Provenance::Fused(provenance), depth.saturating_add(1))
                .add_subclass(Self {
                    sources: sources.unbind(),
                    metadata: object_or_none(py, label.as_ref().map(Bound::as_any)).unbind(),
                }),
        )
    }

    /// Render `label[source, ...]`, with the label `fused` when there is
    /// no metadata.
    fn __str__(slf: &Bound<'_, Self>) -> PyResult<String> {
        PyProvenance::render(slf.as_super())
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        format_dataclass_repr(
            &slf.get_type(),
            &[
                ("sources", this.sources.bind(py).as_any()),
                ("metadata", this.metadata.bind(py)),
            ],
        )
    }

    /// Pickle as a constructor call of the provenance's class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let arguments = PyTuple::new(py, [this.sources.bind(py).as_any(), this.metadata.bind(py)])?;
        Ok((slf.get_type(), arguments))
    }

    /// Return the data payload `{"sources": [<provenance payload>, ...],
    /// "metadata": ..}`, the envelope's `__data__`.
    fn serialize_data_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let sources = self
            .sources
            .bind(py)
            .iter()
            .map(|source| serialize_nested(&source))
            .collect::<PyResult<Vec<_>>>()?;
        build_fields(
            py,
            &[
                ("sources", PyList::new(py, sources)?.as_any()),
                ("metadata", self.metadata.bind(py)),
            ],
        )
    }

    /// Return the fused provenance of a data payload.
    ///
    /// Raises the Python implementation's errors for a malformed payload.
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let [sources, metadata] = read_payload_fields(
            cls,
            data,
            [
                ("sources", FieldShape::PayloadList),
                ("metadata", FieldShape::OptionalStr),
            ],
        )?;
        let provenance_class = PyProvenance::public_class().get(py)?;
        let sources = sources
            .try_iter()?
            .map(|source| deserialize_nested(provenance_class, &source?))
            .collect::<PyResult<Vec<_>>>()?;
        let fields = build_fields(
            py,
            &[
                ("sources", PyTuple::new(py, sources)?.as_any()),
                ("metadata", &metadata),
            ],
        )?;
        construct_from_decoded_fields(cls, &fields)
    }

    /// Register `cls` as the public class.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }
}
