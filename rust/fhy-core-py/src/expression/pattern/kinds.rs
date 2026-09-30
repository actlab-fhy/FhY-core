//! The pattern classes: `fhy_core._rs.Pattern` and its eleven kinds, the
//! bases of the classes of the same names in
//! `fhy_core.symbolic.expression.pattern.core`, built as a closed class
//! hierarchy.
//!
//! `Pattern` holds the Rust [`Pattern`], the depth of the pattern, the
//! field objects of its kind in constructor order, with their names, and
//! its sub-pattern objects in matching order. Each kind extends it with its
//! field objects, read as struct members. Equality, hashing, `repr` and
//! pickling follow the dataclass rules over the field objects, so a capture
//! compares by identity and a predicate by the callable's `==`.

use std::collections::HashSet;
use std::sync::Arc;

use pyo3::exceptions::{PyRecursionError, PyTypeError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyString, PyTuple, PyType};

use fhy_core::expression::pattern::Pattern;
use fhy_core::expression::{BinaryOperation, Callee, LogicalOperation, UnaryOperation};
use fhy_core::foreign::BoxError;

use crate::dataclass::{
    OptionalArgument, build_argument_type_error, collect_tuple, compare_as_dataclass,
    format_dataclass_repr, read_str,
};
use crate::error::IntoPyResult;
use crate::frozen::{refuse_attribute_assignment, refuse_attribute_deletion};
use crate::identifier::restore_identifier;
use crate::public_class::PublicClass;

use super::super::literal::read_literal;
use super::super::node::read_expression;
use super::super::operation::{PythonOperation, operation_from_python};
use super::bindings::{CaptureObjects, PyMatchBindings};
use super::capture::PyCapture;
use super::objects::{ActiveTable, current_object_of};

/// The deepest pattern matched without consulting Python's recursion limit.
const SHALLOW_DEPTH: usize = 64;

/// Return the callback error carrying the Python exception `error`.
pub(super) fn into_callback_error(error: PyErr) -> BoxError {
    Box::new(error)
}

/// Raise `RecursionError` if a pattern of depth `depth` is deeper than
/// Python's recursion limit.
///
/// The core matches a pattern recursively, once per level, on the Rust
/// stack. Refusing the patterns Python's own recursion limit would refuse
/// keeps a deep pattern from overflowing that stack.
pub(super) fn ensure_depth_within_recursion_limit(py: Python<'_>, depth: usize) -> PyResult<()> {
    if depth <= SHALLOW_DEPTH {
        return Ok(());
    }
    let limit: usize = py
        .import(intern!(py, "sys"))?
        .call_method0(intern!(py, "getrecursionlimit"))?
        .extract()?;
    if depth > limit {
        return Err(PyRecursionError::new_err(format!(
            "maximum recursion depth exceeded: the pattern is {depth} levels deep"
        )));
    }
    Ok(())
}

/// Return `value` as a pattern, or raise the `TypeError` naming `owner` and
/// `field`.
fn read_pattern<'a, 'py>(
    value: &'a Bound<'py, PyAny>,
    owner: &str,
    field: &str,
) -> PyResult<&'a Bound<'py, PyPattern>> {
    match value.cast::<PyPattern>() {
        Ok(pattern) => Ok(pattern),
        Err(_not_a_pattern) => Err(build_argument_type_error(owner, field, "a Pattern", value)?),
    }
}

/// Return the items of the iterable `values` as a tuple of patterns, with
/// the pattern objects, or raise the `TypeError` naming `owner` and `field`.
fn read_patterns<'py>(
    values: &Bound<'py, PyAny>,
    owner: &str,
    field: &str,
) -> PyResult<(Bound<'py, PyTuple>, Vec<Bound<'py, PyPattern>>)> {
    let values = collect_tuple(values)?;
    let patterns = values
        .iter()
        .map(|value| read_pattern(&value, owner, field).cloned())
        .collect::<PyResult<Vec<_>>>()?;
    Ok((values, patterns))
}

/// Return the Rust patterns of `patterns`.
fn rust_patterns(patterns: &[Bound<'_, PyPattern>]) -> Vec<Pattern> {
    patterns
        .iter()
        .map(|pattern| pattern.get().pattern.clone())
        .collect()
}

/// Return the Rust operation of `value` and its member, or `None` for
/// `None`, which leaves the operation unconstrained.
fn read_optional_operation<'py, T: PythonOperation>(
    value: &Bound<'py, PyAny>,
) -> PyResult<(Option<T>, Bound<'py, PyAny>)> {
    if value.is_none() {
        return Ok((None, value.clone()));
    }
    let (operation, member) = operation_from_python::<T>(value)?;
    Ok((Some(operation), member))
}

/// Return the tuple of `objects`.
fn build_fields<'py>(
    py: Python<'py>,
    objects: &[&Bound<'py, PyAny>],
) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(py, objects)
}

// ---------------------------------------------------------------------------
// Pattern
// ---------------------------------------------------------------------------

/// A shape an expression may have, with captures, backed by the Rust
/// [`Pattern`]; the base of the pattern kinds.
///
/// The class itself has no constructor: every pattern is an instance of a
/// kind, which sets the Rust pattern.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Pattern")]
pub(crate) struct PyPattern {
    pattern: Pattern,
    /// The number of pattern levels, this one included.
    depth: usize,
    /// The field objects, in constructor order.
    fields: Py<PyTuple>,
    /// The field names, paired with `fields`.
    field_names: &'static [&'static str],
    /// The sub-pattern objects, in matching order.
    sub_patterns: Vec<Py<Self>>,
    /// The capture a capture pattern binds.
    capture: Option<Py<PyCapture>>,
    /// The capture object of every Rust capture in the pattern, collected
    /// on first need.
    captures: PyOnceLock<Arc<CaptureObjects>>,
}

impl PyPattern {
    /// Return the initializer of a pattern of `pattern`, with the field
    /// objects `fields` named `field_names`, the sub-patterns
    /// `sub_patterns` and, for a capture pattern, its `capture`.
    fn initializer(
        pattern: Pattern,
        fields: Bound<'_, PyTuple>,
        field_names: &'static [&'static str],
        sub_patterns: &[Bound<'_, Self>],
        capture: Option<Py<PyCapture>>,
    ) -> PyClassInitializer<Self> {
        let depth = 1 + sub_patterns
            .iter()
            .map(|sub_pattern| sub_pattern.get().depth)
            .max()
            .unwrap_or(0);
        PyClassInitializer::from(Self {
            pattern,
            depth,
            fields: fields.unbind(),
            field_names,
            sub_patterns: sub_patterns
                .iter()
                .map(|sub_pattern| sub_pattern.clone().unbind())
                .collect(),
            capture,
            captures: PyOnceLock::new(),
        })
    }

    /// Return the Rust pattern.
    pub(super) const fn pattern(&self) -> &Pattern {
        &self.pattern
    }

    /// Return the number of pattern levels.
    pub(super) const fn depth(&self) -> usize {
        self.depth
    }

    /// Return the capture object of every Rust capture in the pattern.
    pub(super) fn captures(&self, py: Python<'_>) -> PyResult<&Arc<CaptureObjects>> {
        self.captures.get_or_try_init(py, || {
            let mut captures = CaptureObjects::new();
            let mut visited = HashSet::new();
            let mut pending: Vec<Bound<'_, Self>> = self
                .sub_patterns
                .iter()
                .map(|sub_pattern| sub_pattern.bind(py).clone())
                .collect();
            if let Some(capture) = &self.capture {
                captures.insert(capture.get().capture().clone(), capture.clone_ref(py));
            }
            while let Some(pattern) = pending.pop() {
                if !visited.insert(pattern.as_ptr() as usize) {
                    continue;
                }
                let this = pattern.get();
                if let Some(capture) = &this.capture {
                    captures
                        .entry(capture.get().capture().clone())
                        .or_insert_with(|| capture.clone_ref(py));
                }
                pending.extend(
                    this.sub_patterns
                        .iter()
                        .map(|sub_pattern| sub_pattern.bind(py).clone()),
                );
            }
            Ok(Arc::new(captures))
        })
    }
}

#[pymethods]
impl PyPattern {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.fields)?;
        crate::gc::traverse_all(&visit, &self.sub_patterns)?;
        // `captures` holds `Capture` objects, which hold only their names,
        // and can be read only with the interpreter attached.
        visit.call(self.capture.as_ref())
    }

    /// Match `expression` at the root, and return the captures, or `None`
    /// if it does not match.
    ///
    /// Raises `TypeError` for a value that is not an expression,
    /// `RecursionError` for a pattern deeper than the recursion limit, and
    /// the exception a predicate raises, unchanged.
    #[pyo3(name = "match")]
    fn match_expression<'py>(
        &self,
        expression: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let py = expression.py();
        let expression = read_expression(expression, "Pattern.match", "expression")?;
        ensure_depth_within_recursion_limit(py, self.depth)?;
        let table = ActiveTable::enter(expression);
        let bindings = self
            .pattern
            .matches(expression.get().expression())
            .map_err(crate::exceptions::boxed_error_to_py)?;
        let result = match bindings {
            Some(bindings) => Some(PyMatchBindings::build(py, &bindings, self.captures(py)?)?),
            None => None,
        };
        drop(table);
        Ok(result)
    }

    /// Return whether `other` has exactly the class of this pattern and
    /// equal fields, as a dataclass compares.
    fn __eq__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        compare_as_dataclass(slf, other, |this, other| {
            this.fields.bind(py).eq(other.fields.bind(py))
        })
    }

    /// Return the hash of the field objects, as a dataclass hashes.
    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        self.fields.bind(py).hash()
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let this = slf.get();
        let fields = this.fields.bind(py);
        let pairs = this
            .field_names
            .iter()
            .zip(fields.iter())
            .collect::<Vec<_>>();
        let named = pairs
            .iter()
            .map(|(name, value)| (**name, value))
            .collect::<Vec<_>>();
        format_dataclass_repr(&slf.get_type(), &named)
    }

    /// Pickle as a call of the pattern's class with its fields.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> (Bound<'py, PyType>, Bound<'py, PyTuple>) {
        (slf.get_type(), slf.get().fields.bind(slf.py()).clone())
    }

    /// Always true: patterns are immutable.
    #[getter]
    const fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: patterns are always frozen.
    const fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: patterns are always frozen, and mutating one raises.
    const fn assert_frozen(_slf: &Bound<'_, Self>) {}

    fn __setattr__(slf: &Bound<'_, Self>, name: &str, _value: &Bound<'_, PyAny>) -> PyResult<()> {
        refuse_attribute_assignment(slf, name)
    }

    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        refuse_attribute_deletion(slf, name)
    }
}

/// Implement `public_class` for a pattern kind and its
/// `_register_public_class` class method, in its own `impl` blocks.
macro_rules! impl_public_class {
    ($class:ty, $name:literal) => {
        impl $class {
            /// Return the public Python class registered for this class.
            fn public_class() -> &'static PublicClass {
                static PUBLIC_CLASS: PublicClass = PublicClass::new($name);
                &PUBLIC_CLASS
            }
        }
    };
}

// ---------------------------------------------------------------------------
// WildcardPattern
// ---------------------------------------------------------------------------

/// The pattern every expression matches, capturing nothing.
#[pyclass(
    extends = PyPattern,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "WildcardPattern"
)]
pub(crate) struct PyWildcardPattern;

impl_public_class!(PyWildcardPattern, "WildcardPattern");

impl PyWildcardPattern {
    /// Return a new wildcard pattern of the public class.
    fn build(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        Self::public_class().get(py)?.call0()
    }
}

#[pymethods]
impl PyWildcardPattern {
    /// Create the pattern every expression matches.
    #[new]
    fn new(py: Python<'_>) -> PyClassInitializer<Self> {
        PyPattern::initializer(Pattern::wildcard(), PyTuple::empty(py), &[], &[], None)
            .add_subclass(Self)
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
// CapturePattern
// ---------------------------------------------------------------------------

/// The pattern matching what `sub_pattern` matches and binding `capture` to
/// the matched expression.
///
/// The capture binds after the sub-pattern matched. When the capture is
/// bound already, the match succeeds only for a structurally equal
/// expression, and the first-bound one is kept.
#[pyclass(
    extends = PyPattern,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "CapturePattern"
)]
pub(crate) struct PyCapturePattern {
    /// The `Capture` bound.
    #[pyo3(get)]
    capture: Py<PyAny>,
    /// The pattern the expression must match first.
    #[pyo3(get)]
    sub_pattern: Py<PyAny>,
}

impl_public_class!(PyCapturePattern, "CapturePattern");

#[pymethods]
impl PyCapturePattern {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.capture)?;
        visit.call(&self.sub_pattern)?;
        Ok(())
    }

    /// Create the pattern binding `capture`, a `Capture`, to what
    /// `sub_pattern` matches, a new `WildcardPattern` when omitted.
    ///
    /// Raises `TypeError` for a capture that is not a `Capture` or a
    /// sub-pattern that is not a `Pattern`.
    #[new]
    #[pyo3(signature = (capture, sub_pattern = OptionalArgument::Omitted))]
    fn new(
        capture: &Bound<'_, PyAny>,
        sub_pattern: OptionalArgument<'_>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let py = capture.py();
        let Ok(rust_capture) = capture.cast::<PyCapture>() else {
            return Err(build_argument_type_error(
                "CapturePattern",
                "capture",
                "a Capture",
                capture,
            )?);
        };
        let sub_pattern = match sub_pattern {
            OptionalArgument::Omitted => PyWildcardPattern::build(py)?,
            OptionalArgument::Given(sub_pattern) => sub_pattern,
        };
        let sub = read_pattern(&sub_pattern, "CapturePattern", "sub_pattern")?.clone();
        let pattern = sub
            .get()
            .pattern
            .clone()
            .captured_as(rust_capture.get().capture());
        let fields = build_fields(py, &[capture, &sub_pattern])?;
        Ok(PyPattern::initializer(
            pattern,
            fields,
            &["capture", "sub_pattern"],
            &[sub],
            Some(rust_capture.clone().unbind()),
        )
        .add_subclass(Self {
            capture: capture.clone().unbind(),
            sub_pattern: sub_pattern.unbind(),
        }))
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
// LiteralPattern
// ---------------------------------------------------------------------------

/// The pattern matching a literal: any, or one equal to
/// `LiteralExpression(value)`.
#[pyclass(
    extends = PyPattern,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "LiteralPattern"
)]
pub(crate) struct PyLiteralPattern {
    /// The value as given, or `None` for any literal.
    #[pyo3(get)]
    value: Py<PyAny>,
}

impl_public_class!(PyLiteralPattern, "LiteralPattern");

#[pymethods]
impl PyLiteralPattern {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.value)?;
        Ok(())
    }

    /// Create the pattern matching every literal for `None`, and otherwise
    /// the literals equal to `LiteralExpression(value)`.
    ///
    /// Raises `LiteralExpression`'s errors for a value it refuses.
    #[new]
    #[pyo3(signature = (value = None))]
    fn new(py: Python<'_>, value: Option<&Bound<'_, PyAny>>) -> PyResult<PyClassInitializer<Self>> {
        let value = value.map_or_else(|| py.None().into_bound(py), Clone::clone);
        let pattern = if value.is_none() {
            Pattern::any_literal()
        } else {
            Pattern::literal(read_literal(&value)?.0)
        };
        let fields = build_fields(py, &[&value])?;
        Ok(
            PyPattern::initializer(pattern, fields, &["value"], &[], None).add_subclass(Self {
                value: value.unbind(),
            }),
        )
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
// IdentifierPattern
// ---------------------------------------------------------------------------

/// The pattern matching a reference: any, or one to an identifier with the
/// same id.
#[pyclass(
    extends = PyPattern,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "IdentifierPattern"
)]
pub(crate) struct PyIdentifierPattern {
    /// The `Identifier`, or `None` for any reference.
    #[pyo3(get)]
    identifier: Py<PyAny>,
}

impl_public_class!(PyIdentifierPattern, "IdentifierPattern");

#[pymethods]
impl PyIdentifierPattern {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.identifier)?;
        Ok(())
    }

    /// Create the pattern matching every reference for `None`, and
    /// otherwise the references to `identifier`, an `Identifier`.
    ///
    /// Raises `TypeError` for a value that is neither.
    #[new]
    #[pyo3(signature = (identifier = None))]
    fn new(
        py: Python<'_>,
        identifier: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let identifier = identifier.map_or_else(|| py.None().into_bound(py), Clone::clone);
        let pattern = if identifier.is_none() {
            Pattern::any_identifier()
        } else {
            Pattern::identifier(restore_identifier(
                &identifier,
                "IdentifierPattern",
                "identifier",
            )?)
        };
        let fields = build_fields(py, &[&identifier])?;
        Ok(
            PyPattern::initializer(pattern, fields, &["identifier"], &[], None).add_subclass(
                Self {
                    identifier: identifier.unbind(),
                },
            ),
        )
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
// UnaryExpressionPattern
// ---------------------------------------------------------------------------

/// The pattern matching a unary node whose operand matches `operand`.
#[pyclass(
    extends = PyPattern,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "UnaryExpressionPattern"
)]
pub(crate) struct PyUnaryExpressionPattern {
    /// The `UnaryOperation`, or `None` for any operation.
    #[pyo3(get)]
    operation: Py<PyAny>,
    /// The pattern the operand must match.
    #[pyo3(get)]
    operand: Py<PyAny>,
}

impl_public_class!(PyUnaryExpressionPattern, "UnaryExpressionPattern");

#[pymethods]
impl PyUnaryExpressionPattern {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.operation)?;
        visit.call(&self.operand)?;
        Ok(())
    }

    /// Create the pattern of a unary node of `operation`, any for `None`,
    /// whose operand matches `operand`.
    ///
    /// Raises `ValueError` for an operation that is not a `UnaryOperation`,
    /// and `TypeError` for an operand that is not a `Pattern`.
    #[new]
    fn new(
        operation: &Bound<'_, PyAny>,
        operand: &Bound<'_, PyAny>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let py = operation.py();
        let (rust_operation, member) = read_optional_operation::<UnaryOperation>(operation)?;
        let sub = read_pattern(operand, "UnaryExpressionPattern", "operand")?.clone();
        let sub_pattern = sub.get().pattern.clone();
        let pattern = match rust_operation {
            Some(operation) => Pattern::unary(operation, sub_pattern),
            None => Pattern::unary_any_operation(sub_pattern),
        };
        let fields = build_fields(py, &[&member, operand])?;
        Ok(
            PyPattern::initializer(pattern, fields, &["operation", "operand"], &[sub], None)
                .add_subclass(Self {
                    operation: member.unbind(),
                    operand: operand.clone().unbind(),
                }),
        )
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
// BinaryExpressionPattern
// ---------------------------------------------------------------------------

/// The pattern matching a binary node whose left operand matches `left`
/// and whose right operand then matches `right`.
#[pyclass(
    extends = PyPattern,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "BinaryExpressionPattern"
)]
pub(crate) struct PyBinaryExpressionPattern {
    /// The `BinaryOperation`, or `None` for any operation.
    #[pyo3(get)]
    operation: Py<PyAny>,
    /// The pattern the left operand must match.
    #[pyo3(get)]
    left: Py<PyAny>,
    /// The pattern the right operand must match.
    #[pyo3(get)]
    right: Py<PyAny>,
}

impl_public_class!(PyBinaryExpressionPattern, "BinaryExpressionPattern");

#[pymethods]
impl PyBinaryExpressionPattern {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.operation)?;
        visit.call(&self.left)?;
        visit.call(&self.right)?;
        Ok(())
    }

    /// Create the pattern of a binary node of `operation`, any for `None`,
    /// whose operands match `left` and `right`.
    ///
    /// Raises `ValueError` for an operation that is not a
    /// `BinaryOperation`, and `TypeError` for an operand pattern that is not
    /// a `Pattern`.
    #[new]
    fn new(
        operation: &Bound<'_, PyAny>,
        left: &Bound<'_, PyAny>,
        right: &Bound<'_, PyAny>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let py = operation.py();
        let (rust_operation, member) = read_optional_operation::<BinaryOperation>(operation)?;
        let left_sub = read_pattern(left, "BinaryExpressionPattern", "left")?.clone();
        let right_sub = read_pattern(right, "BinaryExpressionPattern", "right")?.clone();
        let (left_pattern, right_pattern) = (
            left_sub.get().pattern.clone(),
            right_sub.get().pattern.clone(),
        );
        let pattern = match rust_operation {
            Some(operation) => Pattern::binary(operation, left_pattern, right_pattern),
            None => Pattern::binary_any_operation(left_pattern, right_pattern),
        };
        let fields = build_fields(py, &[&member, left, right])?;
        Ok(PyPattern::initializer(
            pattern,
            fields,
            &["operation", "left", "right"],
            &[left_sub, right_sub],
            None,
        )
        .add_subclass(Self {
            operation: member.unbind(),
            left: left.clone().unbind(),
            right: right.clone().unbind(),
        }))
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
// LogicalExpressionPattern
// ---------------------------------------------------------------------------

/// The pattern matching a logical node: its operands, when given, one to
/// one and in order.
///
/// A logical node has at least two operands, so fewer than two operand
/// patterns match nothing.
#[pyclass(
    extends = PyPattern,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "LogicalExpressionPattern"
)]
pub(crate) struct PyLogicalExpressionPattern {
    /// The `LogicalOperation`, or `None` for either connective.
    #[pyo3(get)]
    operation: Py<PyAny>,
    /// The tuple of operand patterns, or `None` for any operands.
    #[pyo3(get)]
    operands: Py<PyAny>,
}

impl_public_class!(PyLogicalExpressionPattern, "LogicalExpressionPattern");

#[pymethods]
impl PyLogicalExpressionPattern {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.operation)?;
        visit.call(&self.operands)?;
        Ok(())
    }

    /// Create the pattern of a logical node of `operation`, either for
    /// `None`, whose operands match the iterable of patterns `operands`
    /// position-wise, or are any for `None`.
    ///
    /// Raises `ValueError` for an operation that is not a
    /// `LogicalOperation`, and `TypeError` for an operand pattern that is
    /// not a `Pattern`.
    #[new]
    fn new(
        operation: &Bound<'_, PyAny>,
        operands: &Bound<'_, PyAny>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let py = operation.py();
        let (rust_operation, member) = read_optional_operation::<LogicalOperation>(operation)?;
        let (operands, subs) = if operands.is_none() {
            (operands.clone(), Vec::new())
        } else {
            let (tuple, subs) = read_patterns(operands, "LogicalExpressionPattern", "operands")?;
            (tuple.into_any(), subs)
        };
        let pattern = match (rust_operation, operands.is_none()) {
            (Some(operation), true) => Pattern::logical_any_operands(operation),
            (None, true) => Pattern::any_logical(),
            (Some(operation), false) => Pattern::logical(operation, rust_patterns(&subs)),
            (None, false) => Pattern::logical_any_operation(rust_patterns(&subs)),
        };
        let fields = build_fields(py, &[&member, &operands])?;
        Ok(
            PyPattern::initializer(pattern, fields, &["operation", "operands"], &subs, None)
                .add_subclass(Self {
                    operation: member.unbind(),
                    operands: operands.unbind(),
                }),
        )
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
// PiecewiseExpressionPattern
// ---------------------------------------------------------------------------

/// The pattern matching a piecewise node: its cases, when given, one to one
/// and in order, each condition then value, and then its otherwise branch.
///
/// A piecewise node has at least one case, so no case pattern matches
/// nothing.
#[pyclass(
    extends = PyPattern,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "PiecewiseExpressionPattern"
)]
pub(crate) struct PyPiecewiseExpressionPattern {
    /// The tuple of `(condition, value)` pattern pairs, or `None` for any
    /// cases.
    #[pyo3(get)]
    cases: Py<PyAny>,
    /// The pattern the otherwise branch must match.
    #[pyo3(get)]
    otherwise: Py<PyAny>,
}

impl_public_class!(PyPiecewiseExpressionPattern, "PiecewiseExpressionPattern");

#[pymethods]
impl PyPiecewiseExpressionPattern {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.cases)?;
        visit.call(&self.otherwise)?;
        Ok(())
    }

    /// Create the pattern of a piecewise node whose cases match the
    /// iterable of `(condition, value)` pattern pairs `cases`
    /// position-wise, or are any for `None`, and whose otherwise branch
    /// matches `otherwise`.
    ///
    /// Raises `TypeError` for a case that is not a pair of patterns, or an
    /// otherwise pattern that is not a `Pattern`.
    #[new]
    fn new(
        cases: &Bound<'_, PyAny>,
        otherwise: &Bound<'_, PyAny>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let py = otherwise.py();
        let mut subs = Vec::new();
        let mut rust_cases = Vec::new();
        let cases = if cases.is_none() {
            cases.clone()
        } else {
            let mut pairs = Vec::new();
            for case in collect_tuple(cases)?.iter() {
                let pair = match collect_tuple(&case) {
                    Ok(pair) if pair.len() == 2 => pair,
                    _ => {
                        return Err(build_argument_type_error(
                            "PiecewiseExpressionPattern",
                            "cases",
                            "(condition, value) pairs of Patterns",
                            &case,
                        )?);
                    }
                };
                let condition = read_pattern(
                    &pair.get_item(0)?,
                    "PiecewiseExpressionPattern",
                    "case conditions",
                )?
                .clone();
                let value = read_pattern(
                    &pair.get_item(1)?,
                    "PiecewiseExpressionPattern",
                    "case values",
                )?
                .clone();
                rust_cases.push((condition.get().pattern.clone(), value.get().pattern.clone()));
                subs.push(condition);
                subs.push(value);
                pairs.push(pair);
            }
            PyTuple::new(py, pairs)?.into_any()
        };
        let otherwise_sub =
            read_pattern(otherwise, "PiecewiseExpressionPattern", "otherwise")?.clone();
        let otherwise_pattern = otherwise_sub.get().pattern.clone();
        subs.push(otherwise_sub);
        let pattern = if cases.is_none() {
            Pattern::piecewise_any_cases(otherwise_pattern)
        } else {
            Pattern::piecewise(rust_cases, otherwise_pattern)
        };
        let fields = build_fields(py, &[&cases, otherwise])?;
        Ok(
            PyPattern::initializer(pattern, fields, &["cases", "otherwise"], &subs, None)
                .add_subclass(Self {
                    cases: cases.unbind(),
                    otherwise: otherwise.clone().unbind(),
                }),
        )
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
// CallExpressionPattern
// ---------------------------------------------------------------------------

/// The pattern matching a call: of the callee named `function_name`, or
/// any, with its arguments, when given, matched one to one and in order.
#[pyclass(
    extends = PyPattern,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "CallExpressionPattern"
)]
pub(crate) struct PyCallExpressionPattern {
    /// The callee's name as given, or `None` for any callee.
    #[pyo3(get)]
    function_name: Py<PyAny>,
    /// The tuple of argument patterns, or `None` for any arguments.
    #[pyo3(get)]
    arguments: Py<PyAny>,
}

impl_public_class!(PyCallExpressionPattern, "CallExpressionPattern");

#[pymethods]
impl PyCallExpressionPattern {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.function_name)?;
        visit.call(&self.arguments)?;
        Ok(())
    }

    /// Create the pattern of a call of the function named `function_name`,
    /// any for `None`, whose arguments match the iterable of patterns
    /// `arguments` position-wise, or are any for `None`.
    ///
    /// The name is parsed as `CallExpression` parses it, so a built-in's
    /// name is the built-in, and calls are compared by callee.
    ///
    /// Raises `TypeError` for a name that is not a `str` or an argument
    /// pattern that is not a `Pattern`, and `ValueError` for an empty name.
    #[new]
    fn new(
        function_name: &Bound<'_, PyAny>,
        arguments: &Bound<'_, PyAny>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let py = function_name.py();
        let callee = if function_name.is_none() {
            None
        } else {
            let name = read_str(function_name, "CallExpressionPattern", "function_name")?;
            Some(name.to_str()?.parse::<Callee>().into_py_result()?)
        };
        let (arguments, subs) = if arguments.is_none() {
            (arguments.clone(), Vec::new())
        } else {
            let (tuple, subs) = read_patterns(arguments, "CallExpressionPattern", "arguments")?;
            (tuple.into_any(), subs)
        };
        let pattern = match (callee, arguments.is_none()) {
            (Some(callee), true) => Pattern::call_any_arguments(callee),
            (None, true) => Pattern::any_call(),
            (Some(callee), false) => Pattern::call(callee, rust_patterns(&subs)),
            (None, false) => Pattern::call_any_callee(rust_patterns(&subs)),
        };
        let fields = build_fields(py, &[function_name, &arguments])?;
        Ok(PyPattern::initializer(
            pattern,
            fields,
            &["function_name", "arguments"],
            &subs,
            None,
        )
        .add_subclass(Self {
            function_name: function_name.clone().unbind(),
            arguments: arguments.unbind(),
        }))
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
// PredicatePattern
// ---------------------------------------------------------------------------

/// The pattern matching the expressions for which `predicate` returns a
/// true value, capturing nothing.
///
/// The predicate receives the expression's node object, and runs each time
/// the pattern is tried; an exception it raises ends the match.
#[pyclass(
    extends = PyPattern,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "PredicatePattern"
)]
pub(crate) struct PyPredicatePattern {
    /// The callable.
    #[pyo3(get)]
    predicate: Py<PyAny>,
    /// The slot the Rust predicate reads the callable from, which this
    /// object owns.
    slots: crate::gc::Slots,
}

impl_public_class!(PyPredicatePattern, "PredicatePattern");

/// Return the Rust predicate calling the Python callable `predicate` with
/// the candidate's node object, read by truthiness.
fn build_predicate(predicate: Py<PyAny>) -> Pattern {
    let predicate = crate::gc::Slot::new(predicate);
    Pattern::try_predicate(move |node| {
        Python::attach(|py| -> PyResult<bool> {
            let object = current_object_of(py, node)?;
            predicate.get(py).call1((object,))?.is_truthy()
        })
        .map_err(into_callback_error)
    })
}

#[pymethods]
impl PyPredicatePattern {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.predicate)?;
        self.slots.traverse(&visit)
    }

    /// Create the pattern matching the expressions for which the callable
    /// `predicate` returns a true value.
    ///
    /// Raises `TypeError` for a value that is not callable.
    #[new]
    fn new(predicate: &Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        let py = predicate.py();
        if !predicate.is_callable() {
            return Err(build_argument_type_error(
                "PredicatePattern",
                "predicate",
                "callable",
                predicate,
            )?);
        }
        let (pattern, slots) =
            crate::gc::collect_slots(|| build_predicate(predicate.clone().unbind()));
        let fields = build_fields(py, &[predicate])?;
        Ok(
            PyPattern::initializer(pattern, fields, &["predicate"], &[], None).add_subclass(Self {
                predicate: predicate.clone().unbind(),
                slots,
            }),
        )
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
// AlternativesPattern
// ---------------------------------------------------------------------------

/// The pattern trying `alternatives` in order: the first that matches
/// decides, a failed one leaves no captures, and the choice is final. No
/// alternative matches nothing.
#[pyclass(
    extends = PyPattern,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "AlternativesPattern"
)]
pub(crate) struct PyAlternativesPattern {
    /// The tuple of alternatives, in order.
    #[pyo3(get)]
    alternatives: Py<PyTuple>,
}

impl_public_class!(PyAlternativesPattern, "AlternativesPattern");

#[pymethods]
impl PyAlternativesPattern {
    /// Visit the Python objects the object holds, for the cycle collector.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "PyO3 hands `__traverse__` its visitor by value"
    )]
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.alternatives)?;
        Ok(())
    }

    /// Create the pattern trying the iterable of patterns `alternatives` in
    /// order.
    ///
    /// Raises `TypeError` for an alternative that is not a `Pattern`.
    #[new]
    fn new(alternatives: &Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        let py = alternatives.py();
        let (alternatives, subs) =
            read_patterns(alternatives, "AlternativesPattern", "alternatives")?;
        let pattern = Pattern::alternatives(rust_patterns(&subs));
        let fields = build_fields(py, &[alternatives.as_any()])?;
        Ok(
            PyPattern::initializer(pattern, fields, &["alternatives"], &subs, None).add_subclass(
                Self {
                    alternatives: alternatives.unbind(),
                },
            ),
        )
    }

    /// Register `cls` as the public class.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }
}

/// Return the `str` of `value`'s type name, for messages.
pub(super) fn type_name(value: &Bound<'_, PyAny>) -> String {
    value
        .get_type()
        .name()
        .map_or_else(|_| "?".to_owned(), |name| name.to_string())
}

/// Raise the `TypeError` for a value `value` of the argument `field` of
/// `owner` that is not an `expected`, as a `PyErr`.
pub(super) fn argument_type_error(
    owner: &str,
    field: &str,
    expected: &str,
    value: &Bound<'_, PyAny>,
) -> PyErr {
    PyTypeError::new_err(format!(
        "{owner} {field} must be {expected}, got {}.",
        type_name(value)
    ))
}

/// Return `value` as a `str` object or `None`, or raise the `TypeError`
/// naming `owner` and `field`.
pub(super) fn read_optional_str<'py>(
    value: &Bound<'py, PyAny>,
    owner: &str,
    field: &str,
) -> PyResult<Option<Bound<'py, PyString>>> {
    if value.is_none() {
        return Ok(None);
    }
    match value.cast::<PyString>() {
        Ok(text) => Ok(Some(text.clone())),
        Err(_not_a_str) => Err(argument_type_error(owner, field, "a str or None", value)),
    }
}
