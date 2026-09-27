//! The expression classes: `fhy_core._rs.Expression` and its seven node
//! classes, the bases of the classes of the same names in
//! `fhy_core.symbolic.expression.core` (pattern P2, as a class hierarchy).
//!
//! `Expression` holds the Rust [`Expression`] handle, the tuple of its
//! children's Python objects in visiting order, and its structural hash,
//! computed once, on the first `hash`. Each node class extends it with the
//! Python objects its fields return, read as struct members.
//!
//! Every Python node's children are Python node objects whose Rust handles
//! are the Rust node's children: a node built from Python keeps the objects
//! it was built from, and a tree the core builds, such as a substitution's
//! result, is turned into Python objects at once by the materializer, which
//! reuses the objects of every subtree the core kept. So reading a field
//! never builds an object, and `node.left is node.left` holds.

use std::collections::HashSet;
use std::sync::OnceLock;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyDict, PyFrozenSet, PyList, PyString, PyTuple, PyType};

use fhy_core::expression::{
    BinaryOperation, Callee, Expression, ExpressionKind, FunctionNameError, LogicalOperation,
    PiecewiseError, RebuildError, UnaryOperation,
};

use crate::dataclass::{build_argument_type_error, collect_tuple, hash_value, read_str};
use crate::error::{IntoPyErr, IntoPyResult};
use crate::frozen::build_frozen_mutation_error;
use crate::identifier::{read_identifier_id, restore_identifier};
use crate::public_class::PublicClass;
use crate::serialization::{FieldShape, construct_from_decoded_fields, read_payload_fields};

use super::literal::{literal_to_python, read_literal};
use super::materialize::substitute;
use super::operation::{PythonOperation, operation_from_python, operation_to_python};
use super::payload::deserialize_expression_payload;
use super::text::{render_formatted, render_repr};
use crate::term::read_renaming;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Raises `ValueError` with the core's text.
impl IntoPyErr for PiecewiseError {
    fn into_py_err(self) -> PyErr {
        PyValueError::new_err(self.to_string())
    }
}

/// Raises `ValueError` with the core's text; for a refused piecewise, the
/// piecewise error's text.
impl IntoPyErr for RebuildError {
    fn into_py_err(self) -> PyErr {
        match self {
            Self::Piecewise(error) => error.into_py_err(),
            _ => PyValueError::new_err(self.to_string()),
        }
    }
}

/// Raises `ValueError` with the core's text.
impl IntoPyErr for FunctionNameError {
    fn into_py_err(self) -> PyErr {
        PyValueError::new_err(self.to_string())
    }
}

// ---------------------------------------------------------------------------
// Arguments and builders
// ---------------------------------------------------------------------------

/// Return `value` as an expression, or raise the `TypeError` naming
/// `owner` and `field`.
pub(super) fn read_expression<'a, 'py>(
    value: &'a Bound<'py, PyAny>,
    owner: &str,
    field: &str,
) -> PyResult<&'a Bound<'py, PyExpression>> {
    match value.cast::<PyExpression>() {
        Ok(expression) => Ok(expression),
        Err(_not_an_expression) => Err(build_argument_type_error(
            owner,
            field,
            "an Expression",
            value,
        )?),
    }
}

/// Return the Rust handles of the expressions `values`, the items of the
/// argument `field` of `owner`.
fn read_expressions(
    values: &Bound<'_, PyTuple>,
    owner: &str,
    field: &str,
) -> PyResult<Vec<Expression>> {
    values
        .iter()
        .map(|value| {
            read_expression(&value, owner, field)
                .map(|expression| expression.get().expression.clone())
        })
        .collect()
}

/// Return the error refusing to coerce the bare Python `bool` `value`,
/// supplied at `position`, to an expression.
pub(super) fn build_bare_bool_error(position: &str, value: bool) -> PyErr {
    let spelled = if value { "True" } else { "False" };
    PyValueError::new_err(format!(
        "{position} is a bare Python bool ({spelled}), not an expression. This \
         is almost always the accidental result of `expr == k`, which compares \
         two expressions structurally and returns a Python bool rather than \
         building an equality; use `.equals()` or `.not_equals()` to build an \
         equality, or write `LiteralExpression({spelled})` for a Boolean \
         constant."
    ))
}

/// Return `value` as an expression: an expression itself, an `Identifier`
/// as its reference, and a number, a decimal or a numeric `str` as its
/// literal.
///
/// # Errors
///
/// Raises `ValueError` for a bare `bool`, for a value of any other type,
/// and for a `str` outside the literal grammar.
pub(crate) fn coerce_to_expression<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = value.py();
    if value.is_instance_of::<PyExpression>() {
        return Ok(value.clone());
    }
    if read_identifier_id(value)?.is_some() {
        return PyIdentifierExpression::public_class()
            .get(py)?
            .call1((value,));
    }
    if let Ok(boolean) = value.cast_exact::<PyBool>() {
        return Err(build_bare_bool_error("Operand", boolean.is_true()));
    }
    let is_number = value.is_instance_of::<pyo3::types::PyInt>()
        || value.is_instance_of::<pyo3::types::PyFloat>()
        || value.is_exact_instance_of::<PyString>()
        || value.is_instance(super::literal::decimal_class(py)?)?;
    if is_number {
        return PyLiteralExpression::public_class().get(py)?.call1((value,));
    }
    Err(PyValueError::new_err(format!(
        "Unable to cast {} with type {} to an expression.",
        value.repr()?,
        value.get_type().str()?
    )))
}

/// Return the unary node of `operation` over the expression `operand`.
fn build_unary<'py>(
    operation: UnaryOperation,
    operand: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = operand.py();
    PyUnaryExpression::public_class()
        .get(py)?
        .call1((operation_to_python(py, operation)?, operand))
}

/// Return the binary node of `operation` over `left` and `right`, each
/// coerced to an expression.
fn build_binary<'py>(
    operation: BinaryOperation,
    left: &Bound<'py, PyAny>,
    right: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = left.py();
    let left = coerce_to_expression(left)?;
    let right = coerce_to_expression(right)?;
    PyBinaryExpression::public_class().get(py)?.call1((
        operation_to_python(py, operation)?,
        left,
        right,
    ))
}

/// Return the logical node of `operation` over `first` and `others`, each
/// coerced to an expression.
///
/// # Errors
///
/// Raises `ValueError` unless there are at least two operands, with the
/// builder `name` in the message.
pub(super) fn build_logical<'py>(
    operation: LogicalOperation,
    name: &str,
    operands: impl IntoIterator<Item = Bound<'py, PyAny>>,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let operands = operands
        .into_iter()
        .map(|operand| coerce_to_expression(&operand))
        .collect::<PyResult<Vec<_>>>()?;
    if operands.len() < 2 {
        return Err(PyValueError::new_err(format!(
            "{name} requires at least two expressions, but got {}.",
            operands.len()
        )));
    }
    PyLogicalExpression::public_class().get(py)?.call1((
        operation_to_python(py, operation)?,
        PyTuple::new(py, operands)?,
    ))
}

/// Return the payload of the serializable `value` from its own
/// `serialize_to_dict`.
fn serialize_nested<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    value.call_method0(intern!(value.py(), "serialize_to_dict"))
}

/// Return the list of the payloads of the serializable `values`.
fn serialize_nested_list<'py>(values: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyList>> {
    let payloads = values
        .iter()
        .map(|value| serialize_nested(&value))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(values.py(), payloads)
}

/// Return the expression a payload encodes, decoded by the public
/// `Expression.deserialize_from_dict`.
fn deserialize_expression<'py>(payload: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = payload.py();
    PyExpression::public_class()
        .get(py)?
        .call_method1(intern!(py, "deserialize_from_dict"), (payload,))
}

/// Return the tuple of the expressions a list of payloads encodes.
fn deserialize_expression_tuple<'py>(
    payloads: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let expressions = payloads
        .try_iter()?
        .map(|payload| deserialize_expression(&payload?))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(payloads.py(), expressions)
}

/// Return the operation named `name`, or raise the
/// `DeserializationValueError` of the field `operation` of `cls`.
fn decode_operation<'py, T: PythonOperation + std::str::FromStr>(
    cls: &Bound<'py, PyType>,
    name: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = cls.py();
    match name.cast::<PyString>()?.to_str()?.parse::<T>() {
        Ok(operation) => operation_to_python(py, operation),
        Err(_unknown) => {
            let error = crate::serialization::deserialization_value_error_class(py)?.call1((
                cls,
                "operation",
                format!("a valid {} value", T::CLASS_NAME),
                name,
            ))?;
            Err(PyErr::from_value(error))
        }
    }
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

/// Raise the rebuild error of a node with `expected` children given
/// `actual`.
fn build_child_count_error(expected: usize, actual: usize) -> PyErr {
    RebuildError::ChildCount { expected, actual }.into_py_err()
}

/// Return `fhy_core.traits.visitable._camel_to_snake`.
fn camel_to_snake(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    static FUNCTION: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    FUNCTION.import(py, "fhy_core.traits.visitable", "_camel_to_snake")
}

// ---------------------------------------------------------------------------
// Expression
// ---------------------------------------------------------------------------

/// A symbolic expression, backed by the Rust [`Expression`]; the base of
/// the node classes.
///
/// The class itself has no constructor: every expression is an instance of
/// a node class, which sets the Rust handle.
#[pyclass(subclass, frozen, module = "fhy_core._rs", name = "Expression")]
pub(crate) struct PyExpression {
    expression: Expression,
    /// The children's Python objects, in visiting order.
    children: Py<PyTuple>,
    /// The structural hash, computed on the first `hash`.
    hash: OnceLock<u64>,
}

impl PyExpression {
    /// Return the public Python class registered for this class.
    pub(crate) fn public_class() -> &'static PublicClass {
        static PUBLIC_CLASS: PublicClass = PublicClass::new("Expression");
        &PUBLIC_CLASS
    }

    /// Return the initializer of a node holding `expression`, whose
    /// children's Python objects are `children`.
    fn initializer(
        expression: Expression,
        children: Bound<'_, PyTuple>,
    ) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Self {
            expression,
            children: children.unbind(),
            hash: OnceLock::new(),
        })
    }

    /// Return the Rust handle.
    pub(crate) fn expression(&self) -> &Expression {
        &self.expression
    }

    /// Return the children's Python objects, in visiting order.
    pub(crate) fn children<'py>(&self, py: Python<'py>) -> &Bound<'py, PyTuple> {
        self.children.bind(py)
    }

    /// Return the structural hash, computing it on the first call.
    fn structural_hash(&self) -> u64 {
        *self.hash.get_or_init(|| hash_value(&self.expression))
    }

    /// Return whether the two expressions are structurally equal.
    ///
    /// Compares the Rust trees, after two shortcuts: shared handles are
    /// equal, and expressions whose hashes are both known and differ are
    /// not.
    pub(crate) fn is_structurally_equal(&self, other: &Self) -> bool {
        if Expression::ptr_eq(&self.expression, &other.expression) {
            return true;
        }
        if let (Some(left), Some(right)) = (self.hash.get(), other.hash.get()) {
            if left != right {
                return false;
            }
        }
        self.expression == other.expression
    }

    /// Return the result of comparing `slf` with `other` for equality, or
    /// `NotImplemented` if `other` is not an expression.
    fn compare_equal<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
        expected: bool,
    ) -> Bound<'py, PyAny> {
        let py = slf.py();
        match other.cast::<Self>() {
            Ok(other) => {
                let is_equal = slf.get().is_structurally_equal(other.get());
                PyBool::new(py, is_equal == expected).to_owned().into_any()
            }
            Err(_not_an_expression) => py.NotImplemented().into_bound(py),
        }
    }
}

#[pymethods]
impl PyExpression {
    /// Return whether `other` is an expression of the same structure:
    /// the same node kinds, operations and callees, identifiers with the
    /// same ids, equal literals, children equal in order.
    fn __eq__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        Self::compare_equal(slf, other, true)
    }

    fn __ne__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        Self::compare_equal(slf, other, false)
    }

    /// Return the structural hash, computed once per expression.
    fn __hash__(&self) -> u64 {
        self.structural_hash()
    }

    /// Refuse a truth value: an expression is symbolic, not a Boolean.
    fn __bool__(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Err(PyTypeError::new_err(format!(
            "{} has no truth value: it is a symbolic expression, not a \
             Boolean. A chained comparison such as `0 <= x <= 5` asks for one, \
             as do the `and`, `or`, and `not` operators, and each would \
             silently keep only one operand. Build the connective with \
             `logical_and`, `logical_or`, or `logical_not` instead.",
            slf.get_type().name()?
        )))
    }

    /// Render the expression in the core's symbolic notation.
    fn __str__(&self) -> String {
        self.expression.to_string()
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(render_repr(
            slf.get_type().name()?.to_str()?,
            &slf.get().expression,
        ))
    }

    /// Render the expression as `pformat_expression` does.
    #[pyo3(signature = (show_id = false, functional = false))]
    fn _format(&self, show_id: bool, functional: bool) -> String {
        render_formatted(&self.expression, show_id, functional)
    }

    /// Return the visitor dispatch suffix of the class: its name in
    /// snake case, computed once per class.
    #[classmethod]
    fn get_visit_method_suffix<'py>(cls: &Bound<'py, PyType>) -> PyResult<Bound<'py, PyAny>> {
        static SUFFIXES: PyOnceLock<Py<PyDict>> = PyOnceLock::new();
        let py = cls.py();
        let suffixes = SUFFIXES
            .get_or_init(py, || PyDict::new(py).unbind())
            .bind(py);
        if let Some(suffix) = suffixes.get_item(cls)? {
            return Ok(suffix);
        }
        let suffix = camel_to_snake(py)?.call1((cls.name()?,))?;
        suffixes.set_item(cls, &suffix)?;
        Ok(suffix)
    }

    /// Return the result of `visitor.visit(self)`.
    fn accept<'py>(
        slf: &Bound<'py, Self>,
        visitor: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        visitor.call_method1(intern!(slf.py(), "visit"), (slf,))
    }

    /// Return the children in visiting order.
    fn get_visit_children<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        self.children.bind(py).clone()
    }

    /// Return the identifiers the expression refers to, every one free.
    fn get_free_identifiers<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyFrozenSet>> {
        let py = slf.py();
        let mut seen_ids = HashSet::new();
        let mut visited = HashSet::new();
        let mut identifiers = Vec::new();
        let mut pending = vec![slf.clone()];
        while let Some(node) = pending.pop() {
            let this = node.get();
            if let ExpressionKind::Identifier(identifier) = this.expression.kind() {
                if seen_ids.insert(identifier.id()) {
                    let leaf = node.cast::<PyIdentifierExpression>()?;
                    identifiers.push(leaf.get().identifier.clone_ref(py));
                }
                continue;
            }
            for child in this.children.bind(py) {
                if visited.insert(child.as_ptr() as usize) {
                    pending.push(child.cast_into::<Self>()?);
                }
            }
        }
        PyFrozenSet::new(py, identifiers)
    }

    /// Return this expression with every reference to a mapped identifier
    /// replaced by its replacement, simultaneously.
    ///
    /// Every subtree without a replaced reference is returned as the same
    /// object, so a mapping that replaces nothing returns this expression
    /// itself.
    ///
    /// Raises `TypeError` if an identifier the expression refers to is
    /// mapped to a value that is not an expression, and `ValueError` if a
    /// replacement puts a literal other than a Boolean in a piecewise case
    /// condition.
    fn substitute<'py>(
        slf: &Bound<'py, Self>,
        replacements: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        substitute(slf, replacements)
    }

    /// Return whether `other` is an expression of the same structure, as
    /// `==` decides.
    fn is_structurally_equivalent(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<Self>()
            .is_ok_and(|other| self.is_structurally_equal(other.get()))
    }

    /// Return whether `other` is this expression up to no renaming, which
    /// for expressions, which bind nothing, is structural equality.
    fn is_alpha_equivalent(&self, other: &Bound<'_, PyAny>) -> bool {
        self.is_structurally_equivalent(other)
    }

    /// Return whether `other` is this expression with its identifiers
    /// renamed by `renaming`, a `fhy_core.term.AlphaRenaming`: its binder
    /// frames and its free renaming.
    ///
    /// Raises `TypeError` if `renaming` is not an `AlphaRenaming`.
    fn is_alpha_equivalent_under(
        &self,
        other: &Bound<'_, PyAny>,
        renaming: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        let renaming = read_renaming(renaming)?.get().value().renaming();
        if renaming.is_empty() {
            return Ok(self.is_structurally_equal(other.get()));
        }
        Ok(self
            .expression
            .is_alpha_equivalent_under(&other.get().expression, renaming))
    }

    /// Return the expression of the envelope payload `data`, an instance
    /// of `cls`.
    ///
    /// Decodes a payload of the expression classes' own shapes in one pass;
    /// any other payload goes through `WrappedFamilySerializable`'s
    /// decoding, which raises the serialization framework's errors.
    #[classmethod]
    fn deserialize_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        deserialize_expression_payload(cls, data)
    }

    /// Return `other` as an expression, as every operator and builder
    /// coerces its operands.
    #[staticmethod]
    fn _get_expression_from_other<'py>(other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        coerce_to_expression(other)
    }

    fn __neg__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        build_unary(UnaryOperation::Negate, slf.as_any())
    }

    fn __pos__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        build_unary(UnaryOperation::Positive, slf.as_any())
    }

    fn __add__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        build_binary(BinaryOperation::Add, slf.as_any(), other)
    }

    fn __radd__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        build_binary(BinaryOperation::Add, other, slf.as_any())
    }

    fn __sub__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        build_binary(BinaryOperation::Subtract, slf.as_any(), other)
    }

    fn __rsub__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        build_binary(BinaryOperation::Subtract, other, slf.as_any())
    }

    fn __mul__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        build_binary(BinaryOperation::Multiply, slf.as_any(), other)
    }

    fn __rmul__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        build_binary(BinaryOperation::Multiply, other, slf.as_any())
    }

    /// Build the true division of the expression by `other`.
    fn __truediv__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        build_binary(BinaryOperation::Divide, slf.as_any(), other)
    }

    fn __rtruediv__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        build_binary(BinaryOperation::Divide, other, slf.as_any())
    }

    fn __floordiv__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        build_binary(BinaryOperation::FloorDivide, slf.as_any(), other)
    }

    fn __rfloordiv__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        build_binary(BinaryOperation::FloorDivide, other, slf.as_any())
    }

    /// Build the remainder of floor division, `BinaryOperation.MODULO`.
    fn __mod__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        build_binary(BinaryOperation::FloorMod, slf.as_any(), other)
    }

    fn __rmod__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        build_binary(BinaryOperation::FloorMod, other, slf.as_any())
    }

    fn __pow__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
        modulo: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        if modulo.is_some_and(|modulo| !modulo.is_none()) {
            return Ok(py.NotImplemented().into_bound(py));
        }
        build_binary(BinaryOperation::Power, slf.as_any(), other)
    }

    fn __rpow__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
        modulo: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        if modulo.is_some_and(|modulo| !modulo.is_none()) {
            return Ok(py.NotImplemented().into_bound(py));
        }
        build_binary(BinaryOperation::Power, other, slf.as_any())
    }

    fn __lt__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        build_binary(BinaryOperation::Less, slf.as_any(), other)
    }

    fn __le__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        build_binary(BinaryOperation::LessEqual, slf.as_any(), other)
    }

    fn __gt__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        build_binary(BinaryOperation::Greater, slf.as_any(), other)
    }

    fn __ge__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        build_binary(BinaryOperation::GreaterEqual, slf.as_any(), other)
    }

    /// Build the equality of the expression and `other`.
    fn equals<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        build_binary(BinaryOperation::Equal, slf.as_any(), other)
    }

    /// Build the inequality of the expression and `other`.
    fn not_equals<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        build_binary(BinaryOperation::NotEqual, slf.as_any(), other)
    }

    /// Build the conjunction of the expression and `others`, one
    /// `LogicalExpression` over all of them.
    ///
    /// Raises `ValueError` unless `others` has at least one operand.
    #[pyo3(signature = (*others))]
    fn logical_and<'py>(
        slf: &Bound<'py, Self>,
        others: &Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let operands = std::iter::once(slf.clone().into_any()).chain(others.iter());
        build_logical(LogicalOperation::And, "logical_and", operands, slf.py())
    }

    /// Build the disjunction of the expression and `others`, one
    /// `LogicalExpression` over all of them.
    ///
    /// Raises `ValueError` unless `others` has at least one operand.
    #[pyo3(signature = (*others))]
    fn logical_or<'py>(
        slf: &Bound<'py, Self>,
        others: &Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let operands = std::iter::once(slf.clone().into_any()).chain(others.iter());
        build_logical(LogicalOperation::Or, "logical_or", operands, slf.py())
    }

    /// Always true: expressions are immutable.
    #[getter]
    fn is_frozen(_slf: &Bound<'_, Self>) -> bool {
        true
    }

    /// Do nothing: expressions are always frozen.
    fn freeze(_slf: &Bound<'_, Self>) {}

    /// Do nothing: expressions are always frozen, and mutating one raises.
    fn assert_frozen(_slf: &Bound<'_, Self>) {}

    fn __setattr__(slf: &Bound<'_, Self>, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let _ = value;
        Err(build_frozen_mutation_error(slf, "modify", name)?)
    }

    fn __delattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<()> {
        Err(build_frozen_mutation_error(slf, "delete", name)?)
    }

    /// Register `cls` as the public class.
    ///
    /// Raises `RuntimeError` if another public class is registered.
    #[classmethod]
    fn _register_public_class(cls: &Bound<'_, PyType>) -> PyResult<()> {
        Self::public_class().register(cls)
    }
}

/// Implement `public_class` and `_register_public_class` for a node class.
macro_rules! impl_public_class {
    ($class:ty, $name:literal) => {
        impl $class {
            /// Return the public Python class registered for this class.
            pub(crate) fn public_class() -> &'static PublicClass {
                static PUBLIC_CLASS: PublicClass = PublicClass::new($name);
                &PUBLIC_CLASS
            }
        }
    };
}

/// Return `children` if it is empty, and raise the rebuild error of a node
/// with no children otherwise; the rebuild of a leaf.
fn rebuild_leaf<'py>(
    slf: &Bound<'py, PyAny>,
    children: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let actual = children.len()?;
    if actual != 0 {
        return Err(build_child_count_error(0, actual));
    }
    Ok(slf.clone())
}

/// Return the items of `children`, which must number `expected`.
fn read_children<'py>(
    children: &Bound<'py, PyAny>,
    expected: usize,
) -> PyResult<Bound<'py, PyTuple>> {
    let children = collect_tuple(children)?;
    if children.len() != expected {
        return Err(build_child_count_error(expected, children.len()));
    }
    Ok(children)
}

// ---------------------------------------------------------------------------
// UnaryExpression
// ---------------------------------------------------------------------------

/// A unary operation applied to one operand, backed by
/// [`ExpressionKind::Unary`].
#[pyclass(
    extends = PyExpression,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "UnaryExpression"
)]
pub(crate) struct PyUnaryExpression {
    /// The `UnaryOperation` member.
    #[pyo3(get)]
    operation: Py<PyAny>,
    /// The operand expression.
    #[pyo3(get)]
    operand: Py<PyAny>,
}

impl_public_class!(PyUnaryExpression, "UnaryExpression");

#[pymethods]
impl PyUnaryExpression {
    /// Create the node of `operation`, a `UnaryOperation`, over the
    /// expression `operand`.
    ///
    /// Raises `ValueError` for an operation that is not a
    /// `UnaryOperation`, and `TypeError` for an operand that is not an
    /// `Expression`.
    #[new]
    fn new(
        operation: &Bound<'_, PyAny>,
        operand: &Bound<'_, PyAny>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let py = operation.py();
        let (rust_operation, member) = operation_from_python::<UnaryOperation>(operation)?;
        let rust_operand = read_expression(operand, "UnaryExpression", "operand")?
            .get()
            .expression
            .clone();
        let expression = Expression::new_unary(rust_operation, rust_operand);
        let children = PyTuple::new(py, [operand])?;
        Ok(
            PyExpression::initializer(expression, children).add_subclass(Self {
                operation: member.unbind(),
                operand: operand.clone().unbind(),
            }),
        )
    }

    /// Return `(operand,)`.
    fn get_operands<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
        slf.as_super().get().children(slf.py()).clone()
    }

    /// Return the node of the same operation over the one child given.
    ///
    /// Raises `ValueError` unless exactly one child is given.
    fn rebuild_with_visit_children<'py>(
        slf: &Bound<'py, Self>,
        new_children: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let children = read_children(new_children, 1)?;
        Self::public_class()
            .get(py)?
            .call1((slf.get().operation.bind(py), children.get_item(0)?))
    }

    /// Pickle as a constructor call of the node's class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let arguments = PyTuple::new(py, [this.operation.bind(py), this.operand.bind(py)])?;
        Ok((slf.get_type(), arguments))
    }

    /// Return the data payload `{"operation": .., "operand": ..}`.
    fn serialize_data_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let ExpressionKind::Unary(node) = slf.as_super().get().expression.kind() else {
            unreachable!("a unary node holds a unary expression");
        };
        build_fields(
            py,
            &[
                (
                    "operation",
                    PyString::new(py, node.operation().as_str()).as_any(),
                ),
                ("operand", &serialize_nested(slf.get().operand.bind(py))?),
            ],
        )
    }

    /// Return the node of a data payload.
    ///
    /// Raises the serialization framework's errors for a malformed payload.
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let [operation, operand] = read_payload_fields(
            cls,
            data,
            [
                ("operation", FieldShape::Str),
                ("operand", FieldShape::Payload),
            ],
        )?;
        let fields = build_fields(
            py,
            &[
                (
                    "operation",
                    &decode_operation::<UnaryOperation>(cls, &operation)?,
                ),
                ("operand", &deserialize_expression(&operand)?),
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
// BinaryExpression
// ---------------------------------------------------------------------------

/// A binary operation applied to a left and a right operand, backed by
/// [`ExpressionKind::Binary`].
#[pyclass(
    extends = PyExpression,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "BinaryExpression"
)]
pub(crate) struct PyBinaryExpression {
    /// The `BinaryOperation` member.
    #[pyo3(get)]
    operation: Py<PyAny>,
    /// The left operand expression.
    #[pyo3(get)]
    left: Py<PyAny>,
    /// The right operand expression.
    #[pyo3(get)]
    right: Py<PyAny>,
}

impl_public_class!(PyBinaryExpression, "BinaryExpression");

#[pymethods]
impl PyBinaryExpression {
    /// Create the node of `operation`, a `BinaryOperation`, over the
    /// expressions `left` and `right`.
    ///
    /// Raises `ValueError` for an operation that is not a
    /// `BinaryOperation`, and `TypeError` for an operand that is not an
    /// `Expression`.
    #[new]
    fn new(
        operation: &Bound<'_, PyAny>,
        left: &Bound<'_, PyAny>,
        right: &Bound<'_, PyAny>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let py = operation.py();
        let (rust_operation, member) = operation_from_python::<BinaryOperation>(operation)?;
        let rust_left = read_expression(left, "BinaryExpression", "left")?
            .get()
            .expression
            .clone();
        let rust_right = read_expression(right, "BinaryExpression", "right")?
            .get()
            .expression
            .clone();
        let expression = Expression::new_binary(rust_operation, rust_left, rust_right);
        let children = PyTuple::new(py, [left, right])?;
        Ok(
            PyExpression::initializer(expression, children).add_subclass(Self {
                operation: member.unbind(),
                left: left.clone().unbind(),
                right: right.clone().unbind(),
            }),
        )
    }

    /// Return `(left, right)`.
    fn get_operands<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
        slf.as_super().get().children(slf.py()).clone()
    }

    /// Return the node of the same operation over the two children given.
    ///
    /// Raises `ValueError` unless exactly two children are given.
    fn rebuild_with_visit_children<'py>(
        slf: &Bound<'py, Self>,
        new_children: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let children = read_children(new_children, 2)?;
        Self::public_class().get(py)?.call1((
            slf.get().operation.bind(py),
            children.get_item(0)?,
            children.get_item(1)?,
        ))
    }

    /// Pickle as a constructor call of the node's class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let arguments = PyTuple::new(
            py,
            [
                this.operation.bind(py),
                this.left.bind(py),
                this.right.bind(py),
            ],
        )?;
        Ok((slf.get_type(), arguments))
    }

    /// Return the data payload `{"operation": .., "left": .., "right": ..}`.
    fn serialize_data_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let ExpressionKind::Binary(node) = slf.as_super().get().expression.kind() else {
            unreachable!("a binary node holds a binary expression");
        };
        let this = slf.get();
        build_fields(
            py,
            &[
                (
                    "operation",
                    PyString::new(py, node.operation().as_str()).as_any(),
                ),
                ("left", &serialize_nested(this.left.bind(py))?),
                ("right", &serialize_nested(this.right.bind(py))?),
            ],
        )
    }

    /// Return the node of a data payload.
    ///
    /// Raises the serialization framework's errors for a malformed payload.
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let [operation, left, right] = read_payload_fields(
            cls,
            data,
            [
                ("operation", FieldShape::Str),
                ("left", FieldShape::Payload),
                ("right", FieldShape::Payload),
            ],
        )?;
        let fields = build_fields(
            py,
            &[
                (
                    "operation",
                    &decode_operation::<BinaryOperation>(cls, &operation)?,
                ),
                ("left", &deserialize_expression(&left)?),
                ("right", &deserialize_expression(&right)?),
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
// LogicalExpression
// ---------------------------------------------------------------------------

/// A conjunction or disjunction of two or more operands, backed by
/// [`ExpressionKind::Logical`].
#[pyclass(
    extends = PyExpression,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "LogicalExpression"
)]
pub(crate) struct PyLogicalExpression {
    /// The `LogicalOperation` member.
    #[pyo3(get)]
    operation: Py<PyAny>,
    /// The operand expressions, in order.
    #[pyo3(get)]
    operands: Py<PyTuple>,
}

impl_public_class!(PyLogicalExpression, "LogicalExpression");

#[pymethods]
impl PyLogicalExpression {
    /// Create the node of `operation`, a `LogicalOperation`, over the
    /// iterable of expressions `operands`, kept in order and never
    /// flattened.
    ///
    /// Raises `ValueError` for an operation that is not a
    /// `LogicalOperation` or fewer than two operands, and `TypeError` for
    /// an operand that is not an `Expression`.
    #[new]
    fn new(
        operation: &Bound<'_, PyAny>,
        operands: &Bound<'_, PyAny>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let (rust_operation, member) = operation_from_python::<LogicalOperation>(operation)?;
        let operands = collect_tuple(operands)?;
        let rust_operands = read_expressions(&operands, "LogicalExpression", "operands")?;
        if rust_operands.len() < 2 {
            return Err(PyValueError::new_err(format!(
                "LogicalExpression requires at least two operands, but got {}.",
                rust_operands.len()
            )));
        }
        let expression = Expression::new_logical(rust_operation, rust_operands);
        Ok(
            PyExpression::initializer(expression, operands.clone()).add_subclass(Self {
                operation: member.unbind(),
                operands: operands.unbind(),
            }),
        )
    }

    /// Return the operands, in order.
    fn get_operands<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        self.operands.bind(py).clone()
    }

    /// Return the node of the same operation over the children given, as
    /// many as it has operands.
    ///
    /// Raises `ValueError` unless the count matches.
    fn rebuild_with_visit_children<'py>(
        slf: &Bound<'py, Self>,
        new_children: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this = slf.get();
        let children = read_children(new_children, this.operands.bind(py).len())?;
        Self::public_class()
            .get(py)?
            .call1((this.operation.bind(py), children))
    }

    /// Pickle as a constructor call of the node's class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let arguments = PyTuple::new(
            py,
            [this.operation.bind(py), this.operands.bind(py).as_any()],
        )?;
        Ok((slf.get_type(), arguments))
    }

    /// Return the data payload `{"operation": .., "operands": [..]}`.
    fn serialize_data_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let ExpressionKind::Logical(node) = slf.as_super().get().expression.kind() else {
            unreachable!("a logical node holds a logical expression");
        };
        build_fields(
            py,
            &[
                (
                    "operation",
                    PyString::new(py, node.operation().as_str()).as_any(),
                ),
                (
                    "operands",
                    serialize_nested_list(slf.get().operands.bind(py))?.as_any(),
                ),
            ],
        )
    }

    /// Return the node of a data payload.
    ///
    /// Raises the serialization framework's errors for a malformed payload.
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let [operation, operands] = read_payload_fields(
            cls,
            data,
            [
                ("operation", FieldShape::Str),
                ("operands", FieldShape::PayloadList),
            ],
        )?;
        let fields = build_fields(
            py,
            &[
                (
                    "operation",
                    &decode_operation::<LogicalOperation>(cls, &operation)?,
                ),
                (
                    "operands",
                    deserialize_expression_tuple(&operands)?.as_any(),
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
// IdentifierExpression
// ---------------------------------------------------------------------------

/// A reference to an identifier, backed by [`ExpressionKind::Identifier`].
#[pyclass(
    extends = PyExpression,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "IdentifierExpression"
)]
pub(crate) struct PyIdentifierExpression {
    /// The Python `Identifier` the node was built from.
    #[pyo3(get)]
    identifier: Py<PyAny>,
}

impl_public_class!(PyIdentifierExpression, "IdentifierExpression");

impl PyIdentifierExpression {
    /// Return the Python `Identifier` the node was built from.
    pub(crate) fn identifier_object(&self) -> &Py<PyAny> {
        &self.identifier
    }
}

#[pymethods]
impl PyIdentifierExpression {
    /// Create the reference to `identifier`, an `Identifier`.
    ///
    /// Raises `TypeError` for a value that is not an `Identifier`.
    #[new]
    fn new(identifier: &Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        let py = identifier.py();
        let rust_identifier = restore_identifier(identifier, "IdentifierExpression", "identifier")?;
        let expression = Expression::from(rust_identifier);
        Ok(
            PyExpression::initializer(expression, PyTuple::empty(py)).add_subclass(Self {
                identifier: identifier.clone().unbind(),
            }),
        )
    }

    /// Return the node itself for no children.
    ///
    /// Raises `ValueError` if children are given.
    fn rebuild_with_visit_children<'py>(
        slf: &Bound<'py, Self>,
        new_children: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        rebuild_leaf(slf.as_any(), new_children)
    }

    /// Pickle as a constructor call of the node's class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        Ok((
            slf.get_type(),
            PyTuple::new(py, [slf.get().identifier.bind(py)])?,
        ))
    }

    /// Return the data payload `{"identifier": <identifier payload>}`.
    fn serialize_data_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        build_fields(
            py,
            &[(
                "identifier",
                &serialize_nested(slf.get().identifier.bind(py))?,
            )],
        )
    }

    /// Return the node of a data payload.
    ///
    /// Raises the serialization framework's errors for a malformed payload.
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let [identifier] = read_payload_fields(cls, data, [("identifier", FieldShape::Payload)])?;
        let identifier = crate::identifier::deserialize_identifier(&identifier)?;
        let fields = build_fields(py, &[("identifier", &identifier)])?;
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
// LiteralExpression
// ---------------------------------------------------------------------------

/// A constant, backed by [`ExpressionKind::Literal`].
#[pyclass(
    extends = PyExpression,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "LiteralExpression"
)]
pub(crate) struct PyLiteralExpression {
    /// The normalized value: a `bool`, `int`, `float` or `decimal.Decimal`.
    #[pyo3(get)]
    value: Py<PyAny>,
}

impl_public_class!(PyLiteralExpression, "LiteralExpression");

/// Return the payload value of a literal: a `bool`, `int` or `float` as
/// itself, and a decimal as its positional text with a decimal point, which
/// the literal grammar reads back as the same decimal.
fn encode_literal_value<'py>(
    py: Python<'py>,
    expression: &Expression,
    value: &Bound<'py, PyAny>,
) -> Bound<'py, PyAny> {
    match expression.kind() {
        ExpressionKind::Literal(fhy_core::expression::LiteralValue::Decimal(decimal)) => {
            let mut text = decimal.to_string();
            if !text.contains('.') {
                text.push_str(".0");
            }
            PyString::new(py, &text).into_any()
        }
        _ => value.clone(),
    }
}

#[pymethods]
impl PyLiteralExpression {
    /// Create the literal of `value`: a `bool`, an `int`, a `float`, a
    /// finite non-negative `decimal.Decimal`, or a `str` of ASCII digits
    /// with at most one decimal point, normalized.
    ///
    /// Raises `TypeError` for a value of another type, and `ValueError` for
    /// a `str` outside the grammar or a refused decimal.
    #[new]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        let py = value.py();
        let (literal, stored) = read_literal(value)?;
        let expression = Expression::from(literal);
        Ok(
            PyExpression::initializer(expression, PyTuple::empty(py)).add_subclass(Self {
                value: stored.unbind(),
            }),
        )
    }

    /// Return the node itself for no children.
    ///
    /// Raises `ValueError` if children are given.
    fn rebuild_with_visit_children<'py>(
        slf: &Bound<'py, Self>,
        new_children: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        rebuild_leaf(slf.as_any(), new_children)
    }

    /// Pickle as a constructor call of the node's class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        Ok((
            slf.get_type(),
            PyTuple::new(py, [slf.get().value.bind(py)])?,
        ))
    }

    /// Return the data payload `{"value": ..}`: a `bool`, `int` or `float`
    /// as itself, and a decimal as its positional text, such as `"1.5"` or
    /// `"100.0"`.
    fn serialize_data_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let value = encode_literal_value(
            py,
            &slf.as_super().get().expression,
            slf.get().value.bind(py),
        );
        build_fields(py, &[("value", &value)])
    }

    /// Return the literal of a data payload.
    ///
    /// Raises the serialization framework's errors for a malformed payload.
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let [value] = read_payload_fields(cls, data, [("value", FieldShape::Literal)])?;
        let fields = build_fields(py, &[("value", &value)])?;
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
// PiecewiseExpression
// ---------------------------------------------------------------------------

/// A first-match-wins choice among cases, with a fallback, backed by
/// [`ExpressionKind::Piecewise`].
#[pyclass(
    extends = PyExpression,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "PiecewiseExpression"
)]
pub(crate) struct PyPiecewiseExpression {
    /// The case conditions, in evaluation order.
    #[pyo3(get)]
    conditions: Py<PyTuple>,
    /// The case values, paired with `conditions` by position.
    #[pyo3(get)]
    values: Py<PyTuple>,
    /// The value when no condition holds.
    #[pyo3(get)]
    otherwise: Py<PyAny>,
}

impl_public_class!(PyPiecewiseExpression, "PiecewiseExpression");

#[pymethods]
impl PyPiecewiseExpression {
    /// Create the piecewise of the cases pairing the iterables `conditions`
    /// and `values` by position, and the fallback `otherwise`.
    ///
    /// Raises `TypeError` for an item that is not an `Expression`, and
    /// `ValueError` for no cases, for `conditions` and `values` of unequal
    /// lengths, or for a literal condition other than a Boolean.
    #[new]
    fn new(
        conditions: &Bound<'_, PyAny>,
        values: &Bound<'_, PyAny>,
        otherwise: &Bound<'_, PyAny>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let py = otherwise.py();
        let conditions = collect_tuple(conditions)?;
        let values = collect_tuple(values)?;
        let rust_conditions = read_expressions(&conditions, "PiecewiseExpression", "conditions")?;
        let rust_values = read_expressions(&values, "PiecewiseExpression", "values")?;
        let rust_otherwise = read_expression(otherwise, "PiecewiseExpression", "otherwise")?
            .get()
            .expression
            .clone();
        if rust_conditions.len() != rust_values.len() {
            return Err(PyValueError::new_err(format!(
                "PiecewiseExpression conditions and values must have equal \
                 length, but got {} conditions and {} values.",
                rust_conditions.len(),
                rust_values.len()
            )));
        }
        let expression =
            Expression::piecewise(rust_conditions.into_iter().zip(rust_values), rust_otherwise)
                .into_py_result()?;
        let mut children = Vec::with_capacity(2 * conditions.len() + 1);
        for (condition, value) in conditions.iter().zip(values.iter()) {
            children.push(condition);
            children.push(value);
        }
        children.push(otherwise.clone());
        Ok(
            PyExpression::initializer(expression, PyTuple::new(py, children)?).add_subclass(Self {
                conditions: conditions.unbind(),
                values: values.unbind(),
                otherwise: otherwise.clone().unbind(),
            }),
        )
    }

    /// Return the `(condition, value)` cases, in evaluation order.
    fn get_cases<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let cases = self
            .conditions
            .bind(py)
            .iter()
            .zip(self.values.bind(py).iter())
            .map(|(condition, value)| PyTuple::new(py, [condition, value]))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, cases)
    }

    /// Return the children, `c0, v0, c1, v1, ..., otherwise`.
    fn get_operands<'py>(slf: &Bound<'py, Self>) -> Bound<'py, PyTuple> {
        slf.as_super().get().children(slf.py()).clone()
    }

    /// Return the piecewise of the children given in visiting order, as
    /// many as it has.
    ///
    /// Raises `ValueError` unless the count matches, or if a new condition
    /// is a literal other than a Boolean.
    fn rebuild_with_visit_children<'py>(
        slf: &Bound<'py, Self>,
        new_children: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let case_count = slf.get().conditions.bind(py).len();
        let children = read_children(new_children, 2 * case_count + 1)?;
        let mut conditions = Vec::with_capacity(case_count);
        let mut values = Vec::with_capacity(case_count);
        for index in 0..case_count {
            conditions.push(children.get_item(2 * index)?);
            values.push(children.get_item(2 * index + 1)?);
        }
        Self::public_class().get(py)?.call1((
            PyTuple::new(py, conditions)?,
            PyTuple::new(py, values)?,
            children.get_item(2 * case_count)?,
        ))
    }

    /// Pickle as a constructor call of the node's class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let arguments = PyTuple::new(
            py,
            [
                this.conditions.bind(py).as_any(),
                this.values.bind(py).as_any(),
                this.otherwise.bind(py),
            ],
        )?;
        Ok((slf.get_type(), arguments))
    }

    /// Return the data payload `{"conditions": [..], "values": [..],
    /// "otherwise": ..}`.
    fn serialize_data_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let this = slf.get();
        build_fields(
            py,
            &[
                (
                    "conditions",
                    serialize_nested_list(this.conditions.bind(py))?.as_any(),
                ),
                (
                    "values",
                    serialize_nested_list(this.values.bind(py))?.as_any(),
                ),
                ("otherwise", &serialize_nested(this.otherwise.bind(py))?),
            ],
        )
    }

    /// Return the piecewise of a data payload.
    ///
    /// Raises the serialization framework's errors for a malformed payload.
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let [conditions, values, otherwise] = read_payload_fields(
            cls,
            data,
            [
                ("conditions", FieldShape::PayloadList),
                ("values", FieldShape::PayloadList),
                ("otherwise", FieldShape::Payload),
            ],
        )?;
        let fields = build_fields(
            py,
            &[
                (
                    "conditions",
                    deserialize_expression_tuple(&conditions)?.as_any(),
                ),
                ("values", deserialize_expression_tuple(&values)?.as_any()),
                ("otherwise", &deserialize_expression(&otherwise)?),
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
// CallExpression
// ---------------------------------------------------------------------------

/// A function applied to arguments, backed by [`ExpressionKind::Call`].
///
/// The callee is a built-in function when its name is one (D-9: built-in
/// names are reserved), and a named user function otherwise.
#[pyclass(
    extends = PyExpression,
    subclass,
    frozen,
    module = "fhy_core._rs",
    name = "CallExpression"
)]
pub(crate) struct PyCallExpression {
    /// The callee's name.
    #[pyo3(get)]
    function_name: Py<PyString>,
    /// The argument expressions, in order.
    #[pyo3(get)]
    arguments: Py<PyTuple>,
}

impl_public_class!(PyCallExpression, "CallExpression");

#[pymethods]
impl PyCallExpression {
    /// Create the call of the function named `function_name` with the
    /// iterable of expressions `arguments`.
    ///
    /// Raises `TypeError` for a name that is not a `str` or an argument
    /// that is not an `Expression`, and `ValueError` for an empty name.
    #[new]
    fn new(
        function_name: &Bound<'_, PyAny>,
        arguments: &Bound<'_, PyAny>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let py = function_name.py();
        let name = read_str(function_name, "CallExpression", "function_name")?;
        let callee = name.to_str()?.parse::<Callee>().into_py_result()?;
        let name = if name.is_exact_instance_of::<PyString>() {
            name.clone()
        } else {
            PyString::new(py, callee.name())
        };
        let arguments = collect_tuple(arguments)?;
        let rust_arguments = read_expressions(&arguments, "CallExpression", "arguments")?;
        let expression = Expression::call(callee, rust_arguments);
        Ok(
            PyExpression::initializer(expression, arguments.clone()).add_subclass(Self {
                function_name: name.unbind(),
                arguments: arguments.unbind(),
            }),
        )
    }

    /// Return whether the callee is a built-in function.
    #[getter]
    fn is_builtin(slf: &Bound<'_, Self>) -> bool {
        matches!(
            slf.as_super().get().expression.kind(),
            ExpressionKind::Call(call) if matches!(call.callee(), Callee::Builtin(_))
        )
    }

    /// Return the arguments, in order.
    fn get_operands<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        self.arguments.bind(py).clone()
    }

    /// Return the call of the same function with the children given, as
    /// many as it has arguments.
    ///
    /// Raises `ValueError` unless the count matches.
    fn rebuild_with_visit_children<'py>(
        slf: &Bound<'py, Self>,
        new_children: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let this = slf.get();
        let children = read_children(new_children, this.arguments.bind(py).len())?;
        Self::public_class()
            .get(py)?
            .call1((this.function_name.bind(py), children))
    }

    /// Pickle as a constructor call of the node's class.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let arguments = PyTuple::new(
            py,
            [
                this.function_name.bind(py).as_any(),
                this.arguments.bind(py).as_any(),
            ],
        )?;
        Ok((slf.get_type(), arguments))
    }

    /// Return the data payload `{"function_name": .., "arguments": [..]}`.
    fn serialize_data_to_dict<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
        let py = slf.py();
        let this = slf.get();
        build_fields(
            py,
            &[
                ("function_name", this.function_name.bind(py).as_any()),
                (
                    "arguments",
                    serialize_nested_list(this.arguments.bind(py))?.as_any(),
                ),
            ],
        )
    }

    /// Return the call of a data payload.
    ///
    /// Raises the serialization framework's errors for a malformed payload.
    #[classmethod]
    fn deserialize_data_from_dict<'py>(
        cls: &Bound<'py, PyType>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let [function_name, arguments] = read_payload_fields(
            cls,
            data,
            [
                ("function_name", FieldShape::Str),
                ("arguments", FieldShape::PayloadList),
            ],
        )?;
        let fields = build_fields(
            py,
            &[
                ("function_name", &function_name),
                (
                    "arguments",
                    deserialize_expression_tuple(&arguments)?.as_any(),
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

/// Return the Python node of the core node `expression` whose children's
/// Python objects are `children`, in visiting order, built through the
/// registered public class of its kind.
pub(super) fn build_node<'py>(
    py: Python<'py>,
    expression: &Expression,
    children: Vec<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    match expression.kind() {
        ExpressionKind::Unary(node) => PyUnaryExpression::public_class().get(py)?.call1((
            operation_to_python(py, node.operation())?,
            PyTuple::new(py, children)?.get_item(0)?,
        )),
        ExpressionKind::Binary(node) => {
            let children = PyTuple::new(py, children)?;
            PyBinaryExpression::public_class().get(py)?.call1((
                operation_to_python(py, node.operation())?,
                children.get_item(0)?,
                children.get_item(1)?,
            ))
        }
        ExpressionKind::Logical(node) => PyLogicalExpression::public_class().get(py)?.call1((
            operation_to_python(py, node.operation())?,
            PyTuple::new(py, children)?,
        )),
        ExpressionKind::Identifier(identifier) => PyIdentifierExpression::public_class()
            .get(py)?
            .call1((crate::identifier::identifier_to_python(py, identifier)?,)),
        ExpressionKind::Literal(value) => PyLiteralExpression::public_class()
            .get(py)?
            .call1((literal_to_python(py, value)?,)),
        ExpressionKind::Piecewise(node) => {
            let case_count = node.cases().len();
            let mut children = children.into_iter();
            let mut conditions = Vec::with_capacity(case_count);
            let mut values = Vec::with_capacity(case_count);
            for _ in 0..case_count {
                if let (Some(condition), Some(value)) = (children.next(), children.next()) {
                    conditions.push(condition);
                    values.push(value);
                }
            }
            let otherwise = children
                .next()
                .unwrap_or_else(|| unreachable!("a piecewise has an otherwise branch"));
            PyPiecewiseExpression::public_class().get(py)?.call1((
                PyTuple::new(py, conditions)?,
                PyTuple::new(py, values)?,
                otherwise,
            ))
        }
        ExpressionKind::Call(node) => PyCallExpression::public_class()
            .get(py)?
            .call1((node.callee().name(), PyTuple::new(py, children)?)),
    }
}
