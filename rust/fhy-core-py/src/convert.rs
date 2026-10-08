//! The conversions between `fhy_core`'s Python objects and `fhy-core`'s Rust
//! values that a downstream binding crate needs to accept and return them.
//!
//! A downstream `-py` crate depends on `fhy-core` for the values and on this
//! crate for their Python objects, and its `#[pyfunction]`s and `#[pymethods]`
//! call these functions at their boundary. Because an aggregate extension
//! links one copy of this crate, an object built here is an instance of the
//! one `fhy_core` class, and a canonical value is the one canonical value.
//! Nothing here keeps the classes or registries of its own.
//!
//! Every function whose name ends in `_from_python` reads an object and
//! raises `TypeError` for one of another type, without changing it. Every
//! one whose name ends in `_to_python` builds the object of a Rust value
//! through the public class, as `fhy_core` itself does, and raises what
//! building it raises. All of them need the GIL, and none holds a lock across
//! a call into Python.
//!
//! The values of `NumPy` arrays, which the evaluators take and return, are
//! converted in the [`numpy`] submodule, which has its own rules and needs
//! `NumPy` at run time, unlike the functions above.
//!
//! The context in which `fhy_core` asks a param question, for a downstream
//! binding crate that asks its own, is [`param::with_param_context`].
//!
//! The objects of `fhy_core.search_space`, and the registration of the
//! `Variable` and `Alternative` kinds a downstream crate defines, are in
//! the [`search_space`] submodule.
//!
//! # Identity
//!
//! - An identifier is converted by id and name hint, in both directions,
//!   and neither direction issues an id: the id counter is the
//!   `fhy-core` one all the crates in the extension share.
//! - An [`OpAttribute`] or a [`ValueDomain`] is a canonical value, and its
//!   Python object is single: converting the same canonical value twice
//!   returns the same object, so `is` holds.
//! - An expression, a type, a param and a diagnostic are values: converting
//!   one to Python builds new objects for it, and converting those back
//!   gives a value equal to the original, sharing its structure.

pub mod numpy;
pub mod param;
pub mod search_space;

use pyo3::prelude::*;

use fhy_core::diagnostic::Diagnostic;
use fhy_core::expression::Expression;
use fhy_core::identifier::Identifier;
use fhy_core::interned::Canonical;
use fhy_core::op_attribute::OpAttribute;
use fhy_core::param::{Param, ParamAssignment};
use fhy_core::types::Type;
use fhy_core::value_domain::ValueDomain;

use crate::{diagnostic, expression, identifier, op_attribute, types, value_domain};

/// Return the Rust identifier with the id and name hint of the Python
/// `Identifier` `object`.
///
/// The identifier keeps the id, and the id counter is advanced past it as
/// deserialization does, which never changes the counter for an identifier
/// the process built.
///
/// # Errors
///
/// Raises `TypeError` if `object` is not an `Identifier`, and
/// `OverflowError` if its id is outside the payload range.
pub fn identifier_from_python(object: &Bound<'_, PyAny>) -> PyResult<Identifier> {
    identifier::restore_identifier(object, "conversion", "value")
}

/// Return the Python `Identifier` with the id and name hint of `identifier`,
/// which is not issued a new id.
///
/// # Errors
///
/// Raises whatever `Identifier.deserialize_from_dict` raises.
pub fn identifier_to_python<'py>(
    py: Python<'py>,
    identifier: &Identifier,
) -> PyResult<Bound<'py, PyAny>> {
    identifier::identifier_to_python(py, identifier)
}

/// Return a new Python `Identifier` named `name_hint`, with a new id drawn
/// from the counter shared by the whole extension.
///
/// # Errors
///
/// Raises whatever the constructor raises, such as `RuntimeError` when the
/// counter is exhausted.
pub fn new_identifier<'py>(py: Python<'py>, name_hint: &str) -> PyResult<Bound<'py, PyAny>> {
    identifier::new_python_identifier(py, name_hint)
}

/// Return the Rust expression of the Python `Expression` `object`, sharing
/// its structure.
///
/// # Errors
///
/// Raises `TypeError` if `object` is not an `Expression`.
pub fn expression_from_python(object: &Bound<'_, PyAny>) -> PyResult<Expression> {
    object
        .cast::<expression::PyExpression>()
        .map(|node| node.get().expression().clone())
        .map_err(|_not_an_expression| not_a("an Expression", object))
}

/// Return a new Python object of `expression`, each node built through the
/// public class of its kind.
///
/// # Errors
///
/// Raises whatever building a node through its public class raises.
pub fn expression_to_python<'py>(
    py: Python<'py>,
    expression: &Expression,
) -> PyResult<Bound<'py, PyAny>> {
    expression::materialize_expression(py, expression)
}

/// Return the Rust type of the Python `Type` `object`: a built-in type's
/// value, or a Python-defined `Type` as an extension.
///
/// # Errors
///
/// Raises `TypeError` if `object` is not a `Type`.
pub fn type_from_python(object: &Bound<'_, PyAny>) -> PyResult<Type> {
    types::read_type_value(object).ok_or_else(|| not_a("a Type", object))
}

/// Return the Python object of `value`: a Python-defined type's own object,
/// and a new instance of the public class of a built-in one.
///
/// # Errors
///
/// Raises whatever building the object raises.
pub fn type_to_python<'py>(py: Python<'py>, value: &Type) -> PyResult<Bound<'py, PyAny>> {
    types::run_in_context(py, None, |context| {
        types::type_to_python(py, context, value)
    })
}

/// Return the Rust param of the Python `Param` `object`.
///
/// # Errors
///
/// Raises `TypeError` if `object` is not a `Param`.
pub fn param_from_python(object: &Bound<'_, PyAny>) -> PyResult<Param> {
    crate::param::param_from_python(object)
}

/// Return a new Python `Param` of `param`, over new objects of its domain,
/// variable and constraints.
///
/// # Errors
///
/// Raises whatever building an object of a part raises.
pub fn param_to_python<'py>(py: Python<'py>, param: &Param) -> PyResult<Bound<'py, PyAny>> {
    crate::param::param_to_python(py, param)
}

/// Return the Rust assignment of the Python `ParamAssignment` `object`.
///
/// # Errors
///
/// Raises `TypeError` if `object` is not a `ParamAssignment`.
pub fn param_assignment_from_python(object: &Bound<'_, PyAny>) -> PyResult<ParamAssignment> {
    crate::param::assignment_from_python(object)
}

/// Return a new Python `ParamAssignment` of `assignment`, over new objects
/// of its param and value. The assignment was checked when it was built, so
/// the object is not checked again.
///
/// # Errors
///
/// Raises whatever building an object of a part raises.
pub fn param_assignment_to_python<'py>(
    py: Python<'py>,
    assignment: &ParamAssignment,
) -> PyResult<Bound<'py, PyAny>> {
    crate::param::assignment_to_python(py, assignment)
}

/// Return the canonical domain of the Python `ValueDomain` `object`.
///
/// # Errors
///
/// Raises `TypeError` if `object` is not a `ValueDomain`.
pub fn value_domain_from_python(object: &Bound<'_, PyAny>) -> PyResult<Canonical<ValueDomain>> {
    value_domain::value_domain_from_python(object)
}

/// Return the one Python `ValueDomain` object of the canonical `domain`,
/// creating it and its ancestors' objects if they have none yet.
///
/// # Errors
///
/// Raises what importing `fhy_core.value_domain` or building the object
/// raises.
pub fn value_domain_to_python(
    py: Python<'_>,
    domain: Canonical<ValueDomain>,
) -> PyResult<Bound<'_, PyAny>> {
    value_domain::value_domain_to_python(py, domain)
}

/// Return the canonical attribute of the Python `OpAttribute` `object`.
///
/// # Errors
///
/// Raises `TypeError` if `object` is not an `OpAttribute`.
pub fn op_attribute_from_python(object: &Bound<'_, PyAny>) -> PyResult<Canonical<OpAttribute>> {
    op_attribute::op_attribute_from_python(object)
}

/// Return the one Python `OpAttribute` object of the canonical `attribute`,
/// creating it if it has none yet.
///
/// # Errors
///
/// Raises what importing `fhy_core.op_attribute` or building the object
/// raises.
pub fn op_attribute_to_python(
    py: Python<'_>,
    attribute: Canonical<OpAttribute>,
) -> PyResult<Bound<'_, PyAny>> {
    op_attribute::op_attribute_to_python(py, attribute)
}

/// Return the Rust diagnostic of the Python `Diagnostic` `object`.
///
/// # Errors
///
/// Raises `TypeError` if `object` is not a `Diagnostic`.
pub fn diagnostic_from_python(object: &Bound<'_, PyAny>) -> PyResult<Diagnostic> {
    diagnostic::borrow_python_diagnostic(object)
        .cloned()
        .ok_or_else(|| not_a("a Diagnostic", object))
}

/// Return a new Python `Diagnostic` of `diagnostic`, with a new `Note` as
/// its message; the note's kind is the one object of its canonical kind.
///
/// # Errors
///
/// Raises whatever building the objects raises.
pub fn diagnostic_to_python<'py>(
    py: Python<'py>,
    diagnostic: &Diagnostic,
) -> PyResult<Bound<'py, PyAny>> {
    diagnostic::diagnostic_to_python(py, diagnostic)
}

/// Return the Rust diagnostics of the Python `ValidationReport` `object`,
/// in the report's order. Its records are not converted.
///
/// # Errors
///
/// Raises `TypeError` if `object` is not a `ValidationReport`.
pub fn validation_report_from_python(object: &Bound<'_, PyAny>) -> PyResult<Vec<Diagnostic>> {
    diagnostic::report_diagnostics_from_python(object)
}

/// Return a new Python `ValidationReport` of `diagnostics`, in order, with
/// no records.
///
/// # Errors
///
/// Raises whatever building the objects raises.
pub fn validation_report_to_python<'py>(
    py: Python<'py>,
    diagnostics: &[Diagnostic],
) -> PyResult<Bound<'py, PyAny>> {
    let objects = diagnostics
        .iter()
        .map(|diagnostic| diagnostic::diagnostic_to_python(py, diagnostic))
        .collect::<PyResult<Vec<_>>>()?;
    diagnostic::report_to_python(
        py,
        pyo3::types::PyTuple::new(py, objects)?,
        pyo3::types::PyTuple::empty(py),
    )
}

/// Return the `TypeError` for an `object` that is not `expected`.
fn not_a(expected: &str, object: &Bound<'_, PyAny>) -> PyErr {
    let actual = object
        .get_type()
        .name()
        .map_or_else(|_| "?".to_owned(), |name| name.to_string());
    pyo3::exceptions::PyTypeError::new_err(format!("expected {expected}, got {actual}."))
}
