//! Literal values: the Python values a `LiteralExpression` accepts and
//! returns, and the Rust [`LiteralValue`] it holds.
//!
//! The Rust core normalizes literals (D-S4-1): a Boolean, an integer of any
//! size, a binary float, or an exact non-negative [`Decimal`], with no
//! spelling kept. Into Rust, a `bool` becomes a Boolean, an `int` (or a
//! subclass other than `bool`, such as an `IntEnum` member) an integer, a
//! `float` (or a subclass, such as `numpy.float64`) a float, a finite
//! non-negative `decimal.Decimal` a decimal, and a `str` is parsed by the
//! core's literal grammar ([`LiteralValue::parse_text`]): ASCII digits are
//! an integer, and ASCII digits with one decimal point a decimal. Back in
//! Python, a literal's value is a `bool`, an `int`, a `float` or a
//! `decimal.Decimal`, the last normalized: `"1.50"` reads back as
//! `Decimal("1.5")` and `"100.0"` as `Decimal("1E+2")`.
//!
//! A negative `decimal.Decimal` is refused: the core's decimals are
//! non-negative, as its literal grammar has no sign, and a negative number
//! is the negation of a literal, `-LiteralExpression(Decimal("1.5"))`. A
//! negative zero is the decimal zero.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString, PyType};

use fhy_core::expression::{BigInt, Decimal, LiteralTextError, LiteralValue};

use crate::error::{IntoPyErr, IntoPyResult};

/// Raises `ValueError` with the core's text, which names the refused text.
impl IntoPyErr for LiteralTextError {
    fn into_py_err(self) -> PyErr {
        PyValueError::new_err(self.to_string())
    }
}

/// Return `decimal.Decimal`.
pub(super) fn decimal_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    CLASS.import(py, "decimal", "Decimal")
}

/// Return the `int` of the Python int `value`, which may be of any size.
pub(crate) fn read_big_int(value: &Bound<'_, PyAny>) -> PyResult<BigInt> {
    if let Ok(small) = value.extract::<i64>() {
        return Ok(BigInt::from(small));
    }
    let text = value
        .py()
        .get_type::<PyInt>()
        .call_method1(intern!(value.py(), "__repr__"), (value,))?;
    text.cast::<PyString>()?
        .to_str()?
        .parse()
        .map_err(|_unparsable| PyValueError::new_err("an int has decimal digits"))
}

/// Return the Python `int` of `value`.
pub(super) fn big_int_to_python<'py>(
    py: Python<'py>,
    value: &BigInt,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(small) = i64::try_from(value) {
        return Ok(small.into_pyobject(py)?.into_any());
    }
    py.get_type::<PyInt>().call1((value.to_string(),))
}

/// Return the Rust decimal of the `decimal.Decimal` `value`.
///
/// # Errors
///
/// Raises `ValueError` if `value` is not finite or is negative.
fn read_decimal(value: &Bound<'_, PyAny>) -> PyResult<Decimal> {
    let py = value.py();
    if !value.call_method0(intern!(py, "is_finite"))?.is_truthy()? {
        return Err(PyValueError::new_err(format!(
            "A literal Decimal must be finite, got {}.",
            value.repr()?
        )));
    }
    let is_zero = value.call_method0(intern!(py, "is_zero"))?.is_truthy()?;
    let is_signed = value.call_method0(intern!(py, "is_signed"))?.is_truthy()?;
    if is_signed && !is_zero {
        return Err(PyValueError::new_err(format!(
            "A literal Decimal must not be negative, got {}; write the \
             negation of the literal of its magnitude instead.",
            value.repr()?
        )));
    }
    let magnitude = value.call_method0(intern!(py, "copy_abs"))?;
    let text = magnitude.call_method1(intern!(py, "__format__"), ("f",))?;
    text.cast::<PyString>()?
        .to_str()?
        .parse::<Decimal>()
        .into_py_result()
}

/// Return the `decimal.Decimal` of `value`, in its normalized form.
pub(super) fn decimal_to_python<'py>(
    py: Python<'py>,
    value: &Decimal,
) -> PyResult<Bound<'py, PyAny>> {
    let text = format!("{}E{}", value.coefficient(), value.exponent());
    decimal_class(py)?.call1((text,))
}

/// Return the Python value of the literal `value`: a `bool`, an `int`, a
/// `float` or a `decimal.Decimal`.
pub(super) fn literal_to_python<'py>(
    py: Python<'py>,
    value: &LiteralValue,
) -> PyResult<Bound<'py, PyAny>> {
    match value {
        LiteralValue::Bool(value) => Ok(PyBool::new(py, *value).to_owned().into_any()),
        LiteralValue::Int(value) => big_int_to_python(py, value),
        LiteralValue::Float(value) => Ok(PyFloat::new(py, *value).into_any()),
        LiteralValue::Decimal(value) => decimal_to_python(py, value),
    }
}

/// Return the Rust literal of the Python `value` and the Python value the
/// literal returns as its `value`.
///
/// An exact `bool`, `int` or `float` is returned as given; any other
/// accepted value as the normalized value of the literal.
///
/// # Errors
///
/// Raises `TypeError` for a value of no accepted type, and `ValueError` for
/// a `str` outside the literal grammar or a decimal the core refuses.
pub(super) fn read_literal<'py>(
    value: &Bound<'py, PyAny>,
) -> PyResult<(LiteralValue, Bound<'py, PyAny>)> {
    let py = value.py();
    if let Ok(boolean) = value.cast::<PyBool>() {
        return Ok((LiteralValue::Bool(boolean.is_true()), value.clone()));
    }
    if value.is_instance_of::<PyInt>() {
        let integer = read_big_int(value)?;
        let stored = if value.is_exact_instance_of::<PyInt>() {
            value.clone()
        } else {
            big_int_to_python(py, &integer)?
        };
        return Ok((LiteralValue::Int(integer), stored));
    }
    if let Ok(float) = value.cast::<PyFloat>() {
        let number = float.value();
        let stored = if value.is_exact_instance_of::<PyFloat>() {
            value.clone()
        } else {
            PyFloat::new(py, number).into_any()
        };
        return Ok((LiteralValue::Float(number), stored));
    }
    if let Ok(text) = value.cast::<PyString>() {
        let literal = LiteralValue::parse_text(text.to_str()?).into_py_result()?;
        let stored = literal_to_python(py, &literal)?;
        return Ok((literal, stored));
    }
    if value.is_instance(decimal_class(py)?)? {
        let decimal = read_decimal(value)?;
        let stored = decimal_to_python(py, &decimal)?;
        return Ok((LiteralValue::Decimal(decimal), stored));
    }
    Err(PyTypeError::new_err(format!(
        "Unsupported type for literal expression value: {}; expected bool, \
         int, float, decimal.Decimal, or str.",
        value.get_type().name()?
    )))
}
