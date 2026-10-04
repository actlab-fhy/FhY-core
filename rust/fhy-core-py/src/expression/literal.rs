//! Literal values: the Python values a `LiteralExpression` accepts and
//! returns, and the Rust [`LiteralValue`] it holds.
//!
//! The Rust core normalizes literals: a Boolean, an integer of any
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

use num_traits::Zero;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyString, PyTuple, PyType};

use fhy_core::expression::{BigInt, Decimal, DecimalPartsError, LiteralTextError, LiteralValue};

use crate::error::{IntoPyErr, IntoPyResult};

/// Raises `ValueError` with the core's text, which names the refused text.
impl IntoPyErr for LiteralTextError {
    fn into_py_err(self) -> PyErr {
        PyValueError::new_err(self.to_string())
    }
}

/// Return `decimal.Decimal`.
pub(crate) fn decimal_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    crate::kit::python::cached_attr!(py, "decimal", "Decimal" => PyType)
}

/// Return the `int` of the Python int `value`, which may be of any size.
pub(crate) fn read_big_int(value: &Bound<'_, PyAny>) -> PyResult<BigInt> {
    if let Ok(small) = value.extract::<i64>() {
        return Ok(BigInt::from(small));
    }
    // Through `int.to_bytes`, signed and little-endian, sized from
    // `bit_length` with room for the sign: linear, and free of CPython's
    // digit limit on decimal text. `int`'s own methods, so an
    // `int` subclass's overrides do not change the value read.
    let py = value.py();
    let int = py.get_type::<PyInt>();
    let bit_length: usize = int
        .call_method1(intern!(py, "bit_length"), (value,))?
        .extract()?;
    let keywords = PyDict::new(py);
    keywords.set_item(intern!(py, "signed"), true)?;
    let bytes = int.call_method(
        intern!(py, "to_bytes"),
        (value, bit_length / 8 + 1, intern!(py, "little")),
        Some(&keywords),
    )?;
    Ok(BigInt::from_signed_bytes_le(
        bytes.cast::<PyBytes>()?.as_bytes(),
    ))
}

/// Return the Python `int` of `value`.
///
/// A big one is built through `int.from_bytes`, so its size is not bounded
/// by `CPython`'s digit limit on decimal text.
pub(crate) fn big_int_to_python<'py>(
    py: Python<'py>,
    value: &BigInt,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(small) = i64::try_from(value) {
        return Ok(small.into_pyobject(py)?.into_any());
    }
    let keywords = PyDict::new(py);
    keywords.set_item(intern!(py, "signed"), true)?;
    py.get_type::<PyInt>().call_method(
        intern!(py, "from_bytes"),
        (
            PyBytes::new(py, &value.to_signed_bytes_le()),
            intern!(py, "little"),
        ),
        Some(&keywords),
    )
}

/// Return the Rust decimal of the `decimal.Decimal` `value`.
///
/// # Errors
///
/// Raises `ValueError` if `value` is not finite or is negative.
pub(crate) fn read_decimal(value: &Bound<'_, PyAny>) -> PyResult<Decimal> {
    let parts = read_decimal_parts(value)?;
    if parts.is_negative && !parts.magnitude.coefficient().is_zero() {
        return Err(PyValueError::new_err(format!(
            "A literal Decimal must not be negative, got {}; write the \
             negation of the literal of its magnitude instead.",
            value.repr()?
        )));
    }
    Ok(parts.magnitude)
}

/// The parts of a finite `decimal.Decimal`: its sign, its magnitude, and
/// whether its exponent is non-negative, so its fixed-point text has no
/// decimal point.
pub(crate) struct DecimalParts {
    pub(crate) is_negative: bool,
    pub(crate) magnitude: Decimal,
    pub(crate) is_integral_form: bool,
}

/// Return the parts of the `decimal.Decimal` `value`, read from
/// `value.as_tuple()` into [`Decimal::from_parts`]: the digits are
/// assembled in Rust, so neither the exponent is expanded nor the digits go
/// through `CPython`'s digit limit, and an exponent beyond
/// [`Decimal::MAX_EXPONENT_MAGNITUDE`] is refused at once.
///
/// # Errors
///
/// Raises `ValueError` if `value` is not finite, or its exponent is beyond
/// the bound, naming it.
pub(crate) fn read_decimal_parts(value: &Bound<'_, PyAny>) -> PyResult<DecimalParts> {
    let py = value.py();
    if !value.call_method0(intern!(py, "is_finite"))?.is_truthy()? {
        return Err(PyValueError::new_err(format!(
            "A literal Decimal must be finite, got {}.",
            value.repr()?
        )));
    }
    let parts = value.call_method0(intern!(py, "as_tuple"))?;
    let sign: u8 = parts.get_item(0)?.extract()?;
    let digit_objects = parts.get_item(1)?;
    let digit_objects = digit_objects.cast::<PyTuple>()?;
    let mut digits = Vec::with_capacity(digit_objects.len());
    for digit in digit_objects {
        let digit: u8 = digit.extract()?;
        if digit > 9 {
            return Err(PyValueError::new_err(format!(
                "A Decimal's digit must lie in 0..=9, got {digit}."
            )));
        }
        digits.push(b'0' + digit);
    }
    let coefficient = if digits.is_empty() {
        BigInt::ZERO
    } else {
        BigInt::parse_bytes(&digits, 10)
            .unwrap_or_else(|| unreachable!("ASCII digits parse as an integer"))
    };
    // The value is not written: its `repr` has every digit.
    let out_of_range = |error: DecimalPartsError| {
        PyValueError::new_err(format!("A literal Decimal is out of range: {error}."))
    };
    let exponent = parts.get_item(2)?;
    let exponent: i64 = exponent.extract().map_err(|_beyond_i64| {
        out_of_range(DecimalPartsError::ExponentOutOfRange {
            exponent: if exponent.gt(0).unwrap_or(true) {
                i64::MAX
            } else {
                i64::MIN
            },
        })
    })?;
    Ok(DecimalParts {
        is_negative: sign == 1,
        magnitude: Decimal::from_parts(coefficient, exponent).map_err(out_of_range)?,
        is_integral_form: exponent >= 0,
    })
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
