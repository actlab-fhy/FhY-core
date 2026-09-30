//! The literal helpers of `native_lowering.py`:
//! `is_decimal_text_exactly_binary` and `coerce_literal_value`, over the
//! core's `Decimal::to_f64_exact`.

use num_traits::{FromPrimitive, ToPrimitive};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString};

use fhy_core::expression::{BigInt, Decimal, LiteralValue};

use crate::error::IntoPyResult;

use super::super::literal::{big_int_to_python, decimal_class, read_decimal_parts};
use super::error::string_literal_precision_error;

/// The number a decimal text denotes, and its sign.
struct SignedLiteral {
    is_negative: bool,
    magnitude: LiteralValue,
}

/// Return the number `value` denotes: integer- or float-grammar text, with
/// an optional sign, or a finite `decimal.Decimal`.
///
/// # Errors
///
/// Raises `ValueError` for text outside the grammar, and `TypeError` for a
/// value that is neither text nor a `decimal.Decimal`.
fn read_signed_literal(value: &Bound<'_, PyAny>) -> PyResult<SignedLiteral> {
    let py = value.py();
    let text = if let Ok(text) = value.cast::<PyString>() {
        text.to_str()?.to_owned()
    } else if value.is_instance(decimal_class(py)?)? {
        // From its `as_tuple()` parts, not its fixed-point text, which
        // expanded every digit of the exponent. A non-negative
        // exponent reads as an integer, as that text, having no decimal
        // point, did.
        let parts = read_decimal_parts(value)?;
        let magnitude = if parts.is_integral_form {
            let exponent = u32::try_from(parts.magnitude.exponent()).unwrap_or_else(|_| {
                unreachable!("an integral form's exponent is bounded and non-negative")
            });
            LiteralValue::Int(parts.magnitude.coefficient() * BigInt::from(10).pow(exponent))
        } else {
            LiteralValue::Decimal(parts.magnitude)
        };
        return Ok(SignedLiteral {
            is_negative: parts.is_negative,
            magnitude,
        });
    } else {
        return Err(PyTypeError::new_err(format!(
            "expected decimal text or a decimal.Decimal, got {}.",
            value.get_type().name()?
        )));
    };
    let (is_negative, digits) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text.strip_prefix('+').unwrap_or(&text)),
    };
    Ok(SignedLiteral {
        is_negative,
        magnitude: LiteralValue::parse_text(digits).into_py_result()?,
    })
}

/// Return the binary float equal to the integer `value`, if one is.
fn exact_float_of_integer(value: &BigInt) -> Option<f64> {
    let float = value.to_f64()?;
    (float.is_finite() && BigInt::from_f64(float).as_ref() == Some(value)).then_some(float)
}

/// Return the binary float equal to the decimal `value`, if one is.
fn exact_float_of_decimal(value: &Decimal) -> Option<f64> {
    value.to_f64_exact()
}

/// Return whether some binary float equals the decimal `text` exactly:
/// `"0.5"` does and `"0.1"` does not.
///
/// `text` is integer- or float-grammar text, with an optional sign, or a
/// finite `decimal.Decimal`. Raises `ValueError` for other text.
#[pyfunction]
pub(crate) fn is_decimal_text_exactly_binary(text: &Bound<'_, PyAny>) -> PyResult<bool> {
    let literal = read_signed_literal(text)?;
    Ok(match &literal.magnitude {
        LiteralValue::Int(integer) => exact_float_of_integer(integer).is_some(),
        LiteralValue::Decimal(decimal) => exact_float_of_decimal(decimal).is_some(),
        LiteralValue::Bool(_) | LiteralValue::Float(_) => {
            unreachable!("literal text parses to an integer or a decimal")
        }
    })
}

/// Return the Python number of a literal value: a `bool`, an `int` or a
/// `float` as given, integer-grammar text as its `int`, and a decimal (text
/// or a `decimal.Decimal`) as the `float` equal to it.
///
/// Raises `StringLiteralPrecisionError` for a decimal no binary float
/// equals, `ValueError` for text outside the literal grammar, and
/// `TypeError` for a value of another type.
#[pyfunction]
pub(crate) fn coerce_literal_value<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = value.py();
    if value.is_instance_of::<PyBool>()
        || value.is_instance_of::<PyInt>()
        || value.is_instance_of::<PyFloat>()
    {
        return Ok(value.clone());
    }
    let literal = read_signed_literal(value)?;
    let sign = if literal.is_negative { -1.0 } else { 1.0 };
    match literal.magnitude {
        LiteralValue::Int(integer) if value.is_instance_of::<PyString>() => big_int_to_python(
            py,
            &if literal.is_negative {
                -integer
            } else {
                integer
            },
        ),
        LiteralValue::Int(integer) => match exact_float_of_integer(&integer) {
            Some(float) => Ok(PyFloat::new(py, sign * float).into_any()),
            None => Err(string_literal_precision_error(
                py,
                &format!("decimal {integer} has no exact binary float"),
            )),
        },
        LiteralValue::Decimal(decimal) => match exact_float_of_decimal(&decimal) {
            Some(float) => Ok(PyFloat::new(py, sign * float).into_any()),
            None => Err(string_literal_precision_error(
                py,
                &format!("decimal {decimal} has no exact binary float"),
            )),
        },
        LiteralValue::Bool(_) | LiteralValue::Float(_) => {
            unreachable!("literal text parses to an integer or a decimal")
        }
    }
}
