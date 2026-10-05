//! Reading a Python integer as a Rust unsigned integer, with one set of
//! errors.
//!
//! The reader is exact about what an argument may be:
//!
//! - a `bool` is a `TypeError`, as is a float or anything else that is not
//!   an `int`: `True` is not the number 1 here, and `2.0` is not 2;
//! - a negative integer is a `ValueError`, worded with the caller's
//!   [`Label`] and [`Minimum`];
//! - an integer above the target's maximum is an `OverflowError`;
//! - a predicate, which answers `False` for an integer no range can hold,
//!   reads leniently ([`read_unsigned_lenient`]).
//!
//! The readers are generic over the target, any of [`UnsignedInteger`]:
//! `u8`, `u16`, `u32`, `u64`, `u128` and `usize`. A caller that chooses its
//! own errors reads with [`classify_unsigned`] and builds the overflow with
//! [`build_too_large_error`].

use std::fmt;

use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyInt};

/// An unsigned machine integer a Python `int` is read as.
///
/// Implemented for `u8`, `u16`, `u32`, `u64`, `u128` and `usize`, and sealed:
/// no other type implements it, so the set of targets can grow without a
/// breaking change.
pub trait UnsignedInteger: sealed::Sealed + Copy + Sized + 'static {
    /// The width of the type in bits, which the `OverflowError` names.
    const BITS: u32;

    /// Return `object` as `Self`.
    ///
    /// # Errors
    ///
    /// Returns the `OverflowError` `PyO3` raises for an `int` that is
    /// negative or does not fit, and the `TypeError` it raises for an
    /// object that is not an integer.
    fn extract_from(object: &Bound<'_, PyAny>) -> PyResult<Self>;
}

/// The supertrait that seals [`UnsignedInteger`]: it is not nameable outside
/// this module.
mod sealed {
    #[expect(
        unnameable_types,
        reason = "the sealed-trait pattern: nothing outside names it"
    )]
    pub trait Sealed {}
}

macro_rules! impl_unsigned_integer {
    ($($target:ty),*) => {$(
        impl sealed::Sealed for $target {}

        impl UnsignedInteger for $target {
            const BITS: u32 = <$target>::BITS;

            fn extract_from(object: &Bound<'_, PyAny>) -> PyResult<Self> {
                object.extract::<$target>()
            }
        }
    )*};
}

impl_unsigned_integer!(u8, u16, u32, u64, u128, usize);

/// The name of an argument in an error message: its name, and the position
/// it holds when it is one component of a sequence.
#[derive(Debug, Clone, Copy)]
pub struct Label<'a> {
    name: &'a str,
    position: Option<Position>,
}

/// How a component is named after its sequence.
#[derive(Debug, Clone, Copy)]
enum Position {
    /// After the name and a space: `Tile shape axis 1`.
    Axis(usize),
    /// In brackets on the name: `origin[1]`.
    Index(usize),
}

impl<'a> Label<'a> {
    /// Return the label of the argument `name`.
    #[must_use]
    pub const fn argument(name: &'a str) -> Self {
        Self {
            name,
            position: None,
        }
    }

    /// Return the label of component `axis` of the sequence `name`, which
    /// reads `name axis`.
    #[must_use]
    pub const fn component(name: &'a str, axis: usize) -> Self {
        Self {
            name,
            position: Some(Position::Axis(axis)),
        }
    }

    /// Return the label of the entry `index` of the sequence `name`, which
    /// reads `name[index]`.
    #[must_use]
    pub const fn entry(name: &'a str, index: usize) -> Self {
        Self {
            name,
            position: Some(Position::Index(index)),
        }
    }
}

impl fmt::Display for Label<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.position {
            Some(Position::Axis(axis)) => write!(formatter, "{} {axis}", self.name),
            Some(Position::Index(index)) => write!(formatter, "{}[{index}]", self.name),
            None => formatter.write_str(self.name),
        }
    }
}

/// The smallest number [`read_unsigned`] accepts, which also words the
/// `ValueError` for a number below it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Minimum {
    /// One or more: zero and a negative number are refused, "must be a
    /// positive integer".
    Positive,
    /// Zero or more: a negative number is refused, "must be a non-negative
    /// integer".
    NonNegative,
}

/// What a Python `int` is as an unsigned integer `T`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[expect(
    clippy::exhaustive_enums,
    reason = "an integer is in range, negative or too large, and callers match all three"
)]
pub enum Reading<T> {
    /// The value, which fits.
    Value(T),
    /// A negative number.
    Negative,
    /// A number above the maximum of `T`.
    TooLarge,
}

/// Read `object` as an `int` and classify it as an unsigned integer `T`,
/// without choosing which exception a value that does not fit is.
///
/// # Errors
///
/// Raises `TypeError`, naming `label`, for a `bool` or anything that is not
/// an `int`, and what the object's comparison with 0 raises for an integer
/// that is out of range.
pub fn classify_unsigned<T: UnsignedInteger>(
    object: &Bound<'_, PyAny>,
    label: &Label<'_>,
) -> PyResult<Reading<T>> {
    if object.is_instance_of::<PyBool>() || !object.is_instance_of::<PyInt>() {
        return Err(PyTypeError::new_err(format!(
            "{label} must be an integer, but got {}",
            object.repr()?
        )));
    }
    match T::extract_from(object) {
        Ok(value) => Ok(Reading::Value(value)),
        Err(error) if error.is_instance_of::<PyOverflowError>(object.py()) => {
            if object.lt(0)? {
                Ok(Reading::Negative)
            } else {
                Ok(Reading::TooLarge)
            }
        }
        Err(error) => Err(error),
    }
}

/// Return the `OverflowError` for the argument `label` above the maximum of
/// `T`.
#[must_use]
pub fn build_too_large_error<T: UnsignedInteger>(label: &Label<'_>) -> PyErr {
    PyOverflowError::new_err(format!(
        "{label} is too large: it must fit in {} bits",
        T::BITS
    ))
}

/// Read `object` as an unsigned integer `T` of at least `minimum`.
///
/// # Errors
///
/// Raises `TypeError` for a `bool` or a non-integer, `ValueError` for a
/// negative number, or for zero under [`Minimum::Positive`], and
/// `OverflowError` for a number above the maximum of `T`, each naming
/// `label`.
pub fn read_unsigned<T: UnsignedInteger>(
    object: &Bound<'_, PyAny>,
    label: &Label<'_>,
    minimum: Minimum,
) -> PyResult<T> {
    let value = match classify_unsigned::<T>(object, label)? {
        Reading::Value(value) => value,
        Reading::Negative => return Err(build_minimum_error(object, label, minimum)),
        Reading::TooLarge => return Err(build_too_large_error::<T>(label)),
    };
    if minimum == Minimum::Positive && object.eq(0)? {
        return Err(build_minimum_error(object, label, minimum));
    }
    Ok(value)
}

/// Return the `ValueError` for the argument `label` below `minimum`.
fn build_minimum_error(object: &Bound<'_, PyAny>, label: &Label<'_>, minimum: Minimum) -> PyErr {
    let kind = match minimum {
        Minimum::Positive => "a positive",
        Minimum::NonNegative => "a non-negative",
    };
    match object.repr() {
        Ok(repr) => {
            PyValueError::new_err(format!("{label} must be {kind} integer, but got {repr}"))
        }
        Err(error) => error,
    }
}

/// Read `object` for a predicate: the value when it is an unsigned integer
/// `T`, and `None` when it is a negative or oversized integer, which is out
/// of every range.
///
/// # Errors
///
/// Raises `TypeError` for a `bool` or a non-integer, naming `label`.
pub fn read_unsigned_lenient<T: UnsignedInteger>(
    object: &Bound<'_, PyAny>,
    label: &Label<'_>,
) -> PyResult<Option<T>> {
    Ok(match classify_unsigned(object, label)? {
        Reading::Value(value) => Some(value),
        Reading::Negative | Reading::TooLarge => None,
    })
}

#[cfg(test)]
mod tests;
