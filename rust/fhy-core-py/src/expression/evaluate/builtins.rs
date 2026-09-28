//! `BuiltinNativeImplementation`: the `implementation` of a native built-in
//! entry, which computes the core's kernel.

use num_traits::{FromPrimitive, Signed, ToPrimitive};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyFloat, PyInt, PyTuple, PyType};

use fhy_core::expression::builtins::BuiltinFunction;
use fhy_core::expression::{BigInt, FunctionSort};

use super::super::literal::{big_int_to_python, read_big_int};
use super::error::non_finite_cast_error;

/// The implementation of a native built-in function, a callable taking one
/// `int` or `float` and computing the core's kernel: IEEE results, so
/// `sqrt(-1.0)` is NaN, and for `round`, `floor` and `ceil` the exact
/// `int` of a finite result.
///
/// One object exists per native built-in; it pickles by name and compares
/// by identity.
#[pyclass(frozen, module = "fhy_core._rs", name = "BuiltinNativeImplementation")]
pub(crate) struct PyBuiltinNativeImplementation {
    function: BuiltinFunction,
}

/// The one implementation object of each native built-in, in catalogue
/// order.
static IMPLEMENTATIONS: PyOnceLock<Vec<(BuiltinFunction, Py<PyBuiltinNativeImplementation>)>> =
    PyOnceLock::new();

/// Return the implementation object of the native built-in `function`.
///
/// # Errors
///
/// Raises `ValueError` if `function` is composed.
pub(in crate::expression) fn builtin_implementation(
    py: Python<'_>,
    function: BuiltinFunction,
) -> PyResult<Bound<'_, PyBuiltinNativeImplementation>> {
    let implementations = IMPLEMENTATIONS.get_or_try_init(py, || {
        BuiltinFunction::iter()
            .filter(|function| function.composed().is_none())
            .map(|function| {
                Py::new(py, PyBuiltinNativeImplementation { function })
                    .map(|object| (function, object))
            })
            .collect::<PyResult<Vec<_>>>()
    })?;
    implementations
        .iter()
        .find(|(native, _)| *native == function)
        .map(|(_, object)| object.bind(py).clone())
        .ok_or_else(|| {
            PyValueError::new_err(format!("{} is not a native built-in", function.name()))
        })
}

/// Return the real argument `value` of `function`: an `int`, as its nearest
/// float and infinite beyond the float range, or a `float`.
fn read_argument(function: BuiltinFunction, value: &Bound<'_, PyAny>) -> PyResult<f64> {
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!(
            "{} takes a real, got bool.",
            function.name()
        )));
    }
    if value.is_instance_of::<PyInt>() {
        let integer: BigInt = read_big_int(value)?;
        return Ok(integer.to_f64().unwrap_or(if integer.is_negative() {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        }));
    }
    if let Ok(float) = value.cast::<PyFloat>() {
        return Ok(float.value());
    }
    Err(PyTypeError::new_err(format!(
        "{} takes an int or a float, got {}.",
        function.name(),
        value.get_type().name()?
    )))
}

#[pymethods]
impl PyBuiltinNativeImplementation {
    /// Return the function's value at `value`, an `int` or a `float`: a
    /// `float`, or for an integer-sorted function the exact `int`.
    ///
    /// Raises `TypeError` for any other argument, a `bool` included, and
    /// `NonFiniteCastError` for an integer-sorted result with no integer.
    fn __call__<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = value.py();
        let function = self.function;
        let argument = read_argument(function, value)?;
        let result = function
            .native_value(argument)
            .unwrap_or_else(|| unreachable!("the implementation is a native built-in's"));
        if function.result_sort() == FunctionSort::Real {
            return Ok(PyFloat::new(py, result).into_any());
        }
        let integer = BigInt::from_f64(result)
            .filter(|_| result.is_finite())
            .ok_or_else(|| {
                non_finite_cast_error(
                    py,
                    &format!("{}({result}) has no integer value", function.name()),
                )
            })?;
        big_int_to_python(py, &integer)
    }

    /// The built-in function's name.
    #[getter]
    fn __name__(&self) -> &'static str {
        self.function.name()
    }

    fn __repr__(&self) -> String {
        format!(
            "<built-in native implementation of {}>",
            self.function.name()
        )
    }

    /// Pickle as `BuiltinNativeImplementation._of(name)`, which returns the
    /// one object of the built-in.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let constructor = py.get_type::<Self>().getattr("_of")?;
        Ok((constructor, PyTuple::new(py, [slf.get().function.name()])?))
    }

    /// Return the implementation of the native built-in named `name`.
    ///
    /// Raises `ValueError` for any other name.
    #[classmethod]
    fn _of<'py>(cls: &Bound<'py, PyType>, name: &str) -> PyResult<Bound<'py, Self>> {
        let function: BuiltinFunction = name
            .parse()
            .map_err(|_unknown| PyValueError::new_err(format!("no built-in is named {name:?}")))?;
        builtin_implementation(cls.py(), function)
    }
}
