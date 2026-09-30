//! The conversions between `NumPy` values and `fhy-core`'s evaluation values,
//! the ones `evaluate_expression_with_numpy` is written over, for a
//! downstream binding crate that evaluates over `NumPy` arrays.
//!
//! `NumPy` stays optional: [`require_numpy`] imports it and is the only
//! place that raises for its absence, and nothing else in the extension
//! touches it. No `rust-numpy` type appears in a signature here, so a
//! downstream crate needs no `numpy` dependency of its own, and no version
//! has to match.
//!
//! # The three domains
//!
//! The core's evaluator computes in three domains: Booleans (`bool`), 64-bit
//! signed integers (`i64`) and binary64 reals (`f64`). A value is either a
//! [`Scalar`] of one of them or an array ([`ArrayValue`], [`ArrayBinding`])
//! of one of them. [`NumpyValue::from_python`] converts as follows:
//!
//! - A Python `bool` is a Boolean scalar, an `int` an integer scalar and a
//!   `float` a real scalar. An `int` outside the 64-bit range raises
//!   `OverflowError`. Python `bool` is checked before `int`, so `True` is a
//!   Boolean.
//! - Anything else goes through `numpy.asarray`, and the dtype's kind picks
//!   the domain: `bool_` is Boolean; every signed or unsigned integer dtype
//!   of at most 64 bits is an integer; every floating-point dtype of at most
//!   64 bits (`float16`, `float32`, `float64`) is real. Any other dtype
//!   (complex, `longdouble`, object, string, datetime, ...) raises
//!   `TypeError`.
//! - An array whose dtype is already `bool_`, `int64` or `float64` in native
//!   byte order is *borrowed*, with no copy, however it is strided. Every
//!   other admitted dtype (`int8`, `uint32`, `float32`, a non-native byte
//!   order, ...) is cast once with `astype`, into a new array the value
//!   owns.
//! - A `uint64` array holding a value above `i64::MAX` raises
//!   `OverflowError`; the other unsigned dtypes fit.
//! - A 0-d array is a scalar of its domain, as a Python number is.
//!
//! [`array_value_to_numpy`] and [`scalar_to_numpy`] convert back, to
//! `numpy.bool_`, `numpy.int64` and `numpy.float64`.
//!
//! A borrowed array is read in place, so a caller that releases the
//! interpreter while it evaluates must not let another thread write the
//! array meanwhile.

use std::fmt;

use numpy::ndarray::{ArrayD, CowArray, IxDyn, arr0};
use numpy::{PyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyImportError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyModule, PySlice};

use fhy_core::expression::builtins::BuiltinFunction;
use fhy_core::expression::evaluate::{
    ArrayBinding, ArrayKernels, ArrayValue, EvaluationError, Scalar,
};
use fhy_core::foreign::BoxError;

/// Import `NumPy`, or raise `ImportError` naming `entry_point` and the
/// `install` hint, caused by the import's failure.
///
/// The text is ``NumPy is required for `{entry_point}`; install it with
/// `{install}`.``, so `require_numpy(py, "evaluate_expression_with_numpy",
/// "pip install fhy_core[numpy]")` raises the error `fhy_core` itself does.
///
/// # Errors
///
/// Raises `ImportError`, with the import's own error as its `__cause__`,
/// when `numpy` cannot be imported.
pub fn require_numpy<'py>(
    py: Python<'py>,
    entry_point: &str,
    install: &str,
) -> PyResult<Bound<'py, PyModule>> {
    py.import(intern!(py, "numpy")).map_err(|error| {
        let guidance = PyImportError::new_err(format!(
            "NumPy is required for `{entry_point}`; install it with `{install}`."
        ));
        guidance.set_cause(py, Some(error));
        guidance
    })
}

/// What a [`NumpyValue`] holds.
enum Held<'py> {
    /// A scalar, with its 0-d array for [`NumpyValue::as_binding`].
    Scalar(Scalar, ArrayValue),
    /// An array of Booleans, borrowed.
    Bool(PyReadonlyArrayDyn<'py, bool>),
    /// An array of 64-bit integers, borrowed.
    Int(PyReadonlyArrayDyn<'py, i64>),
    /// An array of reals, borrowed.
    Real(PyReadonlyArrayDyn<'py, f64>),
}

/// One value converted to the core's three domains: a scalar, or an array
/// that is borrowed from `NumPy` or cast once.
///
/// It keeps the `NumPy` array it reads alive, so [`as_binding`] gives a view
/// of that array's own buffer. See the [module documentation](self) for
/// which values are scalars, which arrays are borrowed and which are cast.
///
/// [`as_binding`]: NumpyValue::as_binding
pub struct NumpyValue<'py> {
    held: Held<'py>,
}

impl fmt::Debug for NumpyValue<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.held {
            Held::Scalar(scalar, _) => f.debug_tuple("NumpyValue::Scalar").field(scalar).finish(),
            Held::Bool(array) => f
                .debug_tuple("NumpyValue::Bool")
                .field(&array.shape())
                .finish(),
            Held::Int(array) => f
                .debug_tuple("NumpyValue::Int")
                .field(&array.shape())
                .finish(),
            Held::Real(array) => f
                .debug_tuple("NumpyValue::Real")
                .field(&array.shape())
                .finish(),
        }
    }
}

/// Return `array` as a `NumPy` array of `T`, casting it with `astype` to the
/// `NumPy` dtype named `dtype` unless it already is one in native byte order.
fn typed_array<'py, T: numpy::Element>(
    numpy: &Bound<'py, PyModule>,
    array: &Bound<'py, PyAny>,
    dtype: &str,
) -> PyResult<Bound<'py, PyArrayDyn<T>>> {
    if let Ok(typed) = array.cast::<PyArrayDyn<T>>() {
        return Ok(typed.clone());
    }
    let cast = array.call_method1(intern!(numpy.py(), "astype"), (numpy.getattr(dtype)?,))?;
    Ok(cast.cast_into::<PyArrayDyn<T>>()?)
}

/// Return the read-only borrow of `array`.
fn borrow<'py, T: numpy::Element>(
    array: &Bound<'py, PyArrayDyn<T>>,
) -> PyResult<PyReadonlyArrayDyn<'py, T>> {
    array
        .try_readonly()
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Return the one element of a 0-d array.
fn only_element<T: numpy::Element + Copy>(array: &PyReadonlyArrayDyn<'_, T>) -> T {
    *array
        .as_array()
        .first()
        .unwrap_or_else(|| unreachable!("a 0-d array has one element"))
}

/// Return the 0-d array of `scalar`.
fn zero_dimensional(scalar: Scalar) -> ArrayValue {
    match scalar {
        Scalar::Bool(value) => ArrayValue::Bool(arr0(value).into_dyn()),
        Scalar::Int(value) => ArrayValue::Int(arr0(value).into_dyn()),
        Scalar::Real(value) => ArrayValue::Real(arr0(value).into_dyn()),
    }
}

impl<'py> NumpyValue<'py> {
    /// Return the scalar `scalar` as a value.
    fn scalar(scalar: Scalar) -> Self {
        Self {
            held: Held::Scalar(scalar, zero_dimensional(scalar)),
        }
    }

    /// Return `value`, bound to `label`, converted to the core's domains.
    ///
    /// `label` names the binding in the error texts, which quote it: it is
    /// an identifier's name hint for `evaluate_expression_with_numpy`. The
    /// conversion rules are the [module documentation](self)'s: a Python
    /// `bool`, `int` or `float` is a scalar, and anything else goes through
    /// `numpy.asarray`, whose `bool_`, `int64` and `float64` arrays in
    /// native byte order are borrowed and whose other admitted dtypes are
    /// cast once. A 0-d array is a scalar.
    ///
    /// # Errors
    ///
    /// Raises `OverflowError` for an `int` outside the 64-bit range and for
    /// a `uint64` array holding a value above `i64::MAX`, `TypeError` for a
    /// dtype that is not Boolean, integer or floating-point of at most 64
    /// bits, `ValueError` when the array is already borrowed mutably, and
    /// whatever `numpy.asarray` or `astype` raises.
    pub fn from_python(
        numpy: &Bound<'py, PyModule>,
        label: &str,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Self> {
        let py = value.py();
        if let Ok(boolean) = value.cast::<PyBool>() {
            return Ok(Self::scalar(Scalar::Bool(boolean.is_true())));
        }
        if value.is_instance_of::<PyInt>() {
            return value
                .extract::<i64>()
                .map(|integer| Self::scalar(Scalar::Int(integer)))
                .map_err(|_out_of_range| {
                    PyOverflowError::new_err(format!(
                        "the int bound to {label:?} is outside the 64-bit range"
                    ))
                });
        }
        if let Ok(float) = value.cast::<PyFloat>() {
            return Ok(Self::scalar(Scalar::Real(float.value())));
        }
        let array = numpy.call_method1(intern!(py, "asarray"), (value,))?;
        let dtype = array.getattr(intern!(py, "dtype"))?;
        let kind: String = dtype.getattr(intern!(py, "kind"))?.extract()?;
        let itemsize: usize = dtype.getattr(intern!(py, "itemsize"))?.extract()?;
        let held = match kind.as_str() {
            "b" => Held::Bool(borrow(&typed_array::<bool>(numpy, &array, "bool_")?)?),
            "i" | "u" if itemsize <= 8 => {
                if kind == "u"
                    && itemsize == 8
                    && array.getattr(intern!(py, "size"))?.extract::<usize>()? > 0
                {
                    let maximum = array.call_method0(intern!(py, "max"))?;
                    if maximum.gt(i64::MAX)? {
                        return Err(PyOverflowError::new_err(format!(
                            "the uint64 array bound to {label:?} holds a value above the \
                             64-bit signed range"
                        )));
                    }
                }
                Held::Int(borrow(&typed_array::<i64>(numpy, &array, "int64")?)?)
            }
            "f" if itemsize <= 8 => {
                Held::Real(borrow(&typed_array::<f64>(numpy, &array, "float64")?)?)
            }
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "the value bound to {label:?} has dtype {}, which is not boolean, integer \
                     or floating-point of at most 64 bits",
                    dtype.str()?
                )));
            }
        };
        Ok(match held {
            Held::Bool(array) if array.ndim() == 0 => {
                Self::scalar(Scalar::Bool(only_element(&array)))
            }
            Held::Int(array) if array.ndim() == 0 => {
                Self::scalar(Scalar::Int(only_element(&array)))
            }
            Held::Real(array) if array.ndim() == 0 => {
                Self::scalar(Scalar::Real(only_element(&array)))
            }
            held => Self { held },
        })
    }

    /// Return the binding the array evaluator takes: a view of the array,
    /// borrowed from `NumPy` or cast, or a 0-d view of a scalar.
    #[must_use]
    pub fn as_binding(&self) -> ArrayBinding<'_> {
        match &self.held {
            Held::Scalar(_, zero) => match zero {
                ArrayValue::Bool(array) => ArrayBinding::Bool(array.view()),
                ArrayValue::Int(array) => ArrayBinding::Int(array.view()),
                ArrayValue::Real(array) => ArrayBinding::Real(array.view()),
            },
            Held::Bool(array) => ArrayBinding::Bool(array.as_array()),
            Held::Int(array) => ArrayBinding::Int(array.as_array()),
            Held::Real(array) => ArrayBinding::Real(array.as_array()),
        }
    }

    /// Return an owned copy of the value, for an API that takes
    /// [`ArrayValue`]s: the array copied, or the 0-d array of a scalar.
    #[must_use]
    pub fn to_array_value(&self) -> ArrayValue {
        match &self.held {
            Held::Scalar(_, zero) => zero.clone(),
            Held::Bool(array) => ArrayValue::Bool(array.as_array().to_owned()),
            Held::Int(array) => ArrayValue::Int(array.as_array().to_owned()),
            Held::Real(array) => ArrayValue::Real(array.as_array().to_owned()),
        }
    }

    /// Return the scalar, when the value is one: a Python number or a 0-d
    /// array. An array of any other shape, even of one element, is not.
    #[must_use]
    pub const fn as_scalar(&self) -> Option<Scalar> {
        match &self.held {
            Held::Scalar(scalar, _) => Some(*scalar),
            _ => None,
        }
    }

    /// Return the borrowed or cast array of reals, when the value is one.
    pub(crate) const fn real_array(&self) -> Option<&PyReadonlyArrayDyn<'py, f64>> {
        match &self.held {
            Held::Real(array) => Some(array),
            _ => None,
        }
    }
}

/// A real array binding `NumPy` already holds, in the standard layout:
/// where its lanes are, and the Python array.
struct HeldInput {
    /// The address of the first lane.
    start: usize,
    /// The number of lanes.
    length: usize,
    array: Py<PyAny>,
}

/// The [`ArrayKernels`] computing the 14 transcendental natives (`exp`,
/// `exp2`, `log`, `log2`, `log10`, `sin`, `cos`, `tan`, `arcsin`, `arccos`,
/// `arctan`, `sinh`, `cosh`, `tanh`) with `NumPy`'s ufuncs, with `NumPy`'s
/// floating-point warnings silenced. `sqrt`, `round`, `floor`, `ceil` and
/// `erf` keep the core's own kernels.
///
/// A kernel reuses a binding's own `NumPy` array, with no copy, when the
/// lanes it is given are a run of one of the `inputs` given to
/// [`new`](Self::new), and copies any other lanes into a `NumPy` array. It
/// takes the interpreter for itself, so an evaluation may release it.
pub struct NumpyKernels {
    numpy: Py<PyModule>,
    inputs: Vec<HeldInput>,
}

impl fmt::Debug for NumpyKernels {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NumpyKernels")
            .field("inputs", &self.inputs.len())
            .finish_non_exhaustive()
    }
}

impl NumpyKernels {
    /// Return the kernels over `numpy`, which may reuse the array of each
    /// real, standard-layout array in `inputs` for lanes that are a run of
    /// it.
    ///
    /// `inputs` are the values the evaluation binds, and only the reals the
    /// kernels could reuse are kept; a scalar, Boolean or integer value is
    /// ignored. The kernels are correct for any `inputs`, and `inputs` only
    /// spare a copy.
    #[must_use]
    pub fn new(numpy: &Bound<'_, PyModule>, inputs: &[&NumpyValue<'_>]) -> Self {
        let inputs = inputs
            .iter()
            .filter_map(|value| {
                let array = value.real_array()?;
                let view = array.as_array();
                view.is_standard_layout().then(|| HeldInput {
                    start: view.as_ptr() as usize,
                    length: view.len(),
                    array: array.as_any().clone().unbind(),
                })
            })
            .collect();
        Self {
            numpy: numpy.clone().unbind(),
            inputs,
        }
    }

    /// Return a `NumPy` view of the lanes of `argument`, when they are a
    /// contiguous run of a binding's: the whole binding, or the slice of it
    /// a chunk of the evaluation reads.
    fn find_input<'py>(
        &self,
        py: Python<'py>,
        argument: &CowArray<'_, f64, IxDyn>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        if !argument.is_view() || !argument.is_standard_layout() {
            return Ok(None);
        }
        let start = argument.as_ptr() as usize;
        let lane_size = std::mem::size_of::<f64>();
        let Some(input) = self.inputs.iter().find(|input| {
            start >= input.start
                && (start - input.start) % lane_size == 0
                && (start - input.start) / lane_size + argument.len() <= input.length
        }) else {
            return Ok(None);
        };
        let offset = (start - input.start) / lane_size;
        let array = input.array.bind(py);
        if offset == 0 && argument.len() == input.length {
            return Ok(Some(array.clone()));
        }
        let flat = array.call_method1(intern!(py, "reshape"), (-1,))?;
        let lanes = flat.get_item(PySlice::new(
            py,
            isize::try_from(offset)?,
            isize::try_from(offset + argument.len())?,
            1,
        ))?;
        Ok(Some(lanes.call_method1(
            intern!(py, "reshape"),
            (argument.shape().to_vec(),),
        )?))
    }

    /// Return the lanes of `NumPy`'s ufunc for `function` over `argument`,
    /// with `NumPy`'s floating-point warnings silenced.
    ///
    /// A binding `NumPy` holds is passed as its own array. Any other lanes
    /// are copied into a `NumPy` array, which the ufunc overwrites; an owned
    /// temporary then receives the result back in place, so no buffer is
    /// allocated on the Rust side.
    fn compute(
        &self,
        py: Python<'_>,
        function: BuiltinFunction,
        argument: CowArray<'_, f64, IxDyn>,
    ) -> PyResult<ArrayD<f64>> {
        let numpy = self.numpy.bind(py);
        let ufunc = numpy.getattr(function.name())?;
        let settings = PyDict::new(py);
        settings.set_item(intern!(py, "all"), intern!(py, "ignore"))?;
        let silenced = numpy
            .getattr(intern!(py, "errstate"))?
            .call((), Some(&settings))?;
        silenced.call_method0(intern!(py, "__enter__"))?;
        let result = self.apply(py, &ufunc, argument);
        silenced.call_method1(intern!(py, "__exit__"), (py.None(), py.None(), py.None()))?;
        result
    }

    /// Return the lanes of `ufunc` over `argument`.
    fn apply(
        &self,
        py: Python<'_>,
        ufunc: &Bound<'_, PyAny>,
        argument: CowArray<'_, f64, IxDyn>,
    ) -> PyResult<ArrayD<f64>> {
        if let Some(input) = self.find_input(py, &argument)? {
            let numpy = self.numpy.bind(py);
            let result = numpy.call_method1(intern!(py, "asarray"), (ufunc.call1((input,))?,))?;
            return Ok(typed_array::<f64>(numpy, &result, "float64")?.to_owned_array());
        }
        let lanes = PyArray::from_array(py, &argument);
        let settings = PyDict::new(py);
        settings.set_item(intern!(py, "out"), &lanes)?;
        ufunc.call((&lanes,), Some(&settings))?;
        let computed = borrow(&lanes)?;
        if argument.is_view() {
            return Ok(computed.as_array().to_owned());
        }
        let mut owned = argument.into_owned();
        owned.assign(&computed.as_array());
        Ok(owned)
    }
}

impl ArrayKernels for NumpyKernels {
    fn handles(&self, function: BuiltinFunction) -> bool {
        matches!(
            function,
            BuiltinFunction::Exp
                | BuiltinFunction::Exp2
                | BuiltinFunction::Log
                | BuiltinFunction::Log2
                | BuiltinFunction::Log10
                | BuiltinFunction::Sin
                | BuiltinFunction::Cos
                | BuiltinFunction::Tan
                | BuiltinFunction::Arcsin
                | BuiltinFunction::Arccos
                | BuiltinFunction::Arctan
                | BuiltinFunction::Sinh
                | BuiltinFunction::Cosh
                | BuiltinFunction::Tanh
        )
    }

    fn native(
        &self,
        function: BuiltinFunction,
        argument: CowArray<'_, f64, IxDyn>,
    ) -> Result<ArrayD<f64>, BoxError> {
        Python::attach(|py| self.compute(py, function, argument))
            .map_err(|error| Box::new(error) as BoxError)
    }
}

/// Return the `NumPy` scalar of `value`: a `numpy.bool_`, `numpy.int64` or
/// `numpy.float64`, by its domain.
///
/// # Errors
///
/// Raises whatever the `NumPy` scalar type's constructor raises.
pub fn scalar_to_numpy<'py>(
    numpy: &Bound<'py, PyModule>,
    value: Scalar,
) -> PyResult<Bound<'py, PyAny>> {
    let py = numpy.py();
    match value {
        Scalar::Bool(value) => numpy.getattr(intern!(py, "bool_"))?.call1((value,)),
        Scalar::Int(value) => numpy.getattr(intern!(py, "int64"))?.call1((value,)),
        Scalar::Real(value) => numpy.getattr(intern!(py, "float64"))?.call1((value,)),
    }
}

/// Return the `NumPy` value of `value`: an array of `bool_`, `int64` or
/// `float64`, by its domain, which takes over the array's buffer with no
/// copy, or the `NumPy` scalar ([`scalar_to_numpy`]) of a 0-d one.
///
/// # Errors
///
/// Raises whatever `NumPy` raises reading the new array's dimensions or
/// indexing its one element.
pub fn array_value_to_numpy<'py>(
    numpy: &Bound<'py, PyModule>,
    value: ArrayValue,
) -> PyResult<Bound<'py, PyAny>> {
    let py = numpy.py();
    let array = match value {
        ArrayValue::Bool(array) => PyArray::from_owned_array(py, array).into_any(),
        ArrayValue::Int(array) => PyArray::from_owned_array(py, array).into_any(),
        ArrayValue::Real(array) => PyArray::from_owned_array(py, array).into_any(),
    };
    if array.getattr(intern!(py, "ndim"))?.extract::<usize>()? == 0 {
        return array.get_item(());
    }
    Ok(array)
}

/// Return the Python exception of the evaluation error `error`: the core's
/// text under the classes `fhy_core`'s Python API documents, the ones
/// `evaluate_expression_with_numpy` raises.
///
/// Errors of the expression map to the exception classes of
/// `fhy_core.symbolic.expression.errors` (unbound variable, bound constant,
/// inexact decimal, unsupported lowering, non-finite cast, and the inliner's
/// and screen's errors); `IntegerOutOfRange` and a lane's overflow or
/// out-of-range cast are `OverflowError`; a Boolean used as a number, or
/// mixed branches, `TypeError`; a shape that does not broadcast, or a
/// broadcast that is too large, `ValueError`; a lane's division by zero
/// `ZeroDivisionError`; `OutOfMemory` `MemoryError`; and a kernel's failure
/// is the Python exception the kernel raised, unchanged. Anything else is
/// `RuntimeError`.
///
/// The `fhy_core` classes are imported on first use, so building one of
/// them needs `fhy_core` importable; when it is not, the returned error is
/// the `ImportError` of that import.
#[must_use]
pub fn evaluation_error_to_python(py: Python<'_>, error: EvaluationError) -> PyErr {
    crate::expression::evaluation_error_to_python(py, error)
}
