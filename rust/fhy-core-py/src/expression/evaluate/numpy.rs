//! `evaluate_expression_with_numpy` (D-S9-10 to D-S9-13): the core's
//! evaluator over `NumPy` values, converted with rust-numpy.
//!
//! `NumPy` stays optional (D-S9-11): every entry point imports it first, and
//! nothing else in the extension touches `NumPy`, so the extension imports
//! and works without it. The bindings the inlined expression refers to are
//! converted to the core's three domains: a Python `bool`, `int` or
//! `float` directly, and anything else through `numpy.asarray`, whose
//! `bool_`, `int64` and `float64` arrays in native byte order are borrowed
//! without a copy, and whose other admitted dtypes `NumPy` casts once. When
//! every binding is a scalar, the scalar backend evaluates; otherwise the
//! array backend does, with the interpreter released, calling `NumPy`'s
//! ufuncs for the 14 transcendental natives (N-S9-2 (b)).

use std::collections::{HashMap, HashSet};

use numpy::ndarray::{ArrayD, CowArray, IxDyn, arr0};
use numpy::{PyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyImportError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyMapping, PyModule, PySlice};

use fhy_core::expression::builtins::{BuiltinConstant, BuiltinFunction};
use fhy_core::expression::evaluate::{
    ArrayBinding, ArrayKernels, ArrayValue, Evaluator, Prepared, Scalar,
};
use fhy_core::expression::registry::FunctionRegistry;
use fhy_core::expression::{Callee, ExpressionKind};
use fhy_core::foreign::BoxError;
use fhy_core::identifier::Identifier;

use crate::identifier::read_identifier_id;

use super::super::node::read_expression;
use super::super::registry;
use super::error::evaluation_error_to_python;

/// Return the `NumPy` module, or raise `ImportError` with the extra to
/// install, caused by the import's failure.
fn import_numpy(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import(intern!(py, "numpy")).map_err(|error| {
        let guidance = PyImportError::new_err(
            "NumPy is required for `evaluate_expression_with_numpy`; install it with \
             `pip install fhy_core[numpy]`.",
        );
        guidance.set_cause(py, Some(error));
        guidance
    })
}

/// A binding converted to one of the core's domains.
enum Converted<'py> {
    /// A scalar.
    Scalar(Scalar),
    /// An array of Booleans, borrowed.
    Bool(PyReadonlyArrayDyn<'py, bool>),
    /// An array of 64-bit integers, borrowed.
    Int(PyReadonlyArrayDyn<'py, i64>),
    /// An array of reals, borrowed.
    Real(PyReadonlyArrayDyn<'py, f64>),
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

/// Return the scalar of a 0-d array's one element.
fn only_element<T: numpy::Element + Copy>(array: &PyReadonlyArrayDyn<'_, T>) -> T {
    *array
        .as_array()
        .first()
        .unwrap_or_else(|| unreachable!("a 0-d array has one element"))
}

/// Return the binding `value` of `identifier` converted.
///
/// # Errors
///
/// Raises `OverflowError` for an integer outside the 64-bit range, and
/// `TypeError` for a dtype that is not Boolean, integer or floating-point,
/// or a floating-point dtype wider than 64 bits.
fn convert<'py>(
    numpy: &Bound<'py, PyModule>,
    identifier: &Identifier,
    value: &Bound<'py, PyAny>,
) -> PyResult<Converted<'py>> {
    let py = value.py();
    if let Ok(boolean) = value.cast::<PyBool>() {
        return Ok(Converted::Scalar(Scalar::Bool(boolean.is_true())));
    }
    if value.is_instance_of::<PyInt>() {
        return value
            .extract::<i64>()
            .map(|integer| Converted::Scalar(Scalar::Int(integer)))
            .map_err(|_out_of_range| {
                PyOverflowError::new_err(format!(
                    "the int bound to {:?} is outside the 64-bit range",
                    identifier.name_hint()
                ))
            });
    }
    if let Ok(float) = value.cast::<PyFloat>() {
        return Ok(Converted::Scalar(Scalar::Real(float.value())));
    }
    let array = numpy.call_method1(intern!(py, "asarray"), (value,))?;
    let dtype = array.getattr(intern!(py, "dtype"))?;
    let kind: String = dtype.getattr(intern!(py, "kind"))?.extract()?;
    let itemsize: usize = dtype.getattr(intern!(py, "itemsize"))?.extract()?;
    let refuse = || -> PyResult<Converted<'py>> {
        Err(PyTypeError::new_err(format!(
            "the value bound to {:?} has dtype {}, which is not boolean, integer or \
             floating-point of at most 64 bits",
            identifier.name_hint(),
            dtype.str()?
        )))
    };
    let converted = match kind.as_str() {
        "b" => Converted::Bool(borrow(&typed_array::<bool>(numpy, &array, "bool_")?)?),
        "i" | "u" if itemsize <= 8 => {
            if kind == "u"
                && itemsize == 8
                && array.getattr(intern!(py, "size"))?.extract::<usize>()? > 0
            {
                let maximum = array.call_method0(intern!(py, "max"))?;
                if maximum.gt(i64::MAX)? {
                    return Err(PyOverflowError::new_err(format!(
                        "the uint64 array bound to {:?} holds a value above the 64-bit \
                         signed range",
                        identifier.name_hint()
                    )));
                }
            }
            Converted::Int(borrow(&typed_array::<i64>(numpy, &array, "int64")?)?)
        }
        "f" if itemsize <= 8 => {
            Converted::Real(borrow(&typed_array::<f64>(numpy, &array, "float64")?)?)
        }
        _ => return refuse(),
    };
    Ok(match converted {
        Converted::Bool(array) if array.ndim() == 0 => {
            Converted::Scalar(Scalar::Bool(only_element(&array)))
        }
        Converted::Int(array) if array.ndim() == 0 => {
            Converted::Scalar(Scalar::Int(only_element(&array)))
        }
        Converted::Real(array) if array.ndim() == 0 => {
            Converted::Scalar(Scalar::Real(only_element(&array)))
        }
        other => other,
    })
}

/// Return the bindings of `environment` that `free` identifiers name,
/// converted.
fn read_bindings<'py>(
    numpy: &Bound<'py, PyModule>,
    environment: &Bound<'py, PyAny>,
    free: &HashSet<Identifier>,
) -> PyResult<Vec<(Identifier, Converted<'py>)>> {
    let by_id: HashMap<u64, &Identifier> = free
        .iter()
        .map(|identifier| (identifier.id(), identifier))
        .collect();
    let mut bindings = Vec::new();
    for item in environment.cast::<PyMapping>()?.items()?.iter() {
        let (key, value) = item.extract::<(Bound<'py, PyAny>, Bound<'py, PyAny>)>()?;
        let Some(identifier) = read_identifier_id(&key)?.and_then(|id| by_id.get(&id)) else {
            continue;
        };
        bindings.push(((*identifier).clone(), convert(numpy, identifier, &value)?));
    }
    Ok(bindings)
}

/// A 0-d array of a scalar binding, for an array evaluation.
enum ZeroDimensional {
    Bool(ArrayD<bool>),
    Int(ArrayD<i64>),
    Real(ArrayD<f64>),
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

/// The array kernels computing the 14 transcendental natives with `NumPy`'s
/// ufuncs (N-S9-2 (b)); `sqrt`, `round`, `floor`, `ceil` and `erf` keep the
/// core's.
struct NumpyKernels {
    numpy: Py<PyModule>,
    inputs: Vec<HeldInput>,
}

impl NumpyKernels {
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
/// `numpy.float64`.
fn scalar_to_numpy<'py>(
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

/// Return the `NumPy` array of `value`, which takes over its buffer, or the
/// `NumPy` scalar of a 0-d one.
fn array_to_numpy<'py>(
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

/// Return the value of the prepared expression when it is a call of a
/// transcendental native over one real array binding: `NumPy`'s ufunc over
/// the binding itself, the kernel N-S9-2 (b) gives that node, with no
/// copy. Return `None` for any other expression, for a binding that is not
/// C-contiguous (the result has the binding's order), or for a binding of a
/// constant's identifier, which the evaluation refuses.
fn evaluate_lone_kernel_call<'py>(
    numpy: &Bound<'py, PyModule>,
    kernels: &NumpyKernels,
    prepared: &Prepared<'_>,
    registry: &FunctionRegistry,
    bindings: &[(Identifier, Converted<'py>)],
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = numpy.py();
    let ExpressionKind::Call(call) = prepared.expression().kind() else {
        return Ok(None);
    };
    let (Callee::Builtin(function), [argument]) = (call.callee(), call.arguments()) else {
        return Ok(None);
    };
    let ExpressionKind::Identifier(identifier) = argument.kind() else {
        return Ok(None);
    };
    if !kernels.handles(*function)
        || BuiltinConstant::of_identifier(identifier).is_some()
        || registry.constant(identifier).is_some()
    {
        return Ok(None);
    }
    let Some((_, Converted::Real(array))) = bindings.iter().find(|(bound, _)| bound == identifier)
    else {
        return Ok(None);
    };
    if !array.is_c_contiguous() {
        return Ok(None);
    }
    let ufunc = numpy.getattr(function.name())?;
    let settings = PyDict::new(py);
    settings.set_item(intern!(py, "all"), intern!(py, "ignore"))?;
    let silenced = numpy
        .getattr(intern!(py, "errstate"))?
        .call((), Some(&settings))?;
    silenced.call_method0(intern!(py, "__enter__"))?;
    let result = ufunc.call1((array.as_any(),));
    silenced.call_method1(intern!(py, "__exit__"), (py.None(), py.None(), py.None()))?;
    Ok(Some(result?))
}

/// Evaluate the prepared expression over `bindings`, converted.
fn evaluate_bindings<'py>(
    numpy: &Bound<'py, PyModule>,
    prepared: &Prepared<'_>,
    registry: &FunctionRegistry,
    bindings: Vec<(Identifier, Converted<'py>)>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = numpy.py();
    if bindings
        .iter()
        .all(|(_, binding)| matches!(binding, Converted::Scalar(_)))
    {
        let environment: HashMap<Identifier, Scalar> = bindings
            .into_iter()
            .map(|(identifier, binding)| match binding {
                Converted::Scalar(value) => (identifier, value),
                _ => unreachable!("every binding is a scalar"),
            })
            .collect();
        let value = prepared
            .evaluate(&environment)
            .map_err(|error| evaluation_error_to_python(py, error))?;
        return scalar_to_numpy(numpy, value);
    }
    let zero_dimensional: Vec<(Identifier, ZeroDimensional)> = bindings
        .iter()
        .filter_map(|(identifier, binding)| match binding {
            Converted::Scalar(Scalar::Bool(value)) => Some((
                identifier.clone(),
                ZeroDimensional::Bool(arr0(*value).into_dyn()),
            )),
            Converted::Scalar(Scalar::Int(value)) => Some((
                identifier.clone(),
                ZeroDimensional::Int(arr0(*value).into_dyn()),
            )),
            Converted::Scalar(Scalar::Real(value)) => Some((
                identifier.clone(),
                ZeroDimensional::Real(arr0(*value).into_dyn()),
            )),
            _ => None,
        })
        .collect();
    let mut inputs = Vec::new();
    let mut environment: HashMap<Identifier, ArrayBinding<'_>> = HashMap::new();
    for (identifier, binding) in &bindings {
        let array = match binding {
            Converted::Scalar(_) => continue,
            Converted::Bool(array) => ArrayBinding::Bool(array.as_array()),
            Converted::Int(array) => ArrayBinding::Int(array.as_array()),
            Converted::Real(array) => {
                let view = array.as_array();
                if view.is_standard_layout() {
                    inputs.push(HeldInput {
                        start: view.as_ptr() as usize,
                        length: view.len(),
                        array: array.as_any().clone().unbind(),
                    });
                }
                ArrayBinding::Real(view)
            }
        };
        environment.insert(identifier.clone(), array);
    }
    for (identifier, array) in &zero_dimensional {
        let binding = match array {
            ZeroDimensional::Bool(array) => ArrayBinding::Bool(array.view()),
            ZeroDimensional::Int(array) => ArrayBinding::Int(array.view()),
            ZeroDimensional::Real(array) => ArrayBinding::Real(array.view()),
        };
        environment.insert(identifier.clone(), binding);
    }
    let kernels = NumpyKernels {
        numpy: numpy.clone().unbind(),
        inputs,
    };
    if let Some(result) = evaluate_lone_kernel_call(numpy, &kernels, prepared, registry, &bindings)?
    {
        return Ok(result);
    }
    let value = py
        .detach(|| prepared.evaluate_array(&environment, &kernels))
        .map_err(|error| evaluation_error_to_python(py, error))?;
    drop(environment);
    array_to_numpy(numpy, value)
}

/// Evaluate `expression` to a `NumPy` value over `environment`, a mapping
/// of identifiers to anything `numpy.asarray` accepts.
///
/// Composed built-ins and user functions are inlined first. Only the
/// bindings the inlined expression refers to are read. A 0-d result is a
/// `NumPy` scalar; any other result is a new array.
///
/// Raises `ImportError` without `NumPy`, and the errors of the core's
/// evaluation under their Python classes, directly.
#[pyfunction]
pub(crate) fn evaluate_expression_with_numpy<'py>(
    expression: &Bound<'py, PyAny>,
    environment: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = expression.py();
    let numpy = import_numpy(py)?;
    let expression = read_expression(expression, "evaluate_expression_with_numpy", "expression")?;
    let snapshot = registry::snapshot();
    let prepared = Evaluator::new(snapshot.registry())
        .prepare(expression.get().expression())
        .map_err(|error| evaluation_error_to_python(py, error))?;
    let bindings = read_bindings(&numpy, environment, prepared.free_identifiers())?;
    evaluate_bindings(&numpy, &prepared, snapshot.registry(), bindings)
}
