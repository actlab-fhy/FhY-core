//! `evaluate_expression_with_numpy`: the core's evaluator over `NumPy`
//! values, converted with [`crate::convert::numpy`].
//!
//! `NumPy` stays optional: every entry point imports it first, and nothing
//! else in the extension touches `NumPy`, so the extension imports and
//! works without it. The bindings the inlined expression refers to are
//! converted to the core's three domains: a Python `bool`, `int` or
//! `float` directly, and anything else through `numpy.asarray`, whose
//! `bool_`, `int64` and `float64` arrays in native byte order are borrowed
//! without a copy, and whose other admitted dtypes `NumPy` casts once. When
//! every binding is a scalar, the scalar backend evaluates; otherwise the
//! array backend does, with the interpreter released, calling `NumPy`'s
//! ufuncs for the 14 transcendental natives.

use std::collections::{HashMap, HashSet};

use numpy::PyUntypedArrayMethods;
use pyo3::prelude::*;
use pyo3::types::{PyMapping, PyModule};

use fhy_core::expression::builtins::BuiltinConstant;
use fhy_core::expression::evaluate::{ArrayBinding, ArrayKernels, Evaluator, Prepared, Scalar};
use fhy_core::expression::registry::FunctionRegistry;
use fhy_core::expression::{Callee, ExpressionKind};
use fhy_core::identifier::Identifier;

use crate::convert::numpy::{
    NumpyKernels, NumpyValue, array_value_to_numpy, evaluation_error_to_python, require_numpy,
    scalar_to_numpy, with_floating_point_warnings_silenced,
};
use crate::identifier::read_identifier_id;

use super::super::node::read_expression;
use super::super::registry;

/// Return the bindings of `environment` that `free` identifiers name,
/// converted.
fn read_bindings<'py>(
    numpy: &Bound<'py, PyModule>,
    environment: &Bound<'py, PyAny>,
    free: &HashSet<Identifier>,
) -> PyResult<Vec<(Identifier, NumpyValue<'py>)>> {
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
        bindings.push((
            (*identifier).clone(),
            NumpyValue::from_python(numpy, identifier.name_hint(), &value)?,
        ));
    }
    Ok(bindings)
}

/// Return the value of the prepared expression when it is a call of a
/// transcendental native over one real array binding: `NumPy`'s ufunc over
/// the binding itself, the kernel gives that node, with no copy. Return
/// `None` for any other expression, for a binding that is not
/// C-contiguous (the result has the binding's order), or for a binding of a
/// constant's identifier, which the evaluation refuses.
fn evaluate_lone_kernel_call<'py>(
    numpy: &Bound<'py, PyModule>,
    kernels: &NumpyKernels,
    prepared: &Prepared<'_>,
    registry: &FunctionRegistry,
    bindings: &[(Identifier, NumpyValue<'py>)],
) -> PyResult<Option<Bound<'py, PyAny>>> {
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
    let Some(array) = bindings
        .iter()
        .find(|(bound, _)| bound == identifier)
        .and_then(|(_, value)| value.real_array())
    else {
        return Ok(None);
    };
    if !array.is_c_contiguous() {
        return Ok(None);
    }
    let ufunc = numpy.getattr(function.name())?;
    with_floating_point_warnings_silenced(numpy, || ufunc.call1((array.as_any(),))).map(Some)
}

/// Evaluate the prepared expression over `bindings`, converted.
fn evaluate_bindings<'py>(
    numpy: &Bound<'py, PyModule>,
    prepared: &Prepared<'_>,
    registry: &FunctionRegistry,
    bindings: &[(Identifier, NumpyValue<'py>)],
) -> PyResult<Bound<'py, PyAny>> {
    let py = numpy.py();
    let scalars: Option<HashMap<Identifier, Scalar>> = bindings
        .iter()
        .map(|(identifier, value)| Some((identifier.clone(), value.as_scalar()?)))
        .collect();
    if let Some(environment) = scalars {
        let value = prepared
            .evaluate(&environment)
            .map_err(|error| evaluation_error_to_python(py, error))?;
        return scalar_to_numpy(numpy, value);
    }
    let values: Vec<&NumpyValue<'py>> = bindings.iter().map(|(_, value)| value).collect();
    let environment: HashMap<Identifier, ArrayBinding<'_>> = bindings
        .iter()
        .map(|(identifier, value)| (identifier.clone(), value.as_binding()))
        .collect();
    let kernels = NumpyKernels::new(numpy, &values);
    if let Some(result) = evaluate_lone_kernel_call(numpy, &kernels, prepared, registry, bindings)?
    {
        return Ok(result);
    }
    let value = py
        .detach(|| prepared.evaluate_array(&environment, &kernels))
        .map_err(|error| evaluation_error_to_python(py, error))?;
    drop(environment);
    array_value_to_numpy(numpy, value)
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
    let numpy = require_numpy(
        py,
        "evaluate_expression_with_numpy",
        "pip install fhy_core[numpy]",
    )?;
    let expression = read_expression(expression, "evaluate_expression_with_numpy", "expression")?;
    let snapshot = registry::snapshot();
    let prepared = Evaluator::new(snapshot.registry())
        .prepare(expression.get().expression())
        .map_err(|error| evaluation_error_to_python(py, error))?;
    let bindings = read_bindings(&numpy, environment, prepared.free_identifiers())?;
    evaluate_bindings(&numpy, &prepared, snapshot.registry(), &bindings)
}
