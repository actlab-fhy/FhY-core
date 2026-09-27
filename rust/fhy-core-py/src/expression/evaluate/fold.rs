//! The fold of `evaluate_expression` (D-S9-8, D-S9-15): the core's
//! `Evaluator::fold` over a registry snapshot, calling each native user
//! function's Python implementation.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString, PyTuple};

use fhy_core::expression::LiteralValue;
use fhy_core::expression::evaluate::{Evaluator, NativeCalls};
use fhy_core::expression::registry::NativeFunction;
use fhy_core::foreign::BoxError;

use super::super::literal::{literal_to_python, read_big_int};
use super::super::materialize::materialize_beside;
use super::super::node::read_expression;
use super::super::registry::{self, RegistryState};
use super::error::{fold_error_to_python, native_result_sort_error};

/// The native user functions of a registry snapshot, computed by their
/// Python implementations.
struct PythonNatives<'s, 'py> {
    py: Python<'py>,
    state: &'s RegistryState,
}

impl PythonNatives<'_, '_> {
    /// Return the literal of the Python value `value` that `function`
    /// returned: a `bool`, an `int` or a `float`.
    fn read_result(
        &self,
        function: &NativeFunction,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<LiteralValue> {
        if let Ok(boolean) = value.cast::<PyBool>() {
            return Ok(LiteralValue::Bool(boolean.is_true()));
        }
        if value.is_instance_of::<PyInt>() {
            return Ok(LiteralValue::Int(read_big_int(value)?));
        }
        if let Ok(float) = value.cast::<PyFloat>() {
            return Ok(LiteralValue::Float(float.value()));
        }
        Err(native_result_sort_error(
            self.py,
            &format!(
                "native function {:?} returned {}, which is not a bool, an int or a float",
                function.name().as_str(),
                value.repr()?
            ),
        ))
    }
}

impl NativeCalls for PythonNatives<'_, '_> {
    fn call(
        &self,
        function: &NativeFunction,
        arguments: &[LiteralValue],
    ) -> Result<LiteralValue, BoxError> {
        let py = self.py;
        let run = || -> PyResult<LiteralValue> {
            let implementation = self
                .state
                .native_implementation(py, function.name().as_str())
                .ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "native function {:?} has no implementation",
                        function.name().as_str()
                    ))
                })?;
            let values = arguments
                .iter()
                .map(|argument| literal_to_python(py, argument))
                .collect::<PyResult<Vec<_>>>()?;
            let result = implementation.call1(PyTuple::new(py, values)?)?;
            self.read_result(function, &result)
        };
        run().map_err(|error| Box::new(error) as BoxError)
    }
}

/// Fold `expression`: replace each native call whose arguments are literals
/// by the literal it computes, and each reference to a constant's
/// identifier by its value.
///
/// Returns the folded expression, `expression` itself when nothing folds,
/// and the names of the functions with a body whose calls were kept, once
/// each, in the order first reached.
///
/// Raises `EntryLookupError` for an unknown name, `FunctionArityError` for
/// a constant called or a wrong argument count, `TypeError` for an argument
/// of the wrong sort, `StringLiteralPrecisionError` for an inexact decimal,
/// `NonFiniteCastError` for an integer-sorted built-in with no integer
/// value, `NativeResultSortError` for a user native's result of the wrong
/// sort, and a user native's own exception unchanged.
#[pyfunction]
pub(crate) fn fold_expression<'py>(
    expression: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
    let py = expression.py();
    let expression = read_expression(expression, "fold_expression", "expression")?;
    let snapshot = registry::snapshot();
    let natives = PythonNatives {
        py,
        state: &snapshot,
    };
    let folding = Evaluator::new(snapshot.registry())
        .fold(expression.get().expression(), &natives)
        .map_err(|error| fold_error_to_python(py, error))?;
    let output = materialize_beside(expression, folding.output())?;
    let names = folding
        .not_inlined()
        .iter()
        .map(|callee| PyString::new(py, callee.name()));
    Ok((output, PyTuple::new(py, names)?))
}
