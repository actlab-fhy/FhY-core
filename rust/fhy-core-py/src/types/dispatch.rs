//! The functions the six dispatchers of `fhy_core.types.dispatch` run for
//! the built-in classes and as their defaults, `unify_expression`, and the
//! promotion helpers of `fhy_core.types.core` (D-S11-8, D-S11-9, D-S11-12).
//!
//! Each dispatcher function converts its arguments, runs the core in a
//! context, and hands the result back as the objects it came from where the
//! core left it unchanged.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyFloat, PyInt, PyTuple, PyType};

use fhy_core::expression::LiteralValue;
use fhy_core::types::{
    CoreDataType, DataType, Type, TypeOperation, UnificationError, unify_expressions,
};

use crate::dataclass::build_argument_type_error;
use crate::error::{IntoPyErr, IntoPyResult};
use crate::expression::read_big_int;

use super::adapter::run_in_context;
use super::classes::{MayCallPython, PyPrimitiveDataType, build_not_a_type_error};
use super::convert::{
    data_type_to_python, environment_to_python, expression_to_python, read_data_type,
    read_data_type_value, read_environment, read_expression, read_type, read_type_value,
    type_to_python,
};
use super::enums::{
    core_data_type_to_python, read_core_data_type, read_type_qualifier, type_qualifier_to_python,
};
use super::environment::PyTypeUnificationEnvironment;

/// Return `environment` as an environment, or raise the `TypeError` naming
/// `function`.
fn read_environment_argument<'a, 'py>(
    environment: &'a Bound<'py, PyAny>,
    function: &str,
) -> PyResult<&'a Bound<'py, PyTypeUnificationEnvironment>> {
    match environment.cast::<PyTypeUnificationEnvironment>() {
        Ok(environment) => Ok(environment),
        Err(_not_an_environment) => Err(build_argument_type_error(
            function,
            "environment",
            "a TypeUnificationEnvironment",
            environment,
        )?),
    }
}

/// Return the `VerificationError` of two values that are not structurally
/// equivalent, one of them no type, as the dispatchers' defaults refuse
/// them.
fn structural_mismatch(
    what: &str,
    joiner: &str,
    pattern: &Bound<'_, PyAny>,
    actual: &Bound<'_, PyAny>,
) -> PyResult<PyErr> {
    Ok(verification_error(format!(
        "cannot {what} {} {joiner} {}: structural mismatch",
        pattern.repr()?,
        actual.repr()?
    )))
}

/// Return the `VerificationError` with `message`.
fn verification_error(message: String) -> PyErr {
    static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    Python::attach(|py| {
        match CLASS
            .import(py, "fhy_core.traits.verifiable", "VerificationError")
            .and_then(|class| class.call1((message,)))
        {
            Ok(error) => PyErr::from_value(error),
            Err(error) => error,
        }
    })
}

/// Return the kind mismatch of a built-in `pattern` meeting `actual`, which
/// is no value of its tier.
fn kind_mismatch(
    operation: TypeOperation,
    expected: String,
    actual: &Bound<'_, PyAny>,
) -> PyResult<PyErr> {
    Ok(UnificationError::KindMismatch {
        operation,
        expected,
        actual: actual.get_type().name()?.to_string(),
    }
    .into_py_err())
}

/// Return whether `left` and `right` are structurally equivalent: two types
/// or two data types, compared by the core and the handlers of
/// Python-defined parts. Anything else is not.
#[pyfunction]
pub(crate) fn types_is_structurally_equivalent(
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let py = left.py();
    if let Some(left) = read_type_value(left) {
        let Some(right) = read_type_value(right) else {
            return Ok(false);
        };
        if !left.may_call_python() && !right.may_call_python() {
            return left
                .is_structurally_equivalent(&right)
                .map_err(IntoPyErr::into_py_err);
        }
        return run_in_context(py, None, |_context| {
            left.is_structurally_equivalent(&right)
                .map_err(IntoPyErr::into_py_err)
        });
    }
    if let Some(left) = read_data_type_value(left) {
        let Some(right) = read_data_type_value(right) else {
            return Ok(false);
        };
        if !left.may_call_python() && !right.may_call_python() {
            return left
                .is_structurally_equivalent(&right)
                .map_err(IntoPyErr::into_py_err);
        }
        return run_in_context(py, None, |_context| {
            left.is_structurally_equivalent(&right)
                .map_err(IntoPyErr::into_py_err)
        });
    }
    Ok(false)
}

/// Bind the type `pattern` against `actual` in `environment`, and return the
/// environment of the bindings learned.
///
/// Raises `VerificationError` when they cannot be bound, and `TypeError`
/// when `environment` is no environment.
#[pyfunction]
pub(crate) fn types_bind_template<'py>(
    pattern: &Bound<'py, PyAny>,
    actual: &Bound<'py, PyAny>,
    environment: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = pattern.py();
    let environment = read_environment_argument(environment, "bind_template")?;
    run_in_context(py, Some(environment), |context| {
        let rust_environment = read_environment(context, environment.as_any())
            .unwrap_or_else(|| unreachable!("the argument is an environment"));
        let rust_pattern = read_type(context, pattern)?;
        let rust_actual = read_type(context, actual)?;
        let bound = match (rust_pattern, rust_actual) {
            (Some(rust_pattern), Some(rust_actual)) => rust_pattern
                .bind_template(&rust_actual, &rust_environment)
                .into_py_result()?,
            (Some(rust_pattern @ (Type::Numerical(_) | Type::Index(_))), None) => {
                return Err(kind_mismatch(
                    TypeOperation::Bind,
                    rust_pattern.kind_name(),
                    actual,
                )?);
            }
            _ => return Err(structural_mismatch("bind", "against", pattern, actual)?),
        };
        environment_to_python(py, context, &bound)
    })
}

/// Return the type `type_` with the placeholders `environment` binds
/// substituted.
///
/// Raises `TypeError` when `type_` is no `Type` or `environment` no
/// environment.
#[pyfunction]
pub(crate) fn types_substitute_template<'py>(
    type_: &Bound<'py, PyAny>,
    environment: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = type_.py();
    let environment = read_environment_argument(environment, "substitute_template")?;
    run_in_context(py, Some(environment), |context| {
        let rust_environment = read_environment(context, environment.as_any())
            .unwrap_or_else(|| unreachable!("the argument is an environment"));
        let Some(rust_type) = read_type(context, type_)? else {
            return Err(build_not_a_type_error(
                "substitute_template",
                "Type",
                type_,
            )?);
        };
        let substituted = rust_type
            .substitute_template(&rust_environment)
            .into_py_result()?;
        type_to_python(py, context, &substituted)
    })
}

/// Unify the type `expected` with `actual` in `environment`, and return the
/// unified type and the environment of the bindings learned.
///
/// Raises `VerificationError` when they cannot be unified, and `TypeError`
/// when `environment` is no environment.
#[pyfunction]
pub(crate) fn types_unify<'py>(
    expected: &Bound<'py, PyAny>,
    actual: &Bound<'py, PyAny>,
    environment: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let py = expected.py();
    let environment = read_environment_argument(environment, "unify")?;
    run_in_context(py, Some(environment), |context| {
        let rust_environment = read_environment(context, environment.as_any())
            .unwrap_or_else(|| unreachable!("the argument is an environment"));
        let rust_expected = read_type(context, expected)?;
        let rust_actual = read_type(context, actual)?;
        let (unified, next) = match (rust_expected, rust_actual) {
            (Some(rust_expected), Some(rust_actual)) => rust_expected
                .unify(&rust_actual, &rust_environment)
                .into_py_result()?,
            (Some(rust_expected @ (Type::Numerical(_) | Type::Index(_))), None) => {
                return Err(kind_mismatch(
                    TypeOperation::Unify,
                    rust_expected.kind_name(),
                    actual,
                )?);
            }
            _ => return Err(structural_mismatch("unify", "with", expected, actual)?),
        };
        PyTuple::new(
            py,
            [
                type_to_python(py, context, &unified)?,
                environment_to_python(py, context, &next)?,
            ],
        )
    })
}

/// Bind the data type `pattern` against `actual` in `environment`, and
/// return the environment of the bindings learned.
///
/// Raises `VerificationError` when they cannot be bound, and `TypeError`
/// when `environment` is no environment.
#[pyfunction]
pub(crate) fn types_bind_data_template<'py>(
    pattern: &Bound<'py, PyAny>,
    actual: &Bound<'py, PyAny>,
    environment: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = pattern.py();
    let environment = read_environment_argument(environment, "bind_data_template")?;
    run_in_context(py, Some(environment), |context| {
        let rust_environment = read_environment(context, environment.as_any())
            .unwrap_or_else(|| unreachable!("the argument is an environment"));
        let rust_pattern = read_data_type(context, pattern)?;
        let rust_actual = read_data_type(context, actual)?;
        let bound = match (rust_pattern, rust_actual) {
            (Some(rust_pattern), Some(rust_actual)) => rust_pattern
                .bind_template(&rust_actual, &rust_environment)
                .into_py_result()?,
            (Some(rust_pattern @ (DataType::Primitive(_) | DataType::Template(_))), None) => {
                return Err(kind_mismatch(
                    TypeOperation::Bind,
                    rust_pattern.kind_name(),
                    actual,
                )?);
            }
            _ => {
                return Err(structural_mismatch(
                    "bind data type",
                    "against",
                    pattern,
                    actual,
                )?);
            }
        };
        environment_to_python(py, context, &bound)
    })
}

/// Return the data type `data_type` with its placeholders substituted from
/// `environment`.
///
/// Raises `TypeError` when `data_type` is no `DataType` or `environment` no
/// environment.
#[pyfunction]
pub(crate) fn types_substitute_data_template<'py>(
    data_type: &Bound<'py, PyAny>,
    environment: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = data_type.py();
    let environment = read_environment_argument(environment, "substitute_data_template")?;
    run_in_context(py, Some(environment), |context| {
        let rust_environment = read_environment(context, environment.as_any())
            .unwrap_or_else(|| unreachable!("the argument is an environment"));
        let Some(rust_data_type) = read_data_type(context, data_type)? else {
            return Err(build_not_a_type_error(
                "substitute_data_template",
                "DataType",
                data_type,
            )?);
        };
        let substituted = rust_data_type
            .substitute_template(&rust_environment)
            .into_py_result()?;
        data_type_to_python(py, context, &substituted)
    })
}

/// Unify the expressions `left` and `right` in `environment`, and return
/// the unified expression and the environment of the bindings learned.
///
/// Raises `VerificationError` when they cannot be unified, and `TypeError`
/// for an argument of the wrong type.
#[pyfunction]
pub(crate) fn types_unify_expression<'py>(
    left: &Bound<'py, PyAny>,
    right: &Bound<'py, PyAny>,
    environment: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let py = left.py();
    let environment = read_environment_argument(environment, "unify_expression")?;
    run_in_context(py, Some(environment), |context| {
        let rust_environment = read_environment(context, environment.as_any())
            .unwrap_or_else(|| unreachable!("the argument is an environment"));
        let Some(rust_left) = read_expression(context, left) else {
            return Err(build_argument_type_error(
                "unify_expression",
                "left",
                "an Expression",
                left,
            )?);
        };
        let Some(rust_right) = read_expression(context, right) else {
            return Err(build_argument_type_error(
                "unify_expression",
                "right",
                "an Expression",
                right,
            )?);
        };
        let (unified, next) =
            unify_expressions(&rust_left, &rust_right, &rust_environment).into_py_result()?;
        PyTuple::new(
            py,
            [
                expression_to_python(py, context, &unified)?,
                environment_to_python(py, context, &next)?,
            ],
        )
    })
}

// ---------------------------------------------------------------------------
// Promotion and literals
// ---------------------------------------------------------------------------

/// Return the bit width of `core_data_type`, or `None` for a weak type.
#[pyfunction]
pub(crate) fn get_core_data_type_bit_width(
    core_data_type: &Bound<'_, PyAny>,
) -> PyResult<Option<u32>> {
    Ok(read_core_data_type(
        core_data_type,
        "get_core_data_type_bit_width",
        "core_data_type",
    )?
    .bit_width())
}

/// Return whether `core_data_type` is a weak literal type.
#[pyfunction]
pub(crate) fn is_weak_core_data_type(core_data_type: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(read_core_data_type(core_data_type, "is_weak_core_data_type", "core_data_type")?.is_weak())
}

/// Return the core data type both arguments promote to.
///
/// Raises `FhYCoreTypeError` when they have none.
#[pyfunction]
pub(crate) fn promote_core_data_types<'py>(
    core_data_type1: &Bound<'py, PyAny>,
    core_data_type2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let left = read_core_data_type(
        core_data_type1,
        "promote_core_data_types",
        "core_data_type1",
    )?;
    let right = read_core_data_type(
        core_data_type2,
        "promote_core_data_types",
        "core_data_type2",
    )?;
    core_data_type_to_python(core_data_type1.py(), left.promote(right).into_py_result()?)
}

/// Return the primitive data type both primitive data types promote to.
///
/// Raises `FhYCoreTypeError` when they have none, and `TypeError` for an
/// argument that is no `PrimitiveDataType`.
#[pyfunction]
pub(crate) fn promote_primitive_data_types<'py>(
    primitive_data_type1: &Bound<'py, PyAny>,
    primitive_data_type2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = primitive_data_type1.py();
    let read = |value: &Bound<'py, PyAny>, field: &str| -> PyResult<CoreDataType> {
        match value.cast::<PyPrimitiveDataType>() {
            Ok(primitive) => match primitive.get().value() {
                DataType::Primitive(core_data_type) => Ok(core_data_type),
                _ => unreachable!("a primitive data type holds a core data type"),
            },
            Err(_not_primitive) => Err(build_argument_type_error(
                "promote_primitive_data_types",
                field,
                "a PrimitiveDataType",
                value,
            )?),
        }
    };
    let left = read(primitive_data_type1, "primitive_data_type1")?;
    let right = read(primitive_data_type2, "primitive_data_type2")?;
    let promoted = left.promote(right).into_py_result()?;
    PyPrimitiveDataType::public_class()
        .get(py)?
        .call1((core_data_type_to_python(py, promoted)?,))
}

/// Return the qualifier of a value computed from values of the two
/// qualifiers.
#[pyfunction]
pub(crate) fn promote_type_qualifiers<'py>(
    type_qualifier1: &Bound<'py, PyAny>,
    type_qualifier2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let left = read_type_qualifier(
        type_qualifier1,
        "promote_type_qualifiers",
        "type_qualifier1",
    )?;
    let right = read_type_qualifier(
        type_qualifier2,
        "promote_type_qualifiers",
        "type_qualifier2",
    )?;
    type_qualifier_to_python(type_qualifier1.py(), left.promote(right))
}

/// Return the concrete core data type the literal `literal` takes in the
/// context `core_data_type`.
///
/// Raises `FhYCoreTypeError` for a literal the context cannot hold, and
/// `TypeError` for a literal that is no `bool`, `int` or `float`.
#[pyfunction]
pub(crate) fn resolve_literal_core_data_type<'py>(
    literal: &Bound<'py, PyAny>,
    core_data_type: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let context = read_core_data_type(
        core_data_type,
        "resolve_literal_core_data_type",
        "core_data_type",
    )?;
    let value = if let Ok(boolean) = literal.cast::<PyBool>() {
        LiteralValue::Bool(boolean.is_true())
    } else if literal.is_instance_of::<PyInt>() {
        LiteralValue::Int(read_big_int(literal)?)
    } else if let Ok(float) = literal.cast::<PyFloat>() {
        LiteralValue::Float(float.value())
    } else {
        return Err(PyTypeError::new_err(format!(
            "resolve_literal_core_data_type literal must be a bool, int or float, got {}.",
            literal.get_type().name()?
        )));
    };
    let resolved = CoreDataType::resolve_literal(&value, context).into_py_result()?;
    core_data_type_to_python(literal.py(), resolved)
}
