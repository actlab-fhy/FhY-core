//! Decoding a whole V1 expression payload in one pass.
//!
//! V1: removed with the V1 wire format.
//!
//! The framework's `WrappedFamilySerializable.deserialize_from_dict`
//! decodes one node per call and checks, at every node, that the node's
//! whole nested payload is a serialized dict, so a tree of depth `d` costs
//! time quadratic in `d`. The fast path here walks the nested envelopes
//! once, with its pending nodes on the heap, and builds each node through
//! its public class, bottom-up.
//!
//! It accepts only payloads of exactly the shapes the expression classes
//! write: envelopes of the seven expression type ids, and data dicts with
//! exactly their fields, of exactly their types. Anything else, including
//! any payload a constructor refuses, falls back to the framework's own
//! path, which raises the framework's errors, so the fast path never
//! changes which payloads decode or how a malformed one fails.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple, PyType};

use fhy_core::expression::{BinaryOperation, LogicalOperation, UnaryOperation};

use crate::identifier::deserialize_identifier;

use super::node::{
    PyBinaryExpression, PyCallExpression, PyIdentifierExpression, PyLiteralExpression,
    PyLogicalExpression, PyPiecewiseExpression, PyUnaryExpression,
};
use super::operation::{PythonOperation, operation_to_python};

/// A node whose children are being decoded, built once they are.
enum Recipe<'py> {
    Unary(Bound<'py, PyAny>),
    Binary(Bound<'py, PyAny>),
    Logical(Bound<'py, PyAny>, usize),
    Piecewise(usize),
    Call(Bound<'py, PyString>, usize),
}

/// One step of the decoding walk.
enum Step<'py> {
    Decode(Bound<'py, PyAny>),
    Build(Recipe<'py>),
}

/// A decoded node: a leaf's object, or an inner node's recipe and the
/// payloads of its children, in visiting order.
enum Decoded<'py> {
    Leaf(Bound<'py, PyAny>),
    Inner(Recipe<'py>, Vec<Bound<'py, PyAny>>),
}

/// Return `data` as a dict of exactly `size` entries, or `None`.
fn read_data<'a, 'py>(data: &'a Bound<'py, PyAny>, size: usize) -> Option<&'a Bound<'py, PyDict>> {
    data.cast_exact::<PyDict>()
        .ok()
        .filter(|data| data.len() == size)
}

/// Return the operation member named by the `str` `name`, or `None`.
fn read_operation<T: PythonOperation + std::str::FromStr>(
    name: Option<Bound<'_, PyAny>>,
) -> PyResult<Option<Bound<'_, PyAny>>> {
    let Some(name) = name else {
        return Ok(None);
    };
    let Ok(text) = name.cast_exact::<PyString>() else {
        return Ok(None);
    };
    match text.to_str()?.parse::<T>() {
        Ok(operation) => operation_to_python(name.py(), operation).map(Some),
        Err(_unknown) => Ok(None),
    }
}

/// Return whether `value` is an exactly-typed dict.
fn is_exact_dict(value: &Bound<'_, PyAny>) -> bool {
    value.is_exact_instance_of::<PyDict>()
}

/// Return the items of `value` if it is a list of exactly-typed dicts.
fn read_payload_list(value: Option<Bound<'_, PyAny>>) -> Option<Vec<Bound<'_, PyAny>>> {
    let value = value?;
    let list = value.cast_exact::<PyList>().ok()?;
    let items: Vec<_> = list.iter().collect();
    items.iter().all(is_exact_dict).then_some(items)
}

/// Return `value` if it is an exactly-typed dict.
fn read_payload(value: Option<Bound<'_, PyAny>>) -> Option<Bound<'_, PyAny>> {
    value.filter(is_exact_dict)
}

/// Decode the data of an identifier reference.
fn decode_identifier<'py>(data: &Bound<'py, PyAny>) -> PyResult<Option<Decoded<'py>>> {
    let py = data.py();
    let Some(data) = read_data(data, 1) else {
        return Ok(None);
    };
    let Some(identifier) = read_payload(data.get_item(intern!(py, "identifier"))?) else {
        return Ok(None);
    };
    let identifier = deserialize_identifier(&identifier)?;
    let node = PyIdentifierExpression::public_class()
        .get(py)?
        .call1((identifier,))?;
    Ok(Some(Decoded::Leaf(node)))
}

/// Decode the data of a literal.
fn decode_literal<'py>(data: &Bound<'py, PyAny>) -> PyResult<Option<Decoded<'py>>> {
    let py = data.py();
    let Some(data) = read_data(data, 1) else {
        return Ok(None);
    };
    let Some(value) = data.get_item(intern!(py, "value"))? else {
        return Ok(None);
    };
    let is_literal_value = value.is_exact_instance_of::<PyBool>()
        || value.is_exact_instance_of::<PyInt>()
        || value.is_exact_instance_of::<PyFloat>()
        || value.is_exact_instance_of::<PyString>();
    if !is_literal_value {
        return Ok(None);
    }
    let node = PyLiteralExpression::public_class()
        .get(py)?
        .call1((value,))?;
    Ok(Some(Decoded::Leaf(node)))
}

/// Decode the data of a unary, binary or logical node.
fn decode_operation_node<'py>(
    type_id: &str,
    data: &Bound<'py, PyAny>,
) -> PyResult<Option<Decoded<'py>>> {
    let py = data.py();
    let size = match type_id {
        "binary_expression" => 3,
        _ => 2,
    };
    let Some(data) = read_data(data, size) else {
        return Ok(None);
    };
    let operation = data.get_item(intern!(py, "operation"))?;
    let decoded = match type_id {
        "unary_expression" => {
            let operation = read_operation::<UnaryOperation>(operation)?;
            let operand = read_payload(data.get_item(intern!(py, "operand"))?);
            operation
                .zip(operand)
                .map(|(operation, operand)| Decoded::Inner(Recipe::Unary(operation), vec![operand]))
        }
        "binary_expression" => {
            let operation = read_operation::<BinaryOperation>(operation)?;
            let left = read_payload(data.get_item(intern!(py, "left"))?);
            let right = read_payload(data.get_item(intern!(py, "right"))?);
            match (operation, left, right) {
                (Some(operation), Some(left), Some(right)) => {
                    Some(Decoded::Inner(Recipe::Binary(operation), vec![left, right]))
                }
                _ => None,
            }
        }
        _ => {
            let operation = read_operation::<LogicalOperation>(operation)?;
            let operands = read_payload_list(data.get_item(intern!(py, "operands"))?);
            operation.zip(operands).map(|(operation, operands)| {
                Decoded::Inner(Recipe::Logical(operation, operands.len()), operands)
            })
        }
    };
    Ok(decoded)
}

/// Decode the data of a piecewise.
fn decode_piecewise<'py>(data: &Bound<'py, PyAny>) -> PyResult<Option<Decoded<'py>>> {
    let py = data.py();
    let Some(data) = read_data(data, 3) else {
        return Ok(None);
    };
    let conditions = read_payload_list(data.get_item(intern!(py, "conditions"))?);
    let values = read_payload_list(data.get_item(intern!(py, "values"))?);
    let otherwise = read_payload(data.get_item(intern!(py, "otherwise"))?);
    let (Some(conditions), Some(values), Some(otherwise)) = (conditions, values, otherwise) else {
        return Ok(None);
    };
    if conditions.len() != values.len() {
        return Ok(None);
    }
    let case_count = conditions.len();
    let mut children: Vec<_> = conditions.into_iter().chain(values).collect();
    children.push(otherwise);
    Ok(Some(Decoded::Inner(
        Recipe::Piecewise(case_count),
        children,
    )))
}

/// Decode the data of a call.
fn decode_call<'py>(data: &Bound<'py, PyAny>) -> PyResult<Option<Decoded<'py>>> {
    let py = data.py();
    let Some(data) = read_data(data, 2) else {
        return Ok(None);
    };
    let name = data
        .get_item(intern!(py, "function_name"))?
        .and_then(|name| name.cast_exact::<PyString>().ok().cloned());
    let arguments = read_payload_list(data.get_item(intern!(py, "arguments"))?);
    Ok(name
        .zip(arguments)
        .map(|(name, arguments)| Decoded::Inner(Recipe::Call(name, arguments.len()), arguments)))
}

/// Decode the envelope `payload` of one node, or return `None` if it is
/// not of an expression class's shape.
fn decode_node<'py>(payload: &Bound<'py, PyAny>) -> PyResult<Option<Decoded<'py>>> {
    let py = payload.py();
    let Ok(envelope) = payload.cast_exact::<PyDict>() else {
        return Ok(None);
    };
    if envelope.len() != 2 {
        return Ok(None);
    }
    let (Some(type_id), Some(data)) = (
        envelope.get_item(intern!(py, "__type__"))?,
        envelope.get_item(intern!(py, "__data__"))?,
    ) else {
        return Ok(None);
    };
    let Ok(type_id) = type_id.cast_exact::<PyString>() else {
        return Ok(None);
    };
    match type_id.to_str()? {
        "identifier_expression" => decode_identifier(&data),
        "literal_expression" => decode_literal(&data),
        type_id @ ("unary_expression" | "binary_expression" | "logical_expression") => {
            decode_operation_node(type_id, &data)
        }
        "piecewise_expression" => decode_piecewise(&data),
        "call_expression" => decode_call(&data),
        _ => Ok(None),
    }
}

/// Build the node of `recipe` from the last results, its children.
fn build<'py>(
    py: Python<'py>,
    recipe: Recipe<'py>,
    results: &mut Vec<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut take = |count: usize| results.split_off(results.len() - count);
    match recipe {
        Recipe::Unary(operation) => {
            let [operand]: [_; 1] = take(1)
                .try_into()
                .unwrap_or_else(|_| unreachable!("one child"));
            PyUnaryExpression::public_class()
                .get(py)?
                .call1((operation, operand))
        }
        Recipe::Binary(operation) => {
            let [left, right]: [_; 2] = take(2)
                .try_into()
                .unwrap_or_else(|_| unreachable!("two children"));
            PyBinaryExpression::public_class()
                .get(py)?
                .call1((operation, left, right))
        }
        Recipe::Logical(operation, count) => PyLogicalExpression::public_class()
            .get(py)?
            .call1((operation, PyTuple::new(py, take(count))?)),
        Recipe::Piecewise(case_count) => {
            let mut children = take(2 * case_count + 1);
            let otherwise = children
                .pop()
                .unwrap_or_else(|| unreachable!("an otherwise child"));
            let values = children.split_off(case_count);
            PyPiecewiseExpression::public_class().get(py)?.call1((
                PyTuple::new(py, children)?,
                PyTuple::new(py, values)?,
                otherwise,
            ))
        }
        Recipe::Call(name, count) => PyCallExpression::public_class()
            .get(py)?
            .call1((name, PyTuple::new(py, take(count))?)),
    }
}

/// Return the expression the payload `data` encodes, decoded in one pass,
/// or `None` if `data` is not of the expression classes' exact shapes or a
/// constructor refuses it.
fn decode_fast<'py>(
    cls: &Bound<'py, PyType>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = cls.py();
    let mut results = Vec::new();
    let mut pending = vec![Step::Decode(data.clone())];
    while let Some(step) = pending.pop() {
        match step {
            Step::Decode(payload) => match decode_node(&payload)? {
                None => return Ok(None),
                Some(Decoded::Leaf(node)) => results.push(node),
                Some(Decoded::Inner(recipe, children)) => {
                    pending.push(Step::Build(recipe));
                    pending.extend(children.into_iter().rev().map(Step::Decode));
                }
            },
            Step::Build(recipe) => {
                let node = build(py, recipe, &mut results)?;
                results.push(node);
            }
        }
    }
    let root = results
        .pop()
        .unwrap_or_else(|| unreachable!("the walk decodes the root"));
    Ok(root.is_instance(cls)?.then_some(root))
}

/// Return `WrappedFamilySerializable.deserialize_from_dict`'s function.
fn framework_decoder(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    static FUNCTION: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    FUNCTION
        .get_or_try_init(py, || {
            let class = py
                .import(intern!(py, "fhy_core.serialization"))?
                .getattr(intern!(py, "WrappedFamilySerializable"))?;
            let method = class
                .getattr(intern!(py, "__dict__"))?
                .get_item(intern!(py, "deserialize_from_dict"))?;
            Ok::<_, PyErr>(method.getattr(intern!(py, "__func__"))?.unbind())
        })
        .map(|function| function.bind(py))
}

/// Return the expression of the payload `data` as an instance of `cls`,
/// through the fast path when it applies and the framework's path
/// otherwise.
///
/// # Errors
///
/// Raises what the framework's path raises for a payload the fast path
/// declines.
pub(super) fn deserialize_expression_payload<'py>(
    cls: &Bound<'py, PyType>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(Some(expression)) = decode_fast(cls, data) {
        return Ok(expression);
    }
    framework_decoder(cls.py())?.call1((cls, data))
}
