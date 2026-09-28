//! Decoding a V2 expression payload, a node table, in one pass (R2-N1).
//!
//! The general path turns the Python dict into a JSON value, decodes the
//! core's `Expression` from it, and then materializes a Python object per
//! node, re-reading every literal from a Python value. This fast path walks
//! the table's Python objects once, in table order, and builds each node's
//! public object as it goes: a literal from the core literal it decodes,
//! through [`literal_from_core`], and an inner node from its children's
//! objects, found by table index, so a node the table shares is built once,
//! as the general path shares it. An identifier's Python `Identifier` is
//! reused per id, as the materializer does.
//!
//! It accepts only tables of exactly the shapes the core writes, with every
//! check the core's decoder makes. Anything else, and any payload a
//! constructor refuses, falls back to the general path, which raises the
//! core's errors, so the fast path never changes which payloads decode or
//! how a malformed one fails.

use std::collections::HashMap;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt, PyList, PyString, PyTuple};

use fhy_core::expression::{
    BinaryOperation, Callee, Expression, ExpressionKind, LiteralValue, LogicalOperation,
    UnaryOperation,
};
use fhy_core::identifier::Identifier;

use crate::identifier::identifier_to_python;

use super::node::{
    PyBinaryExpression, PyCallExpression, PyExpression, PyIdentifierExpression,
    PyLogicalExpression, PyPiecewiseExpression, PyUnaryExpression, literal_from_core,
};
use super::operation::{PythonOperation, operation_to_python};

/// The objects of the table's nodes decoded so far, by index, and whether a
/// later node refers to each.
struct Decoded<'py> {
    objects: Vec<Bound<'py, PyAny>>,
    is_referenced: Vec<bool>,
    /// The Python `Identifier` of each id met so far.
    identifiers: HashMap<u64, Bound<'py, PyAny>>,
}

impl<'py> Decoded<'py> {
    /// Return the object of the child `child` of node `index`, marking it
    /// referenced, or `None` if it is no index preceding `index`.
    fn child(&mut self, index: usize, child: &Bound<'py, PyAny>) -> Option<Bound<'py, PyAny>> {
        let position = child
            .cast_exact::<PyInt>()
            .ok()?
            .extract::<usize>()
            .ok()
            .filter(|&position| position < index)?;
        self.is_referenced[position] = true;
        Some(self.objects[position].clone())
    }

    /// Return the objects of the children `children`, a list of indices, of
    /// node `index`.
    fn children(
        &mut self,
        index: usize,
        children: &Bound<'py, PyAny>,
    ) -> Option<Vec<Bound<'py, PyAny>>> {
        let children = children.cast_exact::<PyList>().ok()?;
        children
            .iter()
            .map(|child| self.child(index, &child))
            .collect()
    }
}

/// Return the fields of `body` if it is an exactly-typed dict of exactly
/// the fields `names`, in any order.
fn read_fields<'py, const N: usize>(
    body: &Bound<'py, PyAny>,
    names: [&Bound<'py, PyString>; N],
) -> PyResult<Option<[Bound<'py, PyAny>; N]>> {
    let Ok(body) = body.cast_exact::<PyDict>() else {
        return Ok(None);
    };
    if body.len() != N {
        return Ok(None);
    }
    let mut fields = Vec::with_capacity(N);
    for name in names {
        let Some(field) = body.get_item(name)? else {
            return Ok(None);
        };
        fields.push(field);
    }
    Ok(fields.try_into().ok())
}

/// Return the operation the exactly-typed `str` `name` names.
fn read_operation<T: PythonOperation + std::str::FromStr>(
    name: &Bound<'_, PyAny>,
) -> PyResult<Option<T>> {
    let Ok(text) = name.cast_exact::<PyString>() else {
        return Ok(None);
    };
    Ok(text.to_str()?.parse::<T>().ok())
}

/// Return the JSON value of a leaf field: an exactly-typed `str`, `bool` or
/// `int`, the only JSON a literal, an identifier or a callee holds.
fn read_scalar(value: &Bound<'_, PyAny>) -> PyResult<Option<serde_json::Value>> {
    if let Ok(text) = value.cast_exact::<PyString>() {
        return Ok(Some(serde_json::Value::String(text.to_str()?.to_owned())));
    }
    if let Ok(flag) = value.cast_exact::<PyBool>() {
        return Ok(Some(serde_json::Value::Bool(flag.is_true())));
    }
    if let Ok(integer) = value.cast_exact::<PyInt>() {
        return Ok(integer.extract::<u64>().ok().map(serde_json::Value::from));
    }
    Ok(None)
}

/// Decode `body`, a one-entry dict of scalars such as `{"int": "1"}`, as a
/// `T` through its serde form, which checks it as the core does.
fn read_tagged<T: serde::de::DeserializeOwned>(body: &Bound<'_, PyAny>) -> PyResult<Option<T>> {
    let Ok(body) = body.cast_exact::<PyDict>() else {
        return Ok(None);
    };
    let mut map = serde_json::Map::with_capacity(body.len());
    for (key, value) in body.iter() {
        let (Ok(key), Some(value)) = (key.cast_exact::<PyString>(), read_scalar(&value)?) else {
            return Ok(None);
        };
        map.insert(key.to_str()?.to_owned(), value);
    }
    Ok(serde_json::from_value(serde_json::Value::Object(map)).ok())
}

/// Return the public object of node `index` of the table, `node`, or `None`
/// if it is not of a node's exact shape or breaks a check the core makes.
#[expect(clippy::too_many_lines, reason = "one arm per node kind")]
fn decode_node<'py>(
    decoded: &mut Decoded<'py>,
    index: usize,
    node: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = node.py();
    let Ok(node) = node.cast_exact::<PyDict>() else {
        return Ok(None);
    };
    let Some((kind, body)) = node.iter().next().filter(|_| node.len() == 1) else {
        return Ok(None);
    };
    let Ok(kind) = kind.cast_exact::<PyString>() else {
        return Ok(None);
    };
    let object = match kind.to_str()? {
        "literal" => {
            let Some(literal) = read_tagged::<LiteralValue>(&body)? else {
                return Ok(None);
            };
            literal_from_core(py, &Expression::from(literal))?
        }
        "identifier" => {
            let Some(identifier) = read_tagged::<Identifier>(&body)? else {
                return Ok(None);
            };
            let object = if let Some(object) = decoded.identifiers.get(&identifier.id()) {
                object.clone()
            } else {
                let object = identifier_to_python(py, &identifier)?;
                decoded.identifiers.insert(identifier.id(), object.clone());
                object
            };
            PyIdentifierExpression::public_class()
                .get(py)?
                .call1((object,))?
        }
        "unary" => {
            let names = [intern!(py, "operation"), intern!(py, "operand")];
            let Some([operation, operand]) = read_fields(&body, names)? else {
                return Ok(None);
            };
            let (Some(operation), Some(operand)) = (
                read_operation::<UnaryOperation>(&operation)?,
                decoded.child(index, &operand),
            ) else {
                return Ok(None);
            };
            PyUnaryExpression::public_class()
                .get(py)?
                .call1((operation_to_python(py, operation)?, operand))?
        }
        "binary" => {
            let names = [
                intern!(py, "operation"),
                intern!(py, "left"),
                intern!(py, "right"),
            ];
            let Some([operation, left, right]) = read_fields(&body, names)? else {
                return Ok(None);
            };
            let (Some(operation), Some(left), Some(right)) = (
                read_operation::<BinaryOperation>(&operation)?,
                decoded.child(index, &left),
                decoded.child(index, &right),
            ) else {
                return Ok(None);
            };
            PyBinaryExpression::public_class().get(py)?.call1((
                operation_to_python(py, operation)?,
                left,
                right,
            ))?
        }
        "logical" => {
            let names = [intern!(py, "operation"), intern!(py, "operands")];
            let Some([operation, operands]) = read_fields(&body, names)? else {
                return Ok(None);
            };
            let (Some(operation), Some(operands)) = (
                read_operation::<LogicalOperation>(&operation)?,
                decoded.children(index, &operands),
            ) else {
                return Ok(None);
            };
            if operands.len() < 2 {
                return Ok(None);
            }
            PyLogicalExpression::public_class().get(py)?.call1((
                operation_to_python(py, operation)?,
                PyTuple::new(py, operands)?,
            ))?
        }
        "piecewise" => {
            let names = [intern!(py, "cases"), intern!(py, "otherwise")];
            let Some([cases, otherwise]) = read_fields(&body, names)? else {
                return Ok(None);
            };
            let (Ok(cases), Some(otherwise)) = (
                cases.cast_exact::<PyList>(),
                decoded.child(index, &otherwise),
            ) else {
                return Ok(None);
            };
            if cases.is_empty() {
                return Ok(None);
            }
            let mut conditions = Vec::with_capacity(cases.len());
            let mut values = Vec::with_capacity(cases.len());
            for case in cases.iter() {
                let Some([condition, value]) = decoded
                    .children(index, &case)
                    .and_then(|case| <[_; 2]>::try_from(case).ok())
                else {
                    return Ok(None);
                };
                if is_non_boolean_literal(&condition) {
                    return Ok(None);
                }
                conditions.push(condition);
                values.push(value);
            }
            PyPiecewiseExpression::public_class().get(py)?.call1((
                PyTuple::new(py, conditions)?,
                PyTuple::new(py, values)?,
                otherwise,
            ))?
        }
        "call" => {
            let names = [intern!(py, "callee"), intern!(py, "arguments")];
            let Some([callee, arguments]) = read_fields(&body, names)? else {
                return Ok(None);
            };
            let (Some(callee), Some(arguments)) = (
                read_tagged::<Callee>(&callee)?,
                decoded.children(index, &arguments),
            ) else {
                return Ok(None);
            };
            PyCallExpression::public_class()
                .get(py)?
                .call1((callee.name(), PyTuple::new(py, arguments)?))?
        }
        _ => return Ok(None),
    };
    Ok(Some(object))
}

/// Return whether `object`, an expression object, is a literal other than a
/// Boolean, which no piecewise case condition may be.
fn is_non_boolean_literal(object: &Bound<'_, PyAny>) -> bool {
    object.cast::<PyExpression>().is_ok_and(|expression| {
        matches!(
            expression.get().expression().kind(),
            ExpressionKind::Literal(literal) if !matches!(literal, LiteralValue::Bool(_))
        )
    })
}

/// Return the public object of the expression the V2 table `data` encodes,
/// decoded in one pass, or `None` if `data` is not a table of exactly the
/// core's shapes, breaks a check the core's decoder makes, or holds a node
/// a constructor refuses; the caller then decodes it through the core.
///
/// Identifiers read before a refusal have advanced the id counter, as they
/// do in the core's decoder.
pub(super) fn decode_table<'py>(data: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = data.py();
    let Ok(data) = data.cast_exact::<PyDict>() else {
        return Ok(None);
    };
    if data.len() != 1 {
        return Ok(None);
    }
    let Some(nodes) = data.get_item(intern!(py, "nodes"))? else {
        return Ok(None);
    };
    let Ok(nodes) = nodes.cast_exact::<PyList>() else {
        return Ok(None);
    };
    let count = nodes.len();
    if count == 0 {
        return Ok(None);
    }
    let mut decoded = Decoded {
        objects: Vec::with_capacity(count),
        is_referenced: Vec::with_capacity(count),
        identifiers: HashMap::new(),
    };
    for (index, node) in nodes.iter().enumerate() {
        let Ok(Some(object)) = decode_node(&mut decoded, index, &node) else {
            return Ok(None);
        };
        decoded.objects.push(object);
        decoded.is_referenced.push(false);
    }
    if decoded.is_referenced[..count - 1].contains(&false) {
        return Ok(None);
    }
    Ok(decoded.objects.pop())
}
