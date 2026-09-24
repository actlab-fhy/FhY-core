//! Serialization of expressions in their wire shape.
//!
//! Decoding reads the whole payload into a JSON value, checks it into an
//! [`ExpressionPayload`] with every identifier still unrestored, and only
//! then builds the expression, restoring the identifiers, so a refused
//! payload leaves the identifier id counter untouched.

use std::fmt;
use std::str::FromStr;

use num_bigint::BigInt;
use num_traits::ToPrimitive;
use serde::de::{self, Deserializer};
use serde::ser::{self, SerializeSeq, SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};

use crate::decode;
use crate::identifier::{Identifier, IdentifierWire};

use super::callee::{Callee, FunctionNameError};
use super::literal::{Decimal, LiteralValue};
use super::node::{
    BinaryExpression, CallExpression, Expression, ExpressionKind, LogicalExpression,
    PiecewiseExpression, UnaryExpression, validate_case_count, validate_condition_literal,
};
use super::operation::{BinaryOperation, LogicalOperation, UnaryOperation};

/// Key of the type id in a node's wire map.
const TYPE_KEY: &str = "__type__";

/// Key of the fields in a node's wire map.
const DATA_KEY: &str = "__data__";

/// Type id of a unary node.
const UNARY_TYPE_ID: &str = "unary_expression";

/// Type id of a binary node.
const BINARY_TYPE_ID: &str = "binary_expression";

/// Type id of a logical node.
const LOGICAL_TYPE_ID: &str = "logical_expression";

/// Type id of an identifier reference.
const IDENTIFIER_TYPE_ID: &str = "identifier_expression";

/// Type id of a literal.
const LITERAL_TYPE_ID: &str = "literal_expression";

/// Type id of a piecewise node.
const PIECEWISE_TYPE_ID: &str = "piecewise_expression";

/// Type id of a call.
const CALL_TYPE_ID: &str = "call_expression";

/// Return the wire type id of `expression`'s node kind.
fn find_type_id(expression: &Expression) -> &'static str {
    match expression.kind() {
        ExpressionKind::Unary(_) => UNARY_TYPE_ID,
        ExpressionKind::Binary(_) => BINARY_TYPE_ID,
        ExpressionKind::Logical(_) => LOGICAL_TYPE_ID,
        ExpressionKind::Identifier(_) => IDENTIFIER_TYPE_ID,
        ExpressionKind::Literal(_) => LITERAL_TYPE_ID,
        ExpressionKind::Piecewise(_) => PIECEWISE_TYPE_ID,
        ExpressionKind::Call(_) => CALL_TYPE_ID,
    }
}

/// The `__data__` fields of a node.
struct NodeFields<'a>(&'a Expression);

/// A literal's `value` field.
struct LiteralWire<'a>(&'a LiteralValue);

/// A piecewise's `conditions` field or its `values` field.
struct CaseColumn<'a> {
    cases: &'a [(Expression, Expression)],
    is_condition: bool,
}

impl Serialize for NodeFields<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0.kind() {
            ExpressionKind::Unary(node) => {
                let mut fields = serializer.serialize_struct("UnaryExpression", 2)?;
                fields.serialize_field("operation", &node.operation())?;
                fields.serialize_field("operand", node.operand())?;
                fields.end()
            }
            ExpressionKind::Binary(node) => {
                let mut fields = serializer.serialize_struct("BinaryExpression", 3)?;
                fields.serialize_field("operation", &node.operation())?;
                fields.serialize_field("left", node.left())?;
                fields.serialize_field("right", node.right())?;
                fields.end()
            }
            ExpressionKind::Logical(node) => {
                let mut fields = serializer.serialize_struct("LogicalExpression", 2)?;
                fields.serialize_field("operation", &node.operation())?;
                fields.serialize_field("operands", node.operands())?;
                fields.end()
            }
            ExpressionKind::Identifier(identifier) => {
                let mut fields = serializer.serialize_struct("IdentifierExpression", 1)?;
                fields.serialize_field("identifier", identifier)?;
                fields.end()
            }
            ExpressionKind::Literal(value) => {
                let mut fields = serializer.serialize_struct("LiteralExpression", 1)?;
                fields.serialize_field("value", &LiteralWire(value))?;
                fields.end()
            }
            ExpressionKind::Piecewise(node) => {
                let mut fields = serializer.serialize_struct("PiecewiseExpression", 3)?;
                fields.serialize_field(
                    "conditions",
                    &CaseColumn {
                        cases: node.cases(),
                        is_condition: true,
                    },
                )?;
                fields.serialize_field(
                    "values",
                    &CaseColumn {
                        cases: node.cases(),
                        is_condition: false,
                    },
                )?;
                fields.serialize_field("otherwise", node.otherwise())?;
                fields.end()
            }
            ExpressionKind::Call(node) => {
                let mut fields = serializer.serialize_struct("CallExpression", 2)?;
                fields.serialize_field("function_name", node.callee().name())?;
                fields.serialize_field("arguments", node.arguments())?;
                fields.end()
            }
        }
    }
}

impl Serialize for CaseColumn<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut column = serializer.serialize_seq(Some(self.cases.len()))?;
        for (condition, value) in self.cases {
            column.serialize_element(if self.is_condition { condition } else { value })?;
        }
        column.end()
    }
}

impl Serialize for LiteralWire<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            LiteralValue::Bool(value) => serializer.serialize_bool(*value),
            LiteralValue::Int(value) => {
                if let Some(small) = value.to_i64() {
                    serializer.serialize_i64(small)
                } else if let Some(unsigned) = value.to_u64() {
                    serializer.serialize_u64(unsigned)
                } else {
                    Number::from_str(&value.to_string())
                        .map_err(ser::Error::custom)?
                        .serialize(serializer)
                }
            }
            LiteralValue::Float(value) if value.is_finite() => serializer.serialize_f64(*value),
            LiteralValue::Float(value) => Err(ser::Error::custom(format_args!(
                "the float literal {value} has no wire form"
            ))),
            LiteralValue::Decimal(value) => serializer.collect_str(value),
        }
    }
}

/// Every node serializes as a two-key map `{"__type__": <type id>,
/// "__data__": <fields>}`, with these type ids and fields, in this order:
///
/// | Type id | Fields |
/// |---|---|
/// | `unary_expression` | `operation` (wire name), `operand` |
/// | `binary_expression` | `operation` (wire name), `left`, `right` |
/// | `logical_expression` | `operation` (wire name), `operands` (list of at least two) |
/// | `identifier_expression` | `identifier` (`{"id", "name_hint"}`) |
/// | `literal_expression` | `value` |
/// | `piecewise_expression` | `conditions` (list), `values` (list), `otherwise` |
/// | `call_expression` | `function_name`, `arguments` (list) |
///
/// A literal's `value` is a JSON Boolean, an integer of any size written as
/// a JSON integer, a float written as a JSON float, or a decimal written as
/// a string holding its `Display` text. Deserializing reads an integer token
/// as an integer literal and a float token as a float literal, never the one
/// as the other, and a string in the literal grammar as a decimal.
///
/// An integer in the `i64` range serializes through `serialize_i64`, and
/// one above it up to `u64::MAX` through `serialize_u64`. Any other integer
/// serializes as a `serde_json` arbitrary-precision number, which only
/// `serde_json` writes as an integer; another format receives it as a
/// struct under `serde_json`'s private number token, holding the decimal
/// digits as a string. A NaN or infinite float literal has no JSON form and
/// fails to serialize.
///
/// Deserializing checks the whole payload before it restores any identifier:
/// a payload refused for its structure, an unknown type id, an unknown or
/// missing field, an unknown operation name, a literal outside the literal
/// grammar, piecewise condition and value lists of different lengths, an
/// empty piecewise, a literal case condition other than a Boolean, a
/// logical node of fewer than two operands, or an empty function name
/// leaves the identifier id counter untouched. Nested
/// levels are read before they are checked, which needs a self-describing
/// format such as JSON.
///
/// Serializing and deserializing recurse once per tree level; see
/// [`Expression`] for the stack a deep tree needs and for the nesting limit
/// `serde_json` puts on decoding JSON text, which refuses a chain of more
/// than 62 unary or binary nodes over a leaf.
impl Serialize for Expression {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut wrapper = serializer.serialize_struct("Expression", 2)?;
        wrapper.serialize_field(TYPE_KEY, find_type_id(self))?;
        wrapper.serialize_field(DATA_KEY, &NodeFields(self))?;
        wrapper.end()
    }
}

/// One checked node of an expression payload, with its identifiers still
/// unrestored.
enum PayloadNode {
    Unary {
        operation: UnaryOperation,
        operand: Box<PayloadNode>,
    },
    Binary {
        operation: BinaryOperation,
        left: Box<PayloadNode>,
        right: Box<PayloadNode>,
    },
    Logical {
        operation: LogicalOperation,
        operands: Vec<PayloadNode>,
    },
    Identifier(IdentifierWire),
    Literal(LiteralValue),
    Piecewise {
        cases: Vec<(PayloadNode, PayloadNode)>,
        otherwise: Box<PayloadNode>,
    },
    Call {
        callee: Callee,
        arguments: Vec<PayloadNode>,
    },
}

/// A whole expression payload, checked but not yet built.
///
/// Decoding one restores no identifier.
struct ExpressionPayload(PayloadNode);

/// Return a short name for the kind of JSON value `value` is.
fn describe_value(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a Boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Object(_) => "a map",
    }
}

/// Prefix `message` with the field it came from.
fn add_field_context(field: &str, message: impl fmt::Display) -> String {
    format!("in `{field}`: {message}")
}

/// Return the values of exactly the fields `names` of the map `value`, in
/// order, refusing a missing or an unknown field.
fn read_fields<'a, const N: usize>(
    value: &'a Value,
    owner: &str,
    names: [&str; N],
) -> Result<[&'a Value; N], String> {
    let Value::Object(map) = value else {
        return Err(format!(
            "expected the fields of {owner} as a map, got {}",
            describe_value(value)
        ));
    };
    if let Some(unknown) = map.keys().find(|key| !names.contains(&key.as_str())) {
        return Err(format!("unknown field `{unknown}` in {owner}"));
    }
    let mut fields = [&Value::Null; N];
    for (field, name) in fields.iter_mut().zip(names) {
        *field = map
            .get(name)
            .ok_or_else(|| format!("missing field `{name}` in {owner}"))?;
    }
    Ok(fields)
}

/// Check the list in `field` of `value` and return its items.
fn read_list<'a>(value: &'a Value, field: &str) -> Result<&'a [Value], String> {
    match value {
        Value::Array(items) => Ok(items),
        other => Err(add_field_context(
            field,
            format_args!("expected a list, got {}", describe_value(other)),
        )),
    }
}

/// Check a literal's `value` field.
fn parse_literal_value(value: &Value) -> Result<LiteralValue, String> {
    match value {
        Value::Bool(flag) => Ok(LiteralValue::from(*flag)),
        Value::Number(number) => {
            let token = number.to_string();
            if token.contains(['.', 'e', 'E']) {
                number
                    .as_f64()
                    .map(LiteralValue::from)
                    .ok_or_else(|| format!("the float {token} does not fit an f64"))
            } else {
                BigInt::parse_bytes(token.as_bytes(), 10)
                    .map(LiteralValue::from)
                    .ok_or_else(|| format!("the integer {token} is malformed"))
            }
        }
        Value::String(text) => text
            .parse::<Decimal>()
            .map(LiteralValue::Decimal)
            .map_err(|error| error.to_string()),
        other => Err(format!(
            "expected a Boolean, a number, or a numeric text as a literal value, got {}",
            describe_value(other)
        )),
    }
}

/// Check a piecewise's fields.
fn parse_piecewise(data: &Value) -> Result<PayloadNode, String> {
    let [conditions, values, otherwise] =
        read_fields(data, "a piecewise", ["conditions", "values", "otherwise"])?;
    let conditions = read_list(conditions, "conditions")?;
    let values = read_list(values, "values")?;
    if conditions.len() != values.len() {
        return Err(format!(
            "a piecewise has {} conditions but {} values",
            conditions.len(),
            values.len()
        ));
    }
    validate_case_count(conditions.len()).map_err(|error| error.to_string())?;
    let mut cases = Vec::with_capacity(conditions.len());
    for (case_index, (condition, value)) in conditions.iter().zip(values).enumerate() {
        let condition =
            parse_node(condition).map_err(|error| add_field_context("conditions", error))?;
        if let PayloadNode::Literal(literal) = &condition {
            validate_condition_literal(case_index, literal).map_err(|error| error.to_string())?;
        }
        let value = parse_node(value).map_err(|error| add_field_context("values", error))?;
        cases.push((condition, value));
    }
    let otherwise = parse_node(otherwise).map_err(|error| add_field_context("otherwise", error))?;
    Ok(PayloadNode::Piecewise {
        cases,
        otherwise: Box::new(otherwise),
    })
}

/// Check a logical node's fields.
fn parse_logical(data: &Value) -> Result<PayloadNode, String> {
    let [operation, operands] = read_fields(data, "a logical node", ["operation", "operands"])?;
    let operation = LogicalOperation::deserialize(operation)
        .map_err(|error| add_field_context("operation", error))?;
    let operands = read_list(operands, "operands")?;
    if operands.len() < 2 {
        return Err(format!(
            "a logical node needs at least 2 operands, got {}",
            operands.len()
        ));
    }
    let operands = operands
        .iter()
        .map(|operand| parse_node(operand).map_err(|error| add_field_context("operands", error)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PayloadNode::Logical {
        operation,
        operands,
    })
}

/// Check a call's fields.
fn parse_call(data: &Value) -> Result<PayloadNode, String> {
    let [function_name, arguments] = read_fields(data, "a call", ["function_name", "arguments"])?;
    let Value::String(function_name) = function_name else {
        return Err(add_field_context(
            "function_name",
            format_args!("expected a string, got {}", describe_value(function_name)),
        ));
    };
    let callee: Callee = function_name
        .parse()
        .map_err(|error: FunctionNameError| error.to_string())?;
    let arguments = read_list(arguments, "arguments")?
        .iter()
        .map(|argument| parse_node(argument).map_err(|error| add_field_context("arguments", error)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PayloadNode::Call { callee, arguments })
}

/// Check the node the wire map `value` describes, and its whole subtree.
fn parse_node(value: &Value) -> Result<PayloadNode, String> {
    let [type_id, data] = read_fields(value, "an expression", [TYPE_KEY, DATA_KEY])?;
    let Value::String(type_id) = type_id else {
        return Err(format!(
            "expected the expression type id as a string, got {}",
            describe_value(type_id)
        ));
    };
    match type_id.as_str() {
        UNARY_TYPE_ID => {
            let [operation, operand] = read_fields(data, "a unary node", ["operation", "operand"])?;
            Ok(PayloadNode::Unary {
                operation: UnaryOperation::deserialize(operation)
                    .map_err(|error| add_field_context("operation", error))?,
                operand: Box::new(
                    parse_node(operand).map_err(|error| add_field_context("operand", error))?,
                ),
            })
        }
        BINARY_TYPE_ID => {
            let [operation, left, right] =
                read_fields(data, "a binary node", ["operation", "left", "right"])?;
            Ok(PayloadNode::Binary {
                operation: BinaryOperation::deserialize(operation)
                    .map_err(|error| add_field_context("operation", error))?,
                left: Box::new(parse_node(left).map_err(|error| add_field_context("left", error))?),
                right: Box::new(
                    parse_node(right).map_err(|error| add_field_context("right", error))?,
                ),
            })
        }
        IDENTIFIER_TYPE_ID => {
            let [identifier] = read_fields(data, "an identifier reference", ["identifier"])?;
            IdentifierWire::deserialize(identifier)
                .map(PayloadNode::Identifier)
                .map_err(|error| add_field_context("identifier", error))
        }
        LITERAL_TYPE_ID => {
            let [literal] = read_fields(data, "a literal", ["value"])?;
            parse_literal_value(literal)
                .map(PayloadNode::Literal)
                .map_err(|error| add_field_context("value", error))
        }
        LOGICAL_TYPE_ID => parse_logical(data),
        PIECEWISE_TYPE_ID => parse_piecewise(data),
        CALL_TYPE_ID => parse_call(data),
        unknown => Err(format!("unknown expression type id `{unknown}`")),
    }
}

/// Build the expression a checked node describes, restoring its
/// identifiers.
fn build_node<E: de::Error>(node: PayloadNode) -> Result<Expression, E> {
    Ok(match node {
        PayloadNode::Unary { operation, operand } => Expression::from_kind(ExpressionKind::Unary(
            UnaryExpression::new(operation, build_node(*operand)?),
        )),
        PayloadNode::Binary {
            operation,
            left,
            right,
        } => Expression::from_kind(ExpressionKind::Binary(BinaryExpression::new(
            operation,
            build_node(*left)?,
            build_node(*right)?,
        ))),
        PayloadNode::Logical {
            operation,
            operands,
        } => {
            let operands = operands
                .into_iter()
                .map(build_node)
                .collect::<Result<Box<[_]>, E>>()?;
            Expression::from_kind(ExpressionKind::Logical(LogicalExpression::new(
                operation, operands,
            )))
        }
        PayloadNode::Identifier(identifier) => {
            Expression::from(Identifier::try_from(identifier).map_err(E::custom)?)
        }
        PayloadNode::Literal(value) => Expression::from(value),
        PayloadNode::Piecewise { cases, otherwise } => {
            let cases = cases
                .into_iter()
                .map(|(condition, value)| Ok((build_node(condition)?, build_node(value)?)))
                .collect::<Result<Vec<_>, E>>()?;
            Expression::from_kind(ExpressionKind::Piecewise(
                PiecewiseExpression::try_new(cases, build_node(*otherwise)?).map_err(E::custom)?,
            ))
        }
        PayloadNode::Call { callee, arguments } => {
            let arguments = arguments
                .into_iter()
                .map(build_node)
                .collect::<Result<Vec<_>, E>>()?;
            Expression::from_kind(ExpressionKind::Call(CallExpression::new(callee, arguments)))
        }
    })
}

impl<'de> Deserialize<'de> for ExpressionPayload {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        parse_node(&value).map(Self).map_err(de::Error::custom)
    }
}

/// Deserialize the wire shape written by the [`Serialize`] implementation,
/// in its map form only, checking the whole payload before restoring any
/// identifier.
impl<'de> Deserialize<'de> for Expression {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let payload: ExpressionPayload = decode::deserialize_map_only(deserializer)?;
        build_node(payload.0)
    }
}
