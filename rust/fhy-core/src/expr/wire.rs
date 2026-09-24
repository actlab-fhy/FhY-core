//! Serialization of expressions as a flat table of their distinct nodes.

use std::collections::HashMap;

use serde::de::{self, Deserializer};
use serde::ser::{SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};

use crate::identifier::Identifier;
use crate::tree::{BuildIdentityHasher, NodeHandle, NodeIdentity, Tree};

use super::callee::Callee;
use super::literal::LiteralValue;
use super::node::{
    BinaryExpression, CallExpression, Expression, ExpressionKind, LogicalExpression,
    PiecewiseExpression, UnaryExpression, validate_condition_literal,
};
use super::operation::{BinaryOperation, LogicalOperation, UnaryOperation};

/// A node of the table as it is read: its data, and its children by index.
#[derive(Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum WireNode {
    Unary {
        operation: UnaryOperation,
        operand: u64,
    },
    Binary {
        operation: BinaryOperation,
        left: u64,
        right: u64,
    },
    Logical {
        operation: LogicalOperation,
        operands: Vec<u64>,
    },
    Identifier(Identifier),
    Literal(LiteralValue),
    Piecewise {
        cases: Vec<(u64, u64)>,
        otherwise: u64,
    },
    Call {
        callee: Callee,
        arguments: Vec<u64>,
    },
}

/// A node of the table as it is written, borrowing its data from the
/// expression: the twin of [`WireNode`], variant for variant.
#[derive(Serialize)]
#[serde(rename = "WireNode", rename_all = "snake_case")]
enum WireNodeRef<'a> {
    Unary {
        operation: UnaryOperation,
        operand: u64,
    },
    Binary {
        operation: BinaryOperation,
        left: u64,
        right: u64,
    },
    Logical {
        operation: LogicalOperation,
        operands: Vec<u64>,
    },
    Identifier(&'a Identifier),
    Literal(&'a LiteralValue),
    Piecewise {
        cases: Vec<(u64, u64)>,
        otherwise: u64,
    },
    Call {
        callee: &'a Callee,
        arguments: Vec<u64>,
    },
}

#[derive(Deserialize)]
#[serde(rename = "Expression", deny_unknown_fields)]
struct ExpressionWire {
    nodes: Vec<WireNode>,
}

fn to_wire_index(position: usize) -> u64 {
    u64::try_from(position).unwrap_or(u64::MAX)
}

/// Build the table node of `node` whose children are at `children`, in
/// [`Expression::children`] order.
fn build_wire_node(node: &Expression, children: Vec<u64>) -> WireNodeRef<'_> {
    let child = |position: usize| children.get(position).copied().unwrap_or(u64::MAX);
    match node.kind() {
        ExpressionKind::Unary(unary) => WireNodeRef::Unary {
            operation: unary.operation(),
            operand: child(0),
        },
        ExpressionKind::Binary(binary) => WireNodeRef::Binary {
            operation: binary.operation(),
            left: child(0),
            right: child(1),
        },
        ExpressionKind::Logical(logical) => WireNodeRef::Logical {
            operation: logical.operation(),
            operands: children,
        },
        ExpressionKind::Identifier(identifier) => WireNodeRef::Identifier(identifier),
        ExpressionKind::Literal(literal) => WireNodeRef::Literal(literal),
        ExpressionKind::Piecewise(_) => {
            let otherwise = children.last().copied().unwrap_or(u64::MAX);
            let cases = children
                .chunks_exact(2)
                .map(|case| (case[0], case[1]))
                .collect();
            WireNodeRef::Piecewise { cases, otherwise }
        }
        ExpressionKind::Call(call) => WireNodeRef::Call {
            callee: call.callee(),
            arguments: children,
        },
    }
}

/// One step of encoding: visit a node, or write it once its children are
/// written.
enum EncodeStep<'a> {
    Visit(&'a Expression),
    Write(&'a Expression, usize),
}

/// Return the table of `root`: each distinct node once, in post-order of
/// first visit, the root last.
///
/// Only a node that may be shared is looked up by identity, since a node
/// with one handle is reached once.
fn encode_nodes(root: &Expression) -> Vec<WireNodeRef<'_>> {
    let mut nodes: Vec<WireNodeRef<'_>> = Vec::new();
    let mut written: HashMap<NodeIdentity, u64, BuildIdentityHasher> = HashMap::default();
    let mut indices: Vec<u64> = Vec::new();
    let mut pending = vec![EncodeStep::Visit(root)];
    while let Some(step) = pending.pop() {
        match step {
            EncodeStep::Visit(node) => {
                if let Some(&index) = node
                    .is_shared()
                    .then(|| written.get(&node.identity()))
                    .flatten()
                {
                    indices.push(index);
                    continue;
                }
                let children: Vec<&Expression> = node.children().collect();
                pending.push(EncodeStep::Write(node, children.len()));
                pending.extend(children.into_iter().rev().map(EncodeStep::Visit));
            }
            EncodeStep::Write(node, child_count) => {
                let children = indices.split_off(indices.len() - child_count);
                let index = to_wire_index(nodes.len());
                nodes.push(build_wire_node(node, children));
                if node.is_shared() {
                    written.insert(node.identity(), index);
                }
                indices.push(index);
            }
        }
    }
    nodes
}

/// An expression serializes as a table of its distinct nodes,
/// `{"nodes": [..]}`, in post-order of first visit with the root last.
/// Each node refers to its children by their indices in the table, which
/// always precede its own:
///
/// | Node | Wire form |
/// |---|---|
/// | unary | `{"unary": {"operation": "negate", "operand": i}}` |
/// | binary | `{"binary": {"operation": "add", "left": i, "right": j}}` |
/// | logical | `{"logical": {"operation": "and", "operands": [i, j, ..]}}` |
/// | identifier | `{"identifier": {"id": 41, "name_hint": "x"}}` |
/// | literal | `{"literal": {"int": "1"}}`, see [`LiteralValue`] |
/// | piecewise | `{"piecewise": {"cases": [[c0, v0], ..], "otherwise": i}}` |
/// | call | `{"call": {"callee": {"builtin": "max"}, "arguments": [i, ..]}}`, or `{"named": "f"}` as the callee |
///
/// So `x + 1`, with `x` of id 41, serializes as
/// `{"nodes":[{"identifier":{"id":41,"name_hint":"x"}},{"literal":{"int":"1"}},{"binary":{"operation":"add","left":0,"right":1}}]}`.
///
/// A node shared by several parents is written once, and decoding shares
/// it again, so a DAG such as `x(k+1) = xk + xk` serializes in space and
/// time linear in its distinct nodes. Neither direction recurses once per
/// tree level, and the serde nesting depth is the same for every tree, so
/// a tree of any depth round-trips through any format on a small stack.
/// Serialization never fails for a well-formed serializer: every literal,
/// NaN and the infinities included, has a wire form.
impl Serialize for Expression {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let nodes = encode_nodes(self);
        let mut table = serializer.serialize_struct("Expression", 1)?;
        table.serialize_field("nodes", &nodes)?;
        table.end()
    }
}

/// The decoded nodes of a table, by index, and whether each one is
/// referenced by a later node.
struct DecodedNodes {
    nodes: Vec<Expression>,
    is_referenced: Vec<bool>,
}

impl DecodedNodes {
    /// Return the decoded node at `child`, referred to by node `index`,
    /// marking it referenced, or the message for a child that does not
    /// precede node `index`.
    fn take_child(&mut self, index: usize, child: u64) -> Result<Expression, String> {
        let position = usize::try_from(child)
            .ok()
            .filter(|&position| position < index);
        let Some(position) = position else {
            return Err(format!(
                "node {index} refers to node {child}, which does not precede it"
            ));
        };
        self.is_referenced[position] = true;
        Ok(self.nodes[position].clone())
    }

    fn take_children(&mut self, index: usize, children: &[u64]) -> Result<Vec<Expression>, String> {
        children
            .iter()
            .map(|&child| self.take_child(index, child))
            .collect()
    }
}

/// Build the expression node `index` of the table describes, or the message
/// refusing it.
fn decode_node(
    decoded: &mut DecodedNodes,
    index: usize,
    node: WireNode,
) -> Result<Expression, String> {
    let kind = match node {
        WireNode::Unary { operation, operand } => ExpressionKind::Unary(UnaryExpression::new(
            operation,
            decoded.take_child(index, operand)?,
        )),
        WireNode::Binary {
            operation,
            left,
            right,
        } => ExpressionKind::Binary(BinaryExpression::new(
            operation,
            decoded.take_child(index, left)?,
            decoded.take_child(index, right)?,
        )),
        WireNode::Logical {
            operation,
            operands,
        } => {
            if operands.len() < 2 {
                return Err(format!(
                    "logical node {index} has {} operands, expected at least 2",
                    operands.len()
                ));
            }
            let operands = decoded.take_children(index, &operands)?;
            ExpressionKind::Logical(LogicalExpression::new(
                operation,
                operands.into_boxed_slice(),
            ))
        }
        WireNode::Identifier(identifier) => ExpressionKind::Identifier(identifier),
        WireNode::Literal(literal) => ExpressionKind::Literal(literal),
        WireNode::Piecewise { cases, otherwise } => {
            if cases.is_empty() {
                return Err(format!("piecewise node {index} has no cases"));
            }
            let mut decoded_cases = Vec::with_capacity(cases.len());
            for (case_index, (condition, value)) in cases.into_iter().enumerate() {
                let condition = decoded.take_child(index, condition)?;
                if let ExpressionKind::Literal(literal) = condition.kind() {
                    validate_condition_literal(case_index, literal).map_err(|_refused| {
                        format!(
                            "condition of case {case_index} of piecewise node {index} is a \
                             non-boolean literal"
                        )
                    })?;
                }
                decoded_cases.push((condition, decoded.take_child(index, value)?));
            }
            let otherwise = decoded.take_child(index, otherwise)?;
            let piecewise = PiecewiseExpression::try_new(decoded_cases, otherwise)
                .map_err(|error| format!("piecewise node {index}: {error}"))?;
            ExpressionKind::Piecewise(piecewise)
        }
        WireNode::Call { callee, arguments } => {
            let arguments = decoded.take_children(index, &arguments)?;
            ExpressionKind::Call(CallExpression::new(callee, arguments))
        }
    };
    Ok(Expression::from_kind(kind))
}

/// Build the expression a table describes, its last node, or the message
/// refusing the table.
fn decode_table(wire: ExpressionWire) -> Result<Expression, String> {
    let count = wire.nodes.len();
    if count == 0 {
        return Err("expression payload has no nodes".to_owned());
    }
    let mut decoded = DecodedNodes {
        nodes: Vec::with_capacity(count),
        is_referenced: Vec::with_capacity(count),
    };
    for (index, node) in wire.nodes.into_iter().enumerate() {
        let expression = decode_node(&mut decoded, index, node)?;
        decoded.nodes.push(expression);
        decoded.is_referenced.push(false);
    }
    if let Some(unreferenced) = decoded.is_referenced[..count - 1]
        .iter()
        .position(|&is_referenced| !is_referenced)
    {
        return Err(format!("node {unreferenced} is not referenced"));
    }
    decoded
        .nodes
        .pop()
        .ok_or_else(|| "expression payload has no nodes".to_owned())
}

/// Deserialize the node table the [`Serialize`] implementation writes,
/// sharing every node that more than one later node refers to.
///
/// The table is refused, with a one-line message, when it is empty
/// (`expression payload has no nodes`), when a node refers to a node that
/// does not precede it (`node {i} refers to node {j}, which does not
/// precede it`), when a logical node has fewer than two operands (`logical
/// node {i} has {n} operands, expected at least 2`), when a piecewise node
/// has no cases (`piecewise node {i} has no cases`), when a piecewise case
/// condition is a literal other than a Boolean (`condition of case {c} of
/// piecewise node {i} is a non-boolean literal`), or when a node other than
/// the root is referred to by no later node (`node {i} is not referenced`).
/// A literal or a callee is refused as [`LiteralValue`] and
/// [`FunctionName`](super::FunctionName) describe; an unknown node kind or
/// field with serde's own error.
///
/// Each identifier advances the identifier id counter past its id as it is
/// read, so a table refused later may already have advanced it.
impl<'de> Deserialize<'de> for Expression {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = ExpressionWire::deserialize(deserializer)?;
        decode_table(wire).map_err(de::Error::custom)
    }
}
