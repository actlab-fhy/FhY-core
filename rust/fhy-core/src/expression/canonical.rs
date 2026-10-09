//! The canonical node table of an expression: each distinct node once,
//! children by table index.
//!
//! The wire format serializes the table under [`Equivalence::Wire`], and the
//! constraint ordering keys render it under [`Equivalence::Structural`].

use std::collections::HashMap;
use std::hash::BuildHasher;

use crate::identifier::Identifier;
use crate::tree::{BuildIdentityHasher, NodeHandle, NodeIdentity, Tree};

use super::BigInt;
use super::callee::Callee;
use super::literal::{Decimal, LiteralValue};
use super::node::{Expression, ExpressionKind};
use super::operation::{BinaryOperation, LogicalOperation, UnaryOperation};

/// When two nodes of a table are one entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Equivalence {
    /// Equal wire texts: floats by their bits, every NaN one (the wire
    /// writes each as `NaN`), and `-0.0` apart from `0.0`; identifiers by id
    /// and name hint, both of which the wire writes.
    Wire,
    /// Equal under `==`: every NaN one, the two zeros one, and identifiers
    /// by id.
    Structural,
}

/// A node of the table: the expression node it stands for, the first one
/// met, and where its children's table indices sit in the table's child
/// list.
struct CanonicalNode<'a> {
    node: &'a Expression,
    children: std::ops::Range<usize>,
}

/// The distinct nodes of an expression, in post-order of first visit, the
/// root last.
pub(crate) struct CanonicalTable<'a> {
    nodes: Vec<CanonicalNode<'a>>,
    /// The children of every node by table index, in
    /// [`Expression::children`] order, one node's after another's.
    children: Vec<usize>,
}

/// The bits a float is compared by: one value for every NaN, and under
/// [`Equivalence::Structural`] one for both zeros.
fn float_bits(value: f64, equivalence: Equivalence) -> u64 {
    if value.is_nan() {
        return f64::NAN.to_bits();
    }
    match equivalence {
        Equivalence::Wire => value.to_bits(),
        Equivalence::Structural => (value + 0.0).to_bits(),
    }
}

/// A literal's data, as the table compares it.
#[derive(PartialEq, Eq, Hash)]
enum LiteralKey<'a> {
    Bool(bool),
    Int(&'a BigInt),
    Float(u64),
    Decimal(&'a Decimal),
}

/// A node's own data, without its children, as the table compares it.
#[derive(PartialEq, Eq, Hash)]
enum DataKey<'a> {
    Unary(UnaryOperation),
    Binary(BinaryOperation),
    Logical(LogicalOperation),
    Identifier(u64, Option<&'a str>),
    Literal(LiteralKey<'a>),
    Piecewise,
    Call(&'a Callee),
}

/// Return the data of `node` under `equivalence`.
fn data_key(node: &Expression, equivalence: Equivalence) -> DataKey<'_> {
    match node.kind() {
        ExpressionKind::Unary(unary) => DataKey::Unary(unary.operation()),
        ExpressionKind::Binary(binary) => DataKey::Binary(binary.operation()),
        ExpressionKind::Logical(logical) => DataKey::Logical(logical.operation()),
        ExpressionKind::Identifier(identifier) => identifier_key(identifier, equivalence),
        ExpressionKind::Literal(literal) => DataKey::Literal(match literal {
            LiteralValue::Bool(value) => LiteralKey::Bool(*value),
            LiteralValue::Int(value) => LiteralKey::Int(value),
            LiteralValue::Float(value) => LiteralKey::Float(float_bits(*value, equivalence)),
            LiteralValue::Decimal(value) => LiteralKey::Decimal(value),
        }),
        ExpressionKind::Piecewise(_) => DataKey::Piecewise,
        ExpressionKind::Call(call) => DataKey::Call(call.callee()),
    }
}

fn identifier_key(identifier: &Identifier, equivalence: Equivalence) -> DataKey<'_> {
    let name_hint = match equivalence {
        Equivalence::Wire => Some(identifier.name_hint()),
        Equivalence::Structural => None,
    };
    DataKey::Identifier(identifier.id(), name_hint)
}

/// One step of building a table: visit a node, or enter it once its
/// children are entered.
enum Step<'a> {
    Visit(&'a Expression),
    Enter(&'a Expression, usize),
}

impl<'a> CanonicalTable<'a> {
    /// Return the table of `root` under `equivalence`.
    ///
    /// A node is added only when no equivalent entry exists. Children are
    /// entered before their parent, so two nodes are equivalent exactly when
    /// their own data are equivalent and their child index lists are equal,
    /// and a lookup is a hash of both and one local comparison. A handle
    /// that may be shared is looked up by identity first, so building is
    /// linear in the distinct handles of `root`, and it does not recurse.
    pub(crate) fn build(root: &'a Expression, equivalence: Equivalence) -> Self {
        // Room for a small expression, so it builds without regrowing.
        const EXPECTED_NODES: usize = 16;
        let mut nodes: Vec<CanonicalNode<'a>> = Vec::with_capacity(EXPECTED_NODES);
        let mut child_list: Vec<usize> = Vec::with_capacity(2 * EXPECTED_NODES);
        // The first entry of each hash of `(data, children)`, and, per entry,
        // the next entry of the same hash: a chain, which collisions alone
        // lengthen, so a lookup allocates nothing.
        let mut first_of_hash: HashMap<u64, usize, BuildIdentityHasher> =
            HashMap::with_capacity_and_hasher(EXPECTED_NODES, BuildIdentityHasher::default());
        let mut next_of_hash: Vec<Option<usize>> = Vec::with_capacity(EXPECTED_NODES);
        let mut entered: HashMap<NodeIdentity, usize, BuildIdentityHasher> = HashMap::default();
        let mut indices: Vec<usize> = Vec::with_capacity(EXPECTED_NODES);
        let mut pending = Vec::with_capacity(EXPECTED_NODES);
        pending.push(Step::Visit(root));
        while let Some(step) = pending.pop() {
            match step {
                Step::Visit(node) => {
                    if let Some(&index) = node
                        .is_shared()
                        .then(|| entered.get(&node.identity()))
                        .flatten()
                    {
                        indices.push(index);
                        continue;
                    }
                    let children = node.children();
                    pending.push(Step::Enter(node, children.len()));
                    pending.extend(children.rev().map(Step::Visit));
                }
                Step::Enter(node, child_count) => {
                    let children = &indices[indices.len() - child_count..];
                    let data = data_key(node, equivalence);
                    let hash = BuildIdentityHasher::default().hash_one((&data, children));
                    let mut candidate = first_of_hash.get(&hash).copied();
                    while let Some(index) = candidate {
                        let entry = &nodes[index];
                        if child_list[entry.children.clone()] == *children
                            && data_key(entry.node, equivalence) == data
                        {
                            break;
                        }
                        candidate = next_of_hash[index];
                    }
                    let index = candidate.unwrap_or_else(|| {
                        let index = nodes.len();
                        next_of_hash.push(first_of_hash.insert(hash, index));
                        let start = child_list.len();
                        child_list.extend_from_slice(children);
                        nodes.push(CanonicalNode {
                            node,
                            children: start..child_list.len(),
                        });
                        index
                    });
                    indices.truncate(indices.len() - child_count);
                    if node.is_shared() {
                        entered.insert(node.identity(), index);
                    }
                    indices.push(index);
                }
            }
        }
        Self {
            nodes,
            children: child_list,
        }
    }

    /// Return the distinct nodes, the root last, each the expression node
    /// it stands for, the first one met, and its children by table index,
    /// in [`Expression::children`] order.
    pub(crate) fn nodes(&self) -> impl ExactSizeIterator<Item = (&'a Expression, &[usize])> {
        self.nodes
            .iter()
            .map(|entry| (entry.node, &self.children[entry.children.clone()]))
    }
}
