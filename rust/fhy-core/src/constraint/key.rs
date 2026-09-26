//! Canonical ordering keys: texts equal for two constraints exactly when
//! they are structurally equivalent.
//!
//! An equation's key renders its tree in pre-order, each node as its kind,
//! its own data and its parenthesized children: a literal's canonical
//! value, an identifier's id, an operation, or a callee. A set
//! constraint's key is its polarity, its variable's id and its members'
//! keys in canonical order. Every text a key embeds is quoted, so no two
//! different constraints render alike.

use std::fmt;

use crate::expression::{Expression, ExpressionKind, LiteralValue};

use super::set::{Polarity, SetConstraint};
use super::value::{Member, MemberKind};

/// Return the key of an equation over `expression`.
pub(super) fn equation_key(expression: &Expression) -> String {
    EquationKey(expression).to_string()
}

/// Return the key of `constraint`.
pub(super) fn set_key(constraint: &SetConstraint) -> String {
    SetKey(constraint).to_string()
}

/// Displays as an equation's key.
struct EquationKey<'a>(&'a Expression);

impl fmt::Display for EquationKey<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("equation|")?;
        write_expression_key(self.0, f)
    }
}

/// Displays as a set constraint's key.
struct SetKey<'a>(&'a SetConstraint);

impl fmt::Display for SetKey<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.0.polarity() {
            Polarity::In => "in_set",
            Polarity::NotIn => "not_in_set",
        };
        write!(f, "{kind}|{}|{{", self.0.variable().id())?;
        write_members(self.0.members().iter(), f)?;
        f.write_str("}")
    }
}

/// One step of the pre-order rendering of a tree.
enum Step<'a> {
    Node(&'a Expression),
    Text(&'static str),
}

/// Write the key of the tree `expression`, on a work list, so a deep tree
/// renders on a small stack.
fn write_expression_key(expression: &Expression, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let mut pending = vec![Step::Node(expression)];
    while let Some(step) = pending.pop() {
        let node = match step {
            Step::Text(text) => {
                f.write_str(text)?;
                continue;
            }
            Step::Node(node) => node,
        };
        match node.kind() {
            ExpressionKind::Unary(unary) => write!(f, "unary[{}]", unary.operation().as_str())?,
            ExpressionKind::Binary(binary) => {
                write!(f, "binary[{}]", binary.operation().as_str())?;
            }
            ExpressionKind::Logical(logical) => {
                write!(f, "logical[{}]", logical.operation().as_str())?;
            }
            ExpressionKind::Identifier(identifier) => write!(f, "identifier[{}]", identifier.id())?,
            ExpressionKind::Literal(value) => {
                f.write_str("literal[")?;
                write_literal_key(value, f)?;
                f.write_str("]")?;
            }
            ExpressionKind::Piecewise(_) => f.write_str("piecewise[]")?,
            ExpressionKind::Call(call) => write!(f, "call[{:?}]", call.callee().name())?,
        }
        f.write_str("(")?;
        pending.push(Step::Text(")"));
        let children: Vec<&Expression> = node.children().collect();
        for (index, child) in children.iter().enumerate().rev() {
            pending.push(Step::Node(child));
            if index > 0 {
                pending.push(Step::Text(","));
            }
        }
    }
    Ok(())
}

/// Write the key of the literal `value`: equal for equal literals, whose
/// NaNs are all equal and whose zeros are one.
fn write_literal_key(value: &LiteralValue, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match value {
        LiteralValue::Bool(value) => write!(f, "bool:{value}"),
        LiteralValue::Int(value) => write!(f, "int:{value}"),
        LiteralValue::Float(value) if value.is_nan() => f.write_str("float:nan"),
        LiteralValue::Float(value) => write!(f, "float:{}", value + 0.0),
        LiteralValue::Decimal(value) => write!(f, "decimal:{value}"),
    }
}

/// Write the key of `member`.
///
/// The recursion follows the nesting of containers, which is shallow.
fn write_member_key(member: &Member, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match member.kind() {
        MemberKind::Bool(value) => write!(f, "bool:{value}"),
        MemberKind::Int(value) => write!(f, "int:{value}"),
        MemberKind::Float(value) => write!(f, "float:{value}"),
        MemberKind::Str(value) => write!(f, "str:{value:?}"),
        MemberKind::Tuple(members) => {
            f.write_str("tuple(")?;
            write_members(members.iter(), f)?;
            f.write_str(")")
        }
        MemberKind::FrozenSet(members) => {
            f.write_str("frozenset(")?;
            write_members(members.iter(), f)?;
            f.write_str(")")
        }
        MemberKind::Opaque(_) => write!(f, "opaque:{:?}", member.opaque_key().unwrap_or_default()),
    }
}

/// Write the keys of `members`, separated by commas.
fn write_members<'a>(
    members: impl Iterator<Item = &'a Member>,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    for (index, member) in members.enumerate() {
        if index > 0 {
            f.write_str(",")?;
        }
        write_member_key(member, f)?;
    }
    Ok(())
}
