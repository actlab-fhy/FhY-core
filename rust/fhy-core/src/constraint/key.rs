//! Canonical ordering keys: texts equal for two constraints exactly when
//! they are structurally equivalent, for every conforming opaque value and
//! custom constraint, whose keys must be equal exactly when they are equal.
//!
//! An equation's key is `equation|` and its expression's canonical node
//! table under structural equivalence (S-1 of
//! `docs/design/rust-port-fixes.md`): each distinct node once, in
//! post-order of first visit with the root last, `;`-separated, as
//! `kind[data](i,j,…)` with its children by table index. A literal writes
//! its canonical value (`int:1`, `float:1e300`, every NaN as `float:NaN` and
//! both zeros as `float:0`), an identifier its id, an operation its name,
//! and a callee `builtin:<name>` or `named:"<name>"`, the name quoted. So
//! `x + x` keys as `equation|identifier[7]();binary[add](0,0)` however its
//! leaves are shared, and a key is linear in the distinct nodes of its
//! expression. A set constraint's key is its polarity, its variable's id and
//! its members' keys in canonical order. Every text a key embeds is quoted,
//! so no two different constraints render alike.

use std::fmt::{self, Write as _};

use crate::expression::{
    Callee, CanonicalTable, Equivalence, Expression, ExpressionKind, LiteralValue, write_float,
};

use super::set::{Polarity, SetConstraint};
use super::value::{Member, MemberKind};

/// Return the key of an equation over `expression`.
pub(super) fn equation_key(expression: &Expression) -> String {
    /// Room for the key of a small expression, so it writes without
    /// regrowing.
    const EXPECTED_LENGTH: usize = 128;
    let mut key = String::with_capacity(EXPECTED_LENGTH);
    write!(key, "{}", EquationKey(expression))
        .unwrap_or_else(|_| unreachable!("a string takes any text"));
    key
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

/// Write the key of the expression `expression`: its canonical node table
/// under structural equivalence, each entry as `kind[data](children)`.
fn write_expression_key(expression: &Expression, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let table = CanonicalTable::build(expression, Equivalence::Structural);
    for (index, (node, children)) in table.nodes().enumerate() {
        if index > 0 {
            f.write_str(";")?;
        }
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
            ExpressionKind::Call(call) => {
                f.write_str("call[")?;
                write_callee_key(call.callee(), f)?;
                f.write_str("]")?;
            }
        }
        f.write_str("(")?;
        for (position, child) in children.iter().enumerate() {
            if position > 0 {
                f.write_str(",")?;
            }
            write!(f, "{child}")?;
        }
        f.write_str(")")?;
    }
    Ok(())
}

/// Write the key of `callee`: `builtin:<name>` for a built-in, whose names
/// are fixed words, and `named:"<name>"` for another function, the name
/// quoted.
fn write_callee_key(callee: &Callee, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match callee {
        Callee::Builtin(function) => write!(f, "builtin:{}", function.name()),
        Callee::Named(name) => {
            f.write_str("named:")?;
            write_quoted(name.as_str(), f)
        }
    }
}

/// Write `text` between double quotes, with each `"` and `\` escaped by a
/// backslash, so the quoted text ends at its closing quote and two texts
/// quote alike exactly when they are equal.
fn write_quoted(text: &str, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_char('"')?;
    for character in text.chars() {
        if matches!(character, '"' | '\\') {
            f.write_char('\\')?;
        }
        f.write_char(character)?;
    }
    f.write_char('"')
}

/// Write the key of the literal `value`: equal for equal literals, whose
/// NaNs are all equal and whose zeros are one.
fn write_literal_key(value: &LiteralValue, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match value {
        LiteralValue::Bool(value) => write!(f, "bool:{value}"),
        LiteralValue::Int(value) => write!(f, "int:{value}"),
        LiteralValue::Float(value) => {
            f.write_str("float:")?;
            write_float(value + 0.0, f)
        }
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
        MemberKind::Float(value) => {
            f.write_str("float:")?;
            write_float(value, f)
        }
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
