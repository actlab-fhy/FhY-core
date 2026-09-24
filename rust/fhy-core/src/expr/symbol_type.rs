//! The value kind a symbolic identifier ranges over.

use serde::{Deserialize, Serialize};

use super::operation::impl_name_text;

/// The value kind a symbolic identifier ranges over.
///
/// Serializes as its [`as_str`](Self::as_str) text, and deserializes and
/// parses with [`FromStr`](std::str::FromStr) only from exactly that text.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::SymbolType;
///
/// assert_eq!(SymbolType::Int.as_str(), "int");
/// assert_eq!(SymbolType::Real.to_string(), "real");
/// ```
#[expect(
    clippy::exhaustive_enums,
    reason = "a closed classification that passes map one to one"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolType {
    /// A real number.
    Real,
    /// An integer.
    Int,
    /// A Boolean.
    Bool,
}

impl SymbolType {
    /// Return the lowercase text of the value kind: `"real"`, `"int"`, or
    /// `"bool"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Real => "real",
            Self::Int => "int",
            Self::Bool => "bool",
        }
    }
}

impl_name_text!(SymbolType, "symbol type");
