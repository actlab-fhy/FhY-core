//! The value kind a symbolic identifier ranges over.
//!
//! [`SymbolType`] declares whether an identifier in a symbolic expression
//! stands for a real number, an integer, or a Boolean. Its text form is the
//! lowercase value name (`"real"`, `"int"`, `"bool"`), which is also its
//! serialized form.

use serde::{Deserialize, Serialize};

use super::operation::impl_name_text;

/// The value kind a symbolic identifier ranges over.
///
/// Serializes as its [`as_str`](Self::as_str) text, and deserializes, like
/// [`FromStr`](std::str::FromStr) parses, only from exactly that text: any
/// other string, including a differently cased one, is refused.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::SymbolType;
///
/// assert_eq!(SymbolType::Int.as_str(), "int");
/// assert_eq!(SymbolType::Real.to_string(), "real");
/// ```
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
