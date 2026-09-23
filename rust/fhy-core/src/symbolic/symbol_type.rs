//! The value kind a symbolic identifier ranges over.
//!
//! [`SymbolType`] declares whether an identifier in a symbolic expression
//! stands for a real number, an integer, or a Boolean. Its text form is the
//! lowercase value name (`"real"`, `"int"`, `"bool"`), which is also its
//! serialized form.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

/// Every symbol type, in declaration order.
const ALL_SYMBOL_TYPES: [SymbolType; 3] = [SymbolType::Real, SymbolType::Int, SymbolType::Bool];

/// The value kind a symbolic identifier ranges over.
///
/// Serializes as its [`as_str`](Self::as_str) text, and deserializes only
/// from exactly that text: any other string, including a differently cased
/// one, is rejected.
///
/// # Examples
///
/// ```
/// use fhy_core::symbolic::symbol_type::SymbolType;
///
/// assert_eq!(SymbolType::Int.as_str(), "int");
/// assert_eq!(SymbolType::Real.to_string(), "real");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

impl fmt::Display for SymbolType {
    /// Write the [`as_str`](Self::as_str) text.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for SymbolType {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for SymbolType {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        ALL_SYMBOL_TYPES
            .into_iter()
            .find(|candidate| candidate.as_str() == text)
            .ok_or_else(|| {
                de::Error::invalid_value(
                    de::Unexpected::Str(&text),
                    &"a symbol type name: real, int, or bool",
                )
            })
    }
}
