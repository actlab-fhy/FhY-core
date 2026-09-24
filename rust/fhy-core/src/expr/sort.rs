//! The value sorts of built-in function parameters and results.

use serde::{Deserialize, Serialize};

use super::operation::impl_name_text;

/// The coarse mathematical sort of a function parameter or result.
///
/// The numeric sorts form the containment chain `Nat < Int < Real`; `Bool`
/// is a separate branch, compatible with no numeric sort.
///
/// Serializes as its [`as_str`](Self::as_str) text, and deserializes and
/// parses with [`FromStr`](std::str::FromStr) only from exactly that text.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::FunctionSort;
///
/// assert_eq!(FunctionSort::Nat.as_str(), "nat");
/// assert_eq!(FunctionSort::Real.to_string(), "real");
/// ```
#[expect(
    clippy::exhaustive_enums,
    reason = "a closed classification that passes map one to one"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FunctionSort {
    /// A Boolean value.
    Bool,
    /// A non-negative integer.
    Nat,
    /// An integer of either sign.
    Int,
    /// A real number.
    Real,
}

impl FunctionSort {
    /// Return the lowercase text of the sort: `"bool"`, `"nat"`, `"int"`, or
    /// `"real"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Nat => "nat",
            Self::Int => "int",
            Self::Real => "real",
        }
    }
}

impl_name_text!(FunctionSort, "function sort");
