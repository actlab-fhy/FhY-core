//! The value sorts of built-in function parameters and results.
//!
//! A [`FunctionSort`] is the coarse mathematical classification of a
//! function's parameter or result. The numeric sorts form the containment
//! chain `Nat < Int < Real`; `Bool` is a separate branch, compatible with no
//! numeric sort. Its text form is the lowercase sort name (`"bool"`,
//! `"nat"`, `"int"`, `"real"`), which is also its serialized form.

use crate::expr::wire_name::impl_wire_name_traits;

/// Every function sort, in declaration order.
const ALL_FUNCTION_SORTS: [FunctionSort; 4] = [
    FunctionSort::Bool,
    FunctionSort::Nat,
    FunctionSort::Int,
    FunctionSort::Real,
];

/// The coarse mathematical sort of a function parameter or result.
///
/// Serializes as its [`as_str`](Self::as_str) text, and deserializes only
/// from exactly that text: any other string, including a differently cased
/// one, is rejected.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::FunctionSort;
///
/// assert_eq!(FunctionSort::Nat.as_str(), "nat");
/// assert_eq!(FunctionSort::Real.to_string(), "real");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

impl_wire_name_traits!(
    FunctionSort,
    ALL_FUNCTION_SORTS,
    "a function sort name: bool, nat, int, or real"
);
