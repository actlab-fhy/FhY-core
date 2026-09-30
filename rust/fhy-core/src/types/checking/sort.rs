//! The mapping between function sorts and core data types.

use crate::expression::FunctionSort;

use super::super::core_data_type::CoreDataType;

impl CoreDataType {
    /// Return whether a value of this type satisfies `sort`: `Bool` the
    /// Boolean sort, the unsigned integers `Nat`, every integer `Int`, and
    /// every integer and real float `Real`.
    #[must_use]
    pub fn is_compatible_with_sort(self, sort: FunctionSort) -> bool {
        match sort {
            FunctionSort::Bool => self == Self::Bool,
            FunctionSort::Nat => self.is_unsigned(),
            FunctionSort::Int => self.is_integral(),
            FunctionSort::Real => self.is_integral() || self.is_real_float(),
        }
    }

    /// Return the concrete type a value of `sort` takes: `Bool`, `Uint32`,
    /// `Int64` or `Float64`.
    #[must_use]
    pub const fn of_sort(sort: FunctionSort) -> Self {
        match sort {
            FunctionSort::Bool => Self::Bool,
            FunctionSort::Nat => Self::Uint32,
            FunctionSort::Int => Self::Int64,
            FunctionSort::Real => Self::Float64,
        }
    }
}
