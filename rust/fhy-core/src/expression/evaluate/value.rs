//! The values of an evaluation, and the conversions of literals to them.

use num_traits::ToPrimitive;

use crate::expression::literal::LiteralValue;
use crate::expression::symbol_type::SymbolType;

use super::error::EvaluationError;

/// A value of one of the three domains an evaluation computes in: a
/// Boolean, a 64-bit integer, or a binary64 real.
///
/// Its [`symbol_type`](Self::symbol_type) names its domain.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::SymbolType;
/// use fhy_core::expression::evaluate::Scalar;
///
/// assert_eq!(Scalar::from(3_i64).symbol_type(), SymbolType::Int);
/// assert_eq!(Scalar::from(0.5), Scalar::Real(0.5));
/// ```
#[expect(
    clippy::exhaustive_enums,
    reason = "the three domains of the value kinds, which callers match"
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    /// A Boolean.
    Bool(bool),
    /// A 64-bit signed integer.
    Int(i64),
    /// A binary64 floating-point number.
    Real(f64),
}

impl Scalar {
    /// Return the value kind of the scalar's domain.
    #[must_use]
    pub fn symbol_type(self) -> SymbolType {
        match self {
            Self::Bool(_) => SymbolType::Bool,
            Self::Int(_) => SymbolType::Int,
            Self::Real(_) => SymbolType::Real,
        }
    }
}

impl From<bool> for Scalar {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<i64> for Scalar {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

impl From<f64> for Scalar {
    fn from(value: f64) -> Self {
        Self::Real(value)
    }
}

/// Return the scalar a literal denotes in an evaluation.
///
/// A decimal becomes the binary float equal to it, and an integer the
/// 64-bit integer.
///
/// # Errors
///
/// Returns [`EvaluationError::InexactDecimal`] for a decimal no binary
/// float equals, and [`EvaluationError::IntegerOutOfRange`] for an integer
/// outside the 64-bit range.
pub(super) fn literal_scalar(value: &LiteralValue) -> Result<Scalar, EvaluationError> {
    match value {
        LiteralValue::Bool(value) => Ok(Scalar::Bool(*value)),
        LiteralValue::Int(integer) => integer
            .to_i64()
            .map(Scalar::Int)
            .ok_or_else(|| EvaluationError::IntegerOutOfRange(integer.clone())),
        LiteralValue::Float(value) => Ok(Scalar::Real(*value)),
        LiteralValue::Decimal(decimal) => decimal
            .to_f64_exact()
            .map(Scalar::Real)
            .ok_or_else(|| EvaluationError::InexactDecimal(decimal.clone())),
    }
}
