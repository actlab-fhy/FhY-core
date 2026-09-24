//! Unary, binary and logical expression operations.
//!
//! Each operation has two text forms: its wire name
//! ([`as_str`](BinaryOperation::as_str), such as `"floor_divide"`), which is
//! also its serialized form and its [`Display`](std::fmt::Display) text, and
//! its operator symbol ([`symbol`](BinaryOperation::symbol), such as `"//"`),
//! which the symbolic notation of the printer uses. Within each enum, no two
//! operations share a wire name or a symbol.

use std::error::Error;
use std::fmt;

use serde::de::IntoDeserializer;
use serde::de::value::{Error as ValueError, StrDeserializer};
use serde::{Deserialize, Serialize};

/// Parse `text` as the variant of `T` whose serialized name it is, exactly,
/// through `T`'s derived `Deserialize`; `expected` names the enum in the
/// error.
///
/// # Errors
///
/// Returns [`UnknownNameError`] if no variant of `T` has the name `text`.
pub(crate) fn parse_variant_name<'a, T: Deserialize<'a>>(
    text: &'a str,
    expected: &'static str,
) -> Result<T, UnknownNameError> {
    let deserializer: StrDeserializer<'a, ValueError> = text.into_deserializer();
    T::deserialize(deserializer).map_err(|_unknown: ValueError| UnknownNameError {
        name: text.into(),
        expected,
    })
}

/// A name that no variant of an enum has, refused by the enum's
/// [`FromStr`](std::str::FromStr).
///
/// `Display` writes the enum's description and the name in backticks, such
/// as ``unknown binary operation `plus` ``.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::{BinaryOperation, UnknownNameError};
///
/// let error: UnknownNameError = "plus".parse::<BinaryOperation>().unwrap_err();
/// assert_eq!(error.name(), "plus");
/// assert_eq!(error.to_string(), "unknown binary operation `plus`");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownNameError {
    name: Box<str>,
    expected: &'static str,
}

impl UnknownNameError {
    /// Return the refused name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for UnknownNameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown {} `{}`", self.expected, self.name)
    }
}

impl Error for UnknownNameError {}

/// Implement `Display`, writing the text the `$text` method returns
/// (`as_str` by default), and `FromStr`, parsing exactly that text through
/// the derived `Deserialize`, for an enum; `$expected` names the enum in a
/// refusal.
macro_rules! impl_name_text {
    ($Type:ty, $expected:literal) => {
        $crate::expr::operation::impl_name_text!($Type, as_str, $expected);
    };
    ($Type:ty, $text:ident, $expected:literal) => {
        impl ::std::fmt::Display for $Type {
            /// Write the
            #[doc = concat!("[`", stringify!($text), "`](Self::", stringify!($text), ")")]
            /// text.
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(self.$text())
            }
        }

        impl ::std::str::FromStr for $Type {
            type Err = $crate::expr::UnknownNameError;

            /// Parse the variant whose
            #[doc = concat!("[`", stringify!($text), "`](Self::", stringify!($text), ")")]
            /// text is exactly `text`.
            ///
            /// # Errors
            ///
            /// Returns [`UnknownNameError`](crate::expr::UnknownNameError)
            /// if no variant has that text.
            fn from_str(text: &str) -> Result<Self, Self::Err> {
                $crate::expr::operation::parse_variant_name(text, $expected)
            }
        }
    };
}

pub(crate) use impl_name_text;

/// The operation of a unary expression.
///
/// Serializes as its wire name ([`as_str`](Self::as_str)), and deserializes
/// and parses with [`FromStr`](std::str::FromStr) only from exactly that
/// text, so a symbol or a differently cased name is refused.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::UnaryOperation;
///
/// assert_eq!(UnaryOperation::LogicalNot.as_str(), "logical_not");
/// assert_eq!(UnaryOperation::LogicalNot.symbol(), "!");
/// ```
#[expect(clippy::exhaustive_enums, reason = "passes match every operation")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnaryOperation {
    /// Arithmetic negation, `-x`.
    Negate,
    /// Arithmetic identity, `+x`.
    Positive,
    /// Boolean negation, `!x`.
    LogicalNot,
}

impl UnaryOperation {
    /// Return the wire name: `"negate"`, `"positive"`, or `"logical_not"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Negate => "negate",
            Self::Positive => "positive",
            Self::LogicalNot => "logical_not",
        }
    }

    /// Return the operator symbol: `"-"`, `"+"`, or `"!"`.
    #[must_use]
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Negate => "-",
            Self::Positive => "+",
            Self::LogicalNot => "!",
        }
    }

    /// Return whether the operation is arithmetic, so its result is a number.
    #[must_use]
    pub(crate) fn is_arithmetic(self) -> bool {
        match self {
            Self::Negate | Self::Positive => true,
            Self::LogicalNot => false,
        }
    }

    /// Return whether the operation is a Boolean connective over its operand.
    #[must_use]
    pub(crate) fn is_logical_connective(self) -> bool {
        match self {
            Self::LogicalNot => true,
            Self::Negate | Self::Positive => false,
        }
    }
}

impl_name_text!(UnaryOperation, "unary operation");

/// The operation of a binary expression.
///
/// Serializes as its wire name ([`as_str`](Self::as_str)), and deserializes
/// and parses with [`FromStr`](std::str::FromStr) only from exactly that
/// text, so a symbol or a differently cased name is refused.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::BinaryOperation;
///
/// assert_eq!(BinaryOperation::FloorDivide.as_str(), "floor_divide");
/// assert_eq!(BinaryOperation::FloorDivide.symbol(), "//");
/// assert_eq!(BinaryOperation::LessEqual.to_string(), "less_equal");
/// ```
#[expect(clippy::exhaustive_enums, reason = "passes match every operation")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinaryOperation {
    /// Addition, `a + b`.
    Add,
    /// Subtraction, `a - b`.
    Subtract,
    /// Multiplication, `a * b`.
    Multiply,
    /// True division, `a / b`: the exact real quotient whatever the operand
    /// types, so `x / 4` over integers is never truncated. Integer division
    /// rounding down is [`FloorDivide`](Self::FloorDivide).
    Divide,
    /// Division rounded toward negative infinity, `a // b`.
    FloorDivide,
    /// Remainder of floor division, `a % b`: its sign follows the divisor,
    /// so `-7 % 3` is `2` and `7 % -3` is `-2`.
    FloorMod,
    /// Exponentiation, `a ** b`.
    Power,
    /// Equality comparison, `a == b`.
    Equal,
    /// Inequality comparison, `a != b`.
    NotEqual,
    /// Strictly-less comparison, `a < b`.
    Less,
    /// Less-or-equal comparison, `a <= b`.
    LessEqual,
    /// Strictly-greater comparison, `a > b`.
    Greater,
    /// Greater-or-equal comparison, `a >= b`.
    GreaterEqual,
}

impl BinaryOperation {
    /// Return the wire name, the variant name in lowercase words joined by
    /// underscores: `"add"`, `"floor_divide"`, `"greater_equal"`, and so on.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Subtract => "subtract",
            Self::Multiply => "multiply",
            Self::Divide => "divide",
            Self::FloorDivide => "floor_divide",
            Self::FloorMod => "floor_mod",
            Self::Power => "power",
            Self::Equal => "equal",
            Self::NotEqual => "not_equal",
            Self::Less => "less",
            Self::LessEqual => "less_equal",
            Self::Greater => "greater",
            Self::GreaterEqual => "greater_equal",
        }
    }

    /// Return the operator symbol: `"+"`, `"-"`, `"*"`, `"/"`, `"//"`, `"%"`,
    /// `"**"`, `"=="`, `"!="`, `"<"`, `"<="`, `">"`, or `">="`, in variant
    /// order.
    #[must_use]
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Subtract => "-",
            Self::Multiply => "*",
            Self::Divide => "/",
            Self::FloorDivide => "//",
            Self::FloorMod => "%",
            Self::Power => "**",
            Self::Equal => "==",
            Self::NotEqual => "!=",
            Self::Less => "<",
            Self::LessEqual => "<=",
            Self::Greater => ">",
            Self::GreaterEqual => ">=",
        }
    }

    /// Return whether the operation is arithmetic, so its result is a number.
    #[must_use]
    pub(crate) fn is_arithmetic(self) -> bool {
        match self {
            Self::Add
            | Self::Subtract
            | Self::Multiply
            | Self::Divide
            | Self::FloorDivide
            | Self::FloorMod
            | Self::Power => true,
            Self::Equal
            | Self::NotEqual
            | Self::Less
            | Self::LessEqual
            | Self::Greater
            | Self::GreaterEqual => false,
        }
    }
}

impl_name_text!(BinaryOperation, "binary operation");

/// The operation of a logical expression: a conjunction or a disjunction of
/// two or more operands.
///
/// Serializes as its wire name ([`as_str`](Self::as_str)), and deserializes
/// and parses with [`FromStr`](std::str::FromStr) only from exactly that
/// text, so a symbol or a differently cased name is refused.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::LogicalOperation;
///
/// assert_eq!(LogicalOperation::And.as_str(), "and");
/// assert_eq!(LogicalOperation::Or.symbol(), "||");
/// ```
#[expect(clippy::exhaustive_enums, reason = "passes match every operation")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogicalOperation {
    /// Boolean conjunction, `a && b && ...`: true when every operand is.
    And,
    /// Boolean disjunction, `a || b || ...`: true when some operand is.
    Or,
}

impl LogicalOperation {
    /// Return the wire name: `"and"` or `"or"`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::And => "and",
            Self::Or => "or",
        }
    }

    /// Return the operator symbol written between the operands: `"&&"` or
    /// `"||"`.
    #[must_use]
    pub fn symbol(self) -> &'static str {
        match self {
            Self::And => "&&",
            Self::Or => "||",
        }
    }
}

impl_name_text!(LogicalOperation, "logical operation");

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::negate(UnaryOperation::Negate, true, false)]
    #[case::positive(UnaryOperation::Positive, true, false)]
    #[case::logical_not(UnaryOperation::LogicalNot, false, true)]
    fn unary_operation_classifies_arithmetic_and_connectives(
        #[case] operation: UnaryOperation,
        #[case] expected_arithmetic: bool,
        #[case] expected_connective: bool,
    ) {
        let is_arithmetic = operation.is_arithmetic();
        let is_connective = operation.is_logical_connective();

        assert_eq!(
            is_arithmetic, expected_arithmetic,
            "is_arithmetic({operation:?})"
        );
        assert_eq!(
            is_connective, expected_connective,
            "is_logical_connective({operation:?})"
        );
    }

    #[rstest]
    #[case::add(BinaryOperation::Add, true)]
    #[case::subtract(BinaryOperation::Subtract, true)]
    #[case::multiply(BinaryOperation::Multiply, true)]
    #[case::divide(BinaryOperation::Divide, true)]
    #[case::floor_divide(BinaryOperation::FloorDivide, true)]
    #[case::floor_mod(BinaryOperation::FloorMod, true)]
    #[case::power(BinaryOperation::Power, true)]
    #[case::equal(BinaryOperation::Equal, false)]
    #[case::not_equal(BinaryOperation::NotEqual, false)]
    #[case::less(BinaryOperation::Less, false)]
    #[case::less_equal(BinaryOperation::LessEqual, false)]
    #[case::greater(BinaryOperation::Greater, false)]
    #[case::greater_equal(BinaryOperation::GreaterEqual, false)]
    fn binary_operation_classifies_arithmetic(
        #[case] operation: BinaryOperation,
        #[case] expected_arithmetic: bool,
    ) {
        let is_arithmetic = operation.is_arithmetic();

        assert_eq!(
            is_arithmetic, expected_arithmetic,
            "is_arithmetic({operation:?})"
        );
    }

    #[rstest]
    #[case::and(LogicalOperation::And, "and", "&&")]
    #[case::or(LogicalOperation::Or, "or", "||")]
    fn logical_operation_has_its_wire_name_and_symbol(
        #[case] operation: LogicalOperation,
        #[case] expected_name: &str,
        #[case] expected_symbol: &str,
    ) {
        assert_eq!(operation.as_str(), expected_name);
        assert_eq!(operation.symbol(), expected_symbol);
    }
}
