//! Unary, binary and logical expression operations.
//!
//! [`UnaryOperation`], [`BinaryOperation`] and [`LogicalOperation`] name the
//! operation of a unary, binary or logical expression node. Each operation
//! has two text forms: its wire
//! name ([`as_str`](BinaryOperation::as_str), such as `"floor_divide"`),
//! which is also its serialized form and its [`Display`](std::fmt::Display)
//! text, and its operator symbol ([`symbol`](BinaryOperation::symbol), such
//! as `"//"`), which the symbolic notation of the printer uses. Within each
//! enum, no two operations share a wire name or a symbol.

use crate::expr::wire_name::impl_wire_name_traits;

/// Every unary operation, in declaration order.
const ALL_UNARY_OPERATIONS: [UnaryOperation; 3] = [
    UnaryOperation::Negate,
    UnaryOperation::Positive,
    UnaryOperation::LogicalNot,
];

/// Every binary operation, in declaration order.
const ALL_BINARY_OPERATIONS: [BinaryOperation; 13] = [
    BinaryOperation::Add,
    BinaryOperation::Subtract,
    BinaryOperation::Multiply,
    BinaryOperation::Divide,
    BinaryOperation::FloorDivide,
    BinaryOperation::FloorMod,
    BinaryOperation::Power,
    BinaryOperation::Equal,
    BinaryOperation::NotEqual,
    BinaryOperation::Less,
    BinaryOperation::LessEqual,
    BinaryOperation::Greater,
    BinaryOperation::GreaterEqual,
];

/// Every logical operation, in declaration order.
const ALL_LOGICAL_OPERATIONS: [LogicalOperation; 2] = [LogicalOperation::And, LogicalOperation::Or];

/// The operation of a unary expression.
///
/// Serializes as its wire name ([`as_str`](Self::as_str)), and deserializes
/// only from exactly that text: a symbol, a differently cased name, or any
/// other string is rejected.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::UnaryOperation;
///
/// assert_eq!(UnaryOperation::LogicalNot.as_str(), "logical_not");
/// assert_eq!(UnaryOperation::LogicalNot.symbol(), "!");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

    /// Return whether the operation is arithmetic, so its result is a number:
    /// true for [`Negate`](Self::Negate) and [`Positive`](Self::Positive).
    #[must_use]
    pub(crate) fn is_arithmetic(self) -> bool {
        match self {
            Self::Negate | Self::Positive => true,
            Self::LogicalNot => false,
        }
    }

    /// Return whether the operation is a Boolean connective over its operand:
    /// true for [`LogicalNot`](Self::LogicalNot) only.
    #[must_use]
    pub(crate) fn is_logical_connective(self) -> bool {
        match self {
            Self::LogicalNot => true,
            Self::Negate | Self::Positive => false,
        }
    }
}

impl_wire_name_traits!(
    UnaryOperation,
    ALL_UNARY_OPERATIONS,
    "a unary operation name such as negate or logical_not"
);

/// The operation of a binary expression.
///
/// Serializes as its wire name ([`as_str`](Self::as_str)), and deserializes
/// only from exactly that text: a symbol, a differently cased name, or any
/// other string is rejected.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

    /// Return whether the operation is arithmetic, so its result is a number:
    /// true for [`Add`](Self::Add), [`Subtract`](Self::Subtract),
    /// [`Multiply`](Self::Multiply), [`Divide`](Self::Divide),
    /// [`FloorDivide`](Self::FloorDivide), [`FloorMod`](Self::FloorMod), and
    /// [`Power`](Self::Power).
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

impl_wire_name_traits!(
    BinaryOperation,
    ALL_BINARY_OPERATIONS,
    "a binary operation name such as add or floor_divide"
);

/// The operation of a logical expression: a conjunction or a disjunction of
/// two or more operands.
///
/// Serializes as its wire name ([`as_str`](Self::as_str)), and deserializes
/// only from exactly that text: a symbol, a differently cased name, or any
/// other string is rejected.
///
/// # Examples
///
/// ```
/// use fhy_core::expr::LogicalOperation;
///
/// assert_eq!(LogicalOperation::And.as_str(), "and");
/// assert_eq!(LogicalOperation::Or.symbol(), "||");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

impl_wire_name_traits!(
    LogicalOperation,
    ALL_LOGICAL_OPERATIONS,
    "a logical operation name: and or or"
);

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// Test the arithmetic unary operations are classified as arithmetic and
    /// not as connectives, and logical negation the other way round.
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

    /// Test the seven arithmetic binary operations are arithmetic and the
    /// six comparisons are not.
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

    /// Test the logical operations' wire names and symbols are distinct.
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
