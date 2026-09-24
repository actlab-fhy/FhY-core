//! Operand coercion, operator overloads, and the expression builders.
//!
//! Every builder takes its operands as [`IntoOperand`]: an expression (owned
//! or borrowed), an identifier (wrapped in an identifier reference), a
//! [`LiteralValue`], or a number (wrapped in a literal). A `bool` is not an
//! operand: a Boolean constant is written `LiteralValue::from(true)`, and an
//! equality is built with [`Expression::equals`], never with `==`, which
//! compares expressions structurally.
//!
//! The arithmetic operators `+ - * /` and unary `-` build binary and unary
//! nodes, with an expression on either side of a binary operator and any
//! operand on the other side; `/` is true division for every operand type.
//! There is no `%`. The other operations are methods named after them:
//! [`Expression::equals`], [`Expression::less`],
//! [`Expression::floor_divide`], [`Expression::floor_mod`],
//! [`Expression::power`],
//! [`Expression::logical_not`], and so on.
//! [`build_logical_and`], [`build_logical_or`], [`build_piecewise`], and
//! [`build_call`] build the nodes whose operand count varies.

use std::ops::{Add, Div, Mul, Neg, Sub};

use num_bigint::BigInt;

use crate::identifier::Identifier;

use super::error::ExpressionBuildError;
use super::literal::LiteralValue;
use super::node::{
    BinaryExpression, CallExpression, Expression, PiecewiseExpression, UnaryExpression,
};
use super::operation::{BinaryOperation, UnaryOperation};

/// The sealing supertrait of [`IntoOperand`], which carries the conversion.
mod sealed {
    use super::Expression;

    #[expect(
        unnameable_types,
        reason = "the sealing supertrait is nameable only inside this module by design"
    )]
    pub trait Sealed {
        /// Convert the value into the operand expression it stands for.
        fn into_expression(self) -> Expression;
    }
}

/// Fold `operands` to the right with the connective `operation`:
/// `a op (b op (c op d))`.
fn fold_logical_operands<I>(
    operation: BinaryOperation,
    operands: I,
) -> Result<Expression, ExpressionBuildError>
where
    I: IntoIterator,
    I::Item: IntoOperand,
{
    let mut operands: Vec<Expression> = operands
        .into_iter()
        .map(sealed::Sealed::into_expression)
        .collect();
    let count = operands.len();
    let (Some(last), true) = (operands.pop(), count >= 2) else {
        return Err(ExpressionBuildError::TooFewLogicalOperands { operation, count });
    };
    Ok(operands.into_iter().rev().fold(last, |folded, operand| {
        Expression::new_binary(operation, operand, folded)
    }))
}

/// A value usable as an operand of an expression builder.
///
/// Implemented for [`Expression`] and `&Expression` (the node itself),
/// [`Identifier`] and `&Identifier` (an identifier reference),
/// [`LiteralValue`] (a literal), and `i64`, `i32`, `u32`, [`BigInt`] and
/// `f64` (an integer or float literal). This trait is sealed: no other type
/// implements it.
///
/// A `bool` is not an operand, so a comparison result cannot stand in for a
/// Boolean constant by accident:
///
/// ```compile_fail,E0277
/// use fhy_core::expr::{Expression, UnaryOperation};
///
/// let negated = Expression::new_unary(UnaryOperation::LogicalNot, true);
/// ```
///
/// Nor is a string; a numeric text becomes an operand through
/// [`LiteralValue::parse_text`]:
///
/// ```compile_fail,E0277
/// use fhy_core::expr::{Expression, UnaryOperation};
///
/// let negated = Expression::new_unary(UnaryOperation::Negate, "1.5");
/// ```
pub trait IntoOperand: sealed::Sealed {}

impl sealed::Sealed for Expression {
    fn into_expression(self) -> Expression {
        self
    }
}

impl IntoOperand for Expression {}

impl sealed::Sealed for &Expression {
    fn into_expression(self) -> Expression {
        self.clone()
    }
}

impl IntoOperand for &Expression {}

impl sealed::Sealed for Identifier {
    fn into_expression(self) -> Expression {
        Expression::from(self)
    }
}

impl IntoOperand for Identifier {}

impl sealed::Sealed for &Identifier {
    fn into_expression(self) -> Expression {
        Expression::from(self.clone())
    }
}

impl IntoOperand for &Identifier {}

impl sealed::Sealed for LiteralValue {
    fn into_expression(self) -> Expression {
        Expression::from(self)
    }
}

impl IntoOperand for LiteralValue {}

impl sealed::Sealed for i64 {
    fn into_expression(self) -> Expression {
        Expression::from(LiteralValue::from(self))
    }
}

impl IntoOperand for i64 {}

impl sealed::Sealed for i32 {
    fn into_expression(self) -> Expression {
        Expression::from(LiteralValue::from(i64::from(self)))
    }
}

impl IntoOperand for i32 {}

impl sealed::Sealed for u32 {
    fn into_expression(self) -> Expression {
        Expression::from(LiteralValue::from(i64::from(self)))
    }
}

impl IntoOperand for u32 {}

impl sealed::Sealed for BigInt {
    fn into_expression(self) -> Expression {
        Expression::from(LiteralValue::from(self))
    }
}

impl IntoOperand for BigInt {}

impl sealed::Sealed for f64 {
    fn into_expression(self) -> Expression {
        Expression::from(LiteralValue::from(self))
    }
}

impl IntoOperand for f64 {}

impl Expression {
    /// Build a unary node applying `operation` to `operand`.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::expr::{Expression, ExpressionKind, UnaryOperation};
    ///
    /// let negated = Expression::new_unary(UnaryOperation::Negate, 5);
    /// let ExpressionKind::Unary(node) = negated.kind() else { panic!("a unary node") };
    /// assert_eq!(node.operation(), UnaryOperation::Negate);
    /// ```
    #[must_use]
    pub fn new_unary(operation: UnaryOperation, operand: impl IntoOperand) -> Self {
        Self::from(UnaryExpression::new(operation, operand.into_expression()))
    }

    /// Build a binary node applying `operation` to `left` and `right`.
    #[must_use]
    pub fn new_binary(
        operation: BinaryOperation,
        left: impl IntoOperand,
        right: impl IntoOperand,
    ) -> Self {
        Self::from(BinaryExpression::new(
            operation,
            left.into_expression(),
            right.into_expression(),
        ))
    }

    /// Build the equality comparison `self == other`.
    #[must_use]
    pub fn equals(&self, other: impl IntoOperand) -> Expression {
        Self::new_binary(BinaryOperation::Equal, self, other)
    }

    /// Build the inequality comparison `self != other`.
    #[must_use]
    pub fn not_equals(&self, other: impl IntoOperand) -> Expression {
        Self::new_binary(BinaryOperation::NotEqual, self, other)
    }

    /// Build the comparison `self < other`.
    #[must_use]
    pub fn less(&self, other: impl IntoOperand) -> Expression {
        Self::new_binary(BinaryOperation::Less, self, other)
    }

    /// Build the comparison `self <= other`.
    #[must_use]
    pub fn less_equal(&self, other: impl IntoOperand) -> Expression {
        Self::new_binary(BinaryOperation::LessEqual, self, other)
    }

    /// Build the comparison `self > other`.
    #[must_use]
    pub fn greater(&self, other: impl IntoOperand) -> Expression {
        Self::new_binary(BinaryOperation::Greater, self, other)
    }

    /// Build the comparison `self >= other`.
    #[must_use]
    pub fn greater_equal(&self, other: impl IntoOperand) -> Expression {
        Self::new_binary(BinaryOperation::GreaterEqual, self, other)
    }

    /// Build the floor division `self // other`: the quotient rounded
    /// toward negative infinity.
    #[must_use]
    pub fn floor_divide(&self, other: impl IntoOperand) -> Expression {
        Self::new_binary(BinaryOperation::FloorDivide, self, other)
    }

    /// Build the floor modulo `self % other`: the remainder of floor
    /// division, whose sign follows the divisor, so `-7 floor_mod 3` is `2`
    /// and `7 floor_mod -3` is `-2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::identifier::Identifier;
    /// use fhy_core::expr::{BinaryOperation, Expression};
    ///
    /// let x = Expression::from(Identifier::new("x"));
    /// assert_eq!(x.floor_mod(3), Expression::new_binary(BinaryOperation::FloorMod, &x, 3));
    /// ```
    #[must_use]
    pub fn floor_mod(&self, other: impl IntoOperand) -> Expression {
        Self::new_binary(BinaryOperation::FloorMod, self, other)
    }

    /// Build the exponentiation `self ** other`.
    #[must_use]
    pub fn power(&self, other: impl IntoOperand) -> Expression {
        Self::new_binary(BinaryOperation::Power, self, other)
    }

    /// Build the arithmetic identity `+self`.
    #[must_use]
    pub fn positive(&self) -> Expression {
        Self::new_unary(UnaryOperation::Positive, self)
    }

    /// Build the Boolean negation `!self`.
    #[must_use]
    pub fn logical_not(&self) -> Expression {
        Self::new_unary(UnaryOperation::LogicalNot, self)
    }
}

/// Implement one arithmetic operator trait for `Expression` and
/// `&Expression` on the left with any operand on the right, and for each
/// listed operand type on the left with `Expression` or `&Expression` on
/// the right.
macro_rules! impl_arithmetic_operator {
    ($operator:ident, $method:ident, $operation:expr, [$($left:ty),*]) => {
        impl<R: IntoOperand> $operator<R> for Expression {
            type Output = Expression;

            fn $method(self, right: R) -> Expression {
                Expression::new_binary($operation, self, right)
            }
        }

        impl<R: IntoOperand> $operator<R> for &Expression {
            type Output = Expression;

            fn $method(self, right: R) -> Expression {
                Expression::new_binary($operation, self, right)
            }
        }

        $(
            impl $operator<Expression> for $left {
                type Output = Expression;

                fn $method(self, right: Expression) -> Expression {
                    Expression::new_binary($operation, self, right)
                }
            }

            impl $operator<&Expression> for $left {
                type Output = Expression;

                fn $method(self, right: &Expression) -> Expression {
                    Expression::new_binary($operation, self, right)
                }
            }
        )*
    };
}

/// Implement every arithmetic operator trait, with each listed operand
/// type on the left.
macro_rules! impl_arithmetic_operators {
    ([$($left:ty),*]) => {
        impl_arithmetic_operator!(Add, add, BinaryOperation::Add, [$($left),*]);
        impl_arithmetic_operator!(Sub, sub, BinaryOperation::Subtract, [$($left),*]);
        impl_arithmetic_operator!(Mul, mul, BinaryOperation::Multiply, [$($left),*]);
        impl_arithmetic_operator!(Div, div, BinaryOperation::Divide, [$($left),*]);
    };
}

// Every `IntoOperand` type other than `Expression` and `&Expression`, which
// the generic impls cover.
impl_arithmetic_operators!([
    i64,
    i32,
    u32,
    BigInt,
    f64,
    Identifier,
    &Identifier,
    LiteralValue
]);

impl Neg for Expression {
    type Output = Expression;

    /// Build the arithmetic negation `-self`.
    fn neg(self) -> Expression {
        Expression::new_unary(UnaryOperation::Negate, self)
    }
}

impl Neg for &Expression {
    type Output = Expression;

    /// Build the arithmetic negation `-self`.
    fn neg(self) -> Expression {
        Expression::new_unary(UnaryOperation::Negate, self)
    }
}

/// Build the conjunction of `operands`, folded to the right:
/// `a && (b && (c && d))`.
///
/// # Errors
///
/// Returns [`ExpressionBuildError::TooFewLogicalOperands`] with the
/// operand count if `operands` holds fewer than two operands.
///
/// # Examples
///
/// ```
/// use fhy_core::identifier::Identifier;
/// use fhy_core::expr::{BinaryOperation, Expression, build_logical_and};
///
/// let x = Expression::from(Identifier::new("x"));
/// let bounded = build_logical_and([x.greater_equal(0), x.less(10)])?;
/// let expected = Expression::new_binary(
///     BinaryOperation::LogicalAnd,
///     x.greater_equal(0),
///     x.less(10),
/// );
/// assert_eq!(bounded, expected);
/// # Ok::<(), fhy_core::expr::ExpressionBuildError>(())
/// ```
pub fn build_logical_and<I>(operands: I) -> Result<Expression, ExpressionBuildError>
where
    I: IntoIterator,
    I::Item: IntoOperand,
{
    fold_logical_operands(BinaryOperation::LogicalAnd, operands)
}

/// Build the disjunction of `operands`, folded to the right:
/// `a || (b || (c || d))`.
///
/// # Errors
///
/// Returns [`ExpressionBuildError::TooFewLogicalOperands`] with the
/// operand count if `operands` holds fewer than two operands.
pub fn build_logical_or<I>(operands: I) -> Result<Expression, ExpressionBuildError>
where
    I: IntoIterator,
    I::Item: IntoOperand,
{
    fold_logical_operands(BinaryOperation::LogicalOr, operands)
}

/// Build a piecewise from `(condition, value)` cases in evaluation order and
/// an `otherwise` branch.
///
/// # Errors
///
/// Returns [`ExpressionBuildError::EmptyPiecewise`] if `cases` is empty, and
/// [`ExpressionBuildError::NonBooleanConditionLiteral`] naming the first
/// case whose condition is a literal other than a Boolean.
pub fn build_piecewise<C, V, O>(
    cases: impl IntoIterator<Item = (C, V)>,
    otherwise: O,
) -> Result<Expression, ExpressionBuildError>
where
    C: IntoOperand,
    V: IntoOperand,
    O: IntoOperand,
{
    let cases = cases
        .into_iter()
        .map(|(condition, value)| (condition.into_expression(), value.into_expression()))
        .collect();
    PiecewiseExpression::try_new(cases, otherwise.into_expression()).map(Expression::from)
}

/// Build a call of `function_name` with `arguments` in order.
///
/// # Errors
///
/// Returns [`ExpressionBuildError::EmptyFunctionName`] if `function_name` is
/// empty.
pub fn build_call<I>(function_name: &str, arguments: I) -> Result<Expression, ExpressionBuildError>
where
    I: IntoIterator,
    I::Item: IntoOperand,
{
    let arguments = arguments
        .into_iter()
        .map(sealed::Sealed::into_expression)
        .collect();
    CallExpression::try_new(function_name, arguments).map(Expression::from)
}
