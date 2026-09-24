//! Operand conversions, operator overloads, and the expression builders.
//!
//! Every builder takes its operands as `impl Into<Expression>`: an
//! expression (owned, or borrowed to share its node), an identifier (owned
//! or borrowed), a [`LiteralValue`], or a number (`i32`, `i64`, `i128`,
//! `u32`, `u64`, `usize`, [`BigInt`] or `f64`). A `bool` is not an operand: a
//! Boolean constant is built with [`Expression::literal`], and an equality
//! with [`Expression::equals`], never with `==`, which compares expressions
//! structurally.
//!
//! The operators `+ - * /`, unary `-` and `!` build binary and unary nodes,
//! with an expression on at least one side of a binary operator; `/` is true
//! division for every operand type, and there is no `%`. The other
//! operations are methods named after them, such as [`Expression::less`] and
//! [`Expression::floor_mod`]. [`Expression::all`], [`Expression::any`],
//! [`Expression::new_logical`], [`Expression::piecewise`], and
//! [`Expression::call`] build the nodes whose operand count varies.

use std::ops::{Add, Div, Mul, Neg, Not, Sub};

use num_bigint::BigInt;

use crate::identifier::Identifier;

use super::callee::Callee;
use super::error::PiecewiseError;
use super::literal::LiteralValue;
use super::node::{
    BinaryExpression, CallExpression, Expression, ExpressionKind, LogicalExpression,
    PiecewiseExpression, UnaryExpression,
};
use super::operation::{BinaryOperation, LogicalOperation, UnaryOperation};

/// Implement `From<$number> for Expression` for each number type, building
/// the literal the number converts to.
macro_rules! impl_from_number {
    ($($number:ty),*) => {
        $(
            impl From<$number> for Expression {
                /// Wrap the number in a literal expression.
                fn from(value: $number) -> Self {
                    Self::from(LiteralValue::from(value))
                }
            }
        )*
    };
}

impl_from_number!(i32, i64, i128, u32, u64, usize, BigInt, f64);

impl From<&Identifier> for Expression {
    /// Wrap a clone of the identifier in an identifier reference.
    fn from(identifier: &Identifier) -> Self {
        Self::from(identifier.clone())
    }
}

impl From<&Expression> for Expression {
    /// Return a handle sharing the node.
    fn from(expression: &Expression) -> Self {
        expression.clone()
    }
}

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
    pub fn new_unary(operation: UnaryOperation, operand: impl Into<Expression>) -> Self {
        Self::from_kind(ExpressionKind::Unary(UnaryExpression::new(
            operation,
            operand.into(),
        )))
    }

    /// Build a binary node applying `operation` to `left` and `right`.
    #[must_use]
    pub fn new_binary(
        operation: BinaryOperation,
        left: impl Into<Expression>,
        right: impl Into<Expression>,
    ) -> Self {
        Self::from_kind(ExpressionKind::Binary(BinaryExpression::new(
            operation,
            left.into(),
            right.into(),
        )))
    }

    /// Build a piecewise from `(condition, value)` cases in evaluation order
    /// and an `otherwise` branch: the value of the first case whose
    /// condition holds, or the otherwise branch when none does.
    ///
    /// # Errors
    ///
    /// Returns [`PiecewiseError::NoCases`] if `cases` is empty, and
    /// [`PiecewiseError::NonBooleanConditionLiteral`] naming the first case
    /// whose condition is a literal other than a Boolean.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::identifier::Identifier;
    /// use fhy_core::expr::{Expression, PiecewiseError};
    ///
    /// let x = Expression::from(Identifier::new("x"));
    /// let sign = Expression::piecewise([(x.greater(0), 1), (x.less(0), -1)], 0)?;
    /// assert_eq!(sign.to_string(), "{1 if (x > 0); -1 if (x < 0); 0 otherwise}");
    ///
    /// let no_cases: [(Expression, Expression); 0] = [];
    /// assert_eq!(Expression::piecewise(no_cases, 0), Err(PiecewiseError::NoCases));
    /// # Ok::<(), PiecewiseError>(())
    /// ```
    pub fn piecewise<C, V, O>(
        cases: impl IntoIterator<Item = (C, V)>,
        otherwise: O,
    ) -> Result<Self, PiecewiseError>
    where
        C: Into<Expression>,
        V: Into<Expression>,
        O: Into<Expression>,
    {
        let cases = cases
            .into_iter()
            .map(|(condition, value)| (condition.into(), value.into()))
            .collect();
        PiecewiseExpression::try_new(cases, otherwise.into())
            .map(|node| Self::from_kind(ExpressionKind::Piecewise(node)))
    }

    /// Build a call of `callee` with `arguments` in order.
    ///
    /// The callee is a [`BuiltinFunction`](super::builtins::BuiltinFunction)
    /// or a [`FunctionName`](super::FunctionName), which is valid by
    /// construction, so the call cannot be refused. The argument count is
    /// not checked against a built-in function's arity.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::identifier::Identifier;
    /// use fhy_core::expr::builtins::BuiltinFunction;
    /// use fhy_core::expr::{Callee, Expression};
    ///
    /// let x = Expression::from(Identifier::new("x"));
    /// let y = Expression::from(Identifier::new("y"));
    /// let larger = Expression::call(BuiltinFunction::Max, [&x, &y]);
    /// assert_eq!(larger.to_string(), "max(x, y)");
    ///
    /// let custom = Expression::call("softplus".parse::<Callee>()?, [&x]);
    /// assert_eq!(custom.to_string(), "softplus(x)");
    /// # Ok::<(), fhy_core::expr::FunctionNameError>(())
    /// ```
    #[must_use]
    pub fn call<I>(callee: impl Into<Callee>, arguments: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<Expression>,
    {
        let arguments = arguments.into_iter().map(Into::into).collect();
        Self::from_kind(ExpressionKind::Call(CallExpression::new(
            callee.into(),
            arguments,
        )))
    }

    /// Build the literal holding `value`, the explicit way to make one,
    /// including a Boolean one.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::expr::{Expression, ExpressionKind, LiteralValue};
    ///
    /// let truth = Expression::literal(true);
    /// assert!(matches!(truth.kind(), ExpressionKind::Literal(LiteralValue::Bool(true))));
    /// ```
    #[must_use]
    pub fn literal(value: impl Into<LiteralValue>) -> Self {
        Self::from(value.into())
    }

    /// Build the conjunction or disjunction `operation` of `operands`.
    ///
    /// Of no operand, this is the literal `true` for
    /// [`And`](LogicalOperation::And) and `false` for
    /// [`Or`](LogicalOperation::Or). Of one operand, it is that operand's
    /// handle, unchanged. Of two or more, it is one logical node over
    /// exactly the given operands, in order. A logical operand of the same
    /// operation is never flattened into the new node, so
    /// `all([x, all([y, z])])` keeps its nested node.
    ///
    /// Build a large conjunction with one call over an iterator:
    /// `acc = acc.and(c)` in a loop nests one level per step.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::identifier::Identifier;
    /// use fhy_core::expr::{Expression, ExpressionKind, LiteralValue, LogicalOperation};
    ///
    /// let x = Expression::from(Identifier::new("x"));
    /// let bounds = Expression::new_logical(LogicalOperation::And, [x.greater_equal(0), x.less(10)]);
    /// let ExpressionKind::Logical(node) = bounds.kind() else { panic!("a logical node") };
    /// assert_eq!(node.operands().len(), 2);
    ///
    /// let nothing: [Expression; 0] = [];
    /// assert_eq!(
    ///     Expression::new_logical(LogicalOperation::Or, nothing),
    ///     Expression::from(LiteralValue::from(false)),
    /// );
    /// assert!(Expression::ptr_eq(&Expression::new_logical(LogicalOperation::And, [&x]), &x));
    /// ```
    #[must_use]
    pub fn new_logical<I>(operation: LogicalOperation, operands: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<Expression>,
    {
        let mut operands = operands.into_iter().map(Into::into);
        let Some(first) = operands.next() else {
            let identity = match operation {
                LogicalOperation::And => true,
                LogicalOperation::Or => false,
            };
            return Self::literal(identity);
        };
        let Some(second) = operands.next() else {
            return first;
        };
        let operands = [first, second].into_iter().chain(operands).collect();
        Self::from_kind(ExpressionKind::Logical(LogicalExpression::new(
            operation, operands,
        )))
    }

    /// Build the conjunction of `operands`, as
    /// [`new_logical`](Self::new_logical) builds it for
    /// [`And`](LogicalOperation::And).
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::identifier::Identifier;
    /// use fhy_core::expr::{Expression, ExpressionKind};
    ///
    /// let x = Expression::from(Identifier::new("x"));
    /// let bounded = Expression::all((0..100).map(|bound| x.less(bound)));
    ///
    /// let ExpressionKind::Logical(node) = bounded.kind() else { panic!("a logical node") };
    /// assert_eq!(node.operands().len(), 100);
    /// ```
    #[must_use]
    pub fn all<I>(operands: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<Expression>,
    {
        Self::new_logical(LogicalOperation::And, operands)
    }

    /// Build the disjunction of `operands`, as
    /// [`new_logical`](Self::new_logical) builds it for
    /// [`Or`](LogicalOperation::Or).
    #[must_use]
    pub fn any<I>(operands: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<Expression>,
    {
        Self::new_logical(LogicalOperation::Or, operands)
    }

    /// Build the two-operand conjunction `self && other`, the same as
    /// `Expression::all([self, other])`.
    #[must_use]
    pub fn and(&self, other: impl Into<Expression>) -> Expression {
        Self::all([self.clone(), other.into()])
    }

    /// Build the two-operand disjunction `self || other`, the same as
    /// `Expression::any([self, other])`.
    #[must_use]
    pub fn or(&self, other: impl Into<Expression>) -> Expression {
        Self::any([self.clone(), other.into()])
    }

    /// Build the equality comparison `self == other`.
    #[must_use]
    pub fn equals(&self, other: impl Into<Expression>) -> Expression {
        Self::new_binary(BinaryOperation::Equal, self, other)
    }

    /// Build the inequality comparison `self != other`.
    #[must_use]
    pub fn not_equals(&self, other: impl Into<Expression>) -> Expression {
        Self::new_binary(BinaryOperation::NotEqual, self, other)
    }

    /// Build the comparison `self < other`.
    #[must_use]
    pub fn less(&self, other: impl Into<Expression>) -> Expression {
        Self::new_binary(BinaryOperation::Less, self, other)
    }

    /// Build the comparison `self <= other`.
    #[must_use]
    pub fn less_equal(&self, other: impl Into<Expression>) -> Expression {
        Self::new_binary(BinaryOperation::LessEqual, self, other)
    }

    /// Build the comparison `self > other`.
    #[must_use]
    pub fn greater(&self, other: impl Into<Expression>) -> Expression {
        Self::new_binary(BinaryOperation::Greater, self, other)
    }

    /// Build the comparison `self >= other`.
    #[must_use]
    pub fn greater_equal(&self, other: impl Into<Expression>) -> Expression {
        Self::new_binary(BinaryOperation::GreaterEqual, self, other)
    }

    /// Build the floor division `self // other`: the quotient rounded
    /// toward negative infinity.
    #[must_use]
    pub fn floor_divide(&self, other: impl Into<Expression>) -> Expression {
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
    pub fn floor_mod(&self, other: impl Into<Expression>) -> Expression {
        Self::new_binary(BinaryOperation::FloorMod, self, other)
    }

    /// Build the exponentiation `self ** other`.
    #[must_use]
    pub fn power(&self, other: impl Into<Expression>) -> Expression {
        Self::new_binary(BinaryOperation::Power, self, other)
    }

    /// Build the arithmetic identity `+self`.
    #[must_use]
    pub fn positive(&self) -> Expression {
        Self::new_unary(UnaryOperation::Positive, self)
    }
}

/// Implement one arithmetic operator trait for `Expression` and
/// `&Expression` on the left with any operand on the right, and for each
/// listed operand type on the left with `Expression` or `&Expression` on
/// the right.
macro_rules! impl_arithmetic_operator {
    ($operator:ident, $method:ident, $operation:expr, [$($left:ty),*]) => {
        impl<R: Into<Expression>> $operator<R> for Expression {
            type Output = Expression;

            fn $method(self, right: R) -> Expression {
                Expression::new_binary($operation, self, right)
            }
        }

        impl<R: Into<Expression>> $operator<R> for &Expression {
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

// Every type converting into an `Expression` other than `Expression` and
// `&Expression`, which the generic impls cover.
impl_arithmetic_operators!([
    i32,
    i64,
    i128,
    u32,
    u64,
    usize,
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

impl Not for Expression {
    type Output = Expression;

    /// Build the Boolean negation `!self`.
    fn not(self) -> Expression {
        Expression::new_unary(UnaryOperation::LogicalNot, self)
    }
}

impl Not for &Expression {
    type Output = Expression;

    /// Build the Boolean negation `!self`, sharing the operand.
    ///
    /// # Examples
    ///
    /// ```
    /// use fhy_core::identifier::Identifier;
    /// use fhy_core::expr::{Expression, UnaryOperation};
    ///
    /// let p = Expression::from(Identifier::new("p"));
    /// assert_eq!(!&p, Expression::new_unary(UnaryOperation::LogicalNot, &p));
    /// ```
    fn not(self) -> Expression {
        Expression::new_unary(UnaryOperation::LogicalNot, self)
    }
}
