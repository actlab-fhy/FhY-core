//! [`AffineForm`]: an expression as an exact linear combination of its
//! free identifiers plus a constant, and [`Expression::affine_form`], which
//! finds it.
//!
//! The analysis is exact: every coefficient and the constant are
//! [`Rational`]s, nothing is rounded, and an expression the analysis cannot
//! prove affine is declined rather than approximated.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;

use num_bigint::BigInt;

use crate::identifier::Identifier;

use super::node::{BinaryExpression, Expression, ExpressionKind};
use super::{
    BinaryOperation, LiteralValue, Rational, UnaryOperation, compute_arithmetic, raise_to_power,
};

/// The deepest tree [`Expression::affine_form`] reads, the ground
/// simplifier's bound: a node nested deeper makes the analysis decline.
const MAX_DEPTH: usize = 256;

/// The most bits a numerator or a denominator of a coefficient or of the
/// constant may have, the ground simplifier's bound on fractions.
const MAX_PART_BITS: u64 = 4096;

/// An expression as `c_1 * x_1 + ... + c_n * x_n + c_0`: a non-zero exact
/// coefficient per free identifier and an exact constant.
///
/// Two affine forms are equal exactly when they denote the same linear
/// function: their coefficients and constants are equal, a coefficient
/// that cancelled to zero is not held, and the identifiers are compared by
/// `==`. `Hash` agrees.
///
/// Displays its [canonical expression](Self::to_expression): the terms in
/// order of their identifiers' ids, then the constant, as
/// `(((3 * i) + ((-1 / 2) * j)) + 4)`.
///
/// # Examples
///
/// ```
/// use fhy_core::expression::{BigInt, Expression, Rational};
/// use fhy_core::identifier::Identifier;
///
/// let [s, t] = ["s", "t"].map(Identifier::new);
/// // (3 * s + t) - t
/// let offset = (Expression::from(3) * Expression::from(&s) + Expression::from(&t))
///     - Expression::from(&t);
///
/// let form = offset.affine_form().expect("an affine expression");
/// assert_eq!(form.coefficient(&s), Rational::from(BigInt::from(3)));
/// assert!(form.coefficient(&t).is_zero());
/// assert!(form.constant().is_zero());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AffineForm {
    /// The non-zero coefficients, by identifier, ordered by id.
    terms: BTreeMap<OrderedIdentifier, Rational>,
    constant: Rational,
}

/// An identifier ordered by its id, so a form's terms iterate in one
/// order in every run.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct OrderedIdentifier(Identifier);

impl PartialOrd for OrderedIdentifier {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedIdentifier {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.id().cmp(&other.0.id())
    }
}

impl AffineForm {
    /// Return the form of the constant `value`.
    fn of_constant(value: Rational) -> Self {
        Self {
            terms: BTreeMap::new(),
            constant: value,
        }
    }

    /// Return the coefficient of `identifier`: zero for an identifier the
    /// form does not hold.
    #[must_use]
    pub fn coefficient(&self, identifier: &Identifier) -> Rational {
        self.terms
            .get(&OrderedIdentifier(identifier.clone()))
            .map_or_else(|| Rational::from(BigInt::from(0)), Rational::clone)
    }

    /// Return the constant term.
    #[must_use]
    pub fn constant(&self) -> &Rational {
        &self.constant
    }

    /// Return each identifier with a non-zero coefficient and its
    /// coefficient, in order of the identifiers' ids.
    pub fn terms(&self) -> impl ExactSizeIterator<Item = (&Identifier, &Rational)> + '_ {
        self.terms
            .iter()
            .map(|(identifier, coefficient)| (&identifier.0, coefficient))
    }

    /// Return whether the form has no term: it denotes its constant.
    #[must_use]
    pub fn is_constant(&self) -> bool {
        self.terms.is_empty()
    }

    /// Return the canonical expression of the form: the sum, left to right,
    /// of each term `c * x` in order of the identifiers' ids (`x` alone for
    /// a coefficient of 1, `-x` for -1), then the constant when it is not
    /// zero or when there is no term. An integer coefficient or constant is
    /// an integer literal and any other the quotient of two integer
    /// literals.
    #[must_use]
    pub fn to_expression(&self) -> Expression {
        let terms = self.terms.iter().map(|(identifier, coefficient)| {
            let reference = Expression::from(&identifier.0);
            if coefficient.numerator() == &BigInt::from(1) && coefficient.is_integer() {
                reference
            } else if coefficient.numerator() == &BigInt::from(-1) && coefficient.is_integer() {
                Expression::new_unary(UnaryOperation::Negate, reference)
            } else {
                Expression::new_binary(
                    BinaryOperation::Multiply,
                    build_number(coefficient),
                    reference,
                )
            }
        });
        let constant = (!self.constant.is_zero() || self.terms.is_empty())
            .then(|| build_number(&self.constant));
        terms
            .chain(constant)
            .reduce(|sum, term| Expression::new_binary(BinaryOperation::Add, sum, term))
            .unwrap_or_else(|| unreachable!("a form with no term writes its constant"))
    }
}

/// Return the integer literal of an integer `number` and the quotient of
/// two integer literals of any other.
fn build_number(number: &Rational) -> Expression {
    let numerator = Expression::literal(number.numerator().clone());
    if number.is_integer() {
        numerator
    } else {
        Expression::new_binary(
            BinaryOperation::Divide,
            numerator,
            Expression::literal(number.denominator().clone()),
        )
    }
}

impl fmt::Display for AffineForm {
    /// Write the [canonical expression](AffineForm::to_expression).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_expression())
    }
}

/// Return `number` when neither of its parts has more bits than a
/// coefficient may have.
fn keep_within_bounds(number: Rational) -> Option<Rational> {
    (number.numerator().bits().max(number.denominator().bits()) <= MAX_PART_BITS).then_some(number)
}

/// Return `left op right`, exact and within the bounds, or `None`.
fn compute(operation: BinaryOperation, left: &Rational, right: &Rational) -> Option<Rational> {
    keep_within_bounds(compute_arithmetic(operation, left, right)?)
}

/// Return the form `left + right`, or `left - right` for a `subtraction`.
fn combine(
    mut left: AffineForm,
    right: AffineForm,
    operation: BinaryOperation,
) -> Option<AffineForm> {
    for (identifier, coefficient) in right.terms {
        let combined = match left.terms.remove(&identifier) {
            Some(held) => compute(operation, &held, &coefficient)?,
            None if operation == BinaryOperation::Subtract => -coefficient,
            None => coefficient,
        };
        if !combined.is_zero() {
            left.terms.insert(identifier, combined);
        }
    }
    left.constant = compute(operation, &left.constant, &right.constant)?;
    Some(left)
}

/// Return `form * factor`, or `form / factor` for a `Divide`, exactly.
fn scale(form: AffineForm, operation: BinaryOperation, factor: &Rational) -> Option<AffineForm> {
    let mut terms = BTreeMap::new();
    for (identifier, coefficient) in form.terms {
        let scaled = compute(operation, &coefficient, factor)?;
        if !scaled.is_zero() {
            terms.insert(identifier, scaled);
        }
    }
    Some(AffineForm {
        terms,
        constant: compute(operation, &form.constant, factor)?,
    })
}

/// Return the form of `node`, a binary operation at nesting `depth`.
fn read_binary(node: &BinaryExpression, depth: usize) -> Option<AffineForm> {
    let operation = node.operation();
    if !matches!(
        operation,
        BinaryOperation::Add
            | BinaryOperation::Subtract
            | BinaryOperation::Multiply
            | BinaryOperation::Divide
            | BinaryOperation::FloorDivide
            | BinaryOperation::FloorMod
            | BinaryOperation::Power
    ) {
        return None;
    }
    let left = read(node.left(), depth + 1)?;
    let right = read(node.right(), depth + 1)?;
    match operation {
        BinaryOperation::Add | BinaryOperation::Subtract => combine(left, right, operation),
        BinaryOperation::Multiply => {
            if left.is_constant() {
                scale(right, operation, &left.constant)
            } else if right.is_constant() {
                scale(left, operation, &right.constant)
            } else {
                None
            }
        }
        BinaryOperation::Divide => {
            if !right.is_constant() || right.constant.is_zero() {
                return None;
            }
            scale(left, operation, &right.constant)
        }
        BinaryOperation::FloorDivide | BinaryOperation::FloorMod => {
            if !(left.is_constant() && right.is_constant()) {
                return None;
            }
            compute(operation, &left.constant, &right.constant).map(AffineForm::of_constant)
        }
        BinaryOperation::Power => {
            if !(left.is_constant() && right.is_constant() && right.constant.is_integer()) {
                return None;
            }
            let power = compute_power(&left.constant, &right.constant)?;
            Some(AffineForm::of_constant(power))
        }
        BinaryOperation::Equal
        | BinaryOperation::NotEqual
        | BinaryOperation::Less
        | BinaryOperation::LessEqual
        | BinaryOperation::Greater
        | BinaryOperation::GreaterEqual => None,
    }
}

/// Return `base ** exponent`, exact and within the bounds, for an integer
/// exponent, or `None`.
fn compute_power(base: &Rational, exponent: &Rational) -> Option<Rational> {
    keep_within_bounds(raise_to_power(base, exponent)?)
}

/// Return the form of `expression` at nesting `depth`, or `None`.
fn read(expression: &Expression, depth: usize) -> Option<AffineForm> {
    if depth > MAX_DEPTH {
        return None;
    }
    match expression.kind() {
        ExpressionKind::Identifier(identifier) => Some(AffineForm {
            terms: BTreeMap::from([(
                OrderedIdentifier(identifier.clone()),
                Rational::from(BigInt::from(1)),
            )]),
            constant: Rational::from(BigInt::from(0)),
        }),
        ExpressionKind::Literal(LiteralValue::Int(value)) => {
            keep_within_bounds(Rational::from(value.clone())).map(AffineForm::of_constant)
        }
        ExpressionKind::Literal(LiteralValue::Decimal(value)) => {
            keep_within_bounds(value.to_rational()).map(AffineForm::of_constant)
        }
        ExpressionKind::Literal(LiteralValue::Bool(_) | LiteralValue::Float(_))
        | ExpressionKind::Logical(_)
        | ExpressionKind::Piecewise(_)
        | ExpressionKind::Call(_) => None,
        ExpressionKind::Unary(node) => match node.operation() {
            UnaryOperation::Positive => read(node.operand(), depth + 1),
            UnaryOperation::Negate => {
                let operand = read(node.operand(), depth + 1)?;
                Some(AffineForm {
                    terms: operand
                        .terms
                        .into_iter()
                        .map(|(identifier, coefficient)| (identifier, -coefficient))
                        .collect(),
                    constant: -operand.constant,
                })
            }
            UnaryOperation::LogicalNot => None,
        },
        ExpressionKind::Binary(node) => read_binary(node, depth),
    }
}

impl Expression {
    /// Return the expression as an exact affine form over its free
    /// identifiers, or `None` when the analysis cannot prove it one.
    ///
    /// It reads, bottom-up:
    ///
    /// - an integer or decimal literal, as its exact rational;
    /// - an identifier, as itself with coefficient 1, whatever its sort;
    /// - negation and unary plus;
    /// - addition and subtraction of two forms;
    /// - multiplication of two forms one of which is constant;
    /// - true division by a constant other than zero;
    /// - floor division, modulo and a power whose operands are both
    ///   constant, folded exactly as the ground simplifier folds them (a
    ///   power only with an integer exponent, and not `0` to a negative
    ///   power).
    ///
    /// It declines (answers `None`) a float or Boolean literal, a
    /// comparison, a logical operation, a piecewise expression, a call, a
    /// product of two non-constant forms, a division by a non-constant form
    /// or by zero, a floor division, modulo or power with a non-constant
    /// operand, a tree nested more than 256 levels, and a coefficient or
    /// constant whose numerator or denominator would exceed 4096 bits.
    /// Terms that cancel are dropped, so `(3 * s + t) - t` gives `3 * s`.
    #[must_use]
    pub fn affine_form(&self) -> Option<AffineForm> {
        read(self, 0)
    }
}
