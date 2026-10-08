//! [`AffineForm`]: an expression as an exact linear combination of its
//! free identifiers plus a constant, and [`Expression::affine_form`], which
//! finds it.
//!
//! The analysis is exact: every coefficient and the constant are
//! [`Rational`]s, nothing is rounded, and an expression the analysis cannot
//! prove affine is declined rather than approximated.

#![expect(
    unused_variables,
    clippy::todo,
    reason = "interface stub; bodies are todo!() until implementation"
)]

use std::collections::BTreeMap;
use std::fmt;

use crate::identifier::Identifier;

use super::Rational;
use super::node::Expression;

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
/// `3 * i + -1/2 * j + 4`.
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
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedIdentifier {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        todo!()
    }
}

impl AffineForm {
    /// Return the coefficient of `identifier`: zero for an identifier the
    /// form does not hold.
    #[must_use]
    pub fn coefficient(&self, identifier: &Identifier) -> Rational {
        todo!()
    }

    /// Return the constant term.
    #[must_use]
    pub fn constant(&self) -> &Rational {
        todo!()
    }

    /// Return each identifier with a non-zero coefficient and its
    /// coefficient, in order of the identifiers' ids.
    pub fn terms(&self) -> impl ExactSizeIterator<Item = (&Identifier, &Rational)> + '_ {
        self.terms
            .iter()
            .map(|entry| -> (&Identifier, &Rational) { todo!() })
    }

    /// Return whether the form has no term: it denotes its constant.
    #[must_use]
    pub fn is_constant(&self) -> bool {
        todo!()
    }

    /// Return the canonical expression of the form: the sum, left to right,
    /// of each term `c * x` in order of the identifiers' ids (`x` alone for
    /// a coefficient of 1, `-x` for -1), then the constant when it is not
    /// zero or when there is no term. An integer coefficient or constant is
    /// an integer literal and any other the quotient of two integer
    /// literals.
    #[must_use]
    pub fn to_expression(&self) -> Expression {
        todo!()
    }
}

impl fmt::Display for AffineForm {
    /// Write the [canonical expression](AffineForm::to_expression).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
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
        todo!()
    }
}
