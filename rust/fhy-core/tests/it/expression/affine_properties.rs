//! Properties of `Expression::affine_form`: the canonical expression
//! evaluates as the expression it came from, the form of a difference is
//! the difference of the forms, the terms are ordered by id, and the
//! canonical expression is a fixed point of the analysis.
//!
//! The trees are sums, differences, negations, constant multiples and
//! divisions by the constants 1, 2, 4 and 8 over three identifiers, so
//! every value an evaluation computes is a dyadic rational of small
//! magnitude, which a binary float holds exactly, so two evaluations differ by
//! zero or by at least 1/8.

use std::collections::HashMap;

use crate::support::expression::IDENTIFIER_POOL;
use crate::support::guard::{GUARD_CASES, draw_guard_cases};

use fhy_core::expression::evaluate::{Evaluator, Scalar};
use fhy_core::expression::registry::FunctionRegistry;
use fhy_core::expression::{
    BigInt, BinaryOperation, Expression, ExpressionKind, LiteralValue, Rational, UnaryOperation,
};
use fhy_core::identifier::Identifier;
use proptest::prelude::*;
use proptest::sample::select;

/// The distance under which two evaluations count as equal. The values are
/// exact multiples of 1/8, so any real difference is far above it; it only
/// satisfies the lint against comparing floats with `==`.
const TOLERANCE: f64 = 1e-9;

/// Return a strategy for affine-shaped trees over the identifier pool: sums,
/// differences, negations, unary pluses, constant multiples on either side
/// and divisions by 1, 2, 4 and 8, up to five levels deep.
fn build_affine_tree_strategy() -> BoxedStrategy<Expression> {
    let leaf = prop_oneof![
        3 => (0..IDENTIFIER_POOL.len())
            .prop_map(|index| Expression::from(IDENTIFIER_POOL[index].clone())),
        2 => (-9_i64..10).prop_map(Expression::from),
    ];
    leaf.prop_recursive(5, 40, 2, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(left, right)| left + right),
            (inner.clone(), inner.clone()).prop_map(|(left, right)| left - right),
            inner.clone().prop_map(|operand| -operand),
            inner
                .clone()
                .prop_map(|operand| Expression::new_unary(UnaryOperation::Positive, operand)),
            (-6_i64..7, inner.clone()).prop_map(|(constant, operand)| constant * operand),
            (inner.clone(), -6_i64..7).prop_map(|(operand, constant)| operand * constant),
            (inner, select(vec![1_i64, 2, 4, 8])).prop_map(|(operand, divisor)| operand / divisor),
        ]
    })
    .boxed()
}

/// Return a strategy for environments binding the pool to small integers.
fn build_environment_strategy() -> BoxedStrategy<HashMap<Identifier, Scalar>> {
    prop::collection::vec(-40_i64..41, IDENTIFIER_POOL.len())
        .prop_map(|values| {
            IDENTIFIER_POOL
                .iter()
                .cloned()
                .zip(values.into_iter().map(Scalar::Int))
                .collect()
        })
        .boxed()
}

/// Return the value of `expression` over `environment` as a float.
///
/// # Panics
///
/// Panics if the evaluation fails or computes a Boolean, which an affine
/// tree over integers does not.
fn evaluate_to_float(expression: &Expression, environment: &HashMap<Identifier, Scalar>) -> f64 {
    let registry = FunctionRegistry::new();
    let value = Evaluator::new(&registry)
        .evaluate(expression, environment)
        .expect("an affine tree evaluates over integers");
    match value {
        Scalar::Int(value) => {
            f64::from(i32::try_from(value).expect("the values stay small enough for an i32"))
        }
        Scalar::Real(value) => value,
        Scalar::Bool(_) => panic!("an affine tree computed a Boolean"),
    }
}

/// Return the difference of two rationals, `a - b`.
fn subtract(a: &Rational, b: &Rational) -> Rational {
    Rational::new(
        a.numerator() * b.denominator() - b.numerator() * a.denominator(),
        a.denominator() * b.denominator(),
    )
    .expect("a product of positive denominators is non-zero")
}

/// Return whether `expression` holds a division node.
fn contains_division(expression: &Expression) -> bool {
    matches!(
        expression.kind(),
        ExpressionKind::Binary(node) if node.operation() == BinaryOperation::Divide
    ) || expression.children().any(contains_division)
}

proptest! {
    /// Test the canonical expression of an affine tree evaluates, over any
    /// integer environment, to the value of the tree.
    #[test]
    fn affine_form_to_expression_evaluates_as_the_original(
        tree in build_affine_tree_strategy(),
        environment in build_environment_strategy(),
    ) {
        let form = tree.affine_form().expect("an affine-shaped tree is affine");
        let canonical = form.to_expression();

        let from_form = evaluate_to_float(&canonical, &environment);
        let from_tree = evaluate_to_float(&tree, &environment);
        prop_assert!(
            (from_form - from_tree).abs() <= TOLERANCE,
            "{} = {} against {} = {}", canonical, from_form, tree, from_tree
        );
    }

    /// Test the form of `a - b` has, for each identifier and for the
    /// constant, the difference of the coefficients of the forms of `a`
    /// and `b`.
    #[test]
    fn affine_form_of_a_difference_is_the_difference_of_the_forms(
        a in build_affine_tree_strategy(),
        b in build_affine_tree_strategy(),
    ) {
        let form_a = a.affine_form().expect("an affine-shaped tree is affine");
        let form_b = b.affine_form().expect("an affine-shaped tree is affine");

        let difference = (&a - &b).affine_form();

        let difference = difference.expect("a difference of affine trees is affine");
        for identifier in IDENTIFIER_POOL.iter() {
            prop_assert_eq!(
                difference.coefficient(identifier),
                subtract(&form_a.coefficient(identifier), &form_b.coefficient(identifier))
            );
        }
        prop_assert_eq!(
            difference.constant(),
            &subtract(form_a.constant(), form_b.constant())
        );
    }

    /// Test the form of the canonical expression of a form is that form.
    #[test]
    fn affine_form_of_the_canonical_expression_is_a_fixed_point(
        tree in build_affine_tree_strategy(),
    ) {
        let form = tree.affine_form().expect("an affine-shaped tree is affine");

        let again = form.to_expression().affine_form();

        prop_assert_eq!(again, Some(form));
    }

    /// Test the terms of a form are in strictly increasing id order, each
    /// has a non-zero coefficient, and the form holds exactly the identifiers
    /// with a non-zero coefficient.
    #[test]
    fn affine_form_terms_are_ordered_by_id_with_non_zero_coefficients(
        tree in build_affine_tree_strategy(),
    ) {
        let form = tree.affine_form().expect("an affine-shaped tree is affine");

        let terms: Vec<_> = form.terms().collect();

        prop_assert!(terms.windows(2).all(|pair| pair[0].0.id() < pair[1].0.id()));
        prop_assert!(terms.iter().all(|(_, coefficient)| !coefficient.is_zero()));
        let held = IDENTIFIER_POOL
            .iter()
            .filter(|identifier| !form.coefficient(identifier).is_zero())
            .count();
        prop_assert_eq!(terms.len(), held);
        prop_assert_eq!(form.is_constant(), terms.is_empty());
    }

    /// Test scaling a tree by a constant scales every coefficient and the
    /// constant.
    #[test]
    fn affine_form_of_a_constant_multiple_scales_the_form(
        tree in build_affine_tree_strategy(),
        factor in -9_i64..10,
    ) {
        let form = tree.affine_form().expect("an affine-shaped tree is affine");
        let factor = Rational::new(BigInt::from(factor), BigInt::from(1)).expect("one");

        let scaled = (factor_expression(&factor) * &tree)
            .affine_form()
            .expect("a constant multiple is affine");

        let multiply = |rational: &Rational| {
            Rational::new(
                rational.numerator() * factor.numerator(),
                rational.denominator().clone(),
            )
            .expect("a positive denominator")
        };
        for identifier in IDENTIFIER_POOL.iter() {
            prop_assert_eq!(scaled.coefficient(identifier), multiply(&form.coefficient(identifier)));
        }
        prop_assert_eq!(scaled.constant(), &multiply(form.constant()));
    }
}

/// Return the integer literal expression of the integer rational `value`.
///
/// # Panics
///
/// Panics if `value` is not an integer.
fn factor_expression(value: &Rational) -> Expression {
    Expression::from(LiteralValue::from(
        value
            .to_integer()
            .expect("the factor is an integer")
            .clone(),
    ))
}

// ---------------------------------------------------------------------------
// The strategy reaches what the properties claim to cover
// ---------------------------------------------------------------------------

/// Test the generated trees mention several identifiers, divide, and are
/// not mostly single leaves, so the properties are not vacuous.
#[test]
fn affine_tree_strategy_draws_trees_with_identifiers_and_divisions() {
    let trees = draw_guard_cases(&build_affine_tree_strategy());

    let with_two_identifiers = trees
        .iter()
        .filter(|tree| tree.free_identifiers().len() >= 2)
        .count();
    let with_division = trees.iter().filter(|tree| contains_division(tree)).count();
    let compound = trees
        .iter()
        .filter(|tree| tree.children().len() > 0)
        .count();

    assert!(
        with_two_identifiers >= GUARD_CASES / 8,
        "{with_two_identifiers}"
    );
    assert!(with_division >= GUARD_CASES / 8, "{with_division}");
    assert!(compound >= GUARD_CASES / 2, "{compound}");
}

/// Test the generated trees reach forms with a fractional coefficient, a
/// cancelled identifier, a constant form, and a form with several terms.
#[test]
fn affine_tree_strategy_reaches_fractions_cancellations_and_constants() {
    let trees = draw_guard_cases(&build_affine_tree_strategy());

    let forms: Vec<_> = trees
        .iter()
        .map(|tree| tree.affine_form().expect("an affine-shaped tree is affine"))
        .collect();

    let fractional = forms
        .iter()
        .filter(|form| {
            form.terms()
                .any(|(_, coefficient)| !coefficient.is_integer())
        })
        .count();
    let cancelled = trees
        .iter()
        .zip(&forms)
        .filter(|(tree, form)| tree.free_identifiers().len() > form.terms().len())
        .count();
    let constant = forms.iter().filter(|form| form.is_constant()).count();
    let several_terms = forms.iter().filter(|form| form.terms().len() >= 2).count();

    assert!(fractional >= 4, "{fractional}");
    assert!(cancelled >= 4, "{cancelled}");
    assert!(constant >= 4, "{constant}");
    assert!(several_terms >= GUARD_CASES / 8, "{several_terms}");
}
