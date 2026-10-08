//! Stories of `Rational` and of `Expression::affine_form`: the exact
//! rational type, the behaviour table of the analysis, every shape it
//! declines, its two bounds, the form's std traits, and the canonical
//! expression the form writes.

use crate::support::expression::{
    build_call_or_panic, build_decimal_literal, build_deep_sum, build_identifier, build_literal,
    build_piecewise_or_panic, expect_binary, expect_literal, expect_unary,
};
use crate::support::hashing::hash_of;

use fhy_core::expression::{
    AffineForm, BigInt, BinaryOperation, Expression, LiteralValue, Rational, UnaryOperation,
};
use fhy_core::identifier::Identifier;
use rstest::rstest;

/// Return the rational `numerator / denominator`, built by `Rational::new`.
fn rat(numerator: i64, denominator: i64) -> Rational {
    Rational::new(BigInt::from(numerator), BigInt::from(denominator))
        .expect("a non-zero denominator")
}

/// Return the two fresh identifiers `x` and `y`, `x` minted first so its id
/// is the smaller, with a reference to each.
fn build_x_and_y() -> ((Identifier, Expression), (Identifier, Expression)) {
    (build_identifier("x"), build_identifier("y"))
}

/// Return the terms of `form` as owned pairs, in iteration order.
fn collect_terms(form: &AffineForm) -> Vec<(Identifier, Rational)> {
    form.terms()
        .map(|(identifier, coefficient)| (identifier.clone(), coefficient.clone()))
        .collect()
}

/// Return `-(-(... leaf))`, `depth` negations deep.
fn build_deep_negation(leaf: &Expression, depth: usize) -> Expression {
    (0..depth).fold(leaf.clone(), |tree, _| -tree)
}

// ---------------------------------------------------------------------------
// Rational
// ---------------------------------------------------------------------------

/// Test `Rational::new` reduces to lowest terms and moves the sign to the
/// numerator.
#[rstest]
#[case::negative_denominator(2, -4, -1, 2)]
#[case::both_negative(-6, -9, 2, 3)]
#[case::already_reduced(3, 7, 3, 7)]
#[case::integer(10, 5, 2, 1)]
#[case::zero_numerator(0, -7, 0, 1)]
fn rational_new_reduces_and_normalizes_the_sign(
    #[case] numerator: i64,
    #[case] denominator: i64,
    #[case] expected_numerator: i64,
    #[case] expected_denominator: i64,
) {
    let rational = rat(numerator, denominator);

    assert_eq!(rational.numerator(), &BigInt::from(expected_numerator));
    assert_eq!(rational.denominator(), &BigInt::from(expected_denominator));
}

/// Test `Rational::new` refuses a zero denominator, whatever the numerator.
#[rstest]
#[case::zero_over_zero(0)]
#[case::positive_over_zero(5)]
#[case::negative_over_zero(-5)]
fn rational_new_refuses_a_zero_denominator(#[case] numerator: i64) {
    let rational = Rational::new(BigInt::from(numerator), BigInt::from(0));

    assert_eq!(rational, None);
}

/// Test equal values built from different parts are equal rationals.
#[test]
fn rational_new_gives_equal_rationals_for_equal_values() {
    assert_eq!(rat(1, 2), rat(2, 4));
    assert_eq!(rat(1, 2), rat(-3, -6));
    assert_ne!(rat(1, 2), rat(-1, 2));
    assert_ne!(rat(1, 2), rat(1, 3));
}

/// Test equal rationals hash equal.
#[test]
fn rational_hash_agrees_with_equality() {
    assert_eq!(hash_of(&rat(1, 2)), hash_of(&rat(2, 4)));
    assert_eq!(hash_of(&rat(-1, 2)), hash_of(&rat(3, -6)));
    assert_eq!(hash_of(&rat(0, 5)), hash_of(&rat(0, -9)));
}

/// Test the display of a rational: the numerator alone for an integer, and
/// `numerator/denominator` otherwise.
#[rstest]
#[case::positive_integer(rat(3, 1), "3")]
#[case::reduced_integer(rat(6, 2), "3")]
#[case::negative_fraction(rat(2, -4), "-1/2")]
#[case::positive_fraction(rat(3, 7), "3/7")]
#[case::zero(rat(0, 9), "0")]
#[case::negative_integer(rat(-8, 1), "-8")]
fn rational_displays_an_integer_alone_and_a_fraction_as_a_quotient(
    #[case] rational: Rational,
    #[case] expected: &str,
) {
    assert_eq!(rational.to_string(), expected);
}

/// Test `is_integer`, `to_integer` and `is_zero` over integers, fractions
/// and zero.
#[rstest]
#[case::integer(rat(4, 2), true, Some(2), false)]
#[case::negative_integer(rat(-4, 2), true, Some(-2), false)]
#[case::fraction(rat(1, 2), false, None, false)]
#[case::zero(rat(0, 3), true, Some(0), true)]
fn rational_classifies_integers_and_zero(
    #[case] rational: Rational,
    #[case] is_integer: bool,
    #[case] integer: Option<i64>,
    #[case] is_zero: bool,
) {
    assert_eq!(rational.is_integer(), is_integer);
    assert_eq!(rational.to_integer(), integer.map(BigInt::from).as_ref());
    assert_eq!(rational.is_zero(), is_zero);
}

/// Test a rational built from a big integer is that integer.
#[test]
fn rational_from_a_big_integer_is_that_integer() {
    let big = BigInt::from(1) << 200_usize;

    let rational = Rational::from(big.clone());

    assert_eq!(rational.to_integer(), Some(&big));
    assert_eq!(rational.denominator(), &BigInt::from(1));
    assert_eq!(rational, Rational::new(big, BigInt::from(1)).expect("one"));
}

/// Test negating a rational flips its sign and keeps its denominator.
#[test]
fn rational_neg_flips_the_sign() {
    assert_eq!(-rat(1, 2), rat(-1, 2));
    assert_eq!(-rat(-3, 1), rat(3, 1));
    assert_eq!(-rat(0, 1), rat(0, 1));
}

/// Test rationals order numerically, not by their parts.
#[test]
fn rational_orders_numerically() {
    let ascending = [
        rat(-1, 2),
        rat(0, 1),
        rat(1, 3),
        rat(1, 2),
        rat(2, 3),
        rat(1, 1),
    ];

    for (index, smaller) in ascending.iter().enumerate() {
        for larger in &ascending[index + 1..] {
            assert!(smaller < larger, "{smaller} < {larger}");
            assert!(larger > smaller, "{larger} > {smaller}");
        }
    }
    assert_eq!(rat(-1, 2).cmp(&rat(-2, 4)), std::cmp::Ordering::Equal);
    assert!(rat(-1, 2) < rat(1, 3));
    assert!(rat(-1, 3) > rat(-1, 2));
}

// ---------------------------------------------------------------------------
// The behaviour table
// ---------------------------------------------------------------------------

/// Test the form of each affine expression: its terms over `x` (index 0)
/// and `y` (index 1), each as `(identifier, numerator, denominator)`, and
/// its constant.
#[rstest]
#[case::offset_cancelled(
    |x: &Expression, y: &Expression| (build_literal(3) * x + y) - y,
    &[(0, 3, 1)],
    (0, 1)
)]
#[case::halves_add_to_one(|x: &Expression, _: &Expression| x / 2 + x / 2, &[(0, 1, 1)], (0, 1))]
#[case::scaled_sum_minus_one(|x: &Expression, _: &Expression| build_literal(2) * (x + 1) - 1, &[(0, 2, 1)], (1, 1))]
#[case::floor_division_folds(|x: &Expression, _: &Expression| build_literal(7).floor_divide(2) + x, &[(0, 1, 1)], (3, 1))]
#[case::modulo_folds(|x: &Expression, _: &Expression| build_literal(7).floor_mod(4) + x, &[(0, 1, 1)], (3, 1))]
#[case::decimal_literal_scales(
    |x: &Expression, _: &Expression| build_decimal_literal("0.25") * x,
    &[(0, 1, 4)],
    (0, 1)
)]
#[case::decimal_literal_constant(
    |x: &Expression, _: &Expression| x + build_decimal_literal("0.5"),
    &[(0, 1, 1)],
    (1, 2)
)]
#[case::negation(|x: &Expression, _: &Expression| -x, &[(0, -1, 1)], (0, 1))]
#[case::unary_plus(
    |x: &Expression, _: &Expression| Expression::new_unary(UnaryOperation::Positive, x),
    &[(0, 1, 1)],
    (0, 1)
)]
#[case::constant(|_: &Expression, _: &Expression| build_literal(5), &[], (5, 1))]
#[case::power_of_constants_scales(|x: &Expression, _: &Expression| build_literal(2).power(3) * x, &[(0, 8, 1)], (0, 1))]
#[case::negative_power_is_exact(|x: &Expression, _: &Expression| x + build_literal(2).power(-1), &[(0, 1, 1)], (1, 2))]
#[case::constant_on_the_left_of_a_product(|x: &Expression, _: &Expression| build_literal(3) * x, &[(0, 3, 1)], (0, 1))]
#[case::constant_on_the_right_of_a_product(|x: &Expression, _: &Expression| x * 3, &[(0, 3, 1)], (0, 1))]
#[case::constant_product_of_constants(|x: &Expression, _: &Expression| (build_literal(2) * 3) * x, &[(0, 6, 1)], (0, 1))]
#[case::division_by_a_constant_expression(
    |x: &Expression, _: &Expression| x / (build_literal(1) + 3),
    &[(0, 1, 4)],
    (0, 1)
)]
#[case::two_identifiers(|x: &Expression, y: &Expression| x - build_literal(2) * y + 4, &[(0, 1, 1), (1, -2, 1)], (4, 1))]
#[case::identifier_with_a_zero_coefficient_dropped(|x: &Expression, y: &Expression| build_literal(0) * x + y, &[(1, 1, 1)], (0, 1))]
#[case::cancelled_to_zero(|x: &Expression, y: &Expression| (x + y) - (y + x), &[], (0, 1))]
#[case::negative_constant(|x: &Expression, _: &Expression| x - 3, &[(0, 1, 1)], (-3, 1))]
fn affine_form_reads_an_affine_expression(
    #[case] build: fn(&Expression, &Expression) -> Expression,
    #[case] expected_terms: &[(usize, i64, i64)],
    #[case] expected_constant: (i64, i64),
) {
    let (x, y) = build_x_and_y();
    let identifiers = [x.0.clone(), y.0.clone()];
    let expression = build(&x.1, &y.1);

    let form = expression.affine_form().expect("an affine expression");

    let expected: Vec<(Identifier, Rational)> = expected_terms
        .iter()
        .map(|&(index, numerator, denominator)| {
            (identifiers[index].clone(), rat(numerator, denominator))
        })
        .collect();
    assert_eq!(collect_terms(&form), expected);
    assert_eq!(
        form.constant(),
        &rat(expected_constant.0, expected_constant.1)
    );
    assert_eq!(form.is_constant(), expected_terms.is_empty());
    for (index, identifier) in identifiers.iter().enumerate() {
        let coefficient = expected_terms
            .iter()
            .find(|term| term.0 == index)
            .map_or_else(|| rat(0, 1), |term| rat(term.1, term.2));
        assert_eq!(form.coefficient(identifier), coefficient);
    }
}

/// Test the offset story: `(3 * s + t) - t` is `3 * s`, the cancelled `t`
/// has a zero coefficient and the constant is zero.
#[test]
fn affine_form_drops_a_cancelled_term() {
    let (s, s_reference) = build_identifier("s");
    let (t, t_reference) = build_identifier("t");
    let offset = (build_literal(3) * &s_reference + &t_reference) - &t_reference;

    let form = offset.affine_form().expect("an affine expression");

    assert_eq!(form.coefficient(&s), rat(3, 1));
    assert!(form.coefficient(&t).is_zero());
    assert!(form.constant().is_zero());
    assert_eq!(form.terms().len(), 1);
    assert!(!form.is_constant());
}

/// Test a constant expression is a constant form with the constant's value.
#[test]
fn affine_form_of_a_constant_is_constant() {
    let five = build_literal(5);

    let form = five.affine_form().expect("a constant is affine");

    assert!(form.is_constant());
    assert_eq!(form.constant(), &rat(5, 1));
    assert_eq!(form.terms().len(), 0);
}

/// Test `x - x` has no term and a zero constant, and writes as the literal
/// `0`.
#[test]
fn affine_form_of_a_cancelling_difference_is_the_literal_zero() {
    let (_, x) = build_identifier("x");
    let difference = &x - &x;

    let form = difference.affine_form().expect("an affine expression");

    assert!(form.is_constant());
    assert!(form.constant().is_zero());
    assert_eq!(form.terms().len(), 0);
    let expression = form.to_expression();
    assert_eq!(expect_literal(&expression), &LiteralValue::from(0));
    assert_eq!(expression.to_string(), "0");
}

/// Test an identifier of any sort is read as itself with coefficient 1.
#[test]
fn affine_form_reads_an_identifier_as_coefficient_one() {
    let (x, reference) = build_identifier("x");

    let form = reference.affine_form().expect("an identifier is affine");

    assert_eq!(collect_terms(&form), vec![(x, rat(1, 1))]);
    assert!(form.constant().is_zero());
}

// ---------------------------------------------------------------------------
// What it declines
// ---------------------------------------------------------------------------

/// Test each expression the analysis cannot prove affine is declined.
#[rstest]
#[case::product_of_identifiers(|x: &Expression, y: &Expression| x * y)]
#[case::product_of_non_constant_forms(|x: &Expression, y: &Expression| (x + 1) * (y + 1))]
#[case::floor_division_of_an_identifier(|x: &Expression, _: &Expression| x.floor_divide(2))]
#[case::modulo_of_an_identifier(|x: &Expression, _: &Expression| x.floor_mod(2))]
#[case::square_of_an_identifier(|x: &Expression, _: &Expression| x.power(2))]
#[case::power_with_a_variable_exponent(|x: &Expression, _: &Expression| build_literal(2).power(x))]
#[case::power_with_a_non_integer_exponent(|_: &Expression, _: &Expression| build_literal(2).power(build_decimal_literal("0.5")))]
#[case::zero_to_a_negative_power(|_: &Expression, _: &Expression| build_literal(0).power(-1))]
#[case::division_by_an_identifier(|x: &Expression, y: &Expression| x / y)]
#[case::division_by_a_constant_zero(|x: &Expression, _: &Expression| x / 0)]
#[case::division_by_a_zero_expression(|x: &Expression, _: &Expression| x / (build_literal(1) - 1))]
#[case::float_literal(|x: &Expression, _: &Expression| x + 0.5_f64)]
#[case::bare_float_literal(|_: &Expression, _: &Expression| build_literal(2.0_f64))]
#[case::boolean_literal(|_: &Expression, _: &Expression| build_literal(true))]
#[case::comparison(|x: &Expression, _: &Expression| x.less(1))]
#[case::logical(|_: &Expression, _: &Expression| Expression::all([build_literal(true), build_literal(false)]))]
#[case::piecewise(|x: &Expression, y: &Expression| build_piecewise_or_panic([(build_literal(true), x.clone())], y.clone()))]
#[case::call(|x: &Expression, _: &Expression| build_call_or_panic("f", [x.clone()]))]
#[case::call_of_a_constant(|_: &Expression, _: &Expression| build_call_or_panic("f", [1]))]
#[case::a_declined_operand_deep_inside(|x: &Expression, y: &Expression| build_literal(2) * (x + (y * y)) - 1)]
fn affine_form_declines_what_it_cannot_prove_affine(
    #[case] build: fn(&Expression, &Expression) -> Expression,
) {
    let (x, y) = build_x_and_y();
    let expression = build(&x.1, &y.1);

    let form = expression.affine_form();

    assert_eq!(form, None);
}

// ---------------------------------------------------------------------------
// Its bounds
// ---------------------------------------------------------------------------

/// Test a left-nested sum folds up to 256 levels deep and is declined past
/// that.
#[rstest]
#[case::at_the_bound(256, true)]
#[case::past_the_bound(257, false)]
fn affine_form_reads_a_sum_nested_up_to_256_levels(#[case] depth: usize, #[case] is_affine: bool) {
    let (x, reference) = build_identifier("x");
    let sum = build_deep_sum(&reference, depth);

    let form = sum.affine_form();

    match form {
        Some(form) => {
            assert!(is_affine, "depth {depth} should be declined");
            assert_eq!(collect_terms(&form), vec![(x, rat(1, 1))]);
            assert_eq!(
                form.constant(),
                &rat(i64::try_from(depth).expect("a small depth"), 1)
            );
        }
        None => assert!(!is_affine, "depth {depth} should be affine"),
    }
}

/// Test the depth bound holds for negations too, not only sums.
#[rstest]
#[case::at_the_bound(256, true)]
#[case::past_the_bound(257, false)]
fn affine_form_reads_negations_nested_up_to_256_levels(
    #[case] depth: usize,
    #[case] is_affine: bool,
) {
    let (_, reference) = build_identifier("x");
    let negations = build_deep_negation(&reference, depth);

    let form = negations.affine_form();

    assert_eq!(form.is_some(), is_affine);
}

/// Test a coefficient whose numerator needs more than 4096 bits is
/// declined and one of 4096 bits is read.
#[rstest]
#[case::numerator_of_4096_bits(4095, true)]
#[case::numerator_of_4097_bits(4096, false)]
#[case::numerator_of_4098_bits(4097, false)]
fn affine_form_bounds_a_coefficient_numerator_to_4096_bits(
    #[case] exponent: usize,
    #[case] is_affine: bool,
) {
    let (x, reference) = build_identifier("x");
    let big = BigInt::from(1) << exponent;
    let scaled = &reference * Expression::from(big.clone());

    let form = scaled.affine_form();

    match form {
        Some(form) => {
            assert!(is_affine, "2^{exponent} should be declined");
            let expected = Rational::new(big, BigInt::from(1)).expect("one");
            assert_eq!(form.coefficient(&x), expected);
        }
        None => assert!(!is_affine, "2^{exponent} should be read"),
    }
}

/// Test a coefficient whose denominator needs more than 4096 bits is
/// declined and one of 4096 bits is read.
#[rstest]
#[case::denominator_of_4096_bits(4095, true)]
#[case::denominator_of_4097_bits(4096, false)]
fn affine_form_bounds_a_coefficient_denominator_to_4096_bits(
    #[case] exponent: usize,
    #[case] is_affine: bool,
) {
    let (_, reference) = build_identifier("x");
    let big = BigInt::from(1) << exponent;
    let divided = &reference / Expression::from(big);

    let form = divided.affine_form();

    assert_eq!(form.is_some(), is_affine);
}

/// Test a constant whose numerator needs more than 4096 bits is declined.
#[rstest]
#[case::constant_of_4096_bits(4095, true)]
#[case::constant_of_4097_bits(4096, false)]
fn affine_form_bounds_the_constant_to_4096_bits(#[case] exponent: usize, #[case] is_affine: bool) {
    let (_, reference) = build_identifier("x");
    let big = BigInt::from(1) << exponent;
    let shifted = &reference + Expression::from(big);

    let form = shifted.affine_form();

    assert_eq!(form.is_some(), is_affine);
}

// ---------------------------------------------------------------------------
// The form
// ---------------------------------------------------------------------------

/// Test `terms` iterates in the order of the identifiers' ids, whatever the
/// order they occur in the expression.
#[test]
fn affine_form_terms_iterate_in_identifier_id_order() {
    let (a, a_reference) = build_identifier("a");
    let (b, b_reference) = build_identifier("b");
    let (c, c_reference) = build_identifier("c");
    let expression = &c_reference + &a_reference + &b_reference + &c_reference;

    let form = expression.affine_form().expect("an affine expression");

    assert_eq!(
        collect_terms(&form),
        vec![(a, rat(1, 1)), (b, rat(1, 1)), (c, rat(2, 1))]
    );
    assert_eq!(form.terms().len(), 3);
}

/// Test the coefficient of an identifier the form does not hold is zero.
#[test]
fn affine_form_coefficient_of_an_absent_identifier_is_zero() {
    let (_, x) = build_identifier("x");
    let absent = Identifier::new("absent");

    let form = (&x + 1).affine_form().expect("an affine expression");

    assert_eq!(form.coefficient(&absent), rat(0, 1));
}

/// Test `x + y` and `y + x` give equal forms with equal hashes, and a
/// different form is unequal.
#[test]
fn affine_form_equality_and_hash_ignore_the_operand_order() {
    let ((_, x), (_, y)) = build_x_and_y();

    let forward = (&x + &y).affine_form().expect("affine");
    let backward = (&y + &x).affine_form().expect("affine");
    let shifted = (&x + &y + 1).affine_form().expect("affine");
    let scaled = (&x + build_literal(2) * &y).affine_form().expect("affine");

    assert_eq!(forward, backward);
    assert_eq!(hash_of(&forward), hash_of(&backward));
    assert_ne!(forward, shifted);
    assert_ne!(forward, scaled);
}

/// Test two syntactically different expressions of one linear function give
/// equal forms.
#[test]
fn affine_form_equality_is_by_the_linear_function() {
    let ((_, x), (_, y)) = build_x_and_y();
    let plain = (build_literal(2) * &x + &y).affine_form().expect("affine");

    let rewritten = ((&x + &x + &y + 5) - 5).affine_form().expect("affine");

    assert_eq!(plain, rewritten);
    assert_eq!(hash_of(&plain), hash_of(&rewritten));
}

/// Test a clone of a form is equal to it.
#[test]
fn affine_form_clone_is_equal() {
    let (_, x) = build_identifier("x");
    let form = (&x + 3).affine_form().expect("affine");

    let copy = form.clone();

    assert_eq!(copy, form);
}

// ---------------------------------------------------------------------------
// The canonical expression
// ---------------------------------------------------------------------------

/// Test the canonical expression of a form, as text: terms in id order,
/// `x` for a coefficient of 1 and `-x` for -1, then the constant when it is
/// not zero or there is no term.
#[rstest]
#[case::scaled_terms_and_constant(|x: &Expression, y: &Expression| build_literal(2) * x + build_literal(3) * y + 4, "(((2 * x) + (3 * y)) + 4)")]
#[case::single_scaled_term(|x: &Expression, _: &Expression| build_literal(2) * x, "(2 * x)")]
#[case::unit_coefficient(|x: &Expression, _: &Expression| x + 0, "x")]
#[case::unit_coefficient_and_constant(|x: &Expression, _: &Expression| x + 3, "(x + 3)")]
#[case::minus_one_coefficient(|x: &Expression, _: &Expression| -x, "(-x)")]
#[case::terms_of_both_signs(|x: &Expression, y: &Expression| -x + y, "((-x) + y)")]
#[case::terms_in_id_order_whatever_the_input_order(|x: &Expression, y: &Expression| y + build_literal(2) * x, "((2 * x) + y)")]
#[case::negative_constant(|x: &Expression, _: &Expression| x - 3, "(x + -3)")]
#[case::constant_alone(|_: &Expression, _: &Expression| build_literal(7), "7")]
#[case::zero_alone(|x: &Expression, _: &Expression| x - x, "0")]
#[case::negative_constant_alone(|_: &Expression, _: &Expression| build_literal(-7), "-7")]
fn affine_form_to_expression_writes_the_canonical_expression(
    #[case] build: fn(&Expression, &Expression) -> Expression,
    #[case] expected: &str,
) {
    let (x, y) = build_x_and_y();
    let form = build(&x.1, &y.1)
        .affine_form()
        .expect("an affine expression");

    let canonical = form.to_expression();

    assert_eq!(canonical.to_string(), expected);
}

/// Test a non-integer coefficient is written as the quotient of two integer
/// literals multiplying its identifier, then the constant.
#[test]
fn affine_form_to_expression_writes_a_fraction_as_a_quotient_of_literals() {
    let (x, reference) = build_identifier("x");
    let form = (&reference / 2 + 3)
        .affine_form()
        .expect("an affine expression");

    let canonical = form.to_expression();

    let sum = expect_binary(&canonical);
    assert_eq!(sum.operation(), BinaryOperation::Add);
    assert_eq!(expect_literal(sum.right()), &LiteralValue::from(3));
    let term = expect_binary(sum.left());
    assert_eq!(term.operation(), BinaryOperation::Multiply);
    let quotient = expect_binary(term.left());
    assert_eq!(quotient.operation(), BinaryOperation::Divide);
    assert_eq!(expect_literal(quotient.left()), &LiteralValue::from(1));
    assert_eq!(expect_literal(quotient.right()), &LiteralValue::from(2));
    assert_eq!(term.right(), &Expression::from(x));
}

/// Test the canonical expression of `1/2 * x - y + 3` is the sum of its
/// two terms in id order and its constant.
#[test]
fn affine_form_to_expression_writes_the_documented_canonical_text() {
    let ((_, x), (_, y)) = build_x_and_y();
    let form = (&x / 2 - &y + 3)
        .affine_form()
        .expect("an affine expression");

    let canonical = form.to_expression();

    assert_eq!(canonical.to_string(), "((((1 / 2) * x) + (-y)) + 3)");
    let sum = expect_binary(&canonical);
    assert_eq!(sum.operation(), BinaryOperation::Add);
    let minus_y = expect_unary(expect_binary(sum.left()).right());
    assert_eq!(minus_y.operation(), UnaryOperation::Negate);
}

/// Test the form of a form's canonical expression is the form itself.
#[rstest]
#[case::integer_coefficients(|x: &Expression, y: &Expression| build_literal(2) * x - build_literal(3) * y + 4)]
#[case::fractions(|x: &Expression, y: &Expression| x / 3 - y / 7 + build_decimal_literal("0.125"))]
#[case::negative_fraction(|x: &Expression, _: &Expression| -x / 2)]
#[case::constant_only(|_: &Expression, _: &Expression| build_literal(9))]
#[case::zero(|x: &Expression, _: &Expression| x - x)]
fn affine_form_of_the_canonical_expression_is_the_form(
    #[case] build: fn(&Expression, &Expression) -> Expression,
) {
    let (x, y) = build_x_and_y();
    let form = build(&x.1, &y.1)
        .affine_form()
        .expect("an affine expression");

    let again = form.to_expression().affine_form();

    assert_eq!(again, Some(form));
}

/// Test a form displays as the text of its canonical expression.
#[rstest]
#[case::terms_and_constant(|x: &Expression, y: &Expression| build_literal(2) * x + build_literal(3) * y + 4, "(((2 * x) + (3 * y)) + 4)")]
#[case::zero(|x: &Expression, _: &Expression| x - x, "0")]
#[case::unit(|x: &Expression, _: &Expression| x + 0, "x")]
fn affine_form_displays_its_canonical_expression(
    #[case] build: fn(&Expression, &Expression) -> Expression,
    #[case] expected: &str,
) {
    let (x, y) = build_x_and_y();
    let form = build(&x.1, &y.1)
        .affine_form()
        .expect("an affine expression");

    let text = form.to_string();

    assert_eq!(text, expected);
    assert_eq!(text, form.to_expression().to_string());
}
