//! Stories for the strategies of the ground simplifier, each alone through
//! its `rewrite` and in a simplifier holding only it, and for the driver
//! that applies them: its order, its fixed point, its bound, and how a
//! caller adds, removes and reorders strategies.
//!
//! The binding's differential tests (`solver::sympy::ground_differential`)
//! check each strategy's rewrites against `SymPy`.

use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use fhy_core::expression::builtins::{BuiltinConstant, BuiltinFunction};
use fhy_core::expression::registry::{FunctionRegistry, NativeConstant};
use fhy_core::expression::{
    BigInt, Callee, Expression, ExpressionKind, FunctionName, FunctionSort, LiteralValue,
};
use fhy_core::solver::strategy::{
    Comparisons, ComposedBuiltins, ExactArithmetic, ExactBuiltins, LogicalOperators,
    NormalizeLiterals, PiecewiseDecision, RegisteredConstants, SimplificationStrategy,
    default_strategies,
};
use fhy_core::solver::{GroundSimplifier, SimplifyContext};
use rstest::rstest;

use crate::support::expression::{build_identifier, build_literal};

fn n(value: i64) -> Expression {
    Expression::from(value)
}

fn decimal(text: &str) -> Expression {
    build_literal(LiteralValue::Decimal(text.parse().expect("a decimal")))
}

fn truth(value: bool) -> Expression {
    build_literal(value)
}

fn rewrite(strategy: &dyn SimplificationStrategy, node: &Expression) -> Option<Expression> {
    strategy.rewrite(node, &SimplifyContext::default())
}

/// Return a simplifier holding only `strategy`.
fn alone(strategy: impl SimplificationStrategy + 'static) -> GroundSimplifier {
    GroundSimplifier::empty().with_strategy(strategy)
}

fn run(simplifier: &GroundSimplifier, expression: &Expression) -> Option<Expression> {
    simplifier.try_simplify(expression, &SimplifyContext::default())
}

// ---------------------------------------------------------------------------
// NormalizeLiterals
// ---------------------------------------------------------------------------

#[rstest]
#[case::no_float_equals(decimal("0.1"), Some(n(1) / n(10)))]
#[case::integer_decimal(decimal("2.0"), Some(n(2)))]
#[case::float_equals(decimal("0.5"), None)]
#[case::integer(n(2), None)]
#[case::boolean(truth(true), None)]
#[case::float(build_literal(0.5), None)]
#[case::not_a_literal(n(1) + n(2), None)]
fn normalize_literals_writes_a_decimal_as_sympy_lifts_it(
    #[case] node: Expression,
    #[case] expected: Option<Expression>,
) {
    assert_eq!(rewrite(&NormalizeLiterals::new(), &node), expected);
}

#[test]
fn normalize_literals_alone_decides_a_decimal_and_nothing_else() {
    let simplifier = alone(NormalizeLiterals::new());

    assert_eq!(run(&simplifier, &decimal("0.1")), Some(n(1) / n(10)));
    assert_eq!(run(&simplifier, &(n(1) + n(2))), None);
}

// ---------------------------------------------------------------------------
// RegisteredConstants
// ---------------------------------------------------------------------------

fn registry_with(
    name: &str,
    value: impl Into<LiteralValue>,
    sort: FunctionSort,
) -> (FunctionRegistry, Expression) {
    let mut registry = FunctionRegistry::new();
    let identifier = registry
        .register_constant(
            NativeConstant::new(FunctionName::new(name).expect("a name"), sort, value)
                .expect("a constant"),
        )
        .expect("registered");
    (registry, Expression::from(identifier))
}

#[test]
fn registered_constants_rewrites_a_reference_to_the_value() {
    let (registry, answer) = registry_with("answer", 42, FunctionSort::Int);

    let rewritten =
        RegisteredConstants::new().rewrite(&answer, &SimplifyContext::from_registry(&registry));

    assert_eq!(rewritten, Some(n(42)));
}

#[test]
fn registered_constants_declines_without_a_registry() {
    let (_, answer) = registry_with("answer", 42, FunctionSort::Int);

    assert_eq!(rewrite(&RegisteredConstants::new(), &answer), None);
}

#[test]
fn registered_constants_declines_a_builtin_constant_and_a_free_identifier() {
    let registry = FunctionRegistry::new();
    let context = SimplifyContext::from_registry(&registry);
    let pi = Expression::from(BuiltinConstant::Pi.identifier().clone());
    let (_, free) = build_identifier("x");

    assert_eq!(RegisteredConstants::new().rewrite(&pi, &context), None);
    assert_eq!(RegisteredConstants::new().rewrite(&free, &context), None);
}

#[test]
fn registered_constants_alone_does_not_decide_the_expression_around_it() {
    let (registry, answer) = registry_with("answer", 42, FunctionSort::Int);
    let simplifier = alone(RegisteredConstants::new());

    let result = simplifier.try_simplify(&answer, &SimplifyContext::from_registry(&registry));

    assert_eq!(result, Some(n(42)));
    assert_eq!(
        simplifier.try_simplify(&(answer + n(1)), &SimplifyContext::from_registry(&registry)),
        None
    );
}

// ---------------------------------------------------------------------------
// ExactArithmetic
// ---------------------------------------------------------------------------

#[rstest]
#[case::sum(n(2) + n(3), Some(n(5)))]
#[case::difference(n(2) - n(5), Some(n(-3)))]
#[case::quotient(n(1) / n(2), Some(decimal("0.5")))]
#[case::quotient_reduces(n(2) / n(6), Some(quotient(1, 3)))]
#[case::already_normal(quotient(1, 3), None)]
#[case::floor_division(n(-7).floor_divide(n(2)), Some(n(-4)))]
#[case::modulo(n(7).floor_mod(n(-3)), Some(n(-2)))]
#[case::power(n(2).power(n(-2)), Some(decimal("0.25")))]
#[case::negation(-n(7), Some(n(-7)))]
#[case::positive(n(7).positive(), Some(n(7)))]
#[case::decimal_operand(decimal("0.5") + decimal("0.25"), Some(decimal("0.75")))]
#[case::quotient_operand(quotient(1, 3) + quotient(1, 6), Some(decimal("0.5")))]
#[case::division_by_zero(n(1) / n(0), None)]
#[case::irrational_root(n(2).power(n(1) / n(2)), None)]
#[case::free_operand(build_identifier("x").1 + n(1), None)]
#[case::undecided_operand((n(1) + n(1)) + n(1), None)]
#[case::float_operand(build_literal(1.5) + n(1), None)]
#[case::boolean_operand(truth(true) + n(1), None)]
#[case::comparison(n(1).less(n(2)), None)]
fn exact_arithmetic_rewrites_a_node_over_exact_numbers(
    #[case] node: Expression,
    #[case] expected: Option<Expression>,
) {
    assert_eq!(rewrite(&ExactArithmetic::new(), &node), expected);
}

fn quotient(numerator: i64, denominator: i64) -> Expression {
    Expression::new_binary(
        fhy_core::expression::BinaryOperation::Divide,
        n(numerator),
        n(denominator),
    )
}

#[test]
fn exact_arithmetic_alone_folds_a_nested_tree_but_not_a_comparison() {
    let simplifier = alone(ExactArithmetic::new());

    assert_eq!(run(&simplifier, &(n(2) * (n(3) + n(4)))), Some(n(14)));
    assert_eq!(run(&simplifier, &(n(2) + n(3)).less(n(9))), None);
}

fn power_of_two(exponent: u32) -> BigInt {
    BigInt::from(1) << exponent
}

fn big(value: BigInt) -> Expression {
    Expression::literal(value)
}

#[rstest]
#[case::floor_division_by_minus_one(n(i64::MIN).floor_divide(n(-1)), Some(big(power_of_two(63))))]
#[case::modulo_by_minus_one(n(i64::MIN).floor_mod(n(-1)), Some(n(0)))]
#[case::negating_the_least_integer(-n(i64::MIN), Some(big(power_of_two(63))))]
#[case::the_least_integer_squared(n(i64::MIN) * n(i64::MIN), Some(big(power_of_two(126))))]
#[case::the_greatest_integer_summed(n(i64::MAX) + n(i64::MAX), Some(big(power_of_two(64) - 2)))]
#[case::the_least_integer_subtracted(n(i64::MIN) - n(1), Some(big(-power_of_two(63) - 1)))]
fn exact_arithmetic_folds_the_extremes_of_the_small_integers(
    #[case] node: Expression,
    #[case] expected: Option<Expression>,
) {
    assert_eq!(rewrite(&ExactArithmetic::new(), &node), expected);
}

#[test]
fn exact_arithmetic_folds_a_product_past_the_small_integers() {
    // (2^63 - 1)^2 + 1 = 2^126 - 2^64 + 2.
    let expression = n(i64::MAX) * n(i64::MAX) + n(1);

    assert_eq!(
        run(&alone(ExactArithmetic::new()), &expression),
        Some(big(power_of_two(126) - power_of_two(64) + 2))
    );
}

#[test]
fn exact_arithmetic_folds_a_product_up_to_a_million_bits_and_declines_past_it() {
    // 2^(2^19) has 2^19 + 1 bits, and 2^(2^19 - 1) has 2^19.
    let fits = big(power_of_two(1 << 19)) * big(power_of_two((1 << 19) - 1));
    let past = big(power_of_two(1 << 19)) * big(power_of_two(1 << 19));

    assert_eq!(
        rewrite(&ExactArithmetic::new(), &fits),
        Some(big(power_of_two((1 << 20) - 1)))
    );
    assert_eq!(rewrite(&ExactArithmetic::new(), &past), None);
}

#[test]
fn exact_arithmetic_declines_a_product_of_two_huge_literals_and_folds_a_zero_one() {
    let huge = || big(power_of_two(4_000_000));

    assert_eq!(rewrite(&ExactArithmetic::new(), &(huge() * huge())), None);
    assert_eq!(
        rewrite(&ExactArithmetic::new(), &(huge() * n(0))),
        Some(n(0))
    );
    assert_eq!(
        rewrite(&ExactArithmetic::new(), &(n(0) * huge())),
        Some(n(0))
    );
}

#[test]
fn exact_arithmetic_declines_a_huge_literal_with_a_fraction_unless_the_result_is_an_integer() {
    let huge = || big(power_of_two(5000));

    assert_eq!(
        rewrite(&ExactArithmetic::new(), &(huge() + quotient(1, 3))),
        None
    );
    assert_eq!(
        rewrite(&ExactArithmetic::new(), &(quotient(1, 3) - huge())),
        None
    );
    // A quotient that is an integer folds, and one that is not declines.
    assert_eq!(
        rewrite(&ExactArithmetic::new(), &(huge() / quotient(1, 3))),
        Some(big(BigInt::from(3) * power_of_two(5000)))
    );
    assert_eq!(
        rewrite(&ExactArithmetic::new(), &(huge() / quotient(3, 7))),
        None
    );
}

#[test]
fn exact_arithmetic_folds_a_floor_division_and_a_modulo_of_a_big_integer() {
    // 2^5000 = 3k + 1, as 2^2 = 4 = 3 + 1 and the exponent is even.
    let k: BigInt = (power_of_two(5000) - 1) / 3;
    let two_to_5000 = || big(power_of_two(5000));
    let minus_two_to_5000 = || big(-power_of_two(5000));
    let strategy = ExactArithmetic::new();

    assert_eq!(
        rewrite(&strategy, &two_to_5000().floor_divide(n(3))),
        Some(big(k.clone()))
    );
    assert_eq!(
        rewrite(&strategy, &two_to_5000().floor_mod(n(3))),
        Some(n(1))
    );
    // Floor division rounds toward negative infinity, and the modulo takes
    // the divisor's sign.
    assert_eq!(
        rewrite(&strategy, &minus_two_to_5000().floor_divide(n(3))),
        Some(big(-k.clone() - 1))
    );
    assert_eq!(
        rewrite(&strategy, &minus_two_to_5000().floor_mod(n(3))),
        Some(n(2))
    );
    assert_eq!(
        rewrite(&strategy, &two_to_5000().floor_divide(n(-3))),
        Some(big(-k.clone() - 1))
    );
    assert_eq!(
        rewrite(&strategy, &two_to_5000().floor_mod(n(-3))),
        Some(n(-2))
    );
    assert_eq!(
        rewrite(&strategy, &minus_two_to_5000().floor_divide(n(-3))),
        Some(big(k))
    );
    assert_eq!(
        rewrite(&strategy, &minus_two_to_5000().floor_mod(n(-3))),
        Some(n(-1))
    );
    // An exact quotient has no remainder.
    assert_eq!(
        rewrite(
            &strategy,
            &two_to_5000().floor_divide(big(power_of_two(4000)))
        ),
        Some(big(power_of_two(1000)))
    );
    assert_eq!(
        rewrite(&strategy, &two_to_5000().floor_mod(big(power_of_two(4000)))),
        Some(n(0))
    );
}

#[test]
fn exact_arithmetic_takes_a_root_up_to_an_index_of_4096_and_declines_past_it() {
    let strategy = ExactArithmetic::new();

    // (2^4096)^(1/4096) = 2, and (2^4097)^(1/4097) would be, past the bound.
    assert_eq!(
        rewrite(&strategy, &big(power_of_two(4096)).power(quotient(1, 4096))),
        Some(n(2))
    );
    assert_eq!(
        rewrite(&strategy, &big(power_of_two(4097)).power(quotient(1, 4097))),
        None
    );
}

#[test]
fn exact_arithmetic_declines_a_big_floor_division_by_zero() {
    let strategy = ExactArithmetic::new();

    assert_eq!(
        rewrite(&strategy, &big(power_of_two(5000)).floor_divide(n(0))),
        None
    );
    assert_eq!(
        rewrite(&strategy, &big(power_of_two(5000)).floor_mod(n(0))),
        None
    );
    assert_eq!(
        rewrite(&strategy, &n(5).floor_divide(big(BigInt::from(0)))),
        None
    );
}

#[test]
fn exact_arithmetic_declines_a_floor_division_only_when_divisor_and_quotient_are_both_large() {
    let strategy = ExactArithmetic::new();
    let dividend = |exponent| big(power_of_two(exponent));

    // A divisor of 4001 bits, or a quotient of 4000, folds.
    assert_eq!(
        rewrite(&strategy, &dividend(10_000).floor_divide(dividend(4000))),
        Some(big(power_of_two(6000)))
    );
    assert_eq!(
        rewrite(&strategy, &dividend(9000).floor_divide(dividend(5000))),
        Some(big(power_of_two(4000)))
    );
    // Both past 4096 bits decline, for the modulo too.
    assert_eq!(
        rewrite(&strategy, &dividend(10_000).floor_divide(dividend(5000))),
        None
    );
    assert_eq!(
        rewrite(&strategy, &dividend(10_000).floor_mod(dividend(5000))),
        None
    );
}

// ---------------------------------------------------------------------------
// Comparisons
// ---------------------------------------------------------------------------

#[rstest]
#[case::less(n(1).less(n(2)), Some(true))]
#[case::less_equal(n(2).less_equal(n(2)), Some(true))]
#[case::greater_fails(n(1).greater(n(2)), Some(false))]
#[case::greater_equal(n(2).greater_equal(n(2)), Some(true))]
#[case::equal(n(2).equals(n(3)), Some(false))]
#[case::not_equal(n(2).not_equals(n(3)), Some(true))]
#[case::rationals(quotient(1, 3).less(decimal("0.5")), Some(true))]
#[case::decimal_equals_integer(decimal("1.0").equals(n(1)), Some(true))]
#[case::booleans_equal(truth(true).equals(truth(true)), Some(true))]
#[case::booleans_not_equal(truth(true).not_equals(truth(true)), Some(false))]
#[case::ordering_booleans(truth(true).less(truth(false)), None)]
#[case::boolean_and_number(truth(true).equals(n(1)), None)]
#[case::float(build_literal(1.5).less(n(2)), None)]
#[case::free(build_identifier("x").1.less(n(2)), None)]
#[case::undecided_operand((n(1) + n(1)).less(n(3)), None)]
#[case::arithmetic(n(1) + n(1), None)]
fn comparisons_decide_a_comparison_of_exact_numbers(
    #[case] node: Expression,
    #[case] expected: Option<bool>,
) {
    assert_eq!(rewrite(&Comparisons::new(), &node), expected.map(truth));
}

#[test]
fn comparisons_alone_decide_a_comparison_only_once_its_operands_are_numbers() {
    let simplifier = alone(Comparisons::new());

    assert_eq!(run(&simplifier, &n(1).less(n(2))), Some(truth(true)));
    assert_eq!(run(&simplifier, &(n(0) + n(1)).less(n(2))), None);
}

// ---------------------------------------------------------------------------
// LogicalOperators
// ---------------------------------------------------------------------------

#[rstest]
#[case::negation(!truth(false), Some(true))]
#[case::conjunction(truth(true).and(truth(false)), Some(false))]
#[case::disjunction(truth(false).or(truth(true)), Some(true))]
#[case::many_operands(Expression::all([truth(true), truth(true), truth(true)]), Some(true))]
#[case::short_circuit_is_not_taken(truth(false).and(build_identifier("x").1.less(n(1))), None)]
#[case::number_operand(Expression::all([n(1), truth(true)]), None)]
#[case::undecided_operand((!(!truth(true))).and(truth(true)), None)]
#[case::a_boolean_builtin(call(BuiltinFunction::Xor, [truth(true), truth(false)]), None)]
#[case::user_function(Expression::call("f".parse::<Callee>().expect("a name"), [truth(true), truth(true)]), None)]
fn logical_operators_decide_an_operator_of_boolean_literals(
    #[case] node: Expression,
    #[case] expected: Option<bool>,
) {
    assert_eq!(
        rewrite(&LogicalOperators::new(), &node),
        expected.map(truth)
    );
}

fn call(function: BuiltinFunction, arguments: impl IntoIterator<Item = Expression>) -> Expression {
    Expression::call(function, arguments)
}

// ---------------------------------------------------------------------------
// PiecewiseDecision
// ---------------------------------------------------------------------------

fn piecewise(cases: Vec<(Expression, Expression)>, otherwise: Expression) -> Expression {
    Expression::piecewise(cases, otherwise).expect("a piecewise")
}

#[rstest]
#[case::first_true_case(piecewise(vec![(truth(true), n(1)), (truth(true), n(2))], n(3)), Some(n(1)))]
#[case::later_true_case(piecewise(vec![(truth(false), n(1)), (truth(true), n(2))], n(3)), Some(n(2)))]
#[case::otherwise(piecewise(vec![(truth(false), n(1))], decimal("0.5")), Some(decimal("0.5")))]
#[case::boolean_value(piecewise(vec![(truth(true), truth(false))], truth(true)), Some(truth(false)))]
#[case::undecided_condition(piecewise(vec![(build_identifier("x").1.less(n(1)), n(1))], n(3)), None)]
#[case::undecided_value(piecewise(vec![(truth(true), n(1))], build_identifier("x").1), None)]
#[case::undecided_value_in_a_case(piecewise(vec![(truth(true), n(1)), (truth(false), n(1) + n(1))], n(3)), None)]
#[case::unnormalized_value(piecewise(vec![(truth(true), decimal("0.1"))], n(3)), None)]
fn piecewise_decision_chooses_among_decided_branches(
    #[case] node: Expression,
    #[case] expected: Option<Expression>,
) {
    assert_eq!(rewrite(&PiecewiseDecision::new(), &node), expected);
}

// ---------------------------------------------------------------------------
// ExactBuiltins
// ---------------------------------------------------------------------------

#[rstest]
#[case::floor(call(BuiltinFunction::Floor, [quotient(7, 2)]), Some(n(3)))]
#[case::ceil(call(BuiltinFunction::Ceil, [quotient(-7, 2)]), Some(n(-3)))]
#[case::round_of_an_integer(call(BuiltinFunction::Round, [n(5)]), Some(n(5)))]
#[case::round_of_a_rational(call(BuiltinFunction::Round, [quotient(1, 3)]), None)]
#[case::sqrt(call(BuiltinFunction::Sqrt, [n(16)]), Some(n(4)))]
#[case::sqrt_of_a_non_square(call(BuiltinFunction::Sqrt, [n(2)]), None)]
#[case::sqrt_of_a_negative(call(BuiltinFunction::Sqrt, [n(-4)]), None)]
#[case::exp2(call(BuiltinFunction::Exp2, [n(-3)]), Some(decimal("0.125")))]
#[case::log2(call(BuiltinFunction::Log2, [n(8)]), Some(n(3)))]
#[case::log10(call(BuiltinFunction::Log10, [quotient(1, 100)]), Some(n(-2)))]
#[case::log2_of_a_non_power(call(BuiltinFunction::Log2, [n(6)]), None)]
#[case::exp_of_zero(call(BuiltinFunction::Exp, [n(0)]), Some(n(1)))]
#[case::exp_of_one(call(BuiltinFunction::Exp, [n(1)]), None)]
#[case::sin_of_zero(call(BuiltinFunction::Sin, [n(0)]), Some(n(0)))]
#[case::cos_of_zero(call(BuiltinFunction::Cos, [n(0)]), Some(n(1)))]
#[case::tan_of_zero(call(BuiltinFunction::Tan, [n(0)]), Some(n(0)))]
#[case::arcsin_of_zero(call(BuiltinFunction::Arcsin, [n(0)]), Some(n(0)))]
#[case::arctan_of_zero(call(BuiltinFunction::Arctan, [n(0)]), Some(n(0)))]
#[case::sinh_of_zero(call(BuiltinFunction::Sinh, [n(0)]), Some(n(0)))]
#[case::cosh_of_zero(call(BuiltinFunction::Cosh, [n(0)]), Some(n(1)))]
#[case::tanh_of_zero(call(BuiltinFunction::Tanh, [n(0)]), Some(n(0)))]
#[case::erf_of_zero(call(BuiltinFunction::Erf, [n(0)]), Some(n(0)))]
#[case::sin_of_one(call(BuiltinFunction::Sin, [n(1)]), None)]
#[case::log_of_one(call(BuiltinFunction::Log, [n(1)]), Some(n(0)))]
#[case::log_of_two(call(BuiltinFunction::Log, [n(2)]), None)]
#[case::acos_of_one(call(BuiltinFunction::Arccos, [n(1)]), Some(n(0)))]
#[case::acos_of_zero(call(BuiltinFunction::Arccos, [n(0)]), None)]
#[case::sqrt_of_zero(call(BuiltinFunction::Sqrt, [n(0)]), Some(n(0)))]
#[case::sqrt_of_a_fraction(call(BuiltinFunction::Sqrt, [quotient(1, 4)]), Some(decimal("0.5")))]
#[case::sigmoid(call(BuiltinFunction::Sigmoid, [n(0)]), None)]
#[case::composed_builtin(call(BuiltinFunction::Max, [n(2), n(5)]), None)]
#[case::boolean_builtin(call(BuiltinFunction::Xor, [truth(true), truth(false)]), None)]
#[case::free_argument(call(BuiltinFunction::Floor, [build_identifier("x").1]), None)]
#[case::undecided_argument(call(BuiltinFunction::Floor, [n(1) + n(1)]), None)]
#[case::user_function(Expression::call("f".parse::<Callee>().expect("a name"), [n(1)]), None)]
fn exact_builtins_rewrite_a_call_with_an_exact_value(
    #[case] node: Expression,
    #[case] expected: Option<Expression>,
) {
    assert_eq!(rewrite(&ExactBuiltins::new(), &node), expected);
}

#[rstest]
#[case::log2_inside_the_bound(call(BuiltinFunction::Log2, [big(power_of_two(4095))]), Some(n(4095)))]
#[case::log2_past_the_bound(call(BuiltinFunction::Log2, [big(power_of_two(4096))]), None)]
#[case::log2_far_past_the_bound(call(BuiltinFunction::Log2, [big(power_of_two(5000))]), None)]
#[case::log10_inside_the_bound(call(BuiltinFunction::Log10, [big(BigInt::from(10).pow(1200))]), Some(n(1200)))]
#[case::log10_past_the_bound(call(BuiltinFunction::Log10, [big(BigInt::from(10).pow(1300))]), None)]
fn exact_builtins_decline_an_argument_past_their_size_bounds(
    #[case] node: Expression,
    #[case] expected: Option<Expression>,
) {
    assert_eq!(rewrite(&ExactBuiltins::new(), &node), expected);
}

// ---------------------------------------------------------------------------
// ComposedBuiltins
// ---------------------------------------------------------------------------

#[rstest]
#[case::max(call(BuiltinFunction::Max, [n(2), n(5)]), Some(n(5)))]
#[case::min(call(BuiltinFunction::Min, [n(2), n(5)]), Some(n(2)))]
#[case::abs(call(BuiltinFunction::Abs, [n(-5)]), Some(n(5)))]
#[case::abs_of_a_positive(call(BuiltinFunction::Abs, [n(5)]), Some(n(5)))]
#[case::abs_of_zero(call(BuiltinFunction::Abs, [n(0)]), Some(n(0)))]
#[case::sign(call(BuiltinFunction::Sign, [n(-5)]), Some(n(-1)))]
#[case::sign_of_a_positive(call(BuiltinFunction::Sign, [n(5)]), Some(n(1)))]
#[case::sign_of_zero(call(BuiltinFunction::Sign, [n(0)]), Some(n(0)))]
#[case::clamp(call(BuiltinFunction::Clamp, [n(15), n(0), n(10)]), Some(n(10)))]
#[case::clamp_inside(call(BuiltinFunction::Clamp, [n(5), n(0), n(10)]), Some(n(5)))]
#[case::clamp_below(call(BuiltinFunction::Clamp, [n(-5), n(0), n(10)]), Some(n(0)))]
#[case::clamp_symmetric(call(BuiltinFunction::ClampSymmetric, [n(-15), n(10)]), Some(n(-10)))]
#[case::relu(call(BuiltinFunction::Relu, [n(-3)]), Some(n(0)))]
#[case::relu_of_a_positive(call(BuiltinFunction::Relu, [n(3)]), Some(n(3)))]
#[case::relu_of_zero(call(BuiltinFunction::Relu, [n(0)]), Some(n(0)))]
#[case::leaky_relu(call(BuiltinFunction::LeakyRelu, [n(-4), quotient(1, 2)]), Some(n(-2)))]
#[case::leaky_relu_of_a_positive(call(BuiltinFunction::LeakyRelu, [n(3), quotient(1, 2)]), Some(n(3)))]
#[case::leaky_relu_of_zero(call(BuiltinFunction::LeakyRelu, [n(0), quotient(1, 2)]), Some(n(0)))]
#[case::max_of_a_free_argument(call(BuiltinFunction::Max, [build_identifier("x").1, n(1)]), None)]
#[case::max_of_an_undecided_argument(call(BuiltinFunction::Max, [n(1) + n(1), n(1)]), None)]
#[case::xor_of_numbers(call(BuiltinFunction::Xor, [n(1), n(2)]), None)]
#[case::xor_of_an_undecided_argument(call(BuiltinFunction::Xor, [truth(true), !(!truth(true))]), None)]
#[case::an_exact_builtin(call(BuiltinFunction::Floor, [n(2)]), None)]
#[case::user_function(Expression::call("f".parse::<Callee>().expect("a name"), [n(1)]), None)]
fn composed_builtins_rewrite_a_call_by_its_definition(
    #[case] node: Expression,
    #[case] expected: Option<Expression>,
) {
    assert_eq!(rewrite(&ComposedBuiltins::new(), &node), expected);
}

/// The truth table of each Boolean built-in, over `(false, false)`,
/// `(false, true)`, `(true, false)` and `(true, true)`.
#[rstest]
#[case::xor(BuiltinFunction::Xor, [false, true, true, false])]
#[case::nand(BuiltinFunction::Nand, [true, true, true, false])]
#[case::nor(BuiltinFunction::Nor, [true, false, false, false])]
#[case::implies(BuiltinFunction::Implies, [true, true, false, true])]
#[case::iff(BuiltinFunction::Iff, [true, false, false, true])]
fn composed_builtins_follow_the_truth_table_of_a_boolean_function(
    #[case] function: BuiltinFunction,
    #[case] table: [bool; 4],
) {
    let inputs = [(false, false), (false, true), (true, false), (true, true)];

    for ((left, right), expected) in inputs.into_iter().zip(table) {
        assert_eq!(
            rewrite(
                &ComposedBuiltins::new(),
                &call(function, [truth(left), truth(right)])
            ),
            Some(truth(expected)),
            "{function:?}({left}, {right})",
        );
    }
}

#[test]
fn composed_builtins_are_not_a_default_strategy() {
    let names: Vec<String> = default_strategies()
        .iter()
        .map(|strategy| strategy.name().into_owned())
        .collect();

    assert!(!names.contains(&ComposedBuiltins::new().name().into_owned()));
    assert_eq!(
        run(
            &GroundSimplifier::new(),
            &call(BuiltinFunction::Max, [n(2), n(5)])
        ),
        None
    );
    assert_eq!(
        run(
            &GroundSimplifier::new().with_strategy(ComposedBuiltins::new()),
            &call(BuiltinFunction::Max, [n(2), n(5)])
        ),
        Some(n(5)),
    );
}

// ---------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------

#[test]
fn the_default_strategies_are_named_and_ordered() {
    let names: Vec<String> = default_strategies()
        .iter()
        .map(|strategy| strategy.name().into_owned())
        .collect();

    assert_eq!(
        names,
        [
            "registered_constants",
            "normalize_literals",
            "exact_arithmetic",
            "comparisons",
            "logical_operators",
            "piecewise",
            "exact_builtins",
        ]
    );
    assert_eq!(GroundSimplifier::new().strategy_names(), names);
}

#[test]
fn a_simplifier_without_strategies_rewrites_nothing() {
    let simplifier = GroundSimplifier::empty();

    assert_eq!(simplifier.strategy_names(), Vec::<String>::new());
    assert_eq!(run(&simplifier, &(n(1) + n(2))), None);
    // An expression that is already decided is returned as it is.
    assert_eq!(run(&simplifier, &n(3)), Some(n(3)));
}

#[test]
fn without_removes_a_strategy_by_name() {
    let simplifier = GroundSimplifier::new().without("comparisons");
    let comparison = (n(1) + n(1)).less(n(3));

    assert!(
        !simplifier
            .strategy_names()
            .contains(&"comparisons".to_owned())
    );
    assert_eq!(run(&simplifier, &comparison), None);
    assert_eq!(
        run(&GroundSimplifier::new(), &comparison),
        Some(truth(true))
    );
    assert_eq!(run(&simplifier, &(n(1) + n(1))), Some(n(2)));
}

#[test]
fn with_default_strategies_appends_them() {
    let simplifier = GroundSimplifier::empty()
        .with_strategy(NormalizeLiterals::new())
        .with_default_strategies();

    assert_eq!(
        simplifier.strategy_names().len(),
        1 + default_strategies().len()
    );
    assert_eq!(simplifier.strategy_names()[0], "normalize_literals");
}

/// A strategy rewriting every binary node to the literal `99`, to see which
/// strategy wins.
#[derive(Debug)]
struct Ninety9;

impl SimplificationStrategy for Ninety9 {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("ninety_nine")
    }

    fn rewrite(&self, node: &Expression, _context: &SimplifyContext<'_>) -> Option<Expression> {
        matches!(node.kind(), ExpressionKind::Binary(_)).then(|| n(99))
    }
}

#[test]
fn the_first_strategy_that_rewrites_a_node_wins() {
    let after = GroundSimplifier::new().with_strategy(Ninety9);
    let first = GroundSimplifier::new().with_strategy_first(Ninety9);

    assert_eq!(run(&after, &(n(1) + n(2))), Some(n(3)));
    assert_eq!(run(&first, &(n(1) + n(2))), Some(n(99)));
    assert_eq!(first.strategy_names()[0], "ninety_nine");
}

/// A strategy renaming the calls of the user function `from` to calls of
/// `to`.
#[derive(Debug)]
struct Renaming {
    from: &'static str,
    to: &'static str,
}

impl SimplificationStrategy for Renaming {
    fn name(&self) -> Cow<'_, str> {
        Cow::Owned(format!("rename_{}_to_{}", self.from, self.to))
    }

    fn rewrite(&self, node: &Expression, _context: &SimplifyContext<'_>) -> Option<Expression> {
        let ExpressionKind::Call(call) = node.kind() else {
            return None;
        };
        let Callee::Named(name) = call.callee() else {
            return None;
        };
        (name.as_str() == self.from).then(|| {
            Expression::call(
                self.to.parse::<Callee>().expect("a name"),
                call.arguments().iter().cloned(),
            )
        })
    }
}

fn named_call(name: &str, argument: Expression) -> Expression {
    Expression::call(name.parse::<Callee>().expect("a name"), [argument])
}

#[test]
fn the_driver_repeats_until_no_strategy_rewrites() {
    // `f(1)` becomes `g(1)`, and then `h(1)`: the chain of rewrites is
    // followed to its end.
    let simplifier = GroundSimplifier::empty()
        .with_strategy(Renaming { from: "f", to: "g" })
        .with_strategy(Renaming { from: "g", to: "h" })
        .with_partial_rewrites();

    let result = run(&simplifier, &named_call("f", n(1)));

    assert_eq!(result, Some(named_call("h", n(1))));
}

#[test]
fn the_driver_simplifies_the_children_of_a_rewritten_node() {
    // The rewrite builds a call whose argument is not yet simplified.
    #[derive(Debug)]
    struct Wrapping;

    impl SimplificationStrategy for Wrapping {
        fn name(&self) -> Cow<'_, str> {
            Cow::Borrowed("wrapping")
        }

        fn rewrite(&self, node: &Expression, _context: &SimplifyContext<'_>) -> Option<Expression> {
            let ExpressionKind::Call(call) = node.kind() else {
                return None;
            };
            let Callee::Named(name) = call.callee() else {
                return None;
            };
            (name.as_str() == "wrap").then(|| call.arguments()[0].clone() + n(1) + n(1))
        }
    }
    let simplifier = GroundSimplifier::new().with_strategy(Wrapping);

    assert_eq!(run(&simplifier, &named_call("wrap", n(5))), Some(n(7)));
}

#[test]
fn strategies_that_undo_each_other_stop_at_the_bound() {
    let flipping = || {
        GroundSimplifier::empty()
            .with_strategy(Renaming { from: "f", to: "g" })
            .with_strategy(Renaming { from: "g", to: "f" })
    };
    let bounded = flipping().with_max_rewrites(5).with_partial_rewrites();

    let result = run(&bounded, &named_call("f", n(1)));

    // Five rewrites flip the name five times: it stopped, cleanly, mid-way.
    assert_eq!(result, Some(named_call("g", n(1))));
    assert_eq!(bounded.max_rewrites(), 5);
    // Without partial rewrites nothing is decided, so the run is declined.
    assert_eq!(
        run(&flipping().with_max_rewrites(5), &named_call("f", n(1))),
        None
    );
}

#[test]
fn a_run_that_reaches_its_bound_keeps_what_it_decided() {
    let simplifier = GroundSimplifier::new().with_max_rewrites(0);

    assert_eq!(run(&simplifier, &(n(1) + n(2))), None);
    assert_eq!(
        run(
            &GroundSimplifier::new().with_max_rewrites(1),
            &(n(1) + n(2))
        ),
        Some(n(3))
    );
}

#[test]
fn a_run_that_stops_at_its_bound_keeps_the_rewrites_it_made() {
    // The one rewrite the bound allows folds the left sum; the right sum is
    // left as it is, so the result is not decided.
    let expression = (n(1) + n(2)) + (n(3) + n(4));
    let bounded = GroundSimplifier::new().with_max_rewrites(1);

    assert_eq!(run(&bounded, &expression), None);
    assert_eq!(
        run(&bounded.with_partial_rewrites(), &expression),
        Some(n(3) + (n(3) + n(4)))
    );
}

#[test]
fn partial_rewrites_are_kept_only_when_asked_for() {
    let (_, x) = build_identifier("x");
    let expression = x.clone() + (n(1) + n(2));

    let declined = run(&GroundSimplifier::new(), &expression);
    let kept = run(
        &GroundSimplifier::new().with_partial_rewrites(),
        &expression,
    );

    assert_eq!(declined, None);
    assert_eq!(kept, Some(x + n(3)));
}

#[test]
fn partial_rewrites_still_decline_what_nothing_rewrites() {
    let (_, x) = build_identifier("x");
    let simplifier = GroundSimplifier::new().with_partial_rewrites();

    assert_eq!(run(&simplifier, &(x + n(1))), None);
}

#[test]
fn a_registered_constant_is_folded_in_place_with_the_context_registry() {
    let (registry, answer) = registry_with("answer", 42, FunctionSort::Int);
    let expression = (answer + n(1)).greater(n(42));

    let result = GroundSimplifier::new()
        .try_simplify(&expression, &SimplifyContext::from_registry(&registry));

    assert_eq!(result, Some(truth(true)));
}

#[test]
fn a_strategy_sees_the_context_it_is_given() {
    #[derive(Debug)]
    struct Spy(Arc<Mutex<Vec<Option<usize>>>>);

    impl SimplificationStrategy for Spy {
        fn name(&self) -> Cow<'_, str> {
            Cow::Borrowed("spy")
        }

        fn rewrite(&self, _node: &Expression, context: &SimplifyContext<'_>) -> Option<Expression> {
            self.0
                .lock()
                .expect("not poisoned")
                .push(context.registry().map(FunctionRegistry::len));
            None
        }
    }
    let sizes = Arc::new(Mutex::new(Vec::new()));
    let registry = FunctionRegistry::new();

    let result = GroundSimplifier::empty()
        .with_strategy(Spy(Arc::clone(&sizes)))
        .try_simplify(&n(1), &SimplifyContext::from_registry(&registry));

    assert_eq!(result, Some(n(1)));

    assert_eq!(*sizes.lock().expect("not poisoned"), vec![Some(0)]);
}
