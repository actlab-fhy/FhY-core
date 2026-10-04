//! Differential tests of the core's `GroundSimplifier` against the SymPy
//! backend as the oracle.
//!
//! The contract under test: wherever the ground simplifier folds an
//! expression, its result is exactly what [`SympySimplifier`] returns for
//! the same input, literal kind and form included; wherever it cannot, it
//! returns `None`, and never an approximation. A composed built-in is
//! checked against SymPy's answer for its inlined form, since the SymPy
//! backend refuses the call itself.
//!
//! The contract is checked at two levels. The whole default pipeline runs on
//! tables of expressions and on random trees. And each strategy, which keeps
//! the contract by itself, runs on its own node tables and on random nodes
//! whose children are decided: whenever it rewrites the node, SymPy's
//! simplification of the node is the rewrite. A new strategy adds its
//! cases here.

use std::collections::HashMap;

use fhy_core::expression::builtins::{BuiltinConstant, BuiltinFunction};
use fhy_core::expression::registry::FunctionRegistry;
use fhy_core::expression::{
    BinaryOperation, Callee, Expression, LiteralValue, LogicalOperation, UnaryOperation,
};
use fhy_core::solver::strategy::{
    Comparisons, ExactArithmetic, ExactBuiltins, LogicalOperators, NormalizeLiterals,
    PiecewiseDecision, SimplificationStrategy, default_strategies,
};
use fhy_core::solver::{GroundSimplifier, GroundWithFallback, SimplifyContext, SolveError, Solver};
use proptest::prelude::*;
use rstest::rstest;

use super::SympySimplifier;
use super::test_support::{backend, build_identifier, build_literal};

fn n(value: i64) -> Expression {
    Expression::from(value)
}

fn decimal(text: &str) -> Expression {
    build_literal(LiteralValue::Decimal(text.parse().expect("a decimal")))
}

fn truth(value: bool) -> Expression {
    build_literal(value)
}

fn call(function: BuiltinFunction, arguments: impl IntoIterator<Item = Expression>) -> Expression {
    Expression::call(function, arguments)
}

/// Return what the ground simplifier folds `expression` to, or `None`.
fn ground(expression: &Expression) -> Option<Expression> {
    let registry = FunctionRegistry::new();
    GroundSimplifier::new().try_simplify(expression, &SimplifyContext::from_registry(&registry))
}

/// Return SymPy's simplification of `expression`, with every composed
/// built-in inlined first.
fn sympy(expression: &Expression) -> Result<Expression, SolveError> {
    backend();
    let registry = FunctionRegistry::new();
    let inlined = registry.inline(expression).expect("inlined");
    Solver::new()
        .with_simplifier(SympySimplifier::new())
        .simplify(
            &inlined,
            &HashMap::new(),
            &SimplifyContext::from_registry(&registry),
        )
}

/// Assert the contract on `expression`: if the ground simplifier folds
/// it, SymPy answers the same expression.
fn assert_agrees(expression: &Expression) -> Option<Expression> {
    let folded = ground(expression)?;
    match sympy(expression) {
        Ok(expected) => assert_eq!(
            folded, expected,
            "{expression} folded to {folded:?}, SymPy says {expected:?}"
        ),
        Err(error) => panic!("{expression} folded to {folded:?}, but SymPy fails: {error}"),
    }
    Some(folded)
}

// ---------------------------------------------------------------------------
// The table: every construct the simplifier folds
// ---------------------------------------------------------------------------

#[rstest]
#[case::integer_literal(n(5))]
#[case::negative_integer_literal(n(-5))]
#[case::true_literal(truth(true))]
#[case::decimal_literal(decimal("0.5"))]
#[case::decimal_no_float_equals(decimal("0.1"))]
#[case::decimal_integer(decimal("2.0"))]
#[case::negative_decimal(-decimal("0.75"))]
#[case::sum(n(2) + n(3))]
#[case::difference(n(2) - n(5))]
#[case::product(n(-4) * n(6))]
#[case::negation(-n(7))]
#[case::double_negation(-(-n(7)))]
#[case::positive(n(7).positive())]
#[case::integer_quotient(n(6) / n(3))]
#[case::quotient_a_float_equals(n(1) / n(2))]
#[case::negative_quotient_a_float_equals(n(-3) / n(8))]
#[case::quotient_no_float_equals(n(1) / n(3))]
#[case::negative_quotient_no_float_equals(n(-1) / n(3))]
#[case::quotient_with_negative_denominator(n(1) / n(-3))]
#[case::quotient_reduces(n(10) / n(4))]
#[case::quotient_of_zero(n(0) / n(5))]
#[case::floor_division(n(7).floor_divide(n(2)))]
#[case::floor_division_of_a_negative(n(-7).floor_divide(n(2)))]
#[case::floor_division_by_a_negative(n(7).floor_divide(n(-2)))]
#[case::floor_division_exact(n(8).floor_divide(n(2)))]
#[case::floor_division_of_a_rational((n(7) / n(2)).floor_divide(n(2)))]
#[case::floor_division_by_a_rational(n(7).floor_divide(n(1) / n(2)))]
#[case::modulo(n(7).floor_mod(n(3)))]
#[case::modulo_by_a_negative(n(7).floor_mod(n(-3)))]
#[case::modulo_of_a_negative(n(-7).floor_mod(n(3)))]
#[case::modulo_of_both_negative(n(-7).floor_mod(n(-3)))]
#[case::modulo_exact(n(9).floor_mod(n(3)))]
#[case::modulo_of_a_rational((n(7) / n(2)).floor_mod(n(2)))]
#[case::modulo_by_a_rational(n(7).floor_mod(n(3) / n(2)))]
#[case::power(n(2).power(n(10)))]
#[case::power_of_a_negative(n(-2).power(n(3)))]
#[case::power_of_a_negative_even(n(-2).power(n(4)))]
#[case::zero_power_zero(n(0).power(n(0)))]
#[case::zero_power(n(0).power(n(5)))]
#[case::power_zero(n(7).power(n(0)))]
#[case::negative_power(n(2).power(n(-2)))]
#[case::negative_power_of_a_negative(n(-2).power(n(-3)))]
#[case::power_of_a_rational((n(2) / n(3)).power(n(2)))]
#[case::negative_power_of_a_rational((n(2) / n(3)).power(n(-2)))]
#[case::power_of_minus_one(n(-1).power(n(1_000_001)))]
#[case::huge_power_of_one(n(1).power(n(1_000_000_000_000)))]
#[case::square_root_power(n(4).power(n(1) / n(2)))]
#[case::fractional_power(n(8).power(n(2) / n(3)))]
#[case::negative_fractional_power(n(8).power(n(-1) / n(3)))]
#[case::rational_square_root((n(4) / n(9)).power(n(1) / n(2)))]
#[case::cube_root_of_a_fraction((n(1) / n(8)).power(n(1) / n(3)))]
#[case::power_of_one_with_a_fractional_exponent(n(1).power(n(1) / n(2)))]
#[case::zero_with_a_fractional_exponent(n(0).power(n(1) / n(2)))]
#[case::decimal_sum(decimal("0.1") + decimal("0.2"))]
#[case::decimal_product(decimal("0.5") * decimal("0.5"))]
#[case::decimal_with_an_integer(decimal("2.50") * n(2))]
#[case::large_integers(n(i64::MAX) * n(i64::MAX) + n(1))]
#[case::large_quotient(n(i64::MAX) / n(i64::MIN))]
#[case::deep_sum(n(1) + (n(2) + (n(3) + (n(4) + n(5)))))]
#[case::less(n(1).less(n(2)))]
#[case::less_fails(n(2).less(n(2)))]
#[case::less_equal(n(2).less_equal(n(2)))]
#[case::greater(n(3).greater(n(2)))]
#[case::greater_equal_fails(n(1).greater_equal(n(2)))]
#[case::equal(n(2).equals(n(2)))]
#[case::equal_fails(n(2).equals(n(3)))]
#[case::not_equal(n(2).not_equals(n(2)))]
#[case::rationals_compare_exactly((n(1) / n(3)).less(decimal("0.34")))]
#[case::integer_equals_decimal(n(1).equals(decimal("1.0")))]
#[case::decimal_equals_quotient(decimal("0.5").equals(n(1) / n(2)))]
#[case::booleans_equal(truth(true).equals(truth(true)))]
#[case::booleans_differ(truth(true).equals(truth(false)))]
#[case::booleans_not_equal(truth(true).not_equals(truth(false)))]
#[case::comparison_of_booleans((n(1).less(n(2))).equals(n(3).less(n(2))))]
#[case::negation_of_a_boolean(!truth(false))]
#[case::negated_comparison(!n(1).less(n(2)))]
#[case::conjunction(n(1).less(n(2)).and(n(2).less(n(3))))]
#[case::conjunction_fails(n(1).less(n(2)).and(n(3).less(n(2))))]
#[case::disjunction(n(5).less(n(2)).or(n(2).less(n(1))))]
#[case::disjunction_holds(n(5).less(n(2)).or(n(2).less(n(3))))]
#[case::three_way_conjunction(Expression::all([n(1).less(n(2)), n(2).less(n(3)), n(3).less(n(4))]))]
#[case::bound_check(n(3).greater_equal(n(0)).and(n(3).less(n(10))))]
#[case::floor(call(BuiltinFunction::Floor, [n(7) / n(2)]))]
#[case::floor_of_a_negative(call(BuiltinFunction::Floor, [n(-7) / n(2)]))]
#[case::floor_of_an_integer(call(BuiltinFunction::Floor, [n(4)]))]
#[case::ceil(call(BuiltinFunction::Ceil, [n(7) / n(2)]))]
#[case::ceil_of_a_negative(call(BuiltinFunction::Ceil, [n(-7) / n(2)]))]
#[case::ceil_of_an_integer(call(BuiltinFunction::Ceil, [n(5)]))]
#[case::round_of_an_integer(call(BuiltinFunction::Round, [n(5)]))]
#[case::sqrt_of_a_perfect_square(call(BuiltinFunction::Sqrt, [n(16)]))]
#[case::sqrt_of_a_rational_square(call(BuiltinFunction::Sqrt, [n(1) / n(4)]))]
#[case::sqrt_of_zero(call(BuiltinFunction::Sqrt, [n(0)]))]
#[case::sqrt_of_one(call(BuiltinFunction::Sqrt, [n(1)]))]
#[case::exp2(call(BuiltinFunction::Exp2, [n(10)]))]
#[case::exp2_of_zero(call(BuiltinFunction::Exp2, [n(0)]))]
#[case::exp2_of_a_negative(call(BuiltinFunction::Exp2, [n(-3)]))]
#[case::exp2_of_a_half_integer(call(BuiltinFunction::Exp2, [n(4) / n(2)]))]
#[case::log2(call(BuiltinFunction::Log2, [n(8)]))]
#[case::log2_of_one(call(BuiltinFunction::Log2, [n(1)]))]
#[case::log2_of_a_unit_fraction(call(BuiltinFunction::Log2, [n(1) / n(8)]))]
#[case::log10(call(BuiltinFunction::Log10, [n(1000)]))]
#[case::log10_of_a_unit_fraction(call(BuiltinFunction::Log10, [n(1) / n(100)]))]
#[case::log_of_one(call(BuiltinFunction::Log, [n(1)]))]
#[case::exp_of_zero(call(BuiltinFunction::Exp, [n(0)]))]
#[case::sin_of_zero(call(BuiltinFunction::Sin, [n(0)]))]
#[case::cos_of_zero(call(BuiltinFunction::Cos, [n(0)]))]
#[case::tan_of_zero(call(BuiltinFunction::Tan, [n(0)]))]
#[case::arcsin_of_zero(call(BuiltinFunction::Arcsin, [n(0)]))]
#[case::arccos_of_one(call(BuiltinFunction::Arccos, [n(1)]))]
#[case::arctan_of_zero(call(BuiltinFunction::Arctan, [n(0)]))]
#[case::sinh_of_zero(call(BuiltinFunction::Sinh, [n(0)]))]
#[case::cosh_of_zero(call(BuiltinFunction::Cosh, [n(0)]))]
#[case::tanh_of_zero(call(BuiltinFunction::Tanh, [n(0)]))]
#[case::erf_of_zero(call(BuiltinFunction::Erf, [n(0)]))]
#[case::max(call(BuiltinFunction::Max, [n(2), n(5)]))]
#[case::max_of_equals(call(BuiltinFunction::Max, [n(5), n(5)]))]
#[case::min(call(BuiltinFunction::Min, [n(2), n(5)]))]
#[case::min_of_rationals(call(BuiltinFunction::Min, [n(1) / n(2), n(1) / n(3)]))]
#[case::abs(call(BuiltinFunction::Abs, [n(-5)]))]
#[case::abs_of_a_positive(call(BuiltinFunction::Abs, [n(5)]))]
#[case::abs_of_a_rational(call(BuiltinFunction::Abs, [n(-1) / n(2)]))]
#[case::sign(call(BuiltinFunction::Sign, [n(-5)]))]
#[case::sign_of_zero(call(BuiltinFunction::Sign, [n(0)]))]
#[case::clamp(call(BuiltinFunction::Clamp, [n(15), n(0), n(10)]))]
#[case::clamp_inside(call(BuiltinFunction::Clamp, [n(5), n(0), n(10)]))]
#[case::clamp_symmetric(call(BuiltinFunction::ClampSymmetric, [n(-15), n(10)]))]
#[case::relu(call(BuiltinFunction::Relu, [n(-3)]))]
#[case::relu_of_a_positive(call(BuiltinFunction::Relu, [n(3)]))]
#[case::leaky_relu(call(BuiltinFunction::LeakyRelu, [n(-4), n(1) / n(2)]))]
#[case::xor(call(BuiltinFunction::Xor, [truth(true), truth(false)]))]
#[case::nand(call(BuiltinFunction::Nand, [truth(true), truth(true)]))]
#[case::nor(call(BuiltinFunction::Nor, [truth(false), truth(false)]))]
#[case::implies(call(BuiltinFunction::Implies, [truth(true), truth(false)]))]
#[case::iff(call(BuiltinFunction::Iff, [truth(false), truth(false)]))]
#[case::piecewise_first_case(
    Expression::piecewise([(n(1).less(n(2)), n(10))], n(20)).expect("a piecewise")
)]
#[case::piecewise_otherwise(
    Expression::piecewise([(n(3).less(n(2)), n(10))], n(1) / n(2)).expect("a piecewise")
)]
#[case::piecewise_second_case(
    Expression::piecewise([(n(3).less(n(2)), n(10)), (n(1).less(n(2)), n(20))], n(30))
        .expect("a piecewise")
)]
#[case::piecewise_boolean_value(
    Expression::piecewise([(n(1).less(n(2)), n(3).less(n(2)))], truth(true)).expect("a piecewise")
)]
#[case::piecewise_in_arithmetic(
    n(1) + Expression::piecewise([(n(1).less(n(2)), n(10))], n(20)).expect("a piecewise")
)]
#[case::floor_divide_of_a_piecewise(
    Expression::piecewise([(n(1).less(n(2)), n(10))], n(20)).expect("a piecewise").floor_divide(n(4))
)]
#[case::modulo_of_a_piecewise(
    Expression::piecewise([(n(1).less(n(2)), n(2))], n(0)).expect("a piecewise").floor_mod(n(-6))
)]
#[case::nested_piecewise(
    Expression::piecewise(
        [(n(1).less(n(2)), Expression::piecewise([(n(3).less(n(2)), n(1))], n(2)).expect("a piecewise"))],
        n(3),
    )
    .expect("a piecewise")
    .floor_divide(n(2))
)]
fn the_ground_simplifier_folds_what_sympy_returns(#[case] expression: Expression) {
    let folded = assert_agrees(&expression);

    assert!(folded.is_some(), "{expression} should fold");
}

// ---------------------------------------------------------------------------
// The table: every case it declines
// ---------------------------------------------------------------------------

#[rstest]
#[case::free_identifier(build_identifier("x").1)]
#[case::free_identifier_in_a_sum(build_identifier("x").1 + n(1))]
#[case::cancellation_sympy_does(build_identifier("x").1 - build_identifier("x").1)]
#[case::free_beside_ground(n(1) + n(2) + build_identifier("x").1)]
#[case::float(build_literal(1.5))]
#[case::float_sum(build_literal(1.5) + n(1))]
#[case::float_product(build_literal(0.1) * n(3))]
#[case::float_comparison(build_literal(1.5).less(n(2)))]
#[case::float_equality(build_literal(0.5).equals(decimal("0.5")))]
#[case::division_by_zero(n(1) / n(0))]
#[case::zero_over_zero(n(0) / n(0))]
#[case::floor_division_by_zero(n(1).floor_divide(n(0)))]
#[case::modulo_by_zero(n(1).floor_mod(n(0)))]
#[case::zero_to_a_negative_power(n(0).power(n(-1)))]
#[case::irrational_root(n(2).power(n(1) / n(2)))]
#[case::partly_exact_root((n(4) / n(3)).power(n(1) / n(2)))]
#[case::partly_exact_power(n(8).power(n(1) / n(2)))]
#[case::root_of_a_negative(n(-4).power(n(1) / n(2)))]
#[case::cube_root_of_a_negative(n(-8).power(n(1) / n(3)))]
#[case::huge_power(n(3).power(n(1_000_000)))]
#[case::huge_exponent(n(3).power(n(1_000_000_000_000)))]
#[case::sqrt_of_a_non_square(call(BuiltinFunction::Sqrt, [n(2)]))]
#[case::sqrt_of_a_negative(call(BuiltinFunction::Sqrt, [n(-4)]))]
#[case::log2_of_a_non_power(call(BuiltinFunction::Log2, [n(6)]))]
#[case::log2_of_zero(call(BuiltinFunction::Log2, [n(0)]))]
#[case::log2_of_a_negative(call(BuiltinFunction::Log2, [n(-2)]))]
#[case::log_of_two(call(BuiltinFunction::Log, [n(2)]))]
#[case::exp_of_one(call(BuiltinFunction::Exp, [n(1)]))]
#[case::sin_of_one(call(BuiltinFunction::Sin, [n(1)]))]
#[case::cos_of_one(call(BuiltinFunction::Cos, [n(1)]))]
#[case::erf_of_one(call(BuiltinFunction::Erf, [n(1)]))]
#[case::round_of_a_rational(call(BuiltinFunction::Round, [n(1) / n(2)]))]
#[case::sigmoid(call(BuiltinFunction::Sigmoid, [n(0)]))]
#[case::silu(call(BuiltinFunction::Silu, [n(0)]))]
#[case::gelu(call(BuiltinFunction::Gelu, [n(0)]))]
#[case::user_function(
    Expression::call("custom".parse::<Callee>().expect("a name"), [n(1)])
)]
#[case::builtin_constant(Expression::from(BuiltinConstant::Pi.identifier().clone()))]
#[case::builtin_constant_comparison(
    Expression::from(BuiltinConstant::Pi.identifier().clone()).greater(n(3))
)]
#[case::infinity(Expression::from(BuiltinConstant::Inf.identifier().clone()) + n(1))]
#[case::piecewise_with_a_branch_it_does_not_take_that_sympy_rejects(
    Expression::piecewise([(n(1).less(n(2)), n(10))], n(1).floor_mod(n(0))).expect("a piecewise")
)]
#[case::piecewise_of_a_free_condition(
    Expression::piecewise([(build_identifier("x").1.less(n(2)), n(10))], n(20)).expect("a piecewise")
)]
fn the_ground_simplifier_declines_what_it_cannot_match(#[case] expression: Expression) {
    assert_eq!(ground(&expression), None, "{expression} should decline");
}

// ---------------------------------------------------------------------------
// Each strategy alone
// ---------------------------------------------------------------------------

/// Assert that `strategy`, which rewrites a node of `nodes` or declines it,
/// rewrites it to SymPy's simplification of the node, and return how many
/// nodes it rewrote.
fn assert_strategy_agrees(strategy: &dyn SimplificationStrategy, nodes: &[Expression]) -> usize {
    let registry = FunctionRegistry::new();
    let context = SimplifyContext::from_registry(&registry);
    let mut rewritten = 0;
    for node in nodes {
        let Some(rewrite) = strategy.rewrite(node, &context) else {
            continue;
        };
        rewritten += 1;
        match sympy(node) {
            Ok(expected) => assert_eq!(
                rewrite,
                expected,
                "{}: {node} rewrote to {rewrite:?}, SymPy says {expected:?}",
                strategy.name()
            ),
            Err(error) => panic!(
                "{}: {node} rewrote to {rewrite:?}, but SymPy fails: {error}",
                strategy.name()
            ),
        }
    }
    rewritten
}

/// The binary operations over two literals, for each pair of `operands`.
fn binary_nodes(operations: &[BinaryOperation], operands: &[Expression]) -> Vec<Expression> {
    let mut nodes = Vec::new();
    for operation in operations {
        for left in operands {
            for right in operands {
                nodes.push(Expression::new_binary(
                    *operation,
                    left.clone(),
                    right.clone(),
                ));
            }
        }
    }
    nodes
}

/// Numbers in the forms the strategies read: integers, decimals, a negated
/// decimal and quotients.
fn number_operands() -> Vec<Expression> {
    vec![
        n(0),
        n(1),
        n(-1),
        n(2),
        n(-3),
        n(7),
        decimal("0.5"),
        decimal("2.25"),
        -decimal("0.75"),
        n(1) / n(3),
        n(-2) / n(3),
        n(-1) / n(6),
    ]
}

#[test]
fn exact_arithmetic_rewrites_what_sympy_returns() {
    let operations = [
        BinaryOperation::Add,
        BinaryOperation::Subtract,
        BinaryOperation::Multiply,
        BinaryOperation::Divide,
        BinaryOperation::FloorDivide,
        BinaryOperation::FloorMod,
    ];
    let mut nodes = binary_nodes(&operations, &number_operands());
    // Powers over every base and the exponents that keep SymPy quick.
    let exponents = [
        n(0),
        n(1),
        n(-1),
        n(3),
        n(-2),
        n(1) / n(2),
        n(1) / n(3),
        n(2) / n(3),
        n(-1) / n(2),
    ];
    for base in
        number_operands()
            .into_iter()
            .chain([n(4), n(8), n(1) / n(4), n(1) / n(8), n(27) / n(8)])
    {
        for exponent in &exponents {
            nodes.push(base.power(exponent.clone()));
        }
    }
    for operand in number_operands() {
        nodes.push(-operand.clone());
        nodes.push(operand.positive());
    }
    // Rewrites of numbers not yet in SymPy's form.
    nodes.extend([
        n(2) / n(6),
        n(10) / n(4),
        n(-9) / n(6),
        n(0) / n(5),
        n(5) / n(-10),
    ]);

    let rewritten = assert_strategy_agrees(&ExactArithmetic::new(), &nodes);

    assert!(rewritten > 400, "only {rewritten} nodes were rewritten");
}

#[test]
fn comparisons_rewrite_what_sympy_returns() {
    let operations = [
        BinaryOperation::Equal,
        BinaryOperation::NotEqual,
        BinaryOperation::Less,
        BinaryOperation::LessEqual,
        BinaryOperation::Greater,
        BinaryOperation::GreaterEqual,
    ];
    let mut operands = number_operands();
    operands.extend([truth(true), truth(false)]);

    let rewritten =
        assert_strategy_agrees(&Comparisons::new(), &binary_nodes(&operations, &operands));

    // Numbers compare under every operation, Booleans under two.
    assert_eq!(rewritten, 6 * 12 * 12 + 2 * 2 * 2);
}

#[test]
fn logical_operators_rewrite_what_sympy_returns() {
    let booleans = [truth(true), truth(false)];
    let mut nodes: Vec<Expression> = booleans.iter().map(|a| !a.clone()).collect();
    for a in &booleans {
        for b in &booleans {
            nodes.push(Expression::new_logical(
                LogicalOperation::And,
                [a.clone(), b.clone()],
            ));
            nodes.push(Expression::new_logical(
                LogicalOperation::Or,
                [a.clone(), b.clone()],
            ));
            for function in [
                BuiltinFunction::Xor,
                BuiltinFunction::Nand,
                BuiltinFunction::Nor,
                BuiltinFunction::Implies,
                BuiltinFunction::Iff,
            ] {
                nodes.push(call(function, [a.clone(), b.clone()]));
            }
            for c in &booleans {
                nodes.push(Expression::all([a.clone(), b.clone(), c.clone()]));
                nodes.push(Expression::any([a.clone(), b.clone(), c.clone()]));
            }
        }
    }

    let rewritten = assert_strategy_agrees(&LogicalOperators::new(), &nodes);

    assert_eq!(rewritten, nodes.len());
}

#[test]
fn piecewise_decision_rewrites_what_sympy_returns() {
    let values = [
        n(1),
        n(-2),
        decimal("0.5"),
        n(1) / n(3),
        truth(true),
        truth(false),
    ];
    let mut nodes = Vec::new();
    for first in [true, false] {
        for second in [true, false] {
            for value in &values {
                // A piecewise's values are Boolean or numeric together.
                let (one, two, otherwise) = if matches!(
                    value.kind(),
                    fhy_core::expression::ExpressionKind::Literal(LiteralValue::Bool(_))
                ) {
                    (truth(true), truth(false), value.clone())
                } else {
                    (n(10), value.clone(), decimal("0.5"))
                };
                nodes.push(
                    Expression::piecewise(
                        [(truth(first), one.clone()), (truth(second), two.clone())],
                        otherwise.clone(),
                    )
                    .expect("a piecewise"),
                );
                nodes.push(
                    Expression::piecewise([(truth(first), one)], otherwise).expect("a piecewise"),
                );
            }
        }
    }

    let rewritten = assert_strategy_agrees(&PiecewiseDecision::new(), &nodes);

    assert_eq!(rewritten, nodes.len());
}

#[test]
fn exact_builtins_rewrite_what_sympy_returns() {
    let operands = number_operands();
    let mut nodes = Vec::new();
    for function in [
        BuiltinFunction::Floor,
        BuiltinFunction::Ceil,
        BuiltinFunction::Round,
        BuiltinFunction::Sqrt,
        BuiltinFunction::Exp2,
        BuiltinFunction::Log2,
        BuiltinFunction::Log10,
        BuiltinFunction::Exp,
        BuiltinFunction::Log,
        BuiltinFunction::Sin,
        BuiltinFunction::Cos,
        BuiltinFunction::Tan,
        BuiltinFunction::Arcsin,
        BuiltinFunction::Arccos,
        BuiltinFunction::Arctan,
        BuiltinFunction::Sinh,
        BuiltinFunction::Cosh,
        BuiltinFunction::Tanh,
        BuiltinFunction::Erf,
        BuiltinFunction::Abs,
        BuiltinFunction::Sign,
        BuiltinFunction::Relu,
    ] {
        for operand in
            operands
                .iter()
                .chain(&[n(4), n(8), n(16), n(100), n(1000), n(1) / n(4), n(1) / n(8)])
        {
            nodes.push(call(function, [operand.clone()]));
        }
    }
    for function in [
        BuiltinFunction::Max,
        BuiltinFunction::Min,
        BuiltinFunction::ClampSymmetric,
        BuiltinFunction::LeakyRelu,
    ] {
        for a in &operands {
            for b in &operands {
                nodes.push(call(function, [a.clone(), b.clone()]));
            }
        }
    }
    for x in [n(-5), n(5), n(1) / n(2)] {
        nodes.push(call(BuiltinFunction::Clamp, [x.clone(), n(-1), n(2)]));
        nodes.push(call(BuiltinFunction::Clamp, [x, n(0), n(1) / n(3)]));
    }

    let rewritten = assert_strategy_agrees(&ExactBuiltins::new(), &nodes);

    assert!(rewritten > 150, "only {rewritten} nodes were rewritten");
}

#[test]
fn normalize_literals_rewrites_what_sympy_returns() {
    let nodes: Vec<Expression> = [
        "0.1", "0.5", "2.0", "2.50", "0.75", "3.141", "100", "0.0625", "0.3",
    ]
    .iter()
    .map(|text| decimal(text))
    .collect();

    let rewritten = assert_strategy_agrees(&NormalizeLiterals::new(), &nodes);

    // The decimals SymPy lifts to another form: all but the ones a binary
    // float equals.
    assert_eq!(rewritten, 5);
}

#[test]
fn registered_constants_rewrite_what_sympy_returns() {
    use fhy_core::expression::registry::NativeConstant;
    use fhy_core::expression::{FunctionName, FunctionSort};

    let mut registry = FunctionRegistry::new();
    let constants: Vec<Expression> = [
        ("whole", FunctionSort::Int, LiteralValue::from(42)),
        ("negative", FunctionSort::Int, LiteralValue::from(-7)),
        ("yes", FunctionSort::Bool, LiteralValue::from(true)),
    ]
    .into_iter()
    .map(|(name, sort, value)| {
        Expression::from(
            registry
                .register_constant(
                    NativeConstant::new(FunctionName::new(name).expect("a name"), sort, value)
                        .expect("a constant"),
                )
                .expect("registered"),
        )
    })
    .collect();
    backend();
    let context = SimplifyContext::from_registry(&registry);
    let strategy = fhy_core::solver::strategy::RegisteredConstants::new();

    for constant in &constants {
        let rewrite = strategy.rewrite(constant, &context).expect("rewritten");
        let expected = Solver::new()
            .with_simplifier(SympySimplifier::new())
            .simplify(constant, &HashMap::new(), &context)
            .expect("simplified");
        assert_eq!(rewrite, expected);
    }
}

/// Return a number or Boolean in the forms the strategies read.
fn decided_leaf() -> BoxedStrategy<Expression> {
    prop_oneof![
        8 => (-9_i64..=9, 1_i64..=9).prop_map(|(a, b)| ground(&(n(a) / n(b))).expect("folded")),
        1 => any::<bool>().prop_map(truth),
    ]
    .boxed()
}

/// Return a node over decided children: the shapes the strategies rewrite,
/// numbers and Booleans mixed, so they decline the ill-typed ones.
fn decided_node() -> BoxedStrategy<Expression> {
    let operations = prop::sample::select(vec![
        BinaryOperation::Add,
        BinaryOperation::Subtract,
        BinaryOperation::Multiply,
        BinaryOperation::Divide,
        BinaryOperation::FloorDivide,
        BinaryOperation::FloorMod,
        BinaryOperation::Equal,
        BinaryOperation::NotEqual,
        BinaryOperation::Less,
        BinaryOperation::LessEqual,
        BinaryOperation::Greater,
        BinaryOperation::GreaterEqual,
    ]);
    let functions = prop::sample::select(vec![
        BuiltinFunction::Floor,
        BuiltinFunction::Ceil,
        BuiltinFunction::Round,
        BuiltinFunction::Sqrt,
        BuiltinFunction::Exp2,
        BuiltinFunction::Log2,
        BuiltinFunction::Log10,
        BuiltinFunction::Exp,
        BuiltinFunction::Log,
        BuiltinFunction::Sin,
        BuiltinFunction::Abs,
        BuiltinFunction::Sign,
        BuiltinFunction::Relu,
    ]);
    let binary_functions = prop::sample::select(vec![
        BuiltinFunction::Max,
        BuiltinFunction::Min,
        BuiltinFunction::ClampSymmetric,
        BuiltinFunction::LeakyRelu,
        BuiltinFunction::Xor,
        BuiltinFunction::Iff,
        BuiltinFunction::Implies,
    ]);
    let exponent = prop_oneof![
        (-3_i64..=4).prop_map(n),
        Just(n(1) / n(2)),
        Just(n(1) / n(3)),
        Just(n(-2) / n(3)),
    ];
    prop_oneof![
        (operations, decided_leaf(), decided_leaf())
            .prop_map(|(operation, a, b)| Expression::new_binary(operation, a, b)),
        (decided_leaf(), exponent).prop_map(|(base, exponent)| base.power(exponent)),
        (decided_leaf(), any::<bool>()).prop_map(|(a, negate)| if negate {
            -a
        } else {
            a.positive()
        }),
        decided_leaf().prop_map(|a| !a),
        (decided_leaf(), decided_leaf())
            .prop_map(|(a, b)| Expression::new_logical(LogicalOperation::And, [a, b])),
        (functions, decided_leaf()).prop_map(|(function, a)| call(function, [a])),
        (binary_functions, decided_leaf(), decided_leaf())
            .prop_map(|(function, a, b)| call(function, [a, b])),
        (
            decided_leaf(),
            decided_leaf(),
            decided_leaf(),
            decided_leaf()
        )
            .prop_map(|(first, a, b, otherwise)| Expression::piecewise(
                [(first, a), (truth(false), b)],
                otherwise
            )
            .unwrap_or_else(|_| truth(true))),
    ]
    .boxed()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    #[test]
    fn every_default_strategy_rewrites_a_decided_node_to_sympy_s_answer_or_declines(
        node in decided_node(),
    ) {
        for strategy in default_strategies() {
            assert_strategy_agrees(strategy.as_ref(), std::slice::from_ref(&node));
        }
    }
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

/// Return an integer or decimal tree: leaves, and every operation the
/// simplifier folds, with exponents kept small enough for SymPy to answer
/// quickly.
fn number_tree() -> BoxedStrategy<Expression> {
    let leaves = prop_oneof![
        8 => (-6_i64..=6).prop_map(n),
        1 => Just(n(i64::MAX)),
        1 => Just(decimal("0.5")),
        1 => Just(decimal("0.1")),
        1 => Just(decimal("2.25")),
        1 => Just(build_literal(1.5)),
        1 => Just(n(1) / n(3)),
    ];
    leaves
        .prop_recursive(4, 32, 2, |inner| {
            let exponent = prop_oneof![
                (-3_i64..=5).prop_map(n),
                Just(n(1) / n(2)),
                Just(n(1) / n(3)),
                Just(n(2) / n(3)),
                Just(n(-1) / n(2)),
            ];
            let integer_argument = inner.clone();
            prop_oneof![
                (inner.clone(), inner.clone()).prop_map(|(a, b)| a + b),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| a - b),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| a * b),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| a / b),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| a.floor_divide(b)),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| a.floor_mod(b)),
                (inner.clone(), exponent).prop_map(|(a, e)| a.power(e)),
                inner.clone().prop_map(|a| -a),
                inner.clone().prop_map(|a| a.positive()),
                (
                    prop::sample::select(vec![
                        BuiltinFunction::Floor,
                        BuiltinFunction::Ceil,
                        BuiltinFunction::Round,
                        BuiltinFunction::Sqrt,
                        BuiltinFunction::Exp2,
                        BuiltinFunction::Log2,
                        BuiltinFunction::Log10,
                        BuiltinFunction::Exp,
                        BuiltinFunction::Log,
                        BuiltinFunction::Sin,
                        BuiltinFunction::Cos,
                        BuiltinFunction::Abs,
                        BuiltinFunction::Sign,
                        BuiltinFunction::Relu,
                    ]),
                    integer_argument,
                )
                    .prop_map(|(function, a)| call(function, [a])),
                (
                    prop::sample::select(vec![BuiltinFunction::Max, BuiltinFunction::Min]),
                    inner.clone(),
                    inner.clone(),
                )
                    .prop_map(|(function, a, b)| call(function, [a, b])),
                (inner.clone(), inner.clone(), inner.clone())
                    .prop_map(|(a, lo, hi)| call(BuiltinFunction::Clamp, [a, lo, hi])),
                (boolean_leaf(), inner.clone(), inner.clone()).prop_map(
                    |(condition, value, otherwise)| {
                        Expression::piecewise([(condition, value)], otherwise).expect("a piecewise")
                    }
                ),
            ]
        })
        .boxed()
}

/// Return a small Boolean expression over numbers.
fn boolean_leaf() -> BoxedStrategy<Expression> {
    let leaves = (-3_i64..=3, -3_i64..=3, 0..6_u8).prop_map(|(a, b, which)| match which {
        0 => n(a).less(n(b)),
        1 => n(a).less_equal(n(b)),
        2 => n(a).greater(n(b)),
        3 => n(a).greater_equal(n(b)),
        4 => n(a).equals(n(b)),
        _ => n(a).not_equals(n(b)),
    });
    prop_oneof![leaves, any::<bool>().prop_map(truth)].boxed()
}

/// Return a Boolean tree: comparisons of number trees, connectives,
/// negations and the Boolean built-ins.
fn boolean_tree() -> BoxedStrategy<Expression> {
    let comparison =
        (number_tree(), number_tree(), 0..6_u8).prop_map(|(a, b, which)| match which {
            0 => a.less(b),
            1 => a.less_equal(b),
            2 => a.greater(b),
            3 => a.greater_equal(b),
            4 => a.equals(b),
            _ => a.not_equals(b),
        });
    prop_oneof![comparison, boolean_leaf()]
        .prop_recursive(3, 12, 2, |inner| {
            prop_oneof![
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| Expression::new_logical(LogicalOperation::And, [a, b])),
                (inner.clone(), inner.clone())
                    .prop_map(|(a, b)| Expression::new_logical(LogicalOperation::Or, [a, b])),
                inner.clone().prop_map(|a| !a),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| a.equals(b)),
                (
                    prop::sample::select(vec![
                        BuiltinFunction::Xor,
                        BuiltinFunction::Nand,
                        BuiltinFunction::Nor,
                        BuiltinFunction::Implies,
                        BuiltinFunction::Iff,
                    ]),
                    inner.clone(),
                    inner,
                )
                    .prop_map(|(function, a, b)| call(function, [a, b])),
            ]
        })
        .boxed()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn a_ground_number_tree_folds_to_sympy_s_answer_or_is_declined(tree in number_tree()) {
        // The assertion is inside `assert_agrees`: a fold SymPy answers
        // differently, or fails on, panics.
        assert_agrees(&tree);
    }

    #[test]
    fn a_ground_boolean_tree_folds_to_sympy_s_answer_or_is_declined(tree in boolean_tree()) {
        assert_agrees(&tree);
    }

    #[test]
    fn a_tree_with_a_free_identifier_is_declined(
        tree in number_tree().prop_map(|tree| tree + build_identifier("free").1),
    ) {
        prop_assert_eq!(ground(&tree), None);
    }
}

/// The generators must not pass vacuously: a good share of their trees must
/// fold, so the property compares them with SymPy.
#[test]
fn the_random_trees_fold_often_enough_to_test_something() {
    use proptest::strategy::ValueTree;
    use proptest::test_runner::TestRunner;

    let mut runner = TestRunner::deterministic();
    for (name, strategy) in [("number", number_tree()), ("boolean", boolean_tree())] {
        let folded = (0..400)
            .filter(|_| {
                let tree = strategy.new_tree(&mut runner).expect("a tree").current();
                ground(&tree).is_some()
            })
            .count();

        assert!(folded >= 80, "only {folded} of 400 {name} trees fold");
    }
}

/// Every binary and unary operation folds, which the table's cases name
/// one by one and this one pins to the operation enums.
#[test]
fn every_operation_the_ground_simplifier_matches_appears_in_the_table() {
    let operations = [
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
    for operation in operations {
        let expression = Expression::new_binary(operation, n(7), n(2));
        assert!(
            assert_agrees(&expression).is_some(),
            "{expression} should fold"
        );
    }
    for operation in [UnaryOperation::Negate, UnaryOperation::Positive] {
        let expression = Expression::new_unary(operation, n(7));
        assert!(assert_agrees(&expression).is_some());
    }
    let expression = Expression::new_unary(UnaryOperation::LogicalNot, truth(true));
    assert!(assert_agrees(&expression).is_some());
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/// Time the three simplifiers through a `Solver`, substituting the
/// environment first as a parameter `assign` does, on the shapes a
/// constraint check takes once its identifiers are bound.
///
/// Not a test: run it with `cargo test --release -p fhy-core-py
/// ground_differential::timing -- --ignored --nocapture`, which prints the
/// microseconds per simplification of each row.
#[test]
#[ignore = "a timing, not a check: run it in a release build with --nocapture"]
#[expect(clippy::print_stdout, reason = "the timing is the output")]
fn timing_of_the_simplifiers_on_constraint_shapes() {
    use std::time::Instant;

    let (x, reference) = build_identifier("x");
    let (a, a_reference) = build_identifier("a");
    let (b, b_reference) = build_identifier("b");
    let shapes: Vec<(&str, Expression, HashMap<_, _>)> = vec![
        (
            "integer comparison",
            reference.greater_equal(0),
            HashMap::from([(x.clone(), n(3))]),
        ),
        (
            "bound check",
            reference
                .greater_equal(0)
                .and(reference.less(64))
                .and(reference.floor_mod(8).equals(0)),
            HashMap::from([(x, n(24))]),
        ),
        (
            "arithmetic tree",
            (&a_reference * &b_reference + 5).floor_divide(4) - a_reference.floor_mod(3),
            HashMap::from([(a, n(7)), (b, n(9))]),
        ),
    ];
    backend();
    let registry = FunctionRegistry::new();
    let context = SimplifyContext::from_registry(&registry);
    let solvers = [
        (
            "ground",
            Solver::new().with_simplifier(GroundSimplifier::new()),
        ),
        (
            "chain",
            Solver::new().with_simplifier(GroundWithFallback::new(SympySimplifier::new())),
        ),
        (
            "sympy",
            Solver::new().with_simplifier(SympySimplifier::new()),
        ),
    ];
    for (shape, expression, environment) in &shapes {
        let mut microseconds = Vec::new();
        for (name, solver) in &solvers {
            let rounds: u32 = if *name == "sympy" { 2_000 } else { 200_000 };
            let answer = solver
                .simplify(expression, environment, &context)
                .expect("simplified");
            let started = Instant::now();
            for _ in 0..rounds {
                std::hint::black_box(
                    solver
                        .simplify(std::hint::black_box(expression), environment, &context)
                        .expect("simplified"),
                );
            }
            let each = started.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
            println!("{shape:20} {name:7} {each:9.3} us  -> {answer}");
            microseconds.push(each);
        }
        let substituted = expression.substitute(environment).expect("substituted");
        let ground = GroundSimplifier::new();
        let started = Instant::now();
        for _ in 0..200_000_u32 {
            std::hint::black_box(ground.try_simplify(std::hint::black_box(&substituted), &context));
        }
        let fold = started.elapsed().as_secs_f64() * 1e6 / 200_000.0;
        println!("{shape:20} fold alone {fold:9.3} us (the substituted tree, no solver)");
        println!(
            "{shape:20} sympy/ground = {:.0}x, sympy/chain = {:.0}x",
            microseconds[2] / microseconds[0],
            microseconds[2] / microseconds[1]
        );
    }
}
