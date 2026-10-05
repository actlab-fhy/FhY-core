//! Stories for the ground simplifier with its default strategies, and its
//! composition: what the whole pipeline folds, in the form `SymPy`'s lifting
//! gives, and what it declines. The strategies and the driver have their
//! own stories in `ground_strategy_stories`.
//!
//! The binding's differential tests (`solver::sympy::ground_differential`)
//! check the same constructs against `SymPy`; these run without Python.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use fhy_core::expression::builtins::{BuiltinConstant, BuiltinFunction};
use fhy_core::expression::registry::{FunctionRegistry, NativeConstant};
use fhy_core::expression::{
    Expression, FunctionName, FunctionSort, LiteralValue, LogicalOperation,
};
use fhy_core::foreign::BoxError;
use fhy_core::solver::strategy::ComposedBuiltins;
use fhy_core::solver::{
    GroundSimplifier, GroundWithFallback, Simplifier, SimplifyContext, SolveError, Solver,
};
use rstest::rstest;

use crate::support::expression::{build_identifier, build_literal};
use crate::support::solver::{FakeBackendError, RecordingSimplifier};

fn n(value: i64) -> Expression {
    Expression::from(value)
}

fn decimal(text: &str) -> Expression {
    build_literal(LiteralValue::Decimal(text.parse().expect("a decimal")))
}

fn float(value: f64) -> Expression {
    build_literal(value)
}

fn truth(value: bool) -> Expression {
    build_literal(value)
}

/// The quotient `numerator / denominator` as an expression, unfolded.
fn quotient(numerator: i64, denominator: i64) -> Expression {
    Expression::new_binary(
        fhy_core::expression::BinaryOperation::Divide,
        n(numerator),
        n(denominator),
    )
}

fn negated(expression: Expression) -> Expression {
    Expression::new_unary(fhy_core::expression::UnaryOperation::Negate, expression)
}

fn call(function: BuiltinFunction, arguments: impl IntoIterator<Item = Expression>) -> Expression {
    Expression::call(function, arguments)
}

fn folded(expression: &Expression) -> Option<Expression> {
    GroundSimplifier::new().try_simplify(expression, &SimplifyContext::default())
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

#[rstest]
#[case::sum(n(2) + n(3), n(5))]
#[case::difference(n(2) - n(5), n(-3))]
#[case::product(n(-4) * n(6), n(-24))]
#[case::negation(-n(7), n(-7))]
#[case::positive(n(7).positive(), n(7))]
#[case::integer_quotient(n(6) / n(3), n(2))]
#[case::quotient_that_a_float_equals(n(1) / n(2), decimal("0.5"))]
#[case::negative_quotient_that_a_float_equals(n(-3) / n(8), negated(decimal("0.375")))]
#[case::quotient_no_float_equals(n(1) / n(3), quotient(1, 3))]
#[case::negative_quotient_no_float_equals(n(-1) / n(3), quotient(-1, 3))]
#[case::quotient_reduces(n(10) / n(4), decimal("2.5"))]
#[case::floor_division(n(7).floor_divide(n(2)), n(3))]
#[case::floor_division_rounds_down(n(-7).floor_divide(n(2)), n(-4))]
#[case::floor_division_by_a_negative(n(7).floor_divide(n(-2)), n(-4))]
#[case::floor_division_of_a_rational(
    (n(7) / n(2)).floor_divide(n(2)),
    n(1)
)]
#[case::modulo(n(7).floor_mod(n(3)), n(1))]
#[case::modulo_takes_the_divisor_s_sign(n(7).floor_mod(n(-3)), n(-2))]
#[case::modulo_of_a_negative(n(-7).floor_mod(n(3)), n(2))]
#[case::modulo_of_a_rational((n(7) / n(2)).floor_mod(n(2)), decimal("1.5"))]
#[case::power(n(2).power(n(10)), n(1024))]
#[case::power_of_a_negative(n(-2).power(n(3)), n(-8))]
#[case::zero_power_zero(n(0).power(n(0)), n(1))]
#[case::negative_power(n(2).power(n(-2)), decimal("0.25"))]
#[case::power_of_a_rational((n(2) / n(3)).power(n(2)), quotient(4, 9))]
#[case::exact_square_root_power(n(4).power(n(1) / n(2)), n(2))]
#[case::exact_fractional_power(n(8).power(n(2) / n(3)), n(4))]
#[case::exact_negative_fractional_power(n(8).power(n(-1) / n(3)), decimal("0.5"))]
#[case::power_of_minus_one(n(-1).power(n(1_000_001)), n(-1))]
#[case::huge_power_of_one(n(1).power(n(1_000_000_000_000)), n(1))]
#[case::decimal_arithmetic_is_exact(decimal("0.1") + decimal("0.2"), quotient(3, 10))]
#[case::decimal_literal_normalizes(decimal("0.1"), quotient(1, 10))]
#[case::decimal_integer_is_an_int(decimal("2.50") * n(2), n(5))]
#[case::large_integers_are_exact(
    n(i64::MAX) * n(i64::MAX) + n(1),
    build_literal("85070591730234615847396907784232501250".parse::<fhy_core::expression::BigInt>().expect("digits"))
)]
fn arithmetic_folds_to_the_literal_sympy_lifts(
    #[case] expression: Expression,
    #[case] expected: Expression,
) {
    assert_eq!(folded(&expression), Some(expected));
}

// ---------------------------------------------------------------------------
// Comparisons, logic and piecewise
// ---------------------------------------------------------------------------

#[rstest]
#[case::less(n(1).less(n(2)), true)]
#[case::less_fails(n(2).less(n(2)), false)]
#[case::less_equal(n(2).less_equal(n(2)), true)]
#[case::greater(n(3).greater(n(2)), true)]
#[case::greater_equal_fails(n(1).greater_equal(n(2)), false)]
#[case::equal(n(2).equals(n(2)), true)]
#[case::not_equal(n(2).not_equals(n(2)), false)]
#[case::rationals_compare_exactly((n(1) / n(3)).less(decimal("0.34")), true)]
#[case::integer_equals_decimal(n(1).equals(decimal("1.0")), true)]
#[case::booleans_equal(truth(true).equals(truth(true)), true)]
#[case::booleans_differ(truth(true).equals(truth(false)), false)]
#[case::booleans_not_equal(truth(true).not_equals(truth(false)), true)]
#[case::negation(!truth(false), true)]
#[case::conjunction(n(1).less(n(2)).and(n(2).less(n(3))), true)]
#[case::disjunction(n(5).less(n(2)).or(n(2).less(n(1))), false)]
#[case::bound_check(n(3).greater_equal(n(0)).and(n(3).less(n(10))), true)]
fn comparisons_and_logic_decide_to_a_boolean(
    #[case] expression: Expression,
    #[case] expected: bool,
) {
    assert_eq!(folded(&expression), Some(truth(expected)));
}

#[test]
fn piecewise_takes_the_first_true_case() {
    let piecewise = Expression::piecewise(
        [(n(1).greater(n(2)), n(10)), (n(1).less(n(2)), n(20))],
        n(30),
    )
    .expect("a piecewise");

    assert_eq!(folded(&piecewise), Some(n(20)));
}

#[test]
fn piecewise_takes_the_otherwise_when_no_case_holds() {
    let piecewise =
        Expression::piecewise([(n(1).greater(n(2)), n(10))], n(1) / n(2)).expect("a piecewise");

    assert_eq!(folded(&piecewise), Some(decimal("0.5")));
}

#[test]
fn piecewise_declines_when_a_branch_it_does_not_take_does_not_fold() {
    let (_, x) = build_identifier("x");
    let free = Expression::piecewise([(n(1).less(n(2)), n(10))], x).expect("a piecewise");
    let undefined = Expression::piecewise([(n(1).less(n(2)), n(10))], n(1).floor_mod(n(0)))
        .expect("a piecewise");

    // SymPy lowers every branch, and fails on the modulo by zero in one it
    // does not take, so the fold leaves both to it.
    assert_eq!((folded(&free), folded(&undefined)), (None, None));
}

#[test]
fn piecewise_of_a_free_condition_declines() {
    let (_, x) = build_identifier("x");
    let piecewise = Expression::piecewise([(x.less(n(2)), n(10))], n(20)).expect("a piecewise");

    assert_eq!(folded(&piecewise), None);
}

#[test]
fn piecewise_with_a_boolean_value_folds_to_it() {
    let piecewise = Expression::piecewise([(n(1).less(n(2)), n(3).less(n(2)))], truth(true))
        .expect("a piecewise");

    assert_eq!(folded(&piecewise), Some(truth(false)));
}

// ---------------------------------------------------------------------------
// Built-in functions
// ---------------------------------------------------------------------------

#[rstest]
#[case::floor(call(BuiltinFunction::Floor, [n(7) / n(2)]), n(3))]
#[case::floor_of_a_negative(call(BuiltinFunction::Floor, [n(-7) / n(2)]), n(-4))]
#[case::ceil(call(BuiltinFunction::Ceil, [n(7) / n(2)]), n(4))]
#[case::ceil_of_a_negative(call(BuiltinFunction::Ceil, [n(-7) / n(2)]), n(-3))]
#[case::ceil_of_an_integer(call(BuiltinFunction::Ceil, [n(5)]), n(5))]
#[case::round_of_an_integer(call(BuiltinFunction::Round, [n(5)]), n(5))]
#[case::sqrt_of_a_perfect_square(call(BuiltinFunction::Sqrt, [n(16)]), n(4))]
#[case::sqrt_of_a_rational_square(call(BuiltinFunction::Sqrt, [n(1) / n(4)]), decimal("0.5"))]
#[case::sqrt_of_zero(call(BuiltinFunction::Sqrt, [n(0)]), n(0))]
#[case::exp2(call(BuiltinFunction::Exp2, [n(10)]), n(1024))]
#[case::exp2_of_a_negative(call(BuiltinFunction::Exp2, [n(-3)]), decimal("0.125"))]
#[case::log2(call(BuiltinFunction::Log2, [n(8)]), n(3))]
#[case::log2_of_a_unit_fraction(call(BuiltinFunction::Log2, [n(1) / n(8)]), n(-3))]
#[case::log10(call(BuiltinFunction::Log10, [n(1000)]), n(3))]
#[case::log_of_one(call(BuiltinFunction::Log, [n(1)]), n(0))]
#[case::exp_of_zero(call(BuiltinFunction::Exp, [n(0)]), n(1))]
#[case::sin_of_zero(call(BuiltinFunction::Sin, [n(0)]), n(0))]
#[case::cos_of_zero(call(BuiltinFunction::Cos, [n(0)]), n(1))]
fn built_ins_fold_where_the_value_is_exact(
    #[case] expression: Expression,
    #[case] expected: Expression,
) {
    assert_eq!(folded(&expression), Some(expected));
}

#[rstest]
#[case::xor(call(BuiltinFunction::Xor, [truth(true), truth(false)]))]
#[case::nand(call(BuiltinFunction::Nand, [truth(true), truth(true)]))]
#[case::nor(call(BuiltinFunction::Nor, [truth(false), truth(false)]))]
#[case::implies(call(BuiltinFunction::Implies, [truth(true), truth(false)]))]
#[case::iff(call(BuiltinFunction::Iff, [truth(false), truth(false)]))]
fn the_default_simplifier_declines_a_boolean_built_in_sympy_refuses(
    #[case] expression: Expression,
) {
    assert_eq!(folded(&expression), None);
}

/// The composed built-ins are not folded by default: `SymPy` refuses them
/// until they are inlined, so the default pipeline declines them, and they
/// are the opt-in `ComposedBuiltins`.
#[rstest]
#[case::max(call(BuiltinFunction::Max, [n(2), n(5)]), n(5))]
#[case::min(call(BuiltinFunction::Min, [n(2), n(5)]), n(2))]
#[case::min_of_rationals(call(BuiltinFunction::Min, [n(1) / n(2), n(1) / n(3)]), quotient(1, 3))]
#[case::abs(call(BuiltinFunction::Abs, [n(-5)]), n(5))]
#[case::abs_of_a_rational(call(BuiltinFunction::Abs, [n(-1) / n(2)]), decimal("0.5"))]
#[case::sign(call(BuiltinFunction::Sign, [n(-5)]), n(-1))]
#[case::sign_of_zero(call(BuiltinFunction::Sign, [n(0)]), n(0))]
#[case::clamp(call(BuiltinFunction::Clamp, [n(15), n(0), n(10)]), n(10))]
#[case::clamp_symmetric(call(BuiltinFunction::ClampSymmetric, [n(-15), n(10)]), n(-10))]
#[case::relu(call(BuiltinFunction::Relu, [n(-3)]), n(0))]
#[case::leaky_relu(call(BuiltinFunction::LeakyRelu, [n(-4), n(1) / n(2)]), n(-2))]
fn the_default_simplifier_declines_a_composed_built_in_and_the_opt_in_strategy_folds_it(
    #[case] expression: Expression,
    #[case] expected: Expression,
) {
    let context = SimplifyContext::default();
    let extended = GroundSimplifier::new().with_strategy(ComposedBuiltins::new());

    assert_eq!(folded(&expression), None);
    assert_eq!(extended.try_simplify(&expression, &context), Some(expected));
}

// ---------------------------------------------------------------------------
// Declining
// ---------------------------------------------------------------------------

#[rstest]
#[case::free_identifier(build_identifier("x").1)]
#[case::free_identifier_in_a_sum(build_identifier("x").1 + n(1))]
#[case::free_identifier_beside_ground(n(1) + n(2) + build_identifier("x").1)]
#[case::float(float(1.5))]
#[case::float_sum(float(1.5) + n(1))]
#[case::float_comparison(float(1.5).less(n(2)))]
#[case::division_by_zero(n(1) / n(0))]
#[case::zero_over_zero(n(0) / n(0))]
#[case::floor_division_by_zero(n(1).floor_divide(n(0)))]
#[case::modulo_by_zero(n(1).floor_mod(n(0)))]
#[case::zero_to_a_negative_power(n(0).power(n(-1)))]
#[case::irrational_root(n(2).power(n(1) / n(2)))]
#[case::partly_exact_root((n(4) / n(3)).power(n(1) / n(2)))]
#[case::root_of_a_negative(n(-4).power(n(1) / n(2)))]
#[case::huge_power(n(3).power(n(1_000_000)))]
#[case::huge_exponent(n(3).power(n(1_000_000_000_000)))]
#[case::sqrt_of_a_non_square(call(BuiltinFunction::Sqrt, [n(2)]))]
#[case::sqrt_of_a_negative(call(BuiltinFunction::Sqrt, [n(-4)]))]
#[case::log2_of_a_non_power(call(BuiltinFunction::Log2, [n(6)]))]
#[case::log2_of_zero(call(BuiltinFunction::Log2, [n(0)]))]
#[case::log_of_two(call(BuiltinFunction::Log, [n(2)]))]
#[case::exp_of_one(call(BuiltinFunction::Exp, [n(1)]))]
#[case::sin_of_one(call(BuiltinFunction::Sin, [n(1)]))]
#[case::round_of_a_rational(call(BuiltinFunction::Round, [n(1) / n(2)]))]
#[case::sigmoid(call(BuiltinFunction::Sigmoid, [n(0)]))]
#[case::silu(call(BuiltinFunction::Silu, [n(0)]))]
#[case::gelu(call(BuiltinFunction::Gelu, [n(0)]))]
#[case::user_function(Expression::call("custom".parse::<fhy_core::expression::Callee>().expect("a name"), [n(1)]))]
#[case::boolean_in_arithmetic(truth(true) + n(1))]
#[case::number_in_a_conjunction(Expression::new_logical(LogicalOperation::And, [n(1), truth(true)]))]
#[case::boolean_compared_with_a_number(truth(true).equals(n(1)))]
#[case::ordering_booleans(truth(true).less(truth(false)))]
#[case::builtin_constant(Expression::from(BuiltinConstant::Pi.identifier().clone()))]
#[case::builtin_constant_comparison(
    Expression::from(BuiltinConstant::Pi.identifier().clone()).greater(n(3))
)]
#[case::infinity(Expression::from(BuiltinConstant::Inf.identifier().clone()) + n(1))]
fn it_declines_what_it_cannot_match_sympy_on(#[case] expression: Expression) {
    assert_eq!(folded(&expression), None);
}

#[test]
fn simplify_returns_the_expression_it_declines_unchanged() {
    let (_, x) = build_identifier("x");
    let expression = x + n(1);

    let simplified = GroundSimplifier::new()
        .simplify(&expression, &SimplifyContext::default())
        .expect("never fails");

    assert!(Expression::ptr_eq(&simplified, &expression));
}

#[test]
fn a_deep_tree_declines_rather_than_overflowing_the_stack() {
    let mut deep = n(1);
    for _ in 0..100_000 {
        deep = deep + n(1);
    }

    assert_eq!(folded(&deep), None);
}

#[test]
fn a_shared_subtree_is_folded_once() {
    let mut shared = n(1);
    for _ in 0..200 {
        shared = &shared + &shared;
    }

    let expected = "1606938044258990275541962092341162602522202993782792835301376"
        .parse::<fhy_core::expression::BigInt>()
        .expect("digits");
    assert_eq!(folded(&shared), Some(build_literal(expected)));
}

// ---------------------------------------------------------------------------
// The context
// ---------------------------------------------------------------------------

#[test]
fn a_registered_constant_folds_to_its_value() {
    let mut registry = FunctionRegistry::new();
    let answer = registry
        .register_constant(
            NativeConstant::new(
                FunctionName::new("answer").expect("a name"),
                FunctionSort::Int,
                42,
            )
            .expect("a constant"),
        )
        .expect("registered");
    let expression = Expression::from(answer) + n(1);

    let result = GroundSimplifier::new()
        .try_simplify(&expression, &SimplifyContext::from_registry(&registry));

    assert_eq!(result, Some(n(43)));
}

#[test]
fn a_registered_float_constant_declines() {
    let mut registry = FunctionRegistry::new();
    let half = registry
        .register_constant(
            NativeConstant::new(
                FunctionName::new("half").expect("a name"),
                FunctionSort::Real,
                0.5,
            )
            .expect("a constant"),
        )
        .expect("registered");

    let result = GroundSimplifier::new().try_simplify(
        &Expression::from(half),
        &SimplifyContext::from_registry(&registry),
    );

    assert_eq!(result, None);
}

#[test]
fn a_constant_without_a_registry_declines() {
    let mut registry = FunctionRegistry::new();
    let answer = registry
        .register_constant(
            NativeConstant::new(
                FunctionName::new("answer").expect("a name"),
                FunctionSort::Int,
                42,
            )
            .expect("a constant"),
        )
        .expect("registered");

    assert_eq!(folded(&Expression::from(answer)), None);
}

#[test]
fn the_solver_substitutes_the_environment_and_then_folds() {
    let (x, reference) = build_identifier("x");
    let solver = Solver::new().with_simplifier(GroundSimplifier::new());
    let bound = reference.greater_equal(n(0));

    let holds = solver
        .simplify(
            &bound,
            &HashMap::from([(x.clone(), n(3))]),
            &SimplifyContext::default(),
        )
        .expect("simplified");
    let fails = solver
        .simplify(
            &bound,
            &HashMap::from([(x, n(-3))]),
            &SimplifyContext::default(),
        )
        .expect("simplified");
    let free = solver
        .simplify(&bound, &HashMap::new(), &SimplifyContext::default())
        .expect("simplified");

    assert_eq!((holds, fails, free), (truth(true), truth(false), bound));
}

// ---------------------------------------------------------------------------
// The composition
// ---------------------------------------------------------------------------

#[test]
fn the_chain_answers_a_ground_expression_without_asking_the_fallback() {
    let fallback = RecordingSimplifier::returning(n(99));
    let chain = GroundWithFallback::from_shared(Arc::clone(&fallback) as Arc<dyn Simplifier>);

    let result = chain
        .simplify(&(n(2) + n(3)), &SimplifyContext::default())
        .expect("simplified");

    assert_eq!(result, n(5));
    assert!(fallback.inputs().is_empty());
}

#[test]
fn the_chain_asks_the_fallback_where_the_ground_simplifier_declines() {
    let (_, x) = build_identifier("x");
    let fallback = RecordingSimplifier::returning(n(99));
    let chain = GroundWithFallback::from_shared(Arc::clone(&fallback) as Arc<dyn Simplifier>);
    let expression = x + n(1);

    let result = chain
        .simplify(&expression, &SimplifyContext::default())
        .expect("simplified");

    assert_eq!(result, n(99));
    assert_eq!(fallback.inputs(), vec![expression]);
}

#[test]
fn the_chain_passes_the_context_on() {
    let (_, x) = build_identifier("x");
    let fallback = RecordingSimplifier::identity();
    let chain = GroundWithFallback::from_shared(Arc::clone(&fallback) as Arc<dyn Simplifier>);
    let registry = FunctionRegistry::new();

    chain
        .simplify(&x, &SimplifyContext::from_registry(&registry))
        .expect("simplified");

    assert_eq!(fallback.registry_sizes(), vec![Some(0)]);
}

#[test]
fn the_chain_reports_the_fallback_s_failure_unchanged() {
    let (_, x) = build_identifier("x");
    let fallback = RecordingSimplifier::failing("boom");
    let chain = GroundWithFallback::from_shared(fallback as Arc<dyn Simplifier>);

    let error: BoxError = chain
        .simplify(&x, &SimplifyContext::default())
        .expect_err("the fallback fails");

    assert_eq!(
        error.downcast_ref::<FakeBackendError>(),
        Some(&FakeBackendError("boom".to_owned()))
    );
}

#[test]
fn the_chain_is_named_for_both_backends() {
    let chain =
        GroundWithFallback::from_shared(RecordingSimplifier::identity() as Arc<dyn Simplifier>);

    assert_eq!(chain.name(), Cow::Borrowed("ground+recording"));
    assert_eq!(GroundSimplifier::new().name(), Cow::Borrowed("ground"));
}

#[test]
fn a_solver_holding_the_chain_names_the_failing_backend() {
    let (_, x) = build_identifier("x");
    let fallback = RecordingSimplifier::failing("boom");
    let solver = Solver::new().with_simplifier(GroundWithFallback::from_shared(fallback));

    let error = solver
        .simplify(&x, &HashMap::new(), &SimplifyContext::default())
        .expect_err("the fallback fails");

    let SolveError::Backend { backend, .. } = error else {
        panic!("expected a backend error, got {error:?}");
    };
    assert_eq!(backend, "ground+recording");
}
