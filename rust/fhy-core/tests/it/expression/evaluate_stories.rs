//! Tests for `Evaluator::evaluate` over scalars: the operations' semantics,
//! Booleans, connectives and piecewise, native built-ins, lane failures and
//! the guards that discard them, the refusals and their order, and depth.

use crate::support::expression as expression_support;
use crate::support::stack as stack_support;

use std::collections::HashMap;

use fhy_core::expression::builtins::{BuiltinConstant, BuiltinFunction};
use fhy_core::expression::evaluate::{EvaluationError, Evaluator, LaneFailure, NearMiss, Scalar};
use fhy_core::expression::registry::{
    FunctionDefinition, FunctionRegistry, InlineError, NativeConstant, NativeFunction,
};
use fhy_core::expression::{
    BigInt, BinaryOperation, Callee, Expression, FunctionName, FunctionSort,
};
use fhy_core::identifier::Identifier;
use rstest::rstest;

use expression_support::{build_decimal_literal, build_identifier, build_literal};
use stack_support::{SMALL_STACK_DEPTH, run_on_small_stack};

fn name(text: &str) -> FunctionName {
    FunctionName::new(text).expect("the test names no built-in")
}

fn call(
    function: impl Into<Callee>,
    arguments: impl IntoIterator<Item = Expression>,
) -> Expression {
    Expression::call(function, arguments)
}

fn binary(
    operation: BinaryOperation,
    left: impl Into<Expression>,
    right: impl Into<Expression>,
) -> Expression {
    Expression::new_binary(operation, left, right)
}

fn piecewise(cases: Vec<(Expression, Expression)>, otherwise: impl Into<Expression>) -> Expression {
    Expression::piecewise(cases, otherwise).expect("a valid piecewise")
}

fn evaluate_in(
    registry: &FunctionRegistry,
    expression: &Expression,
    bindings: &[(&Identifier, Scalar)],
) -> Result<Scalar, EvaluationError> {
    let environment: HashMap<Identifier, Scalar> = bindings
        .iter()
        .map(|(identifier, value)| ((*identifier).clone(), *value))
        .collect();
    Evaluator::new(registry).evaluate(expression, &environment)
}

fn evaluate(
    expression: &Expression,
    bindings: &[(&Identifier, Scalar)],
) -> Result<Scalar, EvaluationError> {
    evaluate_in(&FunctionRegistry::new(), expression, bindings)
}

/// Evaluate `left operation right` over two bound scalars.
fn evaluate_binary(
    operation: BinaryOperation,
    left: Scalar,
    right: Scalar,
) -> Result<Scalar, EvaluationError> {
    let (a, a_reference) = build_identifier("a");
    let (b, b_reference) = build_identifier("b");
    evaluate(
        &binary(operation, a_reference, b_reference),
        &[(&a, left), (&b, right)],
    )
}

fn expect_lane_failure(result: Result<Scalar, EvaluationError>) -> LaneFailure {
    match result {
        Err(EvaluationError::Lane { failure, .. }) => failure,
        other => panic!("expected a lane failure, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Leaves
// ---------------------------------------------------------------------------

#[test]
fn evaluate_reads_literals_in_their_domains() {
    assert_eq!(
        evaluate(&build_literal(true), &[]).unwrap(),
        Scalar::Bool(true)
    );
    assert_eq!(evaluate(&build_literal(3), &[]).unwrap(), Scalar::Int(3));
    assert_eq!(
        evaluate(&build_literal(0.5), &[]).unwrap(),
        Scalar::Real(0.5)
    );
    assert_eq!(
        evaluate(&build_decimal_literal("0.25"), &[]).unwrap(),
        Scalar::Real(0.25)
    );
}

#[test]
fn evaluate_refuses_an_inexact_decimal_and_an_out_of_range_integer() {
    let inexact = evaluate(&build_decimal_literal("0.1"), &[]).expect_err("0.1");
    let big: BigInt = "9223372036854775808".parse().expect("digits");
    let out_of_range = evaluate(&build_literal(big), &[]).expect_err("beyond i64");

    assert_eq!(inexact.to_string(), "decimal 0.1 has no exact binary float");
    assert_eq!(
        out_of_range.to_string(),
        "integer 9223372036854775808 is outside the 64-bit range"
    );
}

#[test]
fn evaluate_reads_bindings_and_ignores_unreferenced_ones() {
    let (x, reference) = build_identifier("x");
    let (unused, _) = build_identifier("unused");

    let value = evaluate(
        &reference,
        &[(&x, Scalar::Real(2.0)), (&unused, Scalar::Bool(true))],
    );

    assert_eq!(value.unwrap(), Scalar::Real(2.0));
}

#[test]
fn evaluate_resolves_builtin_and_user_constants() {
    let mut registry = FunctionRegistry::new();
    let answer = registry
        .register_constant(
            NativeConstant::new(name("answer"), FunctionSort::Int, 42).expect("an integer"),
        )
        .expect("free name");
    let tree =
        Expression::from(BuiltinConstant::Pi.identifier().clone()) + Expression::from(answer);

    let value = evaluate_in(&registry, &tree, &[]).expect("both are constants");

    assert_eq!(value, Scalar::Real(std::f64::consts::PI + 42.0));
}

#[test]
fn evaluate_names_the_near_miss_of_an_unbound_identifier() {
    let mut registry = FunctionRegistry::new();
    registry
        .register_native_function(NativeFunction::new(
            name("softplus"),
            [FunctionSort::Real],
            FunctionSort::Real,
        ))
        .expect("free name");
    let near_miss = |hint: &str| match evaluate_in(&registry, &build_identifier(hint).1, &[]) {
        Err(EvaluationError::Unbound { near_miss, .. }) => near_miss,
        other => panic!("expected an unbound identifier, got {other:?}"),
    };

    assert_eq!(near_miss("x"), None);
    assert_eq!(near_miss("pi"), Some(NearMiss::SharesConstantName));
    assert_eq!(near_miss("relu"), Some(NearMiss::NamesFunction));
    assert_eq!(near_miss("softplus"), Some(NearMiss::NamesFunction));
    let error = evaluate_in(&registry, &build_identifier("relu").1, &[]).expect_err("unbound");
    assert_eq!(
        error.to_string(),
        "identifier \"relu\" is not bound; it names a function, so call it as relu(...) or bind it"
    );
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

#[rstest]
#[case::add(BinaryOperation::Add, 7, 3, 10)]
#[case::subtract(BinaryOperation::Subtract, 7, 3, 4)]
#[case::multiply(BinaryOperation::Multiply, 7, -3, -21)]
#[case::floor_divide(BinaryOperation::FloorDivide, 7, 2, 3)]
#[case::floor_divide_negative(BinaryOperation::FloorDivide, -7, 2, -4)]
#[case::floor_divide_negative_divisor(BinaryOperation::FloorDivide, 7, -2, -4)]
#[case::floor_divide_both_negative(BinaryOperation::FloorDivide, -7, -2, 3)]
#[case::floor_mod(BinaryOperation::FloorMod, 7, 3, 1)]
#[case::floor_mod_negative(BinaryOperation::FloorMod, -7, 3, 2)]
#[case::floor_mod_negative_divisor(BinaryOperation::FloorMod, 7, -3, -2)]
#[case::floor_mod_min(BinaryOperation::FloorMod, i64::MIN, -1, 0)]
#[case::power(BinaryOperation::Power, 3, 4, 81)]
#[case::power_zero(BinaryOperation::Power, 0, 0, 1)]
#[case::power_minus_one(BinaryOperation::Power, -1, 5_000_000_001, -1)]
fn evaluate_computes_integer_arithmetic_exactly(
    #[case] operation: BinaryOperation,
    #[case] left: i64,
    #[case] right: i64,
    #[case] expected: i64,
) {
    let value = evaluate_binary(operation, Scalar::Int(left), Scalar::Int(right));

    assert_eq!(value.unwrap(), Scalar::Int(expected));
}

#[rstest]
#[case::add(BinaryOperation::Add, i64::MAX, 1, LaneFailure::IntegerOverflow)]
#[case::subtract(BinaryOperation::Subtract, i64::MIN, 1, LaneFailure::IntegerOverflow)]
#[case::multiply(BinaryOperation::Multiply, i64::MAX, 2, LaneFailure::IntegerOverflow)]
#[case::floor_divide_min(BinaryOperation::FloorDivide, i64::MIN, -1, LaneFailure::IntegerOverflow)]
#[case::floor_divide_zero(BinaryOperation::FloorDivide, 1, 0, LaneFailure::DivisionByZero)]
#[case::floor_mod_zero(BinaryOperation::FloorMod, 1, 0, LaneFailure::DivisionByZero)]
#[case::power(BinaryOperation::Power, 2, 63, LaneFailure::IntegerOverflow)]
#[case::power_negative(BinaryOperation::Power, 2, -1, LaneFailure::NegativeIntegerExponent)]
fn evaluate_fails_the_lane_of_an_integer_error(
    #[case] operation: BinaryOperation,
    #[case] left: i64,
    #[case] right: i64,
    #[case] expected: LaneFailure,
) {
    let failure = expect_lane_failure(evaluate_binary(
        operation,
        Scalar::Int(left),
        Scalar::Int(right),
    ));

    assert_eq!(failure, expected);
}

#[test]
fn evaluate_negation_of_the_smallest_integer_overflows() {
    let (x, reference) = build_identifier("x");

    let failure = expect_lane_failure(evaluate(&-reference, &[(&x, Scalar::Int(i64::MIN))]));

    assert_eq!(failure, LaneFailure::IntegerOverflow);
}

#[test]
fn evaluate_divides_integers_to_a_real() {
    let value = evaluate_binary(BinaryOperation::Divide, Scalar::Int(7), Scalar::Int(2));
    let by_zero = evaluate_binary(BinaryOperation::Divide, Scalar::Int(1), Scalar::Int(0));

    assert_eq!(value.unwrap(), Scalar::Real(3.5));
    assert_eq!(by_zero.unwrap(), Scalar::Real(f64::INFINITY));
}

#[rstest]
#[case::positive(7.5, 2.0, 3.0, 1.5)]
#[case::negative_dividend(-7.5, 2.0, -4.0, 0.5)]
#[case::negative_divisor(7.5, -2.0, -4.0, -0.5)]
#[case::both_negative(-7.5, -2.0, 3.0, -1.5)]
#[case::inexact(1.0, 0.1, 9.0, 0.099_999_999_999_999_95)]
fn evaluate_floor_divides_reals_as_python_does(
    #[case] left: f64,
    #[case] right: f64,
    #[case] quotient: f64,
    #[case] remainder: f64,
) {
    let divided = evaluate_binary(
        BinaryOperation::FloorDivide,
        Scalar::Real(left),
        Scalar::Real(right),
    );
    let modulo = evaluate_binary(
        BinaryOperation::FloorMod,
        Scalar::Real(left),
        Scalar::Real(right),
    );

    assert_eq!(divided.unwrap(), Scalar::Real(quotient));
    assert_eq!(modulo.unwrap(), Scalar::Real(remainder));
}

#[test]
fn evaluate_floor_divides_a_real_by_zero_as_ieee_divides() {
    let divided = evaluate_binary(
        BinaryOperation::FloorDivide,
        Scalar::Real(1.0),
        Scalar::Real(0.0),
    );
    let modulo = evaluate_binary(
        BinaryOperation::FloorMod,
        Scalar::Real(1.0),
        Scalar::Real(0.0),
    );

    assert_eq!(divided.unwrap(), Scalar::Real(f64::INFINITY));
    assert!(matches!(modulo.unwrap(), Scalar::Real(value) if value.is_nan()));
}

#[test]
fn evaluate_promotes_an_integer_beside_a_real() {
    let sum = evaluate_binary(BinaryOperation::Add, Scalar::Int(1), Scalar::Real(0.5));
    let power = evaluate_binary(BinaryOperation::Power, Scalar::Int(2), Scalar::Real(-1.0));
    let equal = evaluate_binary(BinaryOperation::Equal, Scalar::Int(1), Scalar::Real(1.0));
    let less = evaluate_binary(BinaryOperation::Less, Scalar::Real(0.5), Scalar::Int(1));

    assert_eq!(sum.unwrap(), Scalar::Real(1.5));
    assert_eq!(power.unwrap(), Scalar::Real(0.5));
    assert_eq!(equal.unwrap(), Scalar::Bool(true));
    assert_eq!(less.unwrap(), Scalar::Bool(true));
}

#[rstest]
#[case::equal(BinaryOperation::Equal, false)]
#[case::not_equal(BinaryOperation::NotEqual, true)]
#[case::less(BinaryOperation::Less, true)]
#[case::less_equal(BinaryOperation::LessEqual, true)]
#[case::greater(BinaryOperation::Greater, false)]
#[case::greater_equal(BinaryOperation::GreaterEqual, false)]
fn evaluate_compares_integers(#[case] operation: BinaryOperation, #[case] expected: bool) {
    let value = evaluate_binary(operation, Scalar::Int(i64::MAX - 1), Scalar::Int(i64::MAX));

    assert_eq!(value.unwrap(), Scalar::Bool(expected));
}

#[test]
fn evaluate_compares_nan_as_ieee_does() {
    let equal = evaluate_binary(
        BinaryOperation::Equal,
        Scalar::Real(f64::NAN),
        Scalar::Real(f64::NAN),
    );
    let not_equal = evaluate_binary(
        BinaryOperation::NotEqual,
        Scalar::Real(f64::NAN),
        Scalar::Real(f64::NAN),
    );

    assert_eq!(equal.unwrap(), Scalar::Bool(false));
    assert_eq!(not_equal.unwrap(), Scalar::Bool(true));
}

// ---------------------------------------------------------------------------
// Booleans
// ---------------------------------------------------------------------------

#[test]
fn evaluate_compares_booleans_for_equality() {
    let equal = evaluate_binary(
        BinaryOperation::Equal,
        Scalar::Bool(true),
        Scalar::Bool(true),
    );
    let not_equal = evaluate_binary(
        BinaryOperation::NotEqual,
        Scalar::Bool(true),
        Scalar::Bool(false),
    );

    assert_eq!(equal.unwrap(), Scalar::Bool(true));
    assert_eq!(not_equal.unwrap(), Scalar::Bool(true));
}

#[rstest]
#[case::arithmetic(BinaryOperation::Add, Scalar::Bool(true), Scalar::Int(1))]
#[case::ordering(BinaryOperation::Less, Scalar::Bool(false), Scalar::Bool(true))]
#[case::mixed_equality(BinaryOperation::Equal, Scalar::Bool(true), Scalar::Int(1))]
fn evaluate_refuses_a_boolean_used_as_a_number(
    #[case] operation: BinaryOperation,
    #[case] left: Scalar,
    #[case] right: Scalar,
) {
    let error = evaluate_binary(operation, left, right).expect_err("a boolean is no number");

    assert!(matches!(error, EvaluationError::BooleanArithmetic(_)));
}

#[test]
fn evaluate_refuses_a_boolean_under_negation_and_as_a_native_argument() {
    let (p, reference) = build_identifier("p");

    let negation = evaluate(&-&reference, &[(&p, Scalar::Bool(true))]).expect_err("-true");
    let positive = evaluate(&reference.positive(), &[(&p, Scalar::Bool(true))]).expect_err("+true");
    let argument = evaluate(
        &call(BuiltinFunction::Exp, [reference]),
        &[(&p, Scalar::Bool(true))],
    )
    .expect_err("exp(true)");

    for error in [negation, positive, argument] {
        assert!(
            matches!(error, EvaluationError::BooleanArithmetic(_)),
            "{error:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Connectives and piecewise
// ---------------------------------------------------------------------------

#[test]
fn evaluate_reduces_connectives_in_order() {
    let (p, p_reference) = build_identifier("p");
    let (q, q_reference) = build_identifier("q");
    let conjunction = Expression::all([
        p_reference.clone(),
        q_reference.clone(),
        build_literal(true),
    ]);
    let disjunction = Expression::any([p_reference.clone(), q_reference]);
    let bindings = [(&p, Scalar::Bool(true)), (&q, Scalar::Bool(false))];

    assert_eq!(
        evaluate(&conjunction, &bindings).unwrap(),
        Scalar::Bool(false)
    );
    assert_eq!(
        evaluate(&disjunction, &bindings).unwrap(),
        Scalar::Bool(true)
    );
    assert_eq!(
        evaluate(&!p_reference, &bindings).unwrap(),
        Scalar::Bool(false)
    );
}

#[test]
fn evaluate_takes_the_first_piecewise_case_that_holds() {
    let (x, reference) = build_identifier("x");
    let tree = piecewise(
        vec![
            (reference.greater(10), build_literal(2)),
            (reference.greater(0), build_literal(1)),
            (reference.greater(0), build_literal(99)),
        ],
        0,
    );

    assert_eq!(
        evaluate(&tree, &[(&x, Scalar::Int(20))]).unwrap(),
        Scalar::Int(2)
    );
    assert_eq!(
        evaluate(&tree, &[(&x, Scalar::Int(5))]).unwrap(),
        Scalar::Int(1)
    );
    assert_eq!(
        evaluate(&tree, &[(&x, Scalar::Int(-5))]).unwrap(),
        Scalar::Int(0)
    );
}

#[test]
fn evaluate_mixes_integer_and_real_branches_as_reals() {
    let (x, reference) = build_identifier("x");
    let tree = piecewise(vec![(reference.greater(0), build_literal(1))], 0.5);

    assert_eq!(
        evaluate(&tree, &[(&x, Scalar::Int(1))]).unwrap(),
        Scalar::Real(1.0)
    );
}

#[test]
fn evaluate_refuses_a_piecewise_mixing_booleans_and_numbers() {
    let (x, reference) = build_identifier("x");
    let tree = piecewise(vec![(reference.greater(0), build_literal(true))], 0);

    let error = evaluate(&tree, &[(&x, Scalar::Int(1))]).expect_err("mixed branches");

    assert!(matches!(error, EvaluationError::MixedBranches(_)));
}

// ---------------------------------------------------------------------------
// Lane failures and their guards
// ---------------------------------------------------------------------------

#[test]
fn a_piecewise_discards_the_failure_of_a_branch_it_does_not_select() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let tree = piecewise(
        vec![(
            y_reference.not_equals(0),
            x_reference.floor_divide(&y_reference),
        )],
        0,
    );

    let guarded = evaluate(&tree, &[(&x, Scalar::Int(7)), (&y, Scalar::Int(0))]);
    let selected = evaluate(&tree, &[(&x, Scalar::Int(7)), (&y, Scalar::Int(2))]);

    assert_eq!(guarded.unwrap(), Scalar::Int(0));
    assert_eq!(selected.unwrap(), Scalar::Int(3));
}

#[test]
fn a_piecewise_guards_a_non_finite_cast() {
    let (x, reference) = build_identifier("x");
    let tree = piecewise(
        vec![(
            reference.greater_equal(0),
            call(
                BuiltinFunction::Floor,
                [call(BuiltinFunction::Sqrt, [reference])],
            ),
        )],
        0,
    );

    assert_eq!(
        evaluate(&tree, &[(&x, Scalar::Real(-4.0))]).unwrap(),
        Scalar::Int(0)
    );
    assert_eq!(
        evaluate(&tree, &[(&x, Scalar::Real(10.0))]).unwrap(),
        Scalar::Int(3)
    );
}

#[test]
fn a_piecewise_raises_the_failure_of_the_branch_it_selects() {
    let (x, reference) = build_identifier("x");
    let tree = piecewise(
        vec![(
            reference.less(0),
            call(
                BuiltinFunction::Floor,
                [call(BuiltinFunction::Sqrt, [reference])],
            ),
        )],
        0,
    );

    let error = evaluate(&tree, &[(&x, Scalar::Real(-4.0))]).expect_err("sqrt(-4) is NaN");

    assert!(matches!(
        &error,
        EvaluationError::Lane { failure: LaneFailure::NonFiniteCast, node, lane: None }
            if node.to_string() == "floor(sqrt(x))"
    ));
    assert_eq!(
        error.to_string(),
        "a non-finite value cast to an integer in floor(sqrt(x))"
    );
}

#[test]
fn a_failed_piecewise_condition_fails_its_lane_unless_an_earlier_case_holds() {
    let (x, reference) = build_identifier("x");
    let failing_condition = reference.floor_divide(0).greater(0);
    let tree = piecewise(
        vec![
            (reference.greater(5), build_literal(1)),
            (failing_condition, build_literal(2)),
        ],
        3,
    );

    let earlier = evaluate(&tree, &[(&x, Scalar::Int(10))]);
    let reached = evaluate(&tree, &[(&x, Scalar::Int(1))]);

    assert_eq!(earlier.unwrap(), Scalar::Int(1));
    assert_eq!(expect_lane_failure(reached), LaneFailure::DivisionByZero);
}

#[test]
fn a_nested_piecewise_is_guarded_by_its_outer_condition() {
    let (x, reference) = build_identifier("x");
    let inner = piecewise(vec![(reference.less(0), reference.floor_divide(0))], 1);
    let tree = piecewise(vec![(reference.greater(0), inner.clone())], 2);

    assert_eq!(
        evaluate(&tree, &[(&x, Scalar::Int(-1))]).unwrap(),
        Scalar::Int(2)
    );
    assert_eq!(
        expect_lane_failure(evaluate(&inner, &[(&x, Scalar::Int(-1))])),
        LaneFailure::DivisionByZero
    );
}

#[test]
fn a_conjunction_discards_a_failure_when_another_operand_is_false() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let tree = Expression::all([
        y_reference.not_equals(0),
        x_reference.floor_divide(&y_reference).greater(1),
    ]);
    let reversed = Expression::all([
        x_reference.floor_divide(&y_reference).greater(1),
        y_reference.not_equals(0),
    ]);

    assert_eq!(
        evaluate(&tree, &[(&x, Scalar::Int(7)), (&y, Scalar::Int(0))]).unwrap(),
        Scalar::Bool(false)
    );
    assert_eq!(
        evaluate(&reversed, &[(&x, Scalar::Int(7)), (&y, Scalar::Int(0))]).unwrap(),
        Scalar::Bool(false)
    );
    assert_eq!(
        evaluate(&tree, &[(&x, Scalar::Int(7)), (&y, Scalar::Int(2))]).unwrap(),
        Scalar::Bool(true)
    );
}

#[test]
fn a_disjunction_discards_a_failure_when_another_operand_is_true() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let tree = Expression::any([
        y_reference.equals(0),
        x_reference.floor_divide(&y_reference).greater(1),
    ]);
    let failing = Expression::any([
        y_reference.not_equals(0),
        x_reference.floor_divide(&y_reference).greater(1),
    ]);
    let bindings = [(&x, Scalar::Int(7)), (&y, Scalar::Int(0))];

    assert_eq!(evaluate(&tree, &bindings).unwrap(), Scalar::Bool(true));
    assert_eq!(
        expect_lane_failure(evaluate(&failing, &bindings)),
        LaneFailure::DivisionByZero
    );
}

#[test]
fn a_failure_passes_through_every_other_node() {
    let (x, reference) = build_identifier("x");
    let tree = (reference.floor_divide(0) + 1).greater(0);

    let error = evaluate(&tree, &[(&x, Scalar::Int(1))]).expect_err("the division fails");

    assert!(matches!(
        &error,
        EvaluationError::Lane { failure: LaneFailure::DivisionByZero, node, lane: None } if node.to_string() == "(x // 0)"
    ));
    assert_eq!(error.to_string(), "integer division by zero in (x // 0)");
}

// ---------------------------------------------------------------------------
// Native built-ins, inlining and the order of the checks
// ---------------------------------------------------------------------------

#[test]
fn evaluate_computes_native_builtins_and_casts_integer_results() {
    let (x, reference) = build_identifier("x");
    let bindings = [(&x, Scalar::Real(2.5))];

    let rounded = evaluate(
        &call(BuiltinFunction::Round, [reference.clone()]),
        &bindings,
    );
    let root = evaluate(&call(BuiltinFunction::Sqrt, [build_literal(-1.0)]), &[]);
    let erf = evaluate(&call(BuiltinFunction::Erf, [reference]), &bindings);
    let out_of_range = evaluate(&call(BuiltinFunction::Floor, [build_literal(1e300)]), &[]);

    assert_eq!(rounded.unwrap(), Scalar::Int(2));
    assert!(matches!(root.unwrap(), Scalar::Real(value) if value.is_nan()));
    assert_eq!(
        erf.unwrap(),
        Scalar::Real(BuiltinFunction::Erf.native_value(2.5).unwrap())
    );
    assert_eq!(
        expect_lane_failure(out_of_range),
        LaneFailure::OutOfRangeCast
    );
}

#[test]
fn evaluate_inlines_composed_builtins_and_user_functions() {
    let mut registry = FunctionRegistry::new();
    let (parameter, parameter_reference) = build_identifier("x");
    registry
        .register_function(
            FunctionDefinition::new(
                name("double"),
                [parameter],
                [FunctionSort::Real],
                FunctionSort::Real,
                parameter_reference * 2,
            )
            .expect("one sort"),
        )
        .expect("free name");
    let (x, reference) = build_identifier("x");
    let tree = call(name("double"), [call(BuiltinFunction::Gelu, [reference])]);

    let value = evaluate_in(&registry, &tree, &[(&x, Scalar::Real(1.0))]).expect("inlined");

    let erf = BuiltinFunction::Erf
        .native_value(1.0 / 2.0_f64.sqrt())
        .unwrap();
    let gelu = 1.0_f64.midpoint(erf);
    assert_eq!(value, Scalar::Real(gelu * 2.0));
}

#[test]
fn evaluate_refuses_a_native_user_function() {
    let mut registry = FunctionRegistry::new();
    registry
        .register_native_function(NativeFunction::new(
            name("softplus"),
            [FunctionSort::Real],
            FunctionSort::Real,
        ))
        .expect("free name");

    let error = evaluate_in(
        &registry,
        &call(name("softplus"), [build_literal(1.0)]),
        &[],
    )
    .expect_err("no implementation");

    assert!(
        matches!(&error, EvaluationError::Unsupported(Callee::Named(name)) if name.as_str() == "softplus")
    );
}

#[test]
fn evaluate_reports_an_inlining_error() {
    let error = evaluate(&call(name("nowhere"), [build_literal(1.0)]), &[]).expect_err("unknown");

    assert!(matches!(
        error,
        EvaluationError::Inline(InlineError::UnknownFunction(_))
    ));
}

#[test]
fn evaluate_screens_with_the_value_kinds_of_the_bindings() {
    let (x, reference) = build_identifier("x");
    let tree = Expression::all([reference, build_literal(true)]);

    let error = evaluate(&tree, &[(&x, Scalar::Int(1))]).expect_err("an integer operand");

    assert!(matches!(error, EvaluationError::IllTyped(_)));
}

#[test]
fn evaluate_refuses_a_bound_constant_it_refers_to_and_ignores_one_it_does_not() {
    let pi = BuiltinConstant::Pi.identifier().clone();
    let e = BuiltinConstant::E.identifier().clone();

    let error =
        evaluate(&Expression::from(pi.clone()), &[(&pi, Scalar::Real(3.0))]).expect_err("bound");
    let ignored = evaluate(&build_literal(1), &[(&e, Scalar::Real(3.0))]);

    assert!(
        matches!(&error, EvaluationError::BoundNativeConstant(identifiers) if identifiers == &[pi])
    );
    assert_eq!(
        error.to_string(),
        "cannot bind the constants pi::48: a constant's value is fixed"
    );
    assert_eq!(ignored.unwrap(), Scalar::Int(1));
}

#[test]
fn evaluate_screens_before_refusing_a_bound_constant() {
    let pi = BuiltinConstant::Pi.identifier().clone();
    let (x, reference) = build_identifier("x");
    let tree = Expression::all([reference, Expression::from(pi.clone()).greater(0)]);

    let error =
        evaluate(&tree, &[(&x, Scalar::Int(1)), (&pi, Scalar::Real(3.0))]).expect_err("both");

    assert!(matches!(error, EvaluationError::IllTyped(_)));
}

#[test]
fn prepare_exposes_the_inlined_expression_and_its_free_identifiers() {
    let registry = FunctionRegistry::new();
    let (x, reference) = build_identifier("x");

    let prepared = Evaluator::new(&registry)
        .prepare(&call(BuiltinFunction::Relu, [reference]))
        .expect("relu inlines");

    assert_eq!(
        prepared.expression().to_string(),
        "{x if ((x > 0) || (x != x)); 0 otherwise}"
    );
    assert_eq!(prepared.free_identifiers().len(), 1);
    assert!(prepared.free_identifiers().contains(&x));
}

#[test]
fn evaluate_walks_a_deep_tree_on_a_small_stack() {
    let value = run_on_small_stack(|| {
        let (x, reference) = build_identifier("x");
        let mut tree = reference;
        for _ in 0..SMALL_STACK_DEPTH {
            tree = tree + 1;
        }
        evaluate(&tree, &[(&x, Scalar::Int(0))])
    });

    assert_eq!(
        value.unwrap(),
        Scalar::Int(i64::try_from(SMALL_STACK_DEPTH).unwrap())
    );
}

/// Test a lane error on a 63-level doubling DAG over `x = 1` displays in
/// bounded size: the addition that overflows is written up to a fixed budget
/// of node occurrences, not once per path, and the node stays in the error.
#[test]
fn a_lane_error_on_a_63_level_doubling_dag_displays_in_bounded_size() {
    let (x, reference) = build_identifier("x");
    let dag = crate::support::expression::build_doubling_dag(&reference, 63);

    let error = evaluate(&dag, &[(&x, Scalar::Int(1))]).expect_err("2^63 overflows");

    let text = error.to_string();
    assert!(text.len() < 4096, "{} bytes", text.len());
    assert!(text.starts_with("integer overflow in ((((("), "{text}");
    assert!(text.ends_with('…'), "{text}");
    assert!(
        matches!(&error, EvaluationError::Lane { failure: LaneFailure::IntegerOverflow, node, lane: None } if Expression::ptr_eq(node, &dag)),
        "{error:?}"
    );
}

// =============================================================================
// Unary plus, Boolean piecewise, and the IEEE edges of floor division
// =============================================================================

/// Return whether two scalars are the same value, every NaN equal and the
/// zeros told apart by their sign.
fn is_same_scalar(left: Scalar, right: Scalar) -> bool {
    match (left, right) {
        (Scalar::Real(a), Scalar::Real(b)) => {
            a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
        }
        _ => left == right,
    }
}

/// Test `+x` is `x` for an integer and a real, and a piecewise with
/// Boolean branches selects a Boolean.
#[rstest]
#[case::plus_an_integer(Expression::from(3).positive(), Scalar::Int(3))]
#[case::plus_a_real(Expression::from(2.5).positive(), Scalar::Real(2.5))]
#[case::plus_a_negative_zero(Expression::from(-0.0).positive(), Scalar::Real(-0.0))]
#[case::boolean_piecewise_first_case(
    piecewise(vec![(build_literal(true), build_literal(false))], build_literal(true)),
    Scalar::Bool(false)
)]
#[case::boolean_piecewise_otherwise(
    piecewise(vec![(Expression::from(1).less(0), build_literal(false))], build_literal(true)),
    Scalar::Bool(true)
)]
fn unary_plus_and_boolean_piecewise_evaluate(#[case] tree: Expression, #[case] expected: Scalar) {
    let value = evaluate(&tree, &[]).expect("the tree evaluates");

    assert!(is_same_scalar(value, expected), "{value:?} != {expected:?}");
}

/// Test real floor division and modulo at the IEEE edges agree with
/// `NumPy`'s `floor_divide` and `mod` (2.x), the sign of a zero and every
/// NaN included.
#[rstest]
#[case::zero_by_negative_one(0.0, -1.0, -0.0, -0.0)]
#[case::negative_zero_by_one(-0.0, 1.0, -0.0, 0.0)]
#[case::five_by_infinity(5.0, f64::INFINITY, 0.0, 5.0)]
#[case::negative_five_by_infinity(-5.0, f64::INFINITY, -1.0, f64::INFINITY)]
#[case::five_by_negative_infinity(5.0, f64::NEG_INFINITY, -1.0, f64::NEG_INFINITY)]
#[case::infinity_by_one(f64::INFINITY, 1.0, f64::NAN, f64::NAN)]
#[case::one_by_zero(1.0, 0.0, f64::INFINITY, f64::NAN)]
#[case::negative_one_by_zero(-1.0, 0.0, f64::NEG_INFINITY, f64::NAN)]
#[case::zero_by_zero(0.0, 0.0, f64::NAN, f64::NAN)]
#[case::nan_by_one(f64::NAN, 1.0, f64::NAN, f64::NAN)]
#[case::one_by_nan(1.0, f64::NAN, f64::NAN, f64::NAN)]
#[case::tiny_by_huge(-1e-300, 1e300, -1.0, 1e300)]
#[case::seven_by_negative_zero(7.0, -0.0, f64::NEG_INFINITY, f64::NAN)]
fn real_floor_division_and_modulo_follow_numpy_at_the_ieee_edges(
    #[case] numerator: f64,
    #[case] denominator: f64,
    #[case] quotient: f64,
    #[case] remainder: f64,
) {
    let divide = evaluate_binary(
        BinaryOperation::FloorDivide,
        Scalar::Real(numerator),
        Scalar::Real(denominator),
    )
    .expect("real floor division never fails");
    let modulo = evaluate_binary(
        BinaryOperation::FloorMod,
        Scalar::Real(numerator),
        Scalar::Real(denominator),
    )
    .expect("real modulo never fails");

    assert!(
        is_same_scalar(divide, Scalar::Real(quotient)),
        "{numerator} // {denominator} = {divide:?}"
    );
    assert!(
        is_same_scalar(modulo, Scalar::Real(remainder)),
        "{numerator} % {denominator} = {modulo:?}"
    );
}

/// Test a number in a Boolean position is refused by the Boolean screen as
/// `IllTyped` before the walk runs, for a negation, a connective and a
/// piecewise condition alike: the walk never meets one.
#[rstest]
#[case::negation(|x: &Expression, _: &Expression| !x)]
#[case::connective(|x: &Expression, p: &Expression| Expression::all([x.clone(), p.clone()]))]
#[case::piecewise_condition(|x: &Expression, _: &Expression| piecewise(vec![(x.clone(), Expression::from(1))], 0))]
fn a_number_in_a_boolean_position_is_ill_typed(
    #[case] build: fn(&Expression, &Expression) -> Expression,
) {
    let (x, x_reference) = build_identifier("x");
    let (p, p_reference) = build_identifier("p");
    let tree = build(&x_reference, &p_reference);

    let error = evaluate(&tree, &[(&x, Scalar::Int(1)), (&p, Scalar::Bool(true))])
        .expect_err("a number is no Boolean");

    assert!(matches!(error, EvaluationError::IllTyped(_)), "{error:?}");
}
