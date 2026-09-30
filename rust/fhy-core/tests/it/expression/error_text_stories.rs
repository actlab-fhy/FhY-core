//! The text and the source of every variant of the errors of
//! `expression/error.rs`, `expression/registry/error.rs` and
//! `expression/evaluate/error.rs`.

use std::error::Error;

use fhy_core::expression::builtins::{BuiltinConstant, BuiltinFunction};
use fhy_core::expression::evaluate::{EvaluationError, FoldError, LaneFailure, NearMiss};
use fhy_core::expression::registry::{
    FunctionDefinitionError, InlineError, NativeConstant, RegistrationError,
};
use fhy_core::expression::{
    BigInt, BooleanPosition, BooleanScreen, Callee, Decimal, Expression, FunctionName,
    FunctionSort, LiteralValue, LogicalOperation, PiecewiseError, RebuildError,
};
use rstest::rstest;

use crate::support::error_text::{Source, assert_error_text, fixed, test_error};

fn name(text: &str) -> FunctionName {
    FunctionName::new(text).expect("the test names no built-in")
}

fn x() -> Expression {
    Expression::from(fixed(7, "x"))
}

fn decimal(text: &str) -> Decimal {
    text.parse().expect("a decimal text")
}

/// Return the screen's refusal of `x && 1` over an integer `x`.
fn ill_typed() -> fhy_core::expression::NonBooleanLogicalOperandError {
    BooleanScreen::new()
        .check_predicate(&Expression::all([x().greater(0), Expression::from(1)]))
        .expect_err("a number is no operand of a connective")
}

// =============================================================================
// expression/error.rs
// =============================================================================

#[rstest]
#[case::no_cases(PiecewiseError::NoCases, "piecewise has no cases")]
#[case::non_boolean_condition(
    PiecewiseError::NonBooleanConditionLiteral { case_index: 2 },
    "condition of piecewise case 2 is a non-boolean literal"
)]
fn piecewise_error_text(#[case] error: PiecewiseError, #[case] text: &str) {
    assert_error_text(&error, text, Source::None);
}

#[rstest]
#[case::child_count(
    RebuildError::ChildCount { expected: 2, actual: 3 },
    "expected 2 children, got 3",
    Source::None
)]
#[case::piecewise(
    RebuildError::Piecewise(PiecewiseError::NoCases),
    "invalid piecewise",
    Source::Piecewise
)]
fn rebuild_error_text(#[case] error: RebuildError, #[case] text: &str, #[case] source: Source) {
    assert_error_text(&error, text, source);
}

#[rstest]
#[case::negated_operand(BooleanPosition::NegatedOperand, "the operand of a logical not")]
#[case::logical_and(
    BooleanPosition::LogicalOperand { operation: LogicalOperation::And, operand_index: 1 },
    "operand 1 of a logical and"
)]
#[case::logical_or(
    BooleanPosition::LogicalOperand { operation: LogicalOperation::Or, operand_index: 0 },
    "operand 0 of a logical or"
)]
#[case::case_condition(
    BooleanPosition::CaseCondition { case_index: 3 },
    "the condition of piecewise case 3"
)]
#[case::case_value(BooleanPosition::CaseValue { case_index: 0 }, "the value of piecewise case 0")]
#[case::otherwise(BooleanPosition::Otherwise, "the otherwise branch of a piecewise")]
fn boolean_position_text(#[case] position: BooleanPosition, #[case] text: &str) {
    assert_eq!(position.to_string(), text);
}

#[test]
fn non_boolean_operand_error_text() {
    let in_a_connective = ill_typed();
    let at_the_root = BooleanScreen::new()
        .check_predicate(&Expression::from(1))
        .expect_err("a number is no predicate");

    assert_error_text(
        &in_a_connective,
        "operand 1 of a logical and provably denotes a number but sits in a boolean position",
        Source::None,
    );
    assert_eq!(in_a_connective.operand(), &Expression::from(1));
    assert_error_text(
        &at_the_root,
        "the predicate provably denotes a number",
        Source::None,
    );
    assert!(at_the_root.parent().is_none());
}

// =============================================================================
// expression/registry/error.rs
// =============================================================================

#[rstest]
#[case::one_parameter_two_sorts(
    FunctionDefinitionError::SortCountMismatch { function: name("f"), parameters: 1, sorts: 2 },
    r#"function "f" has 1 parameter but 2 parameter sorts"#
)]
#[case::two_parameters_one_sort(
    FunctionDefinitionError::SortCountMismatch { function: name("f"), parameters: 2, sorts: 1 },
    r#"function "f" has 2 parameters but 1 parameter sort"#
)]
#[case::repeated_parameter(
    FunctionDefinitionError::RepeatedParameter { function: name("f"), parameter: fixed(7, "x") },
    r#"function "f" repeats the parameter "x""#
)]
fn function_definition_error_text(#[case] error: FunctionDefinitionError, #[case] text: &str) {
    assert_error_text(&error, text, Source::None);
}

#[test]
fn constant_value_error_text() {
    let error = NativeConstant::new(name("c"), FunctionSort::Nat, -1)
        .expect_err("a negative integer is no natural");

    assert_error_text(
        &error,
        r#"constant "c" of sort nat cannot hold -1"#,
        Source::None,
    );
    assert_eq!(error.name(), &name("c"));
    assert_eq!(error.sort(), FunctionSort::Nat);
    assert_eq!(error.value(), &LiteralValue::from(-1));
}

#[rstest]
#[case::name_taken(
    RegistrationError::NameTaken(name("f")),
    r#""f" is already registered"#
)]
#[case::builtin_constant_name(
    RegistrationError::BuiltinConstantName(BuiltinConstant::Pi),
    r#""pi" is the name of a built-in constant"#
)]
#[case::captured_identifiers(
    RegistrationError::CapturedIdentifiers {
        function: name("f"),
        identifiers: vec![fixed(7, "x"), fixed(8, "y")],
    },
    r#"function "f" captures identifiers that are not its parameters: x, y"#
)]
fn registration_error_text(#[case] error: RegistrationError, #[case] text: &str) {
    assert_error_text(&error, text, Source::None);
}

#[rstest]
#[case::unknown_function(
    InlineError::UnknownFunction(name("f")),
    r#"no function is registered under "f""#,
    Source::None
)]
#[case::arity_one(
    InlineError::ArityMismatch { callee: Callee::from(name("f")), expected: 1, actual: 2 },
    r#""f" takes 1 argument but the call passes 2"#,
    Source::None
)]
#[case::arity_two(
    InlineError::ArityMismatch { callee: Callee::Builtin(BuiltinFunction::Max), expected: 2, actual: 1 },
    r#""max" takes 2 arguments but the call passes 1"#,
    Source::None
)]
#[case::not_callable(
    InlineError::NotCallable(name("c")),
    r#""c" is a constant, not a function"#,
    Source::None
)]
#[case::recursive(
    InlineError::Recursive(name("f")),
    r#"function "f" is recursive and cannot be inlined"#,
    Source::None
)]
#[case::piecewise(
    InlineError::Piecewise(PiecewiseError::NonBooleanConditionLiteral { case_index: 0 }),
    "inlining built an invalid piecewise",
    Source::Piecewise
)]
fn inline_error_text(#[case] error: InlineError, #[case] text: &str, #[case] source: Source) {
    assert_error_text(&error, text, source);
}

// =============================================================================
// expression/evaluate/error.rs
// =============================================================================

#[rstest]
#[case::integer_overflow(LaneFailure::IntegerOverflow, "integer overflow")]
#[case::division_by_zero(LaneFailure::DivisionByZero, "integer division by zero")]
#[case::negative_integer_exponent(
    LaneFailure::NegativeIntegerExponent,
    "an integer raised to a negative integer power"
)]
#[case::non_finite_cast(LaneFailure::NonFiniteCast, "a non-finite value cast to an integer")]
#[case::out_of_range_cast(
    LaneFailure::OutOfRangeCast,
    "a value outside the 64-bit range cast to an integer"
)]
fn lane_failure_text(#[case] failure: LaneFailure, #[case] text: &str) {
    assert_eq!(failure.to_string(), text);
}

#[rstest]
#[case::inline(
    EvaluationError::Inline(InlineError::Recursive(name("f"))),
    r#"function "f" is recursive and cannot be inlined"#.to_owned(),
    Source::None
)]
#[case::inline_with_a_source(
    EvaluationError::Inline(InlineError::Piecewise(PiecewiseError::NoCases)),
    "inlining built an invalid piecewise".to_owned(),
    Source::Piecewise
)]
#[case::ill_typed(
    EvaluationError::IllTyped(ill_typed()),
    "operand 1 of a logical and provably denotes a number but sits in a boolean position"
        .to_owned(),
    Source::None
)]
#[case::bound_native_constant(
    EvaluationError::BoundNativeConstant(vec![BuiltinConstant::Pi.identifier().clone(), fixed(7, "c")]),
    format!(
        "cannot bind the constants pi::{}, c::7: a constant's value is fixed",
        BuiltinConstant::Pi.identifier().id()
    ),
    Source::None
)]
#[case::unbound(
    EvaluationError::Unbound { identifier: fixed(7, "x"), near_miss: None },
    r#"identifier "x" is not bound"#.to_owned(),
    Source::None
)]
#[case::unbound_naming_a_function(
    EvaluationError::Unbound { identifier: fixed(7, "sin"), near_miss: Some(NearMiss::NamesFunction) },
    r#"identifier "sin" is not bound; it names a function, so call it as sin(...) or bind it"#
        .to_owned(),
    Source::None
)]
#[case::unbound_sharing_a_constant_name(
    EvaluationError::Unbound { identifier: fixed(7, "pi"), near_miss: Some(NearMiss::SharesConstantName) },
    r#"identifier "pi" is not bound; it shares its name with the constant "pi" but is not its identifier"#
        .to_owned(),
    Source::None
)]
#[case::inexact_decimal(
    EvaluationError::InexactDecimal(decimal("0.1")),
    "decimal 0.1 has no exact binary float".to_owned(),
    Source::None
)]
#[case::integer_out_of_range(
    EvaluationError::IntegerOutOfRange(BigInt::from(1_u64 << 63)),
    "integer 9223372036854775808 is outside the 64-bit range".to_owned(),
    Source::None
)]
#[case::unsupported(
    EvaluationError::Unsupported(Callee::from(name("f"))),
    r#"function "f" has no implementation the evaluator can run"#.to_owned(),
    Source::None
)]
#[case::boolean_arithmetic(
    EvaluationError::BooleanArithmetic(x() + Expression::from(LiteralValue::from(true))),
    "a boolean is used as a number in (x + true)".to_owned(),
    Source::None
)]
#[case::mixed_branches(
    EvaluationError::MixedBranches(
        Expression::piecewise([(x().greater(0), Expression::from(LiteralValue::from(true)))], 1)
            .expect("a valid piecewise"),
    ),
    "the branches of {true if (x > 0); 1 otherwise} mix booleans and numbers".to_owned(),
    Source::None
)]
#[case::shape(
    EvaluationError::Shape { left: vec![2], right: vec![3, 1] },
    "shapes [2] and [3, 1] do not broadcast".to_owned(),
    Source::None
)]
#[case::broadcast_too_large(
    EvaluationError::BroadcastTooLarge { shape: vec![1 << 40, 1 << 40] },
    "the broadcast shape [1099511627776, 1099511627776] has more lanes than an array can hold"
        .to_owned(),
    Source::None
)]
#[case::out_of_memory(
    EvaluationError::OutOfMemory { lanes: 1 << 40 },
    "cannot allocate the 1099511627776 lanes of the result".to_owned(),
    Source::None
)]
#[case::lane_of_a_scalar(
    EvaluationError::Lane { failure: LaneFailure::IntegerOverflow, node: x() * 2, lane: None },
    "integer overflow in (x * 2)".to_owned(),
    Source::None
)]
#[case::lane_of_an_array(
    EvaluationError::Lane { failure: LaneFailure::DivisionByZero, node: x().floor_divide(0), lane: Some(4) },
    "integer division by zero at lane 4 in (x // 0)".to_owned(),
    Source::None
)]
#[case::kernel(
    EvaluationError::Kernel { function: BuiltinFunction::Exp, source: test_error() },
    "the array kernel of exp failed".to_owned(),
    Source::TestValue
)]
fn evaluation_error_text(
    #[case] error: EvaluationError,
    #[case] text: String,
    #[case] source: Source,
) {
    assert_error_text(&error, &text, source);
}

/// Test an inlining error's source is the inlining error's own source, so
/// the chain skips the text `Display` already wrote.
#[test]
fn an_evaluation_inline_error_passes_on_the_inline_errors_source() {
    let error = EvaluationError::Inline(InlineError::Piecewise(PiecewiseError::NoCases));

    let source = error.source().expect("the piecewise refusal");

    assert_eq!(
        source.downcast_ref::<PiecewiseError>(),
        Some(&PiecewiseError::NoCases)
    );
}

#[rstest]
#[case::unknown_function(
    FoldError::UnknownFunction(name("f")),
    r#"no function is registered under "f""#,
    Source::None
)]
#[case::not_callable(
    FoldError::NotCallable(name("c")),
    r#""c" is a constant, not a function"#,
    Source::None
)]
#[case::arity_one(
    FoldError::Arity { callee: Callee::from(name("f")), expected: 1, actual: 0 },
    r#""f" takes 1 argument but the call passes 0"#,
    Source::None
)]
#[case::arity_two(
    FoldError::Arity { callee: Callee::from(name("f")), expected: 2, actual: 1 },
    r#""f" takes 2 arguments but the call passes 1"#,
    Source::None
)]
#[case::argument_sort(
    FoldError::ArgumentSort {
        callee: Callee::Builtin(BuiltinFunction::Sqrt),
        position: 0,
        sort: FunctionSort::Real,
        argument: LiteralValue::from(true),
    },
    r#"argument 0 of "sqrt" must be real, got true"#,
    Source::None
)]
#[case::result_sort(
    FoldError::ResultSort { function: name("f"), sort: FunctionSort::Int, value: LiteralValue::from(1.5) },
    r#"native function "f" returned 1.5, which is not of its result sort int"#,
    Source::None
)]
#[case::inexact_decimal(
    FoldError::InexactDecimal(decimal("0.1")),
    "decimal 0.1 has no exact binary float",
    Source::None
)]
#[case::non_finite_cast(
    FoldError::NonFiniteCast { function: BuiltinFunction::Floor, value: f64::INFINITY },
    "floor(inf) has no integer value",
    Source::None
)]
#[case::piecewise(
    FoldError::Piecewise(PiecewiseError::NonBooleanConditionLiteral { case_index: 1 }),
    "folding built an invalid piecewise",
    Source::Piecewise
)]
#[case::native(
    FoldError::Native { function: name("f"), source: test_error() },
    r#"native function "f" failed"#,
    Source::TestValue
)]
fn fold_error_text(#[case] error: FoldError, #[case] text: &str, #[case] source: Source) {
    assert_error_text(&error, text, source);
}
