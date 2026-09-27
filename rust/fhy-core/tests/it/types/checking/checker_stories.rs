//! Tests for `TypeChecker`: literals and weak types, identifiers and
//! qualifiers, the arithmetic, comparison, logical and piecewise rules, the
//! index-type algebra, calls, checking against an expected type, the error
//! frame, and depth.
//!
//! Ported from `tests/types/checking/test_type_checker.py`,
//! `test_type_checker_booleans.py` and `test_type_checker_sorts.py`; the
//! traceability table is in `docs/design/python-switch.md`, "S11b.2
//! implementation notes".

use crate::support::expression::build_call_or_panic;
use crate::support::stack::{SMALL_STACK_DEPTH, run_on_small_stack};
use crate::support::types::{array, index, literal_dimension, scalar};

use std::collections::HashMap;

use fhy_core::expression::builtins::BuiltinConstant;
use fhy_core::expression::registry::{FunctionDefinition, FunctionRegistry, NativeConstant};
use fhy_core::expression::{
    BigInt, Decimal, Expression, FunctionName, FunctionSort, LiteralValue, UnaryOperation,
};
use fhy_core::identifier::Identifier;
use fhy_core::types::checking::{TypeCheckError, TypeChecker, TypeRuleKind};
use fhy_core::types::{CoreDataType, DataType, TemplateDataType, Type, TypeQualifier};
use rstest::rstest;

use CoreDataType::{
    Bool, Complex32, Complex64, Complex128, Float, Float16, Float32, Float64, Int, Int8, Int16,
    Int32, Int64, Uint, Uint8, Uint16,
};

type Bindings = HashMap<Identifier, (Type, TypeQualifier)>;

/// Return the bindings of `pairs`, each a `Param`.
fn params(pairs: &[(&Identifier, Type)]) -> Bindings {
    pairs
        .iter()
        .map(|(identifier, value)| ((*identifier).clone(), (value.clone(), TypeQualifier::Param)))
        .collect()
}

fn reference(identifier: &Identifier) -> Expression {
    Expression::from(identifier.clone())
}

fn boolean(value: bool) -> Expression {
    Expression::from(LiteralValue::Bool(value))
}

fn decimal(text: &str) -> Expression {
    Expression::from(LiteralValue::Decimal(
        text.parse::<Decimal>().expect("a decimal"),
    ))
}

/// Synthesize `expression` over `bindings` and an empty registry.
fn synthesize(
    bindings: &Bindings,
    expression: &Expression,
) -> Result<(Type, TypeQualifier), TypeCheckError> {
    let registry = FunctionRegistry::new();
    TypeChecker::new(bindings, &registry, &registry).synthesize(expression)
}

/// Check `expression` against `expected` over `bindings`.
fn check(
    bindings: &Bindings,
    expression: &Expression,
    expected: &Type,
) -> Result<(Type, TypeQualifier), TypeCheckError> {
    let registry = FunctionRegistry::new();
    TypeChecker::new(bindings, &registry, &registry).check(expression, expected)
}

/// Return the type synthesized for `expression`.
fn type_of(bindings: &Bindings, expression: &Expression) -> Type {
    synthesize(bindings, expression)
        .expect("the expression checks")
        .0
}

/// Return the value of a check that succeeds.
fn ok<T>(result: Result<T, TypeCheckError>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("the check fails: {error}"),
    }
}

/// Return the kind and text of the rule `result` breaks.
fn rule_of<T: std::fmt::Debug>(result: Result<T, TypeCheckError>) -> (TypeRuleKind, String) {
    match result.expect_err("a rule is broken") {
        TypeCheckError::Rule { rule, .. } => (rule.kind(), rule.reason().to_owned()),
        other => panic!("a rule error, got {other}"),
    }
}

// =============================================================================
// Literals
// =============================================================================

#[rstest]
#[case(LiteralValue::from(1), Uint)]
#[case(LiteralValue::from(0), Uint)]
#[case(LiteralValue::from(-1), Int)]
#[case(LiteralValue::from(1.5), Float)]
#[case(LiteralValue::from(true), Bool)]
fn a_literal_synthesizes_its_weak_type(
    #[case] literal: LiteralValue,
    #[case] expected: CoreDataType,
) {
    assert_eq!(
        type_of(&Bindings::new(), &Expression::from(literal)),
        scalar(expected)
    );
    assert_eq!(CoreDataType::of_literal(&LiteralValue::from(7)), Ok(Uint));
}

#[test]
fn a_decimal_literal_is_unsupported() {
    let (kind, reason) = rule_of(synthesize(&Bindings::new(), &decimal("1.5")));

    assert_eq!(kind, TypeRuleKind::Unsupported);
    assert_eq!(reason, "decimal literals are not yet supported");
}

#[test]
fn a_decimal_literal_against_a_concrete_type_is_no_numeric_literal() {
    let (kind, reason) = rule_of(check(&Bindings::new(), &decimal("1.5"), &scalar(Float32)));

    assert_eq!(kind, TypeRuleKind::Literal);
    assert!(
        reason.starts_with("expected a numeric literal value"),
        "{reason}"
    );
}

#[test]
fn a_large_literal_stays_weak_without_a_context() {
    let huge = Expression::from(BigInt::from(1) << 200_u32);

    assert_eq!(type_of(&Bindings::new(), &huge), scalar(Uint));
    assert_eq!(type_of(&Bindings::new(), &(-huge.clone())), scalar(Int));
}

#[test]
fn negating_a_weak_literal_flips_its_sign_family() {
    assert_eq!(
        type_of(&Bindings::new(), &(-Expression::from(3))),
        scalar(Int)
    );
    assert_eq!(
        type_of(&Bindings::new(), &(-Expression::from(0))),
        scalar(Uint)
    );
    assert_eq!(
        type_of(&Bindings::new(), &(-Expression::from(2.5))),
        scalar(Float)
    );
}

#[rstest]
#[case(Expression::from(100), Int8, Int8)]
#[case(Expression::from(200), Uint8, Uint8)]
#[case(Expression::from(300), Int32, Int32)]
#[case(Expression::from(1), Float32, Float32)]
#[case(Expression::from(-1), Float64, Float64)]
#[case(boolean(true), Bool, Bool)]
fn checking_a_literal_against_a_concrete_type_resolves_it(
    #[case] literal: Expression,
    #[case] expected: CoreDataType,
    #[case] resolved: CoreDataType,
) {
    assert_eq!(
        ok(check(&Bindings::new(), &literal, &scalar(expected))),
        (scalar(resolved), TypeQualifier::Param)
    );
}

#[test]
fn checking_a_literal_against_a_weak_type_keeps_its_weak_type() {
    assert_eq!(
        ok(check(&Bindings::new(), &Expression::from(5), &scalar(Int)).map(|typed| typed.0)),
        scalar(Uint)
    );
    assert_eq!(
        ok(check(&Bindings::new(), &Expression::from(5), &scalar(Uint)).map(|typed| typed.0)),
        scalar(Uint)
    );
}

#[rstest]
#[case(Expression::from(256), Uint8, "is wider than the expected type uint8")]
#[case(Expression::from(-1), Uint16, "literal -1 is incompatible with uint16")]
#[case(Expression::from(-1), Uint, "is wider than the expected type uint[]")]
#[case(Expression::from(BigInt::from(1) << 70_u32), Int64, "does not fit in a supported int type")]
#[case(
    boolean(true),
    Int32,
    "boolean literal true is incompatible with int32"
)]
#[case(Expression::from(1), Bool, "incompatible with the bool context")]
fn checking_a_literal_the_expected_type_cannot_hold_is_refused(
    #[case] literal: Expression,
    #[case] expected: CoreDataType,
    #[case] phrase: &str,
) {
    let (_, reason) = rule_of(check(&Bindings::new(), &literal, &scalar(expected)));

    assert!(reason.contains(phrase), "{reason}");
}

#[test]
fn a_literal_cannot_be_checked_against_an_index_type() {
    let (kind, _) = rule_of(check(
        &Bindings::new(),
        &Expression::from(1),
        &index(0, 4, 1),
    ));

    assert_eq!(kind, TypeRuleKind::LiteralAgainstIndex);
}

#[test]
fn the_expected_type_reaches_literal_operands_of_arithmetic() {
    let expression = Expression::from(1) + Expression::from(2);

    assert_eq!(
        ok(check(&Bindings::new(), &expression, &scalar(Int16)).map(|typed| typed.0)),
        scalar(Int16)
    );
}

#[test]
fn a_weak_literal_operand_is_checked_against_a_concrete_other_operand() {
    let x = Identifier::new("x");
    let bindings = params(&[(&x, scalar(Int32))]);

    assert_eq!(type_of(&bindings, &(reference(&x) + 1)), scalar(Int32));
    assert_eq!(
        type_of(&bindings, &(Expression::from(1) + reference(&x))),
        scalar(Int32)
    );
    let (kind, reason) = rule_of(synthesize(
        &bindings,
        &(reference(&x) + (BigInt::from(1) << 40_u32)),
    ));
    assert_eq!(kind, TypeRuleKind::ExpectedType);
    assert!(
        reason.contains("wider than the expected type int32"),
        "{reason}"
    );
}

#[test]
fn a_literal_nested_below_a_negation_escapes_the_range_check() {
    let x = Identifier::new("x");
    let bindings = params(&[(&x, scalar(Int32))]);
    let expression = reference(&x) + (-Expression::from(BigInt::from(1) << 200_u32));

    assert_eq!(type_of(&bindings, &expression), scalar(Int32));
}

// =============================================================================
// Identifiers and qualifiers
// =============================================================================

#[test]
fn an_unbound_identifier_is_refused() {
    let x = Identifier::new("x");

    let (kind, reason) = rule_of(synthesize(&Bindings::new(), &reference(&x)));

    assert_eq!(kind, TypeRuleKind::UnboundIdentifier);
    assert_eq!(reason, format!("identifier `x::{}` is not bound", x.id()));
}

#[test]
fn an_output_identifier_cannot_be_read() {
    let x = Identifier::new("x");
    let bindings = HashMap::from([(x.clone(), (scalar(Int32), TypeQualifier::Output))]);

    let (kind, reason) = rule_of(synthesize(&bindings, &reference(&x)));

    assert_eq!(kind, TypeRuleKind::OutputRead);
    assert!(reason.contains("has type qualifier \"output\" and cannot be read from"));
}

#[rstest]
#[case(TypeQualifier::Input)]
#[case(TypeQualifier::State)]
#[case(TypeQualifier::Param)]
#[case(TypeQualifier::Temp)]
fn other_qualifiers_are_read_and_kept(#[case] qualifier: TypeQualifier) {
    let x = Identifier::new("x");
    let bindings = HashMap::from([(x.clone(), (scalar(Int32), qualifier))]);

    assert_eq!(
        ok(synthesize(&bindings, &reference(&x))),
        (scalar(Int32), qualifier)
    );
    assert_eq!(
        ok(synthesize(&bindings, &(-reference(&x)))),
        (scalar(Int32), qualifier)
    );
}

#[test]
fn qualifiers_promote_to_param_only_from_params() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let bindings = HashMap::from([
        (x.clone(), (scalar(Int32), TypeQualifier::Input)),
        (y.clone(), (scalar(Int32), TypeQualifier::Param)),
    ]);

    assert_eq!(
        ok(synthesize(&bindings, &(reference(&x) + reference(&x))).map(|typed| typed.1)),
        TypeQualifier::Temp
    );
    assert_eq!(
        ok(synthesize(&bindings, &(reference(&y) + 1)).map(|typed| typed.1)),
        TypeQualifier::Param
    );
}

#[test]
fn a_native_constant_types_by_its_sort_and_refuses_a_supplied_type() {
    let pi = BuiltinConstant::Pi.identifier();
    let mut registry = FunctionRegistry::new();
    let user = registry
        .register_constant(
            NativeConstant::new(
                FunctionName::new("tau_c").expect("a name"),
                FunctionSort::Real,
                6.5,
            )
            .expect("a real"),
        )
        .expect("registers");
    let none = Bindings::new();

    let checker = TypeChecker::new(&none, &registry, &registry);
    assert_eq!(
        ok(checker.synthesize(&reference(pi))),
        (scalar(Float64), TypeQualifier::Param)
    );
    assert_eq!(
        ok(checker.synthesize(&reference(&user))),
        (scalar(Float64), TypeQualifier::Param)
    );

    let supplied = params(&[(pi, scalar(Int32))]);
    let (kind, reason) =
        rule_of(TypeChecker::new(&supplied, &registry, &registry).synthesize(&reference(pi)));
    assert_eq!(kind, TypeRuleKind::SuppliedConstantType);
    assert!(reason.contains("names a native constant"));
}

#[test]
fn an_identifier_merely_named_like_a_constant_is_unbound() {
    let pi_named = Identifier::new("pi");

    assert_eq!(
        rule_of(synthesize(&Bindings::new(), &reference(&pi_named))).0,
        TypeRuleKind::UnboundIdentifier
    );
}

#[test]
fn a_tensor_identifier_is_unsupported_and_a_template_one_is_no_value_type() {
    let (t, x) = (Identifier::new("T"), Identifier::new("x"));
    let tensor = params(&[(&x, array(Int32, [literal_dimension(4)]))]);
    let templated = params(&[(&x, array(DataType::Template(TemplateDataType::new(t)), []))]);

    let (kind, reason) = rule_of(synthesize(&tensor, &reference(&x)));
    assert_eq!(kind, TypeRuleKind::Unsupported);
    assert!(
        reason.contains("resolves to tensor type int32[4]"),
        "{reason}"
    );
    let (kind, reason) = rule_of(synthesize(&templated, &reference(&x)));
    assert_eq!(kind, TypeRuleKind::NotAValueType);
    assert!(
        reason.contains("must resolve to a primitive numerical type"),
        "{reason}"
    );
}

// =============================================================================
// Arithmetic
// =============================================================================

#[rstest]
#[case(Int8, Int32, Int32)]
#[case(Uint16, Int16, Int32)]
#[case(Float16, Float32, Float32)]
#[case(Uint8, Uint8, Uint8)]
fn addition_promotes_the_operands(
    #[case] left: CoreDataType,
    #[case] right: CoreDataType,
    #[case] promoted: CoreDataType,
) {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let bindings = params(&[(&x, scalar(left)), (&y, scalar(right))]);

    assert_eq!(
        type_of(&bindings, &(reference(&x) + reference(&y))),
        scalar(promoted)
    );
}

#[rstest]
#[case(Int8, Int16, Float16)]
#[case(Int32, Int32, Float32)]
#[case(Int64, Uint8, Float64)]
#[case(Int, Uint, Float)]
#[case(Int32, Float16, Float32)]
#[case(Int16, Complex64, Complex64)]
#[case(Complex32, Float32, Complex64)]
#[case(Float64, Complex32, Complex128)]
fn true_division_produces_a_float_of_the_wider_width(
    #[case] left: CoreDataType,
    #[case] right: CoreDataType,
    #[case] quotient: CoreDataType,
) {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let bindings = params(&[(&x, scalar(left)), (&y, scalar(right))]);

    assert_eq!(
        type_of(&bindings, &(reference(&x) / reference(&y))),
        scalar(quotient)
    );
}

#[rstest]
#[case(Int8, Int32, Int32)]
#[case(Int32, Float16, Float32)]
#[case(Int64, Float16, Float64)]
#[case(Float16, Float32, Float32)]
fn floor_division_stays_integral_or_lifts_to_a_float(
    #[case] left: CoreDataType,
    #[case] right: CoreDataType,
    #[case] quotient: CoreDataType,
) {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let bindings = params(&[(&x, scalar(left)), (&y, scalar(right))]);
    let expression = Expression::new_binary(
        fhy_core::expression::BinaryOperation::FloorDivide,
        reference(&x),
        reference(&y),
    );

    assert_eq!(type_of(&bindings, &expression), scalar(quotient));
}

#[test]
fn floor_division_of_a_complex_operand_is_refused() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let bindings = params(&[(&x, scalar(Complex64)), (&y, scalar(Int32))]);
    let expression = Expression::new_binary(
        fhy_core::expression::BinaryOperation::FloorDivide,
        reference(&x),
        reference(&y),
    );

    let (_, reason) = rule_of(synthesize(&bindings, &expression));

    assert!(
        reason.starts_with("floor division is not defined for complex numerical types"),
        "{reason}"
    );
}

#[test]
fn arithmetic_across_families_is_a_promotion_error() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let bindings = params(&[(&x, scalar(Int32)), (&y, scalar(Complex64))]);

    let (kind, reason) = rule_of(synthesize(&bindings, &(reference(&x) + reference(&y))));

    assert_eq!(kind, TypeRuleKind::Promotion);
    assert_eq!(
        reason,
        "unsupported primitive data type promotion: int32, complex64"
    );
}

#[rstest]
#[case(
    UnaryOperation::Negate,
    "unary negation is not defined for boolean operands"
)]
#[case(
    UnaryOperation::Positive,
    "unary positive is not defined for boolean operands"
)]
fn unary_arithmetic_refuses_a_boolean(#[case] operation: UnaryOperation, #[case] reason: &str) {
    let (kind, text) = rule_of(synthesize(
        &Bindings::new(),
        &Expression::new_unary(operation, boolean(true)),
    ));

    assert_eq!(kind, TypeRuleKind::Boolean);
    assert_eq!(text, reason);
}

#[test]
fn binary_arithmetic_refuses_a_boolean_operand_by_the_operation_s_name() {
    let (_, reason) = rule_of(synthesize(
        &Bindings::new(),
        &(boolean(true) + Expression::from(1)),
    ));

    assert_eq!(
        reason,
        "the add operation is not defined for boolean operands"
    );
}

// =============================================================================
// Comparisons, logic and piecewise
// =============================================================================

#[test]
fn comparisons_are_boolean_and_check_their_operands_promote() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let numbers = params(&[(&x, scalar(Int32)), (&y, scalar(Float32))]);

    assert_eq!(type_of(&numbers, &reference(&x).less(1)), scalar(Bool));
    assert_eq!(
        type_of(&numbers, &reference(&x).equals(3).equals(boolean(false))),
        scalar(Bool)
    );
    assert_eq!(
        rule_of(synthesize(&numbers, &reference(&x).less(reference(&y)))).0,
        TypeRuleKind::Promotion
    );
}

#[test]
fn boolean_comparisons_follow_the_equality_and_ordering_rules() {
    let (kind, reason) = rule_of(synthesize(
        &Bindings::new(),
        &boolean(true).less(boolean(false)),
    ));
    assert_eq!(kind, TypeRuleKind::Boolean);
    assert_eq!(
        reason,
        "ordering operation less is not defined for boolean operands"
    );

    let (_, reason) = rule_of(synthesize(&Bindings::new(), &boolean(true).equals(1)));
    assert!(reason.starts_with(
        "equality operation equal is not defined between boolean and non-boolean operands"
    ));
    assert_eq!(
        type_of(&Bindings::new(), &boolean(true).equals(boolean(false))),
        scalar(Bool)
    );
}

#[test]
fn logical_connectives_need_boolean_operands() {
    let x = Identifier::new("x");
    let bindings = params(&[(&x, scalar(Int32))]);

    assert_eq!(
        type_of(&bindings, &boolean(true).and(reference(&x).less(0))),
        scalar(Bool)
    );
    let (kind, reason) = rule_of(synthesize(
        &bindings,
        &Expression::all([boolean(true), reference(&x), boolean(false)]),
    ));
    assert_eq!(kind, TypeRuleKind::Boolean);
    assert_eq!(
        reason,
        "logical and requires boolean operands, but got bool[], int32[] and bool[]"
    );
    let (_, reason) = rule_of(synthesize(
        &bindings,
        &Expression::new_unary(UnaryOperation::LogicalNot, reference(&x)),
    ));
    assert_eq!(
        reason,
        "logical NOT requires a boolean operand, but got int32[]"
    );
}

#[test]
fn a_piecewise_promotes_its_branches_and_needs_boolean_conditions() {
    let x = Identifier::new("x");
    let bindings = params(&[(&x, scalar(Int16))]);
    let good = Expression::piecewise(
        [(reference(&x).greater(0), reference(&x))],
        Expression::from(1.5),
    )
    .expect("a valid piecewise");
    let bad_condition = Expression::piecewise(
        [
            (boolean(true), Expression::from(1)),
            (reference(&x), Expression::from(2)),
        ],
        0,
    )
    .expect("a valid piecewise");

    assert_eq!(
        rule_of(synthesize(&bindings, &good)).0,
        TypeRuleKind::Promotion
    );
    let (kind, reason) = rule_of(synthesize(&bindings, &bad_condition));
    assert_eq!(kind, TypeRuleKind::Piecewise);
    assert_eq!(
        reason,
        "piecewise case 1 condition must be boolean, but got int16[]"
    );
    let integral = Expression::piecewise([(reference(&x).greater(0), reference(&x))], 3)
        .expect("a valid piecewise");
    assert_eq!(type_of(&bindings, &integral), scalar(Int16));
}

#[test]
fn a_piecewise_branch_must_be_numerical() {
    let i = Identifier::new("i");
    let bindings = params(&[(&i, index(0, 4, 1))]);
    let expression =
        Expression::piecewise([(boolean(true), reference(&i))], 0).expect("a valid piecewise");

    let (_, reason) = rule_of(synthesize(&bindings, &expression));

    assert!(reason.ends_with("for case 0"), "{reason}");
}

// =============================================================================
// Index types
// =============================================================================

#[test]
fn an_index_shifts_by_an_integral_offset_on_either_side() {
    let (i, n) = (Identifier::new("i"), Identifier::new("N"));
    let bindings = params(&[(&i, index(0, reference(&n), 1))]);

    assert_eq!(
        type_of(&bindings, &(reference(&i) + 1)),
        index(Expression::from(0) + 1, reference(&n) + 1, 1)
    );
    assert_eq!(
        type_of(&bindings, &(Expression::from(2) + reference(&i))),
        index(Expression::from(0) + 2, reference(&n) + 2, 1)
    );
    assert_eq!(
        type_of(&bindings, &(reference(&i) - 1)),
        index(Expression::from(0) - 1, reference(&n) - 1, 1)
    );
}

#[test]
fn an_index_scales_by_a_positive_integer_literal_folding_a_literal_stride() {
    let (i, s) = (Identifier::new("i"), Identifier::new("s"));
    let bindings = params(&[(&i, index(0, 8, 2)), (&s, index(0, 8, reference(&s)))]);
    let three = Expression::from(3);

    assert_eq!(
        type_of(&bindings, &(&three * reference(&i))),
        index(
            &three * Expression::from(0),
            &three * Expression::from(8),
            6
        )
    );
    let symbolic = type_of(&bindings, &(reference(&s) * 3));
    assert_eq!(
        symbolic,
        index(
            &three * Expression::from(0),
            &three * Expression::from(8),
            &three * reference(&s)
        )
    );
}

#[rstest]
#[case(Expression::from(0), "but got scalar value 0")]
#[case(Expression::from(-2), "but got scalar value -2")]
#[case(Expression::from(1.5), "non-integral type float[]")]
fn an_index_refuses_a_scale_that_is_no_positive_integer(
    #[case] factor: Expression,
    #[case] phrase: &str,
) {
    let i = Identifier::new("i");
    let bindings = params(&[(&i, index(0, 8, 1))]);

    let (kind, reason) = rule_of(synthesize(&bindings, &(reference(&i) * factor)));

    assert_eq!(kind, TypeRuleKind::Index);
    assert!(reason.contains(phrase), "{reason}");
}

#[test]
fn an_index_refuses_a_non_literal_scale_a_float_offset_and_other_operations() {
    let (i, x, f) = (
        Identifier::new("i"),
        Identifier::new("x"),
        Identifier::new("f"),
    );
    let bindings = params(&[
        (&i, index(0, 8, 1)),
        (&x, scalar(Int32)),
        (&f, scalar(Float32)),
    ]);
    let cases = [
        (reference(&i) * reference(&x), "is not a literal expression"),
        (
            reference(&i) + reference(&f),
            "index shift requires an integral scalar offset, but the right operand",
        ),
        (
            reference(&f) + reference(&i),
            "the left operand has type float32[]",
        ),
        (
            reference(&i) / 2,
            "division is not defined for operands of index type",
        ),
        (
            reference(&i) + reference(&i),
            "the add operation between two index types is not supported",
        ),
        (
            Expression::from(2) - reference(&i),
            "the subtract operation is not defined for operands of types",
        ),
        (
            -reference(&i),
            "unary negation is not defined for index types",
        ),
    ];
    for (expression, phrase) in cases {
        let (_, reason) = rule_of(synthesize(&bindings, &expression));
        assert!(reason.contains(phrase), "{reason}");
    }
}

#[test]
fn index_comparisons_need_equal_index_types() {
    let (i, j) = (Identifier::new("i"), Identifier::new("j"));
    let bindings = params(&[(&i, index(0, 8, 1)), (&j, index(0, 9, 1))]);

    assert_eq!(
        type_of(&bindings, &reference(&i).equals(reference(&i))),
        scalar(Bool)
    );
    assert!(
        rule_of(synthesize(&bindings, &reference(&i).equals(reference(&j))))
            .1
            .contains("requires structural equivalence")
    );
    assert!(
        rule_of(synthesize(&bindings, &reference(&i).less(reference(&i))))
            .1
            .contains("between two index types")
    );
    assert!(
        rule_of(synthesize(&bindings, &reference(&i).less(1)))
            .1
            .starts_with("comparison less is not defined between index type and")
    );
}

#[test]
fn a_zero_literal_stride_is_refused_wherever_the_index_is_read() {
    let i = Identifier::new("i");
    let bindings = params(&[(&i, index(0, 8, 0))]);

    for expression in [
        reference(&i),
        reference(&i) + 1,
        Expression::new_unary(UnaryOperation::Positive, reference(&i)),
    ] {
        assert_eq!(
            rule_of(synthesize(&bindings, &expression)).0,
            TypeRuleKind::ZeroStride
        );
    }
    let boolean_stride = params(&[(&i, index(0, 8, boolean(false)))]);
    let _ = ok(synthesize(&boolean_stride, &reference(&i)));
}

// =============================================================================
// Calls
// =============================================================================

/// Return a registry with `scale(x: real) -> real = x * 2` and a constant.
fn registry_with_scale() -> FunctionRegistry {
    let x = Identifier::new("x");
    let mut registry = FunctionRegistry::new();
    registry
        .register_function(
            FunctionDefinition::new(
                FunctionName::new("scale").expect("a name"),
                [x.clone()],
                [FunctionSort::Real],
                FunctionSort::Real,
                reference(&x) * 2,
            )
            .expect("a definition"),
        )
        .expect("registers");
    registry
        .register_constant(
            NativeConstant::new(
                FunctionName::new("k").expect("a name"),
                FunctionSort::Int,
                3,
            )
            .expect("an int"),
        )
        .expect("registers");
    registry
}

#[test]
fn a_call_types_by_its_result_sort_after_checking_its_arguments() {
    let registry = registry_with_scale();
    let x = Identifier::new("x");
    let bindings = HashMap::from([(x.clone(), (scalar(Int32), TypeQualifier::State))]);
    let checker = TypeChecker::new(&bindings, &registry, &registry);

    assert_eq!(
        ok(checker.synthesize(&build_call_or_panic("scale", [reference(&x)]))),
        (scalar(Float64), TypeQualifier::Temp)
    );
    assert_eq!(
        ok(checker.synthesize(&build_call_or_panic(
            "max",
            [Expression::from(1), Expression::from(2)]
        ))),
        (scalar(Float64), TypeQualifier::Param)
    );
}

#[test]
fn a_call_refuses_an_argument_outside_its_sort_the_wrong_arity_and_a_constant() {
    let registry = registry_with_scale();
    let none = Bindings::new();
    let checker = TypeChecker::new(&none, &registry, &registry);
    let cases = [
        (
            build_call_or_panic("scale", [boolean(true)]),
            "argument 0 of function 'scale' expects sort real, but got bool[]",
        ),
        (
            build_call_or_panic("scale", [Expression::from(1), Expression::from(2)]),
            "function 'scale' expects 1 argument(s), but got 2",
        ),
        (
            build_call_or_panic("k", [Expression::from(1)]),
            "'k' is a registered constant, not a function",
        ),
        (
            build_call_or_panic("pi", Vec::<Expression>::new()),
            "'pi' is a registered constant, not a function",
        ),
    ];
    for (expression, phrase) in cases {
        let (kind, reason) = rule_of(checker.synthesize(&expression));
        assert_eq!(kind, TypeRuleKind::Call);
        assert!(reason.contains(phrase), "{reason}");
    }
}

#[test]
fn an_unknown_call_is_a_rule_or_deferred() {
    let registry = FunctionRegistry::new();
    let none = Bindings::new();
    let call = build_call_or_panic("never_registered", [Expression::from(1)]);

    let (kind, reason) = rule_of(TypeChecker::new(&none, &registry, &registry).synthesize(&call));
    assert_eq!(kind, TypeRuleKind::UnknownCall);
    assert_eq!(
        reason,
        "call to unknown function 'never_registered': no entry is registered under the name 'never_registered'"
    );
    let deferred = TypeChecker::new(&none, &registry, &registry)
        .with_deferred_unknown_calls()
        .synthesize(&call);
    assert!(matches!(deferred, Err(TypeCheckError::UnknownCall(_))));
}

#[test]
fn the_call_target_is_resolved_before_the_arguments_are_checked() {
    let registry = FunctionRegistry::new();
    let none = Bindings::new();
    let unbound = Identifier::new("u");

    let (kind, _) = rule_of(
        TypeChecker::new(&none, &registry, &registry)
            .synthesize(&build_call_or_panic("gone", [reference(&unbound)])),
    );

    assert_eq!(kind, TypeRuleKind::UnknownCall);
}

// =============================================================================
// Checking against an expected type
// =============================================================================

#[test]
fn a_synthesized_type_must_not_be_wider_than_the_expected_one() {
    let x = Identifier::new("x");
    let bindings = params(&[(&x, scalar(Int32))]);

    let _ = ok(check(&bindings, &reference(&x), &scalar(Int64)));
    let _ = ok(check(&bindings, &reference(&x), &scalar(Int32)));
    let (kind, reason) = rule_of(check(&bindings, &reference(&x), &scalar(Int16)));
    assert_eq!(kind, TypeRuleKind::ExpectedType);
    assert_eq!(
        reason,
        "synthesized type int32[] is wider than the expected type int16[]; promoting them yields int32, which does not match the expected type"
    );
}

#[test]
fn index_types_check_by_structural_equivalence_and_never_against_numbers() {
    let i = Identifier::new("i");
    let bindings = params(&[(&i, index(1, 10, 1))]);

    let _ = ok(check(&bindings, &reference(&i), &index(1, 10, 1)));
    assert!(
        rule_of(check(&bindings, &reference(&i), &index(2, 10, 1)))
            .1
            .contains("is not structurally equivalent")
    );
    assert!(
        rule_of(check(&bindings, &reference(&i), &scalar(Int32)))
            .1
            .contains("one is an index type and the other is a numerical type")
    );
}

// =============================================================================
// The error frame and depth
// =============================================================================

#[test]
fn an_error_at_the_root_is_framed_by_the_root_alone() {
    let i = Identifier::new("i");
    let bindings = params(&[(&i, index(1, 8, 1))]);
    let expression = -reference(&i);

    let error = synthesize(&bindings, &expression).expect_err("negation of an index");

    assert_eq!(
        error.to_string(),
        format!(
            "type error while inferring the type of `(-i::{})`: unary negation is not defined for index types; the resulting bounds and stride cannot be inferred safely",
            i.id()
        )
    );
}

#[test]
fn an_error_below_the_root_names_the_sub_expression() {
    let i = Identifier::new("i");
    let bindings = params(&[(&i, index(1, 8, 1))]);
    let inner = -reference(&i);
    let root = Expression::new_unary(UnaryOperation::Positive, inner);

    let message = synthesize(&bindings, &root)
        .expect_err("negation of an index")
        .to_string();

    assert!(
        message.starts_with(&format!(
            "type error while inferring the type of `(+(-i::{}))` at sub-expression `(-i::{})`",
            i.id(),
            i.id()
        )),
        "{message}"
    );
}

#[test]
fn a_callback_error_passes_through() {
    struct Failing;
    impl fhy_core::types::checking::IdentifierTypes for Failing {
        fn identifier_type(
            &self,
            _identifier: &Identifier,
        ) -> Result<Option<(Type, TypeQualifier)>, fhy_core::foreign::BoxError> {
            Err("lookup failed".into())
        }
    }
    let registry = FunctionRegistry::new();

    let error = TypeChecker::new(&Failing, &registry, &registry)
        .synthesize(&reference(&Identifier::new("x")))
        .expect_err("the lookup fails");

    let TypeCheckError::Callback(source) = error else {
        panic!("a callback error");
    };
    assert_eq!(source.to_string(), "lookup failed");
}

#[test]
fn a_deep_expression_checks_on_a_small_stack() {
    run_on_small_stack(|| {
        let x = Identifier::new("x");
        let bindings = params(&[(&x, scalar(Int64))]);
        let deep = (0..SMALL_STACK_DEPTH).fold(reference(&x), |tree, _| tree + 1);
        let typed = synthesize(&bindings, &deep).map(|typed| typed.0 == scalar(Int64));
        assert!(matches!(typed, Ok(true)));
    });
}

/// Test a rule error on a doubling DAG displays in bounded size: the root
/// and the sub-expression are written up to a budget of node occurrences
/// (R2-010), not once per path.
#[test]
fn a_rule_error_on_a_dag_displays_in_bounded_size() {
    let x = Identifier::new("x");
    let bindings: Bindings = HashMap::from([(x.clone(), (scalar(Int32), TypeQualifier::Param))]);
    let dag = crate::support::expression::build_doubling_dag(&Expression::from(x), 16);
    let tree = &dag + Expression::from(LiteralValue::from(true));

    let error = synthesize(&bindings, &tree).expect_err("a boolean in an addition");

    let text = error.to_string();
    assert!(matches!(error, TypeCheckError::Rule { .. }), "{text}");
    assert!(text.len() < 4096, "{} bytes", text.len());
    assert!(text.contains('…'), "{text}");
}
