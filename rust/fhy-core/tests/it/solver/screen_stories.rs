//! Stories for the hazard screen, [`Hazard::find`]: each of the five
//! hazards, the shapes it refuses and the neighbours it admits, their
//! order, and the walks' depth and sharing.

use std::collections::HashMap;

use fhy_core::expression::builtins::{BuiltinConstant, BuiltinFunction};
use fhy_core::expression::registry::{FunctionRegistry, NativeConstant, NativeFunction};
use fhy_core::expression::{
    BinaryOperation, Expression, FunctionName, FunctionSort, LiteralValue, NoRegisteredSorts,
    SymbolType,
};
use fhy_core::identifier::Identifier;
use fhy_core::solver::Hazard;
use rstest::rstest;

use crate::support::expression::{build_identifier, build_literal};
use crate::support::solver::build_symbol_types;
use crate::support::stack::{SMALL_STACK_DEPTH, run_on_small_stack};

/// Return the hazard of `expression` with no sort lookup.
fn find(expression: &Expression, symbol_types: &HashMap<Identifier, SymbolType>) -> Option<Hazard> {
    Hazard::find(expression, symbol_types, &NoRegisteredSorts)
}

/// Return a decimal literal parsed from `text`.
fn build_decimal(text: &str) -> Expression {
    build_literal(LiteralValue::parse_text(text).expect("a decimal text"))
}

/// Return the reference to a built-in constant.
fn build_constant(constant: BuiltinConstant) -> Expression {
    Expression::from(constant.identifier().clone())
}

/// Return `left op right`.
fn build_binary(operation: BinaryOperation, left: Expression, right: Expression) -> Expression {
    Expression::new_binary(operation, left, right)
}

// ---------------------------------------------------------------------------
// Native constants
// ---------------------------------------------------------------------------

#[rstest]
fn native_constant_hazard_refuses_every_builtin_constant(
    #[values(
        BuiltinConstant::Pi,
        BuiltinConstant::E,
        BuiltinConstant::Inf,
        BuiltinConstant::Nan
    )]
    constant: BuiltinConstant,
) {
    let expression = build_constant(constant).greater(4);

    let hazard = find(&expression, &HashMap::new());

    assert_eq!(
        hazard,
        Some(Hazard::NativeConstant(vec![constant.identifier().clone()]))
    );
}

#[test]
fn native_constant_hazard_names_every_constant_ordered_by_id() {
    let expression = (build_constant(BuiltinConstant::E) + build_constant(BuiltinConstant::Pi))
        .greater(build_constant(BuiltinConstant::E));

    let hazard = find(&expression, &HashMap::new());

    assert_eq!(
        hazard,
        Some(Hazard::NativeConstant(vec![
            BuiltinConstant::Pi.identifier().clone(),
            BuiltinConstant::E.identifier().clone(),
        ]))
    );
}

#[test]
fn native_constant_hazard_refuses_a_constant_the_sort_lookup_reports() {
    let mut registry = FunctionRegistry::new();
    let answer = registry
        .register_constant(
            NativeConstant::new(
                FunctionName::new("answer").expect("a name"),
                FunctionSort::Int,
                42,
            )
            .expect("an int constant"),
        )
        .expect("a new name");
    let expression = Expression::from(answer.clone()).equals(42);

    let hazard = Hazard::find(
        &expression,
        &HashMap::<Identifier, SymbolType>::new(),
        &registry,
    );

    assert_eq!(hazard, Some(Hazard::NativeConstant(vec![answer])));
}

#[test]
fn native_constant_hazard_ignores_an_identifier_merely_named_like_a_constant() {
    let (pi, reference) = build_identifier("pi");
    let symbol_types = build_symbol_types(&[(&pi, SymbolType::Int)]);

    assert_eq!(find(&reference.equals(1), &symbol_types), None);
}

#[test]
fn native_constant_hazard_is_checked_before_every_other_hazard() {
    let (x, reference) = build_identifier("x");
    let symbol_types = build_symbol_types(&[(&x, SymbolType::Real)]);
    let expression = Expression::all([
        (reference.clone() / reference.clone()).equals(build_literal(f64::INFINITY)),
        build_constant(BuiltinConstant::Pi).greater(reference),
    ]);

    assert!(matches!(
        find(&expression, &symbol_types),
        Some(Hazard::NativeConstant(_))
    ));
}

// ---------------------------------------------------------------------------
// Non-finite literals
// ---------------------------------------------------------------------------

#[rstest]
#[case::positive_infinity(f64::INFINITY, BinaryOperation::Less)]
#[case::negative_infinity(f64::NEG_INFINITY, BinaryOperation::Greater)]
#[case::nan(f64::NAN, BinaryOperation::Equal)]
fn non_finite_literal_hazard_refuses_the_literal(
    #[case] value: f64,
    #[case] operation: BinaryOperation,
) {
    let (x, reference) = build_identifier("x");
    let literal = build_literal(value);
    let expression = build_binary(operation, reference, literal.clone());

    let hazard = find(&expression, &build_symbol_types(&[(&x, SymbolType::Real)]));

    assert_eq!(hazard, Some(Hazard::NonFiniteLiteral(literal)));
}

#[rstest]
fn non_finite_literal_hazard_is_found_in_a_divisor_before_the_partial_operation(
    #[values(
        BinaryOperation::Divide,
        BinaryOperation::FloorDivide,
        BinaryOperation::FloorMod
    )]
    operation: BinaryOperation,
    #[values(f64::NAN, f64::INFINITY)] value: f64,
) {
    let (x, reference) = build_identifier("x");
    let divisor = build_literal(value);
    let expression = build_binary(operation, reference, divisor.clone()).equals(0.0);

    let hazard = find(&expression, &build_symbol_types(&[(&x, SymbolType::Real)]));

    assert_eq!(hazard, Some(Hazard::NonFiniteLiteral(divisor)));
}

#[test]
fn non_finite_literal_hazard_admits_finite_floats_and_decimals() {
    let (x, reference) = build_identifier("x");
    let symbol_types = build_symbol_types(&[(&x, SymbolType::Real)]);
    let expression = Expression::all([
        reference.clone().less(1.5),
        reference.greater(build_decimal("0.5")),
    ]);

    assert_eq!(find(&expression, &symbol_types), None);
}

// ---------------------------------------------------------------------------
// Boolean coercion
// ---------------------------------------------------------------------------

#[test]
fn boolean_coercion_hazard_refuses_a_boolean_literal_compared_with_an_int() {
    let (x, reference) = build_identifier("x");
    let expression = reference.equals(build_literal(true));

    let hazard = find(&expression, &build_symbol_types(&[(&x, SymbolType::Int)]));

    assert_eq!(hazard, Some(Hazard::BooleanCoercion(expression)));
}

#[test]
fn boolean_coercion_hazard_admits_a_boolean_literal_compared_with_a_boolean() {
    let (x, reference) = build_identifier("x");

    let hazard = find(
        &reference.equals(build_literal(true)),
        &build_symbol_types(&[(&x, SymbolType::Bool)]),
    );

    assert_eq!(hazard, None);
}

#[test]
fn boolean_coercion_hazard_refuses_a_conjunction_compared_with_an_int() {
    let (x, x_reference) = build_identifier("x");
    let (b, b_reference) = build_identifier("b");
    let (c, c_reference) = build_identifier("c");
    let expression = x_reference.equals(b_reference.and(c_reference));
    let symbol_types = build_symbol_types(&[
        (&x, SymbolType::Int),
        (&b, SymbolType::Bool),
        (&c, SymbolType::Bool),
    ]);

    assert_eq!(
        find(&expression, &symbol_types),
        Some(Hazard::BooleanCoercion(expression))
    );
}

#[test]
fn boolean_coercion_hazard_refuses_a_boolean_operand_of_arithmetic() {
    let (x, x_reference) = build_identifier("x");
    let (b, b_reference) = build_identifier("b");
    let total = x_reference + b_reference;
    let symbol_types = build_symbol_types(&[(&x, SymbolType::Int), (&b, SymbolType::Bool)]);

    assert_eq!(
        find(&total.greater(0), &symbol_types),
        Some(Hazard::BooleanCoercion(total))
    );
}

#[test]
fn boolean_coercion_hazard_refuses_a_piecewise_mixing_a_boolean_and_a_number() {
    let (b, b_reference) = build_identifier("b");
    let mixed = Expression::piecewise([(b_reference, build_literal(1))], build_literal(true))
        .expect("a piecewise");
    let expression = mixed.clone().equals(1);

    assert_eq!(
        find(&expression, &build_symbol_types(&[(&b, SymbolType::Bool)])),
        Some(Hazard::BooleanCoercion(mixed))
    );
}

#[test]
fn boolean_coercion_hazard_admits_connectives_and_negations_of_booleans() {
    let (b, b_reference) = build_identifier("b");
    let (x, x_reference) = build_identifier("x");
    let expression = Expression::all([
        !b_reference.clone(),
        b_reference.and(build_literal(true)),
        (-x_reference).less(0),
    ]);
    let symbol_types = build_symbol_types(&[(&b, SymbolType::Bool), (&x, SymbolType::Int)]);

    assert_eq!(find(&expression, &symbol_types), None);
}

#[test]
fn boolean_coercion_hazard_admits_an_undetermined_operand_compared_with_a_boolean() {
    let (y, y_reference) = build_identifier("y");

    let hazard = find(&y_reference.equals(build_literal(true)), &HashMap::new());

    assert_eq!(hazard, None, "an undeclared {y:?} lowers to no known sort");
}

// ---------------------------------------------------------------------------
// Partial operations
// ---------------------------------------------------------------------------

#[test]
fn partial_operation_hazard_refuses_a_division_by_a_variable() {
    let (x, reference) = build_identifier("x");
    let division = reference.clone() / reference;

    let hazard = find(
        &division.clone().not_equals(1.0),
        &build_symbol_types(&[(&x, SymbolType::Real)]),
    );

    assert_eq!(hazard, Some(Hazard::PartialOperation(division)));
}

#[rstest]
#[case::floor_divide_by_a_negative(BinaryOperation::FloorDivide, -2)]
#[case::floor_mod_by_a_negative(BinaryOperation::FloorMod, -2)]
#[case::floor_divide_by_zero(BinaryOperation::FloorDivide, 0)]
#[case::floor_mod_by_zero(BinaryOperation::FloorMod, 0)]
fn partial_operation_hazard_refuses_a_floor_operation_without_a_positive_divisor(
    #[case] operation: BinaryOperation,
    #[case] divisor: i64,
) {
    let applied = build_binary(operation, build_literal(7), build_literal(divisor));

    let hazard = find(&applied.clone().equals(-4), &HashMap::new());

    assert_eq!(hazard, Some(Hazard::PartialOperation(applied)));
}

#[rstest]
#[case::floor_divide(BinaryOperation::FloorDivide)]
#[case::floor_mod(BinaryOperation::FloorMod)]
fn partial_operation_hazard_admits_a_floor_operation_by_a_positive_literal(
    #[case] operation: BinaryOperation,
    #[values(build_literal(2), build_literal(2.5))] divisor: Expression,
) {
    let (x, reference) = build_identifier("x");
    let expression = build_binary(operation, reference, divisor).greater(1);

    assert_eq!(
        find(&expression, &build_symbol_types(&[(&x, SymbolType::Int)])),
        None
    );
}

#[rstest]
#[case::floor_divide(BinaryOperation::FloorDivide)]
#[case::floor_mod(BinaryOperation::FloorMod)]
#[case::power(BinaryOperation::Power)]
fn partial_operation_hazard_refuses_a_decimal_divisor_or_exponent(
    #[case] operation: BinaryOperation,
) {
    let (x, reference) = build_identifier("x");
    let applied = build_binary(operation, reference, build_decimal("2.0"));

    assert_eq!(
        find(
            &applied.clone().equals(1),
            &build_symbol_types(&[(&x, SymbolType::Int)])
        ),
        Some(Hazard::PartialOperation(applied))
    );
}

#[test]
fn partial_operation_hazard_admits_a_division_by_a_negative_float() {
    let expression = (build_literal(7.0) / -2.0).equals(-3.5);

    assert_eq!(find(&expression, &HashMap::new()), None);
}

#[test]
fn partial_operation_hazard_refuses_a_division_of_two_integers() {
    let division = build_literal(7) / 2;

    assert_eq!(
        find(&division.clone().equals(3.5), &HashMap::new()),
        Some(Hazard::PartialOperation(division))
    );
}

#[rstest]
#[case::real_divisor(build_literal(7) / 2.0)]
#[case::real_dividend(build_literal(7.0) / 2)]
#[case::decimal_dividend(build_decimal("7.5") / 2)]
fn partial_operation_hazard_admits_a_division_with_a_real_operand(#[case] division: Expression) {
    assert_eq!(find(&division.equals(3.5), &HashMap::new()), None);
}

#[rstest]
#[case::identifier(false, SymbolType::Real, true)]
#[case::integer_identifier(false, SymbolType::Int, false)]
#[case::negated(true, SymbolType::Real, true)]
fn partial_operation_hazard_reads_a_real_dividend_through_negation(
    #[case] is_negated: bool,
    #[case] sort: SymbolType,
    #[case] is_admitted: bool,
) {
    let (x, reference) = build_identifier("x");
    let dividend = if is_negated { -reference } else { reference };
    let division = dividend / 2;

    let hazard = find(
        &division.clone().greater(0.0),
        &build_symbol_types(&[(&x, sort)]),
    );

    assert_eq!(
        hazard,
        (!is_admitted).then(|| Hazard::PartialOperation(division))
    );
}

#[test]
fn partial_operation_hazard_reads_a_real_dividend_through_arithmetic_but_not_unary_plus() {
    let (x, reference) = build_identifier("x");
    let symbol_types = build_symbol_types(&[(&x, SymbolType::Real)]);
    let through_sum = (reference.clone() + 1) / 2;
    let through_plus = reference.positive() / 2;

    assert_eq!(find(&through_sum.greater(0.0), &symbol_types), None);
    assert_eq!(
        find(&through_plus.clone().greater(0.0), &symbol_types),
        Some(Hazard::PartialOperation(through_plus))
    );
}

#[rstest]
#[case::zero_to_the_zero(build_literal(0), build_literal(0))]
#[case::negative_exponent(build_identifier("x").1, build_literal(-1))]
#[case::zero_exponent(build_identifier("x").1, build_literal(0))]
#[case::float_exponent(build_identifier("x").1, build_literal(0.5))]
#[case::integral_float_exponent(build_identifier("x").1, build_literal(2.0))]
#[case::variable_exponent(build_literal(2), build_identifier("n").1)]
fn partial_operation_hazard_refuses_an_unsafe_exponent(
    #[case] base: Expression,
    #[case] exponent: Expression,
) {
    let power = base.power(exponent);
    let symbol_types = |_: &Identifier| Some(SymbolType::Int);

    let hazard = Hazard::find(&power.clone().equals(1), &symbol_types, &NoRegisteredSorts);

    assert_eq!(hazard, Some(Hazard::PartialOperation(power)));
}

#[rstest]
fn partial_operation_hazard_admits_an_integer_exponent_of_at_least_one(
    #[values(1, 2, 3)] exponent: i64,
) {
    let (x, reference) = build_identifier("x");
    let expression = reference.power(exponent).equals(4);

    assert_eq!(
        find(&expression, &build_symbol_types(&[(&x, SymbolType::Int)])),
        None
    );
}

#[test]
fn partial_operation_hazard_admits_every_spelling_of_an_integer_divisor() {
    let (x, reference) = build_identifier("x");
    let parsed = build_literal(LiteralValue::parse_text("02").expect("an integer text"));
    let expression = Expression::all([
        reference.clone().floor_divide(parsed.clone()).equals(2),
        reference.clone().floor_mod(parsed.clone()).equals(1),
        reference.power(parsed).equals(4),
    ]);

    assert_eq!(
        find(&expression, &build_symbol_types(&[(&x, SymbolType::Int)])),
        None
    );
}

// ---------------------------------------------------------------------------
// Mixed int/real equality
// ---------------------------------------------------------------------------

#[rstest]
#[case::int_against_a_float(SymbolType::Int, build_literal(1.5), BinaryOperation::Equal)]
#[case::real_against_an_int(SymbolType::Real, build_literal(1), BinaryOperation::Equal)]
#[case::real_unequal_to_an_int(SymbolType::Real, build_literal(1), BinaryOperation::NotEqual)]
#[case::int_against_a_decimal(SymbolType::Int, build_decimal("1.5"), BinaryOperation::Equal)]
#[case::undeclared_against_an_int(SymbolType::Bool, build_literal(1), BinaryOperation::Equal)]
fn mixed_equality_hazard_refuses_a_literal_of_another_kind(
    #[case] sort: SymbolType,
    #[case] literal: Expression,
    #[case] operation: BinaryOperation,
) {
    let (x, reference) = build_identifier("x");
    let expression = build_binary(operation, reference, literal);
    let symbol_types = if sort == SymbolType::Bool {
        HashMap::new()
    } else {
        build_symbol_types(&[(&x, sort)])
    };

    assert_eq!(
        find(&expression, &symbol_types),
        Some(Hazard::MixedIntRealEquality(expression))
    );
}

#[rstest]
#[case::real_against_a_float(SymbolType::Real, build_literal(1.5), BinaryOperation::Equal)]
#[case::real_against_a_whole_float(SymbolType::Real, build_literal(1.0), BinaryOperation::Equal)]
#[case::real_ordered_against_an_int(SymbolType::Real, build_literal(1), BinaryOperation::Less)]
#[case::real_at_least_an_int(SymbolType::Real, build_literal(1), BinaryOperation::GreaterEqual)]
#[case::int_ordered_against_a_float(SymbolType::Int, build_literal(1.5), BinaryOperation::Less)]
#[case::int_against_an_int(SymbolType::Int, build_literal(3), BinaryOperation::Equal)]
fn mixed_equality_hazard_admits_a_literal_of_the_same_kind_or_an_ordering(
    #[case] sort: SymbolType,
    #[case] literal: Expression,
    #[case] operation: BinaryOperation,
) {
    let (x, reference) = build_identifier("x");
    let expression = build_binary(operation, reference, literal);

    assert_eq!(find(&expression, &build_symbol_types(&[(&x, sort)])), None);
}

#[rstest]
#[case::sum(build_identifier("y").1 + 1)]
#[case::product(build_identifier("y").1 * 2)]
#[case::negation(-build_identifier("y").1)]
#[case::power(build_identifier("y").1.power(2))]
#[case::floor_division(build_identifier("y").1.floor_divide(2))]
fn mixed_equality_hazard_follows_the_integer_kind_through_arithmetic(#[case] operand: Expression) {
    let int_types = |_: &Identifier| Some(SymbolType::Int);
    let against_a_float = operand.clone().equals(3.0);
    let against_an_int = operand.clone().equals(3);
    let float_on_the_left = build_literal(3.0).equals(operand);

    assert_eq!(
        Hazard::find(&against_a_float, &int_types, &NoRegisteredSorts),
        Some(Hazard::MixedIntRealEquality(against_a_float))
    );
    assert_eq!(
        Hazard::find(&float_on_the_left, &int_types, &NoRegisteredSorts),
        Some(Hazard::MixedIntRealEquality(float_on_the_left))
    );
    assert_eq!(
        Hazard::find(&against_an_int, &int_types, &NoRegisteredSorts),
        None
    );
}

#[test]
fn mixed_equality_hazard_reads_a_real_operand_of_arithmetic_as_real() {
    let (y, y_reference) = build_identifier("y");
    let (r, r_reference) = build_identifier("r");
    let symbol_types = build_symbol_types(&[(&y, SymbolType::Int), (&r, SymbolType::Real)]);

    assert_eq!(find(&(y_reference * 1.0).equals(3.0), &symbol_types), None);
    assert_eq!(find(&(r_reference + 1).equals(3.0), &symbol_types), None);
}

#[test]
fn mixed_equality_hazard_admits_an_equality_of_two_non_literals() {
    let (y, y_reference) = build_identifier("y");
    let (r, r_reference) = build_identifier("r");
    let symbol_types = build_symbol_types(&[(&y, SymbolType::Int), (&r, SymbolType::Real)]);

    assert_eq!(find(&y_reference.equals(r_reference), &symbol_types), None);
}

#[test]
fn mixed_equality_hazard_reads_a_piecewise_kind_from_its_branches() {
    let (y, y_reference) = build_identifier("y");
    let (r, r_reference) = build_identifier("r");
    let (b, b_reference) = build_identifier("b");
    let symbol_types = build_symbol_types(&[
        (&y, SymbolType::Int),
        (&r, SymbolType::Real),
        (&b, SymbolType::Bool),
    ]);
    let all_real = Expression::piecewise(
        [(b_reference.clone(), r_reference.clone())],
        r_reference.clone(),
    )
    .expect("a piecewise");
    let mixed =
        Expression::piecewise([(b_reference, y_reference)], r_reference).expect("a piecewise");
    let int_against_all_real = build_literal(1).equals(all_real.clone());
    let float_against_mixed = build_literal(3.0).equals(mixed);

    assert_eq!(
        find(&int_against_all_real, &symbol_types),
        Some(Hazard::MixedIntRealEquality(int_against_all_real))
    );
    assert_eq!(
        find(&float_against_mixed, &symbol_types),
        Some(Hazard::MixedIntRealEquality(float_against_mixed))
    );
    assert_eq!(
        find(&build_literal(1.0).equals(all_real), &symbol_types),
        None
    );
}

#[rstest]
#[case::int_literals(build_literal(1), build_literal(1.0), true)]
#[case::parsed_int_literal(build_literal(LiteralValue::parse_text("01").expect("an integer text")), build_literal(1.0), true)]
#[case::decimal_against_an_int(build_decimal("1.0"), build_literal(1), true)]
#[case::two_ints(build_literal(LiteralValue::parse_text("1").expect("an integer text")), build_literal(2), false)]
fn mixed_equality_hazard_classifies_every_literal_spelling_by_its_value(
    #[case] left: Expression,
    #[case] right: Expression,
    #[case] is_refused: bool,
) {
    let expression = left.equals(right);

    assert_eq!(
        find(&expression, &HashMap::new()),
        is_refused.then(|| Hazard::MixedIntRealEquality(expression))
    );
}

#[rstest]
#[case::user_int_function(FunctionSort::Int, true)]
#[case::user_nat_function(FunctionSort::Nat, true)]
#[case::user_real_function(FunctionSort::Real, false)]
fn mixed_equality_hazard_reads_a_call_kind_from_the_sort_lookup(
    #[case] result_sort: FunctionSort,
    #[case] is_refused: bool,
) {
    let name = FunctionName::new("f").expect("a name");
    let mut registry = FunctionRegistry::new();
    registry
        .register_native_function(NativeFunction::new(name.clone(), [], result_sort))
        .expect("a new name");
    let expression = Expression::call(name, Vec::<Expression>::new()).equals(1.5);

    let hazard = Hazard::find(
        &expression,
        &HashMap::<Identifier, SymbolType>::new(),
        &registry,
    );

    assert_eq!(
        hazard,
        is_refused.then(|| Hazard::MixedIntRealEquality(expression))
    );
}

#[test]
fn mixed_equality_hazard_reads_a_builtin_call_kind_from_the_catalogue() {
    let (x, reference) = build_identifier("x");
    let symbol_types = build_symbol_types(&[(&x, SymbolType::Real)]);
    let floored = Expression::call(BuiltinFunction::Floor, [reference.clone()]);
    let absolute = Expression::call(BuiltinFunction::Abs, [reference]);
    let floor_against_a_float = floored.clone().equals(1.5);

    assert_eq!(
        find(&floor_against_a_float, &symbol_types),
        Some(Hazard::MixedIntRealEquality(floor_against_a_float))
    );
    assert_eq!(find(&floored.equals(1), &symbol_types), None);
    assert_eq!(find(&absolute.equals(1.5), &symbol_types), None);
}

#[test]
fn mixed_equality_hazard_refuses_an_unknown_function_against_a_literal() {
    let expression = Expression::call(
        FunctionName::new("g").expect("a name"),
        Vec::<Expression>::new(),
    )
    .equals(1);

    assert_eq!(
        find(&expression, &HashMap::new()),
        Some(Hazard::MixedIntRealEquality(expression))
    );
}

#[test]
fn mixed_equality_hazard_is_found_below_the_root() {
    let (x, reference) = build_identifier("x");
    let hazard = reference.clone().equals(1.5);
    let expression = Expression::all([reference.greater(0), hazard.clone()]);

    assert_eq!(
        find(&expression, &build_symbol_types(&[(&x, SymbolType::Int)])),
        Some(Hazard::MixedIntRealEquality(hazard))
    );
}

// ---------------------------------------------------------------------------
// Order
// ---------------------------------------------------------------------------

#[test]
fn hazard_kinds_are_checked_in_order_whatever_the_node_order() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let (b, b_reference) = build_identifier("b");
    let symbol_types = build_symbol_types(&[
        (&x, SymbolType::Int),
        (&y, SymbolType::Real),
        (&b, SymbolType::Bool),
    ]);
    let equality = x_reference.clone().equals(1.5);
    let division = y_reference.clone() / y_reference;
    let coercion = x_reference.equals(b_reference);
    let expression = Expression::all([
        equality.clone(),
        division.clone().greater(0.0),
        coercion.clone(),
    ]);
    let without_coercion = Expression::all([equality, division.clone().greater(0.0)]);

    assert_eq!(
        find(&expression, &symbol_types),
        Some(Hazard::BooleanCoercion(coercion))
    );
    assert_eq!(
        find(&without_coercion, &symbol_types),
        Some(Hazard::PartialOperation(division))
    );
}

#[test]
fn hazard_of_a_kind_is_the_first_node_in_pre_order() {
    let (a, a_reference) = build_identifier("a");
    let (b, b_reference) = build_identifier("b");
    let symbol_types = build_symbol_types(&[(&a, SymbolType::Real), (&b, SymbolType::Real)]);
    let inner = b_reference.clone() / b_reference;
    let outer = a_reference.clone() / inner.clone();
    let later = a_reference.clone() / a_reference;
    let expression = Expression::all([outer.clone().greater(0.0), later.greater(0.0)]);

    assert_eq!(
        find(&expression, &symbol_types),
        Some(Hazard::PartialOperation(outer))
    );
    assert_eq!(
        find(&inner.clone().greater(0.0), &symbol_types),
        Some(Hazard::PartialOperation(inner))
    );
}

// ---------------------------------------------------------------------------
// Depth and sharing
// ---------------------------------------------------------------------------

#[test]
fn hazard_screen_walks_a_deep_chain_on_a_small_stack() {
    let found = run_on_small_stack(|| {
        let (x, reference) = build_identifier("x");
        let mut chain = reference;
        for _ in 0..SMALL_STACK_DEPTH {
            chain = chain + 1;
        }
        let symbol_types = build_symbol_types(&[(&x, SymbolType::Int)]);
        let admitted = Hazard::find(&chain.clone().equals(3), &symbol_types, &NoRegisteredSorts);
        let refused = Hazard::find(&chain.equals(3.0), &symbol_types, &NoRegisteredSorts);
        (
            admitted.is_none(),
            matches!(refused, Some(Hazard::MixedIntRealEquality(_))),
        )
    });

    assert_eq!(found, (true, true));
}

#[test]
fn hazard_screen_walks_a_deep_piecewise_nest_on_a_small_stack() {
    let found = run_on_small_stack(|| {
        let (x, reference) = build_identifier("x");
        let mut nest = reference.clone();
        for level in 0..SMALL_STACK_DEPTH {
            nest = Expression::piecewise(
                [(reference.clone().greater(level), reference.clone())],
                nest,
            )
            .expect("a piecewise");
        }
        let symbol_types = build_symbol_types(&[(&x, SymbolType::Int)]);
        Hazard::find(&nest.equals(3.5), &symbol_types, &NoRegisteredSorts)
    });

    assert!(matches!(found, Some(Hazard::MixedIntRealEquality(_))));
}

#[test]
fn hazard_screen_classifies_a_shared_dag_in_linear_time() {
    let (x, reference) = build_identifier("x");
    let mut dag = reference;
    for _ in 0..64 {
        dag = dag.clone() + dag;
    }

    let refused = find(
        &dag.clone().equals(1.5),
        &build_symbol_types(&[(&x, SymbolType::Int)]),
    );
    let admitted = find(
        &dag.equals(1.5),
        &build_symbol_types(&[(&x, SymbolType::Real)]),
    );

    assert!(matches!(refused, Some(Hazard::MixedIntRealEquality(_))));
    assert_eq!(admitted, None);
}

// ---------------------------------------------------------------------------
// Text and node
// ---------------------------------------------------------------------------

#[test]
fn hazard_displays_one_lowercase_line_per_kind() {
    let node = build_literal(1);
    let pi = BuiltinConstant::Pi.identifier().clone();

    assert_eq!(
        Hazard::NativeConstant(vec![pi]).to_string(),
        "the expression refers to native constants that smt-lib2 has no term for: pi::48"
    );
    assert_eq!(
        Hazard::NonFiniteLiteral(node.clone()).to_string(),
        "the expression holds a non-finite float, which has no rational value"
    );
    assert_eq!(
        Hazard::BooleanCoercion(node.clone()).to_string(),
        "the expression lowers a boolean operand into a numeric context"
    );
    assert_eq!(
        Hazard::PartialOperation(node.clone()).to_string(),
        "the expression applies a partial operation off the domain its lowering is sound on"
    );
    assert_eq!(
        Hazard::MixedIntRealEquality(node).to_string(),
        "the expression compares a numeric literal for equality with an operand of another or \
         an unknown numeric kind"
    );
}

#[test]
fn hazard_node_is_the_refused_node_except_for_constants() {
    let node = build_literal(1);

    assert_eq!(Hazard::PartialOperation(node.clone()).node(), Some(&node));
    assert_eq!(
        Hazard::NativeConstant(vec![BuiltinConstant::E.identifier().clone()]).node(),
        None
    );
}
