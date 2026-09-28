//! Tests for the built-in catalogue in `fhy_core::expression::builtins`: the
//! function and constant enums and their catalogue order, lookups by name,
//! the declared sorts, the constant values, the parameters, and the exact
//! composed bodies and their printed text.
//!
//! Expected bodies are built with the node constructors
//! (`Expression::new_binary`, `Expression::piecewise`, ...), not the
//! operator builders, and compared structurally, so a literal's kind
//! matters: `0` and `0.0` are different literals.

use crate::support::expression as expression_support;

use std::collections::{HashMap, HashSet};
use std::thread;

use fhy_core::error::UnknownNameError;
use fhy_core::expression::builtins::{BuiltinConstant, BuiltinFunction, ComposedFunction};
use fhy_core::expression::evaluate::{Evaluator, Scalar};
use fhy_core::expression::registry::FunctionRegistry;
use fhy_core::expression::{
    BigInt, BinaryOperation, Callee, Expression, ExpressionKind, FormatOptions, FunctionSort,
    LiteralValue, LogicalOperation, Notation, UnaryOperation,
};
use fhy_core::identifier::Identifier;
use proptest::prelude::*;
use rstest::rstest;

use expression_support::{
    build_identifier, build_literal, build_piecewise_or_panic, expect_binary, expect_call,
    expect_literal, expect_piecewise, is_same_scalar,
};

const COMPOSED_NAMES: [&str; 16] = [
    "max",
    "min",
    "abs",
    "sign",
    "clamp",
    "clamp_symmetric",
    "relu",
    "leaky_relu",
    "xor",
    "nand",
    "nor",
    "implies",
    "iff",
    "sigmoid",
    "silu",
    "gelu",
];

const NATIVE_FUNCTION_NAMES: [&str; 19] = [
    "exp", "exp2", "log", "log2", "log10", "sqrt", "sin", "cos", "tan", "arcsin", "arccos",
    "arctan", "sinh", "cosh", "tanh", "erf", "round", "floor", "ceil",
];

const NATIVE_CONSTANT_NAMES: [&str; 4] = ["pi", "e", "inf", "nan"];

/// Number of parameters across all composed functions.
const TOTAL_PARAMETER_COUNT: usize = 27;

fn find_composed(name: &str) -> &'static ComposedFunction {
    name.parse::<BuiltinFunction>()
        .ok()
        .and_then(BuiltinFunction::composed)
        .unwrap_or_else(|| panic!("{name} is a composed built-in"))
}

fn collect_composed() -> Vec<&'static ComposedFunction> {
    BuiltinFunction::iter()
        .filter_map(BuiltinFunction::composed)
        .collect()
}

fn build_parameter_references(function: &ComposedFunction) -> Vec<Expression> {
    function
        .parameters()
        .iter()
        .cloned()
        .map(Expression::from)
        .collect()
}

fn build_expected_body(name: &str, parameters: &[Expression]) -> Expression {
    match name {
        "max" | "min" | "abs" | "sign" | "leaky_relu" => {
            build_expected_piecewise_body(name, parameters)
        }
        "xor" | "nand" | "nor" | "implies" | "iff" => build_expected_boolean_body(name, parameters),
        _ => build_expected_arithmetic_body(name, parameters),
    }
}

/// Return `comparison || (operand != operand)`, the NaN test of `max` and
/// `min`.
fn build_or_is_nan(comparison: Expression, operand: &Expression) -> Expression {
    Expression::new_logical(
        LogicalOperation::Or,
        [
            comparison,
            Expression::new_binary(BinaryOperation::NotEqual, operand, operand),
        ],
    )
}

fn build_expected_piecewise_body(name: &str, parameters: &[Expression]) -> Expression {
    let zero_float = build_literal(0.0);
    match (name, parameters) {
        ("max", [a, b]) => build_piecewise_or_panic(
            vec![(
                build_or_is_nan(Expression::new_binary(BinaryOperation::Greater, a, b), a),
                a.clone(),
            )],
            b.clone(),
        ),
        ("min", [a, b]) => build_piecewise_or_panic(
            vec![(
                build_or_is_nan(Expression::new_binary(BinaryOperation::Less, a, b), a),
                a.clone(),
            )],
            b.clone(),
        ),
        ("abs", [x]) => build_piecewise_or_panic(
            vec![(
                Expression::new_binary(BinaryOperation::Greater, x, &zero_float),
                x.clone(),
            )],
            Expression::new_binary(BinaryOperation::Subtract, build_literal(0), x),
        ),
        ("sign", [x]) => build_piecewise_or_panic(
            vec![
                (
                    Expression::new_binary(BinaryOperation::Greater, x, &zero_float),
                    build_literal(1),
                ),
                (
                    Expression::new_binary(BinaryOperation::Less, x, &zero_float),
                    build_literal(-1),
                ),
            ],
            build_literal(0),
        ),
        ("leaky_relu", [x, slope]) => build_piecewise_or_panic(
            vec![(
                Expression::new_binary(BinaryOperation::Greater, x, &zero_float),
                x.clone(),
            )],
            Expression::new_binary(BinaryOperation::Multiply, x, slope),
        ),
        _ => panic!(
            "no piecewise body for {name} with {} parameters",
            parameters.len()
        ),
    }
}

fn build_expected_boolean_body(name: &str, parameters: &[Expression]) -> Expression {
    match (name, parameters) {
        ("xor", [a, b]) => Expression::new_logical(
            LogicalOperation::And,
            [
                Expression::new_logical(LogicalOperation::Or, [a, b]),
                Expression::new_unary(
                    UnaryOperation::LogicalNot,
                    Expression::new_logical(LogicalOperation::And, [a, b]),
                ),
            ],
        ),
        ("nand", [a, b]) => Expression::new_unary(
            UnaryOperation::LogicalNot,
            Expression::new_logical(LogicalOperation::And, [a, b]),
        ),
        ("nor", [a, b]) => Expression::new_unary(
            UnaryOperation::LogicalNot,
            Expression::new_logical(LogicalOperation::Or, [a, b]),
        ),
        ("implies", [a, b]) => Expression::new_logical(
            LogicalOperation::Or,
            [&Expression::new_unary(UnaryOperation::LogicalNot, a), b],
        ),
        ("iff", [a, b]) => Expression::new_binary(BinaryOperation::Equal, a, b),
        _ => panic!(
            "no Boolean body for {name} with {} parameters",
            parameters.len()
        ),
    }
}

fn build_expected_arithmetic_body(name: &str, parameters: &[Expression]) -> Expression {
    let one_float = build_literal(1.0);
    match (name, parameters) {
        ("clamp", [x, lo, hi]) => Expression::call(
            BuiltinFunction::Min,
            [
                Expression::call(BuiltinFunction::Max, [x.clone(), lo.clone()]),
                hi.clone(),
            ],
        ),
        ("clamp_symmetric", [x, bound]) => Expression::call(
            BuiltinFunction::Clamp,
            [
                x.clone(),
                Expression::new_unary(UnaryOperation::Negate, bound),
                bound.clone(),
            ],
        ),
        ("relu", [x]) => Expression::call(BuiltinFunction::Max, [x.clone(), build_literal(0)]),
        ("sigmoid", [x]) => Expression::new_binary(
            BinaryOperation::Divide,
            &one_float,
            Expression::new_binary(
                BinaryOperation::Add,
                &one_float,
                Expression::call(
                    BuiltinFunction::Exp,
                    [Expression::new_unary(UnaryOperation::Negate, x)],
                ),
            ),
        ),
        ("silu", [x]) => Expression::new_binary(
            BinaryOperation::Multiply,
            x,
            Expression::call(BuiltinFunction::Sigmoid, [x]),
        ),
        ("gelu", [x]) => Expression::new_binary(
            BinaryOperation::Multiply,
            Expression::new_binary(BinaryOperation::Multiply, build_literal(0.5), x),
            Expression::new_binary(
                BinaryOperation::Add,
                &one_float,
                Expression::call(
                    BuiltinFunction::Erf,
                    [Expression::new_binary(
                        BinaryOperation::Divide,
                        x,
                        Expression::call(BuiltinFunction::Sqrt, [build_literal(2.0)]),
                    )],
                ),
            ),
        ),
        _ => panic!(
            "no arithmetic body for {name} with {} parameters",
            parameters.len()
        ),
    }
}

fn collect_callees(expression: &Expression) -> Vec<Callee> {
    let mut callees = Vec::new();
    let mut pending = vec![expression];
    while let Some(node) = pending.pop() {
        if let ExpressionKind::Call(call) = node.kind() {
            callees.push(call.callee().clone());
        }
        pending.extend(node.children());
    }
    callees
}

fn find_right_literal(expression: &Expression) -> &LiteralValue {
    let node = expect_binary(expression);
    let ExpressionKind::Literal(literal) = node.right().kind() else {
        panic!("expected a literal right operand, got {:?}", node.right());
    };
    literal
}

fn collect_parameter_ids() -> Vec<(BuiltinFunction, Vec<u64>)> {
    collect_composed()
        .into_iter()
        .map(|function| {
            (
                function.function(),
                function.parameters().iter().map(Identifier::id).collect(),
            )
        })
        .collect()
}

/// Test the composed built-ins lead the catalogue, in catalogue order.
#[test]
fn composed_functions_lists_the_composed_builtins_in_catalogue_order() {
    let names: Vec<&str> = collect_composed()
        .into_iter()
        .map(|function| function.function().name())
        .collect();
    let leading: Vec<&str> = BuiltinFunction::iter()
        .take(COMPOSED_NAMES.len())
        .map(BuiltinFunction::name)
        .collect();

    assert_eq!(names, COMPOSED_NAMES);
    assert_eq!(leading, COMPOSED_NAMES);
}

/// Test the native built-ins follow the composed ones, in catalogue order.
#[test]
fn native_functions_lists_the_native_builtins_in_catalogue_order() {
    let names: Vec<&str> = BuiltinFunction::iter()
        .filter(|function| function.composed().is_none())
        .map(BuiltinFunction::name)
        .collect();
    let trailing: Vec<&str> = BuiltinFunction::iter()
        .skip(COMPOSED_NAMES.len())
        .map(BuiltinFunction::name)
        .collect();

    assert_eq!(names, NATIVE_FUNCTION_NAMES);
    assert_eq!(trailing, NATIVE_FUNCTION_NAMES);
}

#[test]
fn native_constants_lists_the_builtin_constants_in_catalogue_order() {
    let names: Vec<&str> = BuiltinConstant::iter().map(BuiltinConstant::name).collect();

    assert_eq!(BuiltinConstant::iter().len(), NATIVE_CONSTANT_NAMES.len());
    assert_eq!(names, NATIVE_CONSTANT_NAMES);
}

#[test]
fn builtin_names_are_unique_and_round_trip_through_from_str() {
    let names: Vec<&str> = BuiltinFunction::iter()
        .map(BuiltinFunction::name)
        .chain(BuiltinConstant::iter().map(BuiltinConstant::name))
        .collect();
    let distinct: HashSet<&str> = names.iter().copied().collect();

    assert_eq!(BuiltinFunction::iter().len(), 35);
    assert_eq!(names.len(), 39);
    assert_eq!(distinct.len(), names.len(), "a name repeats in {names:?}");
    for function in BuiltinFunction::iter() {
        assert_eq!(function.name().parse::<BuiltinFunction>(), Ok(function));
        assert_eq!(function.to_string(), function.name());
    }
    for constant in BuiltinConstant::iter() {
        assert_eq!(constant.name().parse::<BuiltinConstant>(), Ok(constant));
        assert_eq!(constant.to_string(), constant.name());
    }
}

/// Test a lookup by name finds the composed entry of the same variant.
#[test]
fn builtin_function_composed_returns_the_listed_entry() {
    for (function, name) in BuiltinFunction::iter().zip(COMPOSED_NAMES) {
        let found = find_composed(name);

        let composed = function
            .composed()
            .unwrap_or_else(|| panic!("{name} is composed"));
        assert_eq!(found.function(), function);
        assert!(
            std::ptr::eq(found, composed),
            "{name} is not the listed entry"
        );
    }
}

/// Test `FromStr` refuses a name no built-in function has, naming it.
#[rstest]
#[case::constant("pi")]
#[case::empty("")]
#[case::other_case("Max")]
#[case::trailing_space("max ")]
#[case::prefix("ma")]
#[case::extension("maximum")]
#[case::other_spelling("asin")]
#[case::unknown("softplus")]
fn builtin_function_from_str_refuses_other_names(#[case] name: &str) {
    let parsed = name.parse::<BuiltinFunction>();

    let error: UnknownNameError = parsed.expect_err("the name is no built-in function's");
    assert_eq!(error.name(), name);
    assert_eq!(
        error.to_string(),
        format!("unknown built-in function `{name}`")
    );
}

#[rstest]
#[case::native_function("sqrt")]
#[case::composed_function("max")]
#[case::empty("")]
#[case::other_case("PI")]
#[case::other_spelling("infinity")]
#[case::other_case_nan("NaN")]
fn builtin_constant_from_str_refuses_other_names(#[case] name: &str) {
    let parsed = name.parse::<BuiltinConstant>();

    let error = parsed.expect_err("the name is no built-in constant's");
    assert_eq!(
        error.to_string(),
        format!("unknown built-in constant `{name}`")
    );
}

#[rstest]
#[case::max("max", &["a", "b"], &[FunctionSort::Real, FunctionSort::Real], FunctionSort::Real)]
#[case::min("min", &["a", "b"], &[FunctionSort::Real, FunctionSort::Real], FunctionSort::Real)]
#[case::abs("abs", &["x"], &[FunctionSort::Real], FunctionSort::Real)]
#[case::sign("sign", &["x"], &[FunctionSort::Real], FunctionSort::Int)]
#[case::clamp(
    "clamp",
    &["x", "lo", "hi"],
    &[FunctionSort::Real, FunctionSort::Real, FunctionSort::Real],
    FunctionSort::Real
)]
#[case::clamp_symmetric(
    "clamp_symmetric",
    &["x", "bound"],
    &[FunctionSort::Real, FunctionSort::Real],
    FunctionSort::Real
)]
#[case::relu("relu", &["x"], &[FunctionSort::Real], FunctionSort::Real)]
#[case::leaky_relu(
    "leaky_relu",
    &["x", "slope"],
    &[FunctionSort::Real, FunctionSort::Real],
    FunctionSort::Real
)]
#[case::xor("xor", &["a", "b"], &[FunctionSort::Bool, FunctionSort::Bool], FunctionSort::Bool)]
#[case::nand("nand", &["a", "b"], &[FunctionSort::Bool, FunctionSort::Bool], FunctionSort::Bool)]
#[case::nor("nor", &["a", "b"], &[FunctionSort::Bool, FunctionSort::Bool], FunctionSort::Bool)]
#[case::implies(
    "implies",
    &["a", "b"],
    &[FunctionSort::Bool, FunctionSort::Bool],
    FunctionSort::Bool
)]
#[case::iff("iff", &["a", "b"], &[FunctionSort::Bool, FunctionSort::Bool], FunctionSort::Bool)]
#[case::sigmoid("sigmoid", &["x"], &[FunctionSort::Real], FunctionSort::Real)]
#[case::silu("silu", &["x"], &[FunctionSort::Real], FunctionSort::Real)]
#[case::gelu("gelu", &["x"], &[FunctionSort::Real], FunctionSort::Real)]
fn composed_function_declares_its_documented_signature(
    #[case] name: &str,
    #[case] parameter_names: &[&str],
    #[case] parameter_sorts: &[FunctionSort],
    #[case] result_sort: FunctionSort,
) {
    let function = find_composed(name);

    let names: Vec<&str> = function
        .parameters()
        .iter()
        .map(Identifier::name_hint)
        .collect();
    assert_eq!(names, parameter_names);
    assert_eq!(function.function().parameter_sorts(), parameter_sorts);
    assert_eq!(function.function().result_sort(), result_sort);
}

/// Test each native built-in takes one real argument and declares its
/// documented result sort.
#[rstest]
#[case::exp("exp", FunctionSort::Real)]
#[case::exp2("exp2", FunctionSort::Real)]
#[case::log("log", FunctionSort::Real)]
#[case::log2("log2", FunctionSort::Real)]
#[case::log10("log10", FunctionSort::Real)]
#[case::sqrt("sqrt", FunctionSort::Real)]
#[case::sin("sin", FunctionSort::Real)]
#[case::cos("cos", FunctionSort::Real)]
#[case::tan("tan", FunctionSort::Real)]
#[case::arcsin("arcsin", FunctionSort::Real)]
#[case::arccos("arccos", FunctionSort::Real)]
#[case::arctan("arctan", FunctionSort::Real)]
#[case::sinh("sinh", FunctionSort::Real)]
#[case::cosh("cosh", FunctionSort::Real)]
#[case::tanh("tanh", FunctionSort::Real)]
#[case::erf("erf", FunctionSort::Real)]
#[case::round("round", FunctionSort::Int)]
#[case::floor("floor", FunctionSort::Int)]
#[case::ceil("ceil", FunctionSort::Int)]
fn native_function_declares_its_documented_signature(
    #[case] name: &str,
    #[case] result_sort: FunctionSort,
) {
    let function: BuiltinFunction = name.parse().expect("the name is a built-in function's");

    assert!(function.composed().is_none(), "{name} is native");
    assert_eq!(function.parameter_sorts(), [FunctionSort::Real]);
    assert_eq!(function.result_sort(), result_sort);
}

#[test]
fn native_constant_is_real() {
    for constant in BuiltinConstant::iter() {
        assert_eq!(constant.sort(), FunctionSort::Real, "{constant}");
    }
}

#[rstest]
#[case::pi(BuiltinConstant::Pi, std::f64::consts::PI)]
#[case::e(BuiltinConstant::E, std::f64::consts::E)]
#[case::inf(BuiltinConstant::Inf, f64::INFINITY)]
fn native_constant_holds_its_exact_value(#[case] constant: BuiltinConstant, #[case] expected: f64) {
    assert_eq!(
        constant.value().to_bits(),
        expected.to_bits(),
        "{constant} holds {}",
        constant.value()
    );
}

/// Test the `nan` constant holds the positive quiet NaN with no payload.
#[test]
fn native_constant_nan_holds_the_quiet_nan() {
    let constant: BuiltinConstant = "nan".parse().expect("nan is a constant");

    assert_eq!(constant.value().to_bits(), 0x7ff8_0000_0000_0000);
}

/// Test each composed built-in's body is exactly its documented tree over its
/// own parameters, literal kinds included.
#[test]
fn composed_function_body_is_the_documented_tree() {
    for function in collect_composed() {
        let name = function.function().name();

        let expected = build_expected_body(name, &build_parameter_references(function));

        assert_eq!(function.body(), &expected, "{name}");
    }
}

#[rstest]
#[case::max(
    "max",
    "{a if ((a > b) || (a != a)); b otherwise}",
    "(piecewise (or (greater a b) (not_equal a a)) a b)"
)]
#[case::min(
    "min",
    "{a if ((a < b) || (a != a)); b otherwise}",
    "(piecewise (or (less a b) (not_equal a a)) a b)"
)]
#[case::abs(
    "abs",
    "{x if (x > 0); (0 - x) otherwise}",
    "(piecewise (greater x 0) x (subtract 0 x))"
)]
#[case::sign(
    "sign",
    "{1 if (x > 0); -1 if (x < 0); 0 otherwise}",
    "(piecewise (greater x 0) 1 (less x 0) -1 0)"
)]
#[case::clamp("clamp", "min(max(x, lo), hi)", "(min (max x lo) hi)")]
#[case::clamp_symmetric(
    "clamp_symmetric",
    "clamp(x, (-bound), bound)",
    "(clamp x (negate bound) bound)"
)]
#[case::relu("relu", "max(x, 0)", "(max x 0)")]
#[case::leaky_relu(
    "leaky_relu",
    "{x if (x > 0); (x * slope) otherwise}",
    "(piecewise (greater x 0) x (multiply x slope))"
)]
#[case::xor(
    "xor",
    "((a || b) && (!(a && b)))",
    "(and (or a b) (logical_not (and a b)))"
)]
#[case::nand("nand", "(!(a && b))", "(logical_not (and a b))")]
#[case::nor("nor", "(!(a || b))", "(logical_not (or a b))")]
#[case::implies("implies", "((!a) || b)", "(or (logical_not a) b)")]
#[case::iff("iff", "(a == b)", "(equal a b)")]
#[case::sigmoid(
    "sigmoid",
    "(1 / (1 + exp((-x))))",
    "(divide 1 (add 1 (exp (negate x))))"
)]
#[case::silu("silu", "(x * sigmoid(x))", "(multiply x (sigmoid x))")]
#[case::gelu(
    "gelu",
    "((0.5 * x) * (1 + erf((x / sqrt(2)))))",
    "(multiply (multiply 0.5 x) (add 1 (erf (divide x (sqrt 2)))))"
)]
fn composed_function_body_prints_as_its_documented_text(
    #[case] name: &str,
    #[case] expected_symbolic: &str,
    #[case] expected_functional: &str,
) {
    let body = find_composed(name).body();

    let symbolic = body
        .display(FormatOptions::default().with_notation(Notation::Symbolic))
        .to_string();
    let functional = body
        .display(FormatOptions::default().with_notation(Notation::Functional))
        .to_string();

    assert_eq!(symbolic, expected_symbolic);
    assert_eq!(functional, expected_functional);
}

#[test]
fn composed_function_body_refers_to_exactly_its_parameters() {
    for function in collect_composed() {
        let free = function.body().free_identifiers();

        let parameters: HashSet<Identifier> = function.parameters().iter().cloned().collect();
        assert_eq!(free, parameters, "{}", function.function());
    }
}

#[test]
fn composed_function_body_calls_only_builtin_functions() {
    for function in collect_composed() {
        let callees = collect_callees(function.body());

        let named: Vec<&Callee> = callees
            .iter()
            .filter(|callee| !matches!(callee, Callee::Builtin(_)))
            .collect();
        assert!(
            named.is_empty(),
            "{} calls non-built-ins {named:?}",
            function.function()
        );
    }
}

#[test]
fn composed_function_relu_passes_an_integer_zero_to_max() {
    let relu = find_composed("relu");

    let call = expect_call(relu.body());

    assert_eq!(call.callee(), &Callee::Builtin(BuiltinFunction::Max));
    assert_eq!(expect_literal(&call.arguments()[1]), &LiteralValue::from(0));
}

/// Test `abs` compares against a positive float zero.
#[test]
fn composed_function_abs_compares_against_a_float_zero() {
    let abs = find_composed("abs");

    let piecewise = expect_piecewise(abs.body());

    let LiteralValue::Float(zero) = find_right_literal(&piecewise.cases()[0].0) else {
        panic!("abs compares against a float");
    };
    assert_eq!(zero.to_bits(), 0.0_f64.to_bits());
}

/// Test `sign` compares against float zeros but yields the integers 1, -1
/// and 0.
#[test]
fn composed_function_sign_yields_integer_literals() {
    let sign = find_composed("sign");

    let piecewise = expect_piecewise(sign.body());

    let condition_bits: Vec<Option<u64>> = piecewise
        .cases()
        .iter()
        .map(|(condition, _)| match find_right_literal(condition) {
            LiteralValue::Float(value) => Some(value.to_bits()),
            _ => None,
        })
        .collect();
    let values: Vec<Option<&BigInt>> = piecewise
        .cases()
        .iter()
        .map(|(_, value)| expect_literal(value))
        .chain([expect_literal(piecewise.otherwise())])
        .map(|literal| match literal {
            LiteralValue::Int(value) => Some(value),
            _ => None,
        })
        .collect();
    assert_eq!(condition_bits, [Some(0.0_f64.to_bits()); 2]);
    assert_eq!(
        values,
        [
            Some(&BigInt::from(1)),
            Some(&BigInt::from(-1)),
            Some(&BigInt::from(0)),
        ]
    );
}

/// Test no two parameters in the catalogue share an identifier, even when
/// they share a name.
#[test]
fn composed_function_parameters_are_distinct_across_the_catalogue() {
    let ids: Vec<u64> = collect_parameter_ids()
        .into_iter()
        .flat_map(|(_, ids)| ids)
        .collect();

    let distinct: HashSet<u64> = ids.iter().copied().collect();

    assert_eq!(ids.len(), TOTAL_PARAMETER_COUNT);
    assert_eq!(
        distinct.len(),
        ids.len(),
        "a parameter id repeats in {ids:?}"
    );
}

#[test]
fn composed_functions_returns_the_same_table_on_every_call() {
    let first = collect_composed();
    let first_ids = collect_parameter_ids();

    let second = collect_composed();

    for (first, second) in first.into_iter().zip(second) {
        assert!(std::ptr::eq(first, second), "the table was rebuilt");
    }
    assert_eq!(collect_parameter_ids(), first_ids);
}

#[test]
fn composed_functions_agrees_across_threads() {
    let handles: Vec<_> = (0..8)
        .map(|_| thread::spawn(collect_parameter_ids))
        .collect();

    let observed: Vec<Vec<(BuiltinFunction, Vec<u64>)>> = handles
        .into_iter()
        .map(|handle| handle.join().expect("the reader thread finishes"))
        .collect();

    let reference = collect_parameter_ids();
    for (index, parameter_ids) in observed.iter().enumerate() {
        assert_eq!(
            parameter_ids, &reference,
            "thread {index} saw other parameters"
        );
    }
}

#[test]
fn composed_function_parameters_differ_from_new_identifiers() {
    let fresh_x = Identifier::new("x");
    let fresh_a = Identifier::new("a");

    let parameters: Vec<&Identifier> = collect_composed()
        .into_iter()
        .flat_map(ComposedFunction::parameters)
        .collect();

    assert!(
        !parameters.contains(&&fresh_x),
        "{fresh_x:?} is a parameter"
    );
    assert!(
        !parameters.contains(&&fresh_a),
        "{fresh_a:?} is a parameter"
    );
}

/// Test substituting literals for `max`'s parameters, as inlining `max(1, 2)`
/// does, yields `{1 if (1 > 2); 2 otherwise}`.
#[test]
fn composed_function_max_body_with_literal_arguments_yields_the_literal_piecewise() {
    let max = find_composed("max");
    let arguments = HashMap::from([
        (max.parameters()[0].clone(), build_literal(1)),
        (max.parameters()[1].clone(), build_literal(2)),
    ]);

    let inlined = max
        .body()
        .substitute(&arguments)
        .expect("literal arguments substitute into max");

    let expected = build_piecewise_or_panic(
        vec![(
            build_or_is_nan(
                Expression::new_binary(
                    BinaryOperation::Greater,
                    build_literal(1),
                    build_literal(2),
                ),
                &build_literal(1),
            ),
            build_literal(1),
        )],
        build_literal(2),
    );
    assert_eq!(inlined, expected);
}

/// Test the clamp `max(low, min(value, high))`, inlined through the `min`
/// and `max` bodies, yields the nested piecewise.
#[test]
fn composed_function_max_of_min_inlines_to_a_nested_clamp() {
    let (_, low) = build_identifier("low");
    let (_, high) = build_identifier("high");
    let (_, value) = build_identifier("value");
    let min = find_composed("min");
    let max = find_composed("max");

    let inner = min
        .body()
        .substitute(&HashMap::from([
            (min.parameters()[0].clone(), value.clone()),
            (min.parameters()[1].clone(), high.clone()),
        ]))
        .expect("identifier arguments substitute into min");
    let inlined = max
        .body()
        .substitute(&HashMap::from([
            (max.parameters()[0].clone(), low.clone()),
            (max.parameters()[1].clone(), inner),
        ]))
        .expect("the inner clamp substitutes into max");

    let inner_min = build_piecewise_or_panic(
        vec![(
            build_or_is_nan(
                Expression::new_binary(BinaryOperation::Less, &value, &high),
                &value,
            ),
            value,
        )],
        high,
    );
    let expected = build_piecewise_or_panic(
        vec![(
            build_or_is_nan(
                Expression::new_binary(BinaryOperation::Greater, &low, &inner_min),
                &low,
            ),
            low,
        )],
        inner_min,
    );
    assert_eq!(inlined, expected);
}

#[test]
fn build_piecewise_guards_a_fast_path_with_a_fallback() {
    let (_, x) = build_identifier("x");
    let (_, sqrt_path) = build_identifier("sqrt_path");
    let (_, fallback_path) = build_identifier("fallback_path");

    let expression =
        Expression::piecewise([(x.greater(0), sqrt_path.clone())], fallback_path.clone())
            .expect("one guarded case builds");

    let expected = build_piecewise_or_panic(
        vec![(
            Expression::new_binary(BinaryOperation::Greater, &x, build_literal(0)),
            sqrt_path,
        )],
        fallback_path,
    );
    assert_eq!(expression, expected);
}

// ---------------------------------------------------------------------------
// The constants' identifiers
// ---------------------------------------------------------------------------

#[rstest]
#[case::pi(BuiltinConstant::Pi, 48)]
#[case::e(BuiltinConstant::E, 49)]
#[case::inf(BuiltinConstant::Inf, 50)]
#[case::nan(BuiltinConstant::Nan, 51)]
fn builtin_constant_identifier_holds_its_reserved_id_and_name(
    #[case] constant: BuiltinConstant,
    #[case] reserved_id: u64,
) {
    let identifier = constant.identifier();

    assert_eq!(identifier.id(), reserved_id);
    assert_eq!(identifier.name_hint(), constant.name());
}

#[test]
fn builtin_constant_identifier_is_one_value_across_calls_and_threads() {
    let first = BuiltinConstant::Pi.identifier();
    let from_thread = thread::spawn(|| BuiltinConstant::Pi.identifier().clone())
        .join()
        .expect("the thread finishes");

    assert!(std::ptr::eq(first, BuiltinConstant::Pi.identifier()));
    assert_eq!(first, &from_thread);
}

#[test]
fn builtin_constant_identifiers_are_distinct() {
    let identifiers: HashSet<&Identifier> = BuiltinConstant::iter()
        .map(BuiltinConstant::identifier)
        .collect();

    assert_eq!(identifiers.len(), BuiltinConstant::iter().len());
}

#[test]
fn builtin_constant_of_identifier_inverts_identifier() {
    for constant in BuiltinConstant::iter() {
        assert_eq!(
            BuiltinConstant::of_identifier(constant.identifier()),
            Some(constant)
        );
    }
}

#[test]
fn builtin_constant_of_identifier_goes_by_the_id_not_the_name() {
    let look_alike = Identifier::new("pi");
    let restored = Identifier::try_restore(48, "renamed").expect("a reserved id restores");
    let other_reserved = Identifier::try_restore(52, "pi").expect("a reserved id restores");

    assert_eq!(BuiltinConstant::of_identifier(&look_alike), None);
    assert_eq!(
        BuiltinConstant::of_identifier(&restored),
        Some(BuiltinConstant::Pi)
    );
    assert_eq!(BuiltinConstant::of_identifier(&other_reserved), None);
}

// ---------------------------------------------------------------------------
// Native kernels
// ---------------------------------------------------------------------------

#[rstest]
#[case::exp(BuiltinFunction::Exp, 1.0, 1.0_f64.exp())]
#[case::exp2(BuiltinFunction::Exp2, 3.0, 8.0)]
#[case::log(BuiltinFunction::Log, std::f64::consts::E, 1.0)]
#[case::log2(BuiltinFunction::Log2, 8.0, 3.0)]
#[case::log10(BuiltinFunction::Log10, 1000.0, 3.0)]
#[case::sqrt(BuiltinFunction::Sqrt, 2.25, 1.5)]
#[case::sin(BuiltinFunction::Sin, 0.5, 0.5_f64.sin())]
#[case::cos(BuiltinFunction::Cos, 0.5, 0.5_f64.cos())]
#[case::tan(BuiltinFunction::Tan, 0.5, 0.5_f64.tan())]
#[case::arcsin(BuiltinFunction::Arcsin, 0.5, 0.5_f64.asin())]
#[case::arccos(BuiltinFunction::Arccos, 0.5, 0.5_f64.acos())]
#[case::arctan(BuiltinFunction::Arctan, 0.5, 0.5_f64.atan())]
#[case::sinh(BuiltinFunction::Sinh, 0.5, 0.5_f64.sinh())]
#[case::cosh(BuiltinFunction::Cosh, 0.5, 0.5_f64.cosh())]
#[case::tanh(BuiltinFunction::Tanh, 0.5, 0.5_f64.tanh())]
#[case::erf(BuiltinFunction::Erf, 0.5, 0.520_499_877_813_046_5)]
#[case::round(BuiltinFunction::Round, 1.6, 2.0)]
#[case::floor(BuiltinFunction::Floor, -1.5, -2.0)]
#[case::ceil(BuiltinFunction::Ceil, -1.5, -1.0)]
fn native_builtin_computes_its_function(
    #[case] function: BuiltinFunction,
    #[case] argument: f64,
    #[case] expected: f64,
) {
    let value = function.native_value(argument).expect("a native built-in");

    assert!(
        (value - expected).abs() <= 4.0 * f64::EPSILON * expected.abs().max(1.0),
        "{} at {argument} is {value}, expected {expected}",
        function.name()
    );
}

#[rstest]
#[case(0.5, 0.0)]
#[case(1.5, 2.0)]
#[case(2.5, 2.0)]
#[case(-0.5, -0.0)]
#[case(-1.5, -2.0)]
fn round_rounds_half_to_even(#[case] argument: f64, #[case] expected: f64) {
    assert_eq!(
        BuiltinFunction::Round.native_value(argument),
        Some(expected)
    );
}

#[test]
fn native_builtins_follow_ieee_outside_their_domains() {
    let is_nan = |function: BuiltinFunction, argument: f64| {
        function.native_value(argument).is_some_and(f64::is_nan)
    };

    assert!(is_nan(BuiltinFunction::Sqrt, -1.0));
    assert!(is_nan(BuiltinFunction::Log, -1.0));
    assert!(is_nan(BuiltinFunction::Arcsin, 2.0));
    assert_eq!(
        BuiltinFunction::Log.native_value(0.0),
        Some(f64::NEG_INFINITY)
    );
    assert_eq!(
        BuiltinFunction::Exp.native_value(1000.0),
        Some(f64::INFINITY)
    );
    assert_eq!(
        BuiltinFunction::Floor.native_value(f64::INFINITY),
        Some(f64::INFINITY)
    );
    assert!(is_nan(BuiltinFunction::Round, f64::NAN));
}

#[test]
fn composed_builtins_have_no_native_value() {
    for function in BuiltinFunction::iter() {
        assert_eq!(
            function.native_value(1.0).is_none(),
            function.composed().is_some(),
            "{}",
            function.name()
        );
    }
}

// =============================================================================
// NaN through the composed built-ins
// =============================================================================

/// Return the scalar evaluation of `function` over `arguments`, each bound
/// to a parameter of its own.
fn evaluate_builtin(function: BuiltinFunction, arguments: &[Scalar]) -> Scalar {
    let mut environment = HashMap::new();
    let mut references = Vec::new();
    for (index, argument) in arguments.iter().enumerate() {
        let (identifier, reference) = build_identifier(&format!("p{index}"));
        environment.insert(identifier, *argument);
        references.push(reference);
    }
    Evaluator::new(&FunctionRegistry::new())
        .evaluate(&Expression::call(function, references), &environment)
        .unwrap_or_else(|error| panic!("{} evaluates: {error}", function.name()))
}

/// Test NaN propagates through the composed built-ins whichever operand it
/// is, as `NumPy`'s `maximum` does, `abs` of either zero is the positive
/// zero, and `sign(nan)` stays `0`.
#[rstest]
#[case::max_nan_first(BuiltinFunction::Max, &[Scalar::Real(f64::NAN), Scalar::Int(1)], Scalar::Real(f64::NAN))]
#[case::max_nan_second(BuiltinFunction::Max, &[Scalar::Int(1), Scalar::Real(f64::NAN)], Scalar::Real(f64::NAN))]
#[case::min_nan_first(BuiltinFunction::Min, &[Scalar::Real(f64::NAN), Scalar::Int(1)], Scalar::Real(f64::NAN))]
#[case::min_nan_second(BuiltinFunction::Min, &[Scalar::Int(1), Scalar::Real(f64::NAN)], Scalar::Real(f64::NAN))]
#[case::max_of_integers(BuiltinFunction::Max, &[Scalar::Int(2), Scalar::Int(5)], Scalar::Int(5))]
#[case::min_of_reals(BuiltinFunction::Min, &[Scalar::Real(2.5), Scalar::Real(-1.0)], Scalar::Real(-1.0))]
#[case::relu_nan(BuiltinFunction::Relu, &[Scalar::Real(f64::NAN)], Scalar::Real(f64::NAN))]
#[case::relu_negative(BuiltinFunction::Relu, &[Scalar::Real(-3.0)], Scalar::Real(0.0))]
#[case::leaky_relu_nan(BuiltinFunction::LeakyRelu, &[Scalar::Real(f64::NAN), Scalar::Real(0.1)], Scalar::Real(f64::NAN))]
#[case::clamp_nan_value(BuiltinFunction::Clamp, &[Scalar::Real(f64::NAN), Scalar::Int(0), Scalar::Int(1)], Scalar::Real(f64::NAN))]
#[case::clamp_nan_bound(BuiltinFunction::Clamp, &[Scalar::Real(0.5), Scalar::Real(f64::NAN), Scalar::Int(1)], Scalar::Real(f64::NAN))]
#[case::clamp_symmetric_nan(BuiltinFunction::ClampSymmetric, &[Scalar::Real(f64::NAN), Scalar::Real(1.0)], Scalar::Real(f64::NAN))]
#[case::abs_negative_zero(BuiltinFunction::Abs, &[Scalar::Real(-0.0)], Scalar::Real(0.0))]
#[case::abs_positive_zero(BuiltinFunction::Abs, &[Scalar::Real(0.0)], Scalar::Real(0.0))]
#[case::abs_nan(BuiltinFunction::Abs, &[Scalar::Real(f64::NAN)], Scalar::Real(f64::NAN))]
#[case::abs_negative(BuiltinFunction::Abs, &[Scalar::Real(-2.5)], Scalar::Real(2.5))]
#[case::abs_negative_infinity(BuiltinFunction::Abs, &[Scalar::Real(f64::NEG_INFINITY)], Scalar::Real(f64::INFINITY))]
#[case::abs_integer(BuiltinFunction::Abs, &[Scalar::Int(-4)], Scalar::Int(4))]
#[case::sign_nan(BuiltinFunction::Sign, &[Scalar::Real(f64::NAN)], Scalar::Int(0))]
fn composed_builtins_propagate_nan(
    #[case] function: BuiltinFunction,
    #[case] arguments: &[Scalar],
    #[case] expected: Scalar,
) {
    let value = evaluate_builtin(function, arguments);

    assert!(is_same_scalar(value, expected), "{value:?} != {expected:?}");
}

/// Return a float, NaN, an infinity or a zero included.
fn any_float() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(f64::NAN),
        Just(f64::INFINITY),
        Just(f64::NEG_INFINITY),
        Just(0.0),
        Just(-0.0),
        -1e3..1e3_f64,
    ]
}

proptest! {
    /// `max` and `min` are commutative over floats with NaN: swapping
    /// the operands gives the same value, the zeros compared as equal.
    #[test]
    fn max_and_min_are_commutative_over_floats_with_nan(a in any_float(), b in any_float()) {
        for function in [BuiltinFunction::Max, BuiltinFunction::Min] {
            let forward = evaluate_builtin(function, &[Scalar::Real(a), Scalar::Real(b)]);
            let backward = evaluate_builtin(function, &[Scalar::Real(b), Scalar::Real(a)]);
            let (Scalar::Real(forward), Scalar::Real(backward)) = (forward, backward) else {
                panic!("{} of reals is real", function.name());
            };
            prop_assert!(
                forward.partial_cmp(&backward) == Some(std::cmp::Ordering::Equal)
                    || (forward.is_nan() && backward.is_nan()),
                "{}({a}, {b}) = {forward}, but {backward} swapped", function.name()
            );
        }
    }
}
