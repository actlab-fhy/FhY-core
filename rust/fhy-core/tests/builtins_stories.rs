//! Stories for the built-in catalogue in
//! `fhy_core::symbolic::expression::builtins`: the three tables and their
//! order, lookups by name, the declared sorts, the constant values, the
//! parameters, and the exact composed bodies and their printed text. Public
//! API only.
//!
//! Expected bodies are built with the node constructors
//! (`Expression::new_binary`, `PiecewiseExpression::try_new`, ...), not the
//! operator builders, and compared structurally, so a literal's kind
//! matters: `0` and `0.0` are different literals.

#[path = "common/expression.rs"]
pub mod expression_support;

use std::collections::{HashMap, HashSet};
use std::thread;

use fhy_core::identifier::Identifier;
use fhy_core::symbolic::expression::builtins::{
    ComposedFunction, NativeConstantSpec, NativeFunctionSignature, find_composed_function,
    find_native_constant, find_native_function, list_composed_functions, list_native_constants,
    list_native_functions,
};
use fhy_core::symbolic::expression::{
    BigInt, BinaryOperation, Expression, ExpressionKind, FormatOptions, FunctionSort, LiteralKind,
    Notation, UnaryOperation, build_piecewise, format_expression,
};
use rstest::rstest;

use expression_support::{
    build_call_node_or_panic, build_identifier, build_literal, build_piecewise_node_or_panic,
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

/// Return the composed built-in named `name`, failing the test if absent.
fn find_composed(name: &str) -> &'static ComposedFunction {
    find_composed_function(name).unwrap_or_else(|| panic!("{name} is a composed built-in"))
}

/// Return a reference expression to each parameter of `function`, in order.
fn build_parameter_references(function: &ComposedFunction) -> Vec<Expression> {
    function
        .parameters()
        .iter()
        .cloned()
        .map(Expression::from)
        .collect()
}

/// Return the documented body of the composed built-in `name` over
/// `parameters`, the references to its parameters in order.
fn build_expected_body(name: &str, parameters: &[Expression]) -> Expression {
    match name {
        "max" | "min" | "abs" | "sign" | "leaky_relu" => {
            build_expected_piecewise_body(name, parameters)
        }
        "xor" | "nand" | "nor" | "implies" | "iff" => build_expected_boolean_body(name, parameters),
        _ => build_expected_arithmetic_body(name, parameters),
    }
}

/// Return the documented piecewise body of `max`, `min`, `abs`, `sign` or
/// `leaky_relu` over `parameters`.
fn build_expected_piecewise_body(name: &str, parameters: &[Expression]) -> Expression {
    let zero_float = build_literal(0.0);
    match (name, parameters) {
        ("max", [a, b]) => build_piecewise_node_or_panic(
            vec![(
                Expression::new_binary(BinaryOperation::Greater, a, b),
                a.clone(),
            )],
            b.clone(),
        ),
        ("min", [a, b]) => build_piecewise_node_or_panic(
            vec![(
                Expression::new_binary(BinaryOperation::Less, a, b),
                a.clone(),
            )],
            b.clone(),
        ),
        ("abs", [x]) => build_piecewise_node_or_panic(
            vec![(
                Expression::new_binary(BinaryOperation::GreaterEqual, x, &zero_float),
                x.clone(),
            )],
            Expression::new_unary(UnaryOperation::Negate, x),
        ),
        ("sign", [x]) => build_piecewise_node_or_panic(
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
        ("leaky_relu", [x, slope]) => build_piecewise_node_or_panic(
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

/// Return the documented body of the Boolean connective `name` over
/// `parameters`.
fn build_expected_boolean_body(name: &str, parameters: &[Expression]) -> Expression {
    match (name, parameters) {
        ("xor", [a, b]) => Expression::new_binary(
            BinaryOperation::LogicalAnd,
            Expression::new_binary(BinaryOperation::LogicalOr, a, b),
            Expression::new_unary(
                UnaryOperation::LogicalNot,
                Expression::new_binary(BinaryOperation::LogicalAnd, a, b),
            ),
        ),
        ("nand", [a, b]) => Expression::new_unary(
            UnaryOperation::LogicalNot,
            Expression::new_binary(BinaryOperation::LogicalAnd, a, b),
        ),
        ("nor", [a, b]) => Expression::new_unary(
            UnaryOperation::LogicalNot,
            Expression::new_binary(BinaryOperation::LogicalOr, a, b),
        ),
        ("implies", [a, b]) => Expression::new_binary(
            BinaryOperation::LogicalOr,
            Expression::new_unary(UnaryOperation::LogicalNot, a),
            b,
        ),
        ("iff", [a, b]) => Expression::new_binary(BinaryOperation::Equal, a, b),
        _ => panic!(
            "no Boolean body for {name} with {} parameters",
            parameters.len()
        ),
    }
}

/// Return the documented body, built from calls and arithmetic, of `clamp`,
/// `clamp_symmetric`, `relu`, `sigmoid`, `silu` or `gelu` over `parameters`.
fn build_expected_arithmetic_body(name: &str, parameters: &[Expression]) -> Expression {
    let one_float = build_literal(1.0);
    match (name, parameters) {
        ("clamp", [x, lo, hi]) => build_call_node_or_panic(
            "min",
            vec![
                build_call_node_or_panic("max", vec![x.clone(), lo.clone()]),
                hi.clone(),
            ],
        ),
        ("clamp_symmetric", [x, bound]) => build_call_node_or_panic(
            "clamp",
            vec![
                x.clone(),
                Expression::new_unary(UnaryOperation::Negate, bound),
                bound.clone(),
            ],
        ),
        ("relu", [x]) => build_call_node_or_panic("max", vec![x.clone(), build_literal(0)]),
        ("sigmoid", [x]) => Expression::new_binary(
            BinaryOperation::Divide,
            &one_float,
            Expression::new_binary(
                BinaryOperation::Add,
                &one_float,
                build_call_node_or_panic(
                    "exp",
                    vec![Expression::new_unary(UnaryOperation::Negate, x)],
                ),
            ),
        ),
        ("silu", [x]) => Expression::new_binary(
            BinaryOperation::Multiply,
            x,
            build_call_node_or_panic("sigmoid", vec![x.clone()]),
        ),
        ("gelu", [x]) => Expression::new_binary(
            BinaryOperation::Multiply,
            Expression::new_binary(BinaryOperation::Multiply, build_literal(0.5), x),
            Expression::new_binary(
                BinaryOperation::Add,
                &one_float,
                build_call_node_or_panic(
                    "erf",
                    vec![Expression::new_binary(
                        BinaryOperation::Divide,
                        x,
                        build_call_node_or_panic("sqrt", vec![build_literal(2.0)]),
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

/// Return the function names of every call node in `expression`.
fn collect_call_names(expression: &Expression) -> Vec<String> {
    let mut names = Vec::new();
    let mut pending = vec![expression];
    while let Some(node) = pending.pop() {
        if let ExpressionKind::Call(call) = node.kind() {
            names.push(call.function_name().to_owned());
        }
        pending.extend(node.children());
    }
    names
}

/// Return the literal operand of a binary node, failing the test otherwise.
fn find_right_literal(expression: &Expression) -> LiteralKind<'_> {
    let ExpressionKind::Binary(node) = expression.kind() else {
        panic!("expected a binary node, got {expression:?}");
    };
    let ExpressionKind::Literal(literal) = node.right().kind() else {
        panic!("expected a literal right operand, got {:?}", node.right());
    };
    literal.kind()
}

/// Return the literal held by `expression`, failing the test otherwise.
fn find_literal(expression: &Expression) -> LiteralKind<'_> {
    let ExpressionKind::Literal(literal) = expression.kind() else {
        panic!("expected a literal, got {expression:?}");
    };
    literal.kind()
}

/// Return the ids of every composed function's parameters, in catalogue
/// order, each paired with its function's name.
fn collect_parameter_ids() -> Vec<(&'static str, Vec<u64>)> {
    list_composed_functions()
        .iter()
        .map(|function| {
            (
                function.name(),
                function.parameters().iter().map(Identifier::id).collect(),
            )
        })
        .collect()
}

/// Test the composed table lists the 16 composed built-ins in catalogue order.
#[test]
fn composed_functions_lists_the_composed_builtins_in_catalogue_order() {
    let names: Vec<&str> = list_composed_functions()
        .iter()
        .map(ComposedFunction::name)
        .collect();

    assert_eq!(names, COMPOSED_NAMES);
}

/// Test the native table lists the 19 native built-ins in catalogue order.
#[test]
fn native_functions_lists_the_native_builtins_in_catalogue_order() {
    let names: Vec<&str> = list_native_functions()
        .iter()
        .map(NativeFunctionSignature::name)
        .collect();

    assert_eq!(names, NATIVE_FUNCTION_NAMES);
}

/// Test the constant table lists `pi`, `e`, `inf`, `nan` in that order.
#[test]
fn native_constants_lists_the_builtin_constants_in_catalogue_order() {
    let names: Vec<&str> = list_native_constants()
        .iter()
        .map(NativeConstantSpec::name)
        .collect();

    assert_eq!(names, NATIVE_CONSTANT_NAMES);
}

/// Test no name appears twice across the three tables.
#[test]
fn builtin_catalogue_names_are_unique_across_the_tables() {
    let names: Vec<&str> = list_composed_functions()
        .iter()
        .map(ComposedFunction::name)
        .chain(
            list_native_functions()
                .iter()
                .map(NativeFunctionSignature::name),
        )
        .chain(list_native_constants().iter().map(NativeConstantSpec::name))
        .collect();

    let distinct: HashSet<&str> = names.iter().copied().collect();

    assert_eq!(names.len(), 39);
    assert_eq!(distinct.len(), names.len(), "a name repeats in {names:?}");
}

/// Test each composed built-in is found by its name as the listed entry.
#[rstest]
fn find_composed_function_returns_the_listed_entry(
    #[values(
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
        "gelu"
    )]
    name: &str,
) {
    let found = find_composed_function(name);

    let listed = list_composed_functions()
        .iter()
        .find(|function| function.name() == name)
        .unwrap_or_else(|| panic!("{name} is listed"));
    let found = found.unwrap_or_else(|| panic!("{name} is found"));
    assert_eq!(found.name(), name);
    assert!(
        std::ptr::eq(found, listed),
        "{name} is not the listed entry"
    );
}

/// Test each native built-in is found by its name as the listed entry.
#[rstest]
fn find_native_function_returns_the_listed_entry(
    #[values(
        "exp", "exp2", "log", "log2", "log10", "sqrt", "sin", "cos", "tan", "arcsin", "arccos",
        "arctan", "sinh", "cosh", "tanh", "erf", "round", "floor", "ceil"
    )]
    name: &str,
) {
    let found = find_native_function(name);

    let listed = list_native_functions()
        .iter()
        .find(|function| function.name() == name)
        .unwrap_or_else(|| panic!("{name} is listed"));
    let found = found.unwrap_or_else(|| panic!("{name} is found"));
    assert_eq!(found.name(), name);
    assert!(
        std::ptr::eq(found, listed),
        "{name} is not the listed entry"
    );
}

/// Test each built-in constant is found by its name as the listed entry.
#[rstest]
fn find_native_constant_returns_the_listed_entry(#[values("pi", "e", "inf", "nan")] name: &str) {
    let found = find_native_constant(name);

    let listed = list_native_constants()
        .iter()
        .find(|constant| constant.name() == name)
        .unwrap_or_else(|| panic!("{name} is listed"));
    let found = found.unwrap_or_else(|| panic!("{name} is found"));
    assert_eq!(found.name(), name);
    assert!(
        std::ptr::eq(found, listed),
        "{name} is not the listed entry"
    );
}

/// Test a name outside the composed table finds no composed function.
#[rstest]
#[case::native_function("exp")]
#[case::constant("pi")]
#[case::empty("")]
#[case::other_case("Max")]
#[case::trailing_space("max ")]
#[case::prefix("ma")]
#[case::extension("maximum")]
#[case::unknown("softplus")]
fn find_composed_function_rejects_other_names(#[case] name: &str) {
    let found = find_composed_function(name);

    assert!(found.is_none(), "{name:?} found {found:?}");
}

/// Test a name outside the native table finds no native function.
#[rstest]
#[case::composed_function("max")]
#[case::constant("e")]
#[case::empty("")]
#[case::other_case("Exp")]
#[case::leading_space(" exp")]
#[case::other_spelling("asin")]
fn find_native_function_rejects_other_names(#[case] name: &str) {
    let found = find_native_function(name);

    assert!(found.is_none(), "{name:?} found {found:?}");
}

/// Test a name outside the constant table finds no constant.
#[rstest]
#[case::native_function("sqrt")]
#[case::composed_function("max")]
#[case::empty("")]
#[case::other_case("PI")]
#[case::other_spelling("infinity")]
#[case::other_case_nan("NaN")]
fn find_native_constant_rejects_other_names(#[case] name: &str) {
    let found = find_native_constant(name);

    assert!(found.is_none(), "{name:?} found {found:?}");
}

/// Test each composed built-in declares its documented parameter names,
/// parameter sorts, and result sort.
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
    assert_eq!(function.parameter_sorts(), parameter_sorts);
    assert_eq!(function.result_sort(), result_sort);
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
    let function = find_native_function(name).unwrap_or_else(|| panic!("{name} is native"));

    assert_eq!(function.parameter_sorts(), [FunctionSort::Real]);
    assert_eq!(function.result_sort(), result_sort);
}

/// Test each built-in constant is real.
#[rstest]
fn native_constant_is_real(#[values("pi", "e", "inf", "nan")] name: &str) {
    let constant = find_native_constant(name).unwrap_or_else(|| panic!("{name} is a constant"));

    assert_eq!(constant.sort(), FunctionSort::Real);
}

/// Test the finite and infinite constants hold exactly their documented
/// values.
#[rstest]
#[case::pi("pi", std::f64::consts::PI)]
#[case::e("e", std::f64::consts::E)]
#[case::inf("inf", f64::INFINITY)]
fn native_constant_holds_its_exact_value(#[case] name: &str, #[case] expected: f64) {
    let constant = find_native_constant(name).unwrap_or_else(|| panic!("{name} is a constant"));

    assert_eq!(
        constant.value().to_bits(),
        expected.to_bits(),
        "{name} holds {}",
        constant.value()
    );
}

/// Test the `nan` constant holds the positive quiet NaN with no payload.
#[test]
fn native_constant_nan_holds_the_quiet_nan() {
    let constant = find_native_constant("nan").expect("nan is a constant");

    assert_eq!(constant.value().to_bits(), 0x7ff8_0000_0000_0000);
}

/// Test each composed built-in's body is exactly its documented tree over its
/// own parameters, literal kinds included.
#[rstest]
fn composed_function_body_is_the_documented_tree(
    #[values(
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
        "gelu"
    )]
    name: &str,
) {
    let function = find_composed(name);

    let expected = build_expected_body(name, &build_parameter_references(function));

    assert_eq!(function.body(), &expected);
}

/// Test each composed body prints, with identifiers by name hint, as its
/// documented symbolic and functional text.
#[rstest]
#[case::max("max", "{a if (a > b); b otherwise}", "(piecewise (greater a b) a b)")]
#[case::min("min", "{a if (a < b); b otherwise}", "(piecewise (less a b) a b)")]
#[case::abs(
    "abs",
    "{x if (x >= 0.0); (-x) otherwise}",
    "(piecewise (greater_equal x 0.0) x (negate x))"
)]
#[case::sign(
    "sign",
    "{1 if (x > 0.0); -1 if (x < 0.0); 0 otherwise}",
    "(piecewise (greater x 0.0) 1 (less x 0.0) -1 0)"
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
    "{x if (x > 0.0); (x * slope) otherwise}",
    "(piecewise (greater x 0.0) x (multiply x slope))"
)]
#[case::xor(
    "xor",
    "((a || b) && (!(a && b)))",
    "(logical_and (logical_or a b) (logical_not (logical_and a b)))"
)]
#[case::nand("nand", "(!(a && b))", "(logical_not (logical_and a b))")]
#[case::nor("nor", "(!(a || b))", "(logical_not (logical_or a b))")]
#[case::implies("implies", "((!a) || b)", "(logical_or (logical_not a) b)")]
#[case::iff("iff", "(a == b)", "(equal a b)")]
#[case::sigmoid(
    "sigmoid",
    "(1.0 / (1.0 + exp((-x))))",
    "(divide 1.0 (add 1.0 (exp (negate x))))"
)]
#[case::silu("silu", "(x * sigmoid(x))", "(multiply x (sigmoid x))")]
#[case::gelu(
    "gelu",
    "((0.5 * x) * (1.0 + erf((x / sqrt(2.0)))))",
    "(multiply (multiply 0.5 x) (add 1.0 (erf (divide x (sqrt 2.0)))))"
)]
fn composed_function_body_prints_as_its_documented_text(
    #[case] name: &str,
    #[case] expected_symbolic: &str,
    #[case] expected_functional: &str,
) {
    let body = find_composed(name).body();

    let symbolic = format_expression(
        body,
        FormatOptions::default().with_notation(Notation::Symbolic),
    );
    let functional = format_expression(
        body,
        FormatOptions::default().with_notation(Notation::Functional),
    );

    assert_eq!(symbolic, expected_symbolic);
    assert_eq!(functional, expected_functional);
}

/// Test each composed body refers to every one of its parameters and to no
/// other identifier.
#[rstest]
fn composed_function_body_refers_to_exactly_its_parameters(
    #[values(
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
        "gelu"
    )]
    name: &str,
) {
    let function = find_composed(name);

    let free = function.body().free_identifiers();

    let parameters: HashSet<Identifier> = function.parameters().iter().cloned().collect();
    assert_eq!(free, parameters);
}

/// Test every call in a composed body names a built-in function.
#[rstest]
fn composed_function_body_calls_only_builtin_functions(
    #[values(
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
        "gelu"
    )]
    name: &str,
) {
    let function = find_composed(name);

    let call_names = collect_call_names(function.body());

    let unknown: Vec<&String> = call_names
        .iter()
        .filter(|called| {
            find_composed_function(called).is_none() && find_native_function(called).is_none()
        })
        .collect();
    assert!(unknown.is_empty(), "{name} calls non-built-ins {unknown:?}");
}

/// Test `relu` passes an integer zero, not a float zero, to `max`.
#[test]
fn composed_function_relu_passes_an_integer_zero_to_max() {
    let relu = find_composed("relu");

    let ExpressionKind::Call(call) = relu.body().kind() else {
        panic!("relu's body is a call, got {:?}", relu.body());
    };

    assert_eq!(call.function_name(), "max");
    assert_eq!(
        find_literal(&call.arguments()[1]),
        LiteralKind::Int(&BigInt::from(0))
    );
}

/// Test `abs` compares its argument against a positive float zero.
#[test]
fn composed_function_abs_compares_against_a_float_zero() {
    let abs = find_composed("abs");

    let ExpressionKind::Piecewise(piecewise) = abs.body().kind() else {
        panic!("abs's body is a piecewise, got {:?}", abs.body());
    };

    let LiteralKind::Float(zero) = find_right_literal(&piecewise.cases()[0].0) else {
        panic!("abs compares against a float");
    };
    assert_eq!(zero.to_bits(), 0.0_f64.to_bits());
}

/// Test `sign` compares against float zeros but yields the integers 1, -1
/// and 0.
#[test]
fn composed_function_sign_yields_integer_literals() {
    let sign = find_composed("sign");

    let ExpressionKind::Piecewise(piecewise) = sign.body().kind() else {
        panic!("sign's body is a piecewise, got {:?}", sign.body());
    };

    let conditions: Vec<LiteralKind<'_>> = piecewise
        .cases()
        .iter()
        .map(|(condition, _)| find_right_literal(condition))
        .collect();
    let values: Vec<LiteralKind<'_>> = piecewise
        .cases()
        .iter()
        .map(|(_, value)| find_literal(value))
        .chain([find_literal(piecewise.otherwise())])
        .collect();
    assert_eq!(
        conditions,
        [LiteralKind::Float(0.0), LiteralKind::Float(0.0)]
    );
    assert_eq!(
        values,
        [
            LiteralKind::Int(&BigInt::from(1)),
            LiteralKind::Int(&BigInt::from(-1)),
            LiteralKind::Int(&BigInt::from(0)),
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

/// Test repeated calls return the same table with the same parameters.
#[test]
fn composed_functions_returns_the_same_table_on_every_call() {
    let first = list_composed_functions();
    let first_ids = collect_parameter_ids();

    let second = list_composed_functions();

    assert!(std::ptr::eq(first, second), "the table was rebuilt");
    assert_eq!(collect_parameter_ids(), first_ids);
}

/// Test threads racing to first use the catalogue all see one set of
/// parameters.
#[test]
fn composed_functions_agrees_across_threads() {
    let handles: Vec<_> = (0..8)
        .map(|_| thread::spawn(collect_parameter_ids))
        .collect();

    let observed: Vec<Vec<(&'static str, Vec<u64>)>> = handles
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

/// Test a freshly created identifier never equals a catalogue parameter of
/// the same name.
#[test]
fn composed_function_parameters_differ_from_new_identifiers() {
    let (fresh_x, _) = build_identifier("x");
    let (fresh_a, _) = build_identifier("a");

    let parameters: Vec<&Identifier> = list_composed_functions()
        .iter()
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

    let expected = build_piecewise_node_or_panic(
        vec![(
            Expression::new_binary(BinaryOperation::Greater, build_literal(1), build_literal(2)),
            build_literal(1),
        )],
        build_literal(2),
    );
    assert_eq!(inlined, expected);
}

/// Test the clamp `max(low, min(value, high))`, inlined through the `min`
/// and `max` bodies, yields the nested piecewise a scheduling pass expects.
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

    let inner_min = build_piecewise_node_or_panic(
        vec![(
            Expression::new_binary(BinaryOperation::Less, &value, &high),
            value.clone(),
        )],
        high,
    );
    let expected = build_piecewise_node_or_panic(
        vec![(
            Expression::new_binary(BinaryOperation::Greater, &low, &inner_min),
            low.clone(),
        )],
        inner_min,
    );
    assert_eq!(inlined, expected);
}

/// Test a piecewise built for a guarded fast path keeps the guard, the fast
/// path, and the fallback in place.
#[test]
fn build_piecewise_guards_a_fast_path_with_a_fallback() {
    let (_, x) = build_identifier("x");
    let (_, sqrt_path) = build_identifier("sqrt_path");
    let (_, fallback_path) = build_identifier("fallback_path");

    let expression = build_piecewise([(x.greater(0), sqrt_path.clone())], fallback_path.clone())
        .expect("one guarded case builds");

    let expected = build_piecewise_node_or_panic(
        vec![(
            Expression::new_binary(BinaryOperation::Greater, &x, build_literal(0)),
            sqrt_path,
        )],
        fallback_path,
    );
    assert_eq!(expression, expected);
}
