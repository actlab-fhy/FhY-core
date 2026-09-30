//! Stories for lowering expressions to SymPy, [`SympySimplifier::lower`]:
//! each literal form, identifiers and
//! constants, each operation, the Boolean positions, piecewise nodes, the
//! native built-ins, the refusals, and deep trees.

use fhy_core::expression::builtins::{BuiltinConstant, BuiltinFunction};
use fhy_core::expression::registry::{
    FunctionDefinition, FunctionRegistry, NativeConstant, NativeFunction,
};
use fhy_core::expression::{
    BigInt, Callee, Decimal, Expression, FunctionName, FunctionSort, LiteralValue,
    LogicalOperation, NoRegisteredSorts,
};
use fhy_core::identifier::Identifier;
use fhy_core::solver::SimplifyContext;
use pyo3::prelude::*;
use rstest::rstest;

use super::test_support::{attached, backend, build_identifier, build_literal, evaluate, srepr};
use super::{SympyErrorKind, SympyPhase};

/// Return the `srepr` of the lowering of `expression` with no registry.
fn lowered(expression: &Expression) -> String {
    attached(|py| {
        let object = backend()
            .lower(py, expression, &SimplifyContext::new(&NoRegisteredSorts))
            .expect("lowered");
        srepr(&object)
    })
}

/// Return the `srepr` of the Python expression `source`.
fn expected(source: &str) -> String {
    attached(|py| srepr(&evaluate(py, source)))
}

/// Return the name of `identifier`'s symbol.
fn symbol(identifier: &Identifier) -> String {
    format!("{}_{}", identifier.name_hint(), identifier.id())
}

/// Return the kind of the error lowering `expression` with `context`
/// fails with, checking it is a lowering error.
fn refusal(expression: &Expression, context: &SimplifyContext<'_>) -> (String, SympyErrorKind) {
    attached(|py| {
        let error = backend()
            .lower(py, expression, context)
            .expect_err("refused");
        assert_eq!(error.phase(), SympyPhase::Lowering);
        (error.to_string(), error.into_kind())
    })
}

// ---------------------------------------------------------------------------
// Literals
// ---------------------------------------------------------------------------

#[rstest]
#[case::true_literal(LiteralValue::from(true), "sympy.true")]
#[case::false_literal(LiteralValue::from(false), "sympy.false")]
#[case::small_integer(LiteralValue::from(-7), "sympy.Integer(-7)")]
#[case::big_integer(LiteralValue::from(BigInt::from(2).pow(100)), "sympy.Integer(2**100)")]
#[case::binary_tenth(LiteralValue::from(0.1), "sympy.Float(0.1)")]
#[case::infinity(LiteralValue::from(f64::INFINITY), "sympy.oo")]
#[case::negative_infinity(LiteralValue::from(f64::NEG_INFINITY), "-sympy.oo")]
#[case::not_a_number(LiteralValue::from(f64::NAN), "sympy.nan")]
fn literal_lowers_to_the_sympy_number_of_its_value(
    #[case] value: LiteralValue,
    #[case] python: &str,
) {
    assert_eq!(lowered(&Expression::literal(value)), expected(python));
}

#[rstest]
#[case::tenth("0.1", "sympy.Rational(1, 10)")]
#[case::two_and_a_half("2.50", "sympy.Rational(5, 2)")]
#[case::whole("3.0", "sympy.Integer(3)")]
#[case::hundred("100.", "sympy.Integer(100)")]
fn decimal_lowers_to_the_exact_rational_it_denotes(#[case] text: &str, #[case] python: &str) {
    let decimal: Decimal = text.parse().expect("a decimal");
    assert_eq!(
        lowered(&Expression::literal(LiteralValue::Decimal(decimal))),
        expected(python)
    );
}

// ---------------------------------------------------------------------------
// Identifiers and constants
// ---------------------------------------------------------------------------

#[test]
fn identifier_lowers_to_the_symbol_named_by_its_hint_and_id() {
    let (x, reference) = build_identifier("x");

    assert_eq!(
        lowered(&reference),
        expected(&format!("sympy.Symbol({:?})", symbol(&x)))
    );
}

#[rstest]
#[case::pi(BuiltinConstant::Pi, "sympy.pi")]
#[case::e(BuiltinConstant::E, "sympy.E")]
#[case::inf(BuiltinConstant::Inf, "sympy.oo")]
#[case::nan(BuiltinConstant::Nan, "sympy.nan")]
fn builtin_constant_lowers_to_its_sympy_constant(
    #[case] constant: BuiltinConstant,
    #[case] python: &str,
) {
    assert_eq!(
        lowered(&Expression::from(constant.identifier().clone())),
        expected(python)
    );
}

#[test]
fn identifier_named_like_a_constant_is_an_ordinary_symbol() {
    let (pi, reference) = build_identifier("pi");

    assert_eq!(
        lowered(&reference),
        expected(&format!("sympy.Symbol({:?})", symbol(&pi)))
    );
}

#[rstest]
#[case::integer(FunctionSort::Int, LiteralValue::from(42), "sympy.Integer(42)")]
#[case::boolean(FunctionSort::Bool, LiteralValue::from(true), "sympy.true")]
#[case::float(FunctionSort::Real, LiteralValue::from(0.25), "sympy.Float(0.25)")]
#[case::decimal(
    FunctionSort::Real,
    LiteralValue::Decimal("0.1".parse().expect("a decimal")),
    "sympy.Rational(1, 10)"
)]
fn user_constant_lowers_to_its_value_through_the_registry(
    #[case] sort: FunctionSort,
    #[case] value: LiteralValue,
    #[case] python: &str,
) {
    let mut registry = FunctionRegistry::new();
    let constant = registry
        .register_constant(
            NativeConstant::new(FunctionName::new("answer").expect("a name"), sort, value)
                .expect("a constant"),
        )
        .expect("registered");

    let object = attached(|py| {
        let object = backend()
            .lower(
                py,
                &Expression::from(constant),
                &SimplifyContext::from_registry(&registry),
            )
            .expect("lowered");
        srepr(&object)
    });

    assert_eq!(object, expected(python));
}

#[test]
fn user_constant_without_a_registry_is_refused() {
    let mut registry = FunctionRegistry::new();
    let constant = registry
        .register_constant(
            NativeConstant::new(
                FunctionName::new("answer").expect("a name"),
                FunctionSort::Int,
                42,
            )
            .expect("a constant"),
        )
        .expect("registered");

    let (text, kind) = refusal(
        &Expression::from(constant.clone()),
        &SimplifyContext::new(&registry),
    );

    assert!(
        matches!(kind, SympyErrorKind::ConstantValueUnknown(identifier) if identifier == constant)
    );
    assert!(text.contains("answer"), "{text}");
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

#[rstest]
#[case::add(|x: &Expression, y: &Expression| x + y, "x + y")]
#[case::subtract(|x: &Expression, y: &Expression| x - y, "x - y")]
#[case::multiply(|x: &Expression, y: &Expression| x * y, "x * y")]
#[case::divide(|x: &Expression, y: &Expression| x / y, "x / y")]
#[case::floor_divide(|x: &Expression, y: &Expression| x.floor_divide(y), "sympy.floor(x / y)")]
#[case::floor_mod(|x: &Expression, y: &Expression| x.floor_mod(y), "x % y")]
#[case::power(|x: &Expression, y: &Expression| x.power(y), "x ** y")]
#[case::negate(|x: &Expression, _: &Expression| -x, "-x")]
#[case::positive(|x: &Expression, _: &Expression| x.positive(), "+x")]
#[case::equal(|x: &Expression, y: &Expression| x.equals(y), "sympy.Eq(x, y)")]
#[case::not_equal(|x: &Expression, y: &Expression| x.not_equals(y), "sympy.Ne(x, y)")]
#[case::less(|x: &Expression, y: &Expression| x.less(y), "x < y")]
#[case::less_equal(|x: &Expression, y: &Expression| x.less_equal(y), "x <= y")]
#[case::greater(|x: &Expression, y: &Expression| x.greater(y), "x > y")]
#[case::greater_equal(|x: &Expression, y: &Expression| x.greater_equal(y), "x >= y")]
fn operation_lowers_to_the_sympy_operation(
    #[case] build: fn(&Expression, &Expression) -> Expression,
    #[case] python: &str,
) {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let bindings = format!(
        "(lambda x, y: {python})(sympy.Symbol({:?}), sympy.Symbol({:?}))",
        symbol(&x),
        symbol(&y)
    );

    assert_eq!(
        lowered(&build(&x_reference, &y_reference)),
        expected(&bindings)
    );
}

#[test]
fn connectives_lower_to_n_ary_sympy_connectives_and_not() {
    let (a, a_reference) = build_identifier("a");
    let (b, b_reference) = build_identifier("b");
    let (c, c_reference) = build_identifier("c");
    let conjunction = Expression::new_logical(
        LogicalOperation::And,
        [&a_reference, &b_reference, &c_reference],
    );
    let negated_disjunction =
        !Expression::new_logical(LogicalOperation::Or, [&a_reference, &b_reference]);
    let symbols = format!(
        "(sympy.Symbol({:?}), sympy.Symbol({:?}), sympy.Symbol({:?}))",
        symbol(&a),
        symbol(&b),
        symbol(&c)
    );

    assert_eq!(
        lowered(&conjunction),
        expected(&format!("(lambda a, b, c: sympy.And(a, b, c))(*{symbols})"))
    );
    assert_eq!(
        lowered(&negated_disjunction),
        expected(&format!(
            "(lambda a, b, c: sympy.Not(sympy.Or(a, b)))(*{symbols})"
        ))
    );
}

// ---------------------------------------------------------------------------
// Piecewise nodes and Boolean positions
// ---------------------------------------------------------------------------

#[test]
fn piecewise_lowers_to_a_parity_opaque_piecewise_with_a_final_true_branch() {
    let (x, reference) = build_identifier("x");
    let piecewise =
        Expression::piecewise([(reference.greater(0), build_literal(1))], 2).expect("a piecewise");

    let (type_name, branches) = attached(|py| {
        let object = backend()
            .lower(py, &piecewise, &SimplifyContext::default())
            .expect("lowered");
        (
            object.get_type().qualname().expect("a name").to_string(),
            srepr(&object.getattr("args").expect("args")),
        )
    });

    assert_eq!(type_name, "ParityOpaquePiecewise");
    assert_eq!(
        branches,
        expected(&format!(
            "(lambda x: sympy.Piecewise((1, x > 0), (2, True), evaluate=False).args)(sympy.Symbol({:?}))",
            symbol(&x)
        ))
    );
}

#[test]
fn boolean_piecewise_under_a_connective_is_its_first_match_expansion() {
    let (b, b_reference) = build_identifier("b");
    let (c, c_reference) = build_identifier("c");
    let (d, d_reference) = build_identifier("d");
    let choice =
        Expression::piecewise([(b_reference, &c_reference)], &d_reference).expect("a piecewise");
    let conjunction = choice.and(&c_reference);

    assert_eq!(
        lowered(&conjunction),
        expected(&format!(
            "(lambda b, c, d: sympy.And(sympy.Or(sympy.And(b, c), sympy.And(sympy.Not(b), d)), c))\
             (sympy.Symbol({:?}), sympy.Symbol({:?}), sympy.Symbol({:?}))",
            symbol(&b),
            symbol(&c),
            symbol(&d)
        ))
    );
}

#[test]
fn comparison_of_a_boolean_piecewise_compares_its_expansion() {
    let (b, b_reference) = build_identifier("b");
    let choice = Expression::piecewise([(b_reference, build_literal(true))], build_literal(false))
        .expect("a piecewise");

    let comparison = lowered(&choice.equals(build_literal(true)));

    assert_eq!(
        comparison,
        expected(&format!(
            "(lambda b: sympy.Eq(sympy.Or(sympy.And(b, True), sympy.And(sympy.Not(b), False)), True))\
             (sympy.Symbol({:?}))",
            symbol(&b)
        ))
    );
}

#[test]
fn condition_comparing_a_symbol_with_a_relational_negates_the_symbol() {
    let (b, b_reference) = build_identifier("b");
    let (x, x_reference) = build_identifier("x");
    let condition = b_reference.equals(x_reference.less(1));
    let piecewise = Expression::piecewise([(condition, build_literal(1))], 2).expect("a piecewise");

    let branches = attached(|py| {
        let object = backend()
            .lower(py, &piecewise, &SimplifyContext::default())
            .expect("lowered");
        srepr(&object.getattr("args").expect("args"))
    });

    assert!(
        branches.contains("Unequality(Not(Symbol"),
        "the condition is Ne(~b, x < 1): {branches}"
    );
    assert!(branches.contains(&symbol(&b)) && branches.contains(&symbol(&x)));
}

// ---------------------------------------------------------------------------
// Calls
// ---------------------------------------------------------------------------

#[rstest]
#[case::exp(BuiltinFunction::Exp, "sympy.exp(x)")]
#[case::exp2(BuiltinFunction::Exp2, "sympy.Pow(2, x)")]
#[case::log(BuiltinFunction::Log, "sympy.log(x)")]
#[case::log2(BuiltinFunction::Log2, "sympy.log(x, 2)")]
#[case::log10(BuiltinFunction::Log10, "sympy.log(x, 10)")]
#[case::sqrt(BuiltinFunction::Sqrt, "sympy.sqrt(x)")]
#[case::sin(BuiltinFunction::Sin, "sympy.sin(x)")]
#[case::cos(BuiltinFunction::Cos, "sympy.cos(x)")]
#[case::tan(BuiltinFunction::Tan, "sympy.tan(x)")]
#[case::arcsin(BuiltinFunction::Arcsin, "sympy.asin(x)")]
#[case::arccos(BuiltinFunction::Arccos, "sympy.acos(x)")]
#[case::arctan(BuiltinFunction::Arctan, "sympy.atan(x)")]
#[case::sinh(BuiltinFunction::Sinh, "sympy.sinh(x)")]
#[case::cosh(BuiltinFunction::Cosh, "sympy.cosh(x)")]
#[case::tanh(BuiltinFunction::Tanh, "sympy.tanh(x)")]
#[case::erf(BuiltinFunction::Erf, "sympy.erf(x)")]
#[case::round(BuiltinFunction::Round, "prelude.ROUND(x)")]
#[case::floor(BuiltinFunction::Floor, "sympy.floor(x)")]
#[case::ceil(BuiltinFunction::Ceil, "sympy.ceiling(x)")]
fn native_builtin_lowers_to_its_sympy_function(
    #[case] function: BuiltinFunction,
    #[case] python: &str,
) {
    let (x, reference) = build_identifier("x");

    assert_eq!(
        lowered(&Expression::call(function, [reference])),
        expected(&format!(
            "(lambda x: {python})(sympy.Symbol({:?}))",
            symbol(&x)
        ))
    );
}

#[test]
fn round_folds_only_over_an_integer() {
    let (_, reference) = build_identifier("x");

    assert_eq!(
        lowered(&Expression::call(
            BuiltinFunction::Round,
            [build_literal(3)]
        )),
        expected("sympy.Integer(3)")
    );
    assert!(lowered(&Expression::call(BuiltinFunction::Round, [reference])).contains("round"));
}

/// Return a registry holding a function defined by an expression, a
/// native function and a constant.
fn registry_with_every_entry() -> FunctionRegistry {
    let mut registry = FunctionRegistry::new();
    let parameter = Identifier::new("p");
    registry
        .register_function(
            FunctionDefinition::new(
                FunctionName::new("double").expect("a name"),
                [parameter.clone()],
                [FunctionSort::Real],
                FunctionSort::Real,
                Expression::from(parameter) * 2,
            )
            .expect("a definition"),
        )
        .expect("registered");
    registry
        .register_native_function(NativeFunction::new(
            FunctionName::new("mystery").expect("a name"),
            [FunctionSort::Real],
            FunctionSort::Real,
        ))
        .expect("registered");
    registry
        .register_constant(
            NativeConstant::new(
                FunctionName::new("answer").expect("a name"),
                FunctionSort::Int,
                42,
            )
            .expect("a constant"),
        )
        .expect("registered");
    registry
}

#[rstest]
#[case::builtin_with_a_body("max", "inline_functions")]
#[case::user_function("double", "inline_functions")]
#[case::native_function("mystery", "no sympy lowering")]
#[case::constant("answer", "not callable")]
#[case::unknown("nowhere", "unknown function")]
fn call_sympy_has_no_function_for_is_refused_naming_it(#[case] name: &str, #[case] phrase: &str) {
    let callee: Callee = name.parse().expect("a callee");
    let registry = registry_with_every_entry();
    let (_, reference) = build_identifier("x");
    let call = Expression::call(callee.clone(), [reference]);

    let (text, kind) = refusal(&call, &SimplifyContext::from_registry(&registry));

    assert!(text.contains(phrase), "{text}");
    assert!(text.contains(callee.name()), "{text}");
    assert!(matches!(
        kind,
        SympyErrorKind::CallNeedsInlining(_)
            | SympyErrorKind::NoSympyLowering(_)
            | SympyErrorKind::ConstantCalled(_)
            | SympyErrorKind::UnknownFunction(_)
    ));
}

#[test]
fn call_is_refused_before_its_arguments_are_lowered() {
    let registry = FunctionRegistry::new();
    let inner = Expression::call(
        Callee::Named(FunctionName::new("inner").expect("a name")),
        [build_literal(1)],
    );
    let outer = Expression::call(
        Callee::Named(FunctionName::new("outer").expect("a name")),
        [inner],
    );

    let (text, _) = refusal(&outer, &SimplifyContext::from_registry(&registry));

    assert!(text.contains("\"outer\""), "{text}");
}

#[test]
fn ill_typed_expression_is_refused_before_sympy_sees_it() {
    let (_, reference) = build_identifier("b");
    let ill_typed = Expression::new_logical(LogicalOperation::And, [build_literal(1), reference]);

    let (_, kind) = refusal(&ill_typed, &SimplifyContext::default());

    assert!(matches!(kind, SympyErrorKind::IllTyped(_)));
}

// ---------------------------------------------------------------------------
// Sharing and depth
// ---------------------------------------------------------------------------

#[test]
fn shared_node_is_lowered_once() {
    let (_, reference) = build_identifier("x");
    let mut tree = reference;
    for _ in 0..64 {
        tree = &tree * &tree;
    }

    let text = lowered(&tree);

    assert!(
        text.starts_with("Pow("),
        "a doubling DAG is a power: {text}"
    );
}

#[test]
fn deep_tree_lowers_on_a_small_stack() {
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| {
            let (_, x) = build_identifier("x");
            let (_, y) = build_identifier("y");
            let mut tree = x.clone();
            for level in 0..10_000 {
                tree = if level % 2 == 0 {
                    &tree + &x
                } else {
                    &tree * &y
                };
            }
            attached(|py| {
                backend()
                    .lower(py, &tree, &SimplifyContext::default())
                    .map(|_| ())
                    .map_err(|error| error.to_string())
            })
        })
        .expect("a thread");

    let result = handle.join().expect("no stack overflow");

    assert!(
        result.is_ok()
            || result
                .as_ref()
                .is_err_and(|text| text.contains("python raised")),
        "{result:?}"
    );
}
