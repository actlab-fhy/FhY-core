//! Stories for lifting SymPy objects to expressions,
//! [`SympySimplifier::lift`]: sums and
//! products, the native functions, constants, numbers, symbols, the
//! connectives and relationals, the refusals, and deep and shared
//! objects.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use fhy_core::expression::builtins::{BuiltinConstant, BuiltinFunction};
use fhy_core::expression::{
    BigInt, BinaryOperation, Decimal, Expression, LiteralValue, LogicalOperation,
    NoRegisteredSorts, UnaryOperation,
};
use fhy_core::identifier::Identifier;
use fhy_core::solver::SimplifyContext;
use fhy_core::tree::{NodeHandle, NodeIdentity};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rstest::rstest;

use super::test_support::{attached, backend, build_identifier, build_literal, evaluate};
use super::{SympyErrorKind, SympyPhase};

/// Return the lifting of the Python expression `source`.
fn lifted(source: &str) -> Expression {
    attached(|py| backend().lift(&evaluate(py, source)).expect("lifted"))
}

/// Return the kind and text of the error lifting `source` fails with,
/// checking it is a lifting error.
fn refusal(source: &str) -> (String, SympyErrorKind) {
    attached(|py| {
        let error = backend().lift(&evaluate(py, source)).expect_err("refused");
        assert_eq!(error.phase(), SympyPhase::Lifting);
        (error.to_string(), error.into_kind())
    })
}

/// Return the Python expression of the symbol of `identifier`.
fn symbol(identifier: &Identifier) -> String {
    format!(
        "sympy.Symbol({:?})",
        format!("{}_{}", identifier.name_hint(), identifier.id())
    )
}

/// Return a decimal literal of `text`.
fn decimal(text: &str) -> Expression {
    let value: Decimal = text.parse().expect("a decimal");
    Expression::literal(LiteralValue::Decimal(value))
}

#[test]
fn sum_and_product_fold_to_the_right_in_argument_order() {
    let (a, a_reference) = build_identifier("a");
    let (b, b_reference) = build_identifier("b");
    let (c, c_reference) = build_identifier("c");
    let symbols = format!("({}, {}, {})", symbol(&a), symbol(&b), symbol(&c));

    let sum = lifted(&format!(
        "(lambda a, b, c: sympy.Add(a, b, c, evaluate=False))(*{symbols})"
    ));
    let product = lifted(&format!(
        "(lambda a, b, c: sympy.Mul(a, b, c, evaluate=False))(*{symbols})"
    ));

    assert_eq!(sum, &a_reference + &(&b_reference + &c_reference));
    assert_eq!(product, &a_reference * &(&b_reference * &c_reference));
}

#[test]
fn modulo_and_power_are_binary() {
    let (x, reference) = build_identifier("x");

    assert_eq!(
        lifted(&format!("sympy.Mod({}, 3)", symbol(&x))),
        reference.floor_mod(3)
    );
    assert_eq!(
        lifted(&format!("sympy.Pow({}, 3)", symbol(&x))),
        reference.power(3)
    );
}

#[test]
fn negative_integer_powers_lift_as_divisions() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let (z, z_reference) = build_identifier("z");
    let (x_symbol, y_symbol, z_symbol) = (symbol(&x), symbol(&y), symbol(&z));

    assert_eq!(
        lifted(&format!("sympy.Pow({x_symbol}, -1)")),
        build_literal(1) / x_reference.clone()
    );
    assert_eq!(
        lifted(&format!("sympy.Pow({x_symbol}, -3)")),
        build_literal(1) / x_reference.clone().power(3)
    );
    assert_eq!(
        lifted(&format!("{y_symbol} / {x_symbol}")),
        y_reference.clone() / x_reference.clone()
    );
    assert_eq!(
        lifted(&format!(
            "sympy.Mul(2, {y_symbol}, sympy.Pow({x_symbol}, -1), sympy.Pow({z_symbol}, -2), evaluate=False)"
        )),
        (build_literal(2) * y_reference) / (x_reference * z_reference.power(2))
    );
}

#[test]
fn power_of_one_half_is_a_square_root() {
    let (x, reference) = build_identifier("x");

    assert_eq!(
        lifted(&format!("sympy.sqrt({})", symbol(&x))),
        Expression::call(BuiltinFunction::Sqrt, [reference])
    );
}

#[rstest]
#[case::exp("sympy.exp", BuiltinFunction::Exp)]
#[case::sin("sympy.sin", BuiltinFunction::Sin)]
#[case::cos("sympy.cos", BuiltinFunction::Cos)]
#[case::tan("sympy.tan", BuiltinFunction::Tan)]
#[case::asin("sympy.asin", BuiltinFunction::Arcsin)]
#[case::acos("sympy.acos", BuiltinFunction::Arccos)]
#[case::atan("sympy.atan", BuiltinFunction::Arctan)]
#[case::sinh("sympy.sinh", BuiltinFunction::Sinh)]
#[case::cosh("sympy.cosh", BuiltinFunction::Cosh)]
#[case::tanh("sympy.tanh", BuiltinFunction::Tanh)]
#[case::erf("sympy.erf", BuiltinFunction::Erf)]
#[case::log("sympy.log", BuiltinFunction::Log)]
#[case::floor("sympy.floor", BuiltinFunction::Floor)]
#[case::ceiling("sympy.ceiling", BuiltinFunction::Ceil)]
#[case::round("prelude.ROUND", BuiltinFunction::Round)]
fn native_function_lifts_to_its_builtin_call(
    #[case] sympy_function: &str,
    #[case] function: BuiltinFunction,
) {
    let (x, reference) = build_identifier("x");

    assert_eq!(
        lifted(&format!("{sympy_function}({})", symbol(&x))),
        Expression::call(function, [reference])
    );
}

#[rstest]
#[case::pi("sympy.pi", BuiltinConstant::Pi)]
#[case::e("sympy.E", BuiltinConstant::E)]
#[case::infinity("sympy.oo", BuiltinConstant::Inf)]
#[case::nan("sympy.nan", BuiltinConstant::Nan)]
fn constant_lifts_to_the_builtin_constants_identifier(
    #[case] source: &str,
    #[case] constant: BuiltinConstant,
) {
    assert_eq!(
        lifted(source),
        Expression::from(constant.identifier().clone())
    );
}

#[test]
fn negative_infinity_lifts_to_the_negation_of_inf() {
    assert_eq!(
        lifted("-sympy.oo"),
        Expression::new_unary(
            UnaryOperation::Negate,
            Expression::from(BuiltinConstant::Inf.identifier().clone())
        )
    );
}

#[rstest]
#[case::half("sympy.Rational(1, 2)", decimal("0.5"))]
#[case::negative_three_quarters(
    "sympy.Rational(-3, 4)",
    Expression::new_unary(UnaryOperation::Negate, decimal("0.75"))
)]
#[case::tenth(
    "sympy.Rational(1, 10)",
    Expression::new_binary(BinaryOperation::Divide, build_literal(1), build_literal(10))
)]
#[case::third(
    "sympy.Rational(-1, 3)",
    Expression::new_binary(BinaryOperation::Divide, build_literal(-1), build_literal(3))
)]
#[case::big_power_of_two(
    "sympy.Rational(1, 2**60)",
    decimal("0.000000000000000000867361737988403547205962240695953369140625")
)]
fn rational_lifts_to_decimal_text_only_when_a_binary_float_equals_it(
    #[case] source: &str,
    #[case] expected: Expression,
) {
    assert_eq!(lifted(source), expected);
}

#[test]
fn integers_and_floats_lift_to_literals_of_their_values() {
    assert_eq!(lifted("sympy.Integer(-12)"), build_literal(-12));
    assert_eq!(
        lifted("sympy.Integer(2**100)"),
        build_literal(BigInt::from(2).pow(100))
    );
    assert_eq!(lifted("sympy.Float(0.1)"), build_literal(0.1));
}

#[test]
fn symbol_lifts_to_the_identifier_it_names_advancing_the_counter() {
    let probe = Identifier::new("probe");
    let id = probe.id() + 1_000;

    let expression = lifted(&format!("sympy.Symbol('restored_{id}')"));

    let restored = Identifier::try_restore(id, "restored").expect("in range");
    assert_eq!(expression, Expression::from(restored));
    assert!(Identifier::new("after").id() > id);
}

#[test]
fn symbol_with_an_underscore_in_its_hint_lifts_by_its_last_underscore() {
    let (x, reference) = build_identifier("my_name");

    assert_eq!(lifted(&symbol(&x)), reference);
}

#[rstest]
#[case::no_underscore("sympy.Symbol('x')")]
#[case::no_id("sympy.Symbol('x_y')")]
fn symbol_the_lowering_did_not_name_is_refused(#[case] source: &str) {
    let (_, kind) = refusal(source);

    assert!(matches!(kind, SympyErrorKind::UnreadableSymbol(_)));
}

#[test]
fn connectives_lift_in_argument_order() {
    let (a, a_reference) = build_identifier("a");
    let (b, b_reference) = build_identifier("b");
    let (c, c_reference) = build_identifier("c");
    let symbols = format!("({}, {}, {})", symbol(&a), symbol(&b), symbol(&c));
    let call = |body: &str| lifted(&format!("(lambda a, b, c: {body})(*{symbols})"));

    assert_eq!(
        call("sympy.And(a, b, c, evaluate=False)"),
        Expression::new_logical(
            LogicalOperation::And,
            [&a_reference, &b_reference, &c_reference]
        )
    );
    assert_eq!(call("sympy.Not(a)"), !&a_reference);
    assert_eq!(
        call("sympy.Nor(a, b)"),
        !Expression::new_logical(LogicalOperation::Or, [&a_reference, &b_reference])
    );
    assert_eq!(
        call("sympy.Nand(a, b)"),
        !Expression::new_logical(LogicalOperation::And, [&a_reference, &b_reference])
    );
    let either = Expression::new_logical(LogicalOperation::Or, [&a_reference, &b_reference]);
    let both = Expression::new_logical(LogicalOperation::And, [&a_reference, &b_reference]);
    assert_eq!(
        call("sympy.Xor(a, b)"),
        Expression::new_logical(LogicalOperation::And, [either, !both])
    );
    assert_eq!(
        call("sympy.ITE(a, b, c)"),
        Expression::piecewise([(&a_reference, &b_reference)], &c_reference).expect("a piecewise")
    );
    assert_eq!(lifted("sympy.true"), build_literal(true));
    assert_eq!(lifted("sympy.false"), build_literal(false));
}

#[test]
fn nand_and_nor_nodes_lift_as_negated_connectives() {
    let (a, a_reference) = build_identifier("a");
    let (b, b_reference) = build_identifier("b");
    let (c, c_reference) = build_identifier("c");
    let symbols = format!("({}, {}, {})", symbol(&a), symbol(&b), symbol(&c));
    let call = |body: &str| lifted(&format!("(lambda a, b, c: {body})(*{symbols})"));
    let all = [&a_reference, &b_reference, &c_reference];

    // Without `evaluate=False`, SymPy builds `Not(And(..))` and `Not(Or(..))`.
    assert_eq!(
        call("sympy.Nand(a, b, c, evaluate=False)"),
        !Expression::new_logical(LogicalOperation::And, all)
    );
    assert_eq!(
        call("sympy.Nor(a, b, c, evaluate=False)"),
        !Expression::new_logical(LogicalOperation::Or, all)
    );
}

#[test]
fn an_equality_of_three_arguments_is_refused_for_its_arity() {
    let (a, _) = build_identifier("a");
    let (b, _) = build_identifier("b");
    let (c, _) = build_identifier("c");

    let (text, kind) = refusal(&format!(
        "sympy.Basic.__new__(sympy.Eq, {}, {}, {})",
        symbol(&a),
        symbol(&b),
        symbol(&c)
    ));

    assert!(
        matches!(&kind, SympyErrorKind::Arity(expected) if expected == "a binary operation to have exactly two arguments"),
        "{kind:?}"
    );
    assert_eq!(
        text,
        "expected a binary operation to have exactly two arguments"
    );
}

#[test]
fn a_boolean_condition_piecewise_inside_a_relational_lifts_to_a_comparison_of_it() {
    let (a, a_reference) = build_identifier("a");
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");

    let comparison = lifted(&format!(
        "sympy.Lt(sympy.Piecewise(({x}, {a}), ({y}, True)), 3, evaluate=False)",
        x = symbol(&x),
        a = symbol(&a),
        y = symbol(&y)
    ));

    assert_eq!(
        comparison,
        Expression::piecewise([(a_reference, x_reference)], y_reference)
            .expect("a piecewise")
            .less(3)
    );
}

#[rstest]
#[case::equal("sympy.Eq(x, y)", BinaryOperation::Equal)]
#[case::not_equal("sympy.Ne(x, y)", BinaryOperation::NotEqual)]
#[case::less("x < y", BinaryOperation::Less)]
#[case::less_equal("x <= y", BinaryOperation::LessEqual)]
#[case::greater("x > y", BinaryOperation::Greater)]
#[case::greater_equal("x >= y", BinaryOperation::GreaterEqual)]
fn relational_lifts_to_its_comparison(#[case] body: &str, #[case] operation: BinaryOperation) {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");

    assert_eq!(
        lifted(&format!(
            "(lambda x, y: {body})({}, {})",
            symbol(&x),
            symbol(&y)
        )),
        Expression::new_binary(operation, x_reference, y_reference)
    );
}

#[test]
fn piecewise_lifts_to_its_cases_and_otherwise_branch() {
    let (x, reference) = build_identifier("x");

    let expression = lifted(&format!(
        "(lambda x: sympy.Piecewise((1, x > 0), (-1, x < 0), (0, True)))({})",
        symbol(&x)
    ));

    assert_eq!(
        expression,
        Expression::piecewise(
            [
                (reference.greater(0), build_literal(1)),
                (reference.less(0), build_literal(-1))
            ],
            0
        )
        .expect("a piecewise")
    );
}

#[rstest]
#[case::complex_infinity("sympy.zoo", "zoo")]
#[case::partial_piecewise("sympy.Piecewise((1, sympy.Symbol('flag_0')))", "flag_0")]
#[case::implies(
    "sympy.Implies(sympy.Symbol('a_1'), sympy.Symbol('b_2'))",
    "implies is not supported"
)]
#[case::derivative(
    "sympy.Derivative(sympy.Symbol('x_0'), sympy.Symbol('x_0'))",
    "unsupported expression type"
)]
#[case::no_sympy_object("42", "unsupported node type: <class 'int'>")]
fn object_no_expression_denotes_is_refused_naming_it(#[case] source: &str, #[case] phrase: &str) {
    let (text, _) = refusal(source);

    assert!(text.contains(phrase), "{text}");
}

#[test]
fn deep_object_lifts_on_a_small_stack() {
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| {
            attached(|py| {
                let (x, _) = build_identifier("x");
                let object = evaluate(
                    py,
                    &format!(
                        "(lambda x: [t := x] and [t := sympy.Mul(sympy.Add(t, 1, evaluate=False), 2, \
                         evaluate=False) for _ in range(20_000)][-1])({})",
                        symbol(&x)
                    ),
                );
                backend().lift(&object).map(|_| ()).map_err(|error| error.to_string())
            })
        })
        .expect("a thread");

    assert_eq!(handle.join().expect("no stack overflow"), Ok(()));
}

/// Return the doubling DAG `e_{k+1} = sin(e_k) + cos(e_k)` of `depth`
/// levels over `x`: `3 * depth + 1` distinct nodes, and about `2^depth`
/// occurrences of `x`.
fn build_sin_cos_dag(x: Expression, depth: usize) -> Expression {
    let mut dag = x;
    for _ in 0..depth {
        dag = Expression::call(BuiltinFunction::Sin, [dag.clone()])
            + Expression::call(BuiltinFunction::Cos, [dag]);
    }
    dag
}

/// Return the number of distinct nodes of `root`, by identity.
fn count_distinct_nodes(root: &Expression) -> usize {
    let mut seen: HashSet<NodeIdentity> = HashSet::new();
    let mut pending = vec![root];
    while let Some(node) = pending.pop() {
        if seen.insert(node.identity()) {
            pending.extend(node.children());
        }
    }
    seen.len()
}

#[test]
fn lifting_a_depth_16_sin_cos_dag_is_linear() {
    let (_, x) = build_identifier("x");
    let dag = build_sin_cos_dag(x.clone(), 16);
    let small = build_sin_cos_dag(x.clone(), 4);
    let mut small_in_sympy_order = x;
    for _ in 0..4 {
        small_in_sympy_order =
            Expression::call(BuiltinFunction::Cos, [small_in_sympy_order.clone()])
                + Expression::call(BuiltinFunction::Sin, [small_in_sympy_order]);
    }
    let context = SimplifyContext::new(&NoRegisteredSorts);

    let (lifted, elapsed, small_lifted) = attached(|py| {
        let lowered = backend().lower(py, &dag, &context).expect("lowered");
        let started = Instant::now();
        let lifted = backend().lift(&lowered).expect("lifted");
        let elapsed = started.elapsed();
        let small_lowered = backend().lower(py, &small, &context).expect("lowered");
        (
            lifted,
            elapsed,
            backend().lift(&small_lowered).expect("lifted"),
        )
    });

    assert!(
        elapsed < Duration::from_millis(100),
        "lifting took {elapsed:?}"
    );
    assert_eq!(
        count_distinct_nodes(&lifted),
        3 * 16 + 1,
        "the result shares"
    );
    assert_eq!(
        small_lifted, small_in_sympy_order,
        "sympy orders a sum's terms"
    );
}

#[test]
fn substituting_into_a_depth_16_sin_cos_dag_is_linear() {
    let (x, x_reference) = build_identifier("x");
    let (y, _) = build_identifier("y");
    let dag = build_sin_cos_dag(x_reference, 16);
    let context = SimplifyContext::new(&NoRegisteredSorts);

    let (substituted, elapsed) = attached(|py| {
        let lowered = backend().lower(py, &dag, &context).expect("lowered");
        let replacements = PyDict::new(py);
        replacements
            .set_item(evaluate(py, &symbol(&x)), evaluate(py, &symbol(&y)))
            .expect("set");
        let started = Instant::now();
        let substituted = backend()
            .substitute_symbols(&lowered, replacements.as_any())
            .expect("substituted");
        let elapsed = started.elapsed();
        (backend().lift(&substituted).expect("lifted"), elapsed)
    });

    // Each of the 49 rebuilt nodes is a SymPy construction, whose own
    // evaluation dominates; without the memo, the walk takes seconds.
    assert!(
        elapsed < Duration::from_secs(1),
        "substituting took {elapsed:?}"
    );
    assert_eq!(count_distinct_nodes(&substituted), 3 * 16 + 1);
    assert_eq!(substituted.free_identifiers(), HashSet::from([y]));
}
