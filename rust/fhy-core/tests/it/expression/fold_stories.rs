//! Tests for `Evaluator::fold`: native calls with literal arguments folded
//! to literals, constants resolved, everything else kept, the checks of a
//! folded call, user natives through `NativeCalls`, sharing and depth.

use crate::support::expression as expression_support;
use crate::support::stack as stack_support;

use std::cell::RefCell;
use std::collections::HashSet;

use fhy_core::expression::builtins::{BuiltinConstant, BuiltinFunction};
use fhy_core::expression::evaluate::{Evaluator, FoldError, NativeCalls, NoNativeCalls};
use fhy_core::expression::pattern::CallbackError;
use fhy_core::expression::registry::{
    FunctionDefinition, FunctionRegistry, NativeConstant, NativeFunction,
};
use fhy_core::expression::{BigInt, Callee, Expression, FunctionName, FunctionSort, LiteralValue};
use fhy_core::identifier::Identifier;
use fhy_core::tree::{NodeHandle, NodeIdentity};
use rstest::rstest;

use expression_support::{
    build_decimal_literal, build_doubling_dag, build_identifier, build_literal, expect_literal,
};
use stack_support::{SMALL_STACK_DEPTH, run_on_small_stack};

fn name(text: &str) -> FunctionName {
    FunctionName::try_new(text).expect("the test names no built-in")
}

fn call(
    function: impl Into<Callee>,
    arguments: impl IntoIterator<Item = Expression>,
) -> Expression {
    Expression::call(function, arguments)
}

/// Natives that record their calls and compute from a table of closures.
#[derive(Default)]
struct RecordingNatives {
    calls: RefCell<Vec<(String, Vec<LiteralValue>)>>,
}

impl NativeCalls for RecordingNatives {
    fn call(
        &self,
        function: &NativeFunction,
        arguments: &[LiteralValue],
    ) -> Result<LiteralValue, CallbackError> {
        self.calls
            .borrow_mut()
            .push((function.name().as_str().to_owned(), arguments.to_vec()));
        match function.name().as_str() {
            "atan2" => {
                let [LiteralValue::Float(y), LiteralValue::Float(x)] = arguments else {
                    return Err("atan2 takes two floats".into());
                };
                Ok(LiteralValue::Float(y.atan2(*x)))
            }
            "half_int" => Ok(LiteralValue::Float(1.5)),
            "fails" => Err("the implementation failed".into()),
            "twice" => {
                let [LiteralValue::Int(value)] = arguments else {
                    return Err("twice takes an integer".into());
                };
                Ok(LiteralValue::Int(value * 2))
            }
            other => Err(format!("no implementation of {other}").into()),
        }
    }
}

/// Return a registry of the natives `RecordingNatives` computes, a
/// function with a body, and a constant.
fn build_registry() -> (FunctionRegistry, Identifier) {
    let mut registry = FunctionRegistry::new();
    registry
        .register_native_function(NativeFunction::new(
            name("atan2"),
            [FunctionSort::Real, FunctionSort::Real],
            FunctionSort::Real,
        ))
        .expect("free name");
    registry
        .register_native_function(NativeFunction::new(
            name("half_int"),
            [FunctionSort::Real],
            FunctionSort::Int,
        ))
        .expect("free name");
    registry
        .register_native_function(NativeFunction::new(
            name("fails"),
            [FunctionSort::Real],
            FunctionSort::Real,
        ))
        .expect("free name");
    registry
        .register_native_function(NativeFunction::new(
            name("twice"),
            [FunctionSort::Int],
            FunctionSort::Int,
        ))
        .expect("free name");
    let (x, reference) = build_identifier("x");
    registry
        .register_function(
            FunctionDefinition::try_new(
                name("double"),
                [x],
                [FunctionSort::Real],
                FunctionSort::Real,
                reference * 2,
            )
            .expect("one sort per parameter"),
        )
        .expect("free name");
    let answer = registry
        .register_constant(
            NativeConstant::try_new(name("answer"), FunctionSort::Int, 42).expect("an integer"),
        )
        .expect("free name");
    (registry, answer)
}

fn fold(expression: &Expression) -> Result<Expression, FoldError> {
    let (registry, _) = build_registry();
    Evaluator::new(&registry)
        .fold(expression, &RecordingNatives::default())
        .map(fhy_core::expression::evaluate::Folding::into_output)
}

fn fold_literal(expression: &Expression) -> LiteralValue {
    let output = fold(expression).expect("the fold succeeds");
    expect_literal(&output).clone()
}

// ---------------------------------------------------------------------------
// Built-in natives
// ---------------------------------------------------------------------------

#[rstest]
#[case::exp(BuiltinFunction::Exp, 0.0, 1.0)]
#[case::sqrt(BuiltinFunction::Sqrt, 2.25, 1.5)]
#[case::log2(BuiltinFunction::Log2, 8.0, 3.0)]
#[case::tanh(BuiltinFunction::Tanh, 0.0, 0.0)]
fn fold_computes_a_real_builtin(
    #[case] function: BuiltinFunction,
    #[case] argument: f64,
    #[case] expected: f64,
) {
    let folded = fold_literal(&call(function, [build_literal(argument)]));

    assert_eq!(folded, LiteralValue::Float(expected));
}

#[rstest]
#[case::round_half_even(BuiltinFunction::Round, 2.5, 2)]
#[case::round_down(BuiltinFunction::Round, -2.5, -2)]
#[case::floor(BuiltinFunction::Floor, -1.5, -2)]
#[case::ceil(BuiltinFunction::Ceil, 1.2, 2)]
fn fold_makes_an_integer_sorted_builtin_an_integer(
    #[case] function: BuiltinFunction,
    #[case] argument: f64,
    #[case] expected: i64,
) {
    let folded = fold_literal(&call(function, [build_literal(argument)]));

    assert_eq!(folded, LiteralValue::from(expected));
}

#[test]
fn fold_makes_a_huge_integer_sorted_result_its_exact_integer() {
    let folded = fold_literal(&call(BuiltinFunction::Floor, [build_literal(1e300)]));

    let expected: BigInt = format!("{:.0}", 1e300_f64).parse().expect("digits");
    assert_eq!(folded, LiteralValue::Int(expected));
}

#[rstest]
#[case(f64::NAN)]
#[case(f64::INFINITY)]
#[case(f64::NEG_INFINITY)]
fn fold_refuses_a_non_finite_integer_sorted_result(#[case] argument: f64) {
    let error = fold(&call(BuiltinFunction::Round, [build_literal(argument)]))
        .expect_err("no integer is non-finite");

    assert!(matches!(
        error,
        FoldError::NonFiniteCast {
            function: BuiltinFunction::Round,
            ..
        }
    ));
}

#[test]
fn fold_follows_ieee_where_math_would_raise() {
    let root = fold_literal(&call(BuiltinFunction::Sqrt, [build_literal(-1.0)]));
    let logarithm = fold_literal(&call(BuiltinFunction::Log, [build_literal(0.0)]));

    assert!(matches!(root, LiteralValue::Float(value) if value.is_nan()));
    assert_eq!(logarithm, LiteralValue::Float(f64::NEG_INFINITY));
}

#[test]
fn fold_converts_an_integer_argument_to_the_nearest_real() {
    let folded = fold_literal(&call(BuiltinFunction::Sqrt, [build_literal(16)]));
    let huge: BigInt = format!("1{}", "0".repeat(400)).parse().expect("digits");
    let beyond = fold_literal(&call(BuiltinFunction::Exp, [build_literal(huge)]));

    assert_eq!(folded, LiteralValue::Float(4.0));
    assert_eq!(beyond, LiteralValue::Float(f64::INFINITY));
}

#[test]
fn fold_converts_an_exact_decimal_argument_and_refuses_an_inexact_one() {
    let folded = fold_literal(&call(
        BuiltinFunction::Sqrt,
        [build_decimal_literal("2.25")],
    ));
    let error = fold(&call(BuiltinFunction::Sqrt, [build_decimal_literal("0.1")]))
        .expect_err("0.1 has no binary float");

    assert_eq!(folded, LiteralValue::Float(1.5));
    assert!(matches!(error, FoldError::InexactDecimal(decimal) if decimal.to_string() == "0.1"));
}

#[test]
fn fold_folds_nested_native_calls_inside_out() {
    let tree = call(
        BuiltinFunction::Floor,
        [call(BuiltinFunction::Sqrt, [build_literal(10.0)])],
    );

    assert_eq!(fold_literal(&tree), LiteralValue::from(3));
}

// ---------------------------------------------------------------------------
// User natives
// ---------------------------------------------------------------------------

#[test]
fn fold_calls_a_user_native_with_its_literal_arguments() {
    let (registry, _) = build_registry();
    let natives = RecordingNatives::default();
    let tree = call(name("atan2"), [build_literal(1.0), build_literal(1.0)]);

    let folding = Evaluator::new(&registry)
        .fold(&tree, &natives)
        .expect("the fold succeeds");

    assert_eq!(
        expect_literal(folding.output()),
        &LiteralValue::Float(1.0_f64.atan2(1.0))
    );
    assert_eq!(
        *natives.calls.borrow(),
        vec![(
            "atan2".to_owned(),
            vec![LiteralValue::Float(1.0), LiteralValue::Float(1.0)]
        )]
    );
}

#[test]
fn fold_hands_a_user_native_its_decimal_arguments_as_floats() {
    let (registry, _) = build_registry();
    let natives = RecordingNatives::default();
    let tree = call(
        name("atan2"),
        [build_decimal_literal("0.5"), build_literal(1.0)],
    );

    Evaluator::new(&registry)
        .fold(&tree, &natives)
        .expect("the fold succeeds");

    assert_eq!(natives.calls.borrow()[0].1[0], LiteralValue::Float(0.5));
}

#[test]
fn fold_refuses_a_user_native_result_outside_its_result_sort() {
    let error = fold(&call(name("half_int"), [build_literal(0.0)])).expect_err("1.5 is no integer");

    assert!(matches!(
        &error,
        FoldError::ResultSort { function, sort: FunctionSort::Int, value: LiteralValue::Float(_) }
            if function.as_str() == "half_int"
    ));
    assert_eq!(
        error.to_string(),
        "native function \"half_int\" returned 1.5, which is not of its result sort int"
    );
}

#[test]
fn fold_keeps_a_user_native_failure_as_its_source() {
    let error = fold(&call(name("fails"), [build_literal(0.0)])).expect_err("it fails");

    assert!(matches!(&error, FoldError::Native { function, .. } if function.as_str() == "fails"));
    let source = std::error::Error::source(&error).expect("the implementation's error");
    assert_eq!(source.to_string(), "the implementation failed");
    assert_eq!(error.to_string(), "native function \"fails\" failed");
}

#[test]
fn fold_calls_a_shared_user_native_once() {
    let (registry, _) = build_registry();
    let natives = RecordingNatives::default();
    let shared = call(name("twice"), [build_literal(3)]);
    let tree = &shared + &shared;

    let folding = Evaluator::new(&registry)
        .fold(&tree, &natives)
        .expect("the fold succeeds");

    assert_eq!(natives.calls.borrow().len(), 1);
    assert_eq!(folding.output(), &(build_literal(6) + build_literal(6)));
}

#[test]
fn no_native_calls_fails_every_user_native() {
    let (registry, _) = build_registry();

    let error = Evaluator::new(&registry)
        .fold(&call(name("twice"), [build_literal(3)]), &NoNativeCalls)
        .expect_err("no implementation");

    assert!(matches!(error, FoldError::Native { .. }));
}

// ---------------------------------------------------------------------------
// Checks of a folded call
// ---------------------------------------------------------------------------

#[test]
fn fold_refuses_an_unknown_name_and_a_constant_called() {
    let unknown = fold(&call(name("nowhere"), [build_literal(1.0)])).expect_err("unknown");
    let constant = fold(&call(name("answer"), [])).expect_err("a constant");
    let builtin_constant = fold(&call(name("pi"), [])).expect_err("a built-in constant");

    assert!(matches!(&unknown, FoldError::UnknownFunction(name) if name.as_str() == "nowhere"));
    assert_eq!(
        unknown.to_string(),
        "no function is registered under \"nowhere\""
    );
    assert!(matches!(constant, FoldError::NotCallable(_)));
    assert!(matches!(&builtin_constant, FoldError::NotCallable(name) if name.as_str() == "pi"));
    assert_eq!(
        builtin_constant.to_string(),
        "\"pi\" is a constant, not a function"
    );
}

#[rstest]
#[case::builtin_too_many(call(BuiltinFunction::Sin, [build_literal(1.0), build_literal(2.0)]), 1, 2)]
#[case::user_too_few(call(name("atan2"), [build_literal(1.0)]), 2, 1)]
fn fold_checks_the_arity_of_a_folded_call(
    #[case] tree: Expression,
    #[case] expected: usize,
    #[case] actual: usize,
) {
    let error = fold(&tree).expect_err("wrong arity");

    assert!(matches!(
        error,
        FoldError::Arity { expected: e, actual: a, .. } if e == expected && a == actual
    ));
}

#[test]
fn fold_checks_the_argument_sorts_of_a_folded_call() {
    let boolean = fold(&call(BuiltinFunction::Exp, [build_literal(true)])).expect_err("bool");
    let real_for_int = fold(&call(name("twice"), [build_literal(1.5)])).expect_err("a real");

    assert!(matches!(
        boolean,
        FoldError::ArgumentSort {
            position: 0,
            sort: FunctionSort::Real,
            ..
        }
    ));
    assert_eq!(
        real_for_int.to_string(),
        "argument 0 of \"twice\" must be int, got 1.5"
    );
}

#[test]
fn fold_checks_the_arguments_before_the_call_taking_them() {
    let tree = call(
        name("nowhere"),
        [call(BuiltinFunction::Round, [build_literal(f64::NAN)])],
    );

    let error = fold(&tree).expect_err("the argument fails first");

    assert!(matches!(error, FoldError::NonFiniteCast { .. }));
}

#[test]
fn fold_keeps_a_native_call_with_a_non_literal_argument_unchecked() {
    let (_, x) = build_identifier("x");
    let tree = call(BuiltinFunction::Sin, [x.clone(), x]);

    let output = fold(&tree).expect("nothing is folded");

    assert!(Expression::ptr_eq(&output, &tree));
}

// ---------------------------------------------------------------------------
// Functions with a body, constants, and nodes kept
// ---------------------------------------------------------------------------

#[test]
fn fold_keeps_calls_of_functions_with_a_body_and_lists_them_once() {
    let (registry, _) = build_registry();
    let tree = call(name("double"), [build_literal(1.0)])
        + call(BuiltinFunction::Relu, [build_literal(1.0)])
        + call(name("double"), [build_literal(2.0)]);

    let folding = Evaluator::new(&registry)
        .fold(&tree, &RecordingNatives::default())
        .expect("the fold succeeds");

    assert!(Expression::ptr_eq(folding.output(), &tree));
    assert_eq!(
        folding.not_inlined(),
        [
            Callee::Named(name("double")),
            Callee::Builtin(BuiltinFunction::Relu)
        ]
    );
}

#[test]
fn fold_resolves_builtin_and_user_constants() {
    let (registry, answer) = build_registry();
    let pi = Expression::from(BuiltinConstant::Pi.identifier().clone());
    let tree = pi + Expression::from(answer);

    let folding = Evaluator::new(&registry)
        .fold(&tree, &RecordingNatives::default())
        .expect("the fold succeeds");

    assert_eq!(
        folding.output(),
        &(build_literal(std::f64::consts::PI) + build_literal(42))
    );
}

#[test]
fn fold_leaves_an_identifier_merely_named_like_a_constant() {
    let (_, look_alike) = build_identifier("pi");

    let output = fold(&look_alike).expect("nothing to fold");

    assert!(Expression::ptr_eq(&output, &look_alike));
}

#[test]
fn fold_does_not_fold_arithmetic_or_a_literal_piecewise() {
    let arithmetic = build_literal(1) + build_literal(2);
    let negation = -build_literal(1.0);
    let piecewise = Expression::piecewise([(build_literal(true), build_literal(1))], 2)
        .expect("a boolean condition");

    for tree in [arithmetic, negation, piecewise] {
        let output = fold(&tree).expect("nothing to fold");
        assert!(Expression::ptr_eq(&output, &tree), "{tree}");
    }
}

#[test]
fn fold_recurses_into_children_and_shares_what_it_keeps() {
    let (_, x) = build_identifier("x");
    let kept = &x * 2;
    let tree = &kept + call(BuiltinFunction::Exp, [build_literal(0.0)]);

    let output = fold(&tree).expect("the fold succeeds");

    assert_eq!(output, &kept + build_literal(1.0));
    let children: Vec<&Expression> = output.children().collect();
    assert!(Expression::ptr_eq(children[0], &kept));
}

#[test]
fn fold_folds_a_shared_dag_once_per_distinct_node() {
    let leaf = call(BuiltinFunction::Exp, [build_literal(0.0)]);
    let dag = build_doubling_dag(&leaf, 64);

    let output = fold(&dag).expect("the fold succeeds");

    let mut seen: HashSet<NodeIdentity> = HashSet::new();
    let mut pending = vec![&output];
    while let Some(node) = pending.pop() {
        if seen.insert(node.identity()) {
            pending.extend(node.children());
        }
    }
    assert_eq!(seen.len(), 65);
}

#[test]
fn fold_walks_a_deep_tree_on_a_small_stack() {
    let folded_leaf = run_on_small_stack(|| {
        let mut tree = call(BuiltinFunction::Exp, [build_literal(0.0)]);
        for _ in 0..SMALL_STACK_DEPTH {
            tree = -tree;
        }
        let output = fold(&tree).expect("the fold succeeds");
        let mut node = &output;
        while let Some(child) = node.children().next() {
            node = child;
        }
        expect_literal(node).clone()
    });

    assert_eq!(folded_leaf, LiteralValue::Float(1.0));
}
