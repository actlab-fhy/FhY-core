//! Tests for `FunctionRegistry::inline`: user functions and composed
//! built-ins replaced by their bodies, native calls kept with their arity
//! checked, the refusals of unknown names, constants, wrong arities and
//! recursion, the input returned itself when nothing is inlined, sharing,
//! and depth.

use crate::support::expression as expression_support;
use crate::support::stack as stack_support;

use std::collections::HashSet;

use fhy_core::expression::builtins::BuiltinFunction;
use fhy_core::expression::registry::{
    FunctionDefinition, FunctionRegistry, InlineError, NativeConstant, NativeFunction,
};
use fhy_core::expression::{
    Callee, Expression, ExpressionKind, FunctionName, FunctionSort, PiecewiseError,
};
use fhy_core::identifier::Identifier;
use fhy_core::tree::{NodeHandle, NodeIdentity};
use rstest::rstest;

use expression_support::{
    build_doubling_dag, build_identifier, build_literal, call, expect_piecewise,
};
use stack_support::{SMALL_STACK_DEPTH, run_on_small_stack};

fn name(text: &str) -> FunctionName {
    FunctionName::new(text).expect("the test names no built-in")
}

fn call_named(function: &str, arguments: impl IntoIterator<Item = Expression>) -> Expression {
    call(name(function), arguments)
}

/// Define `function(x) = build_body(x)` over the reals.
fn define(
    function: &str,
    build_body: impl FnOnce(&Expression) -> Expression,
) -> FunctionDefinition {
    let (x, reference) = build_identifier("x");
    FunctionDefinition::new(
        name(function),
        [x],
        [FunctionSort::Real],
        FunctionSort::Real,
        build_body(&reference),
    )
    .expect("one parameter with one sort")
}

/// Return a registry holding `double(x) = x * 2`.
fn build_double_registry() -> FunctionRegistry {
    let mut registry = FunctionRegistry::new();
    registry
        .register_function(define("double", |x: &Expression| x * 2))
        .expect("the name is free");
    registry
}

/// Return the number of distinct nodes of `expression`, walking each shared
/// node once, without recursion.
fn count_distinct_nodes(expression: &Expression) -> usize {
    let mut seen: HashSet<NodeIdentity> = HashSet::new();
    let mut pending = vec![expression];
    while let Some(node) = pending.pop() {
        if seen.insert(node.identity()) {
            pending.extend(node.children());
        }
    }
    seen.len()
}

/// Return whether `expression` calls a composed built-in or a user
/// function anywhere.
fn calls_an_inlinable_function(expression: &Expression) -> bool {
    let mut seen: HashSet<NodeIdentity> = HashSet::new();
    let mut pending = vec![expression];
    while let Some(node) = pending.pop() {
        if !seen.insert(node.identity()) {
            continue;
        }
        if let ExpressionKind::Call(call) = node.kind() {
            match call.callee() {
                Callee::Builtin(function) if function.composed().is_some() => return true,
                Callee::Named(_) => return true,
                Callee::Builtin(_) => {}
            }
        }
        pending.extend(node.children());
    }
    false
}

// ---------------------------------------------------------------------------
// What is inlined
// ---------------------------------------------------------------------------

#[test]
fn inline_replaces_a_user_call_by_its_body_over_the_arguments() {
    let registry = build_double_registry();
    let (_, y) = build_identifier("y");

    let inlined = registry
        .inline(&call_named("double", [y.clone()]))
        .expect("double is registered");

    assert_eq!(inlined, &y * 2);
}

#[test]
fn inline_substitutes_the_argument_object_at_every_use() {
    let mut registry = FunctionRegistry::new();
    registry
        .register_function(define("square", |x: &Expression| x * x))
        .expect("free");
    let (_, y) = build_identifier("y");
    let argument = &y + 1;

    let inlined = registry
        .inline(&call_named("square", [argument.clone()]))
        .expect("square is registered");

    let ExpressionKind::Binary(product) = inlined.kind() else {
        panic!("expected a product, got {inlined:?}");
    };
    assert!(Expression::ptr_eq(product.left(), &argument));
    assert!(Expression::ptr_eq(product.right(), &argument));
}

#[test]
fn inline_replaces_a_composed_builtin_call_by_the_catalogue_body() {
    let registry = FunctionRegistry::new();
    let (_, y) = build_identifier("y");

    let inlined = registry
        .inline(&call(BuiltinFunction::Relu, [y.clone()]))
        .expect("relu is built in");

    let expected = build_relu_of(&y);
    assert_eq!(inlined, expected);
}

/// Return `relu(x)` inlined: `{x if (x > 0 || x != x); 0 otherwise}`.
fn build_relu_of(x: &Expression) -> Expression {
    Expression::piecewise([(x.greater(0_i64).or(x.not_equals(x)), x.clone())], 0_i64)
        .expect("a comparison condition")
}

#[rstest]
#[case::max(BuiltinFunction::Max)]
#[case::xor(BuiltinFunction::Xor)]
#[case::clamp_symmetric(BuiltinFunction::ClampSymmetric)]
#[case::gelu(BuiltinFunction::Gelu)]
#[case::silu(BuiltinFunction::Silu)]
fn inline_leaves_no_composed_call_of_a_composed_builtin(#[case] function: BuiltinFunction) {
    let registry = FunctionRegistry::new();
    let arguments: Vec<Expression> = function
        .parameter_sorts()
        .iter()
        .enumerate()
        .map(|(index, _)| Expression::from(Identifier::new(&format!("a{index}"))))
        .collect();

    let inlined = registry
        .inline(&call(function, arguments))
        .expect("the composed built-ins are defined");

    assert!(!calls_an_inlinable_function(&inlined), "{inlined}");
}

#[test]
fn inline_expands_nested_calls_inside_out() {
    let registry = FunctionRegistry::new();
    let (_, y) = build_identifier("y");
    let inner = build_relu_of(&y);

    let inlined = registry
        .inline(&call(
            BuiltinFunction::Relu,
            [call(BuiltinFunction::Relu, [y])],
        ))
        .expect("relu is built in");

    let expected = build_relu_of(&inner);
    assert_eq!(inlined, expected);
    let outer = expect_piecewise(&inlined);
    let (condition, value) = &outer.cases()[0];
    let ExpressionKind::Logical(disjunction) = condition.kind() else {
        panic!("expected a disjunction, got {condition:?}");
    };
    let ExpressionKind::Binary(comparison) = disjunction.operands()[0].kind() else {
        panic!("expected a comparison, got {condition:?}");
    };
    assert!(
        Expression::ptr_eq(comparison.left(), value),
        "the inlined argument is one node at both uses"
    );
}

#[test]
fn inline_follows_a_chain_of_user_functions() {
    let mut registry = FunctionRegistry::new();
    registry
        .register_function(define("h", |x: &Expression| x + 1))
        .expect("free");
    registry
        .register_function(define("g", |x: &Expression| {
            call_named("h", [x.clone()]) * 2
        }))
        .expect("free");
    registry
        .register_function(define("f", |x: &Expression| -call_named("g", [x.clone()])))
        .expect("free");
    let (_, y) = build_identifier("y");

    let inlined = registry
        .inline(&call_named("f", [y.clone()]))
        .expect("the chain is registered");

    assert_eq!(inlined, -((&y + 1) * 2));
}

#[test]
fn inline_expands_a_user_body_calling_a_composed_builtin() {
    let mut registry = FunctionRegistry::new();
    registry
        .register_function(define("f", |x: &Expression| {
            call(BuiltinFunction::Relu, [x.clone()]) + 1
        }))
        .expect("free");
    let (_, y) = build_identifier("y");

    let inlined = registry
        .inline(&call_named("f", [y.clone()]))
        .expect("f is registered");

    assert_eq!(inlined, build_relu_of(&y) + 1);
}

#[test]
fn inline_uses_a_function_registered_after_its_caller() {
    let mut registry = FunctionRegistry::new();
    registry
        .register_function(define("f", |x: &Expression| {
            call_named("later", [x.clone()])
        }))
        .expect("free");
    registry
        .register_function(define("later", |x: &Expression| x - 1))
        .expect("free");
    let (_, y) = build_identifier("y");

    let inlined = registry
        .inline(&call_named("f", [y.clone()]))
        .expect("both are registered by now");

    assert_eq!(inlined, y - 1);
}

// ---------------------------------------------------------------------------
// What is kept
// ---------------------------------------------------------------------------

#[test]
fn inline_keeps_a_native_builtin_call_and_inlines_its_arguments() {
    let registry = build_double_registry();
    let (_, y) = build_identifier("y");

    let inlined = registry
        .inline(&call(
            BuiltinFunction::Exp,
            [call_named("double", [y.clone()])],
        ))
        .expect("exp is built in");

    assert_eq!(inlined, call(BuiltinFunction::Exp, [&y * 2]));
}

#[test]
fn inline_keeps_a_native_user_call() {
    let mut registry = FunctionRegistry::new();
    registry
        .register_native_function(NativeFunction::new(
            name("softplus"),
            [FunctionSort::Real],
            FunctionSort::Real,
        ))
        .expect("free");
    let (_, y) = build_identifier("y");
    let expression = call_named("softplus", [y]);

    let inlined = registry
        .inline(&expression)
        .expect("softplus is registered");

    assert!(Expression::ptr_eq(&inlined, &expression));
}

#[test]
fn inline_returns_the_input_itself_when_nothing_is_inlined() {
    let registry = FunctionRegistry::new();
    let (_, y) = build_identifier("y");
    let expression = (call(BuiltinFunction::Sqrt, [y.clone()]) + &y) * 3;

    let inlined = registry.inline(&expression).expect("nothing to inline");

    assert!(Expression::ptr_eq(&inlined, &expression));
}

#[test]
fn inline_shares_every_subtree_without_a_call_to_inline() {
    let registry = build_double_registry();
    let (_, y) = build_identifier("y");
    let (_, z) = build_identifier("z");
    let untouched = (&z * 5) - 1;
    let expression = call_named("double", [y.clone()]) + untouched.clone();

    let inlined = registry.inline(&expression).expect("double is registered");

    let ExpressionKind::Binary(sum) = inlined.kind() else {
        panic!("expected a sum, got {inlined:?}");
    };
    assert!(Expression::ptr_eq(sum.right(), &untouched));
    assert_eq!(sum.left(), &(&y * 2));
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

#[test]
fn inline_refuses_a_call_of_an_unknown_name() {
    let registry = FunctionRegistry::new();

    let error = registry
        .inline(&call_named("missing", [build_literal(1)]))
        .expect_err("nothing is registered under missing");

    assert_eq!(error, InlineError::UnknownFunction(name("missing")));
    assert_eq!(
        error.to_string(),
        r#"no function is registered under "missing""#
    );
}

#[test]
fn inline_refuses_a_call_of_a_constant() {
    let mut registry = FunctionRegistry::new();
    registry
        .register_constant(
            NativeConstant::new(name("c"), FunctionSort::Int, 3).expect("an integer"),
        )
        .expect("free");

    let error = registry
        .inline(&call_named("c", []))
        .expect_err("a constant is not callable");

    assert_eq!(error, InlineError::NotCallable(name("c")));
    assert_eq!(error.to_string(), r#""c" is a constant, not a function"#);
}

#[rstest]
fn inline_refuses_a_call_of_a_builtin_constant(#[values("pi", "e", "inf", "nan")] constant: &str) {
    let registry = FunctionRegistry::new();

    let error = registry
        .inline(&call_named(constant, []))
        .expect_err("a built-in constant is not callable");

    assert_eq!(error, InlineError::NotCallable(name(constant)));
}

#[rstest]
#[case::too_few_for_a_user_function("double", 0, 1)]
#[case::too_many_for_a_user_function("double", 2, 1)]
#[case::too_many_for_a_native_user_function("softplus", 3, 1)]
fn inline_refuses_a_user_call_of_the_wrong_arity(
    #[case] function: &str,
    #[case] argument_count: usize,
    #[case] parameter_count: usize,
) {
    let mut registry = build_double_registry();
    registry
        .register_native_function(NativeFunction::new(
            name("softplus"),
            [FunctionSort::Real],
            FunctionSort::Real,
        ))
        .expect("free");
    let arguments = vec![build_literal(1); argument_count];

    let error = registry
        .inline(&call_named(function, arguments))
        .expect_err("the argument count is wrong");

    assert_eq!(
        error,
        InlineError::ArityMismatch {
            callee: Callee::Named(name(function)),
            expected: parameter_count,
            actual: argument_count,
        }
    );
}

#[rstest]
#[case::composed_too_few(BuiltinFunction::Max, 1)]
#[case::composed_too_many(BuiltinFunction::Relu, 2)]
#[case::native_too_few(BuiltinFunction::Exp, 0)]
#[case::native_too_many(BuiltinFunction::Floor, 2)]
fn inline_refuses_a_builtin_call_of_the_wrong_arity(
    #[case] function: BuiltinFunction,
    #[case] argument_count: usize,
) {
    let registry = FunctionRegistry::new();

    let error = registry
        .inline(&call(function, vec![build_literal(1.5); argument_count]))
        .expect_err("the argument count is wrong");

    assert_eq!(
        error,
        InlineError::ArityMismatch {
            callee: Callee::Builtin(function),
            expected: function.parameter_sorts().len(),
            actual: argument_count,
        }
    );
}

#[rstest]
#[case::one_expected(1, 2, r#""f" takes 1 argument but the call passes 2"#)]
#[case::two_expected(2, 1, r#""f" takes 2 arguments but the call passes 1"#)]
fn arity_mismatch_displays_both_counts(
    #[case] expected: usize,
    #[case] actual: usize,
    #[case] message: &str,
) {
    let error = InlineError::ArityMismatch {
        callee: Callee::Named(name("f")),
        expected,
        actual,
    };

    assert_eq!(error.to_string(), message);
}

#[test]
fn inline_refuses_a_self_recursive_function() {
    let mut registry = FunctionRegistry::new();
    registry
        .register_function(define("f", |x: &Expression| {
            call_named("f", [x.clone()]) + 1
        }))
        .expect("free");

    let error = registry
        .inline(&call_named("f", [build_literal(1)]))
        .expect_err("f calls itself");

    assert_eq!(error, InlineError::Recursive(name("f")));
    assert_eq!(
        error.to_string(),
        r#"function "f" is recursive and cannot be inlined"#
    );
}

#[test]
fn inline_refuses_mutually_recursive_functions_naming_the_first_reached_again() {
    let mut registry = FunctionRegistry::new();
    registry
        .register_function(define("f", |x: &Expression| call_named("g", [x.clone()])))
        .expect("free");
    registry
        .register_function(define("g", |x: &Expression| {
            call_named("f", [x.clone()]) * 2
        }))
        .expect("free");

    let error = registry
        .inline(&call_named("f", [build_literal(1)]))
        .expect_err("f and g call each other");

    assert_eq!(error, InlineError::Recursive(name("f")));
}

#[test]
fn inline_accepts_a_function_called_twice_but_not_inside_itself() {
    let mut registry = build_double_registry();
    registry
        .register_function(define("quadruple", |x: &Expression| {
            call_named("double", [call_named("double", [x.clone()])])
        }))
        .expect("free");
    let (_, y) = build_identifier("y");

    let inlined = registry
        .inline(&call_named("quadruple", [y.clone()]))
        .expect("double nested in its own argument is no recursion");

    assert_eq!(inlined, (&y * 2) * 2);
}

#[test]
fn inline_checks_the_arguments_before_the_call_taking_them() {
    let registry = FunctionRegistry::new();

    let error = registry
        .inline(&call_named("outer", [call_named("inner", [])]))
        .expect_err("neither is registered");

    assert_eq!(error, InlineError::UnknownFunction(name("inner")));
}

#[test]
fn inline_refuses_a_substitution_that_breaks_a_piecewise_condition() {
    let mut registry = FunctionRegistry::new();
    let (flag, flag_reference) = build_identifier("flag");
    registry
        .register_function(
            FunctionDefinition::new(
                name("choose"),
                [flag],
                [FunctionSort::Bool],
                FunctionSort::Real,
                Expression::piecewise([(flag_reference, build_literal(1))], 0)
                    .expect("an identifier condition"),
            )
            .expect("one parameter with one sort"),
        )
        .expect("free");

    let error = registry
        .inline(&call_named("choose", [build_literal(3)]))
        .expect_err("3 cannot be a condition");

    assert!(
        matches!(
            error,
            InlineError::Piecewise(PiecewiseError::NonBooleanConditionLiteral { case_index: 0 })
        ),
        "{error:?}"
    );
    assert_eq!(error.to_string(), "inlining built an invalid piecewise");
    assert_eq!(
        std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<PiecewiseError>()),
        Some(&PiecewiseError::NonBooleanConditionLiteral { case_index: 0 })
    );
}

// ---------------------------------------------------------------------------
// Sharing and depth
// ---------------------------------------------------------------------------

#[test]
fn inline_expands_a_shared_call_once_per_distinct_node() {
    let registry = FunctionRegistry::new();
    let (_, z) = build_identifier("z");
    let levels = 64;
    let dag = build_doubling_dag(&call(BuiltinFunction::Sigmoid, [z]), levels);

    let inlined = registry.inline(&dag).expect("sigmoid is built in");

    let leaf_nodes = count_distinct_nodes(
        &registry
            .inline(&call(
                BuiltinFunction::Sigmoid,
                [Expression::from(Identifier::new("z"))],
            ))
            .expect("sigmoid is built in"),
    );
    assert_eq!(count_distinct_nodes(&inlined), levels + leaf_nodes);
    assert!(!calls_an_inlinable_function(&inlined));
}

#[test]
fn inline_of_nested_composed_calls_takes_linear_time() {
    let registry = FunctionRegistry::new();
    let depth = 1_000;
    let mut tree = Expression::from(Identifier::new("x"));
    for _ in 0..depth {
        tree = call(BuiltinFunction::Relu, [tree]);
    }

    let inlined = registry
        .inline(&tree)
        .expect("relu is built in, and nesting is no recursion");

    // Each level adds a piecewise, its comparison and the literal zeros, and
    // reuses the level below at both of its uses.
    assert!(count_distinct_nodes(&inlined) <= 5 * depth + 1);
    assert!(!calls_an_inlinable_function(&inlined));
}

#[test]
fn inline_walks_a_deep_tree_on_a_small_stack() {
    run_on_small_stack(|| {
        let registry = build_double_registry();
        let (_, y) = build_identifier("y");
        let mut tree = call_named("double", [y]);
        for _ in 0..SMALL_STACK_DEPTH {
            tree = -tree;
        }

        let inlined = registry.inline(&tree).expect("double is registered");

        assert!(!Expression::ptr_eq(&inlined, &tree));
        assert!(!calls_an_inlinable_function(&inlined));
        assert_eq!(count_distinct_nodes(&inlined), SMALL_STACK_DEPTH + 3);
    });
}

#[test]
fn inline_returns_a_deep_tree_without_calls_itself_on_a_small_stack() {
    run_on_small_stack(|| {
        let registry = FunctionRegistry::new();
        let mut tree = Expression::from(Identifier::new("y"));
        for _ in 0..SMALL_STACK_DEPTH {
            tree = -tree;
        }

        let inlined = registry.inline(&tree).expect("nothing to inline");

        assert!(Expression::ptr_eq(&inlined, &tree));
    });
}
