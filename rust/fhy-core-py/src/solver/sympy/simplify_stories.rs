//! Stories for simplifying with SymPy, [`SympySimplifier`] as a
//! [`Simplifier`] and its SymPy-level operations: decided comparisons, the workarounds, the best-effort cases,
//! the substitution, the phases of errors, threads, and the prelude.

use std::collections::HashMap;
use std::sync::Arc;
use std::thread;

use fhy_core::expression::builtins::BuiltinConstant;
use fhy_core::expression::registry::FunctionRegistry;
use fhy_core::expression::{Expression, ExpressionKind, LiteralValue, NoRegisteredSorts};
use fhy_core::solver::{Simplifier, SimplifyContext, SolveError, Solver};
use pyo3::exceptions::PyKeyboardInterrupt;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::test_support::{
    attached, backend, build_identifier, build_literal, evaluate, srepr, with_patched_sympy,
};
use super::{SympyError, SympyErrorKind, SympyPhase, SympySimplifier};

/// Return the solver holding a new SymPy backend.
fn solver() -> Solver {
    backend();
    Solver::new().with_simplifier(SympySimplifier::new())
}

/// Return the simplification of `expression` after substituting
/// `environment`, with an empty registry.
fn simplified(
    expression: &Expression,
    environment: &HashMap<fhy_core::identifier::Identifier, Expression>,
) -> Result<Expression, SolveError> {
    let registry = FunctionRegistry::new();
    solver().simplify(
        expression,
        environment,
        &SimplifyContext::from_registry(&registry),
    )
}

/// Return the SymPy error of a failed simplification.
fn sympy_error(error: SolveError) -> SympyError {
    let SolveError::Backend { backend, source } = error else {
        panic!("expected a backend error, got {error:?}");
    };
    assert_eq!(backend, "sympy");
    *source.downcast::<SympyError>().expect("a sympy error")
}

#[test]
fn bound_comparison_decides_to_a_boolean_literal() {
    let (x, reference) = build_identifier("x");

    let holds = simplified(
        &reference.greater_equal(0),
        &HashMap::from([(x.clone(), build_literal(3))]),
    );
    let fails = simplified(
        &reference.greater(5),
        &HashMap::from([(x, build_literal(3))]),
    );

    assert_eq!(holds.expect("simplified"), build_literal(true));
    assert_eq!(fails.expect("simplified"), build_literal(false));
}

#[test]
fn symbolic_expression_simplifies() {
    let (_, reference) = build_identifier("x");

    let result = simplified(&(&(&reference + &reference) - &reference), &HashMap::new());

    assert_eq!(result.expect("simplified"), reference);
}

#[test]
fn decimal_arithmetic_is_exact() {
    let tenth = || Expression::literal(LiteralValue::Decimal("0.1".parse().expect("a decimal")));
    let three_tenths =
        Expression::literal(LiteralValue::Decimal("0.3".parse().expect("a decimal")));

    let result = simplified(
        &(tenth() + tenth() + tenth()).equals(three_tenths),
        &HashMap::new(),
    );

    assert_eq!(result.expect("simplified"), build_literal(true));
}

#[test]
fn builtin_constant_simplifies_with_its_sympy_value() {
    let pi = Expression::from(BuiltinConstant::Pi.identifier().clone());

    let result = simplified(&pi.greater(3), &HashMap::new());

    assert_eq!(result.expect("simplified"), build_literal(true));
}

#[test]
fn floor_mod_of_an_all_even_piecewise_keeps_its_parity_unknown() {
    let (_, b) = build_identifier("b");
    let even = Expression::piecewise([(b, build_literal(2))], 0).expect("a piecewise");

    let result = simplified(&even.floor_mod(-6), &HashMap::new()).expect("simplified");

    assert!(
        !matches!(result.kind(), ExpressionKind::Literal(_)),
        "sympy 1.14 would decide it 0: {result}"
    );
}

#[test]
fn comparison_between_booleans_is_simplified_on_its_own() {
    let (_, x) = build_identifier("x");
    let (b, b_reference) = build_identifier("b");
    let expression = x.less(1).equals(&b_reference).and(x.greater(-5));

    let result = simplified(&expression, &HashMap::from([(b, build_literal(true))]));

    assert!(result.is_ok(), "{result:?}");
}

#[test]
fn piecewise_with_a_boolean_symbol_condition_keeps_its_later_branches() {
    let (b, b_reference) = build_identifier("b");
    let (x, x_reference) = build_identifier("x");
    let choice = Expression::piecewise(
        [
            (b_reference, build_literal(1)),
            (x_reference.greater(0), build_literal(2)),
        ],
        3,
    )
    .expect("a piecewise");

    let first = simplified(
        &choice,
        &HashMap::from([
            (b.clone(), build_literal(false)),
            (x.clone(), build_literal(1)),
        ]),
    );
    let last = simplified(
        &choice,
        &HashMap::from([(b, build_literal(false)), (x, build_literal(-1))]),
    );

    assert_eq!(first.expect("simplified"), build_literal(2));
    assert_eq!(last.expect("simplified"), build_literal(3));
}

#[test]
fn precision_exhausted_keeps_the_unsimplified_form() {
    let (_, reference) = build_identifier("x");
    let expression = reference.less(1);

    let result = with_patched_sympy(
        "simplify",
        "lambda *args, **kwargs: (_ for _ in ()).throw(sympy.core.evalf.PrecisionExhausted())",
        || simplified(&expression, &HashMap::new()),
    );

    assert_eq!(result.expect("kept"), expression);
}

#[test]
fn partial_piecewise_result_keeps_the_unsimplified_form() {
    let (_, reference) = build_identifier("x");
    let expression = reference.less(1);

    let result = with_patched_sympy(
        "simplify",
        "lambda *args, **kwargs: sympy.Piecewise((1, sympy.Symbol('flag_0')))",
        || simplified(&expression, &HashMap::new()),
    );

    assert_eq!(result.expect("kept"), expression);
}

#[test]
fn simplify_is_read_from_the_module_at_each_call() {
    let (_, reference) = build_identifier("x");

    let result = with_patched_sympy(
        "simplify",
        "lambda *args, **kwargs: sympy.Integer(7)",
        || simplified(&reference.less(1), &HashMap::new()),
    );

    assert_eq!(result.expect("simplified"), build_literal(7));
}

#[test]
fn keyboard_interrupt_in_simplify_returns_as_the_exception_itself() {
    let (_, reference) = build_identifier("x");

    let error = with_patched_sympy(
        "simplify",
        "lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt())",
        || simplified(&reference.less(1), &HashMap::new()),
    )
    .expect_err("interrupted");

    let error = sympy_error(error);
    assert_eq!(error.phase(), SympyPhase::Simplification);
    let SympyErrorKind::Python(exception) = error.into_kind() else {
        panic!("expected the exception");
    };
    attached(|py| assert!(exception.is_instance_of::<PyKeyboardInterrupt>(py)));
}

#[test]
fn quotient_by_zero_fails_in_the_lifting() {
    let error =
        simplified(&(build_literal(1) / build_literal(0)), &HashMap::new()).expect_err("zoo");

    let error = sympy_error(error);
    assert_eq!(error.phase(), SympyPhase::Lifting);
    assert!(matches!(error.kind(), SympyErrorKind::ComplexInfinity));
}

#[test]
fn backend_is_named_sympy() {
    assert_eq!(SympySimplifier::new().name(), "sympy");
}

// ---------------------------------------------------------------------------
// The SymPy-level operations
// ---------------------------------------------------------------------------

#[test]
fn simplify_object_simplifies_a_sympy_object() {
    let (x, _) = build_identifier("x");

    let text = attached(|py| {
        let object = evaluate(
            py,
            &format!(
                "(lambda x: x + x - x)(sympy.Symbol('{}_{}'))",
                x.name_hint(),
                x.id()
            ),
        );
        srepr(&backend().simplify_object(&object).expect("simplified"))
    });

    assert_eq!(text, format!("Symbol('{}_{}')", x.name_hint(), x.id()));
}

#[test]
fn substitution_is_simultaneous() {
    let (x, x_reference) = build_identifier("x");
    let (y, y_reference) = build_identifier("y");
    let context = SimplifyContext::new(&NoRegisteredSorts);

    let result = attached(|py| {
        let object = backend()
            .lower(py, &x_reference.less(5), &context)
            .expect("lowered");
        let environment = HashMap::from([(x, y_reference.clone()), (y, build_literal(5))]);
        let substituted = backend()
            .substitute(&object, &environment, &context)
            .expect("substituted");
        backend().lift(&substituted).expect("lifted")
    });

    assert_eq!(result, y_reference.less(5));
}

#[test]
fn substitution_keeps_a_bound_boolean_piecewise_boolean() {
    let (b, b_reference) = build_identifier("b");
    let (c, c_reference) = build_identifier("c");
    let context = SimplifyContext::new(&NoRegisteredSorts);
    let choice = Expression::piecewise([(&c_reference, build_literal(true))], build_literal(false))
        .expect("a piecewise");

    let result = attached(|py| {
        let object = backend()
            .lower(py, &b_reference.equals(build_literal(true)), &context)
            .expect("lowered");
        let substituted = backend()
            .substitute(&object, &HashMap::from([(b, choice)]), &context)
            .expect("substituted");
        srepr(&substituted)
    });

    assert!(
        !result.contains("Piecewise") && result.contains(&format!("{}_{}", c.name_hint(), c.id())),
        "the piecewise is compared as a Boolean: {result}"
    );
}

#[test]
fn substitution_refuses_binding_a_referenced_native_constant() {
    let pi = BuiltinConstant::Pi.identifier().clone();
    let context = SimplifyContext::new(&NoRegisteredSorts);

    let error = attached(|py| {
        let object = evaluate(py, &format!("sympy.Symbol('pi_{}') > 3", pi.id()));
        backend()
            .substitute(
                &object,
                &HashMap::from([(pi.clone(), build_literal(1))]),
                &context,
            )
            .expect_err("refused")
    });

    assert_eq!(error.phase(), SympyPhase::Substitution);
    assert!(matches!(error.kind(), SympyErrorKind::BoundNativeConstant(ids) if ids == &[pi]));
    assert!(error.to_string().contains("pi"));
}

#[test]
fn substitute_symbols_applies_a_mapping_of_symbols() {
    let text = attached(|py| {
        let object = evaluate(py, "sympy.Symbol('x_1') + sympy.Symbol('y_2')");
        let replacements = PyDict::new(py);
        replacements
            .set_item(
                evaluate(py, "sympy.Symbol('x_1')"),
                evaluate(py, "sympy.Integer(3)"),
            )
            .expect("set");
        srepr(
            &backend()
                .substitute_symbols(&object, replacements.as_any())
                .expect("substituted"),
        )
    });

    assert_eq!(text, "Add(Symbol('y_2'), Integer(3))");
}

// ---------------------------------------------------------------------------
// Threads and the prelude
// ---------------------------------------------------------------------------

#[test]
fn threads_share_one_backend() {
    let shared = Arc::new(solver());

    let results: Vec<Expression> = (0..8)
        .map(|index| {
            let shared = Arc::clone(&shared);
            thread::spawn(move || {
                let (x, reference) = build_identifier("x");
                let registry = FunctionRegistry::new();
                shared
                    .simplify(
                        &reference.greater(3),
                        &HashMap::from([(x, build_literal(index))]),
                        &SimplifyContext::from_registry(&registry),
                    )
                    .expect("simplified")
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|handle| handle.join().expect("joined"))
        .collect();

    let expected: Vec<Expression> = (0..8).map(|index| build_literal(index > 3)).collect();
    assert_eq!(results, expected);
}

#[test]
fn backends_share_one_prelude_per_interpreter() {
    let (_, b) = build_identifier("b");
    let piecewise = Expression::piecewise([(b, build_literal(1))], 2).expect("a piecewise");
    let context = SimplifyContext::new(&NoRegisteredSorts);
    let first = SympySimplifier::new();
    let second = SympySimplifier::new();

    let is_one_class = attached(|py| {
        let one = first.lower(py, &piecewise, &context).expect("lowered");
        let other = second.lower(py, &piecewise, &context).expect("lowered");
        one.get_type().is(other.get_type())
            && one
                .get_type()
                .is(evaluate(py, "prelude.ParityOpaquePiecewise"))
    });

    assert!(is_one_class);
}

#[test]
fn lowered_round_node_pickles_within_the_process() {
    let (_, reference) = build_identifier("x");
    let call = Expression::call(
        fhy_core::expression::builtins::BuiltinFunction::Round,
        [reference.clone()],
    );
    let context = SimplifyContext::new(&NoRegisteredSorts);

    let restored = attached(|py| {
        let lowered = backend().lower(py, &call, &context).expect("lowered");
        let pickle = py.import("pickle").expect("pickle");
        let bytes = pickle.call_method1("dumps", (lowered,)).expect("pickled");
        let loaded = pickle.call_method1("loads", (bytes,)).expect("unpickled");
        backend().lift(&loaded).expect("lifted")
    });

    assert_eq!(restored, call);
}

#[test]
fn load_succeeds_and_debug_names_the_backend() {
    let fresh = SympySimplifier::new();
    backend();

    attached(|py| fresh.load(py)).expect("loaded");
    assert!(format!("{fresh:?}").starts_with("SympySimplifier"));
}
