//! Tests for `EquationConstraint`: the bindings it keeps and lifts, the
//! order of its checks, the native constant refusal, how it reads the
//! simplifier's result, and its events.
//!
//! The cases are ported from `test_equation_constraint.py` and
//! `test_bindings_evaluation.py`; a fake simplifier stands in for SymPy.

use std::sync::Arc;

use fhy_core::constraint::{
    Binding, Bindings, ConstraintContext, ConstraintError, EquationConstraint, Outcome,
    UnusableBindingReason, Value,
};
use fhy_core::expression::builtins::BuiltinConstant;
use fhy_core::expression::registry::FunctionRegistry;
use fhy_core::expression::{Decimal, Expression, LiteralValue};
use fhy_core::identifier::Identifier;
use fhy_core::solver::{SolveError, Solver};
use rstest::rstest;

use crate::support::constraint::{RecordedEvent, RecordingObserver, TestOpaque, bind, int, text};
use crate::support::solver::RecordingSimplifier;

fn evaluate_with(
    constraint: &EquationConstraint,
    bindings: &Bindings,
    simplifier: &Arc<RecordingSimplifier>,
) -> (Result<Outcome, ConstraintError>, Vec<RecordedEvent>) {
    let solver = simplifier.solver();
    let observer = RecordingObserver::default();
    let context = ConstraintContext::new(&solver).with_observer(&observer);
    let outcome = constraint.evaluate(bindings, &context);
    (outcome, observer.events())
}

#[rstest]
#[case::true_is_satisfied(true, Outcome::Satisfied)]
#[case::false_is_violated(false, Outcome::Violated)]
fn boolean_literal_result_decides_the_equation(#[case] result: bool, #[case] expected: Outcome) {
    let x = Identifier::new("x");
    let constraint = EquationConstraint::new(Expression::from(x.clone()).less(10));
    let simplifier = RecordingSimplifier::returning(Expression::literal(result));

    let (outcome, events) = evaluate_with(
        &constraint,
        &bind([(x, Binding::Value(int(3)))]),
        &simplifier,
    );

    assert_eq!(outcome.expect("a Boolean decides"), expected);
    assert!(events.is_empty(), "{events:?}");
}

#[test]
fn simplifier_receives_the_expression_with_the_bindings_substituted() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let constraint = EquationConstraint::new(
        (&Expression::from(x.clone()) + Expression::from(y.clone())).less(10),
    );
    let simplifier = RecordingSimplifier::returning(Expression::literal(true));

    let (outcome, _) = evaluate_with(
        &constraint,
        &bind([
            (x, Binding::Value(int(3))),
            (y, Binding::Expression(Expression::literal(4))),
        ]),
        &simplifier,
    );

    outcome.expect("decided");
    assert_eq!(
        simplifier.inputs(),
        [(&Expression::literal(3) + Expression::literal(4)).less(10)]
    );
}

#[rstest]
#[case::a_boolean(Value::Bool(true), LiteralValue::Bool(true))]
#[case::an_integer(int(3), LiteralValue::Int(3.into()))]
#[case::a_float(Value::Float(1.5), LiteralValue::Float(1.5))]
#[case::a_decimal(Value::Decimal("1.50".parse::<Decimal>().expect("a decimal")), LiteralValue::Decimal("1.5".parse().expect("a decimal")))]
#[case::integer_text(text("05"), LiteralValue::Int(5.into()))]
#[case::decimal_text(text(".5"), LiteralValue::Decimal("0.5".parse().expect("a decimal")))]
fn literal_value_is_lifted_to_the_literal_it_denotes(
    #[case] value: Value,
    #[case] literal: LiteralValue,
) {
    let x = Identifier::new("x");
    let constraint = EquationConstraint::new(Expression::from(x.clone()).equals(0));
    let simplifier = RecordingSimplifier::returning(Expression::literal(false));

    let (outcome, _) = evaluate_with(
        &constraint,
        &bind([(x, Binding::Value(value))]),
        &simplifier,
    );

    outcome.expect("decided");
    assert_eq!(
        simplifier.inputs(),
        [Expression::literal(literal).equals(0)]
    );
}

#[rstest]
#[case::a_tuple(Value::Tuple(vec![int(1)]))]
#[case::a_frozen_set(Value::FrozenSet(vec![int(1)]))]
#[case::an_opaque_value(TestOpaque::token(1).into_value())]
fn value_that_is_no_literal_is_refused_naming_the_identifier(#[case] value: Value) {
    let x = Identifier::new("x");
    let constraint = EquationConstraint::new(Expression::from(x.clone()).less(10));
    let simplifier = RecordingSimplifier::identity();

    let (outcome, _) = evaluate_with(
        &constraint,
        &bind([(x.clone(), Binding::Value(value))]),
        &simplifier,
    );

    let error = outcome.expect_err("not a literal");
    assert!(
        matches!(
            &error,
            ConstraintError::UnusableBinding { identifier, reason: UnusableBindingReason::NotALiteral }
                if *identifier == x
        ),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        format!("the binding of {x:?} is neither an expression nor a literal")
    );
    assert!(simplifier.inputs().is_empty());
}

#[rstest]
#[case::a_word("abc")]
#[case::an_exponent("1e5")]
#[case::a_sign("-1.5")]
fn text_outside_the_literal_grammar_is_refused_with_its_cause(#[case] refused: &str) {
    let x = Identifier::new("x");
    let constraint = EquationConstraint::new(Expression::from(x.clone()).less(10));

    let (outcome, _) = evaluate_with(
        &constraint,
        &bind([(x.clone(), Binding::Value(text(refused)))]),
        &RecordingSimplifier::identity(),
    );

    let error = outcome.expect_err("not a literal text");
    assert!(
        matches!(
            &error,
            ConstraintError::UnusableBinding {
                reason: UnusableBindingReason::UnparsableText(_),
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        format!("the binding of {x:?} cannot be lifted into a literal")
    );
    assert!(std::error::Error::source(&error).is_some());
}

#[test]
fn bindings_outside_the_scope_are_never_read() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let constraint = EquationConstraint::new(Expression::from(x.clone()).less(10));
    let simplifier = RecordingSimplifier::returning(Expression::literal(true));

    let (outcome, _) = evaluate_with(
        &constraint,
        &bind([
            (y, Binding::Value(Value::Tuple(vec![]))),
            (x, Binding::Value(int(1))),
        ]),
        &simplifier,
    );

    assert_eq!(outcome.expect("y is ignored"), Outcome::Satisfied);
}

#[test]
fn first_unusable_binding_in_the_order_made_is_reported() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let constraint = EquationConstraint::new(
        (&Expression::from(x.clone()) + Expression::from(y.clone())).less(10),
    );

    let (outcome, _) = evaluate_with(
        &constraint,
        &bind([
            (y.clone(), Binding::Value(Value::Tuple(vec![]))),
            (x, Binding::Value(text("abc"))),
        ]),
        &RecordingSimplifier::identity(),
    );

    assert!(matches!(
        outcome,
        Err(ConstraintError::UnusableBinding { identifier, reason: UnusableBindingReason::NotALiteral })
            if identifier == y
    ));
}

#[test]
fn numeric_root_is_ill_typed_before_the_simplifier_is_asked() {
    let x = Identifier::new("x");
    let constraint = EquationConstraint::new(&Expression::from(x) + 1);
    let simplifier = RecordingSimplifier::identity();

    let (outcome, _) = evaluate_with(&constraint, &Bindings::new(), &simplifier);

    assert!(
        matches!(outcome, Err(ConstraintError::IllTyped(_))),
        "{outcome:?}"
    );
    assert!(simplifier.inputs().is_empty());
}

#[test]
fn number_bound_into_a_boolean_position_is_ill_typed() {
    let (p, q) = (Identifier::new("p"), Identifier::new("q"));
    let constraint = EquationConstraint::new(Expression::from(p.clone()).and(Expression::from(q)));

    let (outcome, _) = evaluate_with(
        &constraint,
        &bind([(p, Binding::Value(int(1)))]),
        &RecordingSimplifier::identity(),
    );

    assert!(
        matches!(outcome, Err(ConstraintError::IllTyped(_))),
        "{outcome:?}"
    );
}

#[test]
fn bound_native_constant_is_undecided_and_reported_after_the_screen() {
    let pi = BuiltinConstant::Pi.identifier().clone();
    let x = Identifier::new("x");
    let constraint =
        EquationConstraint::new(Expression::from(x.clone()).less(Expression::from(pi.clone())));
    let simplifier = RecordingSimplifier::identity();

    let (outcome, events) = evaluate_with(
        &constraint,
        &bind([
            (pi.clone(), Binding::Value(int(3))),
            (x, Binding::Value(int(1))),
        ]),
        &simplifier,
    );

    assert_eq!(outcome.expect("refused"), Outcome::Undecided);
    assert_eq!(events, [RecordedEvent::BoundNativeConstants(vec![pi])]);
    assert!(simplifier.inputs().is_empty());
}

#[test]
fn unreferenced_native_constant_binding_is_ignored() {
    let pi = BuiltinConstant::Pi.identifier().clone();
    let x = Identifier::new("x");
    let constraint = EquationConstraint::new(Expression::from(x.clone()).less(1));

    let (outcome, events) = evaluate_with(
        &constraint,
        &bind([(pi, Binding::Value(int(3))), (x, Binding::Value(int(0)))]),
        &RecordingSimplifier::returning(Expression::literal(true)),
    );

    assert_eq!(outcome.expect("decided"), Outcome::Satisfied);
    assert!(events.is_empty());
}

#[test]
fn non_boolean_literal_result_is_ill_typed() {
    let x = Identifier::new("x");
    let predicate = Expression::from(x.clone()).less(10);
    let constraint = EquationConstraint::new(predicate.clone());

    let (outcome, _) = evaluate_with(
        &constraint,
        &bind([(x, Binding::Value(int(1)))]),
        &RecordingSimplifier::returning(Expression::literal(1)),
    );

    let error = outcome.expect_err("a number is no predicate");
    assert!(
        matches!(&error, ConstraintError::NonBooleanResult { predicate: reported, result }
            if *reported == predicate && *result == Expression::literal(1)),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        "the predicate (x < 10) simplified to the literal 1, which is not a boolean, so it \
         denotes a number"
    );
}

#[rstest]
#[case::with_free_identifiers(true)]
#[case::ground(false)]
fn residual_result_is_undecided_and_reported(#[case] has_free_identifiers: bool) {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let constraint = EquationConstraint::new(Expression::from(x.clone()).less(Expression::from(y)));
    let residual = if has_free_identifiers {
        Expression::from(Identifier::new("z")).less(1)
    } else {
        Expression::literal(1).less(Expression::literal(2))
    };

    let (outcome, events) = evaluate_with(
        &constraint,
        &bind([(x, Binding::Value(int(1)))]),
        &RecordingSimplifier::returning(residual.clone()),
    );

    assert_eq!(outcome.expect("undecided"), Outcome::Undecided);
    assert_eq!(
        events,
        [RecordedEvent::Residual(residual, has_free_identifiers)]
    );
}

#[test]
fn simplifier_failure_is_the_solver_error() {
    let x = Identifier::new("x");
    let constraint = EquationConstraint::new(Expression::from(x.clone()).less(10));

    let (outcome, _) = evaluate_with(
        &constraint,
        &bind([(x, Binding::Value(int(1)))]),
        &RecordingSimplifier::failing("sympy gave up"),
    );

    assert!(
        matches!(
            outcome,
            Err(ConstraintError::Solve(SolveError::Backend { .. }))
        ),
        "{outcome:?}"
    );
}

#[test]
fn solver_without_a_simplifier_cannot_decide_an_equation() {
    let x = Identifier::new("x");
    let constraint = EquationConstraint::new(Expression::from(x).less(10));
    let solver = Solver::new();

    let outcome = constraint.evaluate(&Bindings::new(), &ConstraintContext::new(&solver));

    assert!(
        matches!(
            outcome,
            Err(ConstraintError::Solve(SolveError::NoCapableBackend(_)))
        ),
        "{outcome:?}"
    );
}

#[test]
fn simplification_is_asked_with_the_context_registry() {
    let x = Identifier::new("x");
    let constraint = EquationConstraint::new(Expression::from(x).less(10));
    let simplifier = RecordingSimplifier::returning(Expression::literal(true));
    let solver = simplifier.solver();
    let registry = FunctionRegistry::new();

    constraint
        .evaluate(
            &Bindings::new(),
            &ConstraintContext::new(&solver).with_registry(&registry),
        )
        .expect("decided");
    constraint
        .evaluate(&Bindings::new(), &ConstraintContext::new(&solver))
        .expect("decided");

    assert_eq!(simplifier.registry_sizes(), [Some(0), None]);
}

#[test]
fn scope_is_the_expression_free_identifiers() {
    let (x, y) = (Identifier::new("x"), Identifier::new("y"));
    let constraint =
        EquationConstraint::new(Expression::from(x.clone()).less(Expression::from(y.clone())));

    assert_eq!(constraint.free_identifiers(), [x, y].into_iter().collect());
}
